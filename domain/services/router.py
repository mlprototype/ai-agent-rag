"""
ユーザーからの質問を分析し、最適な回答生成ルート(直接回答、計算、検索など)を決定するルーティング処理を担当するファイルです。
エージェントの処理パイプラインの入り口に位置し、後続のワークフロー(RAG、計算ツールなど)を振り分けます。
入力としてユーザーの質問文字列を受け取り、出力として決定したルートや確信度を含む RouteDecision オブジェクトを返します。
ヒューリスティック(ルールベース)とLLMによる判定を組み合わせ、LLMのエラーや遅延時には安全なフォールバックルートへ移行するよう設計されています。
"""
import asyncio
import logging
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from domain.services.prompt_loader import classify_prompt_error, load_prompt, log_prompt_fallback
from domain.services.prompt_registry import ROUTER_PROMPT
from domain.services.heuristic_router import HeuristicRouter, RouteDecision

logger = logging.getLogger(__name__)


class LLMRouteDecision(BaseModel):
    route: Literal["direct_answer", "calculator", "agentic_retrieval"]
    reason: str


class AgentRouter:
    _chain = None
    _PROMPT_NAME = ROUTER_PROMPT
    _CALC_PATTERN = re.compile(r"[\d\s\.\(\)\+\-\*/%]+")
    _SIMPLE_PATTERNS = (
        "こんにちは",
        "こんばんは",
        "おはよう",
        "ありがとう",
        "thanks",
        "thank you",
        "あなたは誰",
        "help",
    )

    @classmethod
    def _get_chain(cls):
        """
        LLMを使用したルーティング判定用のLangChainチェーンを取得・初期化します。
        システムプロンプトやモデル設定をロードし、LLMRouteDecisionの構造化出力を返すチェーンを構築して返却します。
        一度構築したチェーンはクラス変数 `_chain` にキャッシュし、次回以降の呼び出しで再利用(状態更新)します。
        """
        if cls._chain is None:
            fallback_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "あなたは問い合わせルーターです。ユーザーの最新メッセージを以下の3分類のいずれかに必ず分類してください。\n"
                    "- direct_answer: 挨拶、雑談、一般的な短い会話、外部検索が不要な質問\n"
                    "- calculator: 算術計算や数式評価が主目的の質問\n"
                    "- agentic_retrieval: 検索済みナレッジや複数観点の取得が必要な質問\n"
                    "JSONで返し、route と短い reason を含めてください。"
                ),
                ("human", "{query}"),
            ])
            prompt = load_prompt(cls._PROMPT_NAME, fallback_prompt)
            # 0: 生成時のランダム性を排除し、ルーティング判定結果を常に一定(決定的)にするための温度パラメータ値(0)。
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(LLMRouteDecision)
            cls._chain = prompt | llm
        return cls._chain

    @classmethod
    async def route(cls, query: str, timeout_seconds: float | None = None) -> RouteDecision:
        """
        ユーザーの質問文字列を評価し、最終的な処理ルートを決定します。
        入力として質問(query)を受け取り、出力として RouteDecision を返します。
        LLM呼び出しでのタイムアウトやAPIエラー時には、例外を吸収して検索ベース(fallback_retrieval)ルートを返します。
        これにより、ルーティングの失敗がシステム全体のクラッシュを引き起こすのを防ぎます。
        """
        settings = get_settings()

        # パフォーマンス向上のため、LLM呼び出し前にルールベースでの高速かつ確実なルーティングが可能かを確認するために必要
        if settings.router_heuristic_enabled:
            heuristic = HeuristicRouter.route(query, enable_compare=settings.router_heuristic_compare_enabled)
            # router_heuristic_confidence_threshold: ヒューリスティック判定結果を採用するための最低限の確信度スコア。
            # この閾値以上の確信度があれば、LLMを呼び出さずに即座に結果を返す判断に使われる。
            if heuristic is not None and heuristic.confidence >= settings.router_heuristic_confidence_threshold:
                return heuristic

        try:
            chain = cls._get_chain()
            # LLMの応答遅延によるユーザー体験の悪化やシステムのリソース占有を防ぐために必要
            llm_decision: LLMRouteDecision = await asyncio.wait_for(
                chain.ainvoke({"query": query}),
                # router_timeout_seconds: LLMルーティング処理の最大待ち時間(秒)。
                # この時間を超過した場合にタイムアウト例外を発生させ、フォールバックへ移行する判断に使われる。
                timeout=timeout_seconds if timeout_seconds is not None else settings.router_timeout_seconds,
            )
            return RouteDecision(
                query_type="retrieval_complex" if llm_decision.route == "agentic_retrieval" else ("direct" if llm_decision.route == "direct_answer" else "calc"), # LLM result is coarse
                route=llm_decision.route, # type: ignore
                routing_layer="llm",
                source="llm_success",
                # 0.8: LLMによる判定が成功した場合に固定で付与される確信度スコア。
                # 後続の処理でこのルーティング結果の信頼性として評価される。
                confidence=0.8,
                heuristic_matched=False,
                heuristic_rule="",
                llm_router_invoked=True,
                reason=llm_decision.reason
            )

        except Exception as exc:
            error_reason = classify_prompt_error(exc)
            log_prompt_fallback(
                cls._PROMPT_NAME,
                error_reason,
                fallback_target="agentic_retrieval",
                detail=str(exc),
            )
            logger.warning("Router の判定に失敗したため retrieval にフォールバックします: %s", exc)
            fallback_reason = "timeout_fallback" if error_reason == "timeout" else "error_fallback"
            return RouteDecision(
                query_type="retrieval_complex",
                route="fallback_retrieval", # Custom fallback logical route mapping
                routing_layer="fallback",
                source="llm_timeout_fallback" if error_reason == "timeout" else "llm_error_fallback",
                # 0.5: フォールバックルートが選択された場合に固定で付与される確信度スコア。
                # LLMによる正常な判定(0.8)よりも低い信頼性であることを示し、後続処理でより慎重な振る舞いをさせるための判断に使われる。
                confidence=0.5,
                heuristic_matched=False,
                heuristic_rule="",
                llm_router_invoked=True,
                reason=fallback_reason
            )

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from domain.services.coverage_checker import assess_coverage, build_coverage_plan
from domain.services.prompt_loader import classify_prompt_error, load_prompt, log_prompt_fallback
from domain.services.prompt_registry import RETRIEVAL_CRITIC_PROMPT

logger = logging.getLogger(__name__)


class CritiqueResult(BaseModel):
    verdict: Literal["SUFFICIENT", "INSUFFICIENT"]
    reason: str = ""
    coverage_score: float = 0.7
    missing_aspects: list[str] = Field(default_factory=list)


class RetrievalCritic:
    _chain = None
    _PROMPT_NAME = RETRIEVAL_CRITIC_PROMPT

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = item.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(cleaned)
        return ordered

    @classmethod
    def _get_chain(cls):
        if cls._chain is None:
            fallback_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "あなたは検索品質評価者です。質問に対して、検索結果だけで回答可能か判定してください。"
                    "特に expected_aspects と coverage_summary を見て、主要な観点が欠けていれば INSUFFICIENT にしてください。"
                    "JSONで verdict, reason, coverage_score, missing_aspects を返してください。"
                ),
                (
                    "human",
                    "質問: {original_query}\n"
                    "期待観点: {expected_aspects}\n"
                    "意図: {coverage_intent}\n"
                    "比較軸: {comparison_axes}\n"
                    "coverage_summary: {coverage_summary}\n"
                    "検索結果サマリー:\n{chunks_summary}"
                ),
            ])
            prompt = load_prompt(cls._PROMPT_NAME, fallback_prompt)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CritiqueResult)
            cls._chain = prompt | llm
        return cls._chain

    # 関数の役割: 検索結果の回答十分性の評価
    # 入出力: 質問、検索結果サマリーを受け取り、評価結果(CritiqueResult)を返す
    # フォールバック: LLMエラー時はルールベースのカバレッジスコアで暫定評価を行う
    @classmethod
    async def critique(
        cls,
        original_query: str,
        chunks_summary: str,
        timeout_seconds: float | None = None,
    ) -> CritiqueResult:
        if not chunks_summary.strip():
            return CritiqueResult(
                verdict="INSUFFICIENT",
                reason="検索結果が空で、回答に必要な情報が見つかっていません。",
                coverage_score=0.0,
                missing_aspects=["関連情報全体"],
            )

        coverage_plan = build_coverage_plan(original_query)
        coverage = assess_coverage(coverage_plan, chunks_summary)
        if coverage.required_missing_aspects:
            return CritiqueResult(
                verdict="INSUFFICIENT",
                reason=f"coverage_missing: {', '.join(coverage.required_missing_aspects)}",
                coverage_score=coverage.coverage_score,
                missing_aspects=coverage.missing_aspects or coverage.required_missing_aspects,
            )

        settings = get_settings()
        try:
            chain = cls._get_chain()
            result = await asyncio.wait_for(
                chain.ainvoke(
                    {
                        "original_query": original_query,
                        "chunks_summary": chunks_summary,
                        "expected_aspects": ", ".join(coverage_plan.expected_aspects) or "なし",
                        "coverage_intent": coverage_plan.intent,
                        "comparison_axes": ", ".join(coverage_plan.comparison_axes) or "なし",
                        "coverage_summary": coverage.summary,
                    }
                ),
                timeout=timeout_seconds if timeout_seconds is not None else settings.retrieval_critic_timeout_seconds,
            )
            result.missing_aspects = cls._dedupe(coverage.missing_aspects + result.missing_aspects)
            llm_score = max(0.0, min(1.0, float(result.coverage_score)))
            if result.missing_aspects:
                result.coverage_score = min(llm_score, coverage.coverage_score)
            else:
                result.coverage_score = max(llm_score, coverage.coverage_score)
            return result
        except Exception as exc:
            # Critic自体がエラーになった場合でも、ルールベースのカバレッジスコアで暫定評価を行いパイプラインを継続する
            error_reason = classify_prompt_error(exc)
            fallback_verdict = "INSUFFICIENT" if coverage.missing_aspects or coverage.coverage_score < 0.75 else "SUFFICIENT"
            fallback_score = min(coverage.coverage_score, 0.49) if fallback_verdict == "INSUFFICIENT" else min(coverage.coverage_score, 0.7)
            log_prompt_fallback(
                cls._PROMPT_NAME,
                error_reason,
                fallback_target=f"{fallback_verdict.lower()}_fallback",
                detail=str(exc),
            )
            logger.warning(
                "Retrieval Critic が失敗したため %s にフォールバックします: %s",
                fallback_verdict,
                exc,
            )
            return CritiqueResult(
                verdict=fallback_verdict,
                reason=f"critic_fallback:{error_reason}",
                coverage_score=fallback_score,
                missing_aspects=coverage.missing_aspects,
            )

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
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(LLMRouteDecision)
            cls._chain = prompt | llm
        return cls._chain

    @classmethod
    async def route(cls, query: str, timeout_seconds: float | None = None) -> RouteDecision:
        settings = get_settings()

        if settings.router_heuristic_enabled:
            heuristic = HeuristicRouter.route(query, enable_compare=settings.router_heuristic_compare_enabled)
            if heuristic is not None and heuristic.confidence >= settings.router_heuristic_confidence_threshold:
                return heuristic

        try:
            chain = cls._get_chain()
            llm_decision: LLMRouteDecision = await asyncio.wait_for(
                chain.ainvoke({"query": query}),
                timeout=timeout_seconds if timeout_seconds is not None else settings.router_timeout_seconds,
            )
            return RouteDecision(
                query_type="retrieval_complex" if llm_decision.route == "agentic_retrieval" else ("direct" if llm_decision.route == "direct_answer" else "calc"), # LLM result is coarse
                route=llm_decision.route, # type: ignore
                routing_layer="llm",
                source="llm_success",
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
                confidence=0.5,
                heuristic_matched=False,
                heuristic_rule="",
                llm_router_invoked=True,
                reason=fallback_reason
            )

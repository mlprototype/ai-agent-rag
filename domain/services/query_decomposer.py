import asyncio
import logging

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from domain.services.prompt_loader import classify_prompt_error, load_prompt, log_prompt_fallback
from domain.services.prompt_registry import DECOMPOSE_PROMPT, REWRITE_PROMPT

logger = logging.getLogger(__name__)


class SubQueryResponse(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)


class QueryPlanningResult(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None


class QueryDecomposer:
    _decompose_chain = None
    _rewrite_chain = None
    _DECOMPOSE_PROMPT_NAME = DECOMPOSE_PROMPT
    _REWRITE_PROMPT_NAME = REWRITE_PROMPT

    @classmethod
    def _get_decompose_chain(cls):
        if cls._decompose_chain is None:
            fallback_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "あなたは検索クエリ分解の専門家です。元の質問を補うための sub-query を返してください。"
                    "JSONで sub_queries のみを返し、最大4件、重複なし、短く具体的にしてください。"
                ),
                (
                    "human",
                    "元の質問: {original_query}\n不足理由: {critique_reason}\n不足観点: {missing_aspects}"
                ),
            ])
            prompt = load_prompt(cls._DECOMPOSE_PROMPT_NAME, fallback_prompt)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(SubQueryResponse)
            cls._decompose_chain = prompt | llm
        return cls._decompose_chain

    @classmethod
    def _get_rewrite_chain(cls):
        if cls._rewrite_chain is None:
            fallback_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "あなたは sub-query 改善の専門家です。現在の検索クエリ群を改善し、不足観点を埋める新しい sub-query を返してください。"
                    "JSONで sub_queries のみを返し、最大4件、重複なし、説明は不要です。"
                ),
                (
                    "human",
                    "元の質問: {original_query}\n現在の sub_queries: {current_sub_queries}\n"
                    "改善理由: {critique_reason}\n不足観点: {missing_aspects}"
                ),
            ])
            prompt = load_prompt(cls._REWRITE_PROMPT_NAME, fallback_prompt)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(SubQueryResponse)
            cls._rewrite_chain = prompt | llm
        return cls._rewrite_chain

    @staticmethod
    def _normalize(sub_queries: list[str], fallback_query: str) -> list[str]:
        settings = get_settings()
        normalized: list[str] = []
        seen: set[str] = set()
        for query in sub_queries:
            cleaned = query.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(cleaned)
            if len(normalized) >= settings.max_sub_queries:
                break
        return normalized or [fallback_query]

    @classmethod
    async def decompose(
        cls,
        original_query: str,
        critique_reason: str,
        missing_aspects: list[str],
        timeout_seconds: float | None = None,
    ) -> QueryPlanningResult:
        settings = get_settings()
        try:
            chain = cls._get_decompose_chain()
            response = await asyncio.wait_for(
                chain.ainvoke(
                    {
                        "original_query": original_query,
                        "critique_reason": critique_reason,
                        "missing_aspects": ", ".join(missing_aspects) or "なし",
                    }
                ),
                timeout=timeout_seconds if timeout_seconds is not None else settings.decompose_timeout_seconds,
            )
            return QueryPlanningResult(
                sub_queries=cls._normalize(response.sub_queries, original_query),
                fallback_reason=None,
            )
        except Exception as exc:
            error_reason = classify_prompt_error(exc)
            log_prompt_fallback(
                cls._DECOMPOSE_PROMPT_NAME,
                error_reason,
                fallback_target="original_query",
                detail=str(exc),
            )
            logger.warning("Query decomposition に失敗したため original_query にフォールバックします: %s", exc)
            fallback_reason = "timeout_fallback" if error_reason == "timeout" else "error_fallback"
            return QueryPlanningResult(sub_queries=[original_query], fallback_reason=fallback_reason)

    @classmethod
    async def rewrite(
        cls,
        original_query: str,
        current_sub_queries: list[str],
        critique_reason: str,
        missing_aspects: list[str],
        timeout_seconds: float | None = None,
    ) -> QueryPlanningResult:
        settings = get_settings()
        try:
            chain = cls._get_rewrite_chain()
            response = await asyncio.wait_for(
                chain.ainvoke(
                    {
                        "original_query": original_query,
                        "current_sub_queries": " | ".join(current_sub_queries) or original_query,
                        "critique_reason": critique_reason,
                        "missing_aspects": ", ".join(missing_aspects) or "なし",
                    }
                ),
                timeout=timeout_seconds if timeout_seconds is not None else settings.rewrite_timeout_seconds,
            )
            return QueryPlanningResult(
                sub_queries=cls._normalize(response.sub_queries, original_query),
                fallback_reason=None,
            )
        except Exception as exc:
            error_reason = classify_prompt_error(exc)
            log_prompt_fallback(
                cls._REWRITE_PROMPT_NAME,
                error_reason,
                fallback_target="current_sub_queries",
                detail=str(exc),
            )
            logger.warning("Sub-query rewrite に失敗したため current_sub_queries を維持します: %s", exc)
            fallback_reason = "timeout_fallback" if error_reason == "timeout" else "error_fallback"
            return QueryPlanningResult(
                sub_queries=current_sub_queries or [original_query],
                fallback_reason=fallback_reason,
            )

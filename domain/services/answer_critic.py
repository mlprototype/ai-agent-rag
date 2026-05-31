# ファイルの責務: 生成された回答が検索結果に裏付けられているか（ハルシネーションの有無）の検証
# 主な入出力: 質問、回答、引用ソースを受け取り、判定結果(AnswerVerdict)を返す
import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from domain.services.coverage_checker import (
    answer_admits_insufficient_support,
    answer_uses_comparison_language,
    assess_coverage,
    aspects_mentioned_in_text,
    build_coverage_plan,
)
from domain.services.prompt_loader import classify_prompt_error, load_prompt, log_prompt_fallback
from domain.services.prompt_registry import ANSWER_CRITIC_PROMPT

logger = logging.getLogger(__name__)


class AnswerVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    hallucination_risk: Literal["low", "medium", "high"] = "low"
    coverage_ok: bool = True
    confidence_override: float | None = None
    reason: str = ""
    missing_aspects: list[str] = Field(default_factory=list)


class AnswerCritic:
    _chain = None
    _PROMPT_NAME = ANSWER_CRITIC_PROMPT

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
                    "あなたは回答検証者です。回答がソースに忠実か、質問の全側面に答えているかを評価してください。"
                    "特に expected_aspects と coverage_summary を見て、ソースに無い観点を答えていれば FAIL にしてください。"
                    "JSONで verdict, hallucination_risk, coverage_ok, confidence_override, reason, missing_aspects を返してください。"
                ),
                (
                    "human",
                    "元の質問: {original_query}\n"
                    "期待観点: {expected_aspects}\n"
                    "意図: {coverage_intent}\n"
                    "比較軸: {comparison_axes}\n"
                    "coverage_summary: {coverage_summary}\n"
                    "回答: {answer}\n"
                    "引用ソース:\n{sources_summary}"
                ),
            ])
            prompt = load_prompt(cls._PROMPT_NAME, fallback_prompt)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnswerVerdict)
            cls._chain = prompt | llm
        return cls._chain

    # 関数の役割: 回答のソースへの忠実性・ハルシネーション検証
    # 入出力: 質問、回答、引用ソースを受け取り、評価結果(AnswerVerdict)を返す
    # フォールバック: LLMエラー時はルールベースの必須観点不足有無で判定を行う
    @classmethod
    async def verify(
        cls,
        original_query: str,
        answer: str,
        sources_summary: str,
        timeout_seconds: float | None = None,
    ) -> AnswerVerdict:
        coverage_plan = build_coverage_plan(original_query)
        coverage = assess_coverage(coverage_plan, sources_summary)
        mentioned_missing = aspects_mentioned_in_text(coverage.required_missing_aspects, answer)
        mentioned_entities = aspects_mentioned_in_text(coverage_plan.entities, answer)
        compare_without_support = (
            coverage_plan.intent == "compare"
            and (answer_uses_comparison_language(answer) or bool(mentioned_entities))
        )
        if (
            coverage.required_missing_aspects
            and not answer_admits_insufficient_support(answer)
            and (mentioned_missing or compare_without_support)
        ):
            missing = coverage.required_missing_aspects
            return AnswerVerdict(
                verdict="FAIL",
                hallucination_risk="high",
                coverage_ok=False,
                confidence_override=0.35,
                reason=f"coverage_missing: {', '.join(missing)}",
                missing_aspects=missing,
            )

        settings = get_settings()
        try:
            chain = cls._get_chain()
            result = await asyncio.wait_for(
                chain.ainvoke(
                    {
                        "original_query": original_query,
                        "answer": answer,
                        "sources_summary": sources_summary,
                        "expected_aspects": ", ".join(coverage_plan.expected_aspects) or "なし",
                        "coverage_intent": coverage_plan.intent,
                        "comparison_axes": ", ".join(coverage_plan.comparison_axes) or "なし",
                        "coverage_summary": coverage.summary,
                    }
                ),
                timeout=timeout_seconds if timeout_seconds is not None else settings.answer_critic_timeout_seconds,
            )
            result.missing_aspects = cls._dedupe(coverage.missing_aspects + result.missing_aspects)
            if result.confidence_override is not None:
                result.confidence_override = max(0.0, min(0.5, float(result.confidence_override)))
            return result
        except Exception as exc:
            error_reason = classify_prompt_error(exc)
            fallback_missing = coverage.required_missing_aspects or coverage.missing_aspects
            fallback_verdict = "FAIL" if fallback_missing else "PASS"
            confidence_override = 0.35 if fallback_missing else 0.49
            log_prompt_fallback(
                cls._PROMPT_NAME,
                error_reason,
                fallback_target=f"{fallback_verdict.lower()}_fallback",
                detail=str(exc),
            )
            logger.warning(
                "Answer Critic が失敗したため %s にフォールバックします: %s",
                fallback_verdict,
                exc,
            )
            return AnswerVerdict(
                verdict=fallback_verdict,
                hallucination_risk="high" if fallback_missing else "medium",
                coverage_ok=not bool(fallback_missing),
                confidence_override=confidence_override,
                reason=f"critic_fallback:{error_reason}",
                missing_aspects=fallback_missing,
            )

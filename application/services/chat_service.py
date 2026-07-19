# ファイルの責務: 外部インターフェースから呼び出されるチャットと評価Traceのユースケース処理
# 主な入出力: ChatRequestを受け取り、1回のLangGraph実行からChatResponseまたはAgentRunTraceを返す
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, is_dataclass
from threading import Lock
from typing import Any, AsyncGenerator, Mapping
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult

from application.agents.graph import graph
from application.agents.state import AgentState
from application.dto.agent_trace import (
    AgentRunInput,
    AgentRunOutput,
    AgentRunTrace,
    CitationTrace,
    ControlTrace,
    GuardrailTrace,
    SourceTrace,
    TimingTrace,
    ToolCallTrace,
    UsageTrace,
)
from application.dto.chat_models import ChatRequest, ChatResponse, Source
from config.settings import get_settings

_SETTINGS = get_settings()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_TRACE_SCHEMA_VERSION = "1.0"
_TRACE_TARGET = "ai-agent-rag"


def _json_safe(value: Any) -> Any:
    """Toolイベントの値をJSON互換へ変換する。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    return str(value)


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if converted >= 0 else None


def _optional_nonnegative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(converted) or converted < 0:
        return None
    return converted


def _normalize_usage(raw_usage: Mapping[str, Any]) -> dict[str, int | float | None]:
    input_tokens = _optional_nonnegative_int(
        raw_usage.get("input_tokens", raw_usage.get("prompt_tokens"))
    )
    output_tokens = _optional_nonnegative_int(
        raw_usage.get("output_tokens", raw_usage.get("completion_tokens"))
    )
    total_tokens = _optional_nonnegative_int(raw_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    cost_usd = _optional_nonnegative_float(
        raw_usage.get("cost_usd", raw_usage.get("total_cost"))
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


class _GraphRunObserver(BaseCallbackHandler):
    """1回のGraph実行中に伝播した実ToolイベントとLLM usageを収集する。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._tool_calls: list[dict[str, Any]] = []
        self._pending_tools: dict[str, tuple[int, float]] = {}
        self._usage: dict[str, int | float | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(tool_call) for tool_call in self._tool_calls]

    @property
    def usage(self) -> dict[str, int | float | None]:
        with self._lock:
            return dict(self._usage)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        name = str((serialized or {}).get("name") or "unknown_tool")
        arguments: Any = inputs
        if arguments is None:
            try:
                arguments = json.loads(input_str)
            except (json.JSONDecodeError, TypeError):
                arguments = {"input": input_str}
        if not isinstance(arguments, Mapping):
            arguments = {"input": arguments}

        with self._lock:
            index = len(self._tool_calls)
            self._tool_calls.append(
                {
                    "name": name,
                    "arguments": _json_safe(arguments),
                }
            )
            self._pending_tools[str(run_id)] = (index, time.monotonic())

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        self._finish_tool(run_id, result=_json_safe(output))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        self._finish_tool(run_id, error=type(error).__name__)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        del run_id, kwargs
        usage = self._usage_from_response(response)
        if not any(value is not None for value in usage.values()):
            return
        with self._lock:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage[key]
                if value is not None:
                    current = self._usage[key]
                    self._usage[key] = int(current or 0) + int(value)
            cost_usd = usage["cost_usd"]
            if cost_usd is not None:
                current_cost = self._usage["cost_usd"]
                self._usage["cost_usd"] = float(current_cost or 0.0) + float(cost_usd)

    def _finish_tool(
        self,
        run_id: UUID,
        *,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            pending = self._pending_tools.pop(str(run_id), None)
            if pending is None:
                return
            index, started_at = pending
            record = self._tool_calls[index]
            if error is None:
                record["result"] = result
            else:
                record["error"] = error
            record["duration_ms"] = int((time.monotonic() - started_at) * 1000)

    @staticmethod
    def _usage_from_response(response: LLMResult) -> dict[str, int | float | None]:
        llm_output = response.llm_output or {}
        for key in ("token_usage", "usage", "usage_metadata"):
            candidate = llm_output.get(key)
            if isinstance(candidate, Mapping):
                normalized = _normalize_usage(candidate)
                if any(value is not None for value in normalized.values()):
                    return normalized

        aggregated: dict[str, int | float | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                candidate = getattr(message, "usage_metadata", None)
                if not isinstance(candidate, Mapping):
                    continue
                normalized = _normalize_usage(candidate)
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    value = normalized[key]
                    if value is not None:
                        aggregated[key] = int(aggregated[key] or 0) + int(value)
                cost_usd = normalized["cost_usd"]
                if cost_usd is not None:
                    aggregated["cost_usd"] = float(aggregated["cost_usd"] or 0.0) + float(cost_usd)
        return aggregated


@dataclass(frozen=True)
class _GraphRunResult:
    final_state: AgentState
    total_latency_ms: int
    tool_events: list[dict[str, Any]]
    usage: dict[str, int | float | None]


class ChatService:
    """チャット応答と評価専用Traceを同一のGraph実行結果から構築する。"""

    @staticmethod
    def _extract_citation_ids(answer: str) -> list[int]:
        cited_ids: list[int] = []
        for citation_group in re.findall(r"\[([\d,\s]+)\]", answer):
            for citation in citation_group.split(","):
                normalized = citation.strip()
                if normalized.isdigit():
                    citation_id = int(normalized)
                    if citation_id not in cited_ids:
                        cited_ids.append(citation_id)
        return cited_ids

    @staticmethod
    def _extract_sources(state: AgentState, answer: str) -> tuple[list[Source], int]:
        raw_sources = state.get("sources", [])
        if not raw_sources:
            return [], 0

        cited_ids = set(ChatService._extract_citation_ids(answer))
        if not cited_ids:
            return [], len(raw_sources)

        sources = []
        for src in raw_sources:
            if not isinstance(src, Mapping):
                continue
            citation_id = _optional_nonnegative_int(src.get("citation_id"))
            if citation_id in cited_ids:
                sources.append(
                    Source(
                        citation_id=citation_id,
                        doc_id=str(src.get("doc_id", "不明")),
                        chunk_id=str(src.get("chunk_id", "")),
                        snippet=str(src.get("snippet", "")),
                        score=float(src.get("hybrid_score", src.get("score", 0.0))),
                        hybrid_score=float(src.get("hybrid_score", 0.0)),
                        vector_score=float(src.get("vector_score", 0.0)),
                        bm25_score=float(src.get("bm25_score", 0.0)),
                        rerank_score=float(src.get("rerank_score", 0.0)),
                    )
                )
        return sources, len(raw_sources) - len(sources)

    @staticmethod
    async def _run_graph(request: ChatRequest) -> _GraphRunResult:
        inputs = {"messages": [HumanMessage(content=request.question)]}
        observer = _GraphRunObserver()
        config = {
            "configurable": {"thread_id": request.session_id},
            "callbacks": [observer],
        }
        final_state: AgentState = {}
        started_at = time.monotonic()

        async for event in graph.astream(inputs, config=config, stream_mode="values"):
            final_state = event

        total_latency_ms = int((time.monotonic() - started_at) * 1000)
        return _GraphRunResult(
            final_state=final_state,
            total_latency_ms=total_latency_ms,
            tool_events=observer.tool_calls,
            usage=observer.usage,
        )

    @classmethod
    def _build_chat_response(
        cls,
        request: ChatRequest,
        run: _GraphRunResult,
    ) -> ChatResponse:
        final_state = run.final_state
        query_type = final_state.get("query_type")
        route = final_state.get("route")
        answer = final_state.get("answer", "")

        timeout_stages = list(final_state.get("timeout_stages", []))
        fallback_stages = list(final_state.get("fallback_stages", []))
        critique_reason = final_state.get("critique_reason", "")
        warning_codes = list(final_state.get("warning_codes", []))

        critic_degraded = bool(
            final_state.get("retrieval_critic_skipped_reason")
            or final_state.get("answer_critic_skipped_reason")
            or critique_reason.startswith("critic_fallback:")
            or final_state.get("retrieval_degraded")
        )

        if route == "direct_answer":
            sources = None
            filtered_count = 0
            confidence = None
            warning = None
            source_name = None
        elif route == "structured_query_tool":
            sources = None
            filtered_count = 0
            confidence = round(float(final_state.get("confidence", 0.95)), 2)
            warning = final_state.get("warning")
            source_name = final_state.get("structured_query_source_name")
        else:
            sources, filtered_count = cls._extract_sources(final_state, answer)
            confidence = round(float(final_state.get("confidence", 0.5)), 2)
            warning = final_state.get("warning")
            source_name = None

        logger.info(
            {
                "event": "chat_request_summary",
                "session_id": request.session_id,
                "query_type": query_type,
                "route": route,
                "total_latency_ms": run.total_latency_ms,
                "router_timeout": "router" in timeout_stages,
                "decompose_timeout": any(stage in {"decompose", "rewrite"} for stage in timeout_stages),
                "critic_degraded": critic_degraded,
                "final_confidence": confidence,
                "timeout_stage": timeout_stages,
                "fallback_used": bool(fallback_stages),
                "fallback_level": final_state.get("fallback_level", "full_path"),
                "partial_retrieval_used": final_state.get("partial_retrieval_used", False),
                "retrieval_timeout_count": final_state.get("retrieval_timeout_count", 0),
                "retrieval_success_count": final_state.get("retrieval_success_count", 0),
                "warning_codes": warning_codes,
                "retrieval_quality_level": final_state.get("retrieval_quality_level", "high"),
                "remaining_budget_ms_at_generate": final_state.get("remaining_budget_ms_at_generate"),
                "skipped_stages": final_state.get("skipped_stages", []),
                "direct_definition_evidence_found": query_type == "definition" and "low_confidence_definition_guard" not in warning_codes,
                "retrieval_grounding_sufficient": final_state.get("answer_ok", False),
                "citation_filtered_count": filtered_count,
            }
        )

        return ChatResponse(
            answer=answer,
            query_type=query_type,
            route=route,
            sources=sources,
            source_name=source_name,
            confidence=confidence,
            warning=warning,
        )

    @classmethod
    def _build_agent_run_trace(
        cls,
        request: ChatRequest,
        run: _GraphRunResult,
        *,
        case_id: str,
        target: str = _TRACE_TARGET,
        run_id: str | None = None,
    ) -> AgentRunTrace:
        state = run.final_state
        answer = str(state.get("answer", ""))
        query_type = str(state.get("query_type") or "unknown")
        route = str(state.get("route") or "unknown")
        confidence_value = state.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence_value)))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0

        warning_codes = cls._string_list(state.get("warning_codes"))
        sources, citation_source_ids = cls._build_trace_sources(state)
        citations = [
            CitationTrace(
                citation_id=str(citation_id),
                source_id=citation_source_ids.get(
                    citation_id, f"unresolved-citation:{citation_id}"
                ),
            )
            for citation_id in cls._extract_citation_ids(answer)
        ]
        tool_calls = cls._build_tool_calls(request, run, sources)

        retry_count = _optional_nonnegative_int(state.get("retry_count")) or 0
        fallback_stages = cls._string_list(state.get("fallback_stages"))
        timeout_stages = cls._string_list(state.get("timeout_stages"))
        skipped_stages = cls._string_list(state.get("skipped_stages"))
        fallback_level = state.get("fallback_level")
        if not isinstance(fallback_level, (str, int)) or fallback_level == "":
            fallback_level = None
        remaining_budget = _optional_nonnegative_float(
            state.get("remaining_budget_ms_at_generate")
        )
        fallback_used = bool(
            fallback_stages
            or (fallback_level is not None and fallback_level != "full_path")
        )
        strict_insufficient = bool(state.get("strict_insufficient_response"))
        stop_reason = "insufficient_retrieval" if strict_insufficient else "completed"

        guardrail_codes = [
            code
            for code in warning_codes
            if "guard" in code.lower() or code == "strict_insufficient_response"
        ]
        if strict_insufficient and "strict_insufficient_response" not in guardrail_codes:
            guardrail_codes.append("strict_insufficient_response")
        guardrail_messages = []
        warning = state.get("warning")
        if guardrail_codes and isinstance(warning, str) and warning.strip():
            guardrail_messages.append(warning.strip())

        usage = cls._build_usage(run)
        tool_durations = [
            tool.duration_ms
            for tool in tool_calls
            if tool.duration_ms is not None
        ]
        route_latency = (
            _optional_nonnegative_float(state.get("route_decision_latency_ms"))
            if "route_decision_latency_ms" in state
            else None
        )

        return AgentRunTrace(
            schema_version=_TRACE_SCHEMA_VERSION,
            run_id=run_id or str(uuid4()),
            case_id=case_id,
            target=target,
            input=AgentRunInput(question=request.question),
            output=AgentRunOutput(
                answer=answer,
                query_type=query_type,
                route=route,
                confidence=round(confidence, 2),
                warning_codes=warning_codes,
            ),
            tool_calls=tool_calls,
            citations=citations,
            sources=sources,
            guardrail=GuardrailTrace(
                triggered=bool(guardrail_codes),
                blocked=strict_insufficient,
                codes=guardrail_codes,
                messages=guardrail_messages,
            ),
            control=ControlTrace(
                attempt_count=retry_count + 1,
                retry_count=retry_count,
                fallback_used=fallback_used,
                fallback_stages=fallback_stages,
                timeout_stages=timeout_stages,
                skipped_stages=skipped_stages,
                fallback_level=fallback_level,
                remaining_budget_ms_at_generate=remaining_budget,
                stop_reason=stop_reason,
            ),
            usage=usage,
            timing=TimingTrace(
                latency_ms=run.total_latency_ms,
                tool_latency_ms=sum(tool_durations) if tool_durations else None,
                route_decision_latency_ms=route_latency,
            ),
        )

    @classmethod
    def _build_trace_sources(
        cls,
        state: AgentState,
    ) -> tuple[list[SourceTrace], dict[int, str]]:
        sources: list[SourceTrace] = []
        citation_source_ids: dict[int, str] = {}
        seen_source_ids: set[str] = set()

        for raw_source in state.get("sources", []):
            if not isinstance(raw_source, Mapping):
                continue
            doc_id = str(
                raw_source.get("doc_id")
                or raw_source.get("source_name")
                or "unknown-source"
            )
            chunk_id = str(raw_source.get("chunk_id") or "").strip()
            if chunk_id == doc_id or chunk_id.startswith(f"{doc_id}#"):
                source_id = chunk_id
            else:
                source_id = f"{doc_id}#{chunk_id}" if chunk_id else doc_id
            citation_id = _optional_nonnegative_int(raw_source.get("citation_id"))
            if citation_id is not None:
                citation_source_ids[citation_id] = source_id

            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)

            uri_value = raw_source.get("uri") or raw_source.get("path")
            uri = str(uri_value).strip() if uri_value else None
            snippet_value = raw_source.get("snippet")
            snippet = str(snippet_value) if snippet_value is not None else None
            score = cls._trace_source_score(raw_source)
            metadata = {
                key: value
                for key, value in {
                    "citation_id": citation_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id or None,
                    "type": raw_source.get("type"),
                    "hybrid_score": raw_source.get("hybrid_score"),
                    "vector_score": raw_source.get("vector_score"),
                    "bm25_score": raw_source.get("bm25_score"),
                    "rerank_score": raw_source.get("rerank_score"),
                }.items()
                if value is not None
            }
            sources.append(
                SourceTrace(
                    metadata=_json_safe(metadata),
                    source_id=source_id,
                    title=doc_id,
                    uri=uri,
                    snippet=snippet,
                    score=score,
                )
            )

        return sources, citation_source_ids

    @staticmethod
    def _trace_source_score(source: Mapping[str, Any]) -> float | None:
        rerank_score = _optional_nonnegative_float(source.get("rerank_score"))
        if rerank_score is not None and rerank_score > 0:
            return rerank_score
        for key in ("hybrid_score", "score", "vector_score", "bm25_score"):
            score = _optional_nonnegative_float(source.get(key))
            if score is not None:
                return score
        return None

    @classmethod
    def _build_tool_calls(
        cls,
        request: ChatRequest,
        run: _GraphRunResult,
        sources: list[SourceTrace],
    ) -> list[ToolCallTrace]:
        if run.tool_events:
            return [
                cls._tool_call_from_record(record, origin="tool_event")
                for record in run.tool_events
            ]

        normalized_records = run.final_state.get("observed_tool_calls", [])
        if not normalized_records:
            normalized_records = cls._infer_tool_calls(request, run.final_state, sources)
        return [
            cls._tool_call_from_record(record, origin="normalized_from_state")
            for record in normalized_records
            if isinstance(record, Mapping)
        ]

    @staticmethod
    def _tool_call_from_record(
        record: Mapping[str, Any],
        *,
        origin: str,
    ) -> ToolCallTrace:
        arguments = record.get("arguments", {})
        if not isinstance(arguments, Mapping):
            arguments = {"input": arguments}
        error_value = record.get("error")
        error = str(error_value).strip() if error_value else None
        duration_ms = _optional_nonnegative_float(record.get("duration_ms"))
        return ToolCallTrace(
            metadata={"origin": origin},
            name=str(record.get("name") or "unknown_tool"),
            arguments=_json_safe(arguments),
            result=_json_safe(record.get("result")),
            error=error,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _infer_tool_calls(
        request: ChatRequest,
        state: AgentState,
        sources: list[SourceTrace],
    ) -> list[dict[str, Any]]:
        route = state.get("route")
        source_ids = [source.source_id for source in sources]
        if route == "structured_query_tool":
            return [
                {
                    "name": "structured_query_tool",
                    "arguments": {"query": request.question},
                    "result": {
                        "source_name": state.get("structured_query_source_name")
                    },
                }
            ]
        if state.get("compare_path_used") or route == "compare_fast_path":
            return [
                {
                    "name": "compare_retrieval",
                    "arguments": {
                        "targets": state.get("compare_targets"),
                        "aspect": state.get("compare_aspect"),
                    },
                    "result": {"source_ids": source_ids},
                }
            ]
        if route in {"agentic_retrieval", "fallback_retrieval"}:
            return [
                {
                    "name": "hybrid_search",
                    "arguments": {"query": request.question},
                    "result": {"source_ids": source_ids},
                }
            ]
        return []

    @staticmethod
    def _build_usage(run: _GraphRunResult) -> UsageTrace:
        usage_values = dict(run.usage)
        if not any(value is not None for value in usage_values.values()):
            state_usage = run.final_state.get("usage")
            if isinstance(state_usage, Mapping):
                usage_values = _normalize_usage(state_usage)
        observed = any(value is not None for value in usage_values.values())
        values: dict[str, Any] = {
            "input_tokens": _optional_nonnegative_int(usage_values.get("input_tokens")),
            "output_tokens": _optional_nonnegative_int(usage_values.get("output_tokens")),
            "total_tokens": _optional_nonnegative_int(usage_values.get("total_tokens")),
            "cost_usd": _optional_nonnegative_float(usage_values.get("cost_usd")),
        }
        if observed:
            values["metadata"] = {"origin": "langchain_callbacks"}
        return UsageTrace(**values)

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        result = []
        for item in value:
            normalized = str(item).strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    @classmethod
    async def ask_question(cls, request: ChatRequest) -> ChatResponse:
        """通常API向けの後方互換ChatResponseを返す。"""
        run = await cls._run_graph(request)
        return cls._build_chat_response(request, run)

    @classmethod
    async def create_agent_run_trace(
        cls,
        request: ChatRequest,
        *,
        case_id: str,
        target: str = _TRACE_TARGET,
        run_id: str | None = None,
    ) -> AgentRunTrace:
        """1回のGraph実行から評価専用AgentRunTraceを生成する。"""
        run = await cls._run_graph(request)
        return cls._build_agent_run_trace(
            request,
            run,
            case_id=case_id,
            target=target,
            run_id=run_id,
        )

    @classmethod
    async def ask_question_with_trace(
        cls,
        request: ChatRequest,
        *,
        case_id: str,
        target: str = _TRACE_TARGET,
        run_id: str | None = None,
    ) -> tuple[ChatResponse, AgentRunTrace]:
        """同じfinal_stateから通常応答と評価Traceの両方を投影する。"""
        run = await cls._run_graph(request)
        response = cls._build_chat_response(request, run)
        trace = cls._build_agent_run_trace(
            request,
            run,
            case_id=case_id,
            target=target,
            run_id=run_id,
        )
        return response, trace

    # 関数の役割: エージェントが生成したテキストトークンのみをストリーミングする
    # 入出力: ChatRequestを受け取り、文字列トークンのAsyncGeneratorを返す
    @staticmethod
    async def stream_question(request: ChatRequest) -> AsyncGenerator[str, None]:
        if _SETTINGS.answer_critic_retry:
            response = await ChatService.ask_question(request)
            for index in range(0, len(response.answer), 24):
                yield response.answer[index:index + 24]
            return

        inputs = {"messages": [HumanMessage(content=request.question)]}
        config = {"configurable": {"thread_id": request.session_id}}

        async for event in graph.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node")
            if kind == "on_chat_model_stream" and node == "generate":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield chunk

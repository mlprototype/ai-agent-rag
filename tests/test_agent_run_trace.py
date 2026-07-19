import argparse
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from langchain_core.outputs import LLMResult

from application.dto.chat_models import ChatRequest
from application.services.chat_service import (
    ChatService,
    _GraphRunObserver,
    _GraphRunResult,
)
from scripts.run_agent_trace import run as run_trace_cli


def make_run(
    state: dict,
    *,
    latency_ms: int = 42,
    tool_events: list[dict] | None = None,
    usage: dict | None = None,
) -> _GraphRunResult:
    return _GraphRunResult(
        final_state=state,
        total_latency_ms=latency_ms,
        tool_events=tool_events or [],
        usage=usage
        or {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        },
    )


def base_state(**overrides) -> dict:
    state = {
        "answer": "回答",
        "query_type": "direct",
        "route": "direct_answer",
        "confidence": 0.8,
        "warning_codes": [],
        "sources": [],
        "retry_count": 0,
        "fallback_stages": [],
        "timeout_stages": [],
        "skipped_stages": [],
        "fallback_level": "full_path",
        "remaining_budget_ms_at_generate": 3900,
        "route_decision_latency_ms": 2,
        "observed_tool_calls": [],
    }
    state.update(overrides)
    return state


@pytest.mark.anyio
async def test_direct_trace_and_chat_response_share_one_graph_execution():
    request = ChatRequest(session_id="direct-session", question="こんにちは")
    state = base_state(answer="こんにちは！")

    class FakeGraph:
        calls = 0

        async def astream(self, inputs, *, config, stream_mode):
            self.calls += 1
            assert inputs["messages"][0].content == request.question
            assert config["configurable"]["thread_id"] == request.session_id
            assert len(config["callbacks"]) == 1
            assert stream_mode == "values"
            yield state

    fake_graph = FakeGraph()
    with patch("application.services.chat_service.graph", fake_graph):
        response, trace = await ChatService.ask_question_with_trace(
            request,
            case_id="direct-case",
            run_id="direct-run",
        )

    assert fake_graph.calls == 1
    assert response.answer == trace.output.answer == "こんにちは！"
    assert response.route == "direct_answer"
    assert trace.output.route == "direct"
    assert trace.output.metadata["internal_route"] == "direct_answer"
    assert response.confidence is None
    assert trace.output.confidence == 0.8
    assert trace.tool_calls == []
    assert trace.sources == []
    assert trace.usage.input_tokens is None
    assert trace.usage.output_tokens is None
    assert trace.usage.total_tokens is None
    assert trace.usage.cost_usd is None
    assert "passed" not in trace.model_dump()


@pytest.mark.anyio
async def test_structured_query_trace_uses_normalized_tool_fact():
    request = ChatRequest(session_id="structured-session", question="売上の合計は？")
    run = make_run(
        base_state(
            answer="売上の合計は100です。",
            query_type="structured_query",
            route="structured_query_tool",
            confidence=0.95,
            sources=[
                {"source_name": "SQLite (sales)", "type": "structured_data"}
            ],
            structured_query_source_name="SQLite (sales)",
            structured_query_operation="sum",
            structured_query_target_metric="sales",
            structured_query_filters={},
            structured_query_target_dataset="sales",
            observed_tool_calls=[
                {
                    "name": "structured_query_tool",
                    "arguments": {
                        "operation": "sum",
                        "target_metric": "sales",
                        "filters": {},
                        "target_dataset": "sales",
                    },
                    "result": {
                        "success": True,
                        "operation": "sum",
                        "source_name": "SQLite (sales)",
                        "row_count": 1,
                    },
                    "duration_ms": 3,
                }
            ],
        )
    )

    with patch.object(ChatService, "_run_graph", AsyncMock(return_value=run)) as mocked:
        trace = await ChatService.create_agent_run_trace(
            request,
            case_id="structured-case",
            run_id="structured-run",
        )

    mocked.assert_awaited_once_with(request)
    assert trace.output.query_type == "structured_query"
    assert trace.output.route == "structured_query"
    assert trace.output.metadata["internal_route"] == "structured_query_tool"
    assert trace.tool_calls[0].name == "structured_query_tool"
    assert trace.tool_calls[0].metadata["origin"] == "normalized_from_state"
    assert trace.tool_calls[0].arguments == {
        "operation": "sum",
        "target_metric": "sales",
        "filters": {},
        "target_dataset": "sales",
    }
    assert trace.sources[0].source_id == "SQLite (sales)"


@pytest.mark.anyio
async def test_agentic_retrieval_trace_maps_citations_and_control():
    request = ChatRequest(session_id="retrieval-session", question="RAGとは？")
    source_1 = {
        "citation_id": 1,
        "doc_id": "rag.md",
        "chunk_id": "overview",
        "snippet": "RAGは検索拡張生成です。",
        "hybrid_score": 0.91,
        "vector_score": 0.8,
        "bm25_score": 0.7,
        "rerank_score": 0.95,
    }
    source_2 = {
        "citation_id": 2,
        "doc_id": "rag.md",
        "chunk_id": "details",
        "snippet": "外部情報を検索します。",
        "hybrid_score": 0.75,
    }
    run = make_run(
        base_state(
            answer="RAGは検索拡張生成です。[1]",
            query_type="definition",
            route="agentic_retrieval",
            confidence=0.87,
            warning_codes=["partial_retrieval_used"],
            sources=[source_1, source_2],
            retry_count=1,
            fallback_stages=["decompose"],
            timeout_stages=["rewrite"],
            skipped_stages=["rerank"],
            fallback_level="single_retrieval_fallback",
            remaining_budget_ms_at_generate=1240,
            observed_tool_calls=[
                {
                    "name": "hybrid_search",
                    "arguments": {"query": "RAGとは？", "top_k": 5},
                    "result": {"source_ids": ["rag.md#overview", "rag.md#details"]},
                    "duration_ms": 11,
                }
            ],
        ),
        usage={
            "input_tokens": 20,
            "output_tokens": 8,
            "total_tokens": 28,
            "cost_usd": None,
        },
    )

    with patch.object(ChatService, "_run_graph", AsyncMock(return_value=run)):
        response, trace = await ChatService.ask_question_with_trace(
            request,
            case_id="retrieval-case",
            run_id="retrieval-run",
        )

    assert response.sources is not None
    assert response.route == "agentic_retrieval"
    assert trace.output.route == "retrieval"
    assert trace.output.metadata["internal_route"] == "agentic_retrieval"
    assert trace.tool_calls[0].arguments == {"query": "RAGとは？", "top_k": 5}
    assert [source.citation_id for source in response.sources] == [1]
    assert [source.source_id for source in trace.sources] == [
        "rag.md#overview",
        "rag.md#details",
    ]
    assert trace.sources[0].metadata["citation_id"] == 1
    assert trace.citations[0].citation_id == "1"
    assert trace.citations[0].source_id == "rag.md#overview"
    assert trace.control.attempt_count == 2
    assert trace.control.retry_count == 1
    assert trace.control.fallback_used is True
    assert trace.control.fallback_stages == ["decompose"]
    assert trace.control.timeout_stages == ["rewrite"]
    assert trace.control.skipped_stages == ["rerank"]
    assert trace.control.fallback_level == "single_retrieval_fallback"
    assert trace.control.remaining_budget_ms_at_generate == 1240
    assert trace.usage.total_tokens == 28
    assert trace.usage.cost_usd is None
    assert trace.usage.metadata["origin"] == "langchain_callbacks"
    assert trace.timing.tool_latency_ms == 11


@pytest.mark.anyio
async def test_compare_trace_preserves_compare_route_and_tool_fact():
    request = ChatRequest(
        session_id="compare-session",
        question="RAGとFine-tuningの違いは？",
    )
    run = make_run(
        base_state(
            answer="RAGは検索を使い[1]、Fine-tuningは再学習します[2]。",
            query_type="compare",
            route="compare_fast_path",
            confidence=0.8,
            compare_path_used=True,
            compare_targets={"target_a": "RAG", "target_b": "Fine-tuning"},
            compare_aspect="違い",
            sources=[
                {
                    "citation_id": 1,
                    "doc_id": "rag.md",
                    "chunk_id": "rag",
                    "snippet": "RAG",
                    "hybrid_score": 0.9,
                },
                {
                    "citation_id": 2,
                    "doc_id": "ft.md",
                    "chunk_id": "ft",
                    "snippet": "Fine-tuning",
                    "hybrid_score": 0.88,
                },
            ],
            observed_tool_calls=[
                {
                    "name": "compare_retrieval",
                    "arguments": {
                        "targets": ["RAG", "Fine-tuning"],
                        "aspect": "違い",
                    },
                    "result": {
                        "RAG": {"source_ids": ["rag.md#rag"]},
                        "Fine-tuning": {"source_ids": ["ft.md#ft"]},
                    },
                    "duration_ms": 14,
                }
            ],
        )
    )

    with patch.object(ChatService, "_run_graph", AsyncMock(return_value=run)):
        trace = await ChatService.create_agent_run_trace(
            request,
            case_id="compare-case",
            run_id="compare-run",
        )

    assert trace.output.query_type == "compare"
    assert trace.output.route == "compare"
    assert trace.output.metadata["internal_route"] == "compare_fast_path"
    assert trace.tool_calls[0].name == "compare_documents"
    assert trace.tool_calls[0].metadata["origin"] == "normalized_from_state"
    assert trace.tool_calls[0].metadata["internal_tool_name"] == "compare_retrieval"
    assert trace.tool_calls[0].arguments == {
        "left": "RAG",
        "right": "Fine-tuning",
        "aspects": ["違い"],
    }
    assert {citation.citation_id for citation in trace.citations} == {"1", "2"}


def test_real_and_normalized_tool_events_are_merged():
    request = ChatRequest(session_id="event-session", question="実Toolを使う")
    run = make_run(
        base_state(
            route="structured_query_tool",
            query_type="structured_query",
            observed_tool_calls=[{"name": "normalized_tool"}],
        ),
        tool_events=[
            {
                "name": "real_tool",
                "arguments": {"value": 1},
                "result": {"ok": True},
                "duration_ms": 5,
            }
        ],
    )

    trace = ChatService._build_agent_run_trace(
        request,
        run,
        case_id="event-case",
        run_id="event-run",
    )

    assert [tool.name for tool in trace.tool_calls] == [
        "real_tool",
        "normalized_tool",
    ]
    assert trace.tool_calls[0].metadata["origin"] == "tool_event"
    assert trace.tool_calls[1].metadata["origin"] == "normalized_from_state"


def test_duplicate_real_and_normalized_tool_events_prefer_real_event():
    request = ChatRequest(session_id="event-session", question="検索する")
    duplicate = {
        "name": "hybrid_search",
        "arguments": {"query": "検索する", "top_k": 5},
    }
    run = make_run(
        base_state(
            route="agentic_retrieval",
            observed_tool_calls=[duplicate],
        ),
        tool_events=[duplicate],
    )

    trace = ChatService._build_agent_run_trace(
        request,
        run,
        case_id="dedupe-case",
        run_id="dedupe-run",
    )

    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].metadata["origin"] == "tool_event"


def test_usage_from_agent_state_reports_accurate_origin():
    request = ChatRequest(session_id="usage-session", question="usage")
    run = make_run(
        base_state(
            usage={
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            }
        )
    )

    trace = ChatService._build_agent_run_trace(
        request,
        run,
        case_id="usage-case",
        run_id="usage-run",
    )

    assert trace.usage.input_tokens == 4
    assert trace.usage.output_tokens == 2
    assert trace.usage.total_tokens == 6
    assert trace.usage.metadata["origin"] == "agent_state"


def test_fallback_retrieval_route_is_common_and_marked_degraded():
    request = ChatRequest(session_id="fallback-session", question="検索する")
    run = make_run(
        base_state(
            route="fallback_retrieval",
            query_type="definition",
            retrieval_top_k=3,
        )
    )

    trace = ChatService._build_agent_run_trace(
        request,
        run,
        case_id="fallback-case",
        run_id="fallback-run",
    )

    assert trace.output.route == "retrieval"
    assert trace.output.metadata == {
        "internal_route": "fallback_retrieval",
        "degraded": True,
    }
    assert trace.tool_calls[0].arguments == {"query": "検索する", "top_k": 3}


def test_graph_observer_collects_tool_events_and_available_usage():
    observer = _GraphRunObserver()
    tool_run_id = uuid4()
    observer.on_tool_start(
        {"name": "real_tool"},
        '{"value": 1}',
        run_id=tool_run_id,
    )
    observer.on_tool_end({"ok": True}, run_id=tool_run_id)
    observer.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                }
            },
        ),
        run_id=uuid4(),
    )

    assert observer.tool_calls[0]["name"] == "real_tool"
    assert observer.tool_calls[0]["arguments"] == {"value": 1}
    assert observer.tool_calls[0]["result"] == {"ok": True}
    assert observer.tool_calls[0]["duration_ms"] >= 0
    assert observer.usage == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
        "cost_usd": None,
    }


@pytest.mark.anyio
async def test_trace_cli_writes_single_contract_object_with_null_usage(tmp_path):
    request = ChatRequest(session_id="unused", question="こんにちは")
    trace = ChatService._build_agent_run_trace(
        request,
        make_run(base_state(answer="こんにちは！")),
        case_id="cli-case",
        run_id="cli-run",
    )
    output = tmp_path / "trace.json"
    args = argparse.Namespace(
        case_id="cli-case",
        question="こんにちは",
        output=output,
        session_id="cli-session",
        target="ai-agent-rag",
    )

    with patch.object(
        ChatService,
        "create_agent_run_trace",
        AsyncMock(return_value=trace),
    ) as mocked:
        written_path = await run_trace_cli(args)

    mocked.assert_awaited_once()
    payload = json.loads(written_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["case_id"] == "cli-case"
    assert payload["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
    }
    assert "passed" not in payload

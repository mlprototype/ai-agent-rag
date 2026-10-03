import importlib
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from application.dto.chat_models import ChatRequest
from application.services.chat_service import ChatService
from domain.services.answer_critic import AnswerVerdict
from domain.services.heuristic_router import HeuristicRouter

agent = importlib.import_module("application.agents.graph")
chat = importlib.import_module("application.services.chat_service")


@pytest.mark.anyio
@pytest.mark.parametrize("branch", ["skip", "timeout"])
@pytest.mark.parametrize("prior,strict,supported,expected", [
    (False, False, True, False),
    (True, True, True, False),
    (True, False, True, True),
    (True, False, False, False),
])
async def test_critic_skip_cannot_promote_fail(branch, prior, strict, supported, expected):
    state = {
        "original_query": "RAGとは？", "route": "agentic_retrieval",
        "answer_ok": prior, "strict_insufficient_response": strict,
        "sources": [{"snippet": "RAG" if supported else "unrelated"}],
        "initial_budget_ms": 10000, "budget_started_at": time.monotonic(),
        "confidence": 0.7,
    }
    with patch.object(agent, "_should_skip_answer_critic", return_value="remaining_budget_low" if branch == "skip" else None), patch.object(agent, "_stage_timeout_seconds", return_value=0):
        updates = await agent.answer_critic_node(state)
    assert updates["answer_ok"] is expected


@pytest.mark.anyio
async def test_insufficient_then_direct_turn_resets_guardrail_and_usage():
    request = ChatRequest(session_id=str(uuid4()), question="おすすめは？")
    empty = {"context": "", "sources": [], "confidence": 0.0, "chunks": [], "top_k": 0}
    decision = SimpleNamespace(route="agentic_retrieval", reason="test", query_type="retrieval_complex", routing_layer="heuristic", source="heuristic_match", heuristic_matched=True, heuristic_rule="test", confidence=0.9, llm_router_invoked=False)
    with patch.object(agent.AgentRouter, "route", AsyncMock(return_value=decision)), patch.object(agent.RetrievalService, "run", AsyncMock(return_value=empty)), patch.object(agent, "_get_generate_chain"), patch.object(agent.AnswerCritic, "verify", AsyncMock(return_value=AnswerVerdict(verdict="PASS"))):
        _, first = await ChatService.ask_question_with_trace(request, case_id="insufficient")
    assert first.guardrail.blocked is True
    first_state = await agent.graph.aget_state({"configurable": {"thread_id": request.session_id}})
    assert first_state.values["answer_ok"] is False

    direct = HeuristicRouter.route("こんにちは")
    chain = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")]) | FakeListChatModel(responses=["こんにちは！"])
    request.question = "こんにちは"
    with patch.object(agent.AgentRouter, "route", AsyncMock(return_value=direct)), patch.object(agent, "_get_direct_chain", return_value=chain):
        response, second = await ChatService.ask_question_with_trace(request, case_id="direct")
    assert response.answer == "こんにちは！"
    assert second.guardrail.blocked is False
    assert second.control.stop_reason != "insufficient_retrieval"
    state = await agent.graph.aget_state({"configurable": {"thread_id": request.session_id}})
    assert state.values["strict_insufficient_response"] is False
    assert len(state.values["messages"]) == 4

    initialized = await agent.initialize_node({"messages": [HumanMessage(content="こんにちは")], "usage": {"input_tokens": 999}})
    assert initialized["usage"] == {}
    assert "messages" not in initialized


@pytest.mark.anyio
@pytest.mark.parametrize("node,tokens,final", [
    ("direct_generate", [], "こんにちは！"),
    ("structured_query_node", [], "合計は100です。"),
    ("compare_generate", [], "AとBの比較です。"),
    ("generate", [], "検索結果に十分な情報が見つかりませんでした。"),
    ("generate", ["回答", "です。"], "回答です。"),
    ("generate", ["回答"], "回答です。"),
])
async def test_stream_final_answer_complete_in_one_execution(node, tokens, final):
    class Graph:
        calls = 0
        async def astream_events(self, *args, **kwargs):
            self.calls += 1
            for token in tokens:
                yield {"event": "on_chat_model_stream", "metadata": {"langgraph_node": node}, "data": {"chunk": SimpleNamespace(content=token)}, "parent_ids": ["root"]}
            yield {"event": "on_chain_end", "metadata": {}, "parent_ids": [], "data": {"output": {"answer": final}}}
    graph = Graph()
    with patch.object(chat, "graph", graph), patch.object(chat, "_SETTINGS", replace(chat._SETTINGS, answer_critic_retry=False)):
        parts = [part async for part in ChatService.stream_question(ChatRequest(session_id="stream", question="Q"))]
    assert "".join(parts) == final
    assert graph.calls == 1


@pytest.mark.anyio
async def test_definition_stream_uses_guarded_final_text():
    final = "検索結果にこの用語を直接説明する十分な情報が見つかりませんでした。"
    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield {"event": "on_chain_start", "metadata": {"langgraph_node": "generate"}, "data": {"input": {"query_type": "definition"}}, "parent_ids": ["root"]}
            yield {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "generate"}, "data": {"chunk": SimpleNamespace(content="未確定の説明")}, "parent_ids": ["root"]}
            yield {"event": "on_chain_end", "metadata": {}, "parent_ids": [], "data": {"output": {"answer": final}}}
    with patch.object(chat, "graph", Graph()):
        parts = [part async for part in ChatService.stream_question(ChatRequest(session_id="stream", question="Q"))]
    assert parts == [final]


@pytest.mark.anyio
async def test_real_graph_event_stream_returns_direct_answer_once():
    chain = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")]) | FakeListChatModel(responses=["こんにちは！"])
    with patch.object(agent, "_get_direct_chain", return_value=chain):
        parts = [part async for part in ChatService.stream_question(ChatRequest(session_id=str(uuid4()), question="こんにちは"))]
    assert parts == ["こんにちは！"]


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["retrieval", "strict", "definition", "structured", "compare"])
async def test_real_graph_stream_paths_use_one_execution(mode):
    from domain.models.retrieval_models import RetrievedChunk
    from domain.services.retrieval_critic import CritiqueResult
    from domain.services.structured_query_types import StructuredQueryResult
    query = "AとBの違い" if mode == "compare" else "Xとは？" if mode == "definition" else "検索して"
    decision = SimpleNamespace(
        route="structured_query_tool" if mode == "structured" else "agentic_retrieval",
        reason="test", query_type="compare" if mode == "compare" else "definition" if mode == "definition" else "retrieval_complex",
        routing_layer="heuristic", source="heuristic_match", heuristic_matched=True,
        heuristic_rule="test", confidence=0.9, llm_router_invoked=False,
    )
    chunk = RetrievedChunk(doc_id="x", chunk_id="1", content="Xの根拠", hybrid_score=0.8, bm25_score=0.0 if mode == "definition" else 0.8)
    retrieval = {"context": "Xの根拠[1]", "sources": [{"citation_id": 1, "doc_id": "x", "chunk_id": "1"}], "confidence": 0.1 if mode == "definition" else 0.8, "chunks": [chunk], "top_k": 1}
    if mode == "strict":
        retrieval = {"context": "", "sources": [], "confidence": 0.0, "chunks": [], "top_k": 0}
    chain = ChatPromptTemplate.from_messages([("human", "{retrieval_context}")]) | FakeListChatModel(responses=["回答です。[1]"])
    compare = {target: retrieval for target in ("A", "B")}
    structured = StructuredQueryResult(success=True, operation="sum", target_metric="sales", filters={}, rows=[{"result": 100}], summary="合計は100です。", source_name="SQLite (sales)", target_dataset="sales")
    real_graph = agent.graph
    class Spy:
        calls = 0
        async def astream_events(self, *args, **kwargs):
            self.calls += 1
            async for event in real_graph.astream_events(*args, **kwargs):
                yield event
    spy = Spy()
    with (
        patch.object(chat, "graph", spy),
        patch.object(agent.AgentRouter, "route", AsyncMock(return_value=decision)),
        patch.object(agent.RetrievalService, "run", AsyncMock(return_value=retrieval)),
        patch.object(agent.RetrievalCritic, "critique", AsyncMock(return_value=CritiqueResult(verdict="SUFFICIENT"))),
        patch.object(agent.AnswerCritic, "verify", AsyncMock(return_value=AnswerVerdict(verdict="PASS"))),
        patch.object(agent, "_get_generate_chain", return_value=chain),
        patch.object(agent, "run_compare_retrieval", AsyncMock(return_value=compare)),
        patch.object(agent, "ChatOpenAI", return_value=FakeListChatModel(responses=["共通点: AとB。相違点: AとB。[1]"])),
        patch.object(agent.StructuredQueryTool, "run", return_value=structured),
    ):
        request = ChatRequest(session_id=str(uuid4()), question=query)
        parts = [part async for part in ChatService.stream_question(request)]
    final = await real_graph.aget_state({"configurable": {"thread_id": request.session_id}})
    assert "".join(parts) == final.values["answer"]
    assert parts
    assert spy.calls == 1
    if mode == "definition":
        assert "十分な情報が見つかりません" in final.values["answer"]
        assert len(parts) == 1

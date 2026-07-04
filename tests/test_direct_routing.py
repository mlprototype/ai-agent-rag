import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from application.agents.graph import (
    commit_answer_node,
    direct_generate_node,
    route_after_router,
)
from application.agents.state import AgentState


def create_base_state(route: str, query: str, query_type: str = "direct") -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=query)],
        original_query=query,
        route=route,
        query_type=query_type,
        initial_budget_ms=15000,
        budget_started_at=time.monotonic(),
        fallback_level="full_path",
        warning_codes=[],
        budget_pressure_reasons=[],
        timeout_stages=[],
        fallback_stages=[],
        skipped_stages=[],
    )


@pytest.mark.anyio
@patch("application.agents.graph._get_direct_chain")
async def test_direct_generate_node_uses_llm_chain(mock_get_chain):
    state = create_base_state("direct_answer", "こんにちは")

    mock_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "こんにちは！"
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
    mock_get_chain.return_value = mock_chain

    result = await direct_generate_node(state)

    mock_chain.ainvoke.assert_awaited_once()
    assert result["answer"] == "こんにちは！"
    assert result["confidence"] == 0.8
    assert result["answer_ok"] is True


def test_route_after_router_sends_direct_answer_to_generate():
    state = create_base_state("direct_answer", "こんにちは")
    assert route_after_router(state) == "direct_generate"


@pytest.mark.anyio
async def test_commit_answer_holds_answer():
    state = create_base_state("direct_answer", "ありがとう")
    state["answer"] = "どういたしまして！"
    result = await commit_answer_node(state)
    assert result["messages"][0].content == "どういたしまして！"

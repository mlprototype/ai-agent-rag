import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from application.agents.graph import (
    commit_answer_node,
    direct_generate_node,
    route_after_router,
)
from application.agents.state import AgentState

"""
このファイルの責務は、direct および calc 判定時に適切に応答が生成・保持されることを確認することです。
特に calculator route 廃止後、direct_answer ルート内で決定論的に計算が行われることを検証します。
"""

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
async def test_direct_generate_node_greeting(mock_get_chain):
    """挨拶などの通常の direct_answer のテスト"""
    state = create_base_state("direct_answer", "こんにちは", "direct")
    
    mock_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "こんにちは！"
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
    mock_get_chain.return_value = mock_chain
    
    result = await direct_generate_node(state)
    
    assert result["answer"] == "こんにちは！"
    assert result["confidence"] == 0.8
    assert result["answer_ok"] is True

@pytest.mark.anyio
async def test_direct_generate_node_calc_simple():
    """数式の場合、direct_generate_node 内で決定論的に計算されることをテスト (1+1)"""
    state = create_base_state("direct_answer", "1+1", "calc")
    
    # チェーンは呼び出されないはず（決定論的に返る）
    with patch("application.agents.graph._get_direct_chain") as mock_get_chain:
        result = await direct_generate_node(state)
        mock_get_chain.assert_not_called()
    
    assert result["answer"] == "計算結果は 2 です。"
    assert result["confidence"] == 1.0
    assert result["answer_ok"] is True

@pytest.mark.anyio
async def test_direct_generate_node_calc_complex():
    """複雑な数式のテスト (2+3*4)"""
    state = create_base_state("direct_answer", "2+3*4", "calc")
    result = await direct_generate_node(state)
    assert result["answer"] == "計算結果は 14 です。"

@pytest.mark.anyio
async def test_direct_generate_node_calc_japanese():
    """日本語の数式テスト (1足す1)"""
    state = create_base_state("direct_answer", "1足す1", "calc")
    result = await direct_generate_node(state)
    assert result["answer"] == "計算結果は 2 です。"

@pytest.mark.anyio
async def test_route_after_router_integration():
    """calculator ルートが廃止され、direct_answer に統合されていることをテスト"""
    state_direct = create_base_state("direct_answer", "こんにちは", "direct")
    assert route_after_router(state_direct) == "direct_generate"
    
    # 以前は calculator を返していたが、今は direct_generate を返すべき
    state_calc = create_base_state("direct_answer", "1+1", "calc")
    assert route_after_router(state_calc) == "direct_generate"

@pytest.mark.anyio
async def test_commit_answer_holds_answer():
    state = create_base_state("direct_answer", "ありがとう")
    state["answer"] = "どういたしまして！"
    result = await commit_answer_node(state)
    assert result["messages"][0].content == "どういたしまして！"

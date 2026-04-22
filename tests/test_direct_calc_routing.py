import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from application.agents.graph import (
    calc_generate_node,
    commit_answer_node,
    direct_generate_node,
    route_after_router,
)
from application.agents.state import AgentState

"""
このファイルの責務は、direct および calc ルートが検索フェールセーフによって上書きされず、
適切な応答を生成・保持することを確認することです。
主な入出力: ダミーの AgentState を各ノードに入力し、更新後の差分を検証します。
設計上の注意点: LLM を使用する direct_generate_node についてはモックを使用し、API 呼び出しを防ぎます。
"""


def create_base_state(route: str, query: str) -> AgentState:
    """
    テスト用の初期状態を生成するヘルパー関数。
    入力: ルート名とクエリ文字列
    出力: 必須フィールドを埋めた AgentState
    """
    return AgentState(
        original_query=query,
        route=route,
        query_type="direct" if route == "direct_answer" else "calc",
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
async def test_direct_generate_node_success(mock_get_chain):
    """
    direct_answer ルートで、応答が検索不足エラーで上書きされず、
    正しく直接応答として返ることをテストする。
    """
    state = create_base_state("direct_answer", "こんにちは")
    state["messages"] = []
    
    mock_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "こんにちは！何かお手伝いしましょうか？"
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
    mock_get_chain.return_value = mock_chain
    
    result = await direct_generate_node(state)
    
    assert result["answer"] == "こんにちは！何かお手伝いしましょうか？"
    assert result["answer_ok"] is True
    assert result["warning"] is None
    # directルートにはsourcesは存在しない（更新差分に含まれない）
    assert "sources" not in result
    assert result["confidence"] == 0.8


@pytest.mark.anyio
async def test_calc_generate_node_success():
    """
    calculator ルートで、計算結果が自然な文にフォーマットされ、
    不要な warning や sources が付与されないことをテストする。
    """
    state = create_base_state("calculator", "2+3*4")
    state["calculator_result"] = "14"
    state["confidence"] = 0.98
    
    result = await calc_generate_node(state)
    
    assert result["answer"] == "計算結果は 14 です。"
    assert result["answer_ok"] is True
    assert result["warning"] is None
    # calcルートにはsourcesは存在しない（更新差分に含まれない）
    assert "sources" not in result
    assert result["confidence"] == 0.98


@pytest.mark.anyio
async def test_route_after_router_separation():
    """
    ルーティングロジックが direct_answer と calculator を
    generate ノードから分離していることをテストする。
    """
    state_direct = create_base_state("direct_answer", "こんにちは")
    assert route_after_router(state_direct) == "direct_generate"
    
    state_calc = create_base_state("calculator", "2+3*4")
    assert route_after_router(state_calc) == "calculator"


@pytest.mark.anyio
async def test_commit_answer_holds_answer():
    """
    commit_answer が受け取った answer を正しく messages に反映し、
    RAG不足文言で上書きされることなく結果が保持されることをテストする。
    """
    state = create_base_state("direct_answer", "ありがとう")
    state["answer"] = "どういたしまして！"
    
    result = await commit_answer_node(state)
    
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "どういたしまして！"

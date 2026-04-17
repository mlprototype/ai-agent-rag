import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import time

from application.agents.state import AgentState
from application.agents.graph import decompose_node, parallel_retrieve_node, merge_node, generate_node
from domain.models.retrieval_models import RetrievedChunk
from config.settings import get_settings

_SETTINGS = get_settings()

def create_base_state() -> AgentState:
    return AgentState(
        original_query="What is the difference between RAG and Fine-tuning?",
        route="agentic_retrieval",
        sub_queries=["RAG", "Fine-tuning"],
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
async def test_decompose_timeout_triggers_single_retrieval_fallback():
    state = create_base_state()
    # Force timeout by setting budget_started_at to way in the past
    state["budget_started_at"] = time.monotonic() - 20
    
    result = await decompose_node(state)
    
    assert result["sub_queries"] == [state["original_query"]]
    assert result["fallback_level"] == "single_retrieval_fallback"
    assert "decompose_timeout" in result["warning_codes"]

@pytest.mark.anyio
@patch("application.agents.graph.RetrievalService.search")
async def test_partial_retrieval_success_continues(mock_search):
    state = create_base_state()
    
    class TimeoutSearch:
        def __await__(self):
            raise asyncio.TimeoutError()
            yield
            
    mock_result = MagicMock()
    mock_result.selected_chunks = [RetrievedChunk(chunk_id="1", doc_id="1", content="content")]
    
    async def mock_search_impl(query):
        if query == "RAG":
            return mock_result
        raise asyncio.TimeoutError()
        
    mock_search.side_effect = mock_search_impl
    
    result = await parallel_retrieve_node(state)
    
    assert result["partial_retrieval_used"] is True
    assert result["retrieval_success_count"] == 1
    assert result["retrieval_timeout_count"] == 1
    assert "partial_retrieval_used" in result["warning_codes"]
    assert result["fallback_level"] == "full_path"

@pytest.mark.anyio
@patch("application.agents.graph.RetrievalService.search")
async def test_all_subqueries_fail_triggers_single_retrieval_fallback(mock_search):
    state = create_base_state()
    
    async def mock_search_impl(query):
        raise asyncio.TimeoutError()
        
    mock_search.side_effect = mock_search_impl
    
    result = await parallel_retrieve_node(state)
    
    assert result["partial_retrieval_used"] is False
    assert result["retrieval_success_count"] == 0
    assert result["retrieval_timeout_count"] == 2
    assert "all_subqueries_failed" in result["warning_codes"]
    assert result["fallback_level"] == "single_retrieval_fallback"

@pytest.mark.anyio
@patch("application.agents.graph.RetrievalService.search")
async def test_single_retrieval_fails_triggers_minimal_answer(mock_search):
    state = create_base_state()
    state["sub_queries"] = [state["original_query"]]
    
    async def mock_search_impl(query):
        raise asyncio.TimeoutError()
        
    mock_search.side_effect = mock_search_impl
    
    result = await parallel_retrieve_node(state)
    
    assert result["retrieval_success_count"] == 0
    assert result["fallback_level"] == "minimal_answer"

@pytest.mark.anyio
async def test_merge_empty_results_triggers_minimal_answer():
    state = create_base_state()
    state["parallel_results"] = []
    
    result = await merge_node(state)
    
    assert result["fallback_level"] == "minimal_answer"
    assert "empty_retrieval_context" in result["warning_codes"]

@pytest.mark.anyio
@patch("application.agents.graph._get_generate_chain")
async def test_generate_emits_minimal_answer_warning(mock_get_chain):
    state = create_base_state()
    state["fallback_level"] = "minimal_answer"
    state["working_chunks"] = []
    state["sources"] = []
    
    mock_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Minimal Answer"
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
    mock_get_chain.return_value = mock_chain
    
    result = await generate_node(state)
    
    assert result["warning"] == "十分な検索結果が得られず、最小限の回答を行いました。"
    assert result["retrieval_quality_level"] == "low"

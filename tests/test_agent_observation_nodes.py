import time
from unittest.mock import AsyncMock, patch

import pytest

from application.agents.graph import retrieve_node, structured_query_node
from domain.services.structured_query_types import StructuredQueryResult


@pytest.mark.anyio
async def test_retrieve_node_records_actual_top_k():
    retrieval_result = {
        "context": "検索コンテキスト",
        "sources": [],
        "confidence": 0.8,
        "chunks": [],
        "top_k": 5,
    }
    state = {
        "original_query": "RAGとは？",
        "route": "agentic_retrieval",
        "initial_budget_ms": 5000,
        "budget_started_at": time.monotonic(),
        "observed_tool_calls": [],
    }

    with patch(
        "application.agents.graph.RetrievalService.run",
        AsyncMock(return_value=retrieval_result),
    ):
        updates = await retrieve_node(state)

    assert updates["retrieval_top_k"] == 5
    assert updates["observed_tool_calls"][-1]["arguments"] == {
        "query": "RAGとは？",
        "top_k": 5,
    }


@pytest.mark.anyio
async def test_structured_query_node_records_executed_semantics():
    structured_result = StructuredQueryResult(
        success=True,
        operation="count",
        target_metric="units_sold",
        filters={"period": "2025-Q1"},
        rows=[{"result": 3}],
        summary="該当するデータは 3 件です。",
        source_name="SQLite (sales)",
        target_dataset="sales",
    )
    state = {
        "original_query": "Q1の注文件数は？",
        "route": "structured_query_tool",
        "initial_budget_ms": 5000,
        "budget_started_at": time.monotonic(),
        "observed_tool_calls": [],
    }

    with patch(
        "application.agents.graph.StructuredQueryTool.run",
        return_value=structured_result,
    ):
        updates = await structured_query_node(state)

    assert updates["answer"] == "該当するデータは 3 件です。"
    assert updates["observed_tool_calls"][-1]["arguments"] == {
        "operation": "count",
        "target_metric": "units_sold",
        "filters": {"period": "2025-Q1"},
        "target_dataset": "sales",
    }

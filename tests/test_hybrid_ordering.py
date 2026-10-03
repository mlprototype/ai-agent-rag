from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from domain.services.hybrid_search import HybridSearch, _distance_to_similarity, _normalize_scores
from infrastructure.retrieval import vector_store


def test_distance_conversion_is_monotonic():
    scores = [_distance_to_similarity(d) for d in (0.1, 0.9, 1.2)]
    assert scores[0] > scores[1] > scores[2]
    normalized = _normalize_scores(scores)
    assert normalized[0] > normalized[1] > normalized[2]
    assert _normalize_scores([_distance_to_similarity(1.2)]) == [0.5]
    assert _normalize_scores([_distance_to_similarity(0.9)] * 2) == [0.5, 0.5]


@pytest.mark.anyio
async def test_hybrid_fusion_preserves_vector_order_and_disposes_pool():
    engine = SimpleNamespace(dispose=AsyncMock())
    store = SimpleNamespace(asimilarity_search_with_score=AsyncMock(return_value=[
        (Document(page_content=name, metadata={"source": name}), distance)
        for name, distance in (("near", 0.1), ("middle", 0.9), ("far", 1.2))
    ]))
    with patch.object(vector_store, "_create_vector_engine", return_value=engine), patch.object(vector_store, "get_async_vector_store", return_value=store), patch("domain.services.hybrid_search.KeywordSearch.search", AsyncMock(return_value=[])):
        chunks = await HybridSearch.search(["Q"])
    assert [chunk.content for chunk in chunks] == ["near", "middle", "far"]
    engine.dispose.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["query", "construction", "cancel"])
async def test_vector_engine_disposed_on_failure_and_cancellation(failure):
    import asyncio
    engine = SimpleNamespace(dispose=AsyncMock())
    with patch.object(vector_store, "_create_vector_engine", return_value=engine), patch.object(vector_store, "get_async_vector_store", side_effect=ValueError("construction") if failure == "construction" else None):
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else ValueError):
            async with vector_store.managed_async_vector_store():
                if failure == "cancel":
                    raise asyncio.CancelledError()
                raise ValueError("query")
    engine.dispose.assert_awaited_once()


@pytest.mark.anyio
async def test_repeated_requests_close_every_owned_pool_on_its_event_loop():
    import asyncio
    engines = []
    loops = []
    def create_engine():
        async def dispose():
            loops.append(asyncio.get_running_loop())
        engine = SimpleNamespace(dispose=AsyncMock(side_effect=dispose))
        engines.append(engine)
        return engine
    store = SimpleNamespace(asimilarity_search_with_score=AsyncMock(return_value=[]))
    with patch.object(vector_store, "_create_vector_engine", side_effect=create_engine), patch.object(vector_store, "get_async_vector_store", return_value=store), patch("domain.services.hybrid_search.KeywordSearch.search", AsyncMock(return_value=[])):
        for _ in range(20):
            await HybridSearch.search(["original", "rewrite"])
    assert len(engines) == 40
    assert all(engine.dispose.await_count == 1 for engine in engines)
    assert all(loop is asyncio.get_running_loop() for loop in loops)

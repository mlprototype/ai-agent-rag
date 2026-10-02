import asyncio
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from domain.models.retrieval_models import RetrievedChunk
from infrastructure.retrieval import reranker


def chunks():
    return [
        RetrievedChunk(doc_id="a", chunk_id="1", content="first", hybrid_score=0.9),
        RetrievedChunk(doc_id="b", chunk_id="2", content="second", hybrid_score=0.6),
    ]


class RerankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_uses_passthrough_without_cohere_sdk(self):
        with (
            patch.object(reranker, "get_settings", return_value=SimpleNamespace(enable_rerank=False)),
            patch.dict(sys.modules, {"cohere": None}),
        ):
            selected = reranker.build_reranker()
            self.assertIsInstance(selected, reranker.PassthroughReranker)
            result = await selected.rerank("Q", chunks(), 1)
        self.assertEqual([chunk.content for chunk in result], ["first"])
        self.assertEqual(result[0].rerank_score, 0.9)

    async def test_enabled_without_api_key_warns_and_falls_back(self):
        with (
            patch.object(reranker, "get_settings", return_value=SimpleNamespace(enable_rerank=True, cohere_api_key="")),
            self.assertLogs(reranker.logger, level="WARNING") as logs,
        ):
            selected = reranker.build_reranker()
            result = await selected.rerank("Q", chunks(), 1)
        self.assertIsInstance(selected, reranker.CohereReranker)
        self.assertIsNone(selected._client)
        self.assertIn("Passthrough", logs.output[0])
        self.assertEqual(result[0].rerank_score, 0.9)

    async def test_import_failure_warns_and_uses_passthrough(self):
        with patch.dict(sys.modules, {"cohere": None}), self.assertLogs(reranker.logger, level="WARNING") as logs:
            selected = reranker.CohereReranker(api_key="test")
            result = await selected.rerank("Q", chunks(), 1)
        self.assertIn("Passthrough", logs.output[0])
        self.assertEqual(result[0].rerank_score, 0.9)

    async def test_client_failure_warns_and_uses_passthrough(self):
        client = Mock()
        client.rerank.side_effect = RuntimeError("mock client failure")
        with (
            patch.dict(sys.modules, {"cohere": SimpleNamespace(Client=Mock(return_value=client))}),
            patch.object(reranker, "get_settings", return_value=SimpleNamespace(rerank_timeout_seconds=1)),
            self.assertLogs(reranker.logger, level="WARNING") as logs,
        ):
            selected = reranker.CohereReranker(api_key="test")
            result = await selected.rerank("Q", chunks(), 1)
        self.assertIn("Passthrough", logs.output[0])
        self.assertEqual(result[0].rerank_score, 0.9)

    async def test_enabled_client_results_are_used(self):
        client = Mock()
        client.rerank.return_value = SimpleNamespace(results=[SimpleNamespace(index=1, relevance_score=0.97)])
        settings = SimpleNamespace(enable_rerank=True, cohere_api_key="test", rerank_timeout_seconds=1)
        with (
            patch.dict(sys.modules, {"cohere": SimpleNamespace(Client=Mock(return_value=client))}),
            patch.object(reranker, "get_settings", return_value=settings),
        ):
            selected = reranker.build_reranker()
            result = await selected.rerank("Q", chunks(), 1)
        client.rerank.assert_called_once_with(
            model="rerank-multilingual-v3.0", query="Q", documents=["first", "second"], top_n=1
        )
        self.assertEqual(result[0].content, "second")
        self.assertEqual(result[0].rerank_score, 0.97)


def test_optional_sdk_contract_with_mock_http_transport():
    cohere = pytest.importorskip("cohere", reason="requires rerank extra")
    import httpx

    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.path == "/v1/rerank"
        assert json.loads(request.content) == {
            "model": "rerank-multilingual-v3.0", "query": "Q", "documents": ["first", "second"], "top_n": 1
        }
        return httpx.Response(200, json={"id": "offline", "results": [{"index": 1, "relevance_score": 0.97}]})

    client_factory = cohere.Client
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with (
            patch.object(cohere, "Client", side_effect=lambda **kwargs: client_factory(**kwargs, httpx_client=http_client)),
            patch.object(reranker, "get_settings", return_value=SimpleNamespace(rerank_timeout_seconds=1)),
        ):
            selected = reranker.CohereReranker(api_key="offline-test")
            result = asyncio.run(selected.rerank("Q", chunks(), 1))
    assert len(requests) == 1
    assert result[0].content == "second"
    assert result[0].rerank_score == 0.97

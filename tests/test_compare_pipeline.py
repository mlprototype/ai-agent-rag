import asyncio
import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch, AsyncMock

from langchain_core.messages import AIMessage, HumanMessage
from application.agents.graph import graph


@dataclass
class FakeRouteDecision:
    """Fully serialisable mock for RouteDecision – no MagicMock attributes."""
    route: str = "agentic_retrieval"
    reason: str = ""
    query_type: str = "compare"
    routing_layer: str = "heuristic"
    source: str = "heuristic_match"
    heuristic_matched: bool = True
    heuristic_rule: str = "compare"
    confidence: float = 0.95
    llm_router_invoked: bool = False


@dataclass
class FakeAnswerVerdict:
    verdict: str = "PASS"
    reason: str = "ok"
    missing_aspects: list = None
    confidence_override: float = 1.0

    def __post_init__(self):
        if self.missing_aspects is None:
            self.missing_aspects = []


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "LANGSMITH_API_KEY": "test"})
class TestComparePipeline(unittest.IsolatedAsyncioTestCase):

    async def run_graph(self, query: str, thread_id: str = "test_compare"):
        inputs = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": thread_id}}
        final_state = {}
        async for event in graph.astream(inputs, config=config, stream_mode="values"):
            final_state = event
        return final_state

    # ---- Test 1: compare fast-path が正常に動作する ----
    @patch("application.agents.graph.run_compare_retrieval", new_callable=AsyncMock)
    @patch("application.agents.graph.AgentRouter.route", new_callable=AsyncMock)
    @patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock)
    async def test_successful_compare_fast_path(self, mock_ainvoke, mock_router, mock_compare_retrieve):
        from domain.models.retrieval_models import RetrievedChunk
        mock_router.return_value = FakeRouteDecision()
        
        chunk = RetrievedChunk(
            chunk_id="1",
            doc_id="doc1",
            content="text",
            hybrid_score=0.8,
            vector_score=0.8,
            bm25_score=0.8
        )

        mock_compare_retrieve.return_value = {
            "RAG": {"context": "RAGは検索拡張生成の手法です。", "chunks": [chunk], "confidence": 0.8, "top_k": 3, "sources": []},
            "Fine-tuning": {"context": "Fine-tuningはモデルを再学習する手法です。", "chunks": [chunk], "confidence": 0.8, "top_k": 3, "sources": []},
        }

        mock_ainvoke.return_value = AIMessage(content="これが比較結果です")

        with patch("application.agents.graph.AnswerCritic.verify", new_callable=AsyncMock) as mock_critic:
            mock_critic.return_value = FakeAnswerVerdict()

            state = await self.run_graph("RAGとFine-tuningの違い", thread_id="test_compare_1")

            # compare fast-path が使われたこと
            self.assertTrue(state.get("compare_extract_success"))
            self.assertTrue(state.get("compare_path_used"))
            self.assertIsNotNone(state.get("compare_targets"))

            # retrieval が呼ばれたこと
            mock_compare_retrieve.assert_called_once()

            # answer が正しいこと
            self.assertEqual(state.get("answer"), "これが比較結果です")

    # ---- Test 2: 抽出失敗時に agentic_retrieval にフォールバックする ----
    @patch("application.agents.graph.AgentRouter.route", new_callable=AsyncMock)
    @patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock)
    async def test_fallback_on_extraction_failure(self, mock_ainvoke, mock_router):
        mock_router.return_value = FakeRouteDecision()
        mock_ainvoke.return_value = AIMessage(content="fallback_result")

        with patch("application.agents.graph.RetrievalService.run", new_callable=AsyncMock) as mock_retrieval:
            mock_retrieval.return_value = {"context": "", "sources": [], "confidence": 0.5, "chunks": [], "top_k": 0}

            # 「メリットを教えて」は比較対象ペアを抽出できない → fallback
            state = await self.run_graph("メリットを教えて", thread_id="test_compare_2")

            self.assertFalse(state.get("compare_extract_success"))
            self.assertFalse(state.get("compare_path_used"))
            self.assertEqual(state.get("compare_fallback_reason"), "extraction_failed")


if __name__ == "__main__":
    unittest.main()

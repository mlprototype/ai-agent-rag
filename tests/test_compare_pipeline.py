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

        answer = "共通点: RAGとFine-tuningはLLMに関する手法です。相違点: 検索と再学習。"
        mock_ainvoke.return_value = AIMessage(content=answer)

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
            self.assertEqual(state.get("answer"), answer)
            self.assertEqual(state.get("quality_gate_status"), "pass")
            self.assertTrue(state.get("answer_ok"))
            self.assertEqual(state.get("confidence"), state.get("quality_gate_confidence"))
            self.assertNotEqual(state.get("quality_gate_confidence"), 0.8)
            self.assertEqual(state.get("coverage_score"), 1.0)

    @patch("application.agents.graph.run_compare_retrieval", new_callable=AsyncMock)
    @patch("application.agents.graph.AgentRouter.route", new_callable=AsyncMock)
    async def test_missing_target_coverage_keeps_retrieval_fallback(self, mock_router, mock_compare_retrieve):
        from domain.models.retrieval_models import RetrievedChunk
        mock_router.return_value = FakeRouteDecision()
        mock_compare_retrieve.return_value = {
            "RAG": {"chunks": [RetrievedChunk(doc_id="a", chunk_id="1", content="RAG")], "sources": []},
            "Fine-tuning": {"chunks": [], "sources": []},
        }
        with (
            patch("application.agents.graph.RetrievalService.run", new_callable=AsyncMock) as retrieval,
            patch("application.agents.graph._get_generate_chain"),
            patch("application.agents.graph.CompareQualityGate.evaluate") as gate,
            patch("application.agents.graph.AnswerCritic.verify", new_callable=AsyncMock, return_value=FakeAnswerVerdict()),
        ):
            retrieval.return_value = {"context": "", "sources": [], "confidence": 0.0, "chunks": [], "top_k": 0}
            state = await self.run_graph("RAGとFine-tuningの違い", thread_id="compare_missing_target")
        retrieval.assert_awaited_once()
        gate.assert_not_called()
        self.assertTrue(state["compare_route_fallback_used"])
        self.assertEqual(state["route"], "agentic_retrieval")
        self.assertIsNone(state["quality_gate_status"])
        self.assertIsNone(state["quality_gate_confidence"])

    @patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock)
    async def test_gate_verdicts_and_scores_are_propagated(self, mock_ainvoke):
        import time
        from application.agents.graph import compare_generate_node
        from domain.services.compare_quality_gate import CompareQualityGate

        state = {
            "original_query": "RAGとFine-tuningの違い",
            "route": "agentic_retrieval",
            "compare_targets": {"target_a": "RAG", "target_b": "Fine-tuning"},
            "compare_doc_count_a": 1, "compare_doc_count_b": 1,
            "compare_context_coverage_ok": True, "compare_extract_success": True,
            "sources": [{"doc_id": "a"}, {"doc_id": "b"}],
            "budget_started_at": time.monotonic(), "initial_budget_ms": 30000,
        }
        cases = (
            ("共通点: RAGとFine-tuning。相違点: 検索と再学習。", "pass"),
            ("RAGは検索、Fine-tuningは再学習です。", "warning"),
            ("共通点: RAG。相違点: RAG。", "fail"),
        )
        for answer, expected_status in cases:
            with self.subTest(status=expected_status):
                mock_ainvoke.return_value = AIMessage(content=answer)
                with patch.object(CompareQualityGate, "evaluate", wraps=CompareQualityGate.evaluate) as gate:
                    result = await compare_generate_node(state)
                gate.assert_called_once_with(
                    answer=answer, target_a="RAG", target_b="Fine-tuning", doc_count_a=1,
                    doc_count_b=1, coverage_ok=True, sources_count=2, extract_success=True,
                )
                verdict, confidence, warning, missing = CompareQualityGate.evaluate(answer, "RAG", "Fine-tuning", 1, 1, True, 2, True)
                self.assertEqual(result["quality_gate_status"], expected_status)
                self.assertEqual(result["answer_ok"], verdict == "pass")
                self.assertEqual(result["confidence"], confidence)
                self.assertEqual(result["quality_gate_confidence"], confidence)
                self.assertNotEqual(confidence, 0.8)
                self.assertEqual(result["warning"], warning)
                self.assertEqual(result["missing_aspects"], missing)
                if verdict != "pass":
                    self.assertTrue(result["quality_gate_reasons"])

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

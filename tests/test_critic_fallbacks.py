import unittest
from unittest.mock import patch

from domain.services.answer_critic import AnswerCritic
from domain.services.retrieval_critic import RetrievalCritic


class _RaisingChain:
    def __init__(self, exc: Exception):
        self._exc = exc

    async def ainvoke(self, *_args, **_kwargs):
        raise self._exc


class CriticFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_critic_timeout_fallback_caps_confidence(self):
        with patch.object(AnswerCritic, "_get_chain", return_value=_RaisingChain(TimeoutError())):
            verdict = await AnswerCritic.verify(
                original_query="RAG とは何ですか？",
                answer="RAGは外部知識を使って回答する手法です。",
                sources_summary="[1] doc=AI用語集 snippet=RAG (Retrieval-Augmented Generation) は外部ナレッジベースを利用する手法です。",
                timeout_seconds=0.01,
            )

        self.assertEqual(verdict.verdict, "PASS")
        self.assertEqual(verdict.confidence_override, 0.49)
        self.assertEqual(verdict.reason, "critic_fallback:timeout")

    async def test_retrieval_critic_timeout_fallback_is_conservative_for_compare(self):
        with patch.object(RetrievalCritic, "_get_chain", return_value=_RaisingChain(TimeoutError())):
            verdict = await RetrievalCritic.critique(
                original_query="RAGとFine-tuningの違いを教えてください",
                chunks_summary="[1] doc=AI用語集 snippet=RAG (Retrieval-Augmented Generation) は外部ナレッジベースを利用する手法です。",
                timeout_seconds=0.01,
            )

        self.assertEqual(verdict.verdict, "INSUFFICIENT")
        self.assertLessEqual(verdict.coverage_score, 0.49)
        self.assertIn("Fine-tuning", verdict.missing_aspects)


if __name__ == "__main__":
    unittest.main()

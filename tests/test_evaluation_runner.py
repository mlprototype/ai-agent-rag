import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from evaluation import evaluate
from evaluation.schema import EvalRecord


class LazyJudgeTests(unittest.TestCase):
    def test_blank_answer_does_not_initialize_judge(self):
        with patch.object(evaluate, "_get_evaluator_chain") as factory:
            self.assertEqual(evaluate.assess_answer_similarity("expected", "  "), 0.0)
        factory.assert_not_called()

    def test_judge_is_initialized_once_and_reused(self):
        prompt = MagicMock()
        chain = prompt.__or__.return_value
        with (
            patch.object(evaluate, "_EVALUATOR_CHAIN", None),
            patch.object(evaluate, "EVAL_PROMPT", prompt),
            patch.object(evaluate, "ChatOpenAI") as llm,
        ):
            self.assertIs(evaluate._get_evaluator_chain(), chain)
            self.assertIs(evaluate._get_evaluator_chain(), chain)
            llm.assert_called_once_with(model="gpt-4o", temperature=0)
            prompt.__or__.assert_called_once_with(llm.return_value)

    def test_similarity_uses_lazy_judge(self):
        chain = Mock()
        chain.invoke.return_value = SimpleNamespace(content="0.75")
        with patch.object(evaluate, "_get_evaluator_chain", return_value=chain):
            self.assertEqual(evaluate.assess_answer_similarity("expected", "actual"), 0.75)
        chain.invoke.assert_called_once_with({"expected_answer": "expected", "actual_answer": "actual"})

    def test_judge_initialization_error_keeps_existing_fallback(self):
        with patch.object(evaluate, "_get_evaluator_chain", side_effect=ValueError("mock missing key")):
            self.assertEqual(evaluate.assess_answer_similarity("expected", "actual"), 0.0)


class EvaluationRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def run_dataset_item(self, item):
        async def fake_stream(*args, **kwargs):
            yield {
                "answer": "mock answer",
                "query_type": "definition",
                "route": "agentic_retrieval",
                "confidence": 0.7,
                "answer_ok": True,
            }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset.json").write_text(json.dumps([item]), encoding="utf-8")
            with (
                patch.object(evaluate, "__file__", str(root / "evaluate.py")),
                patch.object(evaluate, "graph", SimpleNamespace(astream=fake_stream)),
                patch.object(evaluate, "assess_answer_similarity", return_value=0.9),
            ):
                await evaluate.run_evaluation()
            files = list((root / "results").glob("eval_results_*.json"))
            self.assertEqual(len(files), 1)
            report = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["total_count"], 1)
            return EvalRecord.model_validate(report["records"][0])

    async def test_missing_expectations_remain_optional_and_report_is_saved(self):
        record = await self.run_dataset_item({"question": "Q", "expected_answer": "A"})
        self.assertIsNone(record.expected_query_type)
        self.assertIsNone(record.expected_route)

    async def test_dataset_expectations_propagate_without_inferring_missing_values(self):
        for expectations in (
            {"expected_query_type": "definition", "expected_route": "agentic_retrieval"},
            {"expected_query_type": "compare"},
            {"expected_route": None, "expected_query_type": None},
        ):
            with self.subTest(expectations=expectations):
                record = await self.run_dataset_item(
                    {"question": "Q", "expected_answer": "A", **expectations}
                )
                self.assertEqual(record.expected_query_type, expectations.get("expected_query_type"))
                self.assertEqual(record.expected_route, expectations.get("expected_route"))

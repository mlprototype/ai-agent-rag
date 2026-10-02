import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import evaluate
from evaluation.schema import EvalRecord


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

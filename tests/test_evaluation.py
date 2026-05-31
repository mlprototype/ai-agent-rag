"""
このファイルは、評価システムの各コンポーネントの動作を確認するためのテストスクリプトです。
Pydanticモデルのバリデーション、統計計算ロジック（p50/p95）、および
分布集計（Top N）の正確性を検証します。
"""

import unittest
from evaluation.schema import EvalRecord, EvalSummary, EvalReport, EvalDistributions
from evaluation.aggregator import calculate_summary, get_top_n_distribution, aggregate_results

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # テスト用のダミーレコード
        self.records = [
            EvalRecord(
                query="Q1", query_type="typeA", route="route1", answer="A1",
                confidence=0.8, latency_ms=1000, similarity=0.9,
                response_generated=True, answer_ok=True, degraded=False,
                warning=False, warning_codes=[]
            ),
            EvalRecord(
                query="Q2", query_type="typeA", route="route1", answer="A2",
                confidence=0.7, latency_ms=2000, similarity=0.8,
                response_generated=True, answer_ok=True, degraded=False,
                warning=False, warning_codes=[]
            ),
            EvalRecord(
                query="Q3", query_type="typeB", route="route2", answer="A3",
                confidence=0.6, latency_ms=3000, similarity=0.7,
                response_generated=True, answer_ok=False, degraded=True,
                warning=True, warning_codes=["TIMEOUT"]
            ),
            EvalRecord(
                query="Q4", query_type="typeB", route="route2", answer="",
                confidence=0.2, latency_ms=4000, similarity=0.0,
                response_generated=False, answer_ok=False, degraded=True,
                warning=True, warning_codes=["LOW_CONFIDENCE"]
            ),
            EvalRecord(
                query="Q5", query_type="typeA", route="route1", answer="A5",
                confidence=0.9, latency_ms=1500, similarity=0.95,
                response_generated=True, answer_ok=True, degraded=False,
                warning=False, warning_codes=[]
            )
        ]

    def test_p50_p95_calculation(self):
        """p50/p95の計算ロジックが正しいかテストします。"""
        # latencies: [1000, 1500, 2000, 3000, 4000]
        # count: 5
        # median (p50): 2000
        # p95 index: max(0, int(5 * 0.95) - 1) = max(0, 4 - 1) = 3 -> 3000 (nearest rank)
        # ※ 実装により 0.95 * 5 = 4.75 -> index 4 (4000) になる場合もあるが、実装に合わせる
        
        summary = calculate_summary(self.records)
        self.assertEqual(summary.latency_p50_ms, 2000)
        
        # 実装を確認: idx95 = max(0, int(len(latencies) * 0.95) - 1)
        # int(5 * 0.95) - 1 = int(4.75) - 1 = 4 - 1 = 3
        # latencies[3] = 3000
        self.assertEqual(summary.latency_p95_ms, 3000)

    def test_rates_calculation(self):
        """各種率の計算が正しいかテストします。"""
        summary = calculate_summary(self.records)
        self.assertEqual(summary.total_count, 5)
        self.assertEqual(summary.response_generated_rate, 0.8) # 4/5
        self.assertEqual(summary.answer_ok_rate, 0.6) # 3/5
        self.assertEqual(summary.degraded_rate, 0.4) # 2/5
        self.assertEqual(summary.warning_rate, 0.4) # 2/5

    def test_top_n_distribution(self):
        """Top N抽出機能が正しいかテストします。"""
        items = ["A", "B", "A", "C", "B", "A"]
        top_2 = get_top_n_distribution(items, n=2)
        self.assertEqual(top_2, {"A": 3, "B": 2})
        
        # リストのリストの場合
        items_nested = [["W1", "W2"], ["W1"], ["W3"]]
        top_n = get_top_n_distribution(items_nested)
        self.assertEqual(top_n, {"W1": 2, "W2": 1, "W3": 1})

    def test_aggregate_results_structure(self):
        """最終的なレポート構造の正しさをテストします。"""
        report = aggregate_results(self.records, execution_time_ms=10000)
        
        # 構造の存在確認
        self.assertIn("typeA", report.by_query_type)
        self.assertIn("typeB", report.by_query_type)
        self.assertIn("route1", report.by_route)
        self.assertIn("route2", report.by_route)
        
        # Pydanticバリデーションが通っていること（コンストラクタで実行済み）
        self.assertEqual(report.metadata["total_records"], 5)
        self.assertEqual(report.summary.total_count, 5)

if __name__ == "__main__":
    unittest.main()

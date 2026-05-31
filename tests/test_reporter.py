"""
このファイルは、レポート生成ロジック（reporter.py）の動作を確認するためのテストスクリプトです。
Baseline と Current の突合、差分計算、サマリー生成、および
要対応ケースの抽出ロジックが正しく機能するかを検証します。
"""

import unittest
from datetime import datetime
from evaluation.schema import EvalReport, EvalSummary, EvalRecord, EvalDistributions
from evaluation.reporter import calculate_diffs, generate_summary_points, extract_actionable_cases, build_view_context

class TestReporter(unittest.TestCase):
    def setUp(self):
        # 共通のメタデータ
        metadata = {"timestamp": datetime.now().isoformat(), "total_records": 2, "execution_time_ms": 1000}
        dist = EvalDistributions()

        # Baseline データ
        self.baseline = EvalReport(
            metadata=metadata,
            summary=EvalSummary(
                total_count=2, response_generated_rate=1.0, answer_ok_rate=1.0,
                warning_rate=0.0, fallback_rate=0.0, degraded_rate=0.0,
                avg_confidence=0.9, avg_similarity=0.9, latency_p50_ms=1000, latency_p95_ms=2000
            ),
            by_query_type={}, by_route={}, distributions=dist,
            records=[
                EvalRecord(
                    query="Q1", query_type="T1", answer="A1", confidence=0.9, latency_ms=1000,
                    similarity=0.9, response_generated=True, answer_ok=True
                ),
                EvalRecord(
                    query="Q2", query_type="T1", answer="A2", confidence=0.8, latency_ms=1500,
                    similarity=0.8, response_generated=True, answer_ok=True
                )
            ]
        )

        # Current データ (Q2が悪化、Q3が追加、Q1が正常)
        self.current = EvalReport(
            metadata=metadata,
            summary=EvalSummary(
                total_count=2, response_generated_rate=1.0, answer_ok_rate=0.5,
                warning_rate=0.5, fallback_rate=0.0, degraded_rate=0.5,
                avg_confidence=0.6, avg_similarity=0.6, latency_p50_ms=2000, latency_p95_ms=4000
            ),
            by_query_type={}, by_route={}, distributions=dist,
            records=[
                EvalRecord(
                    query="Q1", query_type="T1", answer="A1", confidence=0.9, latency_ms=1000,
                    similarity=0.9, response_generated=True, answer_ok=True
                ),
                EvalRecord(
                    query="Q2", query_type="T1", answer="A2_bad", confidence=0.4, latency_ms=3000,
                    similarity=0.4, response_generated=True, answer_ok=False, warning=True
                ),
                EvalRecord(
                    query="Q3", query_type="T1", answer="A3", confidence=0.8, latency_ms=1200,
                    similarity=0.8, response_generated=True, answer_ok=True
                )
            ]
        )

    def test_calculate_diffs(self):
        """指標の差分計算が正しいかテストします。"""
        diffs = calculate_diffs(self.baseline, self.current)
        self.assertAlmostEqual(diffs["answer_ok_rate"], -0.5)
        self.assertAlmostEqual(diffs["latency_p50_ms"], 1000)
        self.assertAlmostEqual(diffs["avg_confidence"], -0.3)

    def test_summary_points_generation(self):
        """サマリーの箇条書き生成ロジックをテストします。"""
        points = generate_summary_points(self.baseline, self.current)
        self.assertTrue(any("低下" in p for p in points))
        self.assertTrue(any("悪化" in p for p in points))
        
        # 変化なしの場合
        points_none = generate_summary_points(self.baseline, self.baseline)
        self.assertEqual(points_none, ["主要指標に大きな変動はありません。"])

    def test_extract_actionable_cases_confidence(self):
        """信頼度の絶対閾値・差分閾値による抽出をテストします。"""
        # Q2 は絶対閾値 0.5 未満 (0.4) なので抽出されるはず
        actionable = extract_actionable_cases(self.baseline, self.current)
        q2_case = next((c for c in actionable if c["query"] == "Q2"), None)
        self.assertIsNotNone(q2_case)
        self.assertIn("低信頼度", q2_case["reason_brief"])
        self.assertIn("低類似度", q2_case["reason_brief"])
        self.assertIn("品質基準不合格", q2_case["reason_brief"])

        # 差分閾値テスト用のダミー
        current_v2 = self.current.model_copy(deep=True)
        # Q1 の信頼度を 0.9 -> 0.65 に下げる (差分 0.25 >= 0.2, 絶対値 0.65 >= 0.5)
        current_v2.records[0].confidence = 0.65
        actionable_v2 = extract_actionable_cases(self.baseline, current_v2)
        q1_case = next((c for c in actionable_v2 if c["query"] == "Q1"), None)
        self.assertIsNotNone(q1_case)
        self.assertIn("信頼度の著しい低下", q1_case["reason_brief"])

    def test_added_removed_separation(self):
        """新規追加と削除の分離がCLIレベルで想定通りか（ロジックの確認）。"""
        current_queries = set(r.query for r in self.current.records)
        baseline_queries = set(r.query for r in self.baseline.records)
        
        added = current_queries - baseline_queries
        removed = baseline_queries - current_queries
        
        self.assertEqual(added, {"Q3"})
        self.assertEqual(removed, set()) # baselineにあるQ1, Q2はcurrentにもある

    def test_build_view_context_no_baseline(self):
        """Baselineなしでのコンテキスト構築をテストします。"""
        ctx = build_view_context(None, self.current, "Title", 5)
        self.assertIsNone(ctx["baseline"])
        self.assertIsNone(ctx["diffs"])
        self.assertEqual(len(ctx["summary_points"]), 2)
        self.assertEqual(ctx["title"], "Title")

    def test_sq_analysis_logic(self):
        """structured_query_tool 向けの専用分析ロジックをテストします。"""
        # current に structured_query_tool のレコードを追加
        sq_record = EvalRecord(
            query="SQ1", query_type="calc", route="structured_query_tool", 
            answer="42", confidence=1.0, latency_ms=500, similarity=1.0,
            response_generated=True, answer_ok=True, reason_code="SUCCESS",
            source_name="SQLite (sales)"
        )
        blocked_record = EvalRecord(
            query="SQ2", query_type="calc", route="structured_query_tool", 
            answer="", confidence=0.0, latency_ms=100, similarity=0.0,
            response_generated=False, answer_ok=False, reason_code="validation_failed",
            source_name="SQLite (sales)"
        )
        self.current.records.extend([sq_record, blocked_record])
        
        ctx = build_view_context(None, self.current, "Title", 5)
        sq = ctx["sq_analysis"]
        self.assertIsNotNone(sq)
        self.assertEqual(sq["total_count"], 2)
        self.assertEqual(sq["fail_safe_rate"], 0.5) # 1/2
        self.assertIn("SQLite (sales)", sq["source_counts"])
        self.assertEqual(sq["reason_codes"]["validation_failed"], 1)

if __name__ == "__main__":
    unittest.main()

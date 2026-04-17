import unittest
from domain.services.heuristic_router import HeuristicRouter

class TestHeuristicRouter(unittest.TestCase):
    def test_empty_query(self):
        decision = HeuristicRouter.route("")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.query_type, "direct")
        self.assertEqual(decision.reason, "Matched heuristic rule: empty_query")

    def test_direct_greetings(self):
        valid = ["こんにちは", "おはよう", "ありがとう", "hello", "hi there"]
        for q in valid:
            decision = HeuristicRouter.route(q)
            self.assertIsNotNone(decision, f"Failed on {q}")
            self.assertEqual(decision.query_type, "direct", f"Failed on {q}")

    def test_direct_false_positives(self):
        invalid = [
            "こんにちは、RAGについて教えて",
            "hello, please explain machine learning",
            "おはようございます、昨日の会議の議事録を探して"
        ]
        for q in invalid:
            decision = HeuristicRouter.route(q)
            self.assertIsNone(decision, f"Should be None: {q}")

    def test_calc_expressions(self):
        valid = ["1+1", " 2 * 3 / 4 ", "1足す1は", "1+1は？"]
        for q in valid:
            decision = HeuristicRouter.route(q)
            self.assertIsNotNone(decision, f"Failed on {q}")
            self.assertEqual(decision.query_type, "calc", f"Failed on {q}")

    def test_calc_false_positives(self):
        invalid = ["1+1の仕組みを教えて", "2023年の売上は"]
        for q in invalid:
            decision = HeuristicRouter.route(q)
            self.assertIsNone(decision, f"Should be None: {q}")

    def test_definition(self):
        valid = ["RAGとは何ですか", "LLMの意味を教えて", "ベクトル検索の定義", "APIって何", "RAGとは"]
        for q in valid:
            decision = HeuristicRouter.route(q)
            self.assertIsNotNone(decision, f"Failed on {q}")
            self.assertEqual(decision.query_type, "definition", f"Failed on {q}")

    def test_definition_false_positives(self):
        invalid = ["RAGのチューニングポイントを教えてください", "LLMを使った最新の事例"]
        for q in invalid:
            decision = HeuristicRouter.route(q)
            self.assertIsNone(decision, f"Should be None: {q}")

    def test_compare(self):
        valid = ["RAGとFine-tuningの違い", "Python vs Go", "AとBのメリットを比較", "メリットとデメリットを教えて"]
        for q in valid:
            decision = HeuristicRouter.route(q)
            self.assertIsNotNone(decision, f"Failed on {q}")
            self.assertEqual(decision.query_type, "compare", f"Failed on {q}")

    def test_compare_false_positives(self):
        invalid = [
            "RAGとFine-tuningを併用できますか",
            "AとBのそれぞれの特徴を長く詳しく述べてください、さらに導入事例も含めてください",
            "RAGと呼ばれる手法について教えて",
            "LLMとの付き合い方を教えてください",
            "教えてください"
        ]
        for q in invalid:
            decision = HeuristicRouter.route(q)
            self.assertIsNone(decision, f"Should be None: {q}")

if __name__ == "__main__":
    unittest.main()

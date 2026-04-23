import pytest
from domain.services.heuristic_router import HeuristicRouter

def test_structured_query_router():
    # 構造化クエリの判定
    query1 = "2025年Q3の売上トップ3製品は？"
    decision1 = HeuristicRouter.route(query1)
    assert decision1 is not None
    assert decision1.route == "structured_query_tool"
    assert decision1.query_type == "structured_query"

    query2 = "売上の平均は？"
    decision2 = HeuristicRouter.route(query2)
    assert decision2 is not None
    assert decision2.route == "structured_query_tool"

    # 書き込み要求の判定
    query_write = "売上データをupdateして"
    decision_write = HeuristicRouter.route(query_write)
    assert decision_write is not None
    assert decision_write.route == "structured_query_tool"
    assert decision_write.query_type == "structured_query"

    # 構造化クエリにしない判定 (False positives)
    query3 = "売上を上げる方法は？"
    decision3 = HeuristicRouter.route(query3)
    assert decision3 is None or decision3.route != "structured_query_tool"

    # 定義クエリの維持
    query4 = "RAGとは何ですか"
    decision4 = HeuristicRouter.route(query4)
    assert decision4 is not None
    assert decision4.route == "agentic_retrieval"
    assert decision4.query_type == "definition"

    # 挨拶クエリの維持
    query5 = "こんにちは"
    decision5 = HeuristicRouter.route(query5)
    assert decision5 is not None
    assert decision5.route == "direct_answer"

    # 計算クエリの維持
    query6 = "1+1は？"
    decision6 = HeuristicRouter.route(query6)
    assert decision6 is not None
    assert decision6.route == "calculator"
    assert decision6.query_type == "calc"

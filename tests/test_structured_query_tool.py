from domain.services.structured_query import (
    parse_structured_query_intent,
    StructuredQueryTool
)

def test_parse_structured_query_intent():
    intent = parse_structured_query_intent("売上トップ3")
    assert intent.target_dataset == "sales"
    assert intent.operation == "top_k"
    assert intent.target_metric == "sales"

    intent2 = parse_structured_query_intent("在庫の平均")
    assert intent2.target_dataset == "inventory"
    assert intent2.operation == "avg"

def test_execute_structured_query_success(sqlite_datasource):
    result = StructuredQueryTool.run("売上の合計は？", datasource=sqlite_datasource)
    assert result.success is True
    assert result.operation == "sum"
    assert result.target_dataset == "sales"
    assert "合計は" in result.summary

    result2 = StructuredQueryTool.run("q1の注文件数は？", datasource=sqlite_datasource)
    assert result2.success is True
    assert result2.operation == "count"
    assert "件" in result2.summary

def test_execute_structured_query_fail_safe(sqlite_datasource):
    result = StructuredQueryTool.run("存在しないデータの平均", datasource=sqlite_datasource)
    assert result.success is False
    assert result.target_dataset == "unknown"
    assert result.error_message == "unknown_dataset"
    assert "対象の指標または条件が特定できませんでした" in result.summary

    result2 = StructuredQueryTool.run("売上の割合は？", datasource=sqlite_datasource)
    assert result2.success is False
    assert result2.error_message == "unknown_operation"
    assert "現在の structured query tool では処理できません" in result2.summary

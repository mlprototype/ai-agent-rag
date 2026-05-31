import pytest
from domain.services.structured_query_sql_builder import build_structured_sql
from domain.services.structured_query_types import StructuredQueryIntent

# 責威: SQL Builder の単体テスト。Intent から正しい SQL とパラメータが生成されるか。

def test_build_sql_sum():
    """SUM 操作の SQL 生成テスト"""
    intent = StructuredQueryIntent(
        operation="sum",
        target_metric="sales",
        filters={"category": "Software"},
        target_dataset="sales"
    )
    sql, params = build_structured_sql(intent)
    assert "SELECT sum(sales)" in sql
    assert "FROM sales" in sql
    assert "WHERE category = ?" in sql
    assert params == ("Software",)

def test_build_sql_avg():
    """AVG 操作の SQL 生成テスト"""
    intent = StructuredQueryIntent(
        operation="avg",
        target_metric="stock",
        filters={"status": "In Stock"},
        target_dataset="inventory"
    )
    sql, params = build_structured_sql(intent)
    assert "SELECT avg(stock)" in sql
    assert "WHERE status = ?" in sql

def test_build_sql_count():
    """COUNT 操作の SQL 生成テスト"""
    intent = StructuredQueryIntent(
        operation="count",
        target_metric=None,
        filters={},
        target_dataset="sales"
    )
    sql, params = build_structured_sql(intent)
    assert "SELECT count(*) as result" in sql
    assert "WHERE" not in sql

def test_build_sql_top_k():
    """top_k 操作の SQL 生成テスト"""
    intent = StructuredQueryIntent(
        operation="top_k",
        target_metric="units_sold",
        filters={},
        target_dataset="sales"
    )
    sql, params = build_structured_sql(intent)
    assert "SELECT * FROM sales" in sql
    assert "ORDER BY units_sold DESC LIMIT 3" in sql

def test_build_sql_max():
    """MAX 操作の SQL 生成テスト"""
    intent = StructuredQueryIntent(
        operation="max",
        target_metric="sales",
        filters={"period": "2025-Q1"},
        target_dataset="sales"
    )
    sql, params = build_structured_sql(intent)
    assert "ORDER BY sales DESC LIMIT 1" in sql
    assert params == ("2025-Q1",)

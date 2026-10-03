import pytest
import sqlite3
from domain.services.structured_query_types import StructuredQueryIntent

# 責務: SQLiteDataSource の単体テストおよびセキュリティテスト
# テスト項目: 基本的なクエリ実行、集計、セキュリティ（書き込み拒否、プレースホルダ）

def test_sqlite_datasource_basic_query(sqlite_datasource):
    """基本的な集計クエリ（SUM）で行データが返るか"""
    ds = sqlite_datasource
    intent = StructuredQueryIntent(
        operation="sum",
        target_metric="sales",
        filters={"period": "2025-Q1"},
        target_dataset="sales"
    )
    rows = ds.execute(intent)
    assert isinstance(rows, list)
    assert rows[0]["result"] > 0

def test_sqlite_datasource_top_k(sqlite_datasource):
    """ランキングクエリ（top_k）で行データが返るか"""
    ds = sqlite_datasource
    intent = StructuredQueryIntent(
        operation="top_k",
        target_metric="sales",
        filters={},
        target_dataset="sales"
    )
    rows = ds.execute(intent)
    assert isinstance(rows, list)
    assert len(rows) <= 3

def test_sqlite_datasource_unknown_table(sqlite_datasource):
    """存在しないテーブルへのアクセスが例外になるか"""
    ds = sqlite_datasource
    intent = StructuredQueryIntent(
        operation="count",
        target_metric=None,
        filters={},
        target_dataset="unknown"
    )
    with pytest.raises(sqlite3.OperationalError):
        ds.execute(intent)

def test_sqlite_datasource_readonly_violation_keyword(sqlite_datasource):
    """DROP などの破壊的キーワードが拒否されるか"""
    ds = sqlite_datasource
    # SELECT で始まらない場合は先にそちらのチェックで落ちる
    with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
        ds.execute_readonly("DROP TABLE sales")

    # SELECT で始まりつつ破壊的キーワードを含む場合（例: サブクエリ風）
    with pytest.raises(ValueError, match="Destructive keyword 'DELETE' is not allowed"):
        ds.execute_readonly("SELECT * FROM (DELETE FROM sales)")

def test_sqlite_datasource_readonly_violation_not_select(sqlite_datasource):
    """SELECT 以外のステートメントが拒否されるか"""
    ds = sqlite_datasource
    with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
        ds.execute_readonly("INSERT INTO sales (product_id) VALUES ('X')")

def test_sqlite_datasource_readonly_violation_multistatement(sqlite_datasource):
    """セミコロンによる多文実行が拒否されるか"""
    ds = sqlite_datasource
    with pytest.raises(ValueError, match="Multiple statements or semicolons are not allowed"):
        ds.execute_readonly("SELECT * FROM sales; SELECT * FROM inventory")

def test_sqlite_datasource_placeholder_security(sqlite_datasource):
    """プレースホルダが機能し、SQLインジェクションが無効化されるか"""
    ds = sqlite_datasource
    # インジェクションを試みるフィルタ値
    intent = StructuredQueryIntent(
        operation="count",
        target_metric=None,
        filters={"period": "2025-Q1' OR '1'='1"},
        target_dataset="sales"
    )
    rows = ds.execute(intent)
    assert isinstance(rows, list)
    # プレースホルダが正しく機能すれば、"2025-Q1' OR '1'='1" というリテラル文字列を探し、結果は0件になるはず
    assert rows[0]["result"] == 0

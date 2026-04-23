import pytest
import os
import sqlite3
from domain.services.sqlite_structured_query import SQLiteDataSource
from domain.services.structured_query_types import StructuredQueryIntent
from infrastructure.sqlite.seed_structured_query_db import seed_db

# 責務: SQLiteDataSource の単体テストおよびセキュリティテスト
# テスト項目: 基本的なクエリ実行、集計、セキュリティ（書き込み拒否、プレースホルダ）

DB_PATH = "data/structured_query.db"

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # テスト開始前にDBを初期化
    seed_db()
    yield

def test_sqlite_datasource_basic_query():
    """基本的な集計クエリ（SUM）で行データが返るか"""
    ds = SQLiteDataSource(DB_PATH)
    intent = StructuredQueryIntent(
        operation="sum",
        target_metric="sales",
        filters={"period": "2025-Q1"},
        target_dataset="sales"
    )
    rows = ds.execute(intent)
    assert isinstance(rows, list)
    assert rows[0]["result"] > 0

def test_sqlite_datasource_top_k():
    """ランキングクエリ（top_k）で行データが返るか"""
    ds = SQLiteDataSource(DB_PATH)
    intent = StructuredQueryIntent(
        operation="top_k",
        target_metric="sales",
        filters={},
        target_dataset="sales"
    )
    rows = ds.execute(intent)
    assert isinstance(rows, list)
    assert len(rows) <= 3

def test_sqlite_datasource_unknown_table():
    """存在しないテーブルへのアクセスが例外になるか"""
    ds = SQLiteDataSource(DB_PATH)
    intent = StructuredQueryIntent(
        operation="count",
        target_metric=None,
        filters={},
        target_dataset="unknown"
    )
    with pytest.raises(sqlite3.OperationalError):
        ds.execute(intent)

def test_sqlite_datasource_readonly_violation_keyword():
    """DROP などの破壊的キーワードが拒否されるか"""
    ds = SQLiteDataSource(DB_PATH)
    # SELECT で始まらない場合は先にそちらのチェックで落ちる
    with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
        ds.execute_readonly("DROP TABLE sales")

    # SELECT で始まりつつ破壊的キーワードを含む場合（例: サブクエリ風）
    with pytest.raises(ValueError, match="Destructive keyword 'DELETE' is not allowed"):
        ds.execute_readonly("SELECT * FROM (DELETE FROM sales)")

def test_sqlite_datasource_readonly_violation_not_select():
    """SELECT 以外のステートメントが拒否されるか"""
    ds = SQLiteDataSource(DB_PATH)
    with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
        ds.execute_readonly("INSERT INTO sales (product_id) VALUES ('X')")

def test_sqlite_datasource_readonly_violation_multistatement():
    """セミコロンによる多文実行が拒否されるか"""
    ds = SQLiteDataSource(DB_PATH)
    with pytest.raises(ValueError, match="Multiple statements or semicolons are not allowed"):
        ds.execute_readonly("SELECT * FROM sales; SELECT * FROM inventory")

def test_sqlite_datasource_placeholder_security():
    """プレースホルダが機能し、SQLインジェクションが無効化されるか"""
    ds = SQLiteDataSource(DB_PATH)
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

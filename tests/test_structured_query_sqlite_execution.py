import pytest
from domain.services.structured_query import StructuredQueryTool
from infrastructure.sqlite.seed_structured_query_db import seed_db

# 責務: 構造化クエリツール全体の E2E テスト。自然文から SQLite 実行・回答生成までの流れを検証する。

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # テスト開始前にDBを初期化
    seed_db()
    yield

def test_tool_execution_sum_sales():
    """正常系: 売上の合計を求める"""
    query = "2025年Q1の売上の合計は？"
    result = StructuredQueryTool.run(query)
    assert result.success is True
    assert "合計は" in result.summary
    assert "SQLite (sales)" in result.source_name
    assert result.error_message is None

def test_tool_execution_top_3_inventory():
    """正常系: 在庫のトップ3を求める"""
    query = "在庫のトップ3は？"
    result = StructuredQueryTool.run(query)
    assert result.success is True
    assert "トップ3" in result.summary
    assert len(result.rows) <= 3
    assert "SQLite (inventory)" in result.source_name

def test_tool_execution_validation_fail_write():
    """異常系: 書き込み操作（削除）がバリデーションで弾かれるか"""
    query = "売上データを全て削除して"
    result = StructuredQueryTool.run(query)
    assert result.success is False
    assert result.error_message == "write_operation_blocked"
    assert "禁止されています" in result.summary

def test_tool_execution_parse_fail_unknown():
    """異常系: 意図が特定できない場合にフェイルセーフが機能するか"""
    query = "関係ない質問"
    result = StructuredQueryTool.run(query)
    assert result.success is False
    assert result.error_message == "unknown_dataset"
    assert "特定できませんでした" in result.summary

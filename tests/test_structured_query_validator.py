import pytest
from domain.services.structured_query_types import StructuredQueryIntent
from domain.services.structured_query_validator import validate_structured_query_intent

def test_valid_query():
    intent = StructuredQueryIntent(
        operation="top_k",
        target_metric="sales",
        filters={"period": "2025-Q3"},
        target_dataset="sales"
    )
    result = validate_structured_query_intent(intent, "2025年Q3の売上トップ3製品は？")
    assert result.is_valid is True
    assert result.error_code is None

def test_write_operation_blocked():
    intent = StructuredQueryIntent(
        operation="count",
        target_metric="sales",
        filters={},
        target_dataset="sales"
    )
    result = validate_structured_query_intent(intent, "売上データをupdateして")
    assert result.is_valid is False
    assert result.error_code == "write_operation_blocked"

def test_join_like_query_blocked():
    intent = StructuredQueryIntent(
        operation="sum",
        target_metric="sales",
        filters={},
        target_dataset="sales"
    )
    result = validate_structured_query_intent(intent, "売上と在庫を比較して")
    assert result.is_valid is False
    assert result.error_code == "join_like_query_blocked"

    result2 = validate_structured_query_intent(intent, "売上と販売と在庫のデータ")
    assert result2.is_valid is False
    assert result2.error_code == "join_like_query_blocked"

def test_unknown_dataset():
    intent = StructuredQueryIntent(
        operation="sum",
        target_metric="sales",
        filters={},
        target_dataset="unknown"
    )
    result = validate_structured_query_intent(intent, "利益の合計は？")
    assert result.is_valid is False
    assert result.error_code == "unknown_dataset"

def test_unknown_operation():
    intent = StructuredQueryIntent(
        operation="unknown",
        target_metric="sales",
        filters={},
        target_dataset="sales"
    )
    result = validate_structured_query_intent(intent, "売上の割合は？")
    assert result.is_valid is False
    assert result.error_code == "unknown_operation"

def test_ambiguous_query():
    intent = StructuredQueryIntent(
        operation="sum",
        target_metric=None,
        filters={},
        target_dataset="sales"
    )
    result = validate_structured_query_intent(intent, "売上の何かの合計は？")
    assert result.is_valid is False
    assert result.error_code == "ambiguous_query"

def test_unknown_field():
    intent = StructuredQueryIntent(
        operation="sum",
        target_metric="profit", # Not in ALLOWED_FIELDS["sales"]
        filters={},
        target_dataset="sales"
    )
    result = validate_structured_query_intent(intent, "売上の利益の合計は？")
    assert result.is_valid is False
    assert result.error_code == "unknown_field"

    intent2 = StructuredQueryIntent(
        operation="sum",
        target_metric="sales",
        filters={"unknown_filter": "yes"},
        target_dataset="sales"
    )
    result2 = validate_structured_query_intent(intent2, "売上の合計は？")
    assert result2.is_valid is False
    assert result2.error_code == "unknown_field"

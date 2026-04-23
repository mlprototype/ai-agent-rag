# 責務: 構造化データに対する問い合わせの意図抽出と実行フロー（Parse -> Validate -> Execute -> Format）の制御
# 主な入出力: 自然文クエリを受け取り、整形された StructuredQueryResult を返す
# 設計上の注意点: 各コンポーネント（Parser, Validator, DataSource, Formatter）をオーケストレートし、一貫したフェイルセーフ応答を保証する

import re
from typing import Any, Literal

from domain.services.structured_query_datasets import MOCK_INVENTORY_DATA, MOCK_SALES_DATA
from domain.services.structured_query_types import (
    StructuredQueryIntent, 
    StructuredQueryResult,
    StructuredDataSource
)
from domain.services.structured_query_validator import validate_structured_query_intent
from domain.services.sqlite_structured_query import SQLiteDataSource
from domain.services.structured_query_formatter import format_structured_result

def fail_safe_result(
    reason_code: str, 
    message: str, 
    intent: StructuredQueryIntent | None = None
) -> StructuredQueryResult:
    """
    エラー時の一貫した StructuredQueryResult を生成します。
    """
    return StructuredQueryResult(
        success=False,
        operation=intent.operation if intent else "unknown",
        target_metric=intent.target_metric if intent else None,
        filters=intent.filters if intent else {},
        rows=[],
        summary=message,
        source_name="Unknown",
        error_message=reason_code
    )


def parse_structured_query_intent(query: str) -> StructuredQueryIntent:
    """
    自然文から構造化クエリの意図を抽出します。
    """
    query_lower = query.lower()
    
    # Dataset
    target_dataset = "unknown"
    if "売上" in query_lower or "販売" in query_lower or "注文" in query_lower:
        target_dataset = "sales"
    elif "在庫" in query_lower:
        target_dataset = "inventory"
        
    # Operation
    operation = "unknown"
    if "トップ" in query_lower or "上位" in query_lower or "ランキング" in query_lower:
        operation = "top_k"
    elif "平均" in query_lower:
        operation = "avg"
    elif "合計" in query_lower or "総" in query_lower:
        operation = "sum"
    elif "最大" in query_lower:
        operation = "max"
    elif "最小" in query_lower:
        operation = "min"
    elif "件数" in query_lower or "何件" in query_lower:
        operation = "count"
        
    # Metric
    target_metric = None
    if target_dataset == "sales":
        if "件数" in query_lower:
            target_metric = "units_sold"
        else:
            target_metric = "sales"
    elif target_dataset == "inventory":
        target_metric = "stock"
        
    # Filters
    filters = {}
    if re.search(r'q1|第1四半期', query_lower):
        filters["period"] = "2025-Q1"
    elif re.search(r'q2|第2四半期', query_lower):
        filters["period"] = "2025-Q2"
    elif re.search(r'q3|第3四半期', query_lower):
        filters["period"] = "2025-Q3"
        
    return StructuredQueryIntent(
        operation=operation,  # type: ignore
        target_metric=target_metric,
        filters=filters,
        target_dataset=target_dataset  # type: ignore
    )


class LocalDictDataSource(StructuredDataSource):
    """
    インメモリの辞書リストを対象としたデータソース（テスト・互換性用）
    """
    def __init__(self, dataset_map: dict[str, list[dict[str, Any]]] | None = None):
        if dataset_map is None:
            self.dataset_map = {
                "sales": MOCK_SALES_DATA,
                "inventory": MOCK_INVENTORY_DATA
            }
        else:
            self.dataset_map = dataset_map

    @property
    def name(self) -> str:
        return "MockDB"

    def execute(self, intent: StructuredQueryIntent) -> list[dict[str, Any]]:
        data = self.dataset_map.get(intent.target_dataset, [])
        
        # フィルタ適用
        filtered_data = data
        for k, v in intent.filters.items():
            filtered_data = [row for row in filtered_data if row.get(k) == v]
            
        if not filtered_data:
            return []
            
        # 計算（SQL 実行結果のセットに寄せる）
        metric = intent.target_metric
        
        if intent.operation == "count":
            return [{"result": len(filtered_data)}]
            
        elif intent.operation == "sum":
            val = sum(row[metric] for row in filtered_data)  # type: ignore
            return [{"result": val}]
            
        elif intent.operation == "avg":
            val = sum(row[metric] for row in filtered_data) / len(filtered_data)  # type: ignore
            return [{"result": val}]
            
        elif intent.operation == "max":
            max_row = max(filtered_data, key=lambda x: x[metric])  # type: ignore
            return [max_row]
            
        elif intent.operation == "min":
            min_row = min(filtered_data, key=lambda x: x[metric])  # type: ignore
            return [min_row]
            
        elif intent.operation == "top_k":
            sorted_data = sorted(filtered_data, key=lambda x: x[metric], reverse=True)  # type: ignore
            return sorted_data[:3]
            
        return filtered_data


class StructuredQueryTool:
    @classmethod
    def run(cls, query: str, datasource: StructuredDataSource | None = None) -> StructuredQueryResult:
        """
        構造化クエリの実行パイプライン（Parse -> Validate -> Execute -> Format）を制御します。
        """
        # 1. Parse
        intent = parse_structured_query_intent(query)
        
        # 2. Validate
        validation_result = validate_structured_query_intent(intent, query)
        if not validation_result.is_valid:
            return fail_safe_result(
                reason_code=validation_result.error_code or "validation_failed",
                message=validation_result.error_message or "この問い合わせは現在の structured query tool では処理できません。",
                intent=intent
            )
            
        # 3. Execute
        ds = datasource or SQLiteDataSource()
        try:
            rows = ds.execute(intent)
            
            # 4. Format
            return format_structured_result(
                intent=intent, 
                rows=rows, 
                source_name=f"{ds.name} ({intent.target_dataset})"
            )
            
        except Exception as e:
            return fail_safe_result(
                reason_code=type(e).__name__,
                message=f"構造化クエリの実行中にエラーが発生しました: {str(e)}",
                intent=intent
            )

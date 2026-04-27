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
    ヒューリスティックな正規表現とキーワードマッチングを使用します。
    """
    query_lower = query.lower()
    
    # Dataset 判定 (日本語キーワード + 英名)
    target_dataset = "unknown"
    if any(k in query_lower for k in ["売上", "販売", "注文", "sales"]):
        target_dataset = "sales"
    elif any(k in query_lower for k in ["在庫", "inventory"]):
        target_dataset = "inventory"
        
    # Operation 判定
    operation = "unknown"
    if any(k in query_lower for k in ["トップ", "上位", "ランキング", "top"]):
        operation = "top_k"
    elif "平均" in query_lower or "avg" in query_lower:
        operation = "avg"
    elif any(k in query_lower for k in ["合計", "総", "sum"]):
        operation = "sum"
    elif "最大" in query_lower or "max" in query_lower:
        operation = "max"
    elif "最小" in query_lower or "min" in query_lower:
        operation = "min"
    elif any(k in query_lower for k in ["件数", "何件", "カウント", "count"]):
        operation = "count"
    elif any(k in query_lower for k in ["一覧", "リスト", "list"]):
        operation = "list"
        
    # Metric (指標) 判定 - 明示的な指定を優先
    target_metric = None
    
    # フィールド名の直接言及をチェック
    if "product_id" in query_lower:
        target_metric = "product_id"
    elif "units_sold" in query_lower:
        target_metric = "units_sold"
    elif "sales" in query_lower and target_dataset == "sales":
        # "sales" がデータセット名と指標名の両方に使われるため文脈で判断
        if any(k in query_lower for k in ["合計", "平均", "最大", "最小"]):
            target_metric = "sales"
    elif "stock" in query_lower:
        target_metric = "stock"
    
    # デフォルト指標の設定（明示的な指定がない場合）
    if not target_metric:
        if target_dataset == "sales":
            if operation == "count":
                target_metric = "units_sold" # 件数なら販売数(レコード数)
            elif operation == "list":
                target_metric = "product_id" # 一覧なら商品ID
            else:
                target_metric = "sales"      # それ以外は金額
        elif target_dataset == "inventory":
            target_metric = "stock"
        
    # Filters 判定
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
            
        elif intent.operation == "list":
            return filtered_data
            
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

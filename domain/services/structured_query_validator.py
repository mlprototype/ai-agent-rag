# 責務: StructuredQueryIntent が許可された安全なクエリであるかを検証する（Allowlist 検査）
# 主な入出力: Intent と生のクエリを受け取り、ValidationResult を返す
# 設計上の注意点: 未知の操作や書き込み要求、複雑すぎる JOIN 要求などをアプリケーションレベルで安全にブロックする

from domain.services.structured_query_types import StructuredQueryIntent, ValidationResult

ALLOWED_OPERATIONS = {"count", "sum", "avg", "top_k", "max", "min", "list"}
ALLOWED_DATASETS = {"sales", "inventory"}
ALLOWED_FIELDS = {
    "sales": {"sales", "units_sold", "product_id", "period", "category", "product_name"},
    "inventory": {"stock", "product_id", "status", "product_name"},
    "unknown": set()
}

WRITE_KEYWORDS = ["update", "delete", "insert", "drop", "alter", "create", "削除", "追加", "更新", "変更"]
JOIN_KEYWORDS = ["組み合わせて", "組み合わせ", "比較", "相関"]

def validate_structured_query_intent(intent: StructuredQueryIntent, raw_query: str) -> ValidationResult:
    query_lower = raw_query.lower()
    
    # 1. 書き込み操作 (Write operation)
    if any(k in query_lower for k in WRITE_KEYWORDS):
        return ValidationResult(
            is_valid=False,
            error_code="write_operation_blocked",
            error_message="この問い合わせは現在の structured query tool では処理できません（書き込み操作は禁止されています）。"
        )
        
    # 2. JOIN相当の複雑要求 (Join-like query)
    dataset_mentions = 0
    if "売上" in query_lower or "販売" in query_lower or "注文" in query_lower:
        dataset_mentions += 1
    if "在庫" in query_lower:
        dataset_mentions += 1
        
    if dataset_mentions >= 2 or any(k in query_lower for k in JOIN_KEYWORDS):
        return ValidationResult(
            is_valid=False,
            error_code="join_like_query_blocked",
            error_message="この問い合わせは現在の structured query tool では処理できません（複数データの組み合わせや比較は未対応です）。"
        )
        
    # 3. 未知の dataset または 曖昧なデータセット
    if intent.target_dataset == "unknown" or intent.target_dataset not in ALLOWED_DATASETS:
        return ValidationResult(
            is_valid=False,
            error_code="unknown_dataset",
            error_message="対象の指標または条件が特定できませんでした。"
        )
        
    # 4. 未知の operation または 曖昧な操作
    if intent.operation == "unknown" or intent.operation not in ALLOWED_OPERATIONS:
        return ValidationResult(
            is_valid=False,
            error_code="unknown_operation",
            error_message="この問い合わせは現在の structured query tool では処理できません（対応していない集計操作です）。"
        )
        
    # 5. 曖昧なクエリ（必須項目が欠落）
    if not intent.target_metric and intent.operation != "count":
        return ValidationResult(
            is_valid=False,
            error_code="ambiguous_query",
            error_message="対象の指標または条件が特定できませんでした。"
        )
        
    # 6. 未知の field
    allowed_fields_for_dataset = ALLOWED_FIELDS.get(intent.target_dataset, set())
    if intent.target_metric and intent.target_metric not in allowed_fields_for_dataset:
        return ValidationResult(
            is_valid=False,
            error_code="unknown_field",
            error_message="この問い合わせは現在の structured query tool では処理できません（未定義のフィールドです）。"
        )
        
    for filter_key in intent.filters.keys():
        if filter_key not in allowed_fields_for_dataset:
            return ValidationResult(
                is_valid=False,
                error_code="unknown_field",
                error_message="この問い合わせは現在の structured query tool では処理できません（未定義のフィルタフィールドです）。"
            )
            
    return ValidationResult(is_valid=True)

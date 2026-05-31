from typing import Any
from domain.services.structured_query_types import StructuredQueryIntent

# 責務: バリデーション済みの StructuredQueryIntent から安全な SQL テンプレートを生成する
# 主な入出力: Intent オブジェクト -> (SQL文字列, パラメータタプル)
# 設計上の注意点: 
# 1. SQL インジェクションを防ぐため、フィルタ値には必ずプレースホルダ（?）を使用する。
# 2. テーブル名やカラム名は事前に Validator で許可されたもののみが渡されることを前提とする。
# 3. 操作（operation）ごとに最適な SQL テンプレートを選択して構築する。

def build_structured_sql(intent: StructuredQueryIntent) -> tuple[str, tuple[Any, ...]]:
    """
    StructuredQueryIntent から SQLite 用の SELECT クエリを構築します。
    """
    table_name = intent.target_dataset
    operation = intent.operation
    metric = intent.target_metric
    
    # 1. 基本となる SELECT 文の構築
    if operation == "count":
        sql = f"SELECT count(*) as result FROM {table_name}"
    elif operation == "list":
        sql = f"SELECT {metric} FROM {table_name}"
    elif operation == "sum":
        sql = f"SELECT sum({metric}) as result FROM {table_name}"
    elif operation == "avg":
        sql = f"SELECT avg({metric}) as result FROM {table_name}"
    elif operation in ["max", "min", "top_k"]:
        # これらは後続の ORDER BY / LIMIT で制御するため、一旦 SELECT *
        sql = f"SELECT * FROM {table_name}"
    else:
        sql = f"SELECT * FROM {table_name}"
        
    # 2. WHERE 句（フィルタ）の構築
    params = []
    where_clauses = []
    for k, v in intent.filters.items():
        where_clauses.append(f"{k} = ?")
        params.append(v)
        
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
        
    # 3. 並び替えと制限（max/min/top_k）の付与
    if operation == "max":
        sql += f" ORDER BY {metric} DESC LIMIT 1"
    elif operation == "min":
        sql += f" ORDER BY {metric} ASC LIMIT 1"
    elif operation == "top_k":
        sql += f" ORDER BY {metric} DESC LIMIT 3"
        
    return sql, tuple(params)

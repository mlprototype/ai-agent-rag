# 責務: SQLite を用いた構造化クエリの物理的な実行
# 主な入出力: Intent を受け取り、データベース上で安全にクエリを実行して行データ（rows）を返す
# 設計上の注意点: 読み取り専用（execute_readonly）を強制し、破壊的な操作や多文実行を厳格にブロックする

import sqlite3
import re
from typing import Any
from domain.services.structured_query_types import (
    StructuredQueryIntent, 
    StructuredDataSource
)
from domain.services.structured_query_sql_builder import build_structured_sql

class SQLiteDataSource(StructuredDataSource):
    def __init__(self, db_path: str = "data/structured_query.db"):
        self.db_path = db_path
        
    @property
    def name(self) -> str:
        return "SQLite"

    def execute(self, intent: StructuredQueryIntent) -> list[dict[str, Any]]:
        """
        Intent から SQL を構築し、読み取り専用で実行して結果の行データを返します。
        """
        sql, params = build_structured_sql(intent)
        return self.execute_readonly(sql, params)

    def execute_readonly(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """
        読み取り専用で SQL を実行します。
        SELECT で始まらないもの、多文実行(;)、破壊的キーワードを含むものは例外をスローします。
        """
        sql_upper = sql.strip().upper()
        
        # 1. SELECT で始まらないクエリは拒否
        if not sql_upper.startswith("SELECT"):
            raise ValueError("Only SELECT statements are allowed.")
            
        # 2. 多文実行 (;) は拒否
        if ";" in sql:
            raise ValueError("Multiple statements or semicolons are not allowed.")
            
        # 3. 破壊的キーワードのチェック (Read-only を守る最後の防波堤)
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "REPLACE"]
        for word in forbidden:
            if re.search(rf"\b{word}\b", sql_upper):
                raise ValueError(f"Destructive keyword '{word}' is not allowed.")
                
        # 実行
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = [dict(row) for row in cursor.fetchall()]
            return rows
        finally:
            conn.close()

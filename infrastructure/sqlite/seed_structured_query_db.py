import sqlite3
import os
from domain.services.structured_query_datasets import MOCK_SALES_DATA, MOCK_INVENTORY_DATA

# 責務: SQLite データベースの初期化とシードデータの投入（開発・テスト用）
# 主な入出力: 外部のモックデータセットを SQLite ファイルへ書き込む
# 設計上の注意点: DROP TABLE IF EXISTS により冪等性を確保し、常にクリーンな状態で再生成する

DB_PATH = "data/structured_query.db"

def seed_db():
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 再実行ポリシー: 既存テーブルを削除して再作成
    cursor.execute("DROP TABLE IF EXISTS sales")
    cursor.execute("DROP TABLE IF EXISTS inventory")
    
    # Create tables
    cursor.execute("""
        CREATE TABLE sales (
            product_id TEXT,
            product_name TEXT,
            category TEXT,
            sales INTEGER,
            units_sold INTEGER,
            period TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE inventory (
            product_id TEXT,
            product_name TEXT,
            stock INTEGER,
            status TEXT
        )
    """)
    
    # Insert data
    for row in MOCK_SALES_DATA:
        cursor.execute(
            "INSERT INTO sales (product_id, product_name, category, sales, units_sold, period) VALUES (?, ?, ?, ?, ?, ?)",
            (row["product_id"], row["product_name"], row["category"], row["sales"], row["units_sold"], row["period"])
        )
        
    for row in MOCK_INVENTORY_DATA:
        cursor.execute(
            "INSERT INTO inventory (product_id, product_name, stock, status) VALUES (?, ?, ?, ?)",
            (row["product_id"], row["product_name"], row["stock"], row["status"])
        )
        
    conn.commit()
    conn.close()
    print(f"Database seeded successfully at {DB_PATH}")

if __name__ == "__main__":
    seed_db()

from typing import Any

# 責務: structured query_tool で使用するローカルのモックデータセットを定義する。
# 将来的に SQLite や本番 DB に差し替えるための布石。

MOCK_SALES_DATA: list[dict[str, Any]] = [
    {"product_id": "P001", "product_name": "AI Platform", "category": "Software", "sales": 1500000, "units_sold": 50, "period": "2025-Q1"},
    {"product_id": "P002", "product_name": "Data Analytics API", "category": "Software", "sales": 800000, "units_sold": 200, "period": "2025-Q1"},
    {"product_id": "P003", "product_name": "Cloud Storage", "category": "Infrastructure", "sales": 2500000, "units_sold": 1000, "period": "2025-Q1"},
    {"product_id": "P001", "product_name": "AI Platform", "category": "Software", "sales": 2100000, "units_sold": 70, "period": "2025-Q2"},
    {"product_id": "P002", "product_name": "Data Analytics API", "category": "Software", "sales": 950000, "units_sold": 240, "period": "2025-Q2"},
    {"product_id": "P003", "product_name": "Cloud Storage", "category": "Infrastructure", "sales": 2700000, "units_sold": 1050, "period": "2025-Q2"},
    {"product_id": "P001", "product_name": "AI Platform", "category": "Software", "sales": 3000000, "units_sold": 100, "period": "2025-Q3"},
    {"product_id": "P002", "product_name": "Data Analytics API", "category": "Software", "sales": 1200000, "units_sold": 300, "period": "2025-Q3"},
    {"product_id": "P003", "product_name": "Cloud Storage", "category": "Infrastructure", "sales": 2600000, "units_sold": 1020, "period": "2025-Q3"},
]

MOCK_INVENTORY_DATA: list[dict[str, Any]] = [
    {"product_id": "P001", "product_name": "AI Platform", "stock": 9999, "status": "In Stock"},
    {"product_id": "P002", "product_name": "Data Analytics API", "stock": 9999, "status": "In Stock"},
    {"product_id": "P003", "product_name": "Cloud Storage", "stock": 50000, "status": "In Stock"},
]

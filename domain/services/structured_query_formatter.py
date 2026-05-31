from typing import Any
from domain.services.structured_query_types import StructuredQueryIntent, StructuredQueryResult

# 責務: 構造化クエリの実行結果（行データ）を人間が読みやすい形式に整形する
# 主な入出力: (Intent, 行データリスト, データソース名) -> StructuredQueryResult
# 設計上の注意点: 
# 1. 操作（operation）に応じた適切な自然言語のサマリーを生成する。
# 2. 数値のカンマ区切りや小数点以下の桁数など、表示形式を整える。

def format_structured_result(
    intent: StructuredQueryIntent, 
    rows: list[dict[str, Any]], 
    source_name: str
) -> StructuredQueryResult:
    """
    SQL 実行結果の行データからユーザー向けのサマリーを生成し、StructuredQueryResult に整形します。
    """
    if not rows:
        summary = "指定された条件に一致するデータが見つかりませんでした。"
    else:
        if intent.operation == "count":
            val = rows[0].get("result", 0)
            summary = f"該当するデータは {val} 件です。"
        elif intent.operation == "list":
            vals = [str(r.get(intent.target_metric)) for r in rows if intent.target_metric in r]
            summary = f"該当する {intent.target_metric} は以下の通りです: {', '.join(vals)}。"
        elif intent.operation == "sum":
            val = rows[0].get("result", 0) or 0
            summary = f"合計は {val:,} です。"
        elif intent.operation == "avg":
            val = rows[0].get("result", 0) or 0
            summary = f"平均は {val:,.2f} です。"
        elif intent.operation in ["max", "min"]:
            row = rows[0]
            val = row.get(intent.target_metric) if intent.target_metric else "N/A"
            name = row.get("product_name", "Unknown")
            label = "最大" if intent.operation == "max" else "最小"
            # 数値の場合はカンマ区切りにする
            val_str = f"{val:,}" if isinstance(val, (int, float)) else str(val)
            summary = f"{label}は {name} の {val_str} です。"
        elif intent.operation == "top_k":
            names = [
                f"{r.get('product_name', 'Unknown')} ({r.get(intent.target_metric):,})" 
                for r in rows if intent.target_metric in r and isinstance(r.get(intent.target_metric), (int, float))
            ]
            if not names:
                # 数値でない場合のフォールバック
                names = [f"{r.get('product_name', 'Unknown')}" for r in rows]
            summary = f"トップ3は以下の通りです: {', '.join(names)}。"
        else:
            summary = f"実行に成功しました（{len(rows)}件）。"

    return StructuredQueryResult(
        success=True,
        operation=intent.operation,
        target_metric=intent.target_metric,
        filters=intent.filters,
        rows=rows,
        summary=summary,
        source_name=source_name
    )

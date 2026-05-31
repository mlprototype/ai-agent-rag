"""
このファイルは、評価レコードの集計ロジックを担当します。
各クエリの評価結果（EvalRecord）のリストを受け取り、統計値（p50/p95）や
カテゴリ別の指標を計算して、最終的なレポート（EvalReport）を生成します。
"""

import statistics
from typing import List, Dict, Any
from datetime import datetime
from evaluation.schema import EvalRecord, EvalSummary, EvalDistributions, EvalReport

def calculate_summary(records: List[EvalRecord]) -> EvalSummary:
    """与えられたレコードリストから集計サマリーを計算します。"""
    count = len(records)
    if count == 0:
        return EvalSummary(
            total_count=0, response_generated_rate=0, answer_ok_rate=0,
            warning_rate=0, fallback_rate=0, degraded_rate=0,
            avg_confidence=0, avg_similarity=0, latency_p50_ms=0, latency_p95_ms=0
        )

    # 各種フラグのカウント
    res_gen_count = sum(1 for r in records if r.response_generated)
    ans_ok_count = sum(1 for r in records if r.answer_ok)
    warning_count = sum(1 for r in records if r.warning)
    fallback_count = sum(1 for r in records if r.fallback_level != "NONE")
    degraded_count = sum(1 for r in records if r.degraded)
    
    # 平均値の計算
    avg_conf = sum(r.confidence for r in records) / count
    avg_sim = sum(r.similarity for r in records) / count
    
    # レイテンシの統計（p50, p95）
    latencies = sorted([r.latency_ms for r in records])
    p50 = statistics.median(latencies)
    
    # 95パーセンタイルの計算 (nearest rank method)
    # 1件のみの場合はその値、それ以外はインデックスを計算
    idx95 = max(0, int(len(latencies) * 0.95) - 1)
    p95 = latencies[idx95]

    return EvalSummary(
        total_count=count,
        response_generated_rate=round(res_gen_count / count, 3),
        answer_ok_rate=round(ans_ok_count / count, 3),
        warning_rate=round(warning_count / count, 3),
        fallback_rate=round(fallback_count / count, 3),
        degraded_rate=round(degraded_count / count, 3),
        avg_confidence=round(avg_conf, 3),
        avg_similarity=round(avg_sim, 3),
        latency_p50_ms=p50,
        latency_p95_ms=p95
    )

def get_top_n_distribution(items: List[Any], n: int = 10) -> Dict[str, int]:
    """リスト内の要素の出現頻度をカウントし、上位N個を返します。"""
    counts = {}
    for item in items:
        if isinstance(item, list):
            for sub_item in item:
                counts[str(sub_item)] = counts.get(str(sub_item), 0) + 1
        else:
            counts[str(item)] = counts.get(str(item), 0) + 1
    
    # 頻度順にソートして上位N個を抽出
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:n])

def aggregate_results(records: List[EvalRecord], execution_time_ms: int) -> EvalReport:
    """全レコードからレポートを生成します。"""
    # 1. 全体サマリー
    summary = calculate_summary(records)
    
    # 2. クエリタイプ別サマリー
    by_query_type = {}
    q_types = set(r.query_type for r in records)
    for qt in q_types:
        q_records = [r for r in records if r.query_type == qt]
        by_query_type[qt] = calculate_summary(q_records)
        
    # 3. ルート別サマリー
    by_route = {}
    routes = set(r.route for r in records if r.route)
    for route in routes:
        r_records = [r for r in records if r.route == route]
        by_route[route] = calculate_summary(r_records)
        
    # 4. 分布
    distributions = EvalDistributions(
        warning_codes=get_top_n_distribution([r.warning_codes for r in records]),
        fallback_level=get_top_n_distribution([r.fallback_level for r in records]),
        retrieval_quality_level=get_top_n_distribution([r.retrieval_quality_level for r in records]),
        reason_code=get_top_n_distribution([r.reason_code for r in records])
    )
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_records": len(records),
        "execution_time_ms": execution_time_ms
    }
    
    return EvalReport(
        metadata=metadata,
        summary=summary,
        by_query_type=by_query_type,
        by_route=by_route,
        distributions=distributions,
        records=records
    )

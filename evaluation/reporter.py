"""
このファイルは、評価結果の比較とMarkdownレポートの生成を担当します。
過去の評価結果（Baseline）と最新の結果（Current）を比較し、
主要指標の変動や、改善・悪化が必要なケースを抽出してレポートを出力します。
CLIツールとして動作し、GitHub PR等での利用を想定しています。
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
from evaluation.schema import EvalReport, EvalRecord

def load_report(path: str) -> EvalReport:
    """JSONファイルからEvalReportモデルを読み込みます。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvalReport.model_validate(data)

def calculate_diffs(baseline: EvalReport, current: EvalReport) -> Dict[str, Any]:
    """BaselineとCurrentの集計指標の差分を計算します。"""
    b = baseline.summary
    c = current.summary
    return {
        "total_count": c.total_count - b.total_count,
        "response_generated_rate": c.response_generated_rate - b.response_generated_rate,
        "answer_ok_rate": c.answer_ok_rate - b.answer_ok_rate,
        "warning_rate": c.warning_rate - b.warning_rate,
        "avg_confidence": c.avg_confidence - b.avg_confidence,
        "avg_similarity": c.avg_similarity - b.avg_similarity,
        "latency_p50_ms": c.latency_p50_ms - b.latency_p50_ms,
        "latency_p95_ms": c.latency_p95_ms - b.latency_p95_ms,
    }

def generate_summary_points(baseline: Optional[EvalReport], current: EvalReport) -> List[str]:
    """指標の変動に基づいた要約メッセージを生成します。"""
    points = []
    c = current.summary
    
    if not baseline:
        points.append(f"初回評価結果: {c.total_count} 件のテストを実行しました。")
        points.append(f"品質基準合格率: {c.answer_ok_rate:.1%}")
        return points

    b = baseline.summary
    
    # 合格率の変動
    ok_diff = c.answer_ok_rate - b.answer_ok_rate
    if abs(ok_diff) >= 0.01:
        status = "向上" if ok_diff > 0 else "低下"
        points.append(f"品質基準合格率が {b.answer_ok_rate:.1%} から {c.answer_ok_rate:.1%} へ{status}しました。")
    
    # レイテンシの変動
    lat_diff = c.latency_p50_ms - b.latency_p50_ms
    if abs(lat_diff) >= 50:
        status = "悪化" if lat_diff > 0 else "改善"
        points.append(f"レイテンシ(p50)が {b.latency_p50_ms}ms から {c.latency_p50_ms}ms へ{status}しました。")
        
    # 警告率の変動
    warn_diff = c.warning_rate - b.warning_rate
    if abs(warn_diff) >= 0.05:
        status = "増加" if warn_diff > 0 else "減少"
        points.append(f"警告発生率が {b.warning_rate:.1%} から {c.warning_rate:.1%} へ{status}しました。")

    if not points:
        points.append("主要指標に大きな変動はありません。")
        
    return points

def extract_actionable_cases(baseline: Optional[EvalReport], current: EvalReport) -> List[Dict[str, Any]]:
    """要対応（悪化・低品質）ケースを抽出します。"""
    actionable = []
    
    # クエリをキーにしたマップを作成
    baseline_map = {r.query: r for r in baseline.records} if baseline else {}
    
    for curr in current.records:
        base = baseline_map.get(curr.query)
        reasons = []
        
        # 抽出条件の判定
        if curr.similarity < 0.5:
            reasons.append("低類似度")
        if not curr.answer_ok:
            reasons.append("品質基準不合格")
        if curr.strict_insufficient_response:
            reasons.append("回答不能判定")
        if curr.warning:
            reasons.append("警告発生")
        if curr.fallback_level != "NONE":
            reasons.append("フォールバック発生")
        if curr.reason_code != "SUCCESS" and curr.reason_code != "NONE":
            reasons.append(f"異常終了({curr.reason_code})")
            
        # 信頼度の低下
        if curr.confidence < 0.5:
            reasons.append("低信頼度")
        elif base and (base.confidence - curr.confidence) >= 0.2:
            reasons.append("信頼度の著しい低下")

        if reasons:
            diff_status = "-"
            if base:
                sim_diff = curr.similarity - base.similarity
                if abs(sim_diff) >= 0.1:
                    diff_status = f"類似度 {'+%.2f' % sim_diff if sim_diff > 0 else '%.2f' % sim_diff}"
            else:
                diff_status = "新規"

            actionable.append({
                "query": curr.query,
                "reason_brief": ", ".join(reasons),
                "current": curr,
                "baseline": base,
                "diff_status": diff_status
            })
            
    # 類似度が低い順、または悪化した順にソート（ここでは類似度昇順）
    return sorted(actionable, key=lambda x: x["current"].similarity)

def build_view_context(baseline: Optional[EvalReport], current: EvalReport, title: str, top_n: int) -> Dict[str, Any]:
    """テンプレートレンダリング用の共通コンテキストを構築します。"""
    
    # 基本情報の突合
    current_queries = set(r.query for r in current.records)
    baseline_queries = set(r.query for r in baseline.records) if baseline else set()
    
    added_cases = current_queries - baseline_queries
    removed_cases = baseline_queries - current_queries
    
    # 差分計算
    diffs = calculate_diffs(baseline, current) if baseline else None
    
    # サマリー箇条書き
    summary_points = generate_summary_points(baseline, current)
    
    # 要対応ケース
    actionable_cases = extract_actionable_cases(baseline, current)
    
    # structured_query_tool 専用分析
    sq_records = [r for r in current.records if r.route == "structured_query_tool"]
    sq_analysis = None
    if sq_records:
        sq_count = len(sq_records)
        res_gen = sum(1 for r in sq_records if r.response_generated)
        ans_ok = sum(1 for r in sq_records if r.answer_ok)
        # ブロック系（fail-safe）の判定: reason_code が特定のコード
        block_codes = {"unsupported_query", "validation_failed", "join_like_query_blocked", "write_operation_blocked"}
        fail_safe_count = sum(1 for r in sq_records if r.reason_code in block_codes)
        
        # ソース別件数
        source_counts = {}
        for r in sq_records:
            sname = r.source_name or "unknown"
            source_counts[sname] = source_counts.get(sname, 0) + 1
            
        # 理由コード Top N
        reason_dist = {}
        for r in sq_records:
            if r.reason_code != "SUCCESS":
                reason_dist[r.reason_code] = reason_dist.get(r.reason_code, 0) + 1
        
        sq_analysis = {
            "total_count": sq_count,
            "response_generated_rate": res_gen / sq_count,
            "answer_ok_rate": ans_ok / sq_count,
            "fail_safe_rate": fail_safe_count / sq_count,
            "source_counts": dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]),
            "reason_codes": dict(sorted(reason_dist.items(), key=lambda x: x[1], reverse=True)[:top_n])
        }

    from datetime import datetime
    return {
        "title": title,
        "baseline": baseline,
        "current": current,
        "diffs": diffs,
        "summary_points": summary_points,
        "actionable_cases": actionable_cases,
        "added_cases": list(added_cases),
        "removed_cases": list(removed_cases),
        "top_n": top_n,
        "sq_analysis": sq_analysis,
        "timestamp": current.metadata.get("timestamp", datetime.now().isoformat())
    }

def main():
    parser = argparse.ArgumentParser(description="評価結果レポート生成ツール")
    parser.add_argument("--baseline", type=str, help="Baseline JSON ファイルパス")
    parser.add_argument("--current", type=str, required=True, help="Current JSON ファイルパス")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "reports", "report.md"), help="出力ファイルパス")
    parser.add_argument("--title", type=str, default="AI Agent Evaluation Report", help="レポートタイトル")
    parser.add_argument("--top-n", type=int, default=5, help="Top N 表示件数")
    parser.add_argument("--format", type=str, choices=["markdown", "html", "both"], default="markdown", help="出力形式")
    
    args = parser.parse_args()
    
    current_report = load_report(args.current)
    baseline_report = load_report(args.baseline) if args.baseline else None
    
    # 共通コンテキストの構築
    ctx = build_view_context(baseline_report, current_report, args.title, args.top_n)
    
    # Jinja2 テンプレートのセットアップ
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    
    # 出力処理
    formats = [args.format] if args.format != "both" else ["markdown", "html"]
    
    for fmt in formats:
        if fmt == "markdown":
            template = env.get_template("report_md.j2")
            out_path = args.output if args.output.endswith(".md") else f"{args.output}.md"
        else:
            template = env.get_template("report_html.j2")
            # 出力先パスの調整
            if args.output.endswith(".md"):
                out_path = args.output.replace(".md", ".html")
            elif args.output.endswith(".html"):
                out_path = args.output
            else:
                out_path = f"{args.output}.html"
            
        # ディレクトリの作成
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        rendered = template.render(**ctx)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"レポートを生成しました ({fmt}): {out_path}")

if __name__ == "__main__":
    main()

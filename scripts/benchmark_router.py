import asyncio
import json
import os
import sys
import time
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
import uuid

# Dataset covering all aspects
DATASET = [
    # Direct
    "こんにちは", "hello!", "おはよう", "ありがとう", "こんばんは、今日は疲れたよ",
    # Calc
    "1足す1は", "5 * 52", "100の10%は？", "2+3*4", "1000わる3はいくつ",
    # Definition
    "RAGとは何ですか", "pgvectorって何", "LangChainの意味を教えて", "ベクトル検索の定義", "BM25とは",
    # Compare
    "RAGとFine-tuningの違い", "Python vs Go", "AとBのメリットを比較", "メリットとデメリットを教えて", "LangChainとLlamaIndexの使い分け",
    # Retrieval Complex
    "RAGのチューニングポイントは何でしょうか", "FastAPIでのCORS設定はどうやるか", "ベクトル検索の具体的な実装方法", "LangGraphで複数エージェントを構築する手順", "LLMとの付き合い方を教えてください"
]

async def run_eval_for_config(heuristic_enabled: bool) -> Dict[str, Any]:
    os.environ["ROUTER_HEURISTIC_ENABLED"] = "true" if heuristic_enabled else "false"
    
    # Reload settings module so the lru_cache is cleared or we bypass it
    from config import settings
    settings.get_settings.cache_clear()
    _ = settings.get_settings()

    from application.agents.graph import graph
    
    results = []
    
    for q in DATASET:
        session_id = str(uuid.uuid4())
        inputs = {"messages": [HumanMessage(content=q)]}
        config = {"configurable": {"thread_id": session_id}}
        
        final_state = {}
        started_at = time.monotonic()
        try:
            async for event in graph.astream(inputs, config=config, stream_mode="values"):
                final_state = event
        except Exception as e:
            print(f"Error on {q}: {e}")
            continue
        e2e_latency_ms = int((time.monotonic() - started_at) * 1000)
        
        results.append({
            "query": q,
            "query_type": final_state.get("query_type", "unknown"),
            "routing_layer": final_state.get("routing_layer", "unknown"),
            "llm_router_invoked": final_state.get("llm_router_invoked", False),
            "router_latency_ms": final_state.get("route_decision_latency_ms", 0),
            "e2e_latency_ms": e2e_latency_ms,
            "timeout_stages": list(final_state.get("timeout_stages", [])),
            "fallback_stages": list(final_state.get("fallback_stages", [])),
            "confidence": float(final_state.get("confidence", 0.0)),
            "retrieval_critic_skipped_reason": final_state.get("retrieval_critic_skipped_reason"),
            "answer_critic_skipped_reason": final_state.get("answer_critic_skipped_reason"),
            "warning": final_state.get("warning")
        })
    
    return results

def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    heuristic_hits = sum(1 for r in results if r["routing_layer"] == "heuristic")
    llm_invocations = sum(1 for r in results if r["llm_router_invoked"])
    router_timeouts = sum(1 for r in results if "router" in r["timeout_stages"])
    fallbacks = sum(1 for r in results if len(r["fallback_stages"]) > 0)
    
    router_latencies = [r["router_latency_ms"] for r in results]
    e2e_latencies = [r["e2e_latency_ms"] for r in results]
    
    metrics = {
        "heuristic_hit_rate": heuristic_hits / total,
        "llm_router_invocation_rate": llm_invocations / total,
        "router_timeout_rate": router_timeouts / total,
        "fallback_rate": fallbacks / total,
        "route_decision_latency_p50": np.percentile(router_latencies, 50),
        "route_decision_latency_p95": np.percentile(router_latencies, 95),
        "end_to_end_latency_p50": np.percentile(e2e_latencies, 50),
        "end_to_end_latency_p95": np.percentile(e2e_latencies, 95),
        "query_type_breakdown": {}
    }
    
    for qt in set(r["query_type"] for r in results):
        qt_results = [r for r in results if r["query_type"] == qt]
        qt_count = len(qt_results)
        
        # for compare specifically we need some extras
        qt_metrics = {
            "count": qt_count,
            "heuristic_hit_rate": sum(1 for r in qt_results if r["routing_layer"] == "heuristic") / qt_count,
            "router_timeout_rate": sum(1 for r in qt_results if "router" in r["timeout_stages"]) / qt_count,
        }
        
        if qt in ["compare"]:
            confs = [r["confidence"] for r in qt_results]
            critic_degrades = sum(1 for r in qt_results if r.get("retrieval_critic_skipped_reason") or r.get("answer_critic_skipped_reason"))
            warnings = sum(1 for r in qt_results if r["warning"] is not None)
            
            qt_metrics["avg_confidence"] = np.mean(confs)
            qt_metrics["critic_degraded_rate"] = critic_degrades / qt_count
            qt_metrics["warning_rate"] = warnings / qt_count
            
        metrics["query_type_breakdown"][qt] = qt_metrics
        
    return metrics

def print_metrics(label: str, metrics: Dict[str, Any]):
    print(f"\\n{'='*15} {label} {'='*15}")
    print(f"Heuristic Hit Rate: {metrics['heuristic_hit_rate']*100:.1f}%")
    print(f"LLM Router Invocation: {metrics['llm_router_invocation_rate']*100:.1f}%")
    print(f"Router Timeout Rate: {metrics['router_timeout_rate']*100:.1f}%")
    print(f"Overall Fallback Rate: {metrics['fallback_rate']*100:.1f}%")
    print(f"Route Latency (P50/P95): {metrics['route_decision_latency_p50']:.0f}ms / {metrics['route_decision_latency_p95']:.0f}ms")
    print(f"E2E Latency   (P50/P95): {metrics['end_to_end_latency_p50']:.0f}ms / {metrics['end_to_end_latency_p95']:.0f}ms")
    print("--- Query Type Breakdown ---")
    for qt, qm in metrics["query_type_breakdown"].items():
        print(f"  {qt} ({qm['count']}):")
        print(f"    Heuristic Hit Rate: {qm['heuristic_hit_rate']*100:.1f}%")
        print(f"    Router Timeout Rate: {qm['router_timeout_rate']*100:.1f}%")
        if qt in ["compare"]:
            print(f"    Avg Confidence: {qm['avg_confidence']:.2f}")
            print(f"    Critic Degraded Rate: {qm['critic_degraded_rate']*100:.1f}%")
            print(f"    Warning Rate: {qm['warning_rate']*100:.1f}%")

async def main():
    print("Running Baseline (Heuristic = False)...")
    base_results = await run_eval_for_config(heuristic_enabled=False)
    
    print("\\nRunning Current (Heuristic = True)...")
    curr_results = await run_eval_for_config(heuristic_enabled=True)
    
    base_metrics = aggregate_metrics(base_results)
    curr_metrics = aggregate_metrics(curr_results)
    
    print_metrics("BEFORE HEURISTIC", base_metrics)
    print_metrics("AFTER HEURISTIC", curr_metrics)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())

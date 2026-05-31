import asyncio
import json
import os
import sys
import time
import numpy as np
from typing import Dict, Any, List
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# 10 Compare cases
DATASET = [
    "RAGとFine-tuningの違い", 
    "Python vs Go", 
    "AとBのメリットを比較", 
    "メリットとデメリットを教えて", 
    "LangChainとLlamaIndexの使い分け",
    "Llama 3とGPT-4の特徴の比較",
    "DockerとKubernetesはどちらを使うべきですか？",
    "MySQLとPostgreSQLの違いは何ですか",
    "ReactとVue.jsの比較",
    "AWSとGCPの料金とパフォーマンスの違い"
]

async def run_eval() -> Dict[str, Any]:
    os.environ["ROUTER_HEURISTIC_ENABLED"] = "true"
    
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
            "confidence": float(final_state.get("confidence", 0.0)),
            "retrieval_critic_skipped_reason": final_state.get("retrieval_critic_skipped_reason"),
            "answer_critic_skipped_reason": final_state.get("answer_critic_skipped_reason"),
            "warning": final_state.get("warning")
        })
        print(f"Processed: {q} (Type: {final_state.get('query_type')}, Conf: {final_state.get('confidence', 0):.2f})")
    
    return results

def print_metrics(results: List[Dict[str, Any]]):
    total = len(results)
    if total == 0:
        print("No results.")
        return

    # Filter to only 'compare' query types to strictly evaluate compare
    compare_results = [r for r in results if r["query_type"] == "compare"]
    qt_count = len(compare_results)
    
    if qt_count == 0:
        print("No 'compare' type queries recognized.")
        return
        
    confs = [r["confidence"] for r in compare_results]
    critic_degrades = sum(1 for r in compare_results if r.get("retrieval_critic_skipped_reason") or r.get("answer_critic_skipped_reason"))
    warnings = sum(1 for r in compare_results if r["warning"] is not None)
    
    avg_confidence = np.mean(confs)
    critic_degraded_rate = critic_degrades / qt_count
    warning_rate = warnings / qt_count
    
    print(f"\\n{'='*15} COMPARE BENCHMARK METRICS {'='*15}")
    print(f"Total queries: {total}, Recognized as compare: {qt_count}")
    print(f"Avg Confidence: {avg_confidence:.2f} (Target: >= 0.65)")
    print(f"Critic Degraded Rate: {critic_degraded_rate*100:.1f}% (Target: <= 20.0%)")
    print(f"Warning Rate: {warning_rate*100:.1f}% (Target: Improvement)")
    
    print("\\nDetailed results:")
    for r in compare_results:
        print(f"  Q: {r['query']}")
        ret_skip = r.get('retrieval_critic_skipped_reason')
        ans_skip = r.get('answer_critic_skipped_reason')
        print(f"    Confidence: {r['confidence']:.2f}, Warning: {r['warning']}, Degraded: {bool(ret_skip or ans_skip)}, RetSkip: {ret_skip}, AnsSkip: {ans_skip}")

async def main():
    print("Running Benchmark for Compare Fast-Path...")
    results = await run_eval()
    print_metrics(results)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())

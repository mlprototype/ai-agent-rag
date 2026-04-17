import json
import os
import sys
import asyncio
import time

# applicationやdomainなどからのインポートを可能にするため、プロジェクトのルートをsys.pathに追加します。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from application.agents.graph import graph
from infrastructure.retrieval.vector_store import get_vector_store

# 回答の類似性評価のための審査員（Judge）としてLLMを設定
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "あなたは専門のエバリュエーターです。実際の回答と期待される回答を比較するのがあなたの任務です。"
               "実際の回答の類似性/正確性を0.0から1.0のスケールで評価してください（1.0は完全に正確/類似していることを意味します）。"
               "出力は0.0から1.0の間の単一の浮動小数点数のみにしてください。"),
    ("human", "期待される回答: {expected_answer}\nActual Answer: {actual_answer}\nScore:")
])

evaluator_chain = EVAL_PROMPT | evaluator_llm

def assess_answer_similarity(expected: str, actual: str) -> float:
    try:
        response = evaluator_chain.invoke({"expected_answer": expected, "actual_answer": actual})
        return float(response.content.strip())
    except Exception as e:
        print(f"類似度の評価中にエラーが発生しました: {e}")
        return 0.0

def get_bigrams(text: str) -> set:
    """文字列からスペースを除去し、2文字ずつのペア（バイグラム）のセットを生成します（日本語の一致判定用）。"""
    text = text.replace(" ", "").replace("　", "").replace("\n", "").lower()
    if len(text) < 2:
        return set([text])
    return set([text[i:i+2] for i in range(len(text) - 1)])

def assess_recall_at_k(query: str, expected_snippet: str, k: int = 3) -> bool:
    """期待されるスニペット、またはそれに大きく一致する部分が取得されたドキュメントに含まれているかを確認します。"""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    
    # 日本語対応のため、文字のバイグラム（2文字のペア）で一致率を計算します
    expected_bigrams = get_bigrams(expected_snippet)
    
    # 少なくとも1つのドキュメントに期待される内容の50%以上のバイグラムが含まれている場合、リコールをポジティブと見なします
    for doc in docs:
        doc_bigrams = get_bigrams(doc.page_content)
        overlap = len(expected_bigrams.intersection(doc_bigrams))
        if len(expected_bigrams) > 0 and (overlap / len(expected_bigrams)) > 0.5:
            return True
            
    return False

async def run_evaluation():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    total_similarity = 0.0
    total_recall = 0
    total_retry = 0
    total_iterations = 0
    total_critic_pass = 0
    total_retrieval_queries = 0
    total_router_uncertain = 0
    total_generate_forced = 0
    total_retrieval_degraded = 0
    total_retrieval_critic_skipped = 0
    total_answer_critic_skipped = 0
    query_type_stats: dict[str, dict[str, float]] = {}
    num_questions = len(data)

    print(f"{num_questions} 件の質問の評価を開始します...\n")

    for i, item in enumerate(data):
        question = item["question"]
        expected = item["expected_answer"]

        # 1. ツールが非同期になったため、エージェントグラフを非同期で実行します
        import uuid
        session_id = str(uuid.uuid4())
        inputs = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": session_id}}
        final_state = {}
        started_at = time.monotonic()
        async for event in graph.astream(inputs, config=config, stream_mode="values"):
            final_state = event
        latency_ms = int((time.monotonic() - started_at) * 1000)

        actual_answer = final_state.get("answer", "")
        query_type = final_state.get("query_type", "unknown")
        route = final_state.get("route")
        router_uncertain = bool(final_state.get("router_uncertain", False))
        retry_count = int(final_state.get("retry_count", 0))
        answer_ok = bool(final_state.get("answer_ok", True))
        must_generate = bool(final_state.get("must_generate", False))
        retrieval_degraded = bool(final_state.get("retrieval_degraded", False))
        retrieval_critic_skipped = bool(final_state.get("retrieval_critic_skipped_reason"))
        answer_critic_skipped = bool(final_state.get("answer_critic_skipped_reason"))
        timeout_stages = list(final_state.get("timeout_stages", []))
        fallback_used = bool(final_state.get("fallback_stages", []))
        answer_confidence = round(float(final_state.get("confidence", 0.5)), 2)
        total_router_uncertain += 1 if router_uncertain else 0
        if route == "agentic_retrieval":
            total_retrieval_queries += 1
            total_retry += 1 if retry_count > 0 else 0
            total_iterations += retry_count + 1
            total_critic_pass += 1 if answer_ok else 0
            total_generate_forced += 1 if must_generate else 0
            total_retrieval_degraded += 1 if retrieval_degraded else 0
            total_retrieval_critic_skipped += 1 if retrieval_critic_skipped else 0
            total_answer_critic_skipped += 1 if answer_critic_skipped else 0

        stats = query_type_stats.setdefault(
            query_type,
            {
                "count": 0,
                "latency_total": 0.0,
                "fallback_used": 0.0,
                "router_timeout": 0.0,
                "decompose_timeout": 0.0,
            },
        )
        stats["count"] += 1
        stats["latency_total"] += latency_ms
        stats["fallback_used"] += 1 if fallback_used else 0
        stats["router_timeout"] += 1 if "router" in timeout_stages else 0
        stats["decompose_timeout"] += 1 if any(stage in {"decompose", "rewrite"} for stage in timeout_stages) else 0
        
        # 2. 類似度の評価
        sim_score = assess_answer_similarity(expected, actual_answer)
        total_similarity += sim_score
        
        # 3. Recall@k の評価
        recall_hit = assess_recall_at_k(question, expected)
        total_recall += 1 if recall_hit else 0

        print(f"Q{i+1}: {question}")
        print(f"  Query Type: {query_type}")
        print(f"  Route: {route}")
        print(f"  Latency: {latency_ms}ms")
        print(f"  Fallback Used: {fallback_used}")
        print(f"  Timeout Stage: {', '.join(timeout_stages) if timeout_stages else 'none'}")
        print(f"  Answer Confidence: {answer_confidence:.2f}")
        print(f"  類似度: {sim_score:.2f}")
        print(f"  Recall@3: {'ヒット' if recall_hit else 'ミス'}\n")

    avg_similarity = total_similarity / num_questions
    recall_rate = total_recall / num_questions
    retry_rate = (total_retry / total_retrieval_queries) if total_retrieval_queries else 0.0
    avg_iteration = (total_iterations / total_retrieval_queries) if total_retrieval_queries else 0.0
    critic_pass_rate = (total_critic_pass / total_retrieval_queries) if total_retrieval_queries else 0.0
    router_uncertain_rate = (total_router_uncertain / num_questions) if num_questions else 0.0
    generate_forced_rate = (total_generate_forced / total_retrieval_queries) if total_retrieval_queries else 0.0
    retrieval_degraded_rate = (total_retrieval_degraded / total_retrieval_queries) if total_retrieval_queries else 0.0
    retrieval_critic_skip_rate = (total_retrieval_critic_skipped / total_retrieval_queries) if total_retrieval_queries else 0.0
    answer_critic_skip_rate = (total_answer_critic_skipped / total_retrieval_queries) if total_retrieval_queries else 0.0

    print("===== 評価結果 =====")
    print(f"平均回答類似度: {avg_similarity:.2f}")
    print(f"Recall@3 レート: {recall_rate:.2f}")
    print(f"Retry Rate: {retry_rate:.2f}")
    print(f"Avg Iteration: {avg_iteration:.2f}")
    print(f"Critic Pass Rate: {critic_pass_rate:.2f}")
    print(f"Router Uncertain Rate: {router_uncertain_rate:.2f}")
    print(f"Generate Forced Rate: {generate_forced_rate:.2f}")
    print(f"Retrieval Degraded Rate: {retrieval_degraded_rate:.2f}")
    print(f"Retrieval Critic Skip Rate: {retrieval_critic_skip_rate:.2f}")
    print(f"Answer Critic Skip Rate: {answer_critic_skip_rate:.2f}")
    print("Query Type Breakdown:")
    for query_type, stats in sorted(query_type_stats.items()):
        count = int(stats["count"])
        avg_latency = (stats["latency_total"] / count) if count else 0.0
        fallback_rate = (stats["fallback_used"] / count) if count else 0.0
        router_timeout_rate = (stats["router_timeout"] / count) if count else 0.0
        decompose_timeout_rate = (stats["decompose_timeout"] / count) if count else 0.0
        print(
            f"  {query_type}: count={count}, avg_latency_ms={avg_latency:.0f}, "
            f"fallback_rate={fallback_rate:.2f}, router_timeout_rate={router_timeout_rate:.2f}, "
            f"decompose_timeout_rate={decompose_timeout_rate:.2f}"
        )
    print("====================")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_evaluation())

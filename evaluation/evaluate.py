import json
import os
import sys
import asyncio
import time
from datetime import datetime

# applicationやdomainなどからのインポートを可能にするため、プロジェクトのルートをsys.pathに追加します。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from application.agents.graph import graph
from evaluation.schema import EvalRecord
from evaluation.aggregator import aggregate_results

"""
このファイルは、システム全体の評価を実行するためのメインスクリプトです。
データセットを読み込み、各クエリに対してエージェントを実行し、結果を収集します。
収集されたデータは schema.py で定義されたモデルに変換され、aggregator.py で集計された後、JSONレポートとして出力されます。
"""

_EVALUATOR_CHAIN = None

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "あなたは専門のエバリュエーターです。実際の回答と期待される回答を比較するのがあなたの任務です。"
               "実際の回答の類似性/正確性を0.0から1.0のスケールで評価してください（1.0は完全に正確/類似していることを意味します）。"
               "出力は0.0から1.0の間の単一の浮動小数点数のみにしてください。"),
    ("human", "期待される回答: {expected_answer}\nActual Answer: {actual_answer}\nScore:")
])

def _get_evaluator_chain():
    """Judge を最初の回答類似度評価時に初期化し、以降は再利用する。"""
    global _EVALUATOR_CHAIN
    if _EVALUATOR_CHAIN is None:
        evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        _EVALUATOR_CHAIN = EVAL_PROMPT | evaluator_llm
    return _EVALUATOR_CHAIN

def assess_answer_similarity(expected: str, actual: str) -> float:
    """LLMを使用して期待される回答と実際の回答の類似度を評価します。"""
    if not actual or not actual.strip():
        return 0.0
    try:
        response = _get_evaluator_chain().invoke({"expected_answer": expected, "actual_answer": actual})
        return float(response.content.strip())
    except Exception as e:
        print(f"類似度の評価中にエラーが発生しました: {e}")
        return 0.0


def _critic_degraded(state: dict) -> bool:
    critic_stages = {"answer_critic", "retrieval_critic"}
    return bool(
        state.get("answer_critic_skipped_reason")
        or state.get("retrieval_critic_skipped_reason")
        or str(state.get("critique_reason", "")).startswith("critic_fallback:")
        or critic_stages.intersection(state.get("skipped_stages", []))
        or critic_stages.intersection(state.get("fallback_stages", []))
        or critic_stages.intersection(state.get("timeout_stages", []))
    )

async def run_evaluation():
    """評価データセットを実行し、結果を集計してJSONレポートを保存します。"""
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    records = []
    num_questions = len(data)
    start_time_all = time.monotonic()

    print(f"{num_questions} 件の質問の評価を開始します...\n")

    for i, item in enumerate(data):
        question = item["question"]
        expected = item["expected_answer"]
        expected_qt = item.get("expected_query_type")
        expected_route = item.get("expected_route")

        import uuid
        session_id = str(uuid.uuid4())
        inputs = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": session_id}}
        final_state = {}
        started_at = time.monotonic()
        
        # エージェントグラフの実行
        async for event in graph.astream(inputs, config=config, stream_mode="values"):
            final_state = event
        latency_ms = int((time.monotonic() - started_at) * 1000)

        # 状態情報の抽出
        actual_answer = final_state.get("answer", "")
        query_type = final_state.get("query_type", "unknown")
        route = final_state.get("route")
        router_uncertain = bool(final_state.get("router_uncertain", False))
        answer_ok = bool(final_state.get("answer_ok", True))
        
        # Sprint 3 以降の拡張フィールド
        is_retrieval_degraded = bool(final_state.get("retrieval_degraded", False))
        is_critic_degraded = _critic_degraded(final_state)
        is_strict_insufficient = bool(final_state.get("strict_insufficient_response", False))
        
        timeout_stages = list(final_state.get("timeout_stages", []))
        fallback_stages = list(final_state.get("fallback_stages", []))
        answer_confidence = round(float(final_state.get("confidence", 0.0)), 2)
        
        # 警告コードの取得または生成
        warning_codes = list(final_state.get("warning_codes", []))
        warning_codes.extend(f"TIMEOUT_{s.upper()}" for s in timeout_stages)
        if router_uncertain:
            warning_codes.append("ROUTER_UNCERTAIN")
        if answer_confidence < 0.3 and actual_answer:
            warning_codes.append("LOW_CONFIDENCE")
        warning_codes = list(dict.fromkeys(warning_codes))

        # 検索品質の補正（システム側が high と言っていても degraded なら補正）
        # タイムアウト等で縮退が発生している場合、たとえ graph が high と主張しても実態は中程度以下とする
        retrieval_quality = final_state.get("retrieval_quality_level")
        if not retrieval_quality or (is_retrieval_degraded and retrieval_quality == "high"):
            retrieval_quality = "medium" if is_retrieval_degraded else "high"
        if is_strict_insufficient:
            retrieval_quality = "low"
            
        sources = final_state.get("sources", [])
        source_name = None
        if sources and isinstance(sources, list) and len(sources) > 0:
            # 辞書型であることを確認して取得
            first_source = sources[0]
            if isinstance(first_source, dict):
                source_name = first_source.get("source") or first_source.get("name")
        if not source_name:
            source_name = final_state.get("structured_query_source_name")

        # 回答類似度の評価（expected_answer は Retrieval Ground Truth ではない）
        sim_score = assess_answer_similarity(expected, actual_answer)

        # reason_code の精緻化: 単なる成功・失敗ではなく、ガードの作動や品質不足を区別する
        if not actual_answer:
            reason_code = "ERROR"
        elif is_strict_insufficient:
            reason_code = "NO_DATA"  # 検索結果がゼロで回答拒否
        elif route == "structured_query_tool" and final_state.get("structured_query_reason_code"):
            reason_code = final_state["structured_query_reason_code"]
        elif not answer_ok:
            if "low_confidence_definition_guard" in warning_codes:
                reason_code = "GUARD_BLOCK"  # 低確信度ガードによる回答拒否
            elif is_critic_degraded:
                reason_code = "CRITIC_FAIL"   # Critic判定による品質不足
            else:
                reason_code = "QUALITY_FAIL"  # その他の品質不合格
        else:
            reason_code = "SUCCESS"

        # EvalRecordへの変換
        runtime_fallback_level = final_state.get("fallback_level")
        fallback_level = (
            runtime_fallback_level
            if runtime_fallback_level not in (None, "full_path", "NONE")
            else f"LEVEL_{len(fallback_stages)}" if fallback_stages else "NONE"
        )
        record = EvalRecord(
            query=question,
            query_type=query_type,
            route=route,
            answer=actual_answer,
            confidence=answer_confidence,
            fallback_level=fallback_level,
            latency_ms=latency_ms,
            retrieval_quality_level=retrieval_quality,
            source_name=source_name,
            similarity=sim_score,
            response_generated=bool(actual_answer and actual_answer.strip()),
            answer_ok=answer_ok,
            degraded=is_retrieval_degraded or is_critic_degraded or is_strict_insufficient or fallback_level != "NONE" or bool(final_state.get("partial_retrieval_used")),
            critic_degraded=is_critic_degraded,
            retrieval_degraded=is_retrieval_degraded,
            strict_insufficient_response=is_strict_insufficient,
            warning=bool(final_state.get("warning") or warning_codes),
            warning_codes=warning_codes,
            reason_code=reason_code,
            expected_query_type=expected_qt,
            expected_route=expected_route
        )
        records.append(record)

        print(f"Q{i+1}: {question}")
        print(f"  Query Type: {query_type}, Route: {route}")
        print(f"  Latency: {latency_ms}ms, Confidence: {answer_confidence:.2f}, Similarity: {sim_score:.2f}")
        if warning_codes:
            print(f"  Warnings: {', '.join(warning_codes)}")
        print("-" * 20)

    # 全体の実行時間
    execution_time_ms = int((time.monotonic() - start_time_all) * 1000)
    
    # 集計の実行
    report = aggregate_results(records, execution_time_ms)
    
    # 結果の保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_filename = f"eval_results_{timestamp}.json"
    output_path = os.path.join(results_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    
    print(f"\n評価完了。レポートを保存しました: {output_path}")
    print("\n===== 概要サマリー =====")
    print(f"全件数: {report.summary.total_count}")
    print(f"回答生成率: {report.summary.response_generated_rate:.2%}")
    print(f"品質基準合格率: {report.summary.answer_ok_rate:.2%}")
    print(f"平均類似度: {report.summary.avg_similarity:.2f}")
    print(f"レイテンシ p50: {report.summary.latency_p50_ms}ms")
    print(f"レイテンシ p95: {report.summary.latency_p95_ms}ms")
    print("========================")

if __name__ == "__main__":
    asyncio.run(run_evaluation())

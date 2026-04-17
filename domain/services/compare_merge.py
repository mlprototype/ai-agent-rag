from typing import List, Dict, Any, Tuple


def merge_compare_contexts(
    targets: List[str], 
    retrieval_results: Dict[str, Any]
) -> Tuple[str, int, int, bool, str | None]:
    """
    A/B の retrieval 結果を compare_generate 向けに整形する。
    
    戻り値:
        (packed_context, doc_count_a, doc_count_b, coverage_ok, fallback_reason)
    """
    if len(targets) < 2:
        return "", 0, 0, False, "targets_length_invalid"
        
    target_a = targets[0]
    target_b = targets[1]
    
    res_a = retrieval_results.get(target_a, {})
    res_b = retrieval_results.get(target_b, {})
    
    chunks_a = res_a.get("chunks", [])
    chunks_b = res_b.get("chunks", [])
    
    doc_count_a = len(chunks_a)
    doc_count_b = len(chunks_b)
    
    # Coverage check
    if doc_count_a == 0 and doc_count_b == 0:
        return "", 0, 0, False, "both_retrievals_empty"
    if doc_count_a == 0:
        return "", doc_count_a, doc_count_b, False, "target_a_retrieval_empty"
    if doc_count_b == 0:
        return "", doc_count_a, doc_count_b, False, "target_b_retrieval_empty"

    # Context packaging
    packed_lines = []
    
    packed_lines.append(f"[Item A: {target_a}]")
    if doc_count_a > 0:
        # res_a["context"] generally contains the compressed context from the retrieval service
        packed_lines.append(res_a.get("context", "情報が見つかりませんでした。"))
    else:
        packed_lines.append("関連する情報が見つかりませんでした。")
        
    packed_lines.append("")
    packed_lines.append(f"[Item B: {target_b}]")
    if doc_count_b > 0:
        packed_lines.append(res_b.get("context", "情報が見つかりませんでした。"))
    else:
        packed_lines.append("関連する情報が見つかりませんでした。")
        
    packed_context = "\n".join(packed_lines)
    
    return packed_context, doc_count_a, doc_count_b, True, None

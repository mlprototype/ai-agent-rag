import re
from typing import List, Dict, Any, Tuple


def merge_compare_contexts(
    targets: List[str], 
    retrieval_results: Dict[str, Any]
) -> Tuple[str, int, int, bool, str | None, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    A/B の retrieval 結果を compare_generate 向けに整形し、
    citation_id が衝突しないようにグローバルに再割り当てする。
    
    戻り値:
        (packed_context, doc_count_a, doc_count_b, coverage_ok, fallback_reason, all_chunks, unique_sources)
    """
    if len(targets) < 2:
        return "", 0, 0, False, "targets_length_invalid", [], []
        
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
        return "", 0, 0, False, "both_retrievals_empty", [], []
    if doc_count_a == 0:
        return "", doc_count_a, doc_count_b, False, "target_a_retrieval_empty", [], []
    if doc_count_b == 0:
        return "", doc_count_a, doc_count_b, False, "target_b_retrieval_empty", [], []

    unique_sources = []
    all_chunks = []
    seen = set()
    cid = 1
    
    def process_result(res: Dict[str, Any]) -> str:
        nonlocal cid
        context = res.get("context", "情報が見つかりませんでした。")
        sources = res.get("sources", [])
        all_chunks.extend(res.get("chunks", []))
        
        # 旧citation_idから新citation_idへのマッピング
        mapping = {}
        for s in sources:
            old_id = s.get("citation_id")
            u = f"{s.get('doc_id', '')}::{s.get('chunk_id', '')}"
            if u not in seen:
                seen.add(u)
                s_copy = dict(s)
                s_copy["citation_id"] = cid
                unique_sources.append(s_copy)
                mapping[old_id] = cid
                cid += 1
            else:
                # 既出ソースの場合は既存のcitation_idを引く
                for us in unique_sources:
                    if f"{us.get('doc_id', '')}::{us.get('chunk_id', '')}" == u:
                        mapping[old_id] = us.get("citation_id")
                        break
        
        # コンテキスト内の [x] を新しい [y] に置換
        def replace_citation(match):
            old_str = match.group(1)
            new_ids = []
            for id_str in old_str.split(","):
                id_str = id_str.strip()
                if id_str.isdigit():
                    old_num = int(id_str)
                    if old_num in mapping:
                        new_ids.append(str(mapping[old_num]))
                    else:
                        new_ids.append(id_str)
                else:
                    new_ids.append(id_str)
            return f"[{', '.join(new_ids)}]"
            
        remapped_context = re.sub(r"\[([\d,\s]+)\]", replace_citation, context)
        return remapped_context

    # Context packaging
    packed_lines = []
    
    packed_lines.append(f"[Item A: {target_a}]")
    if doc_count_a > 0:
        packed_lines.append(process_result(res_a))
    else:
        packed_lines.append("関連する情報が見つかりませんでした。")
        
    packed_lines.append("")
    packed_lines.append(f"[Item B: {target_b}]")
    if doc_count_b > 0:
        packed_lines.append(process_result(res_b))
    else:
        packed_lines.append("関連する情報が見つかりませんでした。")
        
    packed_context = "\n".join(packed_lines)
    
    return packed_context, doc_count_a, doc_count_b, True, None, all_chunks, unique_sources

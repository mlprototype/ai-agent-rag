import asyncio
from typing import List, Optional, Dict, Any
from domain.services.retrieval_service import RetrievalService


def build_compare_subquery(target: str, aspect: Optional[str]) -> str:
    """
    対象ごとの独立比較用クエリを生成する。
    """
    base_keywords = "概要 特徴 用途 メリット デメリット"
    if aspect:
        return f"{target} {aspect} {base_keywords}"
    return f"{target} {base_keywords}"


async def run_compare_retrieval(targets: List[str], aspect: Optional[str]) -> Dict[str, Any]:
    """
    複数ターゲットに対して並列で RetrievalService.run を実行する。
    戻り値はターゲット名をキーとした dict を返す。
    例: { "Python": {"context": ..., "chunks": ...}, "Go": {...} }
    """
    results_map = {}
    
    # 検索タスクのリストを作成
    async def fetch(target: str):
        subquery = build_compare_subquery(target, aspect)
        result = await RetrievalService.run(subquery)
        return target, result

    tasks = [fetch(target) for target in targets]
    gathered = await asyncio.gather(*tasks)
    
    for t, r in gathered:
        results_map[t] = r
        
    return results_map

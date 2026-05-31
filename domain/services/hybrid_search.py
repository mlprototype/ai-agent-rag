import asyncio
import logging
import os
from typing import List
from domain.models.retrieval_models import RetrievedChunk
from infrastructure.retrieval.vector_store import get_async_vector_store
from infrastructure.retrieval.keyword_search import KeywordSearch

logger = logging.getLogger(__name__)

# 設定
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))
RETRIEVE_K_VECTOR = int(os.getenv("RETRIEVE_K_VECTOR", "30"))
RETRIEVE_K_KEYWORD = int(os.getenv("RETRIEVE_K_KEYWORD", "30"))
STAGE_TIMEOUT_HYBRID = int(os.getenv("STAGE_TIMEOUT_MS_HYBRID", "5000")) / 1000


def _normalize_scores(scores: List[float]) -> List[float]:
    """min-max 正規化。全て同一値の場合は 0.5 を返す。"""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridSearch:
    """
    Vector Search と Keyword Search の結果を統合する Hybrid Search。
    スコア正規化後に加重平均で hybrid_score を算出する。
    Keyword Search 失敗時は Vector Search のみで続行（フォールバック）。
    """

    @classmethod
    async def search(cls, queries: List[str], top_k: int = 30) -> List[RetrievedChunk]:
        """
        複数のクエリ（original + rewrite）で Hybrid Search を実行する。

        Args:
            queries: 検索クエリのリスト（original_query, rewrite_query）
            top_k: 返却する最大チャンク数

        Returns:
            hybrid_score でソートされた RetrievedChunk のリスト
        """
        # 全クエリの結果を集約するための辞書 (content hash -> RetrievedChunk)
        chunks_map: dict[str, RetrievedChunk] = {}

        for query in queries:
            try:
                merged = await asyncio.wait_for(
                    cls._search_single_query(query),
                    timeout=STAGE_TIMEOUT_HYBRID
                )
                # 重複排除: 同じ content の場合はスコアが高い方を採用
                for chunk in merged:
                    key = hash(chunk.content)
                    if key not in chunks_map or chunk.hybrid_score > chunks_map[key].hybrid_score:
                        chunks_map[key] = chunk
            except asyncio.TimeoutError:
                logger.warning(f"Hybrid search がタイムアウトしました: '{query}'")
            except Exception as e:
                logger.warning(f"Hybrid search に失敗しました: {e}")

        # hybrid_score でソートして top_k 件を返却
        sorted_chunks = sorted(chunks_map.values(), key=lambda c: c.hybrid_score, reverse=True)
        return sorted_chunks[:top_k]

    @classmethod
    async def _search_single_query(cls, query: str) -> List[RetrievedChunk]:
        """単一クエリで Vector + Keyword を並列実行し、スコアを統合する。"""
        vector_store = get_async_vector_store()

        # Vector Search と Keyword Search を並列実行
        vector_task = vector_store.asimilarity_search_with_score(query, k=RETRIEVE_K_VECTOR)
        keyword_task = KeywordSearch.search(query, k=RETRIEVE_K_KEYWORD)

        vector_results, keyword_results = await asyncio.gather(
            vector_task,
            keyword_task,
            return_exceptions=True
        )

        # Vector 結果の処理
        vector_chunks: dict[str, RetrievedChunk] = {}
        if not isinstance(vector_results, Exception) and vector_results:
            # pgvector のスコアはコサイン距離 → 類似度に変換
            raw_scores = []
            for doc, score in vector_results:
                similarity = 1.0 - score if score <= 1.0 else score
                raw_scores.append(similarity)

            norm_scores = _normalize_scores(raw_scores)

            for i, (doc, score) in enumerate(vector_results):
                source = doc.metadata.get("source", "不明")
                chunk = RetrievedChunk(
                    doc_id=source,
                    chunk_id=RetrievedChunk.build_chunk_id(source, doc.page_content, doc.metadata),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    vector_score=norm_scores[i],
                )
                vector_chunks[doc.page_content] = chunk
        elif isinstance(vector_results, Exception):
            logger.warning(f"Vector search に失敗しました: {vector_results}")

        # Keyword 結果の処理
        keyword_map: dict[str, float] = {}
        if not isinstance(keyword_results, Exception) and keyword_results:
            raw_keyword_scores = [r[3] for r in keyword_results]
            norm_keyword_scores = _normalize_scores(raw_keyword_scores)
            for i, (doc_id, chunk_id, content, bm25_score, metadata) in enumerate(keyword_results):
                keyword_map[content] = norm_keyword_scores[i]
        elif isinstance(keyword_results, Exception):
            logger.warning(f"Keyword search に失敗（フォールバック: Vector のみ）: {keyword_results}")

        # スコア統合
        all_contents = set(list(vector_chunks.keys()) + list(keyword_map.keys()))
        merged: List[RetrievedChunk] = []

        for content in all_contents:
            if content in vector_chunks:
                chunk = vector_chunks[content]
                bm25_norm = keyword_map.get(content, 0.0)
                chunk.bm25_score = bm25_norm
                chunk.hybrid_score = round(
                    HYBRID_ALPHA * chunk.vector_score + (1 - HYBRID_ALPHA) * bm25_norm, 4
                )
                merged.append(chunk)
            else:
                # Keyword のみにヒットしたチャンク
                bm25_norm = keyword_map[content]
                # Vector 側のメタが無いため、keyword 結果から復元
                for doc_id, chunk_id, kw_content, _, metadata in keyword_results:
                    if kw_content == content:
                        chunk = RetrievedChunk(
                            doc_id=doc_id,
                            chunk_id=RetrievedChunk.build_chunk_id(doc_id, content, metadata),
                            content=content,
                            metadata=metadata if isinstance(metadata, dict) else {},
                            vector_score=0.0,
                            bm25_score=bm25_norm,
                            hybrid_score=round((1 - HYBRID_ALPHA) * bm25_norm, 4),
                        )
                        merged.append(chunk)
                        break

        return merged

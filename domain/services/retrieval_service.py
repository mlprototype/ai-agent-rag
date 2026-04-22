import asyncio
import json
import logging
from typing import List

from domain.models.retrieval_models import PreparedContext, RetrievalSearchResult, RetrievedChunk
from domain.services.query_rewriter import QueryRewriter
from domain.services.hybrid_search import HybridSearch
from domain.services.confidence import ConfidenceEstimator
from domain.services.compressor import ExtractiveCompressor

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    ナレッジベースの検索に関連するビジネスロジックを統括するドメインサービス。

    Phase 2.5 パイプライン:
      1. Query Rewrite — 検索向けクエリの生成
      2. Hybrid Search — Vector + Keyword の並列検索・スコア統合
      3. Confidence Estimation + Dynamic TopK — 確信度に基づく動的件数制御
      4. Extractive Compression — 関連文の抽出によるノイズ削減
      5. 構造化JSON出力 — LLM へのコンテキスト + Citation メタデータ

    フォールバック順:
      1. Compression skip（圧縮スキップ）
      2. Rewrite skip（書き換えスキップ）
      3. Keyword skip（Vector のみ）
    """

    RETRIEVAL_TIMEOUT_SECONDS = 5.0

    @classmethod
    async def search_knowledge_base(cls, query: str) -> str:
        """
        Phase 2.5 の強化パイプラインを実行し、構造化 JSON 文字列を返す。
        全体タイムアウト 5.0 秒を維持。
        """
        try:
            result = await asyncio.wait_for(
                cls.run(query),
                timeout=cls.RETRIEVAL_TIMEOUT_SECONDS
            )
            return json.dumps(
                {
                    "context": result["context"],
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                },
                ensure_ascii=False
            )
        except asyncio.TimeoutError:
            # 検索パイプライン全体がタイムアウトした場合、例外で落とさずに空結果とエラーメッセージを返してシステムを保護する
            logger.warning(f"パイプライン全体がタイムアウトしました: '{query}'")
            return json.dumps(
                {"results": [], "message": "【Error】検索パイプラインがタイムアウトしました。"},
                ensure_ascii=False
            )
        except Exception as e:
            logger.error(f"検索パイプラインに失敗しました: {e}")
            return json.dumps(
                {"results": [], "message": f"【Error】検索中に予期しないエラーが発生しました: {e}"},
                ensure_ascii=False
            )

    @classmethod
    async def search(cls, query: str) -> RetrievalSearchResult:
        """Query Rewrite + Hybrid Search + Dynamic TopK を実行する。"""
        rewrite_result = await QueryRewriter.rewrite(query)
        logger.info(f"[Pipeline] Stage 1 完了: queries={rewrite_result.combined_queries}")

        chunks = await HybridSearch.search(rewrite_result.combined_queries)
        logger.info(f"[Pipeline] Stage 2 完了: {len(chunks)} チャンクを取得")

        if not chunks:
            return RetrievalSearchResult(
                query=query,
                used_queries=rewrite_result.combined_queries,
                retrieved_chunks=[],
                selected_chunks=[],
                confidence=0.0,
                top_k=0,
            )

        confidence, top_k = ConfidenceEstimator.estimate(chunks)
        selected_chunks = chunks[:top_k]
        logger.info(f"[Pipeline] Stage 3 完了: confidence={confidence}, top_k={top_k}")

        return RetrievalSearchResult(
            query=query,
            used_queries=rewrite_result.combined_queries,
            retrieved_chunks=chunks,
            selected_chunks=selected_chunks,
            confidence=confidence,
            top_k=top_k,
        )

    @classmethod
    async def prepare_context(cls, query: str, chunks: List[RetrievedChunk]) -> PreparedContext:
        """圧縮コンテキストと API 返却用 sources を組み立てる。"""
        if not chunks:
            return PreparedContext(context="", sources=[])

        compression = await ExtractiveCompressor.compress(query, chunks)
        logger.info(f"[Pipeline] Stage 4 完了: 圧縮テキスト {len(compression.compressed_text)} 文字")

        sources = []
        for i, chunk in enumerate(chunks):
            sources.append({
                "citation_id": i + 1,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": chunk.hybrid_score,
                "hybrid_score": chunk.hybrid_score,
                "vector_score": chunk.vector_score,
                "bm25_score": chunk.bm25_score,
                "rerank_score": chunk.rerank_score,
                "snippet": chunk.content[:200],
            })

        return PreparedContext(
            context=compression.compressed_text,
            sources=sources,
        )

    @classmethod
    async def finalize_chunks(
        cls,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> dict:
        """chunks を最終選別し、Generate 用 context と sources を返す。"""
        if not chunks:
            return {"chunks": [], "context": "", "sources": [], "confidence": 0.0, "top_k": 0}

        score_key = "rerank_score" if any(chunk.rerank_score > 0 for chunk in chunks) else "hybrid_score"
        confidence, top_k = ConfidenceEstimator.estimate(chunks, score_key=score_key)
        ordered = sorted(chunks, key=lambda chunk: getattr(chunk, score_key, 0.0), reverse=True)
        selected_chunks = ordered[:top_k]
        prepared = await cls.prepare_context(query, selected_chunks)
        return {
            "chunks": selected_chunks,
            "context": prepared.context,
            "sources": prepared.sources,
            "confidence": confidence,
            "top_k": top_k,
        }

    # 関数の役割: 検索パイプライン全体の実行（内部API用）
    # 入出力: クエリ文字列を受け取り、チャンクやコンテキストを含む辞書を返す
    @classmethod
    async def run(cls, query: str) -> dict:
        search_result = await cls.search(query)
        prepared = await cls.prepare_context(query, search_result.selected_chunks)
        return {
            "context": prepared.context,
            "sources": prepared.sources,
            "confidence": search_result.confidence,
            "chunks": search_result.selected_chunks,
            "top_k": search_result.top_k,
        }

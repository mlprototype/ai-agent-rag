import logging
import os
import re
from typing import List, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

logger = logging.getLogger(__name__)

# 取得件数の設定
RETRIEVE_K_KEYWORD = int(os.getenv("RETRIEVE_K_KEYWORD", "30"))

# 非同期エンジンのシングルトン
_async_engine: AsyncEngine | None = None


def _get_async_engine() -> AsyncEngine:
    """PostgreSQL への非同期接続エンジンを取得（シングルトン）。"""
    global _async_engine
    if _async_engine is None:
        user = os.getenv("POSTGRES_USER", "admin")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "rag_db")
        connection_string = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
        _async_engine = create_async_engine(connection_string, pool_size=5, max_overflow=10)
    return _async_engine


class KeywordSearch:
    """
    PostgreSQL の全文検索（tsvector / ts_rank）を使用したキーワード検索。
    langchain_postgres の PGVector が管理する langchain_pg_embedding テーブルを直接参照する。
    """

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        日本語混在クエリでも BM25 が機能するように正規化を行う。
        - 疑問形の接尾辞の除去
        - CJKとASCIIの境界へのスペース挿入
        - 記号のスペース置換
        """
        # 1. 疑問形などの不要なサフィックスを除去
        suffixes = ["とはなんですか", "とは何ですか", "って何ですか", "って何", "とは", "について教えて", "について", "ですか", "ますか", "？", "?", "教えて", "詳細"]
        norm = query.strip()
        for suffix in suffixes:
            if norm.endswith(suffix):
                norm = norm[:-len(suffix)].strip()
        
        # 2. 記号をスペースに置換
        norm = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', ' ', norm)
        
        # 3. ASCII と CJK の境界にスペースを挿入
        # ASCII -> CJK
        norm = re.sub(r'([a-zA-Z0-9])([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])', r'\1 \2', norm)
        # CJK -> ASCII
        norm = re.sub(r'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])([a-zA-Z0-9])', r'\1 \2', norm)
        
        # 4. 連続するスペースを1つにする
        norm = re.sub(r'\s+', ' ', norm).strip()
        
        # 正規化結果が空になった場合は元のクエリから記号だけ抜いたものを使う
        if not norm:
            norm = re.sub(r'[^\w\s]', ' ', query).strip()
            
        return norm

    @staticmethod
    async def search(query: str, k: int = RETRIEVE_K_KEYWORD) -> List[Tuple[str, str, str, float, dict]]:
        """
        キーワード検索を実行し、(doc_id, chunk_id, content, bm25_score, metadata) のリストを返す。

        PostgreSQL の to_tsvector / plainto_tsquery + ts_rank を使用。
        'simple' コンフィグを使用することで、日本語の形態素解析に依存せず
        文字単位のマッチングで動作する。

        Returns:
            List of tuples: (doc_id, chunk_id, content, bm25_score, metadata)
        """
        engine = _get_async_engine()
        normalized_query = KeywordSearch.normalize_query(query)

        # langchain_pg_embedding テーブルの document カラムに対して FTS を実行
        # cmetadata カラムからメタデータを取得
        sql = text("""
            SELECT
                e.document,
                e.cmetadata,
                ts_rank(
                    to_tsvector('simple', e.document),
                    plainto_tsquery('simple', :query)
                ) AS rank
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = 'agentic_rag_docs'
              AND to_tsvector('simple', e.document) @@ plainto_tsquery('simple', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)

        try:
            async with engine.connect() as conn:
                result = await conn.execute(sql, {"query": normalized_query, "limit": k})
                rows = result.fetchall()

            results = []
            for i, row in enumerate(rows):
                content = row[0]
                metadata = row[1] if row[1] else {}
                bm25_score = float(row[2])
                source = metadata.get("source", "不明") if isinstance(metadata, dict) else "不明"
                chunk_id = f"{source}#k{i+1}"
                results.append((source, chunk_id, content, bm25_score, metadata))

            logger.info({
                "event": "keyword_search_executed",
                "original_query": query,
                "normalized_keyword_query": normalized_query,
                "bm25_hit_count": len(results),
                "all_bm25_zero": len(results) == 0,
            })

            return results

        except Exception as e:
            logger.error(f"Keyword search に失敗しました: {e}")
            return []

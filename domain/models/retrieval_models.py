import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    """
    Hybrid Search の結果として返される検索チャンクを表すデータクラス。
    Vector / Keyword 双方のスコアと統合スコアを保持する。
    """
    doc_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def effective_score(self) -> float:
        return self.rerank_score if self.rerank_score > 0 else self.hybrid_score

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "hybrid_score": self.hybrid_score,
            "rerank_score": self.rerank_score,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "RetrievedChunk":
        return cls(
            doc_id=data.get("doc_id", "不明"),
            chunk_id=data.get("chunk_id", "unknown"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}) or {},
            vector_score=float(data.get("vector_score", 0.0)),
            bm25_score=float(data.get("bm25_score", 0.0)),
            hybrid_score=float(data.get("hybrid_score", 0.0)),
            rerank_score=float(data.get("rerank_score", 0.0)),
        )

    @staticmethod
    def build_chunk_id(doc_id: str, content: str, metadata: Dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        for key in ("chunk_id", "id"):
            value = metadata.get(key)
            if value:
                return str(value)
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        return f"{doc_id}#{digest}"


@dataclass
class RewriteResult:
    """
    Query Rewriter の出力を表すデータクラス。
    元のクエリと書き換え後のクエリを保持する。
    """
    original_query: str
    rewrite_query: Optional[str] = None

    @property
    def combined_queries(self) -> List[str]:
        """検索に使用するクエリのリストを返す。書き換え成功時は両方、失敗時は元のクエリのみ。"""
        if self.rewrite_query and self.rewrite_query != self.original_query:
            return [self.original_query, self.rewrite_query]
        return [self.original_query]


@dataclass
class SourceSpan:
    """Extractive Compression における引用元の位置情報。"""
    doc_id: str
    chunk_id: str
    sentence_idx: int


@dataclass
class CompressionResult:
    """
    Extractive Compressor の出力を表すデータクラス。
    圧縮されたテキストと引用元の位置情報を保持する。
    """
    compressed_text: str
    source_spans: List[SourceSpan] = field(default_factory=list)


@dataclass
class RetrievalSearchResult:
    query: str
    used_queries: List[str]
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    selected_chunks: List[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0
    top_k: int = 0


@dataclass
class PreparedContext:
    context: str
    sources: List[Dict[str, Any]] = field(default_factory=list)

import asyncio
import logging
from abc import ABC, abstractmethod

from config.settings import get_settings
from domain.models.retrieval_models import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankerBase(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class PassthroughReranker(RerankerBase):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        ordered = sorted(chunks, key=lambda chunk: chunk.hybrid_score, reverse=True)
        for chunk in ordered:
            chunk.rerank_score = chunk.hybrid_score
        return ordered[:top_n]


class CohereReranker(RerankerBase):
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._passthrough = PassthroughReranker()
        self._client = None

        if not api_key:
            return

        try:
            import cohere
            self._client = cohere.Client(api_key=api_key)
        except Exception as exc:
            logger.warning("Cohere client を初期化できないため Passthrough にフォールバックします: %s", exc)
            self._client = None

    def _rerank_sync(self, query: str, chunks: list[RetrievedChunk], top_n: int):
        documents = [chunk.content for chunk in chunks]
        return self._client.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=documents,
            top_n=top_n,
        )

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if self._client is None:
            return await self._passthrough.rerank(query, chunks, top_n)

        settings = get_settings()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._rerank_sync, query, chunks, top_n),
                timeout=settings.rerank_timeout_seconds,
            )
            reranked: list[RetrievedChunk] = []
            for item in response.results:
                chunk = chunks[item.index]
                chunk.rerank_score = float(getattr(item, "relevance_score", chunk.hybrid_score))
                reranked.append(chunk)
            return reranked
        except Exception as exc:
            logger.warning("Rerank に失敗したため Passthrough にフォールバックします: %s", exc)
            return await self._passthrough.rerank(query, chunks, top_n)


def build_reranker() -> RerankerBase:
    settings = get_settings()
    if settings.enable_rerank:
        return CohereReranker(api_key=settings.cohere_api_key)
    return PassthroughReranker()

from domain.models.retrieval_models import RetrievedChunk
from config.settings import get_settings


class ResultMerger:
    @staticmethod
    def merge(sub_results: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        settings = get_settings()
        seen: dict[tuple[str, str], RetrievedChunk] = {}
        for chunks in sub_results:
            for chunk in chunks:
                key = (chunk.doc_id, chunk.chunk_id)
                if key not in seen or chunk.hybrid_score > seen[key].hybrid_score:
                    seen[key] = chunk

        merged = sorted(
            seen.values(),
            key=lambda chunk: chunk.hybrid_score,
            reverse=True,
        )
        return merged[:settings.max_merged_chunks]

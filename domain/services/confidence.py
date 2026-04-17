import logging
import os
from typing import List, Tuple
from domain.models.retrieval_models import RetrievedChunk
from domain.services.answer_critic import AnswerVerdict

logger = logging.getLogger(__name__)

# Dynamic TopK 設定
TOP_K_MIN = int(os.getenv("TOP_K_MIN", "3"))
TOP_K_MAX = int(os.getenv("TOP_K_MAX", "8"))


class ConfidenceEstimator:
    """
    Hybrid Search 結果のスコア分布から Confidence を定量化し、
    Dynamic TopK の件数を決定する。
    """

    @staticmethod
    def estimate(
        chunks: List[RetrievedChunk],
        score_key: str = "hybrid_score",
    ) -> Tuple[float, int]:
        """
        RetrievedChunk のリストから Confidence と Dynamic TopK を算出する。

        Returns:
            (confidence, top_k) のタプル
        """
        if not chunks:
            return 0.0, TOP_K_MAX

        ordered = sorted(chunks, key=lambda chunk: getattr(chunk, score_key, 0.0), reverse=True)
        scores = [getattr(chunk, score_key, 0.0) for chunk in ordered]
        top1 = scores[0]  # 既にソート済み前提
        top2 = scores[1] if len(scores) >= 2 else 0.0
        margin = top1 - top2

        # Confidence 算出: clamp(0.2 + 0.6*top1 + 0.2*margin, 0, 1)
        confidence = min(max(0.2 + 0.6 * top1 + 0.2 * margin, 0.0), 1.0)
        confidence = round(confidence, 4)

        # Dynamic TopK: Confidence に応じて文書数を可変化
        if top1 >= 0.85 and margin >= 0.05:
            top_k = TOP_K_MIN  # 高い確信度 → 少数で十分
        elif top1 >= 0.70:
            top_k = 5          # 中程度の確信度
        else:
            top_k = TOP_K_MAX  # 低い確信度 → 多めに取得

        logger.info(
            f"Confidence: {confidence:.4f} (top1={top1:.4f}, margin={margin:.4f}) -> top_k={top_k}"
        )

        return confidence, top_k


class AdvancedConfidenceEstimator:
    """
    Retrieval Critic と最終回答検証の結果を反映した Confidence を算出する。
    """

    @staticmethod
    def estimate(
        critic_coverage: float,
        top_chunks: List[RetrievedChunk],
        answer_verdict: AnswerVerdict | None = None,
    ) -> float:
        if not top_chunks:
            return 0.0

        score_key = "rerank_score" if any(chunk.rerank_score > 0 for chunk in top_chunks) else "hybrid_score"
        ordered = sorted(top_chunks, key=lambda chunk: getattr(chunk, score_key, 0.0), reverse=True)
        top1 = getattr(ordered[0], score_key, 0.0)
        top2 = getattr(ordered[1], score_key, 0.0) if len(ordered) >= 2 else 0.0
        margin = top1 - top2

        base = min(max(0.5 * critic_coverage + 0.3 * top1 + 0.2 * margin, 0.0), 1.0)

        if answer_verdict and answer_verdict.verdict == "FAIL":
            if answer_verdict.confidence_override is not None:
                base = min(base, answer_verdict.confidence_override)
            elif answer_verdict.hallucination_risk == "high":
                base = min(base, 0.3)
            elif answer_verdict.hallucination_risk == "medium":
                base = min(base, 0.5)

        return round(base, 4)

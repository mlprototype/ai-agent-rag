import asyncio
import logging
import os
import re
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from domain.models.retrieval_models import RetrievedChunk, CompressionResult, SourceSpan

logger = logging.getLogger(__name__)

# タイムアウト設定（ミリ秒→秒）
STAGE_TIMEOUT_COMPRESS = int(os.getenv("STAGE_TIMEOUT_MS_COMPRESS", "5000")) / 1000


class ExtractiveCompressor:
    """
    Extractive Compression: 検索結果のチャンクからクエリに関連する文のみを抽出し、
    LLM 入力を最小限にすることでノイズとコストを削減する。
    source_spans により Citation 追跡を維持する。
    """

    _COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "あなたはテキスト抽出の専門家です。\n"
         "以下の検索結果から、ユーザーの質問に回答するために必要な文のみを抽出してください。\n"
         "ルール:\n"
         "- 各文の前にある [docN-sM] タグをそのまま保持すること\n"
         "- 関連のない文は完全に除外すること\n"
         "- 文の内容は改変しないこと（原文のまま抽出）\n"
         "- 出力はタグ付きの抽出文のみ（説明や前置きは不要）"),
        ("human", "質問: {query}\n\n検索結果:\n{tagged_text}")
    ])

    _chain = None

    @classmethod
    def _get_chain(cls):
        """LLM チェーンの遅延初期化。"""
        if cls._chain is None:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=2000)
            cls._chain = cls._COMPRESS_PROMPT | llm
        return cls._chain

    @classmethod
    async def compress(cls, query: str, chunks: List[RetrievedChunk]) -> CompressionResult:
        """
        チャンクを文レベルに分割し、LLM でクエリとの関連度を判定して抽出する。
        タイムアウトまたはエラー時は圧縮をスキップし、元テキストをそのまま返す。
        """
        if not chunks:
            return CompressionResult(compressed_text="", source_spans=[])

        # 各チャンクを文に分割し、タグを付与
        tagged_lines = []
        span_map = {}  # tag -> SourceSpan
        for doc_idx, chunk in enumerate(chunks):
            sentences = cls._split_sentences(chunk.content)
            for sent_idx, sentence in enumerate(sentences):
                tag = f"[doc{doc_idx+1}-s{sent_idx+1}]"
                tagged_lines.append(f"{tag} {sentence}")
                span_map[tag] = SourceSpan(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    sentence_idx=sent_idx
                )

        tagged_text = "\n".join(tagged_lines)

        try:
            chain = cls._get_chain()
            response = await asyncio.wait_for(
                chain.ainvoke({"query": query, "tagged_text": tagged_text}),
                timeout=STAGE_TIMEOUT_COMPRESS
            )
            extracted = response.content.strip()

            # 抽出結果から source_spans を復元
            source_spans = []
            for tag, span in span_map.items():
                if tag in extracted:
                    source_spans.append(span)

            logger.info(
                f"Extractive compression: {len(tagged_lines)} 文 -> {len(source_spans)} 文に圧縮"
            )
            return CompressionResult(
                compressed_text=extracted,
                source_spans=source_spans
            )

        except asyncio.TimeoutError:
            logger.warning(f"Extractive compression がタイムアウトしました（{STAGE_TIMEOUT_COMPRESS}s）")
            return cls._fallback(chunks)
        except Exception as e:
            logger.warning(f"Extractive compression に失敗しました（フォールバック）: {e}")
            return cls._fallback(chunks)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """テキストを文単位に分割する。日本語の句点と英語のピリオドに対応。"""
        # 日本語句点（。）、英語ピリオド+空白、改行で分割
        sentences = re.split(r'(?<=[。.!?])\s*|\n+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _fallback(chunks: List[RetrievedChunk]) -> CompressionResult:
        """圧縮失敗時のフォールバック: 元テキストをそのまま結合して返す。"""
        parts = []
        source_spans = []
        for i, chunk in enumerate(chunks):
            parts.append(f"[{i+1}] {chunk.content}")
            source_spans.append(SourceSpan(
                doc_id=chunk.doc_id, chunk_id=chunk.chunk_id, sentence_idx=0
            ))
        return CompressionResult(
            compressed_text="\n\n".join(parts),
            source_spans=source_spans
        )

import asyncio
import logging
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from domain.models.retrieval_models import RewriteResult

logger = logging.getLogger(__name__)

# タイムアウト設定（ミリ秒→秒）
STAGE_TIMEOUT_REWRITE = int(os.getenv("STAGE_TIMEOUT_MS_REWRITE", "3000")) / 1000


class QueryRewriter:
    """
    ユーザーの自然言語クエリを、検索に適した簡潔なクエリへ書き換える。
    LLM（GPT-4o-mini）を使用して1回のrewriteを生成する。
    タイムアウトや失敗時はoriginal_queryのみで続行するフォールバック設計。
    """

    _REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "あなたは検索クエリの最適化を行う専門家です。\n"
         "ユーザーの質問を受け取り、ベクトル検索とキーワード検索の両方で高い精度を出せるよう、"
         "検索向けの短く明確なクエリに書き換えてください。\n"
         "ルール:\n"
         "- 出力は書き換え後のクエリ文字列のみ（説明や前置きは不要）\n"
         "- 元の意味を変えないこと\n"
         "- 日本語の質問には日本語で、英語の質問には英語で書き換えること"),
        ("human", "{query}")
    ])

    _chain = None

    @classmethod
    def _get_chain(cls):
        """LLM チェーンの遅延初期化。"""
        if cls._chain is None:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)
            cls._chain = cls._REWRITE_PROMPT | llm
        return cls._chain

    @classmethod
    async def rewrite(cls, query: str) -> RewriteResult:
        """
        クエリを書き換え、RewriteResult を返す。
        タイムアウトまたはエラー時は original_query のみの RewriteResult を返す。
        """
        try:
            chain = cls._get_chain()
            response = await asyncio.wait_for(
                chain.ainvoke({"query": query}),
                timeout=STAGE_TIMEOUT_REWRITE
            )
            rewritten = response.content.strip()
            logger.info(f"Query rewrite: '{query}' -> '{rewritten}'")
            return RewriteResult(original_query=query, rewrite_query=rewritten)

        except asyncio.TimeoutError:
            logger.warning(f"Query rewrite がタイムアウトしました（{STAGE_TIMEOUT_REWRITE}s）: '{query}'")
            return RewriteResult(original_query=query)
        except Exception as e:
            logger.warning(f"Query rewrite に失敗しました: {e}")
            return RewriteResult(original_query=query)

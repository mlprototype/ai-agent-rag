from langchain_core.tools import tool
from domain.services.retrieval_service import RetrievalService

@tool
async def retrieval_tool(query: str) -> str:
    """
    クエリに関連する情報をナレッジベースから検索します。
    RAG、LangGraph、pgvector、またはFastAPIに関する質問に答える必要がある場合にこのツールを使用してください。
    """
    # このツールは純粋なフレームワークアダプターとして機能します。
    # すべてのビジネスロジック（検索、タイムアウト、フォーマット、エラー処理）を
    # ドメインサービスに委譲します。
    return await RetrievalService.search_knowledge_base(query)

from langchain_core.tools import tool
from domain.services.retrieval_service import RetrievalService

@tool
async def retrieval_tool(query: str) -> str:
    """
    Search the knowledge base for information relevant to the query.
    Use this tool when you need to answer questions about RAG, LangGraph, pgvector, or FastAPI.
    """
    # The tool now acts purely as a Framework Adapter.
    # It delegates all business logic (search, timeout, formatting, error handling)
    # to the Domain Service.
    return await RetrievalService.search_knowledge_base(query)

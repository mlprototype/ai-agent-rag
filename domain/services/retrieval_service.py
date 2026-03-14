import asyncio
import logging
from infrastructure.retrieval.vector_store import get_async_vector_store

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Domain Service responsible for encapsulating business logic
    related to searching the knowledge base. Handles rules like timeouts,
    error messages, and formatting of search results.
    """
    
    RETRIEVAL_TIMEOUT_SECONDS = 5.0

    @classmethod
    async def search_knowledge_base(cls, query: str, top_k: int = 3) -> str:
        """
        Executes a vector search with timeout and business error handling.
        Returns a formatted string for the LLM to consume.
        """
        vector_store = get_async_vector_store()
        
        try:
            docs = await asyncio.wait_for(
                vector_store.asimilarity_search(query, k=top_k),
                timeout=cls.RETRIEVAL_TIMEOUT_SECONDS
            )
            
            if not docs:
                return "No relevant information found in the knowledge base."
                
            formatted_docs = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                formatted_docs.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}\n")
                
            return "\n".join(formatted_docs)
            
        except asyncio.TimeoutError:
            logger.warning(f"Retrieval timed out for query: '{query}'")
            return "【Error】The retrieval database timed out. Please answer the question without external retrieval context or ask the user to wait."
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return f"【Error】An unexpected error occurred during retrieval: {e}"

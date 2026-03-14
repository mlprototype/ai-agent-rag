import os
from langchain_postgres import PGVector
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.documents import Document
from infrastructure.retrieval.embedding import get_embeddings

def get_connection_string() -> str:
    """Read DB parameters from environment and return psycopg string."""
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "rag_db")
    
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"

def get_vector_store() -> PGVector:
    """Instantiate and return the PGVector store. Automatically creates tables if not exist."""
    connection_string = get_connection_string()
    embeddings = get_embeddings()

    # The collection name could be anything, let's configure a default
    collection_name = "agentic_rag_docs"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True, # Stores metadata as JSONB for better performance
    )
    return vector_store

def get_async_vector_store() -> PGVector:
    """Instantiate and return an async PGVector store using async psycopg engine."""
    connection_string = get_connection_string().replace("postgresql+psycopg", "postgresql+psycopg_async")
    
    embeddings = get_embeddings()
    collection_name = "agentic_rag_docs"
    
    engine = create_async_engine(connection_string, pool_size=5, max_overflow=10)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )
    return vector_store

def seed_database_if_empty():
    """Helper method to seed the database with initial documents for testing."""
    vector_store = get_vector_store()
    
    # Check if there are already documents
    # A simple way to check is to try a dummy search
    results = vector_store.similarity_search("test", k=1)
    if len(results) > 0:
        print("Database already contains documents. Skipping seed.")
        return

    print("Seeding database with sample documents...")
    sample_docs = [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) is a technique that enhances large language models by retrieving relevant information from an external knowledge base before generating a response.",
            metadata={"source": "AI Glossary", "topic": "RAG"}
        ),
        Document(
            page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. It lets you create cyclic graphs for agentic workflows.",
            metadata={"source": "LangChain Docs", "topic": "LangGraph"}
        ),
        Document(
            page_content="pgvector is an open-source vector similarity search for PostgreSQL. It supports exact and approximate nearest neighbor search.",
            metadata={"source": "pgvector README", "topic": "Database"}
        ),
        Document(
            page_content="FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.",
            metadata={"source": "FastAPI Docs", "topic": "Web Framework"}
        )
    ]
    
    vector_store.add_documents(sample_docs)
    print("Seeding complete.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    seed_database_if_empty()

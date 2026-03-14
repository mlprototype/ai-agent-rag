import os
from langchain_openai import OpenAIEmbeddings

def get_embeddings() -> OpenAIEmbeddings:
    """Returns the OpenAI embeddings model."""
    # We use text-embedding-3-small as the default modern embedding model
    return OpenAIEmbeddings(model="text-embedding-3-small")

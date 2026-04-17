import os
from langchain_openai import OpenAIEmbeddings

def get_embeddings() -> OpenAIEmbeddings:
    """OpenAIエンベディングモデルを返します。"""
    # デフォルトのモダンなエンベディングモデルとして text-embedding-3-small を使用します
    return OpenAIEmbeddings(model="text-embedding-3-small")

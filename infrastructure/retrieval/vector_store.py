import os
from langchain_postgres import PGVector
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.documents import Document
from infrastructure.retrieval.embedding import get_embeddings

def get_connection_string() -> str:
    """環境変数からDBパラメータを読み取り、psycopg形式の接続文字列を返します。"""
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "rag_db")
    
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"

def get_vector_store() -> PGVector:
    """PGVectorストアをインスタンス化して返します。テーブルが存在しない場合は自動的に作成されます。"""
    connection_string = get_connection_string()
    embeddings = get_embeddings()

    # コレクション名は任意ですが、ここではデフォルトを設定します
    collection_name = "agentic_rag_docs"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True, # パフォーマンス向上のため、メタデータをJSONBとして保存します
    )
    return vector_store

def get_async_vector_store() -> PGVector:
    """非同期psycopgエンジンを使用して、非同期PGVectorストアをインスタンス化して返します。"""
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
    """テスト用に初期ドキュメントをデータベースにシードするためのヘルパーメソッド。"""
    vector_store = get_vector_store()
    
    # 既にドキュメントがあるか確認
    # シンプルな確認方法として、ダミー検索を試みる
    results = vector_store.similarity_search("test", k=1)
    if len(results) > 0:
        print("データベースに既にドキュメントが存在します。シードをスキップします。")
        return

    print("サンプルドキュメントを使用してデータベースにシード中...")
    sample_docs = [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) は、回答を生成する前に外部ナレッジベースから関連情報を取得することで、大規模言語モデルを強化する手法です。",
            metadata={"source": "AI用語集", "topic": "RAG"}
        ),
        Document(
            page_content="LangGraph は、LLMを使用してステートフルなマルチアクターアプリケーションを構築するためのライブラリです。エージェンティックなワークフローのための循環グラフを作成できます。",
            metadata={"source": "LangChainドキュメント", "topic": "LangGraph"}
        ),
        Document(
            page_content="pgvector は PostgreSQL 用のオープンソースのベクトル類似性検索です。完全一致および近似最近傍探索をサポートしています。",
            metadata={"source": "pgvector README", "topic": "データベース"}
        ),
        Document(
            page_content="FastAPI は、標準の Python 型ヒントに基づいて Python 3.8+ で API を構築するための、モダンで高速な（高性能な）Web フレームワークです。",
            metadata={"source": "FastAPIドキュメント", "topic": "Webフレームワーク"}
        )
    ]
    
    vector_store.add_documents(sample_docs)
    print("シード完了。")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    seed_database_if_empty()

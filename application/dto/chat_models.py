from typing import List, Optional
from pydantic import BaseModel, Field

class Source(BaseModel):
    """
    ナレッジベースから取得されたソースドキュメントを表します。
    """
    doc_id: str = Field(description="ソースドキュメントの識別子（ファイル名等）。")
    chunk_id: str = Field(default="", description="チャンクの識別子。")
    snippet: str = Field(default="", description="取得されたテキストの抜粋。")
    score: float = Field(default=0.0, description="統合スコア（hybrid_score）。")
    hybrid_score: float = Field(default=0.0, description="Hybrid Search の加重平均スコア。")
    vector_score: float = Field(default=0.0, description="Vector Search の正規化スコア。")
    bm25_score: float = Field(default=0.0, description="Keyword Search (BM25) の正規化スコア。")
    rerank_score: float = Field(default=0.0, description="Reranker による再計算スコア。")

class ChatRequest(BaseModel):
    """
    チャットAPIの入力リクエストモデル。
    """
    session_id: str = Field(description="会話のセッションID。複数ターンの文脈保持に使用します。")
    question: str = Field(description="ユーザーからの質問またはメッセージ。")

class ChatResponse(BaseModel):
    """
    チャットAPIの出力レスポンスモデル。
    企業RAGの要件に基づき、answer + sources + confidence を返します。
    """
    answer: str = Field(description="エージェントによって生成された最終的な回答。")
    sources: List[Source] = Field(default_factory=list, description="回答を生成するために使用されたソースのリスト。")
    confidence: float = Field(default=0.0, description="回答の信頼度スコア（0.0〜1.0）。")
    warning: Optional[str] = Field(default=None, description="回答に対する注意メッセージ。")

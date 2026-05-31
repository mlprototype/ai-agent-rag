from langchain_experimental.text_splitter import SemanticChunker
from infrastructure.retrieval.embedding import get_embeddings
from langchain_core.documents import Document
from typing import List

class DocumentChunker:
    """
    ドキュメントのチャンキング（分割）を担当するクラス。
    テキストの意味的なまとまり（文脈）を重視して分割する Semantic Chunking を使用します。
    """
    
    def __init__(self):
        # 既存のEmbeddingモデル（OpenAI）を使用して文の類似度を計算するセマンティックチャンカーを初期化
        embeddings = get_embeddings()
        self.chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile", # コサイン類似度のパーセンタイルで分割点を決定
            breakpoint_threshold_amount=80 # 上位20%の変化がある場所で分割
        )

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        文字列を意味的なチャンクに分割し、LangChain の Document オブジェクトのリストとして返します。
        
        Args:
            text: 分割対象のクリーンなテキスト
            metadata: 各チャンクに付与するメタデータ（ファイル名、パスなど）
        """
        if not metadata:
            metadata = {}
            
        # テキストをDocumentオブジェクトとして作成してからchunkerに渡す
        doc = Document(page_content=text, metadata=metadata)
        
        # split_documentsで分割
        chunks = self.chunker.split_documents([doc])
        return chunks

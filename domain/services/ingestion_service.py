from infrastructure.ingestion.unstructured_loader import DocumentIngestionPipeline
from infrastructure.retrieval.chunking import DocumentChunker
from infrastructure.retrieval.vector_store import get_vector_store
import os

class IngestionService:
    """
    ドキュメント・インジェスチョン（取り込み）の全体フローをオーケストレーションするドメインサービス。
    指定された「loader -> parser -> cleaner -> chunker -> embedder -> vector_store」のパイプラインを管理します。
    """
    
    @staticmethod
    def ingest_file(file_path: str) -> None:
        """
        単一のファイルをパイプラインに通してベクトルDBに登録します。
        
        Args:
            file_path: 取り込むファイルのローカルパス
        """
        print(f"[IngestionPipeline] 開始: {file_path}")
        
        # 1. Loader: ファイルパスのロード（存在確認など）
        loaded_path = DocumentIngestionPipeline.load(file_path)
        print(f"  └ Loader: ロード成功")
        
        # 2. Parser: unstructuredを利用したフォーマット別のテキスト抽出
        parsed_elements = DocumentIngestionPipeline.parse(loaded_path)
        print(f"  └ Parser: 要素数 {len(parsed_elements)} を抽出")
        
        # 3. Cleaner: 抽出されたテキストのクリーニングと正規化
        cleaned_text = DocumentIngestionPipeline.clean(parsed_elements)
        print(f"  └ Cleaner: {len(cleaned_text)} 文字にクリーニング完了")
        
        # 4. Chunker: Semantic Chunking を用いた意味的分割
        # メタデータを付与
        metadata = {
            "source": os.path.basename(file_path),
            "source_path": file_path
        }
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(cleaned_text, metadata=metadata)
        print(f"  └ Chunker: {len(chunks)} 個の意味的チャンクに分割完了")
        
        # 5. Embedder & 6. Vector Store: ベクトル化およびDBへの一括登録（Upsert）
        # LangChainの VectorStore.add_documents は内部で Embedder を呼び出してベクトル化し、DBに保存します
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)
        print(f"  └ Embedder & Vector Store: DBへのチャンク保存成功")
        
        print(f"[IngestionPipeline] 完了: {file_path}")

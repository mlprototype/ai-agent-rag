"""
Agentic RAG アプリケーションのエントリーポイントとなるファイルです。
CLIを通じてユーザーの入力を受け付け、アプリケーション層の ChatService に処理を委譲し、結果を出力します。
入力(ユーザーの質問文字列)と出力(Agentの回答、確信度、情報源)の表示を担当しています。
ビジネスロジックはアプリケーション層に隠蔽しており、ここでは対話のセッションIDの発行とUIの表示のみを行います。
"""
from dotenv import load_dotenv
import os

# .envファイルから環境変数をロード
load_dotenv()

from application.services.chat_service import ChatService
from application.dto.chat_models import ChatRequest

import asyncio
import uuid

"""
メインループを実行する非同期関数。
標準入力からユーザーの質問を受け付け、ChatRequest を構築して ChatService に渡します。
返却された ChatResponse をもとに回答や検索のスコアを画面に出力します。
予期せぬエラーでCLIごとクラッシュするのを防ぐため、例外を吸収してエラー内容のみ出力し継続可能にしています。
"""
async def main():
    print("Welcome to the Agentic RAG Phase 3 Agent!")
    print("Make sure you have set OPENAI_API_KEY in your .env file.")
    print("Type 'exit' to quit.\n")
    
    session_id = str(uuid.uuid4())
    
    while True:
        try:
            user_input = input("User: ")
            
            # ユーザーが明示的に対話を終了したい場合に、無限ループを抜けてプロセスを終了させるために必要
            if user_input.lower() in ["exit", "quit"]:
                break
                
            request = ChatRequest(session_id=session_id, question=user_input)
            
            # 新しいアプリケーションサービスを非同期的に呼び出す
            response = await ChatService.ask_question(request)
            
            print(f"\nAgent: {response.answer}")
            print(f"  Confidence: {response.confidence}")
            
            # RAGパイプライン内で回答の品質や処理に何らかの懸念が生じた場合に、ユーザーへ注意喚起を行うために必要
            if response.warning:
                print(f"  Warning: {response.warning}")
                
            # RAGによる検索結果が利用されたかを確認し、回答の根拠となった情報源をユーザーに提示するために必要
            if response.sources:
                print("  Sources:")
                for source in response.sources:
                    print(
                        f"    - {source.doc_id} "
                        f"(hybrid: {source.hybrid_score:.4f}, vec: {source.vector_score:.4f}, "
                        f"bm25: {source.bm25_score:.4f}, rerank: {source.rerank_score:.4f})"
                    )

        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

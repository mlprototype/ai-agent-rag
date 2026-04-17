from dotenv import load_dotenv
import os

# .envファイルから環境変数をロード
load_dotenv()

from application.services.chat_service import ChatService
from application.dto.chat_models import ChatRequest

import asyncio
import uuid

async def main():
    print("Welcome to the Agentic RAG Phase 2.5 Agent!")
    print("Make sure you have set OPENAI_API_KEY in your .env file.")
    print("Type 'exit' to quit.\n")
    
    session_id = str(uuid.uuid4())
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            request = ChatRequest(session_id=session_id, question=user_input)
            
            # 新しいアプリケーションサービスを同期的に呼び出す
            response = await ChatService.ask_question(request)
            
            print(f"\nAgent: {response.answer}")
            print(f"  Confidence: {response.confidence}")
            if response.warning:
                print(f"  Warning: {response.warning}")
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

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from application.services.chat_service import ChatService
from application.dto.chat_models import ChatRequest

import asyncio

async def main():
    print("Welcome to the Agentic RAG Phase 1 Agent!")
    print("Make sure you have set OPENAI_API_KEY in your .env file.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            request = ChatRequest(question=user_input)
            
            # Using our new application service synchronously
            response = await ChatService.ask_question(request)
            
            print(f"Agent: {response.answer}")
            for source in response.sources:
                print(f" - Source: {source.id}")

        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

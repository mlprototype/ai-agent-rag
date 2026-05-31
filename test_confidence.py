import asyncio
import os
import sys

sys.path.append("/Users/apple/develop/ai-agent-rag")
import dotenv; dotenv.load_dotenv()
from application.agents.graph import graph
from langchain_core.messages import HumanMessage

async def main():
    inputs = {"messages": [HumanMessage(content="Hybrid Searchとはなんですか？")]}
    config = {"configurable": {"thread_id": "test-conf"}}
    async for event in graph.astream(inputs, config=config, stream_mode="values"):
        if "confidence" in event:
            print("Confidence:", event["confidence"])
        if "working_chunks" in event:
            print("All BM25 Zero:", all(c.get("bm25_score", 0.0) == 0.0 for c in event["working_chunks"]))

asyncio.run(main())

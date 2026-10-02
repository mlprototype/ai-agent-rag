"""検索 confidence の手動確認。実行には OpenAI API と PostgreSQL が必要。"""

import asyncio
from pathlib import Path
import sys

async def main():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    from application.agents.graph import graph
    from langchain_core.messages import HumanMessage

    inputs = {"messages": [HumanMessage(content="Hybrid Searchとはなんですか？")]}
    config = {"configurable": {"thread_id": "test-conf"}}
    async for event in graph.astream(inputs, config=config, stream_mode="values"):
        if "confidence" in event:
            print("Confidence:", event["confidence"])
        if "working_chunks" in event:
            print("All BM25 Zero:", all(c.get("bm25_score", 0.0) == 0.0 for c in event["working_chunks"]))

if __name__ == "__main__":
    asyncio.run(main())

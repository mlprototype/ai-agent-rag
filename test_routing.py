import asyncio
import os
import sys

# Add path to sys.path to allow importing domain
sys.path.append("/Users/apple/develop/ai-agent-rag")

from domain.services.router import AgentRouter

async def test():
    decision = await AgentRouter.route("LangGraphの仕組みを教えて")
    print(decision.dict())

asyncio.run(test())

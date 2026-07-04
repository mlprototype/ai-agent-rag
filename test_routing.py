import asyncio
from domain.services.router import AgentRouter


async def main():
    decision = await AgentRouter.route("LangGraphの仕組みを教えて")
    print(decision.model_dump())


if __name__ == "__main__":
    asyncio.run(main())

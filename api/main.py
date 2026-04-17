import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# APIキーのロード（モジュールインポート時にAPIキーが必要になるため、先にロードします）
load_dotenv()

from application.services.chat_service import ChatService
from application.dto.chat_models import ChatRequest, ChatResponse
from api.routers.ingest import router as ingest_router
from domain.services.prompt_loader import prewarm_prompts
from domain.services.prompt_registry import iter_prewarm_prompt_specs
from config.settings import get_settings
import uvicorn

_SETTINGS = get_settings()
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
_PROMPT_SPECS = iter_prewarm_prompt_specs()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(
        prewarm_prompts,
        _PROMPT_SPECS,
        fail_fast=_SETTINGS.prewarm_fail_fast,
    )
    yield


app = FastAPI(title="Agentic RAG API", version="3.0.0", lifespan=lifespan)

# インジェスチョン（ドキュメント取り込み）ルーターを登録
app.include_router(ingest_router, tags=["Ingestion"])

@app.get("/")
def read_root():
    return {"message": "Agentic RAG API is running."}

@app.post("/ask", response_model=ChatResponse)
async def ask_agent(request: ChatRequest):
    """
    LangGraph Agentic RAGと通信するエンドポイント。
    """
    response = await ChatService.ask_question(request)
    return response

@app.post("/ask/stream")
async def ask_agent_stream(request: ChatRequest):
    """
    LangGraph Agentic RAGと通信するストリーミングエンドポイント。
    LLMによって生成されたトークンを順次返します。
    """
    return StreamingResponse(ChatService.stream_question(request), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

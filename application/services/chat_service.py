import logging
import time
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

from application.agents.graph import graph
from application.dto.chat_models import ChatRequest, ChatResponse, Source
from config.settings import get_settings

_SETTINGS = get_settings()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChatService:
    """
    チャットのユースケース操作を処理するアプリケーションサービス層。
    外部インターフェース（API/CLI）とエージェントワークフロー（LangGraph）の架け橋となります。
    Citation（引用）情報の抽出とConfidence（信頼度）の取得もここで行います。
    """

    @staticmethod
    def _extract_sources(state) -> list[Source]:
        sources = []
        for src in state.get("sources", []):
            sources.append(Source(
                doc_id=src.get("doc_id", "不明"),
                chunk_id=src.get("chunk_id", ""),
                snippet=src.get("snippet", ""),
                score=src.get("hybrid_score", src.get("score", 0.0)),
                hybrid_score=src.get("hybrid_score", 0.0),
                vector_score=src.get("vector_score", 0.0),
                bm25_score=src.get("bm25_score", 0.0),
                rerank_score=src.get("rerank_score", 0.0),
            ))
        return sources

    @staticmethod
    async def ask_question(request: ChatRequest) -> ChatResponse:
        """
        エージェントからの完全な生成レスポンスを取得し、
        Citation（引用元）とConfidence（信頼度）を含む構造化レスポンスを返します。
        """
        inputs = {"messages": [HumanMessage(content=request.question)]}
        config = {"configurable": {"thread_id": request.session_id}}
        final_state = {}
        started_at = time.monotonic()

        async for event in graph.astream(inputs, config=config, stream_mode="values"):
            final_state = event

        query_type = final_state.get("query_type")
        route = final_state.get("route")
        
        total_latency_ms = int((time.monotonic() - started_at) * 1000)
        timeout_stages = list(final_state.get("timeout_stages", []))
        fallback_stages = list(final_state.get("fallback_stages", []))
        critique_reason = final_state.get("critique_reason", "")
        critic_degraded = bool(
            final_state.get("retrieval_critic_skipped_reason")
            or final_state.get("answer_critic_skipped_reason")
            or critique_reason.startswith("critic_fallback:")
            or final_state.get("retrieval_degraded")
        )

        logger.info(
            {
                "event": "chat_request_summary",
                "session_id": request.session_id,
                "query_type": query_type,
                "route": route,
                "total_latency_ms": total_latency_ms,
                "router_timeout": "router" in timeout_stages,
                "decompose_timeout": any(stage in {"decompose", "rewrite"} for stage in timeout_stages),
                "critic_degraded": critic_degraded,
                "final_confidence": round(final_state.get("confidence", 0.5), 2),
                "timeout_stage": timeout_stages,
                "fallback_used": bool(fallback_stages),
                "fallback_level": final_state.get("fallback_level", "full_path"),
                "partial_retrieval_used": final_state.get("partial_retrieval_used", False),
                "retrieval_timeout_count": final_state.get("retrieval_timeout_count", 0),
                "retrieval_success_count": final_state.get("retrieval_success_count", 0),
                "warning_codes": final_state.get("warning_codes", []),
                "retrieval_quality_level": final_state.get("retrieval_quality_level", "high"),
                "remaining_budget_ms_at_generate": final_state.get("remaining_budget_ms_at_generate", 0),
                "skipped_stages": final_state.get("skipped_stages", []),
            }
        )

        if route in ("calculator", "direct_answer"):
            sources = None
            confidence = 1.0 if route == "calculator" else None
            warning = None
        else:
            sources = ChatService._extract_sources(final_state)
            confidence = round(final_state.get("confidence", 0.5), 2)
            warning = final_state.get("warning")

        return ChatResponse(
            answer=final_state.get("answer", ""),
            query_type=query_type,
            route=route,
            sources=sources,
            confidence=confidence,
            warning=warning,
        )

    @staticmethod
    async def stream_question(request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        エージェントワークフロー内のLLMによって生成されたテキストトークンのみをストリーミングします。
        """
        if _SETTINGS.answer_critic_retry:
            response = await ChatService.ask_question(request)
            for index in range(0, len(response.answer), 24):
                yield response.answer[index:index + 24]
            return

        inputs = {"messages": [HumanMessage(content=request.question)]}
        config = {"configurable": {"thread_id": request.session_id}}

        async for event in graph.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node")
            if kind == "on_chat_model_stream" and node == "generate":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield chunk

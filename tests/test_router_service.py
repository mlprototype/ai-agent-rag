import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from domain.services.router import AgentRouter, LLMRouteDecision
from config.settings import get_settings

class TestRouterService(unittest.IsolatedAsyncioTestCase):

    @patch("domain.services.router.HeuristicRouter.route")
    async def test_heuristic_hit_no_llm_call(self, mock_hr_route):
        # Mock heuristic to return a hit
        mock_decision = MagicMock()
        mock_decision.query_type = "calc"
        mock_decision.route = "calculator"
        mock_decision.confidence = 1.0
        mock_hr_route.return_value = mock_decision

        with patch.object(AgentRouter, '_get_chain') as mock_get_chain:
            decision = await AgentRouter.route("1+1")
            
            # Heuristic should be called
            mock_hr_route.assert_called_once()
            
            # LLM should NOT be called
            mock_get_chain.assert_not_called()
            self.assertEqual(decision, mock_decision)

    @patch("domain.services.router.HeuristicRouter.route", return_value=None)
    async def test_heuristic_miss_calls_llm(self, mock_hr_route):
        # Mock LLM chain
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = LLMRouteDecision(route="agentic_retrieval", reason="test")
        
        with patch.object(AgentRouter, '_get_chain', return_value=mock_chain):
            decision = await AgentRouter.route("複雑な質問")
            
            mock_chain.ainvoke.assert_called_once_with({"query": "複雑な質問"})
            self.assertEqual(decision.routing_layer, "llm")
            self.assertEqual(decision.source, "llm_success")
            self.assertTrue(decision.llm_router_invoked)
            self.assertEqual(decision.route, "agentic_retrieval")

    @patch("domain.services.router.HeuristicRouter.route", return_value=None)
    async def test_llm_timeout_fallback(self, mock_hr_route):
        # Mock LLM chain to timeout
        mock_chain = AsyncMock()
        mock_chain.ainvoke.side_effect = asyncio.TimeoutError("timeout")
        
        with patch.object(AgentRouter, '_get_chain', return_value=mock_chain):
            decision = await AgentRouter.route("タイムアウトする質問")
            
            self.assertEqual(decision.routing_layer, "fallback")
            self.assertEqual(decision.source, "llm_timeout_fallback")
            self.assertEqual(decision.route, "fallback_retrieval")
            self.assertEqual(decision.reason, "timeout_fallback")

if __name__ == "__main__":
    unittest.main()

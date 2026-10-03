import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evaluation import evaluate
from evaluation.reporter import build_view_context
from evaluation.schema import EvalReport

agent = importlib.import_module("application.agents.graph")


async def evaluate_state(tmp_path, state):
    (tmp_path / "dataset.json").write_text(json.dumps([{"question": "Q", "expected_answer": "A"}]), encoding="utf-8")
    async def stream(*args, **kwargs):
        yield state
    with patch.object(evaluate, "__file__", str(tmp_path / "evaluate.py")), patch.object(evaluate, "graph", SimpleNamespace(astream=stream)), patch.object(evaluate, "assess_answer_similarity", return_value=0.9):
        await evaluate.run_evaluation()
    return EvalReport.model_validate_json(next((tmp_path / "results").glob("*.json")).read_text())


def base_state(**updates):
    return {
        "answer": "A", "answer_ok": True, "route": "agentic_retrieval",
        "query_type": "retrieval_complex", "confidence": 0.7,
        "must_generate": False, "warning_codes": [], "warning": None,
        "retrieval_quality_level": "high", **updates,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("updates,critic,degraded,warning,reason", [
    ({"answer_critic_skipped_reason": "high_confidence"}, True, True, False, "SUCCESS"),
    ({"retrieval_critic_skipped_reason": "remaining_budget_low"}, True, True, False, "SUCCESS"),
    ({"critique_reason": "critic_fallback:timeout", "answer_ok": False}, True, True, False, "CRITIC_FAIL"),
    ({"skipped_stages": ["retrieval_critic"]}, True, True, False, "SUCCESS"),
    ({"fallback_stages": ["answer_critic"]}, True, True, False, "SUCCESS"),
    ({"timeout_stages": ["answer_critic"]}, True, True, True, "SUCCESS"),
    ({"must_generate": True}, False, False, False, "SUCCESS"),
    ({"warning": "user-facing warning"}, False, False, True, "SUCCESS"),
    ({"answer_ok": False, "warning": "incomplete"}, False, False, True, "QUALITY_FAIL"),
    ({"strict_insufficient_response": True, "answer_ok": False, "confidence": 0.25}, False, True, True, "NO_DATA"),
    ({"answer_ok": False, "warning_codes": ["low_confidence_definition_guard"]}, False, False, True, "GUARD_BLOCK"),
    ({"fallback_level": "minimal_answer", "retrieval_degraded": True}, False, True, False, "SUCCESS"),
    ({"fallback_level": "single_retrieval_fallback"}, False, True, False, "SUCCESS"),
    ({"partial_retrieval_used": True}, False, True, False, "SUCCESS"),
])
async def test_evaluation_keeps_runtime_semantics(tmp_path, updates, critic, degraded, warning, reason):
    report = await evaluate_state(tmp_path, base_state(**updates))
    record = report.records[0]
    assert record.critic_degraded is critic
    assert record.degraded is degraded
    assert record.warning is warning
    assert record.reason_code == reason
    assert set(updates.get("warning_codes", [])).issubset(record.warning_codes)
    if updates.get("timeout_stages"):
        assert "TIMEOUT_ANSWER_CRITIC" in record.warning_codes
    if updates.get("fallback_level"):
        assert record.fallback_level == updates["fallback_level"]
    if updates.get("strict_insufficient_response"):
        assert record.retrieval_quality_level == "low"


@pytest.mark.anyio
@pytest.mark.parametrize("query,reason", [
    ("売上データを削除して", "write_operation_blocked"),
    ("売上と在庫を比較して", "join_like_query_blocked"),
    ("売上の割合", "unknown_operation"),
    ("2026年の売上合計", "unknown_field"),
])
async def test_structured_fail_safe_reason_reaches_evaluation_and_reporter(tmp_path, query, reason):
    state = base_state(original_query=query, route="structured_query_tool", query_type="structured_query")
    updates = await agent.structured_query_node(state)
    assert updates["structured_query_reason_code"] == reason
    state.update(updates)
    report = await evaluate_state(tmp_path, state)
    record = report.records[0]
    assert record.reason_code == reason
    assert not record.answer_ok
    assert record.warning
    assert record.warning_codes == []
    assert not record.critic_degraded
    view = build_view_context(None, report, "test", 5)
    assert view["sq_analysis"]["fail_safe_rate"] == 1.0

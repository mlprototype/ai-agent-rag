import time
from unittest.mock import patch
import pytest

from application.agents.state import AgentState
from config.settings import get_settings
from domain.services.retrieval_budget import (
    compute_remaining_budget_ms,
    evaluate_budget_and_fallback,
    should_skip_rerank,
    should_skip_retrieval_critic,
    should_skip_rewrite,
)

_SETTINGS = get_settings()

def create_base_state(remaining_ms: int) -> AgentState:
    initial = _SETTINGS.budget_total_retrieval_complex_ms
    elapsed_ms = initial - remaining_ms
    
    state = AgentState(
        budget_started_at=time.monotonic() - (elapsed_ms / 1000.0),
        initial_budget_ms=initial,
        fallback_level="full_path",
        must_generate=False,
        skipped_stages=[],
        budget_pressure_reasons=[],
    )
    return state

def test_full_path_when_budget_sufficient():
    state = create_base_state(_SETTINGS.budget_total_retrieval_complex_ms)
    
    assert not should_skip_rerank(state)
    assert not should_skip_retrieval_critic(state)
    assert not should_skip_rewrite(state)
    
    updates = evaluate_budget_and_fallback(state, "test_checkpoint")
    assert updates.get("must_generate", False) is False
    assert "fallback_level" not in updates

def test_optimization_skip_when_budget_low_for_rerank():
    reserved = _SETTINGS.budget_reserved_generate_ms + _SETTINGS.budget_reserved_commit_ms
    remaining = reserved + _SETTINGS.budget_min_for_rerank_ms - 100
    
    state = create_base_state(remaining)
    
    assert should_skip_rerank(state) is True
    
    updates = evaluate_budget_and_fallback(state, "test_checkpoint")
    assert updates.get("must_generate", False) is False

def test_critic_skip_when_usable_budget_exhausted():
    reserved = _SETTINGS.budget_reserved_generate_ms + _SETTINGS.budget_reserved_commit_ms
    remaining = reserved - 100
    
    state = create_base_state(remaining)
    
    assert should_skip_rerank(state) is True
    assert should_skip_retrieval_critic(state) is True
    assert should_skip_rewrite(state) is True
    
    updates = evaluate_budget_and_fallback(state, "test_checkpoint")
    assert updates.get("must_generate") is True
    assert updates.get("fallback_level") == "critic_skip"
    assert "usable_budget_exhausted_at_test_checkpoint" in updates.get("budget_pressure_reasons", [])

def test_must_generate_forces_skip_optional_stages():
    state = create_base_state(_SETTINGS.budget_total_retrieval_complex_ms)
    state["must_generate"] = True
    
    assert should_skip_rerank(state) is True
    assert should_skip_retrieval_critic(state) is True
    assert should_skip_rewrite(state) is True

def test_reserved_budget_preserved():
    reserved = _SETTINGS.budget_reserved_generate_ms + _SETTINGS.budget_reserved_commit_ms
    state = create_base_state(reserved + 10)
    
    usable = compute_remaining_budget_ms(state) - reserved
    # tolerance for timing
    assert abs(usable - 10) < 50
    
    assert should_skip_rerank(state) is True

import time
from typing import Any
from application.agents.state import AgentState
from config.settings import get_settings

_SETTINGS = get_settings()

def compute_remaining_budget_ms(state: AgentState) -> int:
    started_at = state.get("budget_started_at")
    initial_budget_ms = int(state.get("initial_budget_ms", 0) or 0)
    if started_at is None or initial_budget_ms <= 0:
        return 0
    elapsed_ms = int((time.monotonic() - float(started_at)) * 1000)
    return max(0, initial_budget_ms - elapsed_ms)

# 関数の役割: チェックポイントでの残予算評価とフォールバックレベルの引き上げ
# 入出力: AgentStateとチェックポイント名を受け取り、更新差分を返す
# state更新: remaining_budget_ms, must_generate, fallback_level などを更新
def evaluate_budget_and_fallback(state: AgentState, checkpoint: str) -> dict[str, Any]:
    remaining = compute_remaining_budget_ms(state)
    usable_for_optional = remaining - _SETTINGS.budget_reserved_generate_ms - _SETTINGS.budget_reserved_commit_ms
    
    updates: dict[str, Any] = {
        "remaining_budget_ms": remaining,
    }
    
    current_level = state.get("fallback_level", "full_path")
    must_generate = state.get("must_generate", False)
    budget_reasons = state.get("budget_pressure_reasons", [])
    
    if usable_for_optional <= 0 and not must_generate:
        updates["must_generate"] = True
        must_generate = True
        
    # 回答生成のための必須時間を差し引いた利用可能時間が尽きた場合、品質向上処理をスキップするレベルへ引き上げる
    if usable_for_optional <= 0 and current_level in ["full_path", "optimization_skip"]:
        updates["fallback_level"] = "critic_skip"
        if f"usable_budget_exhausted_at_{checkpoint}" not in budget_reasons:
            updates["budget_pressure_reasons"] = budget_reasons + [f"usable_budget_exhausted_at_{checkpoint}"]
            
    return updates

def should_skip_rerank(state: AgentState) -> bool:
    if state.get("must_generate") or state.get("fallback_level") in ["optimization_skip", "critic_skip", "single_retrieval_fallback", "minimal_answer"]:
        return True
    remaining = compute_remaining_budget_ms(state)
    usable = remaining - _SETTINGS.budget_reserved_generate_ms - _SETTINGS.budget_reserved_commit_ms
    return usable < _SETTINGS.budget_min_for_rerank_ms

def should_skip_retrieval_critic(state: AgentState) -> bool:
    if state.get("must_generate") or state.get("fallback_level") in ["critic_skip", "single_retrieval_fallback", "minimal_answer"]:
        return True
    remaining = compute_remaining_budget_ms(state)
    usable = remaining - _SETTINGS.budget_reserved_generate_ms - _SETTINGS.budget_reserved_commit_ms
    return usable < _SETTINGS.budget_min_for_critic_ms

def should_skip_rewrite(state: AgentState) -> bool:
    if state.get("must_generate") or state.get("fallback_level") in ["critic_skip", "single_retrieval_fallback", "minimal_answer"]:
        return True
    remaining = compute_remaining_budget_ms(state)
    usable = remaining - _SETTINGS.budget_reserved_generate_ms - _SETTINGS.budget_reserved_commit_ms
    return usable < _SETTINGS.budget_min_for_rewrite_ms

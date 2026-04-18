import os
from dataclasses import dataclass
from functools import lru_cache


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, *aliases: str) -> int:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value is not None:
            return int(value)
    return default


@dataclass(frozen=True)
class Settings:
    enable_agentic: bool
    enable_rerank: bool
    router_heuristic_enabled: bool
    router_heuristic_compare_enabled: bool
    router_heuristic_confidence_threshold: float
    answer_critic_enabled: bool
    answer_critic_retry: bool
    max_retrieval_retry: int
    max_sub_queries: int
    max_merged_chunks: int
    router_timeout_seconds: float
    router_budget_ms: int
    router_uncertain_confidence_cap: float
    prompt_load_timeout_seconds: float
    prompt_failure_ttl_seconds: float
    prewarm_fail_fast: bool
    retrieval_critic_timeout_seconds: float
    answer_critic_timeout_seconds: float
    decompose_timeout_seconds: float
    rewrite_timeout_seconds: float
    rerank_timeout_seconds: float
    complex_budget_low_ms: int
    complex_budget_medium_ms: int
    complex_budget_high_ms: int
    retrieval_degrade_threshold_ms: int
    force_generate_threshold_ms: int
    retrieval_critic_skip_confidence: float
    answer_critic_skip_confidence: float
    cohere_api_key: str
    prompt_namespace: str
    budget_total_retrieval_complex_ms: int
    budget_reserved_generate_ms: int
    budget_reserved_commit_ms: int
    budget_min_for_rerank_ms: int
    budget_min_for_critic_ms: int
    budget_min_for_rewrite_ms: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        enable_agentic=_get_bool("ENABLE_AGENTIC", True),
        enable_rerank=_get_bool("ENABLE_RERANK", False),
        router_heuristic_enabled=_get_bool("ROUTER_HEURISTIC_ENABLED", True),
        router_heuristic_compare_enabled=_get_bool("ROUTER_HEURISTIC_COMPARE_ENABLED", True),
        router_heuristic_confidence_threshold=_get_int("ROUTER_HEURISTIC_CONFIDENCE_THRESHOLD_PCT", 85) / 100,
        answer_critic_enabled=_get_bool("ANSWER_CRITIC", True),
        answer_critic_retry=_get_bool("ANSWER_CRITIC_RETRY", False),
        max_retrieval_retry=_get_int("MAX_RETRY", 3, "MAX_RETRIEVAL_RETRY"),
        max_sub_queries=_get_int("MAX_SUB_QUERIES", 4),
        max_merged_chunks=_get_int("MAX_MERGED_CHUNKS", 20),
        router_timeout_seconds=_get_int("STAGE_TIMEOUT_MS_ROUTER", 1800) / 1000,
        router_budget_ms=_get_int("ROUTER_BUDGET_MS", 1500),
        router_uncertain_confidence_cap=_get_int("ROUTER_UNCERTAIN_CONFIDENCE_CAP_PCT", 80) / 100,
        prompt_load_timeout_seconds=_get_int("PROMPT_LOAD_TIMEOUT_MS", 1200) / 1000,
        prompt_failure_ttl_seconds=_get_int("PROMPT_FAILURE_TTL_MS", 30000) / 1000,
        prewarm_fail_fast=_get_bool("PREWARM_FAIL_FAST", True),
        retrieval_critic_timeout_seconds=_get_int("STAGE_TIMEOUT_MS_RETRIEVAL_CRITIC", 2500) / 1000,
        answer_critic_timeout_seconds=_get_int("STAGE_TIMEOUT_MS_ANSWER_CRITIC", 2500) / 1000,
        decompose_timeout_seconds=_get_int("STAGE_TIMEOUT_MS_DECOMPOSE", 1800) / 1000,
        rewrite_timeout_seconds=_get_int("STAGE_TIMEOUT_MS_REWRITE_SUBQUERY", 1800) / 1000,
        rerank_timeout_seconds=_get_int("STAGE_TIMEOUT_MS_RERANK", 2500) / 1000,
        complex_budget_low_ms=_get_int("COMPLEX_BUDGET_MS_LOW", 4000),
        complex_budget_medium_ms=_get_int("COMPLEX_BUDGET_MS_MEDIUM", 7000),
        complex_budget_high_ms=_get_int("COMPLEX_BUDGET_MS_HIGH", 9000),
        retrieval_degrade_threshold_ms=_get_int("RETRIEVAL_DEGRADE_THRESHOLD_MS", 2000),
        force_generate_threshold_ms=_get_int("FORCE_GENERATE_THRESHOLD_MS", 1500),
        retrieval_critic_skip_confidence=_get_int("RETRIEVAL_CRITIC_SKIP_CONFIDENCE_PCT", 85) / 100,
        answer_critic_skip_confidence=_get_int("ANSWER_CRITIC_SKIP_CONFIDENCE_PCT", 80) / 100,
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        prompt_namespace=os.getenv("PROMPT_NAMESPACE", "my-rag"),
        budget_total_retrieval_complex_ms=_get_int("BUDGET_TOTAL_RETRIEVAL_COMPLEX_MS", 15000),
        budget_reserved_generate_ms=_get_int("BUDGET_RESERVED_GENERATE_MS", 3000),
        budget_reserved_commit_ms=_get_int("BUDGET_RESERVED_COMMIT_MS", 500),
        budget_min_for_rerank_ms=_get_int("BUDGET_MIN_FOR_RERANK_MS", 1000),
        budget_min_for_critic_ms=_get_int("BUDGET_MIN_FOR_CRITIC_MS", 1500),
        budget_min_for_rewrite_ms=_get_int("BUDGET_MIN_FOR_REWRITE_MS", 3000),
    )

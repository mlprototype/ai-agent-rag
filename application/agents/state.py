import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    session_id: str
    original_query: str
    route: Literal["direct_answer", "calculator", "agentic_retrieval", "fallback_retrieval"]
    router_reason: str
    router_uncertain: bool
    query_type: Literal["direct", "calc", "compare", "definition", "retrieval_complex"]
    routing_layer: Literal["heuristic", "llm", "fallback"]
    route_decision_source: Literal["heuristic_match", "llm_success", "llm_timeout_fallback", "llm_error_fallback"]
    heuristic_matched: bool
    heuristic_rule: str
    route_decision_latency_ms: int
    route_decision_confidence: float
    llm_router_invoked: bool
    query_complexity: Literal["low", "medium", "high"]
    coverage_intent: str
    coverage_entities: list[str]
    comparison_axes: list[str]
    expected_aspects: list[str]
    query_phase: Literal["initial", "decomposed"]
    sub_queries: list[str]
    retrieval_ok: bool
    answer_ok: bool
    retry_count: int
    confidence: float
    warning: str | None
    sources: list[dict[str, Any]]
    retrieval_context: str
    working_chunks: list[dict[str, Any]]
    parallel_results: list[list[dict[str, Any]]]
    critique_reason: str
    missing_aspects: list[str]
    coverage_score: float
    answer: str
    calculator_expression: str
    calculator_result: str
    force_generate: bool
    must_generate: bool
    retrieval_degraded: bool
    confidence_cap: float | None
    budget_started_at: float
    initial_budget_ms: int
    remaining_budget_ms: int
    retrieval_critic_skipped_reason: str | None
    answer_critic_skipped_reason: str | None
    timeout_stages: list[str]
    fallback_stages: list[str]
    # Compare
    compare_targets: dict[str, Any] | None
    compare_aspect: str | None
    compare_extract_success: bool
    compare_path_used: bool
    compare_doc_count_a: int
    compare_doc_count_b: int
    compare_context_coverage_ok: bool
    compare_route_fallback_used: bool
    compare_fallback_reason: str | None
    quality_gate_status: Literal["pass", "warning", "fail"] | None
    quality_gate_reasons: list[str] | None
    quality_gate_confidence: float | None

    # Sprint 3: Budget and Fallback
    fallback_level: Literal["full_path", "optimization_skip", "critic_skip", "single_retrieval_fallback", "minimal_answer"]
    skipped_stages: list[str]
    budget_pressure_reasons: list[str]
    remaining_budget_ms_at_generate: int
    partial_retrieval_used: bool
    retrieval_timeout_count: int
    retrieval_success_count: int
    warning_codes: list[str]
    retrieval_quality_level: str

"""spec-rag-qa と受け渡す評価専用 AgentRunTrace DTO。"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class AgentTraceModel(BaseModel):
    """未知フィールドを拒否し、拡張は metadata に限定する基底モデル。"""

    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunInput(AgentTraceModel):
    question: NonEmptyStr


class AgentRunOutput(AgentTraceModel):
    answer: str
    query_type: NonEmptyStr
    route: NonEmptyStr
    confidence: Confidence
    warning_codes: list[NonEmptyStr] = Field(default_factory=list)


class ToolCallTrace(AgentTraceModel):
    name: NonEmptyStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: NonEmptyStr | None = None
    duration_ms: NonNegativeFloat | None = None


class CitationTrace(AgentTraceModel):
    citation_id: NonEmptyStr
    source_id: NonEmptyStr


class SourceTrace(AgentTraceModel):
    source_id: NonEmptyStr
    title: NonEmptyStr | None = None
    uri: NonEmptyStr | None = None
    snippet: str | None = None
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None


class GuardrailTrace(AgentTraceModel):
    triggered: bool = False
    blocked: bool = False
    codes: list[NonEmptyStr] = Field(default_factory=list)
    messages: list[NonEmptyStr] = Field(default_factory=list)


class ControlTrace(AgentTraceModel):
    attempt_count: Annotated[int, Field(ge=1)] = 1
    retry_count: NonNegativeInt = 0
    fallback_used: bool = False
    fallback_stages: list[NonEmptyStr] = Field(default_factory=list)
    timeout_stages: list[NonEmptyStr] = Field(default_factory=list)
    skipped_stages: list[NonEmptyStr] = Field(default_factory=list)
    fallback_level: NonEmptyStr | int | None = None
    remaining_budget_ms_at_generate: NonNegativeFloat | None = None
    stop_reason: NonEmptyStr | None = None


class UsageTrace(AgentTraceModel):
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    total_tokens: NonNegativeInt | None = None
    cost_usd: NonNegativeFloat | None = None


class TimingTrace(AgentTraceModel):
    latency_ms: NonNegativeFloat
    tool_latency_ms: NonNegativeFloat | None = None
    route_decision_latency_ms: NonNegativeFloat | None = None


class AgentRunTrace(AgentTraceModel):
    """1回のAgent実行で観測した事実。評価結果は保持しない。"""

    schema_version: NonEmptyStr
    run_id: NonEmptyStr
    case_id: NonEmptyStr
    target: NonEmptyStr
    input: AgentRunInput
    output: AgentRunOutput
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    citations: list[CitationTrace] = Field(default_factory=list)
    sources: list[SourceTrace] = Field(default_factory=list)
    guardrail: GuardrailTrace
    control: ControlTrace
    usage: UsageTrace
    timing: TimingTrace

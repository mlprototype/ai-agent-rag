from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    key: str
    hub_name: str | None
    local_path: str
    has_fallback: bool
    critical: bool
    used_by: tuple[str, ...]
    prewarm: bool = True

    @property
    def prompt_name(self) -> str:
        return self.hub_name or self.key


ROUTER_PROMPT = "agentic-rag-router"
DECOMPOSE_PROMPT = "agentic-rag-decompose"
REWRITE_PROMPT = "agentic-rag-rewrite"
RETRIEVAL_CRITIC_PROMPT = "agentic-rag-retrieval-critic"
ANSWER_CRITIC_PROMPT = "agentic-rag-answer-critic"
GENERATE_PROMPT = "agentic-rag-generate"

PROMPT_REGISTRY: dict[str, PromptSpec] = {
    "router": PromptSpec(
        key="router",
        hub_name=ROUTER_PROMPT,
        local_path="prompts/router/v1.yaml",
        has_fallback=True,
        critical=True,
        used_by=("router",),
        prewarm=True,
    ),
    "decompose": PromptSpec(
        key="decompose",
        hub_name=DECOMPOSE_PROMPT,
        local_path="prompts/decompose/v1.yaml",
        has_fallback=True,
        critical=True,
        used_by=("decompose",),
        prewarm=True,
    ),
    "rewrite": PromptSpec(
        key="rewrite",
        hub_name=REWRITE_PROMPT,
        local_path="prompts/rewrite/v1.yaml",
        has_fallback=True,
        critical=True,
        used_by=("rewrite",),
        prewarm=True,
    ),
    "retrieval_critic": PromptSpec(
        key="retrieval_critic",
        hub_name=RETRIEVAL_CRITIC_PROMPT,
        local_path="prompts/retrieval_critic/v1.yaml",
        has_fallback=True,
        critical=True,
        used_by=("critic", "retrieval"),
        prewarm=True,
    ),
    "answer_critic": PromptSpec(
        key="answer_critic",
        hub_name=ANSWER_CRITIC_PROMPT,
        local_path="prompts/answer_critic/v1.yaml",
        has_fallback=True,
        critical=True,
        used_by=("critic", "answer"),
        prewarm=True,
    ),
    "generate": PromptSpec(
        key="generate",
        hub_name=GENERATE_PROMPT,
        local_path="prompts/generate/v1.yaml",
        has_fallback=True,
        critical=True,
        used_by=("generate",),
        prewarm=True,
    ),
    "compare_generate": PromptSpec(
        key="compare_generate",
        hub_name=None,
        local_path="prompts/compare_generate.yaml",
        has_fallback=True,
        critical=False,
        used_by=("compare_generate_node",),
        prewarm=True,
    ),
}

PROMPT_NAME_REGISTRY: dict[str, PromptSpec] = {
    spec.prompt_name: spec for spec in PROMPT_REGISTRY.values() if spec.hub_name
}


def get_prompt_spec(key_or_name: str) -> PromptSpec:
    if key_or_name in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[key_or_name]
    if key_or_name in PROMPT_NAME_REGISTRY:
        return PROMPT_NAME_REGISTRY[key_or_name]
    raise KeyError(f"Unknown prompt spec: {key_or_name}")


def iter_prewarm_prompt_specs() -> tuple[PromptSpec, ...]:
    return tuple(spec for spec in PROMPT_REGISTRY.values() if spec.prewarm)


def iter_critical_prompt_specs() -> tuple[PromptSpec, ...]:
    return tuple(spec for spec in PROMPT_REGISTRY.values() if spec.critical)

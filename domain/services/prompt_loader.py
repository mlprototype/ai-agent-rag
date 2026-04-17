import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from config.settings import get_settings
from domain.services.prompt_formats import (
    build_prompt_from_document,
    classify_prompt_error,
    load_prompt_document_from_path,
)
from domain.services.prompt_registry import PromptSpec, get_prompt_spec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PromptResolution:
    prompt_key: str
    prompt: Any | None
    resolution_source: str
    used_fallback: bool
    failure_reason: str | None
    prewarm_phase: str
    cache_hit_type: str
    latency_ms: int
    candidate_reasons: dict[str, str]
    success: bool
    has_fallback: bool
    prompt_version: str | None
    used_by: tuple[str, ...]
    critical: bool = False


@dataclass(frozen=True)
class RepositoryLoadResult:
    prompt: Any | None
    source: str
    failure_reason: str | None = None
    cache_hit_type: str = "none"
    candidate_reasons: dict[str, str] = field(default_factory=dict)
    prompt_version: str | None = None


class PromptRepository(Protocol):
    def load(self, spec: PromptSpec) -> RepositoryLoadResult:
        ...


def log_prompt_fallback(prompt_name: str, reason: str, **fields) -> None:
    payload = {
        "event": "prompt_fallback",
        "prompt_name": prompt_name,
        "reason": reason,
    }
    payload.update({key: value for key, value in fields.items() if value is not None})
    logger.warning(payload)


def _log_prompt_load_failed(prompt_name: str, reason: str, detail: str) -> None:
    logger.warning(
        {
            "event": "prompt_load_failed",
            "prompt_name": prompt_name,
            "reason": reason,
            "detail": detail,
        }
    )


def clear_prompt_cache() -> None:
    return None


class LocalPromptRepository:
    def load(self, spec: PromptSpec) -> RepositoryLoadResult:
        prompt_path = _REPO_ROOT / spec.local_path
        candidate_key = f"local:{spec.local_path}"
        try:
            document = load_prompt_document_from_path(prompt_path)
            prompt = build_prompt_from_document(document)
            return RepositoryLoadResult(
                prompt=prompt,
                source="local",
                failure_reason=None,
                cache_hit_type="none",
                candidate_reasons={},
                prompt_version=document.version,
            )
        except Exception as exc:
            reason = classify_prompt_error(exc)
            _log_prompt_load_failed(spec.prompt_name, reason, f"{candidate_key}: {exc}")
            return RepositoryLoadResult(
                prompt=None,
                source="none",
                failure_reason=reason,
                cache_hit_type="none",
                candidate_reasons={candidate_key: reason},
                prompt_version=None,
            )


_LOCAL_REPOSITORY = LocalPromptRepository()


def _log_prompt_resolution(resolution: PromptResolution) -> None:
    logger.info(
        {
            "event": "prompt_resolution",
            "prompt_key": resolution.prompt_key,
            "resolution_source": resolution.resolution_source,
            "used_fallback": resolution.used_fallback,
            "failure_reason": resolution.failure_reason,
            "prewarm_phase": resolution.prewarm_phase,
            "cache_hit_type": resolution.cache_hit_type,
            "latency_ms": resolution.latency_ms,
            "success": resolution.success,
            "has_fallback": resolution.has_fallback,
            "prompt_version": resolution.prompt_version,
            "used_by": list(resolution.used_by),
            "critical": resolution.critical,
        }
    )


def resolve_prompt(
    prompt_name: str,
    fallback=None,
    *,
    fallback_source: str = "default",
    prewarm_phase: str = "runtime",
    has_fallback: bool | None = None,
    critical: bool = False,
) -> PromptResolution:
    started_at = time.monotonic()
    spec = get_prompt_spec(prompt_name)
    fallback_available = (fallback is not None) if has_fallback is None else has_fallback
    repository_result = _LOCAL_REPOSITORY.load(spec)

    if repository_result.prompt is not None:
        resolution = PromptResolution(
            prompt_key=prompt_name,
            prompt=repository_result.prompt,
            resolution_source=repository_result.source,
            used_fallback=False,
            failure_reason=None,
            prewarm_phase=prewarm_phase,
            cache_hit_type=repository_result.cache_hit_type,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            candidate_reasons=repository_result.candidate_reasons,
            success=True,
            has_fallback=fallback_available,
            prompt_version=repository_result.prompt_version,
            used_by=spec.used_by,
            critical=critical,
        )
        _log_prompt_resolution(resolution)
        return resolution

    resolution = PromptResolution(
        prompt_key=prompt_name,
        prompt=fallback,
        resolution_source=fallback_source if fallback is not None else "none",
        used_fallback=fallback is not None,
        failure_reason=repository_result.failure_reason,
        prewarm_phase=prewarm_phase,
        cache_hit_type=repository_result.cache_hit_type,
        latency_ms=int((time.monotonic() - started_at) * 1000),
        candidate_reasons=repository_result.candidate_reasons,
        success=fallback is not None,
        has_fallback=fallback_available,
        prompt_version=None,
        used_by=spec.used_by,
        critical=critical,
    )
    _log_prompt_resolution(resolution)
    return resolution


def load_prompt(prompt_name: str, fallback):
    return resolve_prompt(prompt_name, fallback).prompt


def prewarm_prompts(
    prompt_specs: list[PromptSpec] | tuple[PromptSpec, ...],
    *,
    fail_fast: bool | None = None,
) -> None:
    settings = get_settings()
    should_fail_fast = settings.prewarm_fail_fast if fail_fast is None else fail_fast
    total = len(prompt_specs)
    local_resolved = 0
    fallback_resolved = 0
    fail_fast_candidates = 0
    final_decision = "continue"

    for spec in prompt_specs:
        resolution = resolve_prompt(
            spec.prompt_name,
            fallback=None,
            prewarm_phase="startup",
            has_fallback=spec.has_fallback,
            critical=spec.critical,
        )
        if resolution.resolution_source == "local":
            local_resolved += 1
            continue

        action = "warning_continue"
        if resolution.used_fallback:
            fallback_resolved += 1
        if resolution.resolution_source == "none" and spec.critical and not spec.has_fallback:
            fail_fast_candidates += 1
        if resolution.resolution_source == "none" and spec.critical and not spec.has_fallback and should_fail_fast:
            action = "fail_fast"
            final_decision = "fail_fast"

        logger.warning(
            {
                "event": "prompt_prewarm_fallback",
                "prompt_name": spec.prompt_name,
                "source": resolution.resolution_source,
                "reason": resolution.failure_reason,
                "critical": spec.critical,
                "has_fallback": spec.has_fallback,
                "startup_action": action,
                "candidate_reasons": resolution.candidate_reasons,
            }
        )

        if action == "fail_fast":
            logger.warning(
                {
                    "event": "prompt_prewarm_summary",
                    "total": total,
                    "local_resolved": local_resolved,
                    "fallback_resolved": fallback_resolved,
                    "fail_fast_candidates": fail_fast_candidates,
                    "final_decision": final_decision,
                }
            )
            raise RuntimeError(
                f"Critical prompt '{spec.prompt_name}' is unavailable during prewarm and no local fallback exists."
            )

    logger.info(
        {
            "event": "prompt_prewarm_summary",
            "total": total,
            "local_resolved": local_resolved,
            "fallback_resolved": fallback_resolved,
            "fail_fast_candidates": fail_fast_candidates,
            "final_decision": final_decision,
        }
    )

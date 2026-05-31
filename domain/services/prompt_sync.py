import difflib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from langsmith import Client

from config.settings import get_settings
from domain.services.prompt_formats import (
    build_prompt_from_document,
    classify_prompt_error,
    dump_prompt_document,
    extract_prompt_error_detail,
    load_prompt_document_from_path,
    prompt_to_document,
)
from domain.services.prompt_registry import PROMPT_REGISTRY, PromptSpec, get_prompt_spec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HubPullResult:
    prompt: Any | None
    resolved_hub_name: str | None
    failure_reason: str | None
    candidate_reasons: dict[str, str]
    candidate_details: dict[str, str] = field(default_factory=dict)


@dataclass
class PromptSyncResult:
    key: str
    hub_name: str | None
    resolved_hub_name: str | None
    local_path: str
    prompt_version: str | None
    pulled: bool
    changed: bool
    validated: bool
    saved: bool
    error_reason: str | None
    candidate_reasons: dict[str, str]
    latency_ms: int
    candidate_details: dict[str, str] = field(default_factory=dict)


@dataclass
class PromptSyncOutcome:
    results: list[PromptSyncResult]
    diffs: dict[str, str]

    @property
    def changed_count(self) -> int:
        return sum(1 for result in self.results if result.changed)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.error_reason is not None)

    @property
    def saved_count(self) -> int:
        return sum(1 for result in self.results if result.saved)

    @property
    def changed_keys(self) -> list[str]:
        return [result.key for result in self.results if result.changed]

    @property
    def failed_keys(self) -> list[str]:
        return [result.key for result in self.results if result.error_reason is not None]


class SyncPromptRepository(Protocol):
    def load(self, spec: PromptSpec, *, revision: str = "latest") -> HubPullResult:
        ...


class LangSmithPromptRepository:
    def __init__(self, client: Client | None = None, *, prompt_namespace: str | None = None):
        settings = get_settings()
        self._client = client or Client()
        self._prompt_namespace = prompt_namespace if prompt_namespace is not None else settings.prompt_namespace

    def load(self, spec: PromptSpec, *, revision: str = "latest") -> HubPullResult:
        if not spec.hub_name:
            return HubPullResult(
                prompt=None,
                resolved_hub_name=None,
                failure_reason="hub_name_missing",
                candidate_reasons={},
                candidate_details={},
            )

        candidate_reasons: dict[str, str] = {}
        candidate_details: dict[str, str] = {}
        for hub_name in _build_hub_candidates(spec.hub_name, self._prompt_namespace):
            identifier = f"{hub_name}:{revision}" if revision else hub_name
            try:
                prompt = self._client.pull_prompt(identifier, skip_cache=True)
                return HubPullResult(
                    prompt=prompt,
                    resolved_hub_name=hub_name,
                    failure_reason=None,
                    candidate_reasons=candidate_reasons,
                    candidate_details=candidate_details,
                )
            except Exception as exc:
                candidate_reasons[hub_name] = classify_prompt_error(exc)
                candidate_details[hub_name] = extract_prompt_error_detail(exc)

        failure_reason = _select_failure_reason(candidate_reasons) if candidate_reasons else "hub_pull_failed"
        return HubPullResult(
            prompt=None,
            resolved_hub_name=None,
            failure_reason=failure_reason,
            candidate_reasons=candidate_reasons,
            candidate_details=candidate_details,
        )


def _build_hub_candidates(hub_name: str, namespace: str) -> tuple[str, ...]:
    candidates: list[str] = []
    if namespace and "_" not in hub_name and not hub_name.startswith(f"{namespace}_"):
        candidates.append(f"{namespace}_{hub_name}")
    candidates.append(hub_name)
    if namespace and "/" not in hub_name and not hub_name.startswith(f"{namespace}/"):
        candidates.append(f"{namespace}/{hub_name}")

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return tuple(ordered)


def _select_failure_reason(candidate_reasons: dict[str, str]) -> str:
    priority = {
        "workspace_required": 0,
        "prompt_owner_required": 1,
        "auth_error": 2,
        "langsmith_user_error": 3,
        "rate_limited": 4,
        "connection_error": 5,
        "hub_server_error": 6,
        "timeout": 7,
        "prompt_not_found": 8,
    }
    return min(
        candidate_reasons.values(),
        key=lambda reason: (priority.get(reason, 99), reason),
    )


def _resolve_local_path(repo_root: Path, local_path: str) -> Path:
    return repo_root / local_path


def _build_diff(local_path: str, current_text: str, new_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            current_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=local_path,
            tofile=local_path,
        )
    )


def _log_prompt_sync_result(result: PromptSyncResult) -> None:
    logger.info(
        {
            "event": "prompt_sync_result",
            "key": result.key,
            "hub_name": result.hub_name,
            "resolved_hub_name": result.resolved_hub_name,
            "local_path": result.local_path,
            "prompt_version": result.prompt_version,
            "pulled": result.pulled,
            "changed": result.changed,
            "validated": result.validated,
            "saved": result.saved,
            "error_reason": result.error_reason,
            "candidate_reasons": result.candidate_reasons,
            "candidate_details": result.candidate_details,
            "latency_ms": result.latency_ms,
        }
    )


def _log_prompt_sync_summary(outcome: PromptSyncOutcome, *, dry_run: bool, fail_on_diff: bool) -> None:
    logger.info(
        {
            "event": "prompt_sync_summary",
            "total": len(outcome.results),
            "changed_count": outcome.changed_count,
            "changed_keys": outcome.changed_keys,
            "failed_count": outcome.failed_count,
            "failed_keys": outcome.failed_keys,
            "saved_count": outcome.saved_count,
            "dry_run": dry_run,
            "fail_on_diff": fail_on_diff,
        }
    )


def _iter_target_specs(only_keys: list[str] | None = None) -> tuple[PromptSpec, ...]:
    if not only_keys:
        return tuple(sorted(PROMPT_REGISTRY.values(), key=lambda spec: spec.key))
    return tuple(sorted((get_prompt_spec(key) for key in only_keys), key=lambda spec: spec.key))


def sync_prompts_from_hub(
    *,
    only_keys: list[str] | None = None,
    specs: list[PromptSpec] | tuple[PromptSpec, ...] | None = None,
    dry_run: bool = False,
    fail_on_diff: bool = False,
    revision: str = "latest",
    repo_root: Path | None = None,
    hub_repository: SyncPromptRepository | None = None,
) -> PromptSyncOutcome:
    repo_root_path = repo_root or _REPO_ROOT
    repository = hub_repository or LangSmithPromptRepository()
    specs = tuple(specs) if specs is not None else _iter_target_specs(only_keys)
    results: list[PromptSyncResult] = []
    diffs: dict[str, str] = {}
    staged_writes: list[tuple[str, str, Path]] = []

    with TemporaryDirectory(prefix=".prompt-sync-", dir=repo_root_path) as staging_dir:
        staging_root = Path(staging_dir)
        for spec in specs:
            started_at = time.monotonic()
            local_path = _resolve_local_path(repo_root_path, spec.local_path)
            current_text = ""
            version: str | None = None

            try:
                current_document = load_prompt_document_from_path(local_path)
                version = current_document.version
                current_text = dump_prompt_document(current_document)
            except Exception as exc:
                result = PromptSyncResult(
                    key=spec.key,
                    hub_name=spec.hub_name,
                    resolved_hub_name=None,
                    local_path=spec.local_path,
                    prompt_version=None,
                    pulled=False,
                    changed=False,
                    validated=False,
                    saved=False,
                    error_reason=f"local_version_unavailable:{classify_prompt_error(exc)}",
                    candidate_reasons={},
                    candidate_details={},
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                )
                results.append(result)
                _log_prompt_sync_result(result)
                continue

            hub_result = repository.load(spec, revision=revision)
            if hub_result.prompt is None:
                result = PromptSyncResult(
                    key=spec.key,
                    hub_name=spec.hub_name,
                    resolved_hub_name=hub_result.resolved_hub_name,
                    local_path=spec.local_path,
                    prompt_version=version,
                    pulled=False,
                    changed=False,
                    validated=False,
                    saved=False,
                    error_reason=hub_result.failure_reason or "hub_pull_failed",
                    candidate_reasons=hub_result.candidate_reasons,
                    candidate_details=hub_result.candidate_details,
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                )
                results.append(result)
                _log_prompt_sync_result(result)
                continue

            try:
                document = prompt_to_document(hub_result.prompt, version=version)
                build_prompt_from_document(document)
                new_text = dump_prompt_document(document)
                changed = current_text != new_text
            except Exception as exc:
                result = PromptSyncResult(
                    key=spec.key,
                    hub_name=spec.hub_name,
                    resolved_hub_name=hub_result.resolved_hub_name,
                    local_path=spec.local_path,
                    prompt_version=version,
                    pulled=True,
                    changed=False,
                    validated=False,
                    saved=False,
                    error_reason=classify_prompt_error(exc),
                    candidate_reasons=hub_result.candidate_reasons,
                    candidate_details=hub_result.candidate_details,
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                )
                results.append(result)
                _log_prompt_sync_result(result)
                continue

            if changed:
                staged_path = staging_root / f"{spec.key}.yaml"
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                staged_path.write_text(new_text, encoding="utf-8")
                diffs[spec.key] = _build_diff(spec.local_path, current_text, new_text)
                staged_writes.append((spec.key, spec.local_path, staged_path))

            result = PromptSyncResult(
                key=spec.key,
                hub_name=spec.hub_name,
                resolved_hub_name=hub_result.resolved_hub_name,
                local_path=spec.local_path,
                prompt_version=version,
                pulled=True,
                changed=changed,
                validated=True,
                saved=False,
                error_reason=None,
                candidate_reasons=hub_result.candidate_reasons,
                candidate_details=hub_result.candidate_details,
                latency_ms=int((time.monotonic() - started_at) * 1000),
            )
            results.append(result)
            _log_prompt_sync_result(result)

        outcome = PromptSyncOutcome(results=results, diffs=diffs)
        if outcome.failed_count > 0:
            _log_prompt_sync_summary(outcome, dry_run=dry_run, fail_on_diff=fail_on_diff)
            return outcome

        if not dry_run:
            results_by_key = {result.key: result for result in results}
            for key, local_path, staged_path in staged_writes:
                target_path = _resolve_local_path(repo_root_path, local_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, target_path)
                document = load_prompt_document_from_path(target_path)
                build_prompt_from_document(document)
                results_by_key[key].saved = True

    _log_prompt_sync_summary(outcome, dry_run=dry_run, fail_on_diff=fail_on_diff)
    return outcome

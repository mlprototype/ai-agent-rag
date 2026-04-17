import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.prompts import ChatPromptTemplate

from domain.services.prompt_formats import (
    PromptDocument,
    PromptMessageDocument,
    build_prompt_from_document,
    dump_prompt_document,
    load_prompt_document_from_path,
)
from domain.services.prompt_registry import PromptSpec
from domain.services.prompt_sync import HubPullResult, PromptSyncOutcome, PromptSyncResult, sync_prompts_from_hub
from tools import sync_prompts_from_hub as sync_cli


class FakeHubRepository:
    def __init__(self, responses: dict[str, HubPullResult]):
        self._responses = responses

    def load(self, spec: PromptSpec, *, revision: str = "latest") -> HubPullResult:
        return self._responses[spec.key]


def _write_prompt(path: Path, *, version: str, template: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = PromptDocument(
        version=version,
        messages=[PromptMessageDocument(role="system", template=template)],
    )
    path.write_text(dump_prompt_document(document), encoding="utf-8")


class PromptSyncTests(unittest.TestCase):
    def test_sync_dry_run_reports_change_without_saving(self):
        with TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "prompts/router/v1.yaml"
            _write_prompt(local_path, version="v7", template="old")
            spec = PromptSpec(
                key="router",
                hub_name="agentic-rag-router",
                local_path=str(local_path),
                has_fallback=True,
                critical=True,
                used_by=("router",),
            )
            fake_repo = FakeHubRepository(
                {
                    "router": HubPullResult(
                        prompt=ChatPromptTemplate.from_messages([("system", "new")]),
                        resolved_hub_name="my-rag/agentic-rag-router",
                        failure_reason=None,
                        candidate_reasons={},
                        candidate_details={},
                    )
                }
            )

            outcome = sync_prompts_from_hub(
                specs=[spec],
                dry_run=True,
                repo_root=Path(temp_dir),
                hub_repository=fake_repo,
            )

            self.assertEqual(outcome.changed_count, 1)
            self.assertEqual(outcome.changed_keys, ["router"])
            self.assertEqual(outcome.failed_keys, [])
            self.assertEqual(outcome.failed_count, 0)
            self.assertFalse(outcome.results[0].saved)
            self.assertEqual(outcome.results[0].prompt_version, "v7")
            self.assertIn("new", outcome.diffs["router"])
            self.assertIn("old", local_path.read_text(encoding="utf-8"))

    def test_sync_is_all_or_nothing_on_failure(self):
        with TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "prompts/router/v1.yaml"
            second_path = Path(temp_dir) / "prompts/decompose/v1.yaml"
            _write_prompt(first_path, version="v1", template="router-old")
            _write_prompt(second_path, version="v1", template="decompose-old")
            specs = [
                PromptSpec("router", "agentic-rag-router", str(first_path), True, True, ("router",)),
                PromptSpec("decompose", "agentic-rag-decompose", str(second_path), True, True, ("decompose",)),
            ]
            fake_repo = FakeHubRepository(
                {
                    "router": HubPullResult(
                        prompt=ChatPromptTemplate.from_messages([("system", "router-new")]),
                        resolved_hub_name="my-rag/agentic-rag-router",
                        failure_reason=None,
                        candidate_reasons={},
                        candidate_details={},
                    ),
                    "decompose": HubPullResult(
                        prompt=None,
                        resolved_hub_name=None,
                        failure_reason="prompt_not_found",
                        candidate_reasons={"my-rag/agentic-rag-decompose": "prompt_not_found"},
                        candidate_details={"my-rag/agentic-rag-decompose": "Resource not found"},
                    ),
                }
            )

            outcome = sync_prompts_from_hub(
                specs=specs,
                dry_run=False,
                repo_root=Path(temp_dir),
                hub_repository=fake_repo,
            )

            self.assertEqual(outcome.failed_count, 1)
            self.assertEqual(outcome.changed_keys, ["router"])
            self.assertEqual(outcome.failed_keys, ["decompose"])
            self.assertFalse(any(result.saved for result in outcome.results))
            self.assertIn("router-old", first_path.read_text(encoding="utf-8"))
            self.assertIn("decompose-old", second_path.read_text(encoding="utf-8"))

    def test_sync_saves_after_all_validation_succeeds(self):
        with TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "prompts/rewrite/v1.yaml"
            _write_prompt(local_path, version="v3", template="rewrite-old")
            spec = PromptSpec(
                key="rewrite",
                hub_name="agentic-rag-rewrite",
                local_path=str(local_path),
                has_fallback=True,
                critical=True,
                used_by=("rewrite",),
            )
            fake_repo = FakeHubRepository(
                {
                    "rewrite": HubPullResult(
                        prompt=ChatPromptTemplate.from_messages([("human", "rewrite-new")]),
                        resolved_hub_name="my-rag/agentic-rag-rewrite",
                        failure_reason=None,
                        candidate_reasons={},
                        candidate_details={},
                    )
                }
            )

            outcome = sync_prompts_from_hub(
                specs=[spec],
                dry_run=False,
                repo_root=Path(temp_dir),
                hub_repository=fake_repo,
            )

            self.assertEqual(outcome.failed_count, 0)
            self.assertEqual(outcome.saved_count, 1)
            self.assertEqual(outcome.changed_keys, ["rewrite"])
            self.assertIn("rewrite-new", local_path.read_text(encoding="utf-8"))
            self.assertIn("version: v3", local_path.read_text(encoding="utf-8"))
            document = load_prompt_document_from_path(local_path)
            prompt = build_prompt_from_document(document)
            self.assertEqual(prompt.input_variables, [])

    def test_sync_uses_normalized_diff_instead_of_raw_yaml_formatting(self):
        with TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "prompts/router/v1.yaml"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(
                'version: "v1"\nformat: chat_prompt\nmessages:\n- role: system\n  template: "same"\n',
                encoding="utf-8",
            )
            spec = PromptSpec(
                key="router",
                hub_name="agentic-rag-router",
                local_path=str(local_path),
                has_fallback=True,
                critical=True,
                used_by=("router",),
            )
            fake_repo = FakeHubRepository(
                {
                    "router": HubPullResult(
                        prompt=ChatPromptTemplate.from_messages([("system", "same")]),
                        resolved_hub_name="my-rag/agentic-rag-router",
                        failure_reason=None,
                        candidate_reasons={},
                        candidate_details={},
                    )
                }
            )

            outcome = sync_prompts_from_hub(
                specs=[spec],
                dry_run=True,
                repo_root=Path(temp_dir),
                hub_repository=fake_repo,
            )

            self.assertEqual(outcome.changed_count, 0)
            self.assertEqual(outcome.diffs, {})

    def test_cli_returns_nonzero_when_fail_on_diff_is_requested(self):
        outcome = PromptSyncOutcome(
            results=[
                PromptSyncResult(
                    key="router",
                    hub_name="agentic-rag-router",
                    resolved_hub_name="my-rag/agentic-rag-router",
                    local_path="prompts/router/v1.yaml",
                    prompt_version="v1",
                    pulled=True,
                    changed=True,
                    validated=True,
                    saved=False,
                    error_reason=None,
                    candidate_reasons={},
                    candidate_details={},
                    latency_ms=10,
                )
            ],
            diffs={"router": "--- prompts/router/v1.yaml\n+++ prompts/router/v1.yaml\n"},
        )

        with patch.object(sync_cli, "sync_prompts_from_hub", return_value=outcome):
            exit_code = sync_cli.main(["--dry-run", "--fail-on-diff"])

        self.assertEqual(exit_code, sync_cli.EXIT_DIFF_FOUND)


if __name__ == "__main__":
    unittest.main()

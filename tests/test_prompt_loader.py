import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.prompts import ChatPromptTemplate

from domain.services import prompt_loader
from domain.services.prompt_registry import PromptSpec


class PromptLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        prompt_loader.clear_prompt_cache()

    def tearDown(self) -> None:
        prompt_loader.clear_prompt_cache()

    def test_resolve_prompt_prefers_local_snapshot(self):
        fallback = ChatPromptTemplate.from_messages([("human", "fallback")])
        resolution = prompt_loader.resolve_prompt("agentic-rag-router", fallback)

        self.assertEqual(resolution.resolution_source, "local")
        self.assertFalse(resolution.used_fallback)
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.prompt_version, "v1")
        self.assertEqual(resolution.used_by, ("router",))

    def test_resolve_prompt_returns_default_fallback_when_local_missing(self):
        fallback = ChatPromptTemplate.from_messages([("human", "fallback")])
        local_miss = prompt_loader.RepositoryLoadResult(
            prompt=None,
            source="none",
            failure_reason="prompt_not_found",
            cache_hit_type="none",
            candidate_reasons={"local:test": "prompt_not_found"},
            prompt_version=None,
        )

        with patch.object(prompt_loader.LocalPromptRepository, "load", return_value=local_miss):
            resolution = prompt_loader.resolve_prompt("agentic-rag-router", fallback)

        self.assertEqual(resolution.resolution_source, "default")
        self.assertTrue(resolution.used_fallback)
        self.assertEqual(resolution.failure_reason, "prompt_not_found")
        self.assertIsNone(resolution.prompt_version)

    def test_local_prompt_repository_is_strict_on_invalid_schema(self):
        with TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "broken.yaml"
            invalid_path.write_text(
                "version: v1\nformat: chat_prompt\nmessages:\n  - role: messages_placeholder\n",
                encoding="utf-8",
            )
            spec = PromptSpec(
                key="broken",
                hub_name="agentic-rag-broken",
                local_path=str(invalid_path),
                has_fallback=True,
                critical=False,
                used_by=("test",),
            )
            with patch.object(prompt_loader, "_REPO_ROOT", Path("/")):
                result = prompt_loader._LOCAL_REPOSITORY.load(spec)

        self.assertIsNone(result.prompt)
        self.assertEqual(result.failure_reason, "schema_validation_error")

    def test_prewarm_continues_for_prompt_with_local_fallback(self):
        with patch.object(
            prompt_loader,
            "resolve_prompt",
            return_value=prompt_loader.PromptResolution(
                prompt_key="agentic-rag-router",
                prompt=None,
                resolution_source="none",
                used_fallback=False,
                failure_reason="prompt_not_found",
                prewarm_phase="startup",
                cache_hit_type="none",
                latency_ms=12,
                candidate_reasons={"local:test": "prompt_not_found"},
                success=False,
                has_fallback=True,
                prompt_version=None,
                used_by=("router",),
                critical=True,
            ),
        ):
            prompt_loader.prewarm_prompts(
                [PromptSpec("router", "agentic-rag-router", "prompts/router/v1.yaml", has_fallback=True, critical=True, used_by=("router",))],
                fail_fast=True,
            )

    def test_prewarm_fail_fast_for_critical_prompt_without_fallback(self):
        with patch.object(
            prompt_loader,
            "resolve_prompt",
            return_value=prompt_loader.PromptResolution(
                prompt_key="agentic-rag-custom",
                prompt=None,
                resolution_source="none",
                used_fallback=False,
                failure_reason="prompt_not_found",
                prewarm_phase="startup",
                cache_hit_type="none",
                latency_ms=9,
                candidate_reasons={"local:test": "prompt_not_found"},
                success=False,
                has_fallback=False,
                prompt_version=None,
                used_by=("test",),
                critical=True,
            ),
        ):
            with self.assertRaises(RuntimeError):
                prompt_loader.prewarm_prompts(
                    [PromptSpec("custom", "agentic-rag-custom", "prompts/custom/v1.yaml", has_fallback=False, critical=True, used_by=("test",))],
                    fail_fast=True,
                )


if __name__ == "__main__":
    unittest.main()

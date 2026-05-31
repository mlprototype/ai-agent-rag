import argparse
import logging
import sys

from domain.services.prompt_sync import PromptSyncOutcome, sync_prompts_from_hub

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DIFF_FOUND = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync prompt snapshots from Hub into local YAML files.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Registry key to sync. Repeat to target multiple prompts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and diff only. Do not write local files.",
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit non-zero when synced content would change local snapshots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print unified diffs for changed prompts.",
    )
    return parser


def _normalize_only_keys(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        for key in value.split(","):
            stripped = key.strip()
            if stripped:
                normalized.append(stripped)
    return normalized


def _print_outcome(outcome: PromptSyncOutcome, *, show_diffs: bool) -> None:
    for result in outcome.results:
        status = "ERROR" if result.error_reason else ("CHANGED" if result.changed else "OK")
        detail = [
            f"key={result.key}",
            f"local_path={result.local_path}",
            f"version={result.prompt_version}",
        ]
        if result.hub_name:
            detail.append(f"hub_name={result.hub_name}")
        if result.resolved_hub_name:
            detail.append(f"resolved_hub_name={result.resolved_hub_name}")
        detail.append(f"pulled={result.pulled}")
        detail.append(f"validated={result.validated}")
        detail.append(f"saved={result.saved}")
        detail.append(f"latency_ms={result.latency_ms}")
        if result.error_reason:
            detail.append(f"error_reason={result.error_reason}")
        if result.candidate_reasons:
            detail.append(f"candidate_reasons={result.candidate_reasons}")
        if result.candidate_details:
            detail.append(f"candidate_details={result.candidate_details}")
        print(f"[{status}] " + " ".join(detail))
        if show_diffs and result.key in outcome.diffs:
            print(outcome.diffs[result.key], end="" if outcome.diffs[result.key].endswith("\n") else "\n")


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    only_keys = _normalize_only_keys(args.only)

    try:
        outcome = sync_prompts_from_hub(
            only_keys=only_keys or None,
            dry_run=args.dry_run,
            fail_on_diff=args.fail_on_diff,
        )
    except Exception as exc:
        print(f"[ERROR] sync failed before completion: {exc}", file=sys.stderr)
        return EXIT_FAILED

    _print_outcome(outcome, show_diffs=args.verbose or args.dry_run)

    if outcome.failed_count > 0:
        return EXIT_FAILED
    if args.fail_on_diff and outcome.changed_count > 0:
        return EXIT_DIFF_FOUND
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

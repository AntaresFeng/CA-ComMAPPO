import argparse
import sys
from dataclasses import replace
from pathlib import Path

from ca_commappo.evaluation.highway_metrics import (
    format_summary_lines,
    save_results_json,
)
from ca_commappo.evaluation.sanity_baseline_runner import (
    SUPPORTED_POLICIES,
    SanityConfig,
    load_sanity_config,
    run_sanity_baseline,
)
from ca_commappo.evaluation.sanity_eval_artifacts import (
    write_sanity_eval_artifacts,
)

DEFAULT_CONFIG = Path("configs/sanity/highway_intersection.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random and idle-only sanity baselines for highway intersection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a YAML sanity baseline config.",
    )
    parser.add_argument(
        "--policy",
        choices=SUPPORTED_POLICIES + ("all",),
        default="all",
        help="Policy to evaluate. Use 'all' for all policies in the config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path for full episode records and summaries.",
    )
    parser.add_argument(
        "--episode-seed-base",
        type=int,
        default=None,
        help="Override the first explicit final-test episode seed.",
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        default=None,
        help="Override the number of contiguous final-test episode seeds.",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Artifact identity seed; it does not replace explicit episode seeds.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Write strict per-policy test artifacts below "
            "<root>/<policy>_highway/<run>/ for final-result aggregation."
        ),
    )
    return parser.parse_args(argv)


def print_summary(results: dict) -> None:
    for policy, policy_results in results["policies"].items():
        summary = policy_results["summary"]
        print(f"policy={policy}")
        for line in format_summary_lines(summary, indent="  "):
            print(line)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_sanity_config(args.config)
        config = _apply_protocol_overrides(config, args)
        results = run_sanity_baseline(config, policy=args.policy)
        print_summary(results)
        if args.output is not None:
            save_results_json(results, args.output)
            print(f"saved={args.output}")
        if args.artifact_root is not None:
            artifact_paths = write_sanity_eval_artifacts(
                results=results,
                config=config,
                artifact_root=args.artifact_root,
            )
            for policy, records_path in artifact_paths.items():
                print(f"artifact[{policy}]={records_path}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports failures as exit 1.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _apply_protocol_overrides(
    config: SanityConfig, args: argparse.Namespace
) -> SanityConfig:
    episode_seed_base = (
        config.episode_seed_base
        if args.episode_seed_base is None
        else args.episode_seed_base
    )
    episode_count = (
        config.episode_count if args.episode_count is None else args.episode_count
    )
    if (episode_seed_base is None) != (episode_count is None):
        raise ValueError(
            "episode_seed_base and episode_count must either both be set or both "
            "be omitted"
        )
    if episode_count is not None and episode_count <= 0:
        raise ValueError("episode_count must be a positive integer")
    return replace(
        config,
        episode_seed_base=episode_seed_base,
        episode_count=episode_count,
        run_seed=config.run_seed if args.run_seed is None else args.run_seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())

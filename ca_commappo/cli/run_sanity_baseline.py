import argparse
import sys
from pathlib import Path

from ca_commappo.evaluation.highway_metrics import (
    format_summary_lines,
    save_results_json,
)
from ca_commappo.evaluation.sanity_baseline_runner import (
    load_sanity_config,
    run_sanity_baseline,
    SUPPORTED_POLICIES,
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
        results = run_sanity_baseline(config, policy=args.policy)
        print_summary(results)
        if args.output is not None:
            save_results_json(results, args.output)
            print(f"saved={args.output}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

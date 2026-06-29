import argparse
import sys
from pathlib import Path

from ca_commappo.evaluation.sanity_baselines import (
    load_sanity_config,
    run_sanity_baseline,
    save_results_json,
)

DEFAULT_CONFIG = Path("examples/sanity/highway_intersection.yaml")


def parse_args() -> argparse.Namespace:
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
        choices=["random", "idle-only", "all"],
        default="all",
        help="Policy to evaluate. Use 'all' for all policies in the config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path for full episode records and summaries.",
    )
    return parser.parse_args()


def print_summary(results: dict) -> None:
    for policy, policy_results in results["policies"].items():
        summary = policy_results["summary"]
        print(f"policy={policy}")
        print(f"  episodes={summary['episodes']}")
        print(f"  mean_episode_reward={summary['mean_episode_reward']:.3f}")
        print(f"  mean_agent_reward={summary['mean_agent_reward']:.3f}")
        print(f"  mean_episode_length={summary['mean_episode_length']:.3f}")
        print(f"  collision_rate={summary['collision_rate']:.3f}")
        print(f"  arrival_rate={summary['arrival_rate']:.3f}")
        print(f"  truncation_rate={summary['truncation_rate']:.3f}")


def main() -> int:
    args = parse_args()
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

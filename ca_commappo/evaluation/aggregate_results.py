"""Aggregate eval_records.jsonl across seeds into a mean±std table (Phase 5.1).

Reads the ``eval_records.jsonl`` files produced by ``--mode benchmark`` / ``test``
(see ``ca_commappo/evaluation/mappo_eval_artifacts.py``), picks the best summary
per run by the highway best-model sort key (arrival↑ -> collision↓ -> reward↑ ->
length↓), groups runs by algorithm (derived from the ``runs/<algo>_highway/``
path), and emits mean±std per metric across seeds.

Usage:
    python -m ca_commappo.evaluation.aggregate_results \\
        --runs-glob "runs/*_highway/*benchmark_seed_*/seed_*/eval_records.jsonl" \\
        --output results/aggregated.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

METRICS = [
    "arrival_rate",
    "collision_rate",
    "mean_episode_reward",
    "mean_agent_reward",
    "mean_episode_length",
    "mean_agent_collision_fraction",
    "mean_agent_arrival_fraction",
    "mean_throughput",
    "mean_waiting_time",
    "mean_travel_time",
]


def _sort_key(summary: dict) -> tuple[float, float, float, float]:
    """Highway best-model sort key: arrival↑ -> collision↓ -> reward↑ -> length↓."""
    return (
        float(summary.get("arrival_rate", 0.0)),
        -float(summary.get("collision_rate", 0.0)),
        float(summary.get("mean_episode_reward", 0.0)),
        -float(summary.get("mean_episode_length", 0.0)),
    )


def best_summary(records: list[dict]) -> dict:
    """Pick the best summary (by the sort key) from a run's eval records."""
    summaries = [r["summary"] for r in records if "summary" in r]
    if not summaries:
        raise ValueError("no summary in records")
    return max(summaries, key=_sort_key)


def algo_from_path(path: str | Path) -> str:
    """Derive the algorithm name from a ``runs/<algo>_highway/...`` path."""
    for part in Path(path).parts:
        if part.endswith("_highway"):
            return part[: -len("_highway")]
    return "unknown"


def load_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(jsonl_paths: Iterable[str]) -> list[dict]:
    by_algo: dict[str, list[dict]] = {}
    for path in jsonl_paths:
        records = load_records(path)
        if not records:
            continue
        by_algo.setdefault(algo_from_path(path), []).append(best_summary(records))

    rows: list[dict] = []
    for algo in sorted(by_algo):
        summaries = by_algo[algo]
        row: dict[str, object] = {"algorithm": algo, "seeds": len(summaries)}
        for metric in METRICS:
            values = [float(s[metric]) for s in summaries if metric in s]
            if values:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                )
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-glob",
        required=True,
        help="glob pattern for eval_records.jsonl files",
    )
    parser.add_argument("--output", default=None, help="output CSV path")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.runs_glob))
    if not paths:
        print(f"No files matched: {args.runs_glob}", file=sys.stderr)
        return 1
    rows = aggregate(paths)
    if not rows:
        print("No eval records found.", file=sys.stderr)
        return 1

    if args.output:
        write_csv(rows, args.output)
        print(f"Wrote {args.output}")
    for row in rows:
        algo = row["algorithm"]
        seeds = row["seeds"]
        arr = row.get("arrival_rate_mean")
        col = row.get("collision_rate_mean")
        rew = row.get("mean_episode_reward_mean")
        print(
            f"{algo:20s} seeds={seeds} "
            f"arrival={_fmt(arr)} collision={_fmt(col)} reward={_fmt(rew)}"
        )
    return 0


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())

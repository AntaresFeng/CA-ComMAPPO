"""Tests for aggregate_results (Phase 5.1)."""

from __future__ import annotations

import json
from pathlib import Path

from ca_commappo.evaluation.aggregate_results import (
    aggregate,
    algo_from_path,
    best_summary,
    load_records,
)


def _record(arrival, collision, reward, length=10.0):
    return {
        "epoch": 0,
        "is_best": True,
        "summary": {
            "episodes": 1,
            "arrival_rate": arrival,
            "collision_rate": collision,
            "mean_episode_reward": reward,
            "mean_episode_length": length,
        },
    }


def _write_run(tmp_path: Path, algo: str, run_name: str, records: list[dict]) -> Path:
    d = tmp_path / f"runs/{algo}_highway/{run_name}/seed_0"
    d.mkdir(parents=True)
    p = d / "eval_records.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def test_best_summary_prefers_higher_arrival():
    records = [_record(0.0, 1.0, 1.0), _record(1.0, 0.0, 0.5)]
    s = best_summary(records)
    assert s["arrival_rate"] == 1.0


def test_best_summary_prefers_lower_collision_on_arrival_tie():
    records = [_record(0.5, 0.9, 0.0), _record(0.5, 0.1, 0.0)]
    s = best_summary(records)
    assert s["collision_rate"] == 0.1


def test_algo_from_path():
    assert (
        algo_from_path("runs/ca_commappo_highway/x/seed_0/eval_records.jsonl")
        == "ca_commappo"
    )
    assert algo_from_path("runs/mappo_highway/y/seed_1/eval_records.jsonl") == "mappo"


def test_aggregate_groups_by_algo_and_computes_mean_std(tmp_path):
    # Two seeds for ca_commappo, one for mappo.
    _write_run(
        tmp_path,
        "ca_commappo",
        "run_a",
        [_record(1.0, 0.0, 2.0, 8.0)],
    )
    _write_run(
        tmp_path,
        "ca_commappo",
        "run_b",
        # run picks the best summary: arrival=1.0 wins over arrival=0.0
        [_record(0.0, 1.0, 0.0, 5.0), _record(1.0, 0.0, 4.0, 12.0)],
    )
    _write_run(tmp_path, "mappo", "run_a", [_record(0.5, 0.5, 1.0, 10.0)])

    paths = sorted(tmp_path.glob("runs/*_highway/**/eval_records.jsonl"))
    rows = aggregate(paths)

    by_algo = {r["algorithm"]: r for r in rows}
    assert set(by_algo) == {"ca_commappo", "mappo"}
    ca = by_algo["ca_commappo"]
    assert ca["seeds"] == 2
    # both ca runs have arrival 1.0 -> mean 1.0, std 0.0
    assert ca["arrival_rate_mean"] == 1.0
    assert ca["arrival_rate_std"] == 0.0
    # rewards 2.0 and 4.0 -> mean 3.0
    assert ca["mean_episode_reward_mean"] == 3.0
    m = by_algo["mappo"]
    assert m["seeds"] == 1
    assert m["arrival_rate_mean"] == 0.5


def test_load_records_ignores_blank_lines(tmp_path):
    p = tmp_path / "eval_records.jsonl"
    p.write_text(
        json.dumps(_record(1.0, 0.0, 1.0))
        + "\n\n"
        + json.dumps(_record(0.0, 1.0, 0.0)),
        encoding="utf-8",
    )
    assert len(load_records(p)) == 2

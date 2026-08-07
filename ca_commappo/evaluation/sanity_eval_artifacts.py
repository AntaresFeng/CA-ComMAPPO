"""Write strict sanity results in the same final-test schema as MAPPO.

The legacy ``--output`` JSON remains useful for interactive inspection.  These
artifacts are deliberately stricter: one isolated run directory per policy,
one metadata file, and exactly one final-test JSONL record.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ca_commappo.evaluation.mappo_eval_artifacts import (
    append_eval_record_jsonl,
    build_eval_metadata,
    build_eval_record,
    write_eval_metadata,
)
from ca_commappo.evaluation.sanity_baseline_runner import SanityConfig


def write_sanity_eval_artifacts(
    *,
    results: dict[str, Any],
    config: SanityConfig,
    artifact_root: str | Path,
    created_at: str | None = None,
) -> dict[str, Path]:
    """Write one aggregator-compatible final-test run per requested policy."""
    artifact_root = Path(artifact_root)
    timestamp = created_at or datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    paths: dict[str, Path] = {}

    for policy, policy_results in results["policies"].items():
        protocol = policy_results.get("evaluation_protocol")
        if not isinstance(protocol, dict):
            raise ValueError(
                "comparable sanity artifacts require a per-policy strict "
                "evaluation_protocol"
            )
        _validate_policy_result(
            policy_results["episodes"],
            policy_results["summary"],
            protocol,
            policy,
        )

    for policy, policy_results in results["policies"].items():
        protocol = policy_results["evaluation_protocol"]
        algorithm = policy.replace("-", "_")
        run_dir = (
            artifact_root
            / f"{algorithm}_highway"
            / f"{timestamp}_test_seed_{config.run_seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)

        episodes = policy_results["episodes"]
        summary = policy_results["summary"]

        metadata_config = Namespace(
            config_path=config.config_path,
            env_name=config.env_name,
            env_id=config.env_id,
            seed=config.run_seed,
            logger="sanity",
            policy=policy,
            action_selection_mode=protocol["action_selection_mode"],
            test_episode=config.episode_count,
            n_agents=int(config.highway_config["controlled_vehicles"]),
            highway_config=config.highway_config,
        )
        agents = SimpleNamespace(log_dir=str(run_dir))
        metadata = build_eval_metadata(
            metadata_config,
            agents,
            "test",
            created_at=timestamp,
            evaluation_protocol=protocol,
        )
        record = build_eval_record(
            mode="test",
            phase="test",
            epoch=None,
            step=0,
            is_initial_eval=False,
            is_best=True,
            eval_result={
                "scores": [float(episode["score"]) for episode in episodes],
                "summary": summary,
                "episodes": episodes,
            },
            evaluation_protocol=protocol,
        )
        write_eval_metadata(metadata, run_dir)
        records_path = append_eval_record_jsonl(record, run_dir)
        paths[policy] = records_path

    return paths


def _validate_policy_result(
    episodes: Any,
    summary: Any,
    protocol: dict[str, Any],
    policy: str,
) -> None:
    expected_seeds = protocol["episode_seeds"]
    if not isinstance(episodes, list) or len(episodes) != len(expected_seeds):
        raise ValueError(
            f"policy {policy!r} does not contain exactly {len(expected_seeds)} episodes"
        )
    observed_seeds = [episode.get("seed") for episode in episodes]
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"policy {policy!r} episode seeds do not match evaluation_protocol"
        )
    for episode_index, episode in enumerate(episodes):
        score = episode.get("score")
        reward = episode.get("episode_reward")
        if not isinstance(score, (int, float)) or float(score) != float(reward):
            raise ValueError(
                f"policy {policy!r} episode {episode_index} score must equal "
                "episode_reward"
            )
    if not isinstance(summary, dict) or summary.get("episodes") != len(episodes):
        raise ValueError(f"policy {policy!r} summary episode count is inconsistent")

    required_metrics = ("mean_travel_time", "mean_throughput", "mean_waiting_time")
    missing = [metric for metric in required_metrics if metric not in summary]
    if missing:
        raise ValueError(
            f"policy {policy!r} is missing final-test metrics: {', '.join(missing)}"
        )

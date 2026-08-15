"""Deterministic local-observation order perturbations and paired artifacts."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ca_commappo.evaluation.mappo_eval_artifacts import json_safe

ORDER_ROBUSTNESS_SCHEMA_VERSION = 1
DEFAULT_PERMUTATION_SEEDS = (30_000, 30_001, 30_002)
ORDER_ROBUSTNESS_METADATA_FILENAME = "order_robustness_metadata.json"
ORDER_ROBUSTNESS_PAIRS_FILENAME = "order_robustness_pairs.jsonl"
ORDER_ROBUSTNESS_SUMMARY_FILENAME = "order_robustness_summary.json"
_PAIRED_METRICS = (
    "arrival",
    "collision",
    "truncated",
    "episode_reward",
    "steps",
    "agent_collision_fraction",
    "agent_arrival_fraction",
    "throughput",
    "waiting_time",
    "travel_time",
)


def observation_permutation_protocol(
    *,
    permutation_seeds: Sequence[int],
    vehicles_count: int,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    seeds = _validate_permutation_seeds(permutation_seeds)
    if vehicles_count < 2:
        raise ValueError(
            "vehicles_count must include ego and at least one neighbour row"
        )
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    return {
        "schema_version": ORDER_ROBUSTNESS_SCHEMA_VERSION,
        "type": "deterministic_neighbour_row_permutation",
        "target": "actor_local_observation_only",
        "ego_row": 0,
        "permuted_rows": list(range(1, int(vehicles_count))),
        "feature_names": [str(name) for name in feature_names],
        "permutation_seeds": seeds,
        "schedule": "seed_sequence(permutation_seed,episode_seed,action_step,agent_index)",
        "global_state_unchanged": True,
        "agent_mask_unchanged": True,
    }


def make_neighbour_row_permuter(
    *,
    permutation_seed: int,
    vehicles_count: int,
    feature_count: int,
) -> Callable[[dict[str, Any], int, int], dict[str, Any]]:
    if permutation_seed < 0:
        raise ValueError("permutation_seed must be non-negative")
    if vehicles_count < 2 or feature_count <= 0:
        raise ValueError("vehicles_count and feature_count are invalid")

    def transform(
        observations: dict[str, Any], episode_seed: int, action_step: int
    ) -> dict[str, Any]:
        if episode_seed < 0 or action_step < 0:
            raise ValueError("episode_seed and action_step must be non-negative")
        transformed: dict[str, Any] = {}
        for agent_index, agent in enumerate(sorted(observations)):
            value = np.asarray(observations[agent])
            expected_size = vehicles_count * feature_count
            if value.size != expected_size:
                raise ValueError(
                    f"agent {agent!r} observation has {value.size} values; "
                    f"expected {expected_size}"
                )
            rows = value.reshape(vehicles_count, feature_count)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [permutation_seed, episode_seed, action_step, agent_index]
                )
            )
            row_order = np.concatenate(
                (np.array([0]), rng.permutation(np.arange(1, vehicles_count)))
            )
            transformed[agent] = rows[row_order].reshape(value.shape).copy()
        return transformed

    return transform


def build_order_robustness_pairs(
    *,
    baseline_result: dict[str, Any],
    perturbed_results: Mapping[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_seed = _episodes_by_seed(baseline_result)
    perturbed_by_seed = {
        int(permutation_seed): _episodes_by_seed(result)
        for permutation_seed, result in perturbed_results.items()
    }
    expected_episode_seeds = set(baseline_by_seed)
    for permutation_seed, episodes in perturbed_by_seed.items():
        if set(episodes) != expected_episode_seeds:
            raise ValueError(
                f"permutation seed {permutation_seed} does not match baseline cases"
            )

    pairs = []
    for episode_seed in sorted(expected_episode_seeds):
        baseline = baseline_by_seed[episode_seed]
        perturbations = []
        for permutation_seed in sorted(perturbed_by_seed):
            perturbed = perturbed_by_seed[permutation_seed][episode_seed]
            perturbations.append(
                {
                    "permutation_seed": permutation_seed,
                    "episode": perturbed,
                    "deltas": _episode_deltas(baseline, perturbed),
                }
            )
        pairs.append(
            {
                "schema_version": ORDER_ROBUSTNESS_SCHEMA_VERSION,
                "episode_seed": episode_seed,
                "baseline": baseline,
                "perturbations": perturbations,
            }
        )
    return pairs


def build_order_robustness_summary(
    *,
    baseline_result: dict[str, Any],
    perturbed_results: Mapping[int, dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    variants = []
    for permutation_seed in sorted(perturbed_results):
        result = perturbed_results[permutation_seed]
        deltas = [
            perturbation["deltas"]
            for pair in pairs
            for perturbation in pair["perturbations"]
            if perturbation["permutation_seed"] == permutation_seed
        ]
        variants.append(
            {
                "permutation_seed": permutation_seed,
                "summary": result["summary"],
                "summary_deltas": _numeric_deltas(
                    baseline_result["summary"], result["summary"]
                ),
                "paired_case_deltas": _summarize_paired_deltas(deltas),
            }
        )
    return {
        "schema_version": ORDER_ROBUSTNESS_SCHEMA_VERSION,
        "baseline_summary": baseline_result["summary"],
        "variants": variants,
    }


def write_order_robustness_artifacts(
    *,
    output_dir: str | Path,
    config: Any,
    evaluation_protocol: dict[str, Any],
    permutation_protocol: dict[str, Any],
    model_provenance: dict[str, Any],
    baseline_result: dict[str, Any],
    perturbed_results: Mapping[int, dict[str, Any]],
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pairs = build_order_robustness_pairs(
        baseline_result=baseline_result,
        perturbed_results=perturbed_results,
    )
    summary = build_order_robustness_summary(
        baseline_result=baseline_result,
        perturbed_results=perturbed_results,
        pairs=pairs,
    )
    metadata = {
        "schema_version": ORDER_ROBUSTNESS_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": json_safe(config),
        "evaluation_protocol": json_safe(evaluation_protocol),
        "permutation_protocol": json_safe(permutation_protocol),
        "model_provenance": json_safe(model_provenance),
    }
    paths = {
        "metadata": destination / ORDER_ROBUSTNESS_METADATA_FILENAME,
        "pairs": destination / ORDER_ROBUSTNESS_PAIRS_FILENAME,
        "summary": destination / ORDER_ROBUSTNESS_SUMMARY_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite order-robustness artifacts: {existing}"
        )
    _write_json_exclusive(paths["metadata"], metadata)
    try:
        with paths["pairs"].open("x", encoding="utf-8") as output:
            for pair in pairs:
                output.write(json.dumps(json_safe(pair), sort_keys=True) + "\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite order-robustness pairs: {paths['pairs']}"
        ) from error
    _write_json_exclusive(paths["summary"], summary)
    return paths


def _validate_permutation_seeds(seeds: Sequence[int]) -> list[int]:
    values = list(seeds)
    if not values:
        raise ValueError("permutation_seeds must not be empty")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in values
    ):
        raise ValueError("permutation_seeds must be non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError("permutation_seeds must not contain duplicates")
    return values


def _episodes_by_seed(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("evaluation result must contain episode records")
    by_seed: dict[int, dict[str, Any]] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            raise TypeError("episode records must be objects")
        seed = episode.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("each episode record must have an integer seed")
        if seed in by_seed:
            raise ValueError(f"duplicate episode seed: {seed}")
        by_seed[seed] = episode
    return by_seed


def _episode_deltas(
    baseline: dict[str, Any], perturbed: dict[str, Any]
) -> dict[str, Any]:
    deltas = _numeric_deltas(baseline, perturbed, keys=_PAIRED_METRICS)
    baseline_rewards = baseline.get("agent_rewards")
    perturbed_rewards = perturbed.get("agent_rewards")
    if isinstance(baseline_rewards, dict) and isinstance(perturbed_rewards, dict):
        if set(baseline_rewards) != set(perturbed_rewards):
            raise ValueError("paired episodes have different agent reward keys")
        deltas["agent_rewards"] = {
            agent: float(perturbed_rewards[agent]) - float(baseline_rewards[agent])
            for agent in sorted(baseline_rewards)
        }
    return deltas


def _numeric_deltas(
    baseline: Mapping[str, Any],
    perturbed: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, float]:
    selected = keys or sorted(set(baseline) & set(perturbed))
    deltas: dict[str, float] = {}
    for key in selected:
        if key not in baseline or key not in perturbed:
            continue
        baseline_value = baseline[key]
        perturbed_value = perturbed[key]
        if isinstance(baseline_value, bool) and isinstance(perturbed_value, bool):
            deltas[key] = float(int(perturbed_value) - int(baseline_value))
        elif (
            not isinstance(baseline_value, bool)
            and not isinstance(perturbed_value, bool)
            and isinstance(baseline_value, (int, float))
            and isinstance(perturbed_value, (int, float))
        ):
            deltas[key] = float(perturbed_value) - float(baseline_value)
    return deltas


def _summarize_paired_deltas(deltas: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            key
            for delta in deltas
            for key, value in delta.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    result = {}
    for metric in metric_names:
        values = [float(delta[metric]) for delta in deltas if metric in delta]
        result[metric] = {
            "cases": len(values),
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(
                json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
            )
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite artifact: {path}") from error


__all__ = [
    "DEFAULT_PERMUTATION_SEEDS",
    "ORDER_ROBUSTNESS_METADATA_FILENAME",
    "ORDER_ROBUSTNESS_PAIRS_FILENAME",
    "ORDER_ROBUSTNESS_SCHEMA_VERSION",
    "ORDER_ROBUSTNESS_SUMMARY_FILENAME",
    "build_order_robustness_pairs",
    "build_order_robustness_summary",
    "make_neighbour_row_permuter",
    "observation_permutation_protocol",
    "write_order_robustness_artifacts",
]

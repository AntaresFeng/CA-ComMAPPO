"""Aggregate independent final-test results across training seeds.

Only the 100+-episode record produced by ``--mode test`` is a final result.
Benchmark records are checkpoint-selection evidence and are deliberately rejected
here; in particular, this module never takes a best-of-benchmark summary.

Artifacts may live at the run root (W&B) or one logger-created child (TensorBoard)::

    runs/<algo>_highway/<timestamp>_test_seed_<seed>/eval_metadata.json
    runs/<algo>_highway/<timestamp>_test_seed_<seed>/eval_records.jsonl
    runs/<algo>_highway/<timestamp>_test_seed_<seed>/<logger-child>/eval_*.json*

Usage::

    python -m ca_commappo.evaluation.aggregate_results \
        --runs-glob "runs/*_highway/*_test_seed_*/**/eval_records.jsonl" \
        --expected-seeds 1 2 3 4 5 \
        --output results/aggregated.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from argparse import Namespace
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ca_commappo.config import TRAINING_ENV_SEED_SCHEME, derive_training_env_seed
from ca_commappo.evaluation.highway_metrics import summarize_episode_records
from ca_commappo.evaluation.mappo_eval_artifacts import SCHEMA_VERSION
from ca_commappo.evaluation.model_provenance import (
    BEST_MODEL_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    VERIFIED_PROVENANCE,
    provenance_path_for_model,
    validate_model_provenance_payload,
)

MIN_FINAL_TEST_EPISODES = 100
_STATIC_BASELINE_ALGORITHMS = {"random", "idle_only"}

METRICS = [
    "arrival_rate",
    "collision_rate",
    "truncation_rate",
    "mean_episode_reward",
    "mean_agent_reward",
    "mean_episode_length",
    "mean_agent_collision_fraction",
    "mean_agent_arrival_fraction",
    "mean_throughput",
    "mean_waiting_time",
    "mean_travel_time",
]


class AggregationError(ValueError):
    """Raised when an input cannot be used as a comparable final result."""


@dataclass(frozen=True)
class _FinalResult:
    path: Path
    algorithm: str
    seed: int
    summary: dict
    protocol: tuple[tuple[str, str], ...]
    checkpoint_sha256: str | None
    training_config_sha256: str | None
    source_commit: str | None
    runtime_source_sha256: str | None
    training_device: str | None


def _sort_key(summary: dict) -> tuple[float, float, float, float]:
    """Highway best-model sort key: arrival up, collision down, reward up."""
    return (
        float(summary.get("arrival_rate", 0.0)),
        -float(summary.get("collision_rate", 0.0)),
        float(summary.get("mean_episode_reward", 0.0)),
        -float(summary.get("mean_episode_length", 0.0)),
    )


def best_summary(records: list[dict]) -> dict:
    """Compatibility helper for analysing benchmark checkpoint records.

    Final-result aggregation intentionally does not call this function.
    """
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


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except FileNotFoundError as error:
        raise AggregationError(
            f"missing metadata beside eval records: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise AggregationError(f"invalid JSON metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise AggregationError(f"metadata must be a JSON object: {path}")
    return value


def _require_seed(metadata: dict, metadata_path: Path) -> int:
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AggregationError(
            f"metadata seed must be an integer in {metadata_path}; got {seed!r}"
        )
    return seed


def _require_episode_count(summary: dict, path: Path) -> int:
    episodes = summary.get("episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise AggregationError(
            f"summary.episodes must be an integer in {path}; got {episodes!r}"
        )
    if episodes < MIN_FINAL_TEST_EPISODES:
        raise AggregationError(
            f"final test in {path} has only {episodes} episodes; "
            f"at least {MIN_FINAL_TEST_EPISODES} are required"
        )
    return episodes


def _require_metrics(summary: dict, path: Path) -> None:
    missing = [metric for metric in METRICS if metric not in summary]
    if missing:
        raise AggregationError(
            f"final-test summary in {path} is missing metrics: {', '.join(missing)}"
        )
    for metric in METRICS:
        value = summary[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AggregationError(
                f"summary.{metric} must be numeric in {path}; got {value!r}"
            )
        if not math.isfinite(float(value)):
            raise AggregationError(
                f"summary.{metric} must be finite in {path}; got {value!r}"
            )


def _canonical(value: object) -> str:
    """Return a stable representation suitable for a protocol fingerprint."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_evaluation_protocol(
    metadata: dict, record: dict, episodes: int, path: Path
) -> dict:
    metadata_protocol = metadata.get("evaluation_protocol")
    record_protocol = record.get("evaluation_protocol")
    if not isinstance(metadata_protocol, dict) or not isinstance(record_protocol, dict):
        raise AggregationError(
            f"final test in {path} must include evaluation_protocol in both "
            "metadata and record"
        )
    if metadata_protocol != record_protocol:
        raise AggregationError(
            f"evaluation_protocol mismatch between metadata and record in {path}"
        )

    required_values = {
        "type": "fixed_explicit_episode_seeds",
        "purpose": "final_test",
        "reproducible_actions": True,
        "episode_count": episodes,
    }
    for field, expected in required_values.items():
        if metadata_protocol.get(field) != expected:
            raise AggregationError(
                f"invalid final evaluation_protocol in {path}: expected "
                f"{field}={expected!r}, got {metadata_protocol.get(field)!r}"
            )

    action_selection_mode = metadata_protocol.get("action_selection_mode")
    deterministic_actions = metadata_protocol.get("deterministic_actions")
    expected_determinism = {
        "deterministic_argmax": True,
        "fixed_policy": True,
        "seeded_random": False,
    }
    if action_selection_mode not in expected_determinism:
        raise AggregationError(
            f"invalid evaluation_protocol.action_selection_mode in {path}: "
            f"{action_selection_mode!r}"
        )
    if deterministic_actions is not expected_determinism[action_selection_mode]:
        raise AggregationError(
            f"evaluation_protocol deterministic_actions is inconsistent with "
            f"action_selection_mode={action_selection_mode!r} in {path}"
        )

    base_seed = metadata_protocol.get("base_seed")
    parallels = metadata_protocol.get("parallels")
    episode_seeds = metadata_protocol.get("episode_seeds")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise AggregationError(
            f"evaluation_protocol.base_seed must be an integer in {path}"
        )
    if (
        isinstance(parallels, bool)
        or not isinstance(parallels, int)
        or not 1 <= parallels <= episodes
    ):
        raise AggregationError(
            f"evaluation_protocol.parallels must be in [1, {episodes}] in {path}"
        )
    expected_seeds = list(range(base_seed, base_seed + episodes))
    if episode_seeds != expected_seeds:
        raise AggregationError(
            f"evaluation_protocol.episode_seeds must be the explicit contiguous "
            f"range [{base_seed}, {base_seed + episodes}) in {path}"
        )
    return metadata_protocol


def _protocol_fingerprint(
    metadata: dict,
    record: dict,
    episodes: int,
    evaluation_protocol: dict,
    path: Path,
) -> tuple[tuple[str, str], ...]:
    """Capture evaluation settings that must agree across final-test runs.

    Training/model hyperparameters and the training seed intentionally do not belong
    here.  The environment/task snapshot and evaluation episode count do.
    """
    config = metadata.get("config")
    if config is not None and not isinstance(config, dict):
        raise AggregationError(f"metadata.config must be an object in {path}")
    config = config or {}

    configured_episodes = config.get("test_episode")
    if configured_episodes is not None and configured_episodes != episodes:
        raise AggregationError(
            f"test protocol mismatch within {path}: metadata config.test_episode="
            f"{configured_episodes!r}, summary.episodes={episodes}"
        )

    metadata_schema = metadata.get("schema_version")
    record_schema = record.get("schema_version")
    if (
        metadata_schema is not None
        and record_schema is not None
        and metadata_schema != record_schema
    ):
        raise AggregationError(
            f"schema mismatch within {path}: metadata={metadata_schema!r}, "
            f"record={record_schema!r}"
        )

    protocol = {
        "schema_version": (
            metadata_schema if metadata_schema is not None else record_schema
        ),
        "env_name": metadata.get("env_name", config.get("env_name")),
        "env_id": metadata.get("env_id", config.get("env_id")),
        "highway_config": config.get("highway_config"),
        "n_agents": config.get("n_agents"),
        "episodes": episodes,
        # Parallel execution width is an implementation detail: explicit seeds
        # and deterministic actions make a sequential sanity policy comparable
        # to a vectorized MAPPO policy.  It is validated per run above but is
        # deliberately absent from the cross-run semantic fingerprint.
        "evaluation_protocol": {
            key: evaluation_protocol[key]
            for key in (
                "type",
                "purpose",
                "reproducible_actions",
                "base_seed",
                "episode_count",
                "episode_seeds",
            )
        },
    }
    return tuple((key, _canonical(value)) for key, value in protocol.items())


def _require_model_provenance(
    *,
    metadata: dict,
    record: dict,
    algorithm: str,
    training_seed: int,
    path: Path,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    metadata_provenance = metadata.get("model_provenance")
    record_provenance = record.get("model_provenance")
    evaluation_protocol = metadata.get("evaluation_protocol")
    expected_action_mode = {
        "random": "seeded_random",
        "idle_only": "fixed_policy",
    }.get(algorithm, "deterministic_argmax")
    if (
        not isinstance(evaluation_protocol, dict)
        or evaluation_protocol.get("action_selection_mode") != expected_action_mode
    ):
        raise AggregationError(
            f"algorithm={algorithm!r} requires action_selection_mode="
            f"{expected_action_mode!r} in {path}"
        )
    if algorithm in _STATIC_BASELINE_ALGORITHMS:
        if metadata_provenance is not None or record_provenance is not None:
            raise AggregationError(
                f"static baseline {algorithm!r} must not claim model provenance in {path}"
            )
        return None, None, None, None, None

    if not isinstance(metadata_provenance, dict) or not isinstance(
        record_provenance, dict
    ):
        raise AggregationError(
            f"learned-policy final test in {path} must include model_provenance "
            "in both metadata and record"
        )
    if metadata_provenance != record_provenance:
        raise AggregationError(
            f"model_provenance mismatch between metadata and record in {path}"
        )
    provenance = metadata_provenance
    expected_values = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "verification": VERIFIED_PROVENANCE,
        "checkpoint_name": BEST_MODEL_FILENAME,
        "training_seed": training_seed,
        "training_env_seed": derive_training_env_seed(training_seed),
        "training_env_seed_scheme": TRAINING_ENV_SEED_SCHEME,
    }
    for field, expected in expected_values.items():
        if provenance.get(field) != expected:
            raise AggregationError(
                f"invalid model_provenance in {path}: expected {field}={expected!r}, "
                f"got {provenance.get(field)!r}"
            )

    checkpoint_hash = provenance.get("checkpoint_sha256")
    training_config_hash = provenance.get("training_config_sha256")
    source_commit = provenance.get("source_commit")
    source_hash = provenance.get("runtime_source_sha256")
    training_device = provenance.get("training_device")
    checkpoint_path = Path(str(provenance.get("checkpoint_path", "")))
    if checkpoint_path.name != BEST_MODEL_FILENAME or not checkpoint_path.is_file():
        raise AggregationError(
            f"retained best-model checkpoint is missing for {path}: {checkpoint_path}"
        )
    sidecar_value = provenance.get("provenance_sidecar")
    if not isinstance(sidecar_value, str) or not sidecar_value:
        raise AggregationError(
            f"model_provenance.provenance_sidecar must be retained in {path}"
        )
    sidecar_path = Path(sidecar_value).expanduser().resolve()
    expected_sidecar_path = provenance_path_for_model(checkpoint_path).resolve()
    if sidecar_path != expected_sidecar_path or not sidecar_path.is_file():
        raise AggregationError(
            f"retained provenance sidecar is missing or misplaced for {path}: "
            f"{sidecar_path}"
        )
    try:
        sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AggregationError(
            f"invalid retained provenance sidecar for {path}: {error}"
        ) from error
    if not isinstance(sidecar_payload, dict):
        raise AggregationError(f"retained provenance sidecar must be an object: {path}")
    embedded_payload = dict(provenance)
    embedded_payload.pop("provenance_sidecar", None)
    if embedded_payload != sidecar_payload:
        raise AggregationError(
            f"embedded model_provenance does not match its retained sidecar in {path}"
        )

    final_config = metadata.get("config")
    if not isinstance(final_config, dict):
        raise AggregationError(f"metadata.config must be an object in {path}")
    try:
        validate_model_provenance_payload(
            sidecar_payload,
            checkpoint_path=checkpoint_path,
            expected_training_seed=training_seed,
            expected_configs=Namespace(**final_config),
            verify_current_source=False,
            require_clean_source=True,
        )
    except (OSError, TypeError, ValueError) as error:
        raise AggregationError(
            f"invalid retained model provenance in {path}: {error}"
        ) from error
    return (
        str(checkpoint_hash),
        str(training_config_hash),
        str(source_commit),
        str(source_hash),
        str(training_device),
    )


def _validate_episode_payload(
    *,
    record: dict,
    summary: dict,
    evaluation_protocol: dict,
    episodes: int,
    path: Path,
) -> None:
    episode_records = record.get("episodes")
    if not isinstance(episode_records, list) or len(episode_records) != episodes:
        actual = len(episode_records) if isinstance(episode_records, list) else None
        raise AggregationError(
            f"episode payload count mismatch in {path}: "
            f"summary={episodes}, payload={actual!r}"
        )

    observed_episode_seeds = []
    observed_scores = []
    for index, episode_record in enumerate(episode_records):
        if not isinstance(episode_record, dict):
            raise AggregationError(
                f"episode payload {index} must be an object in {path}"
            )
        if episode_record.get("episode_index") != index:
            raise AggregationError(
                f"episode payload indices must be contiguous in {path}; "
                f"position {index} has {episode_record.get('episode_index')!r}"
            )
        episode_seed = episode_record.get("seed")
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise AggregationError(
                f"episode payload {index} is missing an integer seed in {path}"
            )
        score = episode_record.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise AggregationError(
                f"episode payload {index} is missing a finite score in {path}"
            )
        episode_reward = episode_record.get("episode_reward")
        agent_rewards = episode_record.get("agent_rewards")
        if (
            isinstance(episode_reward, bool)
            or not isinstance(episode_reward, (int, float))
            or not math.isfinite(float(episode_reward))
            or not isinstance(agent_rewards, dict)
            or not agent_rewards
        ):
            raise AggregationError(
                f"episode payload {index} has invalid reward data in {path}"
            )
        reward_values = list(agent_rewards.values())
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in reward_values
        ):
            raise AggregationError(
                f"episode payload {index} has non-finite agent rewards in {path}"
            )
        mean_agent_reward = math.fsum(float(value) for value in reward_values) / len(
            reward_values
        )
        if not math.isclose(
            float(score), float(episode_reward), rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(
            float(episode_reward),
            mean_agent_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AggregationError(
                f"episode payload {index} score/reward/agent mean disagree in {path}"
            )
        observed_episode_seeds.append(episode_seed)
        observed_scores.append(float(score))

    if observed_episode_seeds != evaluation_protocol["episode_seeds"]:
        raise AggregationError(
            f"episode payload seeds do not match evaluation_protocol in {path}"
        )

    scores = record.get("scores")
    if not isinstance(scores, list) or len(scores) != episodes:
        actual = len(scores) if isinstance(scores, list) else None
        raise AggregationError(
            f"score count mismatch in {path}: summary={episodes}, scores={actual!r}"
        )
    for index, (stored_score, episode_score) in enumerate(zip(scores, observed_scores)):
        if (
            isinstance(stored_score, bool)
            or not isinstance(stored_score, (int, float))
            or not math.isfinite(float(stored_score))
            or not math.isclose(
                float(stored_score), episode_score, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise AggregationError(
                f"top-level score {index} does not match episode score in {path}"
            )

    try:
        recomputed = summarize_episode_records(episode_records)
    except (KeyError, TypeError, ValueError) as error:
        raise AggregationError(
            f"cannot recompute final summary from episode payload in {path}: {error}"
        ) from error
    for field in ("episodes", *METRICS):
        if field not in recomputed:
            raise AggregationError(
                f"episode payload cannot reproduce summary.{field} in {path}"
            )
        stored_value = summary[field]
        recomputed_value = recomputed[field]
        if field == "episodes":
            matches = stored_value == recomputed_value
        else:
            matches = math.isclose(
                float(stored_value),
                float(recomputed_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        if not matches:
            raise AggregationError(
                f"stored summary.{field} does not match episode payload in {path}: "
                f"stored={stored_value}, recomputed={recomputed_value}"
            )


def _load_final_result(path: str | Path) -> _FinalResult:
    records_path = Path(path)
    algorithm = algo_from_path(records_path)
    if algorithm == "unknown":
        raise AggregationError(
            f"cannot derive algorithm from {records_path}; expected a parent named "
            "<algorithm>_highway"
        )

    metadata_path = records_path.parent / "eval_metadata.json"
    metadata = _load_json(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise AggregationError(
            f"metadata schema_version must be {SCHEMA_VERSION} in {metadata_path}"
        )
    if metadata.get("mode") != "test":
        raise AggregationError(
            f"not a test run: expected metadata mode='test' in {metadata_path}, "
            f"got {metadata.get('mode')!r}"
        )
    seed = _require_seed(metadata, metadata_path)

    try:
        records = load_records(records_path)
    except FileNotFoundError as error:
        raise AggregationError(
            f"eval records file does not exist: {records_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise AggregationError(f"invalid JSONL in {records_path}: {error}") from error
    if len(records) != 1:
        raise AggregationError(
            f"final-test artifact must contain exactly one record: {records_path} "
            f"contains {len(records)}"
        )

    record = records[0]
    if not isinstance(record, dict):
        raise AggregationError(f"eval record must be a JSON object: {records_path}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise AggregationError(
            f"record schema_version must be {SCHEMA_VERSION} in {records_path}"
        )
    required_identity = {
        "mode": "test",
        "phase": "test",
        "epoch": None,
    }
    for field, expected in required_identity.items():
        if field not in record or record[field] != expected:
            raise AggregationError(
                f"not a final test record in {records_path}: expected "
                f"{field}={expected!r}, got {record.get(field)!r}"
            )
    if record.get("is_best") is not True or record.get("is_initial_eval") is not False:
        raise AggregationError(
            f"final test record must have is_best=true and is_initial_eval=false in "
            f"{records_path}"
        )

    summary = record.get("summary")
    if not isinstance(summary, dict):
        raise AggregationError(
            f"final test record has no summary object: {records_path}"
        )
    episodes = _require_episode_count(summary, records_path)
    _require_metrics(summary, records_path)
    evaluation_protocol = _require_evaluation_protocol(
        metadata, record, episodes, records_path
    )
    protocol_fingerprint = _protocol_fingerprint(
        metadata,
        record,
        episodes,
        evaluation_protocol,
        records_path,
    )

    _validate_episode_payload(
        record=record,
        summary=summary,
        evaluation_protocol=evaluation_protocol,
        episodes=episodes,
        path=records_path,
    )
    (
        checkpoint_hash,
        training_config_hash,
        source_commit,
        source_hash,
        training_device,
    ) = _require_model_provenance(
        metadata=metadata,
        record=record,
        algorithm=algorithm,
        training_seed=seed,
        path=records_path,
    )

    return _FinalResult(
        path=records_path,
        algorithm=algorithm,
        seed=seed,
        summary=summary,
        protocol=protocol_fingerprint,
        checkpoint_sha256=checkpoint_hash,
        training_config_sha256=training_config_hash,
        source_commit=source_commit,
        runtime_source_sha256=source_hash,
        training_device=training_device,
    )


def _normalize_expected_seeds(expected_seeds: Iterable[int] | None) -> set[int] | None:
    if expected_seeds is None:
        return None
    values = list(expected_seeds)
    if not values:
        raise AggregationError("expected seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values):
        raise AggregationError("expected seeds must all be integers")
    duplicates = sorted(seed for seed in set(values) if values.count(seed) > 1)
    if duplicates:
        raise AggregationError(f"duplicate expected seeds: {duplicates}")
    return set(values)


def _describe_protocol_difference(
    expected: tuple[tuple[str, str], ...], actual: tuple[tuple[str, str], ...]
) -> str:
    expected_map = dict(expected)
    actual_map = dict(actual)
    differences = [
        f"{key}: {expected_map.get(key)} != {actual_map.get(key)}"
        for key in expected_map.keys() | actual_map.keys()
        if expected_map.get(key) != actual_map.get(key)
    ]
    return "; ".join(sorted(differences))


def aggregate(
    jsonl_paths: Iterable[str | Path],
    *,
    expected_seeds: Iterable[int] | None = None,
    expected_seeds_by_algorithm: Mapping[str, Iterable[int]] | None = None,
) -> list[dict]:
    """Aggregate one independently evaluated final result per algorithm/seed.

    ``expected_seeds`` applies to every algorithm present.  Each algorithm must have
    exactly that seed set; missing and unexpected seeds are errors, never omissions.
    """
    rows, _per_seed_rows = aggregate_with_per_seed(
        jsonl_paths,
        expected_seeds=expected_seeds,
        expected_seeds_by_algorithm=expected_seeds_by_algorithm,
    )
    return rows


def aggregate_with_per_seed(
    jsonl_paths: Iterable[str | Path],
    *,
    expected_seeds: Iterable[int] | None = None,
    expected_seeds_by_algorithm: Mapping[str, Iterable[int]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return strict aggregate rows and their validated per-seed observations."""
    by_algo = _collect_final_results(
        jsonl_paths,
        expected_seeds=expected_seeds,
        expected_seeds_by_algorithm=expected_seeds_by_algorithm,
    )

    rows: list[dict] = []
    per_seed_rows: list[dict] = []
    for algorithm in sorted(by_algo):
        results = sorted(by_algo[algorithm], key=lambda result: result.seed)
        row: dict[str, object] = {"algorithm": algorithm, "seeds": len(results)}
        for result in results:
            per_seed_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": result.seed,
                    "checkpoint_sha256": result.checkpoint_sha256,
                    "source_commit": result.source_commit,
                    "runtime_source_sha256": result.runtime_source_sha256,
                    "training_device": result.training_device,
                    **{metric: float(result.summary[metric]) for metric in METRICS},
                }
            )
        for metric in METRICS:
            values = [float(result.summary[metric]) for result in results]
            row[f"{metric}_mean"] = statistics.mean(values)
            if algorithm in _STATIC_BASELINE_ALGORITHMS:
                row[f"{metric}_std"] = None
            else:
                row[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        rows.append(row)
    return rows, per_seed_rows


def _collect_final_results(
    jsonl_paths: Iterable[str | Path],
    *,
    expected_seeds: Iterable[int] | None,
    expected_seeds_by_algorithm: Mapping[str, Iterable[int]] | None,
) -> dict[str, list[_FinalResult]]:
    if expected_seeds is not None and expected_seeds_by_algorithm is not None:
        raise AggregationError(
            "use either global expected_seeds or expected_seeds_by_algorithm"
        )
    required_seeds = _normalize_expected_seeds(expected_seeds)
    required_by_algorithm = None
    if expected_seeds_by_algorithm is not None:
        required_by_algorithm = {
            str(algorithm): _normalize_expected_seeds(seeds)
            for algorithm, seeds in expected_seeds_by_algorithm.items()
        }
    by_key: dict[tuple[str, int], _FinalResult] = {}
    reference: _FinalResult | None = None
    learned_contracts: dict[str, tuple[str, str]] = {}
    learned_source: tuple[str, str] | None = None
    learned_checkpoint_owners: dict[tuple[str, str], _FinalResult] = {}

    for path in jsonl_paths:
        result = _load_final_result(path)
        key = (result.algorithm, result.seed)
        if key in by_key:
            previous = by_key[key]
            raise AggregationError(
                f"duplicate final result for algorithm={result.algorithm!r}, "
                f"seed={result.seed}: {previous.path} and {result.path}"
            )
        if reference is None:
            reference = result
        elif result.protocol != reference.protocol:
            difference = _describe_protocol_difference(
                reference.protocol, result.protocol
            )
            raise AggregationError(
                f"inconsistent final-test protocol between {reference.path} and "
                f"{result.path}: {difference}"
            )
        if result.checkpoint_sha256 is not None:
            contract = (
                str(result.training_config_sha256),
                str(result.training_device),
            )
            previous_contract = learned_contracts.get(result.algorithm)
            if previous_contract is not None and contract != previous_contract:
                raise AggregationError(
                    f"learned seeds for algorithm={result.algorithm!r} do not share "
                    "the same training config and device"
                )
            learned_contracts[result.algorithm] = contract

            source = (
                str(result.source_commit),
                str(result.runtime_source_sha256),
            )
            if learned_source is not None and source != learned_source:
                raise AggregationError(
                    "learned-policy artifacts do not share the same source commit "
                    "and runtime-source manifest"
                )
            learned_source = source

            checkpoint_key = (result.algorithm, result.checkpoint_sha256)
            previous_checkpoint = learned_checkpoint_owners.get(checkpoint_key)
            if (
                previous_checkpoint is not None
                and previous_checkpoint.seed != result.seed
            ):
                raise AggregationError(
                    f"same checkpoint reused for multiple training seeds in "
                    f"algorithm={result.algorithm!r}: {previous_checkpoint.path} and "
                    f"{result.path}"
                )
            learned_checkpoint_owners[checkpoint_key] = result
        by_key[key] = result

    by_algo: dict[str, list[_FinalResult]] = {}
    for (algorithm, _seed), result in by_key.items():
        by_algo.setdefault(algorithm, []).append(result)

    if required_seeds is not None:
        for algorithm, results in sorted(by_algo.items()):
            actual = {result.seed for result in results}
            if actual != required_seeds:
                missing = sorted(required_seeds - actual)
                unexpected = sorted(actual - required_seeds)
                raise AggregationError(
                    f"seed set mismatch for algorithm={algorithm!r}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
    if required_by_algorithm is not None:
        actual_algorithms = set(by_algo)
        expected_algorithms = set(required_by_algorithm)
        if actual_algorithms != expected_algorithms:
            raise AggregationError(
                "algorithm set mismatch: "
                f"missing={sorted(expected_algorithms - actual_algorithms)}, "
                f"unexpected={sorted(actual_algorithms - expected_algorithms)}"
            )
        for algorithm, results in sorted(by_algo.items()):
            expected = required_by_algorithm[algorithm]
            actual = {result.seed for result in results}
            if actual != expected:
                raise AggregationError(
                    f"seed set mismatch for algorithm={algorithm!r}: "
                    f"missing={sorted(expected - actual)}, "
                    f"unexpected={sorted(actual - expected)}"
                )

    return by_algo


def load_runs_manifest(path: str | Path) -> tuple[list[str], dict[str, list[int]]]:
    manifest_path = Path(path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise AggregationError(
            f"runs manifest schema_version must be 1: {manifest_path}"
        )
    artifacts = manifest.get("eval_records")
    expected = manifest.get("expected_seeds")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or not all(isinstance(item, str) and item for item in artifacts)
    ):
        raise AggregationError(
            f"runs manifest eval_records must be a non-empty string list: {manifest_path}"
        )
    if not isinstance(expected, dict) or not expected:
        raise AggregationError(
            f"runs manifest expected_seeds must be a non-empty object: {manifest_path}"
        )
    expected_seeds: dict[str, list[int]] = {}
    for algorithm, seeds in expected.items():
        if not isinstance(algorithm, str) or not isinstance(seeds, list):
            raise AggregationError(
                f"invalid expected_seeds entry in runs manifest: {algorithm!r}"
            )
        normalized = _normalize_expected_seeds(seeds)
        expected_seeds[algorithm] = sorted(normalized or set())
    return artifacts, expected_seeds


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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--runs-glob",
        help="glob for final test run-root eval_records.jsonl files",
    )
    input_group.add_argument(
        "--runs-manifest",
        help=(
            "JSON manifest with explicit eval_records paths and per-algorithm "
            "expected_seeds; preferred for final tables"
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "exact training seed set required for every algorithm (for example: 1 2 3)"
        ),
    )
    parser.add_argument("--output", default=None, help="output CSV path")
    args = parser.parse_args(argv)

    expected_by_algorithm = None
    if args.runs_manifest:
        if args.expected_seeds is not None:
            parser.error("--expected-seeds cannot be combined with --runs-manifest")
        try:
            paths, expected_by_algorithm = load_runs_manifest(args.runs_manifest)
        except (AggregationError, OSError) as error:
            print(f"Manifest failed: {error}", file=sys.stderr)
            return 1
    else:
        paths = sorted(glob.glob(args.runs_glob, recursive=True))
    if not paths:
        print(f"No final-test artifacts selected: {args.runs_glob}", file=sys.stderr)
        return 1
    try:
        rows = aggregate(
            paths,
            expected_seeds=args.expected_seeds,
            expected_seeds_by_algorithm=expected_by_algorithm,
        )
    except (AggregationError, OSError) as error:
        print(f"Aggregation failed: {error}", file=sys.stderr)
        return 1
    if not rows:
        print("No final test records found.", file=sys.stderr)
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

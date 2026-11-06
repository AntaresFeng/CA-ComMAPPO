import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def episode_outcome(
    crashed: Sequence[bool],
    arrived: Sequence[bool],
    truncated: bool,
) -> dict[str, bool]:
    collision = any(crashed)
    arrival = bool(arrived) and all(arrived) and not collision
    timeout = bool(truncated) and not collision and not arrival
    return {
        "collision": collision,
        "arrival": arrival,
        "truncated": timeout,
    }


def build_episode_record(
    *,
    episode_index: int,
    steps: int,
    agent_rewards: dict[str, float],
    crashed_agents: Sequence[bool],
    arrived_agents: Sequence[bool],
    truncated: bool,
    phase: str | None = None,
    policy: str | None = None,
    seed: int | None = None,
    env_id: int | None = None,
    score: float | None = None,
    waiting_steps: Sequence[float] | None = None,
    dt: float | None = None,
    arrival_times: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    rewards = {agent: float(reward) for agent, reward in agent_rewards.items()}
    num_agents = len(rewards)
    if num_agents <= 0:
        raise ValueError("agent_rewards must contain at least one agent")

    crashed = [bool(value) for value in crashed_agents]
    arrived = [bool(value) for value in arrived_agents]
    outcome = episode_outcome(crashed, arrived, truncated)

    record: dict[str, Any] = {}
    if phase is not None:
        record["phase"] = phase
    if policy is not None:
        record["policy"] = policy
    if seed is not None:
        record["seed"] = int(seed)
    if env_id is not None:
        record["env_id"] = int(env_id)
    record.update(
        {
            "episode_index": int(episode_index),
            "steps": int(steps),
            "episode_reward": float(sum(rewards.values()) / num_agents),
        }
    )
    if score is not None:
        record["score"] = float(score)
    record.update(
        {
            "agent_rewards": rewards,
            **outcome,
            "crashed_agents": crashed,
            "arrived_agents": arrived,
            "agent_collision_fraction": _mean_bool(crashed),
            "agent_arrival_fraction": _mean_bool(arrived),
        }
    )

    # These efficiency metrics do not enter the current best-model sort key.
    if dt is not None:
        travel_time = float(int(steps) * float(dt))
        duration = travel_time
        arrived_count = sum(1 for value in arrived if value)
        throughput = float(arrived_count / duration) if duration > 0 else 0.0
        # Historical artifacts use this name for episode duration. Actual per-car
        # completion times are recorded separately to avoid reinterpreting old data.
        record["travel_time"] = travel_time
        record["episode_duration"] = duration
        record["throughput"] = throughput
        if waiting_steps is not None:
            waiting = [float(value) for value in waiting_steps]
            mean_waiting_steps = sum(waiting) / len(waiting) if waiting else 0.0
            record["waiting_time"] = float(mean_waiting_steps * float(dt))
    if arrival_times is not None:
        if len(arrival_times) != num_agents or len(arrived) != num_agents:
            raise ValueError("arrival_times must contain one value per agent")
        times = []
        for reached, value in zip(arrived, arrival_times, strict=True):
            if reached != (value is not None):
                raise ValueError(
                    "arrival_times must be present exactly for arrived agents"
                )
            if value is not None:
                value = float(value)
                if (
                    not math.isfinite(value)
                    or value < 0
                    or (dt is not None and value > steps * dt + 1e-9)
                ):
                    raise ValueError("arrival time must lie within the episode")
            times.append(value)
        record["arrival_times"] = times
        completed = [value for value in times if value is not None]
        record["completed_travel_time"] = _mean(completed) if completed else None
    return record


def summarize_episode_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("records must not be empty")

    episode_rewards = [float(record["episode_reward"]) for record in records]
    episode_lengths = [float(record["steps"]) for record in records]
    all_agent_rewards = [
        float(reward)
        for record in records
        for reward in record["agent_rewards"].values()
    ]

    summary: dict[str, float | int] = {
        "episodes": len(records),
        "mean_episode_reward": _mean(episode_rewards),
        "mean_agent_reward": _mean(all_agent_rewards),
        "mean_episode_length": _mean(episode_lengths),
        "collision_rate": _mean_bool(record["collision"] for record in records),
        "arrival_rate": _mean_bool(record["arrival"] for record in records),
        "truncation_rate": _mean_bool(record["truncated"] for record in records),
    }
    for key in ("agent_collision_fraction", "agent_arrival_fraction"):
        if all(key in record for record in records):
            summary[f"mean_{key}"] = _mean(float(record[key]) for record in records)
    for key in ("travel_time", "episode_duration", "throughput", "waiting_time"):
        if all(key in record for record in records):
            summary[f"mean_{key}"] = _mean(float(record[key]) for record in records)
    if all("arrival_times" in record for record in records):
        completed = [
            value
            for record in records
            for value in record["arrival_times"]
            if value is not None
        ]
        summary["completed_agents"] = len(completed)
        if completed:
            summary["mean_completed_travel_time"] = _mean(completed)
    return summary


def summary_to_log_infos(
    prefix: str,
    summary: dict[str, float | int],
) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in summary.items()}


def format_summary_lines(
    summary: dict[str, float | int],
    *,
    indent: str = "",
) -> list[str]:
    ordered_keys = [
        "episodes",
        "mean_episode_reward",
        "mean_agent_reward",
        "mean_episode_length",
        "collision_rate",
        "arrival_rate",
        "truncation_rate",
        "mean_agent_collision_fraction",
        "mean_agent_arrival_fraction",
        "mean_travel_time",
        "mean_episode_duration",
        "completed_agents",
        "mean_completed_travel_time",
        "mean_throughput",
        "mean_waiting_time",
    ]
    lines = []
    for key in ordered_keys:
        if key not in summary:
            continue
        value = summary[key]
        if key == "episodes":
            lines.append(f"{indent}{key}={int(value)}")
        else:
            lines.append(f"{indent}{key}={float(value):.3f}")
    return lines


def print_highway_summary(
    summary: dict[str, float | int], *, indent: str = "  "
) -> None:
    for line in format_summary_lines(summary, indent=indent):
        print(line)


def save_results_json(results: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot compute mean of empty values")
    return float(sum(items) / len(items))


def _mean_bool(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return float(sum(1 for value in items if value) / len(items))

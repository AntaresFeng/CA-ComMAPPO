import json
from pathlib import Path
from typing import Any, Iterable, Sequence


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
    score: float | None = None,
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

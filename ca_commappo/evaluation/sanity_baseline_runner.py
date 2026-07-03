import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces

from ca_commappo.envs.highway_intersection_wrapper import (
    DEFAULT_HIGHWAY_ENV_ID,
    IDLE_ACTION,
    HighwayIntersectionMultiAgentEnv,
)


SUPPORTED_POLICIES = ("random", "idle-only")


@dataclass(frozen=True)
class SanityConfig:
    env_id: str
    env_seed: int
    episodes: int
    seeds: list[int]
    policies: list[str]
    highway_config: dict[str, Any]


def load_sanity_config(path: str | Path) -> SanityConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    episodes = int(raw_config.get("episodes", 1))
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer")

    env_seed = int(raw_config.get("env_seed", 1))
    seeds = [int(seed) for seed in raw_config.get("seeds", [])]
    if not seeds:
        seeds = [env_seed]

    policies = list(raw_config.get("policies", SUPPORTED_POLICIES))
    _validate_policies(policies)

    return SanityConfig(
        env_id=str(raw_config.get("env_id", DEFAULT_HIGHWAY_ENV_ID)),
        env_seed=env_seed,
        episodes=episodes,
        seeds=seeds,
        policies=policies,
        highway_config=dict(raw_config.get("highway_config", {}) or {}),
    )


def select_actions(env: Any, policy: str, rng: np.random.Generator) -> dict[str, int]:
    if policy == "idle-only":
        return {agent: IDLE_ACTION for agent in env.agents}
    if policy == "random":
        actions = {}
        for agent in env.agents:
            action_space = env.action_space[agent]
            if not isinstance(action_space, spaces.Discrete):
                raise TypeError("random sanity policy requires Discrete action spaces")
            actions[agent] = int(rng.integers(action_space.n))
        return actions
    raise ValueError(_unknown_policy_message(policy))


def run_episode(
    config: SanityConfig,
    policy: str,
    seed: int,
    episode_index: int,
) -> dict[str, Any]:
    _validate_policies([policy])
    rng = np.random.default_rng(seed)
    env = _make_env(config)
    try:
        env.reset(seed=seed)
        steps = 0
        agent_rewards = {agent: 0.0 for agent in env.agents}
        truncated = False

        while True:
            actions = select_actions(env, policy, rng)
            _obs, rewards, _terminated, truncated, _info = env.step(actions)
            steps += 1

            for agent, reward in rewards.items():
                agent_rewards[agent] += float(reward)

            crashed_agents, arrived_agents = _controlled_vehicle_flags(env)
            collision = any(crashed_agents)
            all_arrived = all(arrived_agents)
            if collision or all_arrived or truncated:
                break

        collision = any(crashed_agents)
        arrival = all(arrived_agents) and not collision
        timeout = bool(truncated) and not collision and not arrival
        num_agents = len(env.agents)

        return {
            "policy": policy,
            "seed": seed,
            "episode_index": episode_index,
            "steps": steps,
            "episode_reward": float(sum(agent_rewards.values()) / num_agents),
            "agent_rewards": agent_rewards,
            "collision": collision,
            "arrival": arrival,
            "truncated": timeout,
            "crashed_agents": crashed_agents,
            "arrived_agents": arrived_agents,
            "agent_collision_fraction": float(np.mean(crashed_agents)),
            "agent_arrival_fraction": float(np.mean(arrived_agents)),
        }
    finally:
        env.close()


def run_sanity_baseline(
    config: SanityConfig,
    policy: str = "all",
) -> dict[str, Any]:
    policies = _resolve_requested_policies(policy, config.policies)
    results = {
        "config": {
            "env_id": config.env_id,
            "episodes": config.episodes,
            "seeds": config.seeds,
        },
        "policies": {},
    }

    for policy_name in policies:
        records = []
        for seed_index, base_seed in enumerate(config.seeds):
            for episode_index in range(config.episodes):
                episode_seed = base_seed + episode_index * len(config.seeds)
                record = run_episode(
                    config,
                    policy_name,
                    seed=episode_seed,
                    episode_index=seed_index * config.episodes + episode_index,
                )
                records.append(record)
        results["policies"][policy_name] = {
            "episodes": records,
            "summary": summarize_episode_records(records),
        }

    return results


def summarize_episode_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("records must not be empty")

    episode_rewards = np.array([record["episode_reward"] for record in records])
    episode_lengths = np.array([record["steps"] for record in records])
    all_agent_rewards = [
        reward for record in records for reward in record["agent_rewards"].values()
    ]

    return {
        "episodes": len(records),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_agent_reward": float(np.mean(all_agent_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "collision_rate": float(np.mean([record["collision"] for record in records])),
        "arrival_rate": float(np.mean([record["arrival"] for record in records])),
        "truncation_rate": float(np.mean([record["truncated"] for record in records])),
    }


def save_results_json(results: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _make_env(config: SanityConfig) -> HighwayIntersectionMultiAgentEnv:
    return HighwayIntersectionMultiAgentEnv(
        Namespace(env_id=config.env_id, highway_config=config.highway_config)
    )


def _controlled_vehicle_flags(
    env: HighwayIntersectionMultiAgentEnv,
) -> tuple[list[bool], list[bool]]:
    base_env = env.env.unwrapped
    crashed = [bool(vehicle.crashed) for vehicle in base_env.controlled_vehicles]
    arrived = [
        bool(base_env.has_arrived(vehicle)) for vehicle in base_env.controlled_vehicles
    ]
    return crashed, arrived


def _resolve_requested_policies(
    policy: str,
    configured_policies: list[str],
) -> list[str]:
    if policy == "all":
        _validate_policies(configured_policies)
        return configured_policies
    _validate_policies([policy])
    return [policy]


def _validate_policies(policies: list[str]) -> None:
    for policy in policies:
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(_unknown_policy_message(policy))


def _unknown_policy_message(policy: str) -> str:
    return (
        f"Unknown policy {policy!r}; expected one of "
        f"{', '.join(SUPPORTED_POLICIES)} or 'all'."
    )

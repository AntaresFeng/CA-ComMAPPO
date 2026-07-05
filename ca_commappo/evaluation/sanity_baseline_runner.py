from argparse import Namespace
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np
import yaml
from gymnasium import spaces

from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    summarize_episode_records,
)
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
            _obs, rewards, _terminated, truncated, info = env.step(actions)
            steps += 1

            for agent, reward in rewards.items():
                agent_rewards[agent] += float(reward)

            crashed_agents = info["crashed"]
            arrived_agents = info["arrived"]
            collision = any(crashed_agents)
            all_arrived = all(arrived_agents)
            if collision or all_arrived or truncated:
                break

        return build_episode_record(
            policy=policy,
            seed=seed,
            episode_index=episode_index,
            steps=steps,
            agent_rewards=agent_rewards,
            crashed_agents=crashed_agents,
            arrived_agents=arrived_agents,
            truncated=truncated,
        )
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


def _make_env(config: SanityConfig) -> HighwayIntersectionMultiAgentEnv:
    return HighwayIntersectionMultiAgentEnv(
        Namespace(env_id=config.env_id, highway_config=config.highway_config)
    )


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

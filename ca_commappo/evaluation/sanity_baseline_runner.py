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
from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    summarize_episode_records,
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
    env_name: str = "HighwayIntersection"
    episode_seed_base: int | None = None
    episode_count: int | None = None
    eval_parallels: int = 1
    run_seed: int = 1
    config_path: str = ""


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

    episode_seed_base = _optional_int(raw_config, "episode_seed_base")
    episode_count = _optional_int(raw_config, "episode_count")
    if (episode_seed_base is None) != (episode_count is None):
        raise ValueError(
            "episode_seed_base and episode_count must either both be set or both "
            "be omitted"
        )
    if episode_count is not None and episode_count <= 0:
        raise ValueError("episode_count must be a positive integer")

    eval_parallels = int(raw_config.get("eval_parallels", 1))
    if eval_parallels != 1:
        raise ValueError(
            "sanity baseline currently executes sequentially; eval_parallels must be 1"
        )

    highway_config = _load_highway_config(raw_config, config_path)

    return SanityConfig(
        env_id=str(raw_config.get("env_id", DEFAULT_HIGHWAY_ENV_ID)),
        env_name=str(raw_config.get("env_name", "HighwayIntersection")),
        env_seed=env_seed,
        episodes=episodes,
        seeds=seeds,
        policies=policies,
        highway_config=highway_config,
        episode_seed_base=episode_seed_base,
        episode_count=episode_count,
        eval_parallels=eval_parallels,
        run_seed=int(raw_config.get("run_seed", 1)),
        config_path=str(config_path),
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


def make_action_rng(episode_seed: int) -> np.random.Generator:
    """Create an action RNG stream distinct from the environment seed stream."""
    action_seed_sequence = np.random.SeedSequence(int(episode_seed)).spawn(1)[0]
    return np.random.default_rng(action_seed_sequence)


def run_episode(
    config: SanityConfig,
    policy: str,
    seed: int,
    episode_index: int,
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    _validate_policies([policy])
    rng = make_action_rng(seed)
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
            phase=phase,
            policy=policy,
            seed=seed,
            episode_index=episode_index,
            steps=steps,
            agent_rewards=agent_rewards,
            crashed_agents=crashed_agents,
            arrived_agents=arrived_agents,
            truncated=truncated,
            score=float(sum(agent_rewards.values()) / len(agent_rewards)),
            waiting_steps=info.get("waiting_steps"),
            dt=info.get("dt"),
            arrival_times=info.get("arrival_times"),
        )
    finally:
        env.close()


def run_sanity_baseline(
    config: SanityConfig,
    policy: str = "all",
) -> dict[str, Any]:
    policies = _resolve_requested_policies(policy, config.policies)
    fixed_episode_seeds = strict_episode_seeds(config)
    results = {
        "config": {
            "env_name": config.env_name,
            "env_id": config.env_id,
            "episodes": config.episodes,
            "seeds": config.seeds,
            "highway_config": config.highway_config,
        },
        "policies": {},
    }
    if fixed_episode_seeds is not None:
        results["evaluation_protocols"] = {}
        results["config"].update(
            {
                "run_seed": config.run_seed,
                "episode_seed_base": config.episode_seed_base,
                "episode_count": config.episode_count,
                "episode_seeds": fixed_episode_seeds,
            }
        )

    for policy_name in policies:
        records = []
        if fixed_episode_seeds is not None:
            for episode_index, episode_seed in enumerate(fixed_episode_seeds):
                record = run_episode(
                    config,
                    policy_name,
                    seed=episode_seed,
                    episode_index=episode_index,
                    phase="test",
                )
                records.append(record)
        else:
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
        policy_results = {
            "episodes": records,
            "summary": summarize_episode_records(records),
        }
        if fixed_episode_seeds is not None:
            protocol = strict_evaluation_protocol(config, policy_name)
            policy_results["evaluation_protocol"] = protocol
            results["evaluation_protocols"][policy_name] = protocol
        results["policies"][policy_name] = policy_results

    return results


def strict_episode_seeds(config: SanityConfig) -> list[int] | None:
    """Return the explicit, ordered final-test cases, or ``None`` for legacy mode."""
    if config.episode_seed_base is None or config.episode_count is None:
        return None
    return list(
        range(config.episode_seed_base, config.episode_seed_base + config.episode_count)
    )


def strict_evaluation_protocol(
    config: SanityConfig, policy: str
) -> dict[str, Any] | None:
    """Describe task cases and truthful action semantics for one policy."""
    _validate_policies([policy])
    episode_seeds = strict_episode_seeds(config)
    if episode_seeds is None:
        return None
    is_fixed_policy = policy == "idle-only"
    return {
        "type": "fixed_explicit_episode_seeds",
        "purpose": "final_test",
        "deterministic_actions": is_fixed_policy,
        "reproducible_actions": True,
        "action_selection_mode": (
            "fixed_policy" if is_fixed_policy else "seeded_random"
        ),
        "base_seed": config.episode_seed_base,
        "episode_count": config.episode_count,
        "parallels": config.eval_parallels,
        "episode_seeds": episode_seeds,
    }


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


def _optional_int(raw_config: dict[str, Any], key: str) -> int | None:
    value = raw_config.get(key)
    return None if value is None else int(value)


def _load_highway_config(
    raw_config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    inline_config = dict(raw_config.get("highway_config", {}) or {})
    referenced_path = raw_config.get("highway_config_path")
    if referenced_path is None:
        return inline_config

    highway_path = Path(referenced_path)
    if not highway_path.is_absolute():
        highway_path = config_path.parent / highway_path
    with highway_path.open("r", encoding="utf-8") as config_file:
        referenced_config = yaml.safe_load(config_file) or {}
    if not isinstance(referenced_config, dict):
        raise TypeError("highway_config_path must contain a YAML mapping")
    return _deep_merge(referenced_config, inline_config)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

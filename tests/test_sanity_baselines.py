import math

import numpy as np
from gymnasium import spaces

from ca_commappo.evaluation.sanity_baselines import (
    SanityConfig,
    load_sanity_config,
    run_episode,
    run_sanity_baseline,
    select_actions,
    summarize_episode_records,
)


class DummyEnv:
    agents = ["agent_0", "agent_1", "agent_2"]
    action_space = {agent: spaces.Discrete(3) for agent in agents}


def test_load_sanity_config_from_yaml(tmp_path):
    config_path = tmp_path / "sanity.yaml"
    config_path.write_text(
        """
env_id: "intersection-v1"
env_seed: 7
episodes: 2
seeds: [7, 8]
policies: ["random", "idle-only"]
highway_config:
  controlled_vehicles: 3
  duration: 13
""".strip(),
        encoding="utf-8",
    )

    config = load_sanity_config(config_path)

    assert config.env_id == "intersection-v1"
    assert config.env_seed == 7
    assert config.episodes == 2
    assert config.seeds == [7, 8]
    assert config.policies == ["random", "idle-only"]
    assert config.highway_config == {"controlled_vehicles": 3, "duration": 13}


def test_idle_only_policy_returns_idle_for_every_agent():
    actions = select_actions(DummyEnv(), "idle-only", np.random.default_rng(1))

    assert actions == {"agent_0": 1, "agent_1": 1, "agent_2": 1}


def test_random_policy_is_reproducible_with_same_rng_seed():
    env = DummyEnv()
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first_actions = [select_actions(env, "random", first_rng) for _ in range(5)]
    second_actions = [select_actions(env, "random", second_rng) for _ in range(5)]

    assert first_actions == second_actions
    for action_dict in first_actions:
        assert set(action_dict) == set(env.agents)
        assert all(0 <= action <= 2 for action in action_dict.values())


def test_summarize_episode_records_computes_means_and_rates():
    records = [
        {
            "steps": 5,
            "episode_reward": 3.0,
            "agent_rewards": {"agent_0": 1.0, "agent_1": 2.0},
            "collision": True,
            "arrival": False,
            "truncated": False,
        },
        {
            "steps": 7,
            "episode_reward": 9.0,
            "agent_rewards": {"agent_0": 4.0, "agent_1": 5.0},
            "collision": False,
            "arrival": True,
            "truncated": False,
        },
    ]

    summary = summarize_episode_records(records)

    assert summary["episodes"] == 2
    assert math.isclose(summary["mean_episode_reward"], 6.0)
    assert math.isclose(summary["mean_agent_reward"], 3.0)
    assert math.isclose(summary["mean_episode_length"], 6.0)
    assert math.isclose(summary["collision_rate"], 0.5)
    assert math.isclose(summary["arrival_rate"], 0.5)
    assert math.isclose(summary["truncation_rate"], 0.0)


def test_run_episode_uses_real_highway_intersection_env():
    config = SanityConfig(
        env_id="intersection-v1",
        env_seed=1,
        episodes=1,
        seeds=[1],
        policies=["idle-only"],
        highway_config={
            "controlled_vehicles": 2,
            "duration": 2,
            "initial_vehicle_count": 1,
            "spawn_probability": 0.0,
        },
    )

    record = run_episode(config, "idle-only", seed=1, episode_index=0)

    assert record["policy"] == "idle-only"
    assert record["seed"] == 1
    assert record["episode_index"] == 0
    assert record["steps"] >= 1
    assert set(record["agent_rewards"]) == {"agent_0", "agent_1"}
    assert isinstance(record["collision"], bool)
    assert isinstance(record["arrival"], bool)
    assert isinstance(record["truncated"], bool)
    assert len(record["crashed_agents"]) == 2
    assert len(record["arrived_agents"]) == 2


def test_run_sanity_baseline_runs_all_configured_policies():
    config = SanityConfig(
        env_id="intersection-v1",
        env_seed=1,
        episodes=1,
        seeds=[1],
        policies=["random", "idle-only"],
        highway_config={
            "controlled_vehicles": 1,
            "duration": 1,
            "initial_vehicle_count": 0,
            "spawn_probability": 0.0,
        },
    )

    results = run_sanity_baseline(config, policy="all")

    assert set(results["policies"]) == {"random", "idle-only"}
    assert results["config"] == {
        "env_id": "intersection-v1",
        "episodes": 1,
        "seeds": [1],
    }
    for policy_result in results["policies"].values():
        assert len(policy_result["episodes"]) == 1
        assert policy_result["summary"]["episodes"] == 1

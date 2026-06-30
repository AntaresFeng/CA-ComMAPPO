import json
from argparse import Namespace
from pathlib import Path

from gymnasium.spaces import Tuple
from xuance.environment import make_envs

from ca_commappo.envs.highway_intersection import (
    HighwayIntersectionMultiAgentEnv,
    build_intersection_config,
    register_highway_intersection_env,
)


def test_default_highway_config_matches_runtime_env_config():
    expected_config = json.loads(
        Path("tests/default_config.json").read_text(encoding="utf-8")
    )
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(env_id="intersection-v1", highway_config={})
    )

    try:
        assert env.env.unwrapped.config == expected_config
    finally:
        env.close()


def test_build_config_deep_merges_highway_overrides():
    config = build_intersection_config(
        {
            "controlled_vehicles": 3,
            "duration": 7,
            "observation": {
                "observation_config": {
                    "vehicles_count": 9,
                    "features": ["presence", "x", "y"],
                }
            },
            "action": {
                "action_config": {
                    "target_speeds": [0, 5, 10],
                }
            },
        },
    )

    assert config["controlled_vehicles"] == 3
    assert config["duration"] == 7
    assert config["observation"]["type"] == "MultiAgentObservation"
    assert config["observation"]["observation_config"]["type"] == "Kinematics"
    assert config["observation"]["observation_config"]["vehicles_count"] == 9
    assert config["observation"]["observation_config"]["features"] == [
        "presence",
        "x",
        "y",
    ]
    assert config["action"]["type"] == "MultiAgentAction"
    assert config["action"]["action_config"]["type"] == "DiscreteMetaAction"
    assert config["action"]["action_config"]["target_speeds"] == [0, 5, 10]


def test_adapter_uses_intersection_v1_with_configurable_agent_count_and_target_speeds():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={"controlled_vehicles": 3},
        )
    )

    try:
        assert env.env_id == "intersection-v1"
        assert env.num_agents == 3
        assert env.agents == ["agent_0", "agent_1", "agent_2"]
        assert isinstance(env.env.observation_space, Tuple)
        assert isinstance(env.env.action_space, Tuple)
        assert len(env.env.observation_space.spaces) == 3
        assert len(env.env.action_space.spaces) == 3

        action_speeds = [
            action_type.target_speeds.tolist()
            for action_type in env.env.unwrapped.action_type.agents_action_types
        ]
        vehicle_speeds = [
            vehicle.target_speeds.tolist()
            for vehicle in env.env.unwrapped.controlled_vehicles
        ]

        assert action_speeds == [[0, 4.5, 9]] * 3
        assert vehicle_speeds == [[0, 4.5, 9]] * 3
    finally:
        env.close()


def test_reset_step_and_state_are_xuance_compatible():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={"controlled_vehicles": 3, "duration": 2},
        )
    )

    try:
        obs, info = env.reset(seed=0)
        assert set(obs) == set(env.agents)
        assert info == {}
        assert env.state().shape == env.state_space.shape

        actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        next_obs, rewards, terminated, truncated, step_info = env.step(actions)

        assert set(next_obs) == set(env.agents)
        assert set(rewards) == set(env.agents)
        assert set(terminated) == set(env.agents)
        assert isinstance(truncated, bool)
        assert "agents_rewards" in step_info
        assert env.agent_mask() == {agent: True for agent in env.agents}
        assert set(env.avail_actions()) == set(env.agents)
    finally:
        env.close()


def test_collision_global_termination_marks_all_agents_done():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={
                "controlled_vehicles": 3,
                "duration": 13,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "normalize_reward": False,
            },
        )
    )

    try:
        env.reset(seed=3)
        terminated = {agent: False for agent in env.agents}
        step_info = {}

        for _ in range(13):
            _obs, _rewards, terminated, _truncated, step_info = env.step(
                {agent: 1 for agent in env.agents}
            )
            if step_info["rewards"]["collision_reward"] > 0:
                break

        assert step_info["rewards"]["collision_reward"] > 0
        assert all(terminated.values())
    finally:
        env.close()


def test_arrival_rewards_and_termination_match_highway_info():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={
                "controlled_vehicles": 3,
                "duration": 13,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "normalize_reward": False,
            },
        )
    )

    try:
        env.reset(seed=1)
        partial_arrival = None
        final_arrival = None

        for _ in range(13):
            _obs, rewards, terminated, truncated, step_info = env.step(
                {agent: 1 for agent in env.agents}
            )
            arrived_ratio = step_info["rewards"]["arrived_reward"]
            if 0 < arrived_ratio < 1 and partial_arrival is None:
                partial_arrival = rewards, terminated, truncated, step_info
            if step_info["global_terminated"]:
                final_arrival = rewards, terminated, truncated, step_info
                break

        assert partial_arrival is not None
        rewards, terminated, truncated, step_info = partial_arrival
        assert not truncated
        assert not step_info["global_terminated"]
        assert step_info["agents_terminated"] == (False, True, True)
        assert terminated == {
            "agent_0": False,
            "agent_1": True,
            "agent_2": True,
        }
        assert (
            tuple(rewards[agent] for agent in env.agents) == step_info["agents_rewards"]
        )
        assert step_info["agents_rewards"] == (1.0, 1, 1)
        assert step_info["raw_agents_rewards"] == (1.0, 1, 1)

        assert final_arrival is not None
        rewards, terminated, truncated, step_info = final_arrival
        assert not truncated
        assert step_info["global_terminated"]
        assert step_info["agents_terminated"] == (True, True, True)
        assert terminated == {
            "agent_0": True,
            "agent_1": True,
            "agent_2": True,
        }
        assert (
            tuple(rewards[agent] for agent in env.agents) == step_info["agents_rewards"]
        )
        assert step_info["agents_rewards"] == (1, 0.0, 0.0)
        assert step_info["raw_agents_rewards"] == (1, 1, 1)
    finally:
        env.close()


def test_terminal_agents_are_masked_and_stop_receiving_rewards_after_arrival():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={
                "controlled_vehicles": 3,
                "duration": 13,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "normalize_reward": False,
            },
        )
    )

    try:
        env.reset(seed=1)
        partial_arrival = None

        for _ in range(13):
            _obs, _rewards, terminated, _truncated, step_info = env.step(
                {agent: 1 for agent in env.agents}
            )
            if 0 < step_info["rewards"]["arrived_reward"] < 1:
                partial_arrival = terminated, env.agent_mask()
                break

        assert partial_arrival is not None
        terminated, agent_mask = partial_arrival
        assert terminated == {
            "agent_0": False,
            "agent_1": True,
            "agent_2": True,
        }
        assert agent_mask == {agent: True for agent in env.agents}

        _obs, rewards, _terminated, _truncated, _step_info = env.step(
            {agent: 1 for agent in env.agents}
        )

        assert env.agent_mask() == {
            "agent_0": True,
            "agent_1": False,
            "agent_2": False,
        }
        assert rewards["agent_1"] == 0.0
        assert rewards["agent_2"] == 0.0
    finally:
        env.close()


def test_register_highway_intersection_env_with_xuance_make_envs():
    register_highway_intersection_env()
    envs = make_envs(
        Namespace(
            env_name="HighwayIntersection",
            env_id="intersection-v1",
            env_seed=1,
            parallels=1,
            vectorize="DummyVecMultiAgentEnv",
            distributed_training=False,
            render_mode="rgb_array",
            highway_config={"controlled_vehicles": 2, "duration": 2},
        )
    )

    try:
        obs, _info = envs.reset()
        assert len(obs) == 1
        assert set(obs[0]) == {"agent_0", "agent_1"}
        assert envs.num_agents == 2
    finally:
        envs.close()

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


def test_adapter_uses_intersection_v1_with_configurable_agent_count():
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
    finally:
        env.close()


def test_adapter_preserves_intersection_discrete_target_speeds():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={"controlled_vehicles": 3},
        )
    )

    try:
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

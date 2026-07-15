import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.spaces import Tuple
from highway_env import utils
from xuance.environment import REGISTRY_MULTI_AGENT_ENV, make_envs

from ca_commappo.envs.highway_intersection_wrapper import (
    DEFAULT_ENV_NAME,
    GLOBAL_OBSERVATION_FEATURES,
    HighwayIntersectionMultiAgentEnv,
    build_intersection_config,
)


def _expected_vehicle_features(env, vehicle):
    record = vehicle.to_dict(origin_vehicle=None, observe_intentions=False)
    observation_config = env.config["observation"]["observation_config"]
    normalize = observation_config.get("normalize", True)
    clip = observation_config.get("clip", True)
    feature_ranges = observation_config.get("features_range", {})
    values = []
    for feature in GLOBAL_OBSERVATION_FEATURES:
        value = float(record[feature])
        if normalize and feature in feature_ranges:
            value = float(utils.lmap(value, feature_ranges[feature], [-1, 1]))
            if clip:
                value = float(np.clip(value, -1, 1))
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _npc_sort_key(vehicle):
    record = vehicle.to_dict(origin_vehicle=None, observe_intentions=False)
    x, y = float(record["x"]), float(record["y"])
    return (
        x * x + y * y,
        x,
        y,
        float(record["vx"]),
        float(record["vy"]),
        float(record["cos_h"]),
        float(record["sin_h"]),
    )


def test_default_highway_config_matches_runtime_env_config():
    expected_config = json.loads(
        Path("tests/default_config.json").read_text(encoding="utf-8")
    )
    env = HighwayIntersectionMultiAgentEnv(Namespace(highway_config={}))

    try:
        assert env.env_id == "intersection-multi-agent-v1"
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


def test_adapter_uses_multi_agent_intersection_by_default():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            highway_config={"controlled_vehicles": 3},
        )
    )

    try:
        assert env.env_id == "intersection-multi-agent-v1"
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


def test_adapter_keeps_intersection_v1_compatible_when_explicitly_requested():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={"controlled_vehicles": 3},
        )
    )

    try:
        assert env.env_id == "intersection-v1"
        assert env.env.unwrapped.__class__.__name__ == "ContinuousIntersectionEnv"
        assert isinstance(env.env.observation_space, Tuple)
        assert isinstance(env.env.action_space, Tuple)
        assert len(env.env.observation_space.spaces) == 3
        assert len(env.env.action_space.spaces) == 3
    finally:
        env.close()


def test_reset_step_and_state_are_xuance_compatible():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
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


@pytest.mark.parametrize(
    ("highway_config", "expected_capacity", "expected_state_shape"),
    [
        (
            {
                "controlled_vehicles": 2,
                "duration": 20,
                "initial_vehicle_count": 10,
            },
            30,
            (254,),
        ),
        (
            {
                "controlled_vehicles": 3,
                "duration": 13,
                "initial_vehicle_count": 10,
            },
            23,
            (205,),
        ),
        (
            {
                "controlled_vehicles": 2,
                "duration": 13,
                "initial_vehicle_count": 0,
            },
            14,
            (126,),
        ),
    ],
)
def test_global_npc_capacity_defaults_to_safe_bound(
    highway_config, expected_capacity, expected_state_shape
):
    env = HighwayIntersectionMultiAgentEnv(Namespace(highway_config=highway_config))

    try:
        assert env.global_npc_capacity == expected_capacity
        assert env.state_space.shape == expected_state_shape
    finally:
        env.close()


def test_global_npc_capacity_accepts_positive_integer_override():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(global_npc_capacity=4, highway_config={"controlled_vehicles": 2})
    )

    try:
        assert env.global_npc_capacity == 4
        assert env.state_space.shape == (46,)
    finally:
        env.close()


@pytest.mark.parametrize("invalid_capacity", [0, -1, 1.5, True, "3"])
def test_global_npc_capacity_rejects_invalid_override(invalid_capacity):
    with pytest.raises(
        ValueError, match="global_npc_capacity must be a positive integer"
    ):
        HighwayIntersectionMultiAgentEnv(
            Namespace(
                global_npc_capacity=invalid_capacity,
                highway_config={"controlled_vehicles": 2},
            )
        )


def test_global_observation_is_zero_and_independent_before_reset():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(global_npc_capacity=4, highway_config={"controlled_vehicles": 2})
    )

    try:
        first = env.global_observation()
        second = env.global_observation()

        assert set(first) == {"controlled", "npc", "npc_mask"}
        assert first["controlled"].shape == (2, 7)
        assert first["npc"].shape == (4, 7)
        assert first["npc_mask"].shape == (4,)
        assert all(value.dtype == np.float32 for value in first.values())
        assert all(not np.any(value) for value in first.values())

        first["npc"][0, 0] = 1.0
        assert not np.any(second["npc"])
        assert not np.any(env.state())
    finally:
        env.close()


def test_global_observation_uses_fixed_controlled_rows_and_sorted_npcs():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            global_npc_capacity=2,
            highway_config={
                "controlled_vehicles": 2,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.0,
            },
        )
    )

    try:
        env.reset(seed=0)
        global_obs = env.global_observation()
        base_env = env.env.unwrapped

        for index, vehicle in enumerate(base_env.controlled_vehicles):
            np.testing.assert_allclose(
                global_obs["controlled"][index],
                _expected_vehicle_features(env, vehicle),
            )

        controlled_ids = {id(vehicle) for vehicle in base_env.controlled_vehicles}
        expected_npcs = sorted(
            (
                vehicle
                for vehicle in base_env.road.vehicles
                if id(vehicle) not in controlled_ids
            ),
            key=_npc_sort_key,
        )[: env.global_npc_capacity]
        assert len(expected_npcs) == env.global_npc_capacity
        for index, vehicle in enumerate(expected_npcs):
            np.testing.assert_allclose(
                global_obs["npc"][index],
                _expected_vehicle_features(env, vehicle),
            )
        np.testing.assert_array_equal(global_obs["npc_mask"], [1.0, 1.0])

        expected_state = np.concatenate(
            [
                global_obs["controlled"].reshape(-1),
                global_obs["npc"].reshape(-1),
                global_obs["npc_mask"],
            ]
        ).astype(np.float32)
        np.testing.assert_array_equal(env.state(), expected_state)
        assert env.state_space.contains(env.state())
    finally:
        env.close()


def test_global_observation_pads_npcs_and_rebuilds_distance_order():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            global_npc_capacity=20,
            highway_config={
                "controlled_vehicles": 2,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.0,
            },
        )
    )

    try:
        env.reset(seed=0)
        base_env = env.env.unwrapped
        controlled_ids = {id(vehicle) for vehicle in base_env.controlled_vehicles}
        npcs = [
            vehicle
            for vehicle in base_env.road.vehicles
            if id(vehicle) not in controlled_ids
        ]
        assert len(npcs) >= 2

        first, second = npcs[:2]
        first.position = np.asarray([1.0, 0.0])
        second.position = np.asarray([10.0, 0.0])
        first_order = env.global_observation()
        np.testing.assert_allclose(
            first_order["npc"][0], _expected_vehicle_features(env, first)
        )

        first.position = np.asarray([20.0, 0.0])
        second.position = np.asarray([2.0, 0.0])
        second_order = env.global_observation()
        np.testing.assert_allclose(
            second_order["npc"][0], _expected_vehicle_features(env, second)
        )

        active_npc_count = min(len(npcs), env.global_npc_capacity)
        np.testing.assert_array_equal(
            second_order["npc_mask"][:active_npc_count],
            np.ones(active_npc_count, dtype=np.float32),
        )
        assert not np.any(second_order["npc_mask"][active_npc_count:])
        assert not np.any(second_order["npc"][active_npc_count:])
    finally:
        env.close()


def test_step_reobserves_post_spawn_scene_for_actor_and_global_state():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            flatten_observations=True,
            global_npc_capacity=20,
            highway_config={
                "controlled_vehicles": 2,
                "duration": 2,
                "initial_vehicle_count": 10,
                "spawn_probability": 1.0,
            },
        )
    )

    try:
        env.reset(seed=0)
        base_env = env.env.unwrapped
        before_ids = {id(vehicle) for vehicle in base_env.road.vehicles}

        next_obs, *_ = env.step({agent: 1 for agent in env.agents})
        current_observation = base_env.observation_type.observe()
        for index, agent in enumerate(env.agents):
            np.testing.assert_allclose(
                next_obs[agent],
                np.asarray(current_observation[index], dtype=np.float32).reshape(-1),
            )

        new_vehicles = [
            vehicle
            for vehicle in base_env.road.vehicles
            if id(vehicle) not in before_ids
        ]
        assert len(new_vehicles) == 1
        global_obs = env.global_observation()
        active_npcs = global_obs["npc"][global_obs["npc_mask"].astype(bool)]
        expected_new_vehicle = _expected_vehicle_features(env, new_vehicles[0])
        assert any(
            np.allclose(row, expected_new_vehicle, rtol=1e-6, atol=1e-7)
            for row in active_npcs
        )
    finally:
        env.close()


def test_flatten_observations_option_exposes_one_dimensional_spaces_and_obs():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            flatten_observations=True,
            highway_config={
                "controlled_vehicles": 2,
                "observation": {
                    "observation_config": {
                        "vehicles_count": 15,
                        "features": [
                            "presence",
                            "x",
                            "y",
                            "vx",
                            "vy",
                            "cos_h",
                            "sin_h",
                        ],
                    }
                },
            },
        )
    )

    try:
        assert env.observation_space["agent_0"].shape == (105,)
        assert env.observation_space["agent_1"].shape == (105,)

        obs, _info = env.reset(seed=0)
        assert obs["agent_0"].shape == (105,)
        assert obs["agent_1"].shape == (105,)
        assert env.global_npc_capacity == 23
        assert env.state().shape == (198,)
    finally:
        env.close()


def test_collision_global_termination_marks_all_agents_done():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
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
        assert not np.any(env.global_observation()["controlled"])
    finally:
        env.close()


def test_arrival_rewards_and_termination_match_highway_info():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
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
        assert step_info["arrived"] == (False, True, True)
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
        assert step_info["arrived"] == (True, True, True)
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
        controlled_state = env.global_observation()["controlled"]
        assert np.any(controlled_state[0])
        assert not np.any(controlled_state[1])
        assert not np.any(controlled_state[2])

        _obs, rewards, _terminated, _truncated, step_info = env.step(
            {agent: 1 for agent in env.agents}
        )

        assert env.agent_mask() == {
            "agent_0": True,
            "agent_1": False,
            "agent_2": False,
        }
        assert rewards["agent_1"] == 0.0
        assert rewards["agent_2"] == 0.0
        assert (
            tuple(rewards[agent] for agent in env.agents) == step_info["agents_rewards"]
        )
        assert step_info["agents_rewards"] == (1, 0.0, 0.0)
        assert step_info["raw_agents_rewards"] == (1, 1, 1)
    finally:
        env.close()


def test_highway_intersection_env_auto_registers_with_xuance_make_envs():
    assert (
        REGISTRY_MULTI_AGENT_ENV[DEFAULT_ENV_NAME] is HighwayIntersectionMultiAgentEnv
    )

    envs = make_envs(
        Namespace(
            env_name="HighwayIntersection",
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
        assert envs.state_space.shape == (110,)
        assert np.asarray(envs.buf_state[0]).shape == (110,)
    finally:
        envs.close()


def test_render_accepts_xuance_render_mode_argument():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            render_mode="rgb_array",
            highway_config={"controlled_vehicles": 2, "duration": 2},
        )
    )

    try:
        env.reset(seed=0)
        image = env.render("rgb_array")

        assert image is not None
    finally:
        env.close()

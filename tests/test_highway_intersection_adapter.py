from argparse import Namespace

import numpy as np
import pytest
from gymnasium.spaces import Tuple
from xuance.environment import REGISTRY_MULTI_AGENT_ENV, make_envs

from ca_commappo.envs.highway_intersection_wrapper import (
    DEFAULT_ENV_NAME,
    HighwayIntersectionMultiAgentEnv,
    build_intersection_config,
)


@pytest.mark.parametrize("count", [1, 2, 4, 6, 8])
def test_population_and_routes_are_valid_across_resets(count):
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(highway_config={"controlled_vehicles": count})
    )
    try:
        for seed in range(10):
            obs, info = env.reset(seed=seed)
            raw = env.env.unwrapped
            assert len(raw.road.vehicles) == count
            assert set(raw.controlled_vehicles) == set(raw.road.vehicles)
            for vehicle, spec in zip(
                raw.controlled_vehicles, info["agents_navigation"], strict=True
            ):
                assert len(vehicle.route) == 3
                assert spec["entry"] != spec["destination"]
                assert vehicle.route[-1][1] == f"exit{spec['destination']}"
                assert vehicle.route[-1][1] not in raw.road.network.graph
            initial = [v.position.copy() for v in raw.controlled_vehicles]
            _, _, terminated, _, step_info = env.step(
                {agent: 1 for agent in env.agents}
            )
            assert not any(step_info["crashed"])
            assert not any(terminated.values())
            assert all(
                np.linalg.norm(v.position - pos) > 0
                for v, pos in zip(raw.controlled_vehicles, initial, strict=True)
            )
            replay, replay_info = env.reset(seed=seed)
            assert replay_info["agents_navigation"] == info["agents_navigation"]
            for agent in env.agents:
                np.testing.assert_array_equal(replay[agent], obs[agent])
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
        assert env.env_id == "ca-intersection-v0"
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
        assert info["arrival_times"] == (None, None, None)
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


def test_adapter_enforces_declared_max_episode_steps(monkeypatch):
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            highway_config={
                "controlled_vehicles": 1,
                "duration": 0.4,
                "policy_frequency": 5,
                "initial_vehicle_count": 0,
                "spawn_probability": 0,
            }
        )
    )

    try:
        monkeypatch.setattr(env.env.unwrapped, "_is_truncated", lambda: False)
        env.reset(seed=0)
        actions = {agent: 1 for agent in env.agents}

        *_rest, first_truncated, _info = env.step(actions)
        *_rest, second_truncated, _info = env.step(actions)

        assert env.max_episode_steps == 2
        assert first_truncated is False
        assert second_truncated is True
    finally:
        env.close()


def test_step_info_exposes_speed_waiting_and_dt_for_reporting():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(highway_config={"controlled_vehicles": 3, "duration": 2})
    )
    try:
        env.reset(seed=0)
        actions = {agent: 1 for agent in env.agents}  # IDLE

        _obs, _rewards, _terminated, _truncated, step_info = env.step(actions)
        assert "agents_speeds" in step_info
        assert "waiting_steps" in step_info
        assert "dt" in step_info
        assert len(step_info["agents_speeds"]) == env.num_agents
        assert len(step_info["waiting_steps"]) == env.num_agents
        assert all(isinstance(v, float) for v in step_info["agents_speeds"])
        assert all(isinstance(v, int) and v >= 0 for v in step_info["waiting_steps"])
        assert step_info["dt"] > 0.0

        # Accumulation is non-decreasing across steps and resets on reset().
        first_waiting = step_info["waiting_steps"]
        _obs, _r, _t, _tr, step_info2 = env.step(actions)
        assert all(
            step_info2["waiting_steps"][i] >= first_waiting[i]
            for i in range(env.num_agents)
        )

        env.reset(seed=0)
        _obs, _r, _t, _tr, step_info3 = env.step(actions)
        assert all(v == 0 for v in step_info3["waiting_steps"])
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
        assert env.state().shape == (210,)
    finally:
        env.close()


def test_env_seed_is_only_injected_on_first_implicit_reset(monkeypatch):
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_seed=17,
            highway_config={"controlled_vehicles": 2, "duration": 2},
        )
    )
    reset_calls = []
    original_reset = env.env.reset

    def reset_spy(**kwargs):
        reset_calls.append(kwargs.copy())
        return original_reset(**kwargs)

    monkeypatch.setattr(env.env, "reset", reset_spy)

    try:
        env.reset()
        env.reset()
        env.reset(seed=23)
        env.reset()

        assert reset_calls == [{"seed": 17}, {}, {"seed": 23}, {}]
    finally:
        env.close()


def _place_on_exit(raw, vehicle, destination, longitudinal=26):
    lane_index = (f"il{destination}", f"exit{destination}", 0)
    lane = raw.road.network.get_lane(lane_index)
    vehicle.position = lane.position(longitudinal, 0)
    vehicle.heading = lane.heading_at(longitudinal)
    vehicle.lane_index = vehicle.target_lane_index = lane_index
    vehicle.lane = lane
    vehicle.route = [lane_index]


def test_collision_global_termination_marks_all_agents_done():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(highway_config={"controlled_vehicles": 3})
    )
    try:
        env.reset(seed=3)
        raw = env.env.unwrapped
        victim = raw.controlled_vehicles[1]
        _place_on_exit(raw, victim, raw._vehicle_specs[victim]["destination"])
        victim.crashed = True
        _, rewards, terminated, _, info = env.step({agent: 1 for agent in env.agents})
        assert info["crashed"] == (False, True, False)
        assert not info["arrived"][1]
        assert info["arrival_times"][1] is None
        assert rewards["agent_1"] == -40
        assert info["global_terminated"] and all(terminated.values())
    finally:
        env.close()


def test_terminal_agents_are_masked_and_retired_after_destination_arrival():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(highway_config={"controlled_vehicles": 2})
    )
    try:
        env.reset(seed=1)
        raw = env.env.unwrapped
        first, second = raw.controlled_vehicles
        destination = raw._vehicle_specs[first]["destination"]
        _place_on_exit(raw, first, (destination + 1) % 4)
        assert not raw.has_arrived(first)
        _place_on_exit(raw, first, destination)
        _, rewards, terminated, _, info = env.step({agent: 1 for agent in env.agents})
        assert rewards["agent_0"] == 10
        assert terminated == {"agent_0": True, "agent_1": False}
        assert env.agent_mask() == {"agent_0": True, "agent_1": True}
        assert info["next_agent_mask"] == {"agent_0": False, "agent_1": True}
        assert info["arrival_times"] == (0.2, None)
        assert first not in raw.road.vehicles and second in raw.road.vehicles
        position = first.position.copy()
        target_speed = first.target_speed
        _, rewards, _, _, info = env.step({"agent_0": 2, "agent_1": 1})
        assert rewards["agent_0"] == info["raw_agents_rewards"][0] == 0
        assert env.agent_mask() == {"agent_0": False, "agent_1": True}
        assert info["arrival_times"] == (0.2, None)
        np.testing.assert_array_equal(first.position, position)
        assert first.target_speed == target_speed
        _place_on_exit(raw, second, raw._vehicle_specs[second]["destination"])
        _, rewards, terminated, _, info = env.step({"agent_1": 1})
        assert rewards == {"agent_0": 0, "agent_1": 10}
        assert info["arrived"] == (True, True)
        assert info["arrival_times"] == pytest.approx((0.2, 0.6))
        assert info["global_terminated"] and all(terminated.values())
        _, info = env.reset(seed=1)
        assert info["arrival_times"] == (None, None)
        assert len(raw.road.vehicles) == 2
    finally:
        env.close()


def test_explicit_scenario_controls_spawn_route_and_rejects_overlap():
    specs = [
        {"entry": 0, "destination": 2, "longitudinal": 50, "speed": 4.5},
        {"entry": 0, "destination": 1, "longitudinal": 25, "speed": 9},
    ]
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            highway_config={
                "controlled_vehicles": 2,
                "controlled_vehicle_specs": specs,
                "access_length": 120,
                "perception_distance": 10,
            }
        )
    )
    try:
        obs, info = env.reset(seed=7)
        raw = env.env.unwrapped
        assert info["agents_navigation"] == tuple(specs)
        for vehicle, spec in zip(raw.controlled_vehicles, specs, strict=True):
            assert vehicle.lane.local_coordinates(vehicle.position)[0] == pytest.approx(
                spec["longitudinal"]
            )
            assert vehicle.speed == spec["speed"]
            assert vehicle.route[-1][1] == f"exit{spec['destination']}"
        assert all(view[:, 0].sum() == 1 for view in obs.values())
        with pytest.raises(ValueError, match="minimum_spawn_gap"):
            env.reset(
                options={"config": {"controlled_vehicle_specs": [specs[0], specs[0]]}}
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {
                "controlled_vehicle_specs": [
                    {"entry": 0, "destination": 0, "longitudinal": 50, "speed": 9}
                ]
            },
            "destination",
        ),
        (
            {
                "controlled_vehicle_specs": [
                    {"entry": 0, "destination": 1, "longitudinal": -1, "speed": 9}
                ]
            },
            "longitudinal",
        ),
        ({"controlled_vehicle_specs": []}, "one entry"),
        ({"spawn_spacing": 5}, "spawn_spacing"),
        ({"controlled_vehicles": 0}, "positive"),
    ],
)
def test_invalid_scenarios_fail_before_training(overrides, match):
    with pytest.raises(ValueError, match=match):
        HighwayIntersectionMultiAgentEnv(
            Namespace(highway_config={"controlled_vehicles": 1, **overrides})
        )


def test_background_population_is_explicit_and_does_not_remove_controlled_vehicles():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            highway_config={
                "controlled_vehicles": 2,
                "initial_vehicle_count": 5,
                "spawn_probability": 0,
            }
        )
    )
    try:
        env.reset(seed=4)
        raw = env.env.unwrapped
        assert len(raw.road.vehicles) == 7
        for _ in range(3):
            env.step({agent: 1 for agent in env.agents})
        assert len(raw.road.vehicles) == 7
        assert all(v in raw.road.vehicles for v in raw.controlled_vehicles)
    finally:
        env.close()


@pytest.mark.parametrize(
    "entry,destination", [(i, j) for i in range(4) for j in range(4) if i != j]
)
def test_all_movements_drive_to_their_destination(entry, destination):
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            highway_config={
                "controlled_vehicles": 1,
                "controlled_vehicle_specs": [
                    {
                        "entry": entry,
                        "destination": destination,
                        "longitudinal": 60,
                        "speed": 9,
                    }
                ],
            }
        )
    )
    try:
        env.reset(seed=7)
        arrival_rewards = 0
        for step in range(1, env.max_episode_steps + 1):
            _, rewards, terminated, truncated, info = env.step({"agent_0": 1})
            if rewards["agent_0"] == 10:
                arrival_rewards += 1
            if terminated["agent_0"] or truncated:
                break
        assert info["arrived"] == (True,)
        assert info["crashed"] == (False,)
        assert info["arrival_times"] == pytest.approx((step / 5,))
        assert arrival_rewards == 1
        assert (
            env.env.unwrapped.controlled_vehicles[0].lane_index[1]
            == f"exit{destination}"
        )
    finally:
        env.close()


def test_arrival_and_termination_remain_sticky_when_agents_arrive_sequentially(
    monkeypatch,
):
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(highway_config={"controlled_vehicles": 2, "duration": 3})
    )

    try:
        observations, _info = env.reset(seed=0)
        raw_observations = tuple(observations[agent] for agent in env.agents)
        arrival_sequence = [(True, False), (False, True)]
        step_state = {"index": 0, "arrived": arrival_sequence[0]}
        vehicle_indices = {
            id(vehicle): index
            for index, vehicle in enumerate(env.env.unwrapped.controlled_vehicles)
        }

        def fake_step(_actions):
            arrived = arrival_sequence[step_state["index"]]
            step_state["arrived"] = arrived
            step_state["index"] += 1
            rewards = tuple(10.0 if value else 0.0 for value in arrived)
            return raw_observations, rewards, arrived, False, {}

        def fake_has_arrived(vehicle):
            return step_state["arrived"][vehicle_indices[id(vehicle)]]

        monkeypatch.setattr(env.env, "step", fake_step)
        monkeypatch.setattr(env.env.unwrapped, "has_arrived", fake_has_arrived)

        _obs, rewards, terminated, truncated, info = env.step(
            {agent: 1 for agent in env.agents}
        )

        assert rewards == {"agent_0": 10.0, "agent_1": 0.0}
        assert terminated == {"agent_0": True, "agent_1": False}
        assert truncated is False
        assert info["current_arrived"] == (True, False)
        assert info["arrived"] == (True, False)
        assert info["raw_agents_terminated"] == (True, False)
        assert info["agents_terminated"] == (True, False)
        assert info["global_terminated"] is False

        _obs, rewards, terminated, truncated, info = env.step({"agent_1": 1})

        assert rewards == {"agent_0": 0.0, "agent_1": 10.0}
        assert terminated == {"agent_0": True, "agent_1": True}
        assert truncated is False
        assert info["current_arrived"] == (False, True)
        assert info["arrived"] == (True, True)
        assert info["raw_agents_terminated"] == (False, True)
        assert info["agents_terminated"] == (True, True)
        assert info["global_terminated"] is True
        assert env.agent_mask() == {"agent_0": False, "agent_1": True}
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

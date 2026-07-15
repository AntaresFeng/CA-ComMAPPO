from copy import deepcopy
from math import ceil
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401 - importing registers highway-env ids.
import numpy as np
from gymnasium import spaces
from highway_env import utils
from highway_env.envs.intersection_env import IntersectionEnv, MultiAgentIntersectionEnv
from xuance.environment import REGISTRY_MULTI_AGENT_ENV, RawMultiAgentEnv


DEFAULT_ENV_NAME = "HighwayIntersection"
DEFAULT_HIGHWAY_ENV_ID = "intersection-multi-agent-v1"
IDLE_ACTION = IntersectionEnv.ACTIONS_INDEXES["IDLE"]  # 1: "IDLE"
GLOBAL_OBSERVATION_FEATURES = ("presence", "x", "y", "vx", "vy", "cos_h", "sin_h")
INTERSECTION_CENTER = np.zeros(2, dtype=np.float32) # (0.0, 0.0)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _highway_multi_agent_intersection_defaults() -> dict[str, Any]:
    intersection_config = IntersectionEnv.default_config()
    multi_agent_config = MultiAgentIntersectionEnv.default_config()
    config = _deep_merge(intersection_config, multi_agent_config)
    config["observation"] = _deep_merge(
        {
            "type": multi_agent_config["observation"]["type"],
            "observation_config": intersection_config["observation"],
        },
        multi_agent_config["observation"],
    )
    config["action"] = _deep_merge(
        {
            "type": multi_agent_config["action"]["type"],
            "action_config": intersection_config["action"],
        },
        multi_agent_config["action"],
    )
    return config


def build_intersection_config(
    highway_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete multi-agent config for highway-env intersection.

    highway-env 1.11 applies user config with a shallow top-level update, so the
    adapter must create complete nested `observation` and `action` dictionaries
    before calling `gym.make`.
    """
    base_config = _highway_multi_agent_intersection_defaults()
    config = _deep_merge(base_config, highway_config or {})
    if int(config["controlled_vehicles"]) <= 0:
        raise ValueError("controlled_vehicles must be a positive integer")
    return config


class HighwayIntersectionMultiAgentEnv(RawMultiAgentEnv):
    """XuanCe RawMultiAgentEnv adapter for highway-env multi-agent intersection."""

    def __init__(self, env_config):
        super().__init__()
        self.env_id = getattr(env_config, "env_id", DEFAULT_HIGHWAY_ENV_ID)
        self.render_mode = getattr(env_config, "render_mode", None)
        self.flatten_observations = bool(
            getattr(env_config, "flatten_observations", False)
        )
        highway_config = getattr(env_config, "highway_config", None) or {}

        self.config = build_intersection_config(highway_config)
        self.num_agents = int(self.config["controlled_vehicles"])
        self.global_npc_capacity = self._resolve_global_npc_capacity(
            getattr(env_config, "global_npc_capacity", None)
        )
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_groups = [self.agents]
        self.env = gym.make(
            self.env_id, config=self.config, render_mode=self.render_mode
        )
        self._validate_multi_agent_spaces()

        self.observation_space = self._build_observation_space()
        self.action_space = {
            agent: self.env.action_space.spaces[i]
            for i, agent in enumerate(self.agents)
        }
        self.state_space = self._build_state_space()
        self.max_episode_steps = self._max_episode_steps()
        self._initial_seed = getattr(env_config, "env_seed", None)
        self._active_agents = self._all_agents_active()
        self._last_agent_mask = self._all_agents_active()
        self._scene_ready = False

        observation_config = self.config["observation"]["observation_config"]
        self._global_observation_normalize = bool(
            observation_config.get("normalize", True)
        )
        self._global_observation_clip = bool(observation_config.get("clip", True))
        self._global_observation_feature_ranges = deepcopy(
            observation_config.get("features_range", {})
        )

    def _validate_multi_agent_spaces(self) -> None:
        if not isinstance(self.env.observation_space, spaces.Tuple):
            raise ValueError(
                f"{self.env_id} observation_space is not Tuple; ensure "
                "observation.type is MultiAgentObservation."
            )
        if not isinstance(self.env.action_space, spaces.Tuple):
            raise ValueError(
                f"{self.env_id} action_space is not Tuple; ensure "
                "action.type is MultiAgentAction."
            )
        if len(self.env.observation_space.spaces) != self.num_agents:
            raise ValueError(
                "observation_space length does not match controlled_vehicles"
            )
        if len(self.env.action_space.spaces) != self.num_agents:
            raise ValueError("action_space length does not match controlled_vehicles")

    def _build_state_space(self) -> spaces.Box:
        feature_count = len(GLOBAL_OBSERVATION_FEATURES)
        flat_dim = (
            self.num_agents * feature_count
            + self.global_npc_capacity * feature_count
            + self.global_npc_capacity
        )
        return spaces.Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32)

    def _resolve_global_npc_capacity(self, configured_capacity: Any) -> int:
        if configured_capacity is not None:
            if (
                isinstance(configured_capacity, bool)
                or not isinstance(configured_capacity, (int, np.integer))
                or int(configured_capacity) <= 0
            ):
                raise ValueError("global_npc_capacity must be a positive integer")
            return int(configured_capacity)

        initial_npc_bound = max(1, int(self.config["initial_vehicle_count"]))
        step_spawn_bound = max(
            1,
            ceil(
                float(self.config["duration"]) * float(self.config["policy_frequency"])
            ),
        )
        return initial_npc_bound + step_spawn_bound

    def _build_observation_space(self) -> dict[str, spaces.Space]:
        observation_space = {}
        for i, agent in enumerate(self.agents):
            space = self.env.observation_space.spaces[i]
            if self.flatten_observations:
                space = self._flatten_box_space(space)
            observation_space[agent] = space
        return observation_space

    def _flatten_box_space(self, space: spaces.Space) -> spaces.Box:
        if not isinstance(space, spaces.Box):
            raise ValueError("flatten_observations requires Box observation spaces")
        return spaces.Box(
            low=np.asarray(space.low, dtype=np.float32).reshape(-1),
            high=np.asarray(space.high, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )

    def _max_episode_steps(self) -> int:
        duration = float(self.env.unwrapped.config.get("duration", 1))
        policy_frequency = float(self.env.unwrapped.config.get("policy_frequency", 1))
        return max(1, int(duration * policy_frequency))

    def _tuple_to_agent_dict(self, values: tuple[Any, ...]) -> dict[str, Any]:
        return {agent: values[i] for i, agent in enumerate(self.agents)}

    def _obs_tuple_to_agent_dict(
        self, values: tuple[Any, ...]
    ) -> dict[str, np.ndarray]:
        obs = {}
        for i, agent in enumerate(self.agents):
            value = np.asarray(values[i], dtype=np.float32)
            obs[agent] = value.reshape(-1) if self.flatten_observations else value
        return obs

    def _all_agents_active(self) -> dict[str, bool]:
        return {agent: True for agent in self.agents}

    def _actions_to_tuple(
        self,
        action_dict: dict[str, Any],
        active_agents: dict[str, bool] | None = None,
    ) -> tuple[Any, ...]:
        if active_agents is None:
            active_agents = self._all_agents_active()
        missing = [
            agent
            for agent in self.agents
            if active_agents[agent] and agent not in action_dict
        ]
        if missing:
            raise KeyError(f"Missing actions for agents: {missing}")
        return tuple(
            action_dict[agent] if active_agents[agent] else IDLE_ACTION
            for agent in self.agents
        )

    def _mask_rewards(
        self,
        rewards: dict[str, Any],
        active_agents: dict[str, bool],
    ) -> dict[str, Any]:
        return {
            agent: rewards[agent] if active_agents[agent] else 0.0
            for agent in self.agents
        }

    def _empty_global_observation(self) -> dict[str, np.ndarray]:
        feature_count = len(GLOBAL_OBSERVATION_FEATURES)
        return {
            "controlled": np.zeros((self.num_agents, feature_count), dtype=np.float32),
            "npc": np.zeros(
                (self.global_npc_capacity, feature_count), dtype=np.float32
            ),
            "npc_mask": np.zeros(self.global_npc_capacity, dtype=np.float32),
        }

    def _vehicle_feature_values(self, vehicle) -> tuple[dict[str, float], np.ndarray]:
        record = vehicle.to_dict(
            origin_vehicle=None,
            observe_intentions=False,
        )
        features = []
        for feature in GLOBAL_OBSERVATION_FEATURES:
            value = float(record[feature])
            if self._global_observation_normalize and feature in (
                self._global_observation_feature_ranges
            ):
                value = float(
                    utils.lmap(
                        value,
                        self._global_observation_feature_ranges[feature],
                        [-1, 1],
                    )
                )
                if self._global_observation_clip:
                    value = float(np.clip(value, -1, 1))
            features.append(value)
        return record, np.asarray(features, dtype=np.float32)

    @staticmethod
    def _npc_sort_key(record: dict[str, float]) -> tuple[float, ...]:
        position = np.asarray([record["x"], record["y"]], dtype=np.float64)
        relative_position = position - INTERSECTION_CENTER
        distance_squared = float(np.dot(relative_position, relative_position))
        return (
            distance_squared,
            float(record["x"]),
            float(record["y"]),
            float(record["vx"]),
            float(record["vy"]),
            float(record["cos_h"]),
            float(record["sin_h"]),
        )

    def global_observation(self) -> dict[str, np.ndarray]:
        """Return the current centralized scene observation.

        Controlled vehicles use fixed agent-aligned rows. NPC rows are rebuilt
        from the current scene and deterministically sorted by distance to the
        intersection center before truncation and zero-padding.
        """
        global_obs = self._empty_global_observation()
        base_env = self.env.unwrapped
        if not self._scene_ready or base_env.road is None:
            return global_obs

        controlled_vehicles = list(base_env.controlled_vehicles)
        for index, (agent, vehicle) in enumerate(
            zip(self.agents, controlled_vehicles, strict=True)
        ):
            if self._active_agents[agent]:
                _record, features = self._vehicle_feature_values(vehicle)
                global_obs["controlled"][index] = features

        controlled_vehicle_ids = {id(vehicle) for vehicle in controlled_vehicles}
        npc_records = []
        for vehicle in base_env.road.vehicles:
            if id(vehicle) in controlled_vehicle_ids:
                continue
            record, features = self._vehicle_feature_values(vehicle)
            npc_records.append((self._npc_sort_key(record), features))

        npc_records.sort(key=lambda item: item[0])
        for index, (_sort_key, features) in enumerate(
            npc_records[: self.global_npc_capacity]
        ):
            global_obs["npc"][index] = features
            global_obs["npc_mask"][index] = 1.0
        return global_obs

    def reset(self, **kwargs):
        if "seed" not in kwargs and self._initial_seed is not None:
            kwargs["seed"] = self._initial_seed
        observation, _info = self.env.reset(**kwargs)
        obs_dict = self._obs_tuple_to_agent_dict(observation)
        self._active_agents = self._all_agents_active()
        self._last_agent_mask = self._all_agents_active()
        self._scene_ready = True
        return obs_dict, {}

    def step(self, action_dict: dict[str, Any]):
        active_before_step = dict(self._active_agents)
        action_tuple = self._actions_to_tuple(action_dict, active_before_step)
        # intersection-multi-agent-v1's MultiAgentWrapper returns per-agent
        # tuples for reward and terminated directly from step().
        _observation, reward, terminated, truncated, info = self.env.step(action_tuple)
        # IntersectionEnv creates and removes NPCs after AbstractEnv has built
        # its observation. Re-observe here so actor observations and the
        # centralized state describe the same post-step vehicle snapshot.
        observation = self.env.unwrapped.observation_type.observe()
        obs_dict = self._obs_tuple_to_agent_dict(observation)
        raw_rewards = self._tuple_to_agent_dict(reward)
        rewards = self._mask_rewards(raw_rewards, active_before_step)
        terminated = self._tuple_to_agent_dict(terminated)

        # Compute environment-level done signal for XuanCe: any crash or
        # all agents arrived.  MultiAgentWrapper discarded the upstream
        # scalar terminated, so we recompute from per-agent state.
        crashed = tuple(bool(v.crashed) for v in self.env.unwrapped.controlled_vehicles)
        arrived = tuple(
            self.env.unwrapped.has_arrived(v)
            for v in self.env.unwrapped.controlled_vehicles
        )
        info["global_terminated"] = any(crashed) or all(terminated.values())
        if info["global_terminated"]:
            terminated = {agent: True for agent in self.agents}
        info["raw_agents_rewards"] = tuple(raw_rewards[agent] for agent in self.agents)
        info["agents_rewards"] = tuple(rewards[agent] for agent in self.agents)

        # Upstream info["crashed"] is self.vehicle.crashed, i.e. only
        # controlled_vehicles[0].  In a multi-agent setting this silently
        # ignores collisions on all other agents (see abstract.py:179).
        # Replace it with a per-agent tuple so downstream consumers see the
        # true crash state of every controlled vehicle.
        info["crashed"] = crashed
        info["arrived"] = arrived
        self._last_agent_mask = active_before_step
        self._active_agents = {
            agent: active_before_step[agent] and not bool(terminated[agent])
            for agent in self.agents
        }
        return obs_dict, rewards, terminated, bool(truncated), info

    def state(self):
        global_obs = self.global_observation()
        return np.concatenate(
            [
                global_obs["controlled"].reshape(-1),
                global_obs["npc"].reshape(-1),
                global_obs["npc_mask"],
            ]
        ).astype(np.float32, copy=False)

    def agent_mask(self):
        return dict(self._last_agent_mask)

    def avail_actions(self):
        actions = {}
        for agent in self.agents:
            action_space = self.action_space[agent]
            if not isinstance(action_space, spaces.Discrete):
                return None
            actions[agent] = np.ones(action_space.n, dtype=np.bool_)
        return actions

    def render(self, *_args, **_kwargs):
        # XuanCe passes render_mode here, but gymnasium fixes it in gym.make().
        return self.env.render()

    def close(self):
        return self.env.close()


def register_highway_intersection_env(env_name: str = DEFAULT_ENV_NAME) -> None:
    REGISTRY_MULTI_AGENT_ENV[env_name] = HighwayIntersectionMultiAgentEnv


register_highway_intersection_env()

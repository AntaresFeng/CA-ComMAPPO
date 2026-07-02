from copy import deepcopy
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401 - importing registers highway-env ids.
import numpy as np
from gymnasium import spaces
from highway_env.envs.intersection_env import IntersectionEnv, MultiAgentIntersectionEnv
from xuance.environment import REGISTRY_MULTI_AGENT_ENV, RawMultiAgentEnv


DEFAULT_ENV_NAME = "HighwayIntersection"
IDLE_ACTION = IntersectionEnv.ACTIONS_INDEXES[
    "IDLE"
]  # {0: "SLOWER", 1: "IDLE", 2: "FASTER"}


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
    """Build a complete multi-agent config for highway-env intersection-v1.

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
    """XuanCe RawMultiAgentEnv adapter for highway-env intersection-v1."""

    def __init__(self, env_config):
        super().__init__()
        self.env_id = getattr(env_config, "env_id", "intersection-v1")
        self.render_mode = getattr(env_config, "render_mode", None)
        self.flatten_observations = bool(
            getattr(env_config, "flatten_observations", False)
        )
        highway_config = getattr(env_config, "highway_config", None) or {}

        self.config = build_intersection_config(highway_config)
        self.num_agents = int(self.config["controlled_vehicles"])
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
        self._last_obs: dict[str, np.ndarray] | None = None
        self._initial_seed = getattr(env_config, "env_seed", None)
        self._active_agents = self._all_agents_active()
        self._last_agent_mask = self._all_agents_active()

    def _validate_multi_agent_spaces(self) -> None:
        if not isinstance(self.env.observation_space, spaces.Tuple):
            raise ValueError(
                "intersection-v1 observation_space is not Tuple; ensure "
                "observation.type is MultiAgentObservation."
            )
        if not isinstance(self.env.action_space, spaces.Tuple):
            raise ValueError(
                "intersection-v1 action_space is not Tuple; ensure "
                "action.type is MultiAgentAction."
            )
        if len(self.env.observation_space.spaces) != self.num_agents:
            raise ValueError(
                "observation_space length does not match controlled_vehicles"
            )
        if len(self.env.action_space.spaces) != self.num_agents:
            raise ValueError("action_space length does not match controlled_vehicles")

    def _build_state_space(self) -> spaces.Box:
        flat_dim = sum(
            int(np.prod(space.shape)) for space in self.observation_space.values()
        )
        return spaces.Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32)

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

    def _remember_obs(self, obs: dict[str, Any]) -> None:
        self._last_obs = {
            agent: np.asarray(value, dtype=np.float32) for agent, value in obs.items()
        }

    def reset(self, **kwargs):
        if "seed" not in kwargs and self._initial_seed is not None:
            kwargs["seed"] = self._initial_seed
        observation, _info = self.env.reset(**kwargs)
        obs_dict = self._obs_tuple_to_agent_dict(observation)
        self._active_agents = self._all_agents_active()
        self._last_agent_mask = self._all_agents_active()
        self._remember_obs(obs_dict)
        return obs_dict, {}

    def step(self, action_dict: dict[str, Any]):
        active_before_step = dict(self._active_agents)
        action_tuple = self._actions_to_tuple(action_dict, active_before_step)
        observation, _reward, terminated_global, truncated, info = self.env.step(
            action_tuple
        )
        obs_dict = self._obs_tuple_to_agent_dict(observation)
        raw_rewards = self._tuple_to_agent_dict(info["agents_rewards"])
        rewards = self._mask_rewards(raw_rewards, active_before_step)
        terminated = self._tuple_to_agent_dict(info["agents_terminated"])
        info["global_terminated"] = bool(terminated_global)
        if terminated_global:
            terminated = {agent: True for agent in self.agents}
        info["raw_agents_rewards"] = tuple(raw_rewards[agent] for agent in self.agents)
        info["agents_rewards"] = tuple(rewards[agent] for agent in self.agents)
        self._last_agent_mask = active_before_step
        self._active_agents = {
            agent: active_before_step[agent] and not bool(terminated[agent])
            for agent in self.agents
        }
        self._remember_obs(obs_dict)
        return obs_dict, rewards, terminated, bool(truncated), info

    def state(self):
        if self._last_obs is None:
            return np.zeros(self.state_space.shape, dtype=np.float32)
        return np.concatenate(
            [self._last_obs[agent].reshape(-1) for agent in self.agents]
        ).astype(np.float32)

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

from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401 - importing registers highway-env ids.
import numpy as np
from gymnasium import spaces
from highway_env.envs.intersection_env import IntersectionEnv
from xuance.environment import REGISTRY_MULTI_AGENT_ENV, RawMultiAgentEnv

from ca_commappo.envs.intersection import PROJECT_ENV_ID, ResearchIntersectionEnv

DEFAULT_ENV_NAME = "HighwayIntersection"
DEFAULT_HIGHWAY_ENV_ID = PROJECT_ENV_ID
IDLE_ACTION = IntersectionEnv.ACTIONS_INDEXES["IDLE"]  # 1: "IDLE"
# Speeds below this threshold (m/s) count toward waiting time.
WAITING_SPEED_THRESHOLD = 1.0


def build_intersection_config(
    highway_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ResearchIntersectionEnv.resolve_config(highway_config)


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
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_groups = [self.agents]
        self.env = gym.make(
            self.env_id,
            config=highway_config if self.env_id == PROJECT_ENV_ID else self.config,
            render_mode=self.render_mode,
        )
        self.config = self.env.unwrapped.config
        self._validate_multi_agent_spaces()

        self.observation_space = self._build_observation_space()
        self.action_space = {
            agent: self.env.action_space.spaces[i]
            for i, agent in enumerate(self.agents)
        }
        self.state_space = self._build_state_space()
        self.max_episode_steps = self._max_episode_steps()
        self._episode_step = 0
        self._last_obs: dict[str, np.ndarray] | None = None
        self._initial_seed = getattr(env_config, "env_seed", None)
        self._has_reset = False
        self._active_agents = self._all_agents_active()
        self._last_agent_mask = self._all_agents_active()
        self._arrived_agents = {agent: False for agent in self.agents}
        self._waiting_steps = {agent: 0 for agent in self.agents}
        self._dt = self._compute_dt()

    def _compute_dt(self) -> float:
        policy_frequency = float(
            self.env.unwrapped.config.get("policy_frequency", 1) or 1
        )
        return 1.0 / policy_frequency if policy_frequency > 0 else 0.0

    def _validate_multi_agent_spaces(self) -> None:
        if not isinstance(self.env.observation_space, spaces.Tuple):
            raise TypeError(
                f"{self.env_id} observation_space is not Tuple; ensure "
                "observation.type is MultiAgentObservation."
            )
        if not isinstance(self.env.action_space, spaces.Tuple):
            raise TypeError(
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
            raise TypeError("flatten_observations requires Box observation spaces")
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
        if (
            not self._has_reset
            and "seed" not in kwargs
            and self._initial_seed is not None
        ):
            kwargs["seed"] = self._initial_seed
        observation, info = self.env.reset(**kwargs)
        self._has_reset = True
        obs_dict = self._obs_tuple_to_agent_dict(observation)
        self._active_agents = self._all_agents_active()
        self._last_agent_mask = self._all_agents_active()
        self._arrived_agents = {agent: False for agent in self.agents}
        self._episode_step = 0
        self._waiting_steps = {agent: 0 for agent in self.agents}
        self._remember_obs(obs_dict)
        return obs_dict, info

    def step(self, action_dict: dict[str, Any]):
        active_before_step = dict(self._active_agents)
        action_tuple = self._actions_to_tuple(action_dict, active_before_step)
        # MultiAgentWrapper returns per-agent
        # tuples for reward and terminated directly from step().
        observation, reward, terminated, truncated, info = self.env.step(action_tuple)
        self._episode_step += 1
        # highway-env advances time with floating-point increments. At 5 Hz,
        # 100 * 0.2 can remain just below a 20-second duration, so upstream may
        # truncate on step 101 even though XuanCe allocates an episode buffer
        # from the declared 100-step limit. Enforce that public limit exactly.
        truncated = bool(truncated) or self._episode_step >= self.max_episode_steps
        obs_dict = self._obs_tuple_to_agent_dict(observation)
        raw_rewards = self._tuple_to_agent_dict(reward)
        rewards = self._mask_rewards(raw_rewards, active_before_step)
        raw_terminated = self._tuple_to_agent_dict(terminated)

        # Compute environment-level done signal for XuanCe: any crash or
        # all agents arrived. Arrival is an episode-level event: after an agent
        # crosses the exit threshold it can leave that geometric region while
        # the other agents continue, but it must remain terminal and count as
        # arrived. MultiAgentWrapper discarded the upstream scalar terminated,
        # so recompute the global signal from sticky per-agent state.
        crashed = tuple(bool(v.crashed) for v in self.env.unwrapped.controlled_vehicles)
        current_arrived = tuple(
            bool(self.env.unwrapped.has_arrived(v))
            for v in self.env.unwrapped.controlled_vehicles
        )
        for i, agent in enumerate(self.agents):
            self._arrived_agents[agent] = (
                self._arrived_agents[agent] or current_arrived[i]
            )
        arrived = tuple(self._arrived_agents[agent] for agent in self.agents)
        terminated = {
            agent: not active_before_step[agent] or bool(raw_terminated[agent])
            for agent in self.agents
        }
        info["global_terminated"] = (
            bool(info.get("global_terminated", False)) or any(crashed) or all(arrived)
        )
        if info["global_terminated"]:
            terminated = {agent: True for agent in self.agents}
        info["raw_agents_rewards"] = tuple(raw_rewards[agent] for agent in self.agents)
        info["agents_rewards"] = tuple(rewards[agent] for agent in self.agents)
        info["raw_agents_terminated"] = tuple(
            bool(raw_terminated[agent]) for agent in self.agents
        )
        info["agents_terminated"] = tuple(
            bool(terminated[agent]) for agent in self.agents
        )

        # Upstream info["crashed"] is self.vehicle.crashed, i.e. only
        # controlled_vehicles[0].  In a multi-agent setting this silently
        # ignores collisions on all other agents (see abstract.py:179).
        # Replace it with a per-agent tuple so downstream consumers see the
        # true crash state of every controlled vehicle.
        info["crashed"] = crashed
        info["current_arrived"] = current_arrived
        info["arrived"] = arrived
        # Count the transition that ends an agent's episode, excluding later
        # placeholder actions from its waiting time.
        agents_speeds = tuple(
            float(v.speed) for v in self.env.unwrapped.controlled_vehicles
        )
        for i, agent in enumerate(self.agents):
            if active_before_step[agent] and agents_speeds[i] < WAITING_SPEED_THRESHOLD:
                self._waiting_steps[agent] += 1
        info["agents_speeds"] = agents_speeds
        info["waiting_steps"] = tuple(
            self._waiting_steps[agent] for agent in self.agents
        )
        info["dt"] = self._dt
        self._last_agent_mask = active_before_step
        self._active_agents = {
            agent: active_before_step[agent] and not bool(terminated[agent])
            for agent in self.agents
        }
        info["next_agent_mask"] = dict(self._active_agents)
        self._remember_obs(obs_dict)
        return obs_dict, rewards, terminated, truncated, info

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


register_highway_intersection_env()

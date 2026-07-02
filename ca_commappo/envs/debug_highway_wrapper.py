import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium as gym
import highway_env  # noqa: F401 - importing registers highway-env ids.
import numpy as np
from gymnasium import spaces

from ca_commappo.envs.highway_intersection import (
    HighwayIntersectionMultiAgentEnv,
    build_intersection_config,
)


class StepData(NamedTuple):
    obs: Any
    rewards: tuple[Any, ...]
    terminated: tuple[Any, ...]
    global_terminated: bool
    truncated: bool
    info: dict[str, Any]
    agent_mask: tuple[Any, ...] | None = None
    state: np.ndarray | None = None


DEFAULT_ENV_ID = "intersection-v1"
DEFAULT_HIGHWAY_CONFIG = {
    "controlled_vehicles": 2,
    "duration": 15,
    "spawn_probability": 1.0,
    "initial_vehicle_count": 20,
    "arrived_reward": 5,
    "collision_reward": -10,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete highway-env intersection episode through the raw "
            "Gymnasium env, the CA-ComMAPPO adapter, or both."
        ),
        epilog=(
            "Examples:\n"
            "  uv run python -m ca_commappo.envs.debug_highway_wrapper\n"
            "  uv run python -m ca_commappo.envs.debug_highway_wrapper --actions '1,1,1'\n"
            "  uv run python -m ca_commappo.envs.debug_highway_wrapper --target wrapper "
            "--render-mode human --pause 0.001\n"
            "  uv run python -m ca_commappo.envs.debug_highway_wrapper --target wrapper "
            "--seed 7 --print-observations"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=["raw", "wrapper", "both"],
        default="both",
        help="Which env surface to run.",
    )
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="highway-env id to create.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Episode seed. Omit it to let each run use fresh entropy. "
            "Random actions use the same seed by default when this is set."
        ),
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        default=None,
        help="Optional separate seed for random action generation.",
    )
    parser.add_argument(
        "--actions",
        default=None,
        help=(
            "Manual fixed action vector, such as '1,1,1'. A single value, "
            "such as '1', is broadcast to every agent."
        ),
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=None,
        help="Override highway_config.controlled_vehicles.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override highway_config.duration.",
    )
    parser.add_argument(
        "--spawn-probability",
        type=float,
        default=None,
        help="Override highway_config.spawn_probability.",
    )
    parser.add_argument(
        "--initial-vehicle-count",
        type=int,
        default=None,
        help="Override highway_config.initial_vehicle_count.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional JSON file containing extra highway_config overrides.",
    )
    parser.add_argument(
        "--highway-config-json",
        default=None,
        help="Optional JSON object containing extra highway_config overrides.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Safety cap. Defaults to duration * policy_frequency from config.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default=None,
        help="Pass a render_mode to highway-env.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep after each step, useful with --render-mode human.",
    )
    parser.add_argument(
        "--print-observations",
        action="store_true",
        help="Print full observation arrays instead of compact shape summaries.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the complete merged highway config used to create the envs.",
    )
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")
    if args.pause < 0:
        parser.error("--pause must be non-negative")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        highway_config = build_highway_config(args)
        complete_config = build_intersection_config(highway_config)
        max_steps = args.max_steps or max_episode_steps(complete_config)
        action_plan = build_action_plan(args, complete_config, max_steps)

        print_run_header(args, highway_config, complete_config, max_steps, action_plan)
        if args.target in {"raw", "both"}:
            run_raw_episode(args, complete_config, action_plan)
        if args.target in {"wrapper", "both"}:
            run_wrapper_episode(args, highway_config, action_plan)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_highway_config(args: argparse.Namespace) -> dict[str, Any]:
    highway_config = dict(DEFAULT_HIGHWAY_CONFIG)
    highway_config = _merge_json_source(
        highway_config, args.config_file, "--config-file"
    )
    highway_config = _merge_json_source(
        highway_config, args.highway_config_json, "--highway-config-json"
    )
    for key, value in {
        "controlled_vehicles": args.controlled_vehicles,
        "duration": args.duration,
        "spawn_probability": args.spawn_probability,
        "initial_vehicle_count": args.initial_vehicle_count,
    }.items():
        if value is not None:
            highway_config[key] = value
    return highway_config


def _merge_json_source(base: dict[str, Any], source: Any, label: str) -> dict[str, Any]:
    if source is None:
        return base
    loaded = json.loads(
        source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    )
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return deep_merge(base, loaded)


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def max_episode_steps(config: dict[str, Any]) -> int:
    duration = float(config.get("duration", 1))
    policy_frequency = float(config.get("policy_frequency", 1))
    return max(1, int(duration * policy_frequency))


def build_action_plan(
    args: argparse.Namespace,
    complete_config: dict[str, Any],
    max_steps: int,
) -> list[tuple[Any, ...]]:
    probe_env = gym.make(args.env_id, config=complete_config, render_mode=None)
    try:
        if not isinstance(probe_env.action_space, spaces.Tuple):
            raise TypeError("Expected raw action_space to be gymnasium.spaces.Tuple")
        action_space = probe_env.action_space
        if args.actions is not None:
            action = parse_manual_actions(args.actions, action_space)
            return [action for _ in range(max_steps)]

        rng = np.random.default_rng(
            args.action_seed if args.action_seed is not None else args.seed
        )
        return [sample_tuple_action(action_space, rng) for _ in range(max_steps)]
    finally:
        probe_env.close()


def parse_manual_actions(
    raw_actions: str,
    action_space: spaces.Tuple,
) -> tuple[int, ...]:
    values = [int(token) for token in raw_actions.replace(",", " ").split() if token]
    if not values:
        raise ValueError("--actions must include at least one integer action id")

    num_agents = len(action_space.spaces)
    if len(values) == 1:
        values = values * num_agents
    if len(values) != num_agents:
        raise ValueError(
            f"--actions expected 1 value or {num_agents} values, got {len(values)}"
        )

    action = tuple(values)
    if not action_space.contains(action):
        raise ValueError(f"manual action {action} is outside {action_space}")
    return action


def sample_tuple_action(
    action_space: spaces.Tuple,
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    sampled_actions = []
    for space in action_space.spaces:
        if isinstance(space, spaces.Discrete):
            sampled_actions.append(int(rng.integers(space.n)))
        else:
            space.seed(int(rng.integers(np.iinfo(np.int32).max)))
            sampled_actions.append(space.sample())
    return tuple(sampled_actions)


def print_run_header(
    args: argparse.Namespace,
    highway_config: dict[str, Any],
    complete_config: dict[str, Any],
    max_steps: int,
    action_plan: list[tuple[Any, ...]],
) -> None:
    print("=== highway intersection debug episode ===")
    print(f"target={args.target}")
    print(f"env_id={args.env_id}")
    print(f"episode_seed={args.seed}")
    action_mode = "manual" if args.actions is not None else "random"
    action_seed = args.action_seed if args.action_seed is not None else args.seed
    print(f"action={action_mode} action_seed={action_seed}")
    print(f"max_steps={max_steps}")
    print(f"highway_config={json.dumps(highway_config, sort_keys=True)}")
    print(f"first_actions={action_plan[: min(5, len(action_plan))]}")
    if args.print_config:
        print("complete_highway_config=")
        print(json.dumps(complete_config, indent=2, sort_keys=True))


def run_raw_episode(
    args: argparse.Namespace,
    complete_config: dict[str, Any],
    action_plan: list[tuple[Any, ...]],
) -> None:
    env = gym.make(args.env_id, config=complete_config, render_mode=args.render_mode)
    try:

        def step(env, action):
            obs, reward, terminated, truncated, info = env.step(action)
            return StepData(
                obs=obs,
                rewards=tuple(info.get("agents_rewards", (reward,))),
                terminated=tuple(info.get("agents_terminated", (terminated,))),
                global_terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )

        def done(sd):
            return sd.global_terminated or sd.truncated

        run_episode("raw", env, args, action_plan, step, done, env)
    finally:
        env.close()


def run_wrapper_episode(
    args: argparse.Namespace,
    highway_config: dict[str, Any],
    action_plan: list[tuple[Any, ...]],
) -> None:
    env = HighwayIntersectionMultiAgentEnv(
        argparse.Namespace(
            env_id=args.env_id,
            render_mode=args.render_mode,
            highway_config=highway_config,
        )
    )
    try:

        def step(env, action):
            action_dict = {agent: action[i] for i, agent in enumerate(env.agents)}
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            return StepData(
                obs=obs,
                rewards=tuple(float(rewards[agent]) for agent in env.agents),
                terminated=tuple(bool(terminated[agent]) for agent in env.agents),
                global_terminated=bool(info.get("global_terminated", False)),
                truncated=bool(truncated),
                info=info,
                agent_mask=tuple(bool(env.agent_mask()[agent]) for agent in env.agents),
                state=env.state(),
            )

        def done(sd):
            return sd.global_terminated or sd.truncated

        run_episode("wrapper", env, args, action_plan, step, done, env.env)
    finally:
        env.close()


def run_episode(
    target: str,
    env: Any,
    args: argparse.Namespace,
    action_plan: list[tuple[Any, ...]],
    step_fn: Any,
    done_fn: Any,
    raw_env: Any,
) -> None:
    print_env_surface_header(target, env)
    obs, info = env.reset(seed=args.seed)
    print_reset(target, obs, info, args.print_observations)
    if args.render_mode == "human":
        env.render()

    total_rewards = np.zeros(len(action_plan[0]), dtype=np.float64)
    last_truncated = False
    last_terminated = False
    steps = 0
    for step_index, action in enumerate(action_plan, start=1):
        sd = step_fn(env, action)
        steps = step_index
        total_rewards += np.asarray(sd.rewards, dtype=np.float64)
        last_terminated = sd.global_terminated
        last_truncated = sd.truncated
        print_step(
            target=target,
            step_index=step_index,
            action=action,
            rewards=sd.rewards,
            terminated=sd.terminated,
            global_terminated=sd.global_terminated,
            truncated=sd.truncated,
            info=sd.info,
            obs=sd.obs,
            print_observations=args.print_observations,
            agent_mask=sd.agent_mask,
            state=sd.state,
        )
        if args.render_mode == "human":
            env.render()
        if args.pause:
            time.sleep(args.pause)
        if done_fn(sd):
            break
    print_episode_summary(
        target,
        steps,
        total_rewards,
        last_terminated,
        last_truncated,
        controlled_vehicle_flags(raw_env),
    )


def print_env_surface_header(target: str, env: Any) -> None:
    raw_env = env.env if isinstance(env, HighwayIntersectionMultiAgentEnv) else env
    print(f"\n=== {target} surface ===")
    print(f"observation_space={getattr(env, 'observation_space', None)}")
    print(f"action_space={getattr(env, 'action_space', None)}")
    print_action_labels(raw_env)


def print_action_labels(raw_env: Any) -> None:
    action_type = getattr(raw_env.unwrapped, "action_type", None)
    action_types = getattr(action_type, "agents_action_types", [])
    if not action_types:
        return

    print("action_labels=")
    for index, agent_action_type in enumerate(action_types):
        labels = action_label_map(agent_action_type)
        if labels:
            label_text = ", ".join(
                f"{action_id}:{label}" for action_id, label in sorted(labels.items())
            )
            print(f"  agent_{index}: {label_text}")


def action_label_map(action_type: Any) -> dict[int, str]:
    raw_mapping = getattr(action_type, "actions_indexes", None)
    if not isinstance(raw_mapping, dict):
        return {}

    labels = {}
    for key, value in raw_mapping.items():
        if isinstance(key, str) and isinstance(value, int):
            labels[value] = key
        elif isinstance(key, int) and isinstance(value, str):
            labels[key] = value
    return labels


def print_reset(
    target: str,
    obs: Any,
    info: dict[str, Any],
    print_observations: bool,
) -> None:
    print(f"[{target}] reset info={info}")
    print(f"[{target}] reset obs={format_obs(obs, print_observations)}")


def print_step(
    target: str,
    step_index: int,
    action: Any,
    rewards: tuple[Any, ...],
    terminated: tuple[Any, ...],
    global_terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    obs: Any,
    print_observations: bool,
    agent_mask: tuple[Any, ...] | None = None,
    state: np.ndarray | None = None,
) -> None:
    print(f"[{target}] step={step_index:03d}")
    print(f"  action={action}")
    print(f"  rewards={format_agent_values(rewards)} total={sum(rewards):.3f}")
    print(
        "  done="
        f"agents={tuple(bool(value) for value in terminated)} "
        f"global={global_terminated} truncated={truncated}"
    )
    if agent_mask is not None:
        print(f"  mask={tuple(bool(value) for value in agent_mask)}")
    print(f"  info={format_info(info)}")
    print(f"  obs={format_obs(obs, print_observations)}")
    if state is not None:
        print(f"  state={array_summary(state)}")


def print_episode_summary(
    target: str,
    steps: int,
    total_rewards: np.ndarray,
    global_terminated: bool,
    truncated: bool,
    vehicle_flags: tuple[list[bool], list[bool]],
) -> None:
    crashed, arrived = vehicle_flags
    print(f"[{target}] summary")
    print(f"  steps={steps}")
    print(f"  total_rewards={format_agent_values(tuple(total_rewards.tolist()))}")
    print(f"  global_terminated={global_terminated} truncated={truncated}")
    print(f"  crashed_agents={crashed}")
    print(f"  arrived_agents={arrived}")


def format_agent_values(values: tuple[Any, ...]) -> str:
    return (
        "["
        + ", ".join(f"a{i}={float(value):.3f}" for i, value in enumerate(values))
        + "]"
    )


def format_info(info: dict[str, Any]) -> str:
    interesting_keys = [
        "speed",
        "crashed",
        "action",
        "rewards",
        "agents_rewards",
        "raw_agents_rewards",
        "agents_terminated",
        "global_terminated",
    ]
    compact_info = {key: info[key] for key in interesting_keys if key in info}
    return repr(compact_info)


def format_obs(obs: Any, print_observations: bool) -> str:
    if print_observations:
        return repr(obs)
    if isinstance(obs, dict):
        return (
            "{"
            + ", ".join(
                f"{agent}: {array_summary(value)}" for agent, value in obs.items()
            )
            + "}"
        )
    if isinstance(obs, tuple):
        return "(" + ", ".join(array_summary(value) for value in obs) + ")"
    return array_summary(obs)


def array_summary(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return f"shape={array.shape} empty"
    return (
        f"shape={array.shape} "
        f"dtype={array.dtype} "
        f"min={float(np.min(array)):.3f} "
        f"max={float(np.max(array)):.3f} "
        f"mean={float(np.mean(array)):.3f}"
    )


def controlled_vehicle_flags(raw_env: Any) -> tuple[list[bool], list[bool]]:
    base_env = raw_env.unwrapped
    controlled_vehicles = getattr(base_env, "controlled_vehicles", [])
    crashed = [bool(vehicle.crashed) for vehicle in controlled_vehicles]
    arrived = [bool(base_env.has_arrived(vehicle)) for vehicle in controlled_vehicles]
    return crashed, arrived


if __name__ == "__main__":
    raise SystemExit(main())

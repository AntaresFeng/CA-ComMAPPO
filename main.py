import argparse
import importlib
import sys
from argparse import Namespace
from collections.abc import Sequence
from types import ModuleType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CA-ComMAPPO project entrypoint for local experiments and checks."
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    subparsers.add_parser(
        "debug",
        add_help=False,
        help="Run one raw/wrapper highway-env debug episode.",
        description=(
            "Forward arguments to examples/debug_highway_env_episode.py. "
            "Example: uv run python main.py debug --target wrapper --actions 1"
        ),
    )

    subparsers.add_parser(
        "sanity",
        add_help=False,
        help="Run random and idle-only sanity baselines.",
        description=(
            "Forward arguments to examples/random_highway_intersection.py. "
            "Example: uv run python main.py sanity --policy random"
        ),
    )

    subparsers.add_parser(
        "smoke",
        help="Run the legacy one-step wrapper smoke check.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "debug":
        return _run_module_main("examples.debug_highway_env_episode", forwarded_args)
    if args.command == "sanity":
        return _run_module_main("examples.random_highway_intersection", forwarded_args)
    if args.command == "smoke":
        if forwarded_args:
            parser.error(f"smoke does not accept arguments: {forwarded_args}")
        return run_smoke_check()

    parser.error(f"unknown command: {args.command}")
    return 2


def _run_module_main(module_name: str, forwarded_args: Sequence[str]) -> int:
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [_module_argv0(module, module_name), *forwarded_args]
    try:
        return int(module.main() or 0)
    finally:
        sys.argv = old_argv


def _module_argv0(module: ModuleType, fallback: str) -> str:
    module_file = getattr(module, "__file__", None)
    return str(module_file) if module_file else fallback


def run_smoke_check() -> int:
    from ca_commappo.envs.highway_intersection import HighwayIntersectionMultiAgentEnv

    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={
                "controlled_vehicles": 3,
                "duration": 13,
                "spawn_probability": 0.6,
            },
        )
    )
    try:
        obs, _info = env.reset(seed=1)
        actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        _next_obs, rewards, terminated, truncated, step_info = env.step(actions)
        print(f"env_id={env.env_id}")
        print(f"agents={env.agents}")
        print(f"state_shape={env.state().shape}")
        print(f"reward_keys={list(rewards)}")
        print(f"terminated={terminated}, truncated={truncated}")
        print(f"global_terminated={step_info.get('global_terminated')}")
        print(f"obs_keys={list(obs)}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from importlib import import_module
from typing import Any


COMMAND_MODULES = {
    "debug-wrapper": "ca_commappo.envs.debug_highway_wrapper",
    "mappo": "ca_commappo.training.mappo_highway_intersection",
    "sanity": "ca_commappo.cli.run_sanity_baseline",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CA-ComMAPPO project command dispatcher."
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMAND_MODULES),
        help="Project command to run.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module = import_module(COMMAND_MODULES[args.command])
    command_main: Any = getattr(module, "main")
    return int(command_main(args.args))


if __name__ == "__main__":
    raise SystemExit(main())

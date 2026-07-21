"""Backward-compatible alias for the unified training script.

This module re-exports the unified train_highway_intersection entry points
so that existing imports (e.g. from tests or external scripts) continue to work.
"""

from xuance.common import load_yaml

from ca_commappo.evaluation.highway_metrics import print_highway_summary
from ca_commappo.evaluation.mappo_highway_metrics import evaluate_highway_policy
from ca_commappo.evaluation.mappo_video_recorder import record_mappo_policy_videos
from ca_commappo.training.train_highway_intersection import (
    AGENT_REGISTRY,
    CONFIG_DIR,
    EVAL_ENV_SEED_OFFSET,
    append_eval_run_record,
    benchmark,
    build_parser,
    cli_overrides,
    config_path_for,
    get_agent_class,
    load_configs,
    main,
    print_train_information,
    run,
    test,
    train,
    write_eval_run_metadata,
)

__all__ = [
    "AGENT_REGISTRY",
    "CONFIG_DIR",
    "EVAL_ENV_SEED_OFFSET",
    "append_eval_run_record",
    "benchmark",
    "build_parser",
    "cli_overrides",
    "config_path_for",
    "get_agent_class",
    "load_configs",
    "load_yaml",
    "main",
    "print_train_information",
    "run",
    "test",
    "train",
    "write_eval_run_metadata",
]

if __name__ == "__main__":
    raise SystemExit(main())

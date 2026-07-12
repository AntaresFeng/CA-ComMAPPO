from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("runs/mappo_highway")


def prepare_run_directory(
    configs: Namespace,
    mode: str,
    created_at: datetime | None = None,
) -> Path:
    """Create one isolated output directory and derive all per-run paths."""
    configured_output = getattr(configs, "output_dir", None) or DEFAULT_OUTPUT_DIR
    output_root = Path(configured_output).expanduser()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = (created_at or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S_%f")
    base_run_id = f"{timestamp}_{mode}_seed_{configs.seed}"

    for collision_index in range(1000):
        suffix = "" if collision_index == 0 else f"_{collision_index:03d}"
        run_id = f"{base_run_id}{suffix}"
        run_dir = output_root / run_id
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(
            f"Could not create a unique run directory under {output_root}"
        )

    configs.output_dir = str(output_root)
    configs.run_id = run_id
    configs.run_dir = str(run_dir)
    configs.log_dir = str(run_dir)
    configs.model_dir = str(run_dir / "models")
    if getattr(configs, "video_dir", None) is None:
        configs.video_dir = str(run_dir / "videos")
    return run_dir


def align_wandb_run_name(configs: Namespace, wandb_run: Any | None) -> None:
    if getattr(configs, "logger", None) == "wandb" and wandb_run is not None:
        wandb_run.name = configs.run_id

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
EVAL_METADATA_FILENAME = "eval_metadata.json"
EVAL_RECORDS_FILENAME = "eval_records.jsonl"


def eval_artifact_paths(log_dir: str | Path | None) -> tuple[Path, Path]:
    if log_dir is None or str(log_dir) == "":
        raise ValueError("agents.log_dir is required for MAPPO eval artifacts")

    run_dir = Path(log_dir)
    return run_dir / EVAL_METADATA_FILENAME, run_dir / EVAL_RECORDS_FILENAME


def json_safe(value: Any) -> Any:
    if isinstance(value, Namespace):
        return json_safe(vars(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_eval_metadata(
    configs: Namespace,
    agents: Any,
    mode: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    log_dir = getattr(agents, "log_dir", "")
    eval_artifact_paths(log_dir)
    timestamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    config_snapshot = json_safe(vars(configs))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": timestamp,
        "mode": mode,
        "log_dir": str(log_dir),
        "env_name": json_safe(getattr(configs, "env_name", None)),
        "env_id": json_safe(getattr(configs, "env_id", None)),
        "seed": json_safe(getattr(configs, "seed", None)),
        "logger": json_safe(getattr(configs, "logger", None)),
        "config": config_snapshot,
    }


def write_eval_metadata(metadata: dict[str, Any], log_dir: str | Path) -> Path:
    metadata_path, _records_path = eval_artifact_paths(log_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def build_eval_record(
    mode: str,
    phase: str,
    epoch: int | None,
    step: int,
    is_initial_eval: bool,
    is_best: bool,
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "phase": phase,
        "epoch": json_safe(epoch),
        "step": json_safe(step),
        "is_initial_eval": bool(is_initial_eval),
        "is_best": bool(is_best),
        "scores": json_safe(eval_result["scores"]),
        "summary": json_safe(eval_result["summary"]),
        "episodes": json_safe(eval_result["episodes"]),
    }


def append_eval_record_jsonl(record: dict[str, Any], log_dir: str | Path) -> Path:
    _metadata_path, records_path = eval_artifact_paths(log_dir)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(json_safe(record), sort_keys=True))
        output.write("\n")
    return records_path

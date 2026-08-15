"""Integrity metadata for benchmark-selected MAPPO checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

from ca_commappo.config import (
    TRAINING_ENV_SEED_SCHEME,
    derive_training_env_seed,
)
from ca_commappo.evaluation.mappo_eval_artifacts import json_safe

BEST_MODEL_FILENAME = "best_model.pth"
BEST_MODEL_PROVENANCE_FILENAME = "best_model_provenance.json"
PROVENANCE_SCHEMA_VERSION = 1
VERIFIED_PROVENANCE = "sidecar_sha256_verified"
UNVERIFIED_PROVENANCE = "unverified_no_sidecar"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NON_TRAINING_CONFIG_KEYS = {
    "allow_unverified_model",
    "config_path",
    "device",
    # env_seed is intentionally seed-specific.  Its fixed derivation from the
    # training seed is recorded and checked separately while the cross-seed
    # training contract remains invariant.
    "env_seed",
    "highway_config_path",
    "log_dir",
    "logger",
    "model_dir",
    "model_dir_load",
    "output_dir",
    "record_video",
    "run_dir",
    "run_id",
    "seed",
    "test_mode",
    "video_combined",
    "video_contact_sheet",
    "video_dir",
    "video_episodes",
    "video_seed",
    "wandb_user_name",
}


def sha256_file(path: str | Path) -> str:
    checkpoint_path = Path(path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path_for_model(model_path: str | Path) -> Path:
    return Path(model_path).with_name(BEST_MODEL_PROVENANCE_FILENAME)


def canonical_training_config(configs: Namespace) -> dict[str, Any]:
    """Return the path/seed/logger-independent training contract snapshot."""

    snapshot = json_safe(vars(configs))
    return {
        key: value
        for key, value in sorted(snapshot.items())
        if key not in _NON_TRAINING_CONFIG_KEYS
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_source_sha256(repository_root: str | Path | None = None) -> str:
    """Fingerprint local Python runtime sources and the locked dependencies."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    candidates = [root / "main.py", root / "pyproject.toml", root / "uv.lock"]
    candidates.extend(sorted((root / "ca_commappo").rglob("*.py")))
    digest = hashlib.sha256()
    for path in sorted({candidate.resolve() for candidate in candidates}):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def write_best_model_provenance(
    *,
    model_path: str | Path,
    configs: Namespace,
    selected_epoch: int,
    selected_step: int,
    selection_summary: dict[str, Any],
    selection_protocol: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(model_path).expanduser().resolve()
    if checkpoint_path.name != BEST_MODEL_FILENAME:
        raise ValueError(
            f"benchmark provenance requires {BEST_MODEL_FILENAME}, got "
            f"{checkpoint_path.name}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"saved benchmark checkpoint does not exist: {checkpoint_path}"
        )
    training_seed = int(configs.seed)
    training_env_seed = int(configs.env_seed)
    expected_env_seed = derive_training_env_seed(training_seed)
    if training_env_seed != expected_env_seed:
        raise ValueError(
            "formal checkpoint provenance requires the configured training-env "
            f"seed mapping; expected env_seed={expected_env_seed}, "
            f"got env_seed={training_env_seed} for seed={training_seed}"
        )

    training_config = canonical_training_config(configs)
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "verification": VERIFIED_PROVENANCE,
        "checkpoint_name": BEST_MODEL_FILENAME,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_seed": training_seed,
        "training_env_seed": training_env_seed,
        "training_env_seed_scheme": TRAINING_ENV_SEED_SCHEME,
        "source_run_id": str(getattr(configs, "run_id", "")),
        "source_config_path": str(getattr(configs, "config_path", "")),
        "training_device": str(getattr(configs, "device", "")),
        "training_config": training_config,
        "training_config_sha256": canonical_json_sha256(training_config),
        "runtime_source_sha256": runtime_source_sha256(),
        "selected_epoch": int(selected_epoch),
        "selected_step": int(selected_step),
        "selection_summary": json_safe(selection_summary),
        "selection_protocol": json_safe(selection_protocol),
    }
    sidecar_path = provenance_path_for_model(checkpoint_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(sidecar_path)
    return payload


def load_model_provenance(
    *,
    model_path: str | Path,
    expected_training_seed: int,
    expected_configs: Namespace | None = None,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(model_path).expanduser().resolve()
    if checkpoint_path.name != BEST_MODEL_FILENAME:
        raise ValueError(
            f"formal test must load {BEST_MODEL_FILENAME}, got {checkpoint_path.name}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {checkpoint_path}")

    sidecar_path = provenance_path_for_model(checkpoint_path)
    if not sidecar_path.is_file():
        if not allow_unverified:
            raise FileNotFoundError(
                f"missing checkpoint provenance sidecar: {sidecar_path}; use "
                "--allow-unverified-model only for diagnostic legacy evaluation"
            )
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "verification": UNVERIFIED_PROVENANCE,
            "checkpoint_name": BEST_MODEL_FILENAME,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_seed": None,
            "provenance_sidecar": None,
        }

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid checkpoint provenance JSON: {sidecar_path}"
        ) from error
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint provenance must be an object: {sidecar_path}")

    _validate_sidecar_payload(
        payload,
        checkpoint_path=checkpoint_path,
        expected_training_seed=int(expected_training_seed),
        expected_configs=expected_configs,
    )
    verified = dict(payload)
    verified["checkpoint_path"] = str(checkpoint_path)
    verified["provenance_sidecar"] = str(sidecar_path.resolve())
    return verified


def _validate_sidecar_payload(
    payload: dict[str, Any],
    *,
    checkpoint_path: Path,
    expected_training_seed: int,
    expected_configs: Namespace | None,
) -> None:
    expected_values = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "verification": VERIFIED_PROVENANCE,
        "checkpoint_name": BEST_MODEL_FILENAME,
        "training_seed": expected_training_seed,
        "training_env_seed": derive_training_env_seed(expected_training_seed),
        "training_env_seed_scheme": TRAINING_ENV_SEED_SCHEME,
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"checkpoint provenance mismatch: expected {field}={expected!r}, "
                f"got {payload.get(field)!r}"
            )

    stored_hash = payload.get("checkpoint_sha256")
    if not isinstance(stored_hash, str) or not _SHA256_PATTERN.fullmatch(stored_hash):
        raise ValueError("checkpoint provenance has an invalid SHA-256")
    observed_hash = sha256_file(checkpoint_path)
    if observed_hash != stored_hash:
        raise ValueError(
            "checkpoint SHA-256 does not match its provenance sidecar: "
            f"expected {stored_hash}, observed {observed_hash}"
        )

    training_config = payload.get("training_config")
    training_config_hash = payload.get("training_config_sha256")
    if not isinstance(training_config, dict):
        raise TypeError("checkpoint provenance training_config must be an object")
    if canonical_json_sha256(training_config) != training_config_hash:
        raise ValueError("checkpoint provenance training-config SHA-256 is invalid")
    if expected_configs is not None:
        expected_training_config = canonical_training_config(expected_configs)
        expected_hash = canonical_json_sha256(expected_training_config)
        if expected_hash != training_config_hash:
            raise ValueError(
                "test config does not match the checkpoint training contract: "
                f"expected {training_config_hash}, observed {expected_hash}"
            )

    source_hash = payload.get("runtime_source_sha256")
    if not isinstance(source_hash, str) or not _SHA256_PATTERN.fullmatch(source_hash):
        raise ValueError("checkpoint provenance runtime-source SHA-256 is invalid")
    current_source_hash = runtime_source_sha256()
    if current_source_hash != source_hash:
        raise ValueError(
            "runtime sources differ from the checkpoint provenance: "
            f"expected {source_hash}, observed {current_source_hash}"
        )

    for field in ("selected_epoch", "selected_step"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint provenance {field} must be non-negative")
    if (
        not isinstance(payload.get("source_run_id"), str)
        or not payload["source_run_id"]
    ):
        raise ValueError("checkpoint provenance source_run_id must be non-empty")
    if (
        not isinstance(payload.get("training_device"), str)
        or not payload["training_device"]
    ):
        raise ValueError("checkpoint provenance training_device must be non-empty")
    if not isinstance(payload.get("selection_summary"), dict):
        raise TypeError("checkpoint provenance selection_summary must be an object")
    selection_protocol = payload.get("selection_protocol")
    if (
        not isinstance(selection_protocol, dict)
        or selection_protocol.get("purpose") != "checkpoint_selection"
    ):
        raise ValueError(
            "checkpoint provenance must contain the checkpoint-selection protocol"
        )


__all__ = [
    "BEST_MODEL_FILENAME",
    "BEST_MODEL_PROVENANCE_FILENAME",
    "PROVENANCE_SCHEMA_VERSION",
    "UNVERIFIED_PROVENANCE",
    "VERIFIED_PROVENANCE",
    "canonical_json_sha256",
    "canonical_training_config",
    "load_model_provenance",
    "provenance_path_for_model",
    "runtime_source_sha256",
    "sha256_file",
    "write_best_model_provenance",
]

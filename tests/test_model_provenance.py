import json
from argparse import Namespace

import pytest

from ca_commappo.config import TRAINING_ENV_SEED_SCHEME, derive_training_env_seed
from ca_commappo.evaluation.model_provenance import (
    BEST_MODEL_PROVENANCE_FILENAME,
    UNVERIFIED_PROVENANCE,
    VERIFIED_PROVENANCE,
    load_model_provenance,
    sha256_file,
    write_best_model_provenance,
)


def _configs(tmp_path, seed=3):
    return Namespace(
        seed=seed,
        env_seed=derive_training_env_seed(seed),
        run_id=f"run_seed_{seed}",
        config_path="configs/mappo/test.yaml",
        model_dir=str(tmp_path / "models"),
        learning_rate=3e-4,
        running_steps=1_000_000,
        actor_hidden_size=[128, 128],
        device="cpu",
    )


def _write_checkpoint(tmp_path, contents=b"checkpoint"):
    model_path = tmp_path / "models" / "best_model.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(contents)
    return model_path


def _selection_protocol():
    return {
        "type": "fixed_explicit_episode_seeds",
        "purpose": "checkpoint_selection",
        "deterministic_actions": True,
        "reproducible_actions": True,
        "action_selection_mode": "deterministic_argmax",
        "base_seed": 10_000,
        "episode_count": 50,
        "parallels": 10,
        "episode_seeds": list(range(10_000, 10_050)),
    }


def test_write_and_load_verified_best_model_provenance(tmp_path):
    model_path = _write_checkpoint(tmp_path)
    written = write_best_model_provenance(
        model_path=model_path,
        configs=_configs(tmp_path),
        selected_epoch=7,
        selected_step=70_000,
        selection_summary={"episodes": 50, "arrival_rate": 0.6},
        selection_protocol=_selection_protocol(),
    )

    loaded = load_model_provenance(
        model_path=model_path,
        expected_training_seed=3,
    )

    assert written["verification"] == VERIFIED_PROVENANCE
    assert loaded["checkpoint_sha256"] == sha256_file(model_path)
    assert loaded["training_seed"] == 3
    assert loaded["training_env_seed"] == 3_000_000
    assert loaded["training_env_seed_scheme"] == TRAINING_ENV_SEED_SCHEME
    assert loaded["selected_epoch"] == 7
    assert loaded["selected_step"] == 70_000
    assert loaded["provenance_sidecar"].endswith(BEST_MODEL_PROVENANCE_FILENAME)


def test_missing_sidecar_is_only_allowed_for_explicit_legacy_diagnostics(tmp_path):
    model_path = _write_checkpoint(tmp_path)

    with pytest.raises(FileNotFoundError, match="allow-unverified-model"):
        load_model_provenance(model_path=model_path, expected_training_seed=3)

    loaded = load_model_provenance(
        model_path=model_path,
        expected_training_seed=3,
        allow_unverified=True,
    )
    assert loaded["verification"] == UNVERIFIED_PROVENANCE
    assert loaded["training_seed"] is None
    assert loaded["checkpoint_sha256"] == sha256_file(model_path)


def test_write_rejects_invalid_training_environment_seed_mapping(tmp_path):
    model_path = _write_checkpoint(tmp_path)
    configs = _configs(tmp_path, seed=3)
    configs.env_seed = 1

    with pytest.raises(ValueError, match="expected env_seed=3000000"):
        write_best_model_provenance(
            model_path=model_path,
            configs=configs,
            selected_epoch=1,
            selected_step=10_000,
            selection_summary={"episodes": 50},
            selection_protocol=_selection_protocol(),
        )


def test_load_rejects_training_seed_mismatch(tmp_path):
    model_path = _write_checkpoint(tmp_path)
    write_best_model_provenance(
        model_path=model_path,
        configs=_configs(tmp_path, seed=3),
        selected_epoch=1,
        selected_step=10_000,
        selection_summary={"episodes": 50},
        selection_protocol=_selection_protocol(),
    )

    with pytest.raises(ValueError, match="training_seed=4"):
        load_model_provenance(model_path=model_path, expected_training_seed=4)


def test_load_rejects_training_env_seed_that_does_not_match_training_seed(tmp_path):
    model_path = _write_checkpoint(tmp_path)
    write_best_model_provenance(
        model_path=model_path,
        configs=_configs(tmp_path, seed=3),
        selected_epoch=1,
        selected_step=10_000,
        selection_summary={"episodes": 50},
        selection_protocol=_selection_protocol(),
    )
    sidecar_path = model_path.with_name(BEST_MODEL_PROVENANCE_FILENAME)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["training_env_seed"] = 1
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training_env_seed=3000000"):
        load_model_provenance(model_path=model_path, expected_training_seed=3)


def test_load_rejects_checkpoint_modified_after_sidecar_write(tmp_path):
    model_path = _write_checkpoint(tmp_path)
    write_best_model_provenance(
        model_path=model_path,
        configs=_configs(tmp_path),
        selected_epoch=1,
        selected_step=10_000,
        selection_summary={"episodes": 50},
        selection_protocol=_selection_protocol(),
    )
    model_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        load_model_provenance(model_path=model_path, expected_training_seed=3)


def test_load_rejects_test_config_that_differs_from_training_contract(tmp_path):
    model_path = _write_checkpoint(tmp_path)
    training_configs = _configs(tmp_path)
    write_best_model_provenance(
        model_path=model_path,
        configs=training_configs,
        selected_epoch=1,
        selected_step=10_000,
        selection_summary={"episodes": 50},
        selection_protocol=_selection_protocol(),
    )
    test_configs = _configs(tmp_path)
    test_configs.learning_rate = 1e-4

    with pytest.raises(ValueError, match="training contract"):
        load_model_provenance(
            model_path=model_path,
            expected_training_seed=3,
            expected_configs=test_configs,
        )


def test_load_rejects_sidecar_without_checkpoint_selection_protocol(tmp_path):
    model_path = _write_checkpoint(tmp_path)
    write_best_model_provenance(
        model_path=model_path,
        configs=_configs(tmp_path),
        selected_epoch=1,
        selected_step=10_000,
        selection_summary={"episodes": 50},
        selection_protocol=_selection_protocol(),
    )
    sidecar_path = model_path.with_name(BEST_MODEL_PROVENANCE_FILENAME)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["selection_protocol"]["purpose"] = "final_test"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint-selection protocol"):
        load_model_provenance(model_path=model_path, expected_training_seed=3)

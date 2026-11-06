import json
from argparse import Namespace

import pytest
from xuance.environment import make_envs

from ca_commappo.config import (
    TRAINING_ENV_SEED_SCHEME,
    derive_training_env_seed,
    load_configs,
)
from ca_commappo.envs.highway_intersection_wrapper import DEFAULT_HIGHWAY_ENV_ID
from ca_commappo.evaluation import model_provenance
from ca_commappo.evaluation.model_provenance import (
    BEST_MODEL_PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    UNVERIFIED_LEGACY_PROVENANCE,
    UNVERIFIED_PROVENANCE,
    VERIFIED_PROVENANCE,
    canonical_json_sha256,
    load_model_provenance,
    sha256_file,
    write_best_model_provenance,
)

SOURCE_COMMIT = "a" * 40
SOURCE_MANIFEST = {
    "main.py": "b" * 64,
    "pyproject.toml": "c" * 64,
    "uv.lock": "d" * 64,
}


def _source_snapshot(*, clean=True, commit=SOURCE_COMMIT, manifest=None):
    source_manifest = dict(manifest or SOURCE_MANIFEST)
    return {
        "source_commit": commit,
        "source_tracked_tree_clean": clean,
        "runtime_source_manifest": source_manifest,
        "runtime_source_sha256": canonical_json_sha256(source_manifest),
    }


@pytest.fixture(autouse=True)
def _stable_source_snapshot(monkeypatch):
    monkeypatch.setattr(
        model_provenance,
        "repository_source_snapshot",
        lambda: _source_snapshot(),
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
        smoke=False,
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
    assert written["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert written["source_commit"] == SOURCE_COMMIT
    assert written["source_tracked_tree_clean"] is True
    assert written["runtime_source_manifest"] == SOURCE_MANIFEST
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


def test_v1_sidecar_is_only_allowed_as_unverified_legacy_diagnostic(tmp_path):
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
    payload["schema_version"] = 1
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version=2"):
        load_model_provenance(model_path=model_path, expected_training_seed=3)

    loaded = load_model_provenance(
        model_path=model_path,
        expected_training_seed=3,
        allow_unverified=True,
    )
    assert loaded["schema_version"] == 1
    assert loaded["verification"] == UNVERIFIED_LEGACY_PROVENANCE


def test_formal_sidecar_write_rejects_dirty_tracked_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_provenance,
        "repository_source_snapshot",
        lambda: _source_snapshot(clean=False),
    )

    with pytest.raises(RuntimeError, match="clean tracked Git worktree"):
        write_best_model_provenance(
            model_path=_write_checkpoint(tmp_path),
            configs=_configs(tmp_path),
            selected_epoch=1,
            selected_step=10_000,
            selection_summary={"episodes": 50},
            selection_protocol=_selection_protocol(),
        )


def test_smoke_sidecar_can_record_dirty_tree_without_becoming_formal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        model_provenance,
        "repository_source_snapshot",
        lambda: _source_snapshot(clean=False),
    )
    configs = _configs(tmp_path)
    configs.smoke = True
    model_path = _write_checkpoint(tmp_path)
    written = write_best_model_provenance(
        model_path=model_path,
        configs=configs,
        selected_epoch=1,
        selected_step=10_000,
        selection_summary={"episodes": 50},
        selection_protocol=_selection_protocol(),
    )

    loaded = load_model_provenance(
        model_path=model_path,
        expected_training_seed=3,
        expected_configs=configs,
    )
    assert written["source_tracked_tree_clean"] is False
    assert loaded["source_tracked_tree_clean"] is False


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


def test_ippo_formal_env_construction_preserves_provenance_seed(tmp_path):
    configs = load_configs(
        DEFAULT_HIGHWAY_ENV_ID,
        config_path="configs/ippo/ca-intersection-v0.yaml",
        overrides={"seed": 1, "parallels": 2},
    )
    configs.model_dir = str(tmp_path / "models")
    expected_env_seed = derive_training_env_seed(configs.seed)

    assert configs.smoke is False
    assert configs.vectorize == "SubprocVecMultiAgentEnv"
    envs = make_envs(configs)
    try:
        assert configs.env_seed == expected_env_seed
    finally:
        envs.close()

    payload = write_best_model_provenance(
        model_path=_write_checkpoint(tmp_path),
        configs=configs,
        selected_epoch=0,
        selected_step=0,
        selection_summary={"episodes": 50, "arrival_rate": 0.0},
        selection_protocol=_selection_protocol(),
    )

    assert payload["training_env_seed"] == expected_env_seed
    assert payload["training_config"]["vectorize"] == "SubprocVecMultiAgentEnv"


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


@pytest.mark.parametrize(
    "changed",
    [
        {"learning_rate": 1e-4},
        {"env_id": "ca-intersection-v0"},
        {"highway_config": {"spawn_spacing": 30}},
    ],
)
def test_load_rejects_test_config_that_differs_from_training_contract(
    tmp_path, changed
):
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
    for key, value in changed.items():
        setattr(test_configs, key, value)

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


def test_load_rejects_runtime_manifest_that_does_not_match_its_hash(tmp_path):
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
    payload["runtime_source_manifest"]["main.py"] = "e" * 64
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest hash is invalid"):
        load_model_provenance(model_path=model_path, expected_training_seed=3)


def test_load_rejects_different_current_git_commit(tmp_path, monkeypatch):
    model_path = _write_checkpoint(tmp_path)
    write_best_model_provenance(
        model_path=model_path,
        configs=_configs(tmp_path),
        selected_epoch=1,
        selected_step=10_000,
        selection_summary={"episodes": 50},
        selection_protocol=_selection_protocol(),
    )
    monkeypatch.setattr(
        model_provenance,
        "repository_source_snapshot",
        lambda: _source_snapshot(commit="f" * 40),
    )

    with pytest.raises(ValueError, match="Git commit differs"):
        load_model_provenance(model_path=model_path, expected_training_seed=3)

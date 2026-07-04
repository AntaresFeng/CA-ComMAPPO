import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from ca_commappo.evaluation.mappo_eval_artifacts import (
    EVAL_METADATA_FILENAME,
    EVAL_RECORDS_FILENAME,
    append_eval_record_jsonl,
    build_eval_metadata,
    build_eval_record,
    eval_artifact_paths,
    json_safe,
    write_eval_metadata,
)


class FakeAgent:
    def __init__(self, log_dir: Path):
        self.log_dir = str(log_dir)
        self.current_step = 42


def test_eval_artifact_paths_use_agent_log_dir(tmp_path):
    metadata_path, records_path = eval_artifact_paths(tmp_path / "run")

    assert metadata_path == tmp_path / "run" / EVAL_METADATA_FILENAME
    assert records_path == tmp_path / "run" / EVAL_RECORDS_FILENAME


def test_eval_artifact_paths_require_log_dir():
    with pytest.raises(ValueError, match="agents.log_dir"):
        eval_artifact_paths("")


def test_eval_artifact_paths_require_non_none_log_dir():
    with pytest.raises(ValueError, match="agents.log_dir"):
        eval_artifact_paths(None)


def test_json_safe_converts_nested_config_values():
    value = {
        "path": Path("models/run"),
        "int": np.int64(3),
        "float": np.float32(1.5),
        "array": np.array([1, 2]),
        "tuple": (Path("a"), np.bool_(True)),
    }

    assert json_safe(value) == {
        "path": "models/run",
        "int": 3,
        "float": pytest.approx(1.5),
        "array": [1, 2],
        "tuple": ["a", True],
    }


def test_json_safe_recursively_converts_unwrapped_numpy_scalar_values():
    value = {"date": json_safe(np.datetime64("2026-07-04"))}

    dumped = json.dumps(value)

    assert json.loads(dumped) == {"date": "2026-07-04"}


def test_build_eval_metadata_keeps_full_config_under_config(tmp_path):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=1,
        logger="tensorboard",
        running_steps=16,
        eval_interval=16,
        test_episode=1,
        highway_config={"duration": 13},
        custom_path=Path("models/run"),
        custom_np=np.int64(7),
    )
    agent = FakeAgent(tmp_path / "seed_1_run")

    metadata = build_eval_metadata(
        configs=configs,
        agents=agent,
        mode="benchmark",
        created_at="2026-07-04T19:02:28+08:00",
    )

    assert metadata["schema_version"] == 1
    assert metadata["created_at"] == "2026-07-04T19:02:28+08:00"
    assert metadata["mode"] == "benchmark"
    assert metadata["log_dir"] == str(tmp_path / "seed_1_run")
    assert metadata["env_name"] == "HighwayIntersection"
    assert metadata["env_id"] == "intersection-multi-agent-v1"
    assert metadata["seed"] == 1
    assert metadata["logger"] == "tensorboard"
    assert metadata["config"]["custom_path"] == "models/run"
    assert metadata["config"]["custom_np"] == 7
    assert metadata["config"]["highway_config"] == {"duration": 13}


def test_build_eval_metadata_supports_requested_positional_signature(tmp_path):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=2,
        logger="tensorboard",
    )
    agent = FakeAgent(tmp_path / "seed_2_run")

    metadata = build_eval_metadata(
        configs,
        agent,
        "test",
        "2026-07-04T20:00:00+08:00",
    )

    assert metadata["created_at"] == "2026-07-04T20:00:00+08:00"
    assert metadata["mode"] == "test"
    assert metadata["seed"] == 2


def test_write_metadata_and_append_jsonl_records(tmp_path):
    metadata = {"schema_version": 1, "mode": "benchmark", "config": {"seed": 1}}
    metadata_path = write_eval_metadata(metadata, tmp_path)
    first_record = {"schema_version": 1, "step": 0, "summary": {"arrival_rate": 0.0}}
    second_record = {
        "schema_version": 1,
        "step": 16,
        "summary": {"arrival_rate": 1.0},
    }

    records_path = append_eval_record_jsonl(first_record, tmp_path)
    append_eval_record_jsonl(second_record, tmp_path)

    assert metadata_path == tmp_path / EVAL_METADATA_FILENAME
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    lines = records_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first_record, second_record]


def test_build_eval_record_captures_evaluation_result():
    result = {
        "scores": [2.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0, "arrival": True}],
    }

    record = build_eval_record(
        mode="benchmark",
        phase="benchmark",
        epoch=1,
        step=16,
        is_initial_eval=False,
        is_best=True,
        eval_result=result,
    )

    assert record == {
        "schema_version": 1,
        "mode": "benchmark",
        "phase": "benchmark",
        "epoch": 1,
        "step": 16,
        "is_initial_eval": False,
        "is_best": True,
        "scores": [2.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0, "arrival": True}],
    }


def test_build_eval_record_supports_requested_positional_signature():
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "collision_rate": 0.0},
        "episodes": [{"episode_index": 0}],
    }

    record = build_eval_record("test", "test", None, 42, False, True, result)

    assert record["mode"] == "test"
    assert record["phase"] == "test"
    assert record["epoch"] is None
    assert record["step"] == 42
    assert record["is_initial_eval"] is False
    assert record["is_best"] is True

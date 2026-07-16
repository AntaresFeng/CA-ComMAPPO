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
from ca_commappo.training import mappo_highway_intersection as mappo_training


class FakeAgent:
    def __init__(self, log_dir: Path):
        self.log_dir = str(log_dir)
        self.current_step = 42

    def load_model(self, path):
        self.loaded_model_path = path


class FakeBenchmarkAgent:
    def __init__(self, log_dir: Path):
        self.log_dir = str(log_dir)
        self.current_step = 0

    def train(self, steps: int):
        self.current_step += steps

    def save_model(self, *args, **kwargs):
        raise AssertionError("save_model should not be called when save_model=False")


class FakeBenchmarkEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


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


def test_write_metadata_refuses_to_overwrite_existing_run(tmp_path):
    write_eval_metadata({"mode": "test"}, tmp_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_eval_metadata({"mode": "benchmark"}, tmp_path)


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


def test_test_mode_record_uses_null_epoch_and_best_true():
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "collision_rate": 0.0},
        "episodes": [{"episode_index": 0}],
    }

    record = build_eval_record(
        mode="test",
        phase="test",
        epoch=None,
        step=42,
        is_initial_eval=False,
        is_best=True,
        eval_result=result,
    )

    assert record["mode"] == "test"
    assert record["phase"] == "test"
    assert record["epoch"] is None
    assert record["step"] == 42
    assert record["is_initial_eval"] is False
    assert record["is_best"] is True


def test_benchmark_initial_record_marks_initial_and_best():
    result = {
        "scores": [2.0],
        "summary": {"episodes": 1, "arrival_rate": 0.0},
        "episodes": [{"episode_index": 0}],
    }

    record = build_eval_record(
        mode="benchmark",
        phase="benchmark",
        epoch=0,
        step=0,
        is_initial_eval=True,
        is_best=True,
        eval_result=result,
    )

    assert record["epoch"] == 0
    assert record["is_initial_eval"] is True
    assert record["is_best"] is True


def test_benchmark_epoch_record_marks_non_initial_best_status():
    result = {
        "scores": [0.5],
        "summary": {"episodes": 1, "arrival_rate": 0.0},
        "episodes": [{"episode_index": 0}],
    }

    record = build_eval_record(
        mode="benchmark",
        phase="benchmark",
        epoch=1,
        step=16,
        is_initial_eval=False,
        is_best=False,
        eval_result=result,
    )

    assert record["epoch"] == 1
    assert record["is_initial_eval"] is False
    assert record["is_best"] is False


def test_benchmark_writes_metadata_once_and_appends_initial_and_epoch_records(
    tmp_path, monkeypatch
):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=3,
        env_seed=21,
        logger="tensorboard",
        running_steps=16,
        parallels=1,
        eval_interval=16,
        test_episode=1,
    )
    agent = FakeBenchmarkAgent(tmp_path / "benchmark_run")
    fake_env = FakeBenchmarkEnv()
    eval_results = [
        {
            "scores": [0.1],
            "summary": {"episodes": 1, "arrival_rate": 0.0},
            "episodes": [{"episode_index": 0}],
        },
        {
            "scores": [0.9],
            "summary": {"episodes": 1, "arrival_rate": 1.0},
            "episodes": [{"episode_index": 0}],
        },
    ]
    make_envs_configs = []

    def fake_make_envs(configs_test):
        make_envs_configs.append(configs_test)
        return fake_env

    monkeypatch.setattr(mappo_training, "make_envs", fake_make_envs)
    monkeypatch.setattr(
        mappo_training,
        "evaluate_highway_policy",
        lambda **_kwargs: eval_results.pop(0),
    )

    mappo_training.benchmark(configs, agent, save_model=False)

    metadata_path = tmp_path / "benchmark_run" / EVAL_METADATA_FILENAME
    records_path = tmp_path / "benchmark_run" / EVAL_RECORDS_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]

    assert metadata["mode"] == "benchmark"
    assert len(rows) == 2
    assert rows[0]["epoch"] == 0
    assert rows[0]["is_initial_eval"] is True
    assert rows[0]["is_best"] is True
    assert rows[0]["step"] == 0
    assert rows[1]["epoch"] == 1
    assert rows[1]["is_initial_eval"] is False
    assert rows[1]["is_best"] is True
    assert rows[1]["step"] == 16
    assert len(make_envs_configs) == 1
    assert make_envs_configs[0].env_seed == configs.seed + 10_000
    assert configs.env_seed == 21
    assert fake_env.closed is True


def test_benchmark_closes_test_env_when_metadata_write_fails(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=3,
        logger="tensorboard",
        running_steps=16,
        parallels=1,
        eval_interval=16,
        test_episode=1,
    )
    agent = FakeBenchmarkAgent(tmp_path / "benchmark_run")
    fake_env = FakeBenchmarkEnv()

    def fail_metadata(*_args, **_kwargs):
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(mappo_training, "make_envs", lambda _configs: fake_env)
    monkeypatch.setattr(mappo_training, "write_eval_run_metadata", fail_metadata)

    with pytest.raises(RuntimeError, match="metadata failed"):
        mappo_training.benchmark(configs, agent, save_model=False)

    assert fake_env.closed is True


def test_test_mode_calls_video_recorder_when_enabled(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=3,
        logger="tensorboard",
        test_episode=8,
        model_dir="models/default",
        model_dir_load="models/best_model.pth",
        record_video=True,
        video_episodes=None,
        video_dir=str(tmp_path / "videos"),
        video_seed=0,
        video_contact_sheet=True,
        video_combined=True,
    )
    agent = FakeAgent(tmp_path / "test_run")
    calls = []
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0}],
    }

    monkeypatch.setattr(
        mappo_training, "evaluate_highway_policy", lambda **_kwargs: result
    )
    monkeypatch.setattr(mappo_training, "print_highway_summary", lambda _summary: None)
    monkeypatch.setattr(
        mappo_training,
        "record_mappo_policy_videos",
        lambda **kwargs: (
            calls.append(kwargs)
            or {
                "summary_path": str(tmp_path / "videos" / "video_eval_summary.json"),
                "video_dir": str(tmp_path / "videos"),
                "combined_video_path": None,
                "contact_sheet_path": None,
            }
        ),
    )

    mappo_training.test(configs, agent, envs=object())

    assert calls
    assert calls[0]["model_path"] == "models/best_model.pth"
    assert calls[0]["episode_count"] == 6
    assert calls[0]["base_seed"] == 0
    assert calls[0]["make_contact_sheet"] is True
    assert calls[0]["make_combined_video"] is True


def test_test_mode_skips_video_recorder_by_default(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=3,
        logger="tensorboard",
        test_episode=1,
        model_dir="models/default",
        record_video=False,
    )
    agent = FakeAgent(tmp_path / "test_run")
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0}],
    }

    monkeypatch.setattr(
        mappo_training, "evaluate_highway_policy", lambda **_kwargs: result
    )
    monkeypatch.setattr(mappo_training, "print_highway_summary", lambda _summary: None)
    monkeypatch.setattr(
        mappo_training,
        "record_mappo_policy_videos",
        lambda **_kwargs: pytest.fail("video recorder should not be called"),
    )

    mappo_training.test(configs, agent, envs=object())


def test_load_configs_stores_config_path(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        mappo_training,
        "load_yaml",
        lambda file_dir: {
            "env_name": "HighwayIntersection",
            "env_id": "intersection-multi-agent-v1",
        },
    )

    configs = mappo_training.load_configs(config_path=str(config_path))

    assert configs.config_path == str(config_path)

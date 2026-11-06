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
from ca_commappo.training import train_highway_intersection as mappo_training


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
        env_id="ca-intersection-v0",
        seed=1,
        logger="tensorboard",
        running_steps=16,
        eval_interval=16,
        test_episode=1,
        checkpoint_eval_episodes=1,
        checkpoint_eval_seed=10_000,
        final_eval_seed=20_000,
        eval_parallels=1,
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
    assert metadata["env_id"] == "ca-intersection-v0"
    assert metadata["seed"] == 1
    assert metadata["logger"] == "tensorboard"
    assert metadata["config"]["custom_path"] == "models/run"
    assert metadata["config"]["custom_np"] == 7
    assert metadata["config"]["highway_config"] == {"duration": 13}


def test_build_eval_metadata_supports_requested_positional_signature(tmp_path):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="ca-intersection-v0",
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
        env_id="ca-intersection-v0",
        seed=3,
        env_seed=21,
        logger="tensorboard",
        running_steps=16,
        parallels=1,
        eval_interval=16,
        test_episode=1,
        checkpoint_eval_episodes=1,
        checkpoint_eval_seed=10_000,
        final_eval_seed=20_000,
        eval_parallels=1,
    )
    agent = FakeBenchmarkAgent(tmp_path / "benchmark_run")
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
    evaluation_calls = []

    def fake_evaluate_fixed_seed_policy(**kwargs):
        evaluation_calls.append(kwargs)
        return eval_results.pop(0)

    monkeypatch.setattr(
        mappo_training,
        "evaluate_fixed_seed_policy",
        fake_evaluate_fixed_seed_policy,
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
    assert len(evaluation_calls) == 2
    assert all(
        call["protocol"]["episode_seeds"] == [10_000] for call in evaluation_calls
    )
    assert configs.env_seed == 21
    assert metadata["evaluation_protocol"]["episode_seeds"] == [10_000]


def test_benchmark_closes_test_env_when_metadata_write_fails(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="ca-intersection-v0",
        seed=3,
        logger="tensorboard",
        running_steps=16,
        parallels=1,
        eval_interval=16,
        test_episode=1,
        checkpoint_eval_episodes=1,
        checkpoint_eval_seed=10_000,
        final_eval_seed=20_000,
        eval_parallels=1,
    )
    agent = FakeBenchmarkAgent(tmp_path / "benchmark_run")
    make_envs_called = False

    def fail_metadata(*_args, **_kwargs):
        raise RuntimeError("metadata failed")

    def fail_if_make_envs_called(_configs):
        nonlocal make_envs_called
        make_envs_called = True
        raise AssertionError("evaluation env must not be allocated before metadata")

    monkeypatch.setattr(mappo_training, "make_envs", fail_if_make_envs_called)
    monkeypatch.setattr(mappo_training, "write_eval_run_metadata", fail_metadata)

    with pytest.raises(RuntimeError, match="metadata failed"):
        mappo_training.benchmark(configs, agent, save_model=False)

    assert make_envs_called is False


def test_test_mode_calls_video_recorder_when_enabled(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="ca-intersection-v0",
        seed=3,
        logger="tensorboard",
        test_episode=8,
        model_dir="models/default",
        model_dir_load="models/best_model.pth",
        record_video=True,
        video_episodes=None,
        video_dir=str(tmp_path / "videos"),
        video_seed=None,
        video_contact_sheet=True,
        video_combined=True,
        final_eval_seed=20_000,
        eval_parallels=2,
    )
    agent = FakeAgent(tmp_path / "test_run")
    calls = []
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0}],
    }

    monkeypatch.setattr(
        mappo_training, "evaluate_fixed_seed_policy", lambda **_kwargs: result
    )
    monkeypatch.setattr(
        mappo_training,
        "load_model_provenance",
        lambda **_kwargs: {"verification": "sidecar_sha256_verified"},
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
    assert calls[0]["base_seed"] == 20_000
    assert calls[0]["make_contact_sheet"] is True
    assert calls[0]["make_combined_video"] is True
    assert calls[0]["model_provenance"]["verification"] == ("sidecar_sha256_verified")


def test_test_mode_skips_video_recorder_by_default(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="ca-intersection-v0",
        seed=3,
        logger="tensorboard",
        test_episode=1,
        model_dir="models/default",
        record_video=False,
        final_eval_seed=20_000,
        eval_parallels=1,
    )
    agent = FakeAgent(tmp_path / "test_run")
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0}],
    }

    monkeypatch.setattr(
        mappo_training, "evaluate_fixed_seed_policy", lambda **_kwargs: result
    )
    monkeypatch.setattr(
        mappo_training,
        "load_model_provenance",
        lambda **_kwargs: {"verification": "sidecar_sha256_verified"},
    )
    monkeypatch.setattr(mappo_training, "print_highway_summary", lambda _summary: None)
    monkeypatch.setattr(
        mappo_training,
        "record_mappo_policy_videos",
        lambda **_kwargs: pytest.fail("video recorder should not be called"),
    )

    mappo_training.test(configs, agent, envs=object())


def test_load_configs_stores_config_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    configs = mappo_training.load_configs(config_path=str(config_path))

    assert configs.config_path == str(config_path)


def test_formal_benchmark_checks_clean_source_before_creating_run(monkeypatch):
    configs = Namespace(smoke=False)
    prepared = False

    def reject_dirty_source():
        raise RuntimeError("dirty source")

    def mark_prepared(*_args):
        nonlocal prepared
        prepared = True

    monkeypatch.setattr(
        mappo_training, "require_clean_source_checkout", reject_dirty_source
    )
    monkeypatch.setattr(mappo_training, "prepare_run_directory", mark_prepared)

    with pytest.raises(RuntimeError, match="dirty source"):
        mappo_training.run(configs, mode="benchmark")

    assert prepared is False


def test_order_eval_runs_baseline_and_each_permutation_seed(tmp_path, monkeypatch):
    configs = Namespace(
        seed=1,
        smoke=True,
        model_dir_load="models/best_model.pth",
        final_eval_seed=20_000,
        test_episode=1,
        eval_parallels=1,
        order_permutation_seeds=[30_000, 30_001],
        highway_config={
            "observation": {
                "observation_config": {
                    "vehicles_count": 3,
                    "features": ["presence", "x"],
                }
            }
        },
    )
    agent = FakeAgent(tmp_path / "order_run")
    calls = []
    artifact_call = {}

    monkeypatch.setattr(
        mappo_training,
        "load_model_provenance",
        lambda **_kwargs: {"verification": "sidecar_sha256_verified"},
    )
    monkeypatch.setattr(
        mappo_training,
        "evaluate_fixed_seed_policy",
        lambda **kwargs: (
            calls.append(kwargs)
            or {
                "episodes": [{"seed": 20_000}],
                "summary": {"episodes": 1},
                "scores": [0.0],
            }
        ),
    )
    monkeypatch.setattr(
        mappo_training,
        "write_order_robustness_artifacts",
        lambda **kwargs: (
            artifact_call.update(kwargs)
            or {
                "metadata": tmp_path / "metadata.json",
                "pairs": tmp_path / "pairs.jsonl",
                "summary": tmp_path / "summary.json",
            }
        ),
    )
    monkeypatch.setattr(mappo_training, "print_highway_summary", lambda _summary: None)

    mappo_training.order_robustness_test(configs, agent)

    assert agent.loaded_model_path == "models/best_model.pth"
    assert len(calls) == 3
    assert calls[0].get("observation_transform") is None
    assert callable(calls[1]["observation_transform"])
    assert callable(calls[2]["observation_transform"])
    assert sorted(artifact_call["perturbed_results"]) == [30_000, 30_001]


def test_ca_trace_smoke_uses_one_case_and_writes_artifacts(tmp_path, monkeypatch):
    configs = Namespace(
        agent="CA_COMMAPPO",
        seed=1,
        smoke=True,
        model_dir_load="models/best_model.pth",
        final_eval_seed=20_000,
    )
    agent = FakeAgent(tmp_path / "trace_run")
    collected = {}

    monkeypatch.setattr(
        mappo_training,
        "load_model_provenance",
        lambda **_kwargs: {"verification": "sidecar_sha256_verified"},
    )
    monkeypatch.setattr(
        mappo_training,
        "collect_ca_trace_episodes",
        lambda **kwargs: collected.update(kwargs) or [{"episode_seed": 20_000}],
    )
    monkeypatch.setattr(
        mappo_training,
        "write_ca_trace_artifacts",
        lambda **_kwargs: {
            "metadata": tmp_path / "metadata.json",
            "records": tmp_path / "records.jsonl",
            "summary": tmp_path / "summary.json",
            "plots": [tmp_path / "plot.png", tmp_path / "plot.pdf"],
            "summary_payload": {"episode_summary": {"episodes": 1}},
        },
    )
    monkeypatch.setattr(mappo_training, "print_highway_summary", lambda _summary: None)

    mappo_training.ca_trace_test(configs, agent)

    assert agent.loaded_model_path == "models/best_model.pth"
    assert collected["protocol"]["episode_seeds"] == [20_000]


def test_benchmark_train_segments_preserve_exact_budget_and_short_tail():
    configs = Namespace(running_steps=48, parallels=4, eval_interval=20)

    assert mappo_training.benchmark_train_segments(configs) == [5, 5, 2]


def test_evaluate_fixed_seed_policy_batches_and_returns_exact_sorted_seed_set(
    monkeypatch,
):
    configs = Namespace(env_seed=1, parallels=16)
    protocol = mappo_training.fixed_seed_evaluation_protocol(
        base_seed=100,
        episode_count=5,
        parallels=2,
        purpose="final_test",
    )
    envs = []
    calls = []

    class Env:
        def __init__(self, env_seed, parallels):
            self.env_seed = env_seed
            self.parallels = parallels
            self.closed = False

        def close(self):
            self.closed = True

    class Agent:
        current_step = 123

        def __init__(self):
            self.logs = []

        def log_infos(self, values, step):
            self.logs.append((values, step))

    def fake_make_envs(batch_configs):
        env = Env(batch_configs.env_seed, batch_configs.parallels)
        envs.append(env)
        return env

    def fake_evaluate_highway_policy(**kwargs):
        seeds = kwargs["episode_seeds"]
        calls.append(kwargs)
        records = []
        for seed in reversed(seeds):
            records.append(
                {
                    "seed": seed,
                    "score": float(seed),
                    "episode_reward": float(seed),
                    "agent_rewards": {"agent_0": float(seed)},
                    "steps": 1,
                    "collision": False,
                    "arrival": True,
                    "truncated": False,
                }
            )
        return {"episodes": records}

    monkeypatch.setattr(mappo_training, "make_envs", fake_make_envs)
    monkeypatch.setattr(
        mappo_training, "evaluate_highway_policy", fake_evaluate_highway_policy
    )
    agent = Agent()

    result = mappo_training.evaluate_fixed_seed_policy(
        configs=configs,
        agents=agent,
        protocol=protocol,
        phase="test",
        log_prefix="Test-Highway",
    )

    assert [record["seed"] for record in result["episodes"]] == list(range(100, 105))
    assert [env.env_seed for env in envs] == [100, 102, 104]
    assert [env.parallels for env in envs] == [2, 2, 1]
    assert all(env.closed for env in envs)
    assert all(call["deterministic"] is True for call in calls)
    assert all(call["record_once_per_env"] is True for call in calls)
    assert configs.env_seed == 1
    assert configs.parallels == 16
    assert agent.logs[0][1] == 123


@pytest.mark.parametrize(
    "configs, message",
    [
        (Namespace(running_steps=10, parallels=4, eval_interval=8), "running_steps"),
        (Namespace(running_steps=16, parallels=4, eval_interval=10), "eval_interval"),
    ],
)
def test_benchmark_train_segments_reject_ambiguous_vector_budgets(configs, message):
    with pytest.raises(ValueError, match=message):
        mappo_training.benchmark_train_segments(configs)

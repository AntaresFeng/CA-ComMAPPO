# MAPPO Evaluation JSONL Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save MAPPO formal test/benchmark evaluation metadata and per-evaluation records beside the main TensorBoard log files.

**Architecture:** Add a focused artifact helper module for JSON-safe conversion, metadata construction, and JSONL appends. Keep `evaluate_highway_policy(...)` responsible only for collecting records and summaries; wire persistence from `ca_commappo/training/mappo_highway_intersection.py` using `agents.log_dir` as the run directory.

**Tech Stack:** Python 3.12, pytest, JSON/JSONL, pathlib, NumPy, XuanCe MAPPO TensorBoard log directories.

---

## File Structure

- Create `ca_commappo/evaluation/mappo_eval_artifacts.py`: owns `eval_metadata.json` / `eval_records.jsonl` filenames, JSON-safe conversion, metadata construction, record construction, JSON writing, and JSONL append.
- Add `tests/test_mappo_eval_artifacts.py`: focused unit tests for artifact paths, metadata schema, JSON-safe conversion, JSONL append behavior, and record construction.
- Modify `ca_commappo/training/mappo_highway_intersection.py`: writes metadata once for `test` / `benchmark`, appends one JSONL line after each formal evaluation, and leaves pure `train` unchanged.
- Optionally extend `tests/test_mappo_highway_metrics.py` only if existing fake-agent fixtures need a `log_dir` attribute for integration-facing assertions.
- Do not modify vendored `examples/` or XuanCe `.venv` code.

### Task 1: Artifact Helper Module

**Files:**
- Create: `ca_commappo/evaluation/mappo_eval_artifacts.py`
- Test: `tests/test_mappo_eval_artifacts.py`

- [ ] **Step 1: Write failing metadata and JSON-safe tests**

Create `tests/test_mappo_eval_artifacts.py` with these tests:

```python
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


def test_write_metadata_and_append_jsonl_records(tmp_path):
    metadata = {
        "schema_version": 1,
        "mode": "benchmark",
        "config": {"seed": 1},
    }
    metadata_path = write_eval_metadata(metadata, tmp_path)
    first_record = {"schema_version": 1, "step": 0, "summary": {"arrival_rate": 0.0}}
    second_record = {"schema_version": 1, "step": 16, "summary": {"arrival_rate": 1.0}}

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'ca_commappo.evaluation.mappo_eval_artifacts'`.

- [ ] **Step 3: Implement the artifact helper module**

Create `ca_commappo/evaluation/mappo_eval_artifacts.py`:

```python
import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
EVAL_METADATA_FILENAME = "eval_metadata.json"
EVAL_RECORDS_FILENAME = "eval_records.jsonl"


def eval_artifact_paths(log_dir: str | Path) -> tuple[Path, Path]:
    if log_dir is None or str(log_dir) == "":
        raise ValueError("agents.log_dir is required for MAPPO eval artifacts")
    run_dir = Path(log_dir)
    return run_dir / EVAL_METADATA_FILENAME, run_dir / EVAL_RECORDS_FILENAME


def json_safe(value: Any) -> Any:
    if isinstance(value, Namespace):
        return json_safe(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_eval_metadata(
    *,
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
        "env_name": getattr(configs, "env_name", None),
        "env_id": getattr(configs, "env_id", None),
        "seed": getattr(configs, "seed", None),
        "logger": getattr(configs, "logger", None),
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
    *,
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
        "epoch": epoch,
        "step": int(step),
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
```

- [ ] **Step 4: Run artifact helper tests**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit artifact helper**

Run:

```powershell
git add ca_commappo/evaluation/mappo_eval_artifacts.py tests/test_mappo_eval_artifacts.py
git commit -m "feat: add mappo eval artifact writers"
```

Expected: commit succeeds with only the helper module and its tests.

### Task 2: Test Mode JSONL Wiring

**Files:**
- Modify: `ca_commappo/training/mappo_highway_intersection.py`
- Test: `tests/test_mappo_eval_artifacts.py`

- [ ] **Step 1: Add a focused test for test-mode persistence helper behavior**

Append this test to `tests/test_mappo_eval_artifacts.py`:

```python
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
```

- [ ] **Step 2: Run the new focused test**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py::test_test_mode_record_uses_null_epoch_and_best_true -q
```

Expected: pass once Task 1 is implemented.

- [ ] **Step 3: Import artifact helpers in the training entrypoint**

Modify the imports near the top of `ca_commappo/training/mappo_highway_intersection.py`:

```python
from ca_commappo.evaluation.mappo_eval_artifacts import (
    append_eval_record_jsonl,
    build_eval_metadata,
    build_eval_record,
    write_eval_metadata,
)
```

- [ ] **Step 4: Add a metadata writer helper**

Add this helper below `print_train_information(...)` in `ca_commappo/training/mappo_highway_intersection.py`:

```python
def write_eval_run_metadata(
    configs: argparse.Namespace,
    agents: MAPPO_Agents,
    mode: str,
) -> None:
    metadata = build_eval_metadata(configs=configs, agents=agents, mode=mode)
    metadata_path = write_eval_metadata(metadata, agents.log_dir)
    print(f"Eval metadata saved: {metadata_path}")
```

- [ ] **Step 5: Wire JSONL output in `test()`**

In `test(...)`, call `write_eval_run_metadata(...)` before evaluation and append the test record after `evaluate_highway_policy(...)`:

```python
def test(configs: argparse.Namespace, agents: MAPPO_Agents, envs) -> None:
    model_path = getattr(configs, "model_dir_load", configs.model_dir)
    agents.load_model(path=model_path)
    write_eval_run_metadata(configs, agents, mode="test")
    result = evaluate_highway_policy(
        agents=agents,
        envs=envs,
        test_episodes=configs.test_episode,
        phase="test",
        log_prefix="Test-Highway",
    )
    record = build_eval_record(
        mode="test",
        phase="test",
        epoch=None,
        step=agents.current_step,
        is_initial_eval=False,
        is_best=True,
        eval_result=result,
    )
    records_path = append_eval_record_jsonl(record, agents.log_dir)
    scores = result["scores"]
    print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
    print_highway_summary(result["summary"])
    print(f"Eval records saved: {records_path}")
    print("Finish testing.")
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py tests/test_mappo_highway_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit test-mode wiring**

Run:

```powershell
git add ca_commappo/training/mappo_highway_intersection.py tests/test_mappo_eval_artifacts.py
git commit -m "feat: save mappo test eval jsonl"
```

Expected: commit succeeds with training entrypoint and test additions.

### Task 3: Benchmark JSONL Wiring

**Files:**
- Modify: `ca_commappo/training/mappo_highway_intersection.py`
- Test: `tests/test_mappo_eval_artifacts.py`

- [ ] **Step 1: Add benchmark record tests**

Append these tests to `tests/test_mappo_eval_artifacts.py`:

```python
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
```

- [ ] **Step 2: Run benchmark record tests**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py::test_benchmark_initial_record_marks_initial_and_best tests/test_mappo_eval_artifacts.py::test_benchmark_epoch_record_marks_non_initial_best_status -q
```

Expected: both tests pass once Task 1 is implemented.

- [ ] **Step 3: Add a small append helper in the training entrypoint**

Add this helper below `write_eval_run_metadata(...)`:

```python
def append_eval_run_record(
    *,
    agents: MAPPO_Agents,
    mode: str,
    phase: str,
    epoch: int | None,
    is_initial_eval: bool,
    is_best: bool,
    eval_result: dict[str, Any],
) -> Path:
    record = build_eval_record(
        mode=mode,
        phase=phase,
        epoch=epoch,
        step=agents.current_step,
        is_initial_eval=is_initial_eval,
        is_best=is_best,
        eval_result=eval_result,
    )
    return append_eval_record_jsonl(record, agents.log_dir)
```

- [ ] **Step 4: Refactor `test()` to use the append helper**

Replace the inline `build_eval_record(...)` / `append_eval_record_jsonl(...)` block in `test()` with:

```python
    records_path = append_eval_run_record(
        agents=agents,
        mode="test",
        phase="test",
        epoch=None,
        is_initial_eval=False,
        is_best=True,
        eval_result=result,
    )
```

- [ ] **Step 5: Wire metadata and initial benchmark record**

At the start of `benchmark(...)`, after `test_envs = make_envs(configs_test)` and before the first evaluation, write metadata:

```python
    write_eval_run_metadata(configs, agents, mode="benchmark")
```

After the initial `best_scores_info` is created and before optional model save, append the initial evaluation record:

```python
        records_path = append_eval_run_record(
            agents=agents,
            mode="benchmark",
            phase="benchmark",
            epoch=0,
            is_initial_eval=True,
            is_best=True,
            eval_result=eval_result,
        )
        print(f"Eval records saved: {records_path}")
```

- [ ] **Step 6: Wire per-epoch benchmark records**

Inside the benchmark loop, compute `is_best` once, append the record immediately after evaluation, and then update best state:

```python
            is_best = is_better_highway_summary(
                eval_result["summary"], best_scores_info["summary"]
            )
            records_path = append_eval_run_record(
                agents=agents,
                mode="benchmark",
                phase="benchmark",
                epoch=epoch + 1,
                is_initial_eval=False,
                is_best=is_best,
                eval_result=eval_result,
            )
            print(f"Eval records saved: {records_path}")
            if is_best:
                best_scores_info = {
                    "mean": mean_score,
                    "std": np.std(test_scores),
                    "step": agents.current_step,
                    "summary": eval_result["summary"],
                }
                if save_model:
                    agents.save_model(model_name="best_model.pth")
```

- [ ] **Step 7: Run focused tests**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py tests/test_mappo_highway_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit benchmark wiring**

Run:

```powershell
git add ca_commappo/training/mappo_highway_intersection.py tests/test_mappo_eval_artifacts.py
git commit -m "feat: save mappo benchmark eval jsonl"
```

Expected: commit succeeds with benchmark persistence changes.

### Task 4: Smoke Verification And Documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-07-04-mappo-eval-jsonl-output-design.md` if implementation reveals a schema correction
- Modify: `docs/sanity_baselines.md` only if adding a short MAPPO artifact note is useful

- [ ] **Step 1: Run full focused test set**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py tests/test_mappo_highway_metrics.py tests/test_highway_metrics.py tests/test_highway_intersection_adapter.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke benchmark**

Run:

```powershell
uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode benchmark --no-save
```

Expected: command completes, prints highway metric summaries, and prints paths for `eval_metadata.json` and `eval_records.jsonl`.

- [ ] **Step 3: Inspect generated artifact files**

Run:

```powershell
Get-ChildItem logs\mappo_highway\seed_* -Filter eval_metadata.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1 FullName,Length,LastWriteTime
Get-ChildItem logs\mappo_highway\seed_* -Filter eval_records.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1 FullName,Length,LastWriteTime
```

Expected: both commands show the latest run directory.

- [ ] **Step 4: Validate JSON and JSONL shape**

Run:

```powershell
$metadata = Get-ChildItem logs\mappo_highway\seed_* -Filter eval_metadata.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$records = Get-ChildItem logs\mappo_highway\seed_* -Filter eval_records.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python -c "import json, pathlib; p=pathlib.Path(r'$($metadata.FullName)'); data=json.loads(p.read_text(encoding='utf-8')); assert data['schema_version']==1; assert 'config' in data; print(data['mode'], data['env_id'])"
python -c "import json, pathlib; p=pathlib.Path(r'$($records.FullName)'); rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]; assert rows; assert all('summary' in row and 'episodes' in row for row in rows); print(len(rows), rows[-1]['step'])"
```

Expected: first Python command prints the run mode and env id; second prints the number of JSONL rows and last step.

- [ ] **Step 5: Run diff and whitespace checks**

Run:

```powershell
git diff --check
git status --short --untracked-files=all
```

Expected: no whitespace errors. Status should include only intended source, test, and optional docs changes.

- [ ] **Step 6: Commit verification/docs adjustments**

If Task 4 modified docs, run:

```powershell
git add docs/superpowers/specs/2026-07-04-mappo-eval-jsonl-output-design.md docs/sanity_baselines.md
git commit -m "docs: document mappo eval artifacts"
```

Expected: commit succeeds if docs changed. If no docs changed, skip this commit.

## Self-Review

Spec coverage:
- Metadata JSON under `Path(agents.log_dir)` is implemented in Task 1 and wired in Tasks 2-3.
- Append-only `eval_records.jsonl` is implemented in Task 1 and wired for both `test` and `benchmark` in Tasks 2-3.
- Full `vars(configs)` under `config` is covered by `test_build_eval_metadata_keeps_full_config_under_config`.
- Immediate append after every formal evaluation is covered by Task 3 benchmark wiring and Task 4 smoke validation.
- Pure `train()` remains unchanged by design.

Placeholder scan:
- The plan contains concrete filenames, function names, test names, command lines, and expected outputs.
- No `TBD`, `TODO`, or unspecified "add tests" steps remain.

Type consistency:
- Artifact helper functions use the same names in tests, implementation snippets, and training-entrypoint snippets.
- JSONL filenames match the user-approved prefixless names: `eval_metadata.json` and `eval_records.jsonl`.

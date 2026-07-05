# Highway Metrics Training Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared highway task metrics and wire them into sanity, MAPPO training telemetry, benchmark evaluation, and test evaluation.

**Architecture:** The adapter exposes raw episode facts in `info`; `highway_metrics.py` owns pure metric semantics; `mappo_highway_metrics.py` bridges XuanCe callbacks and learned-policy evaluation; the training entrypoint orchestrates the integration. Sanity and MAPPO evaluation share summary code.

**Tech Stack:** Python 3.12, pytest, Gymnasium/highway-env, XuanCe MAPPO, TensorBoard/W&B via XuanCe `log_infos`.

---

## File Structure

- Create `ca_commappo/evaluation/highway_metrics.py`: pure helpers for outcomes, episode records, summaries, log key formatting, JSON persistence. It must not reach through env wrappers for vehicle state.
- Create `ca_commappo/evaluation/mappo_highway_metrics.py`: `HighwayMetricsCallback`, learned-policy evaluation loop, best-model comparison helper.
- Modify `ca_commappo/envs/highway_intersection_wrapper.py`: add per-agent `info["arrived"]`.
- Modify `ca_commappo/evaluation/sanity_baseline_runner.py`: reuse `highway_metrics` helpers.
- Modify `ca_commappo/cli/run_sanity_baseline.py`: use shared summary formatter.
- Modify `ca_commappo/training/mappo_highway_intersection.py`: attach callback and use explicit highway evaluator for `test`/`benchmark`.
- Add `tests/test_highway_metrics.py`: pure metric and summary tests.
- Add `tests/test_mappo_highway_metrics.py`: fake-agent/fake-env tests for evaluator and callback behavior.
- Modify `tests/test_highway_intersection_adapter.py`: assert `info["arrived"]` contract.

### Task 1: Shared Metric Semantics

**Files:**
- Create: `ca_commappo/evaluation/highway_metrics.py`
- Test: `tests/test_highway_metrics.py`

- [ ] **Step 1: Write failing tests**

Add tests covering collision precedence, arrival, truncation, optional mean fields, and log key formatting.

Run: `uv run pytest tests/test_highway_metrics.py -q`
Expected: fail because `ca_commappo.evaluation.highway_metrics` does not exist.

- [ ] **Step 2: Implement pure helpers**

Implement `episode_outcome`, `build_episode_record`, `summarize_episode_records`, `summary_to_log_infos`, `format_summary_lines`, and `save_results_json`.

- [ ] **Step 3: Verify shared metric tests**

Run: `uv run pytest tests/test_highway_metrics.py -q`
Expected: pass.

### Task 2: Adapter Episode Facts

**Files:**
- Modify: `ca_commappo/envs/highway_intersection_wrapper.py`
- Modify: `tests/test_highway_intersection_adapter.py`

- [ ] **Step 1: Write failing adapter assertions**

Assert `step_info["arrived"]` is a per-agent tuple and matches observed arrival states in the existing arrival test.

Run: `uv run pytest tests/test_highway_intersection_adapter.py::test_arrival_rewards_and_termination_match_highway_info -q`
Expected: fail on missing `arrived`.

- [ ] **Step 2: Add adapter `arrived` info**

In `step()`, compute `arrived = tuple(bool(self.env.unwrapped.has_arrived(v)) for v in self.env.unwrapped.controlled_vehicles)` and set `info["arrived"] = arrived`.

- [ ] **Step 3: Verify adapter assertions**

Run: `uv run pytest tests/test_highway_intersection_adapter.py::test_arrival_rewards_and_termination_match_highway_info -q`
Expected: pass.

### Task 3: Sanity Runner Refactor

**Files:**
- Modify: `ca_commappo/evaluation/sanity_baseline_runner.py`
- Modify: `ca_commappo/cli/run_sanity_baseline.py`
- Test: `tests/test_highway_metrics.py`

- [ ] **Step 1: Add sanity compatibility test**

Add a test that two known episode records summarize through `summarize_episode_records` with existing sanity keys.

Run: `uv run pytest tests/test_highway_metrics.py -q`
Expected: pass once shared module is present.

- [ ] **Step 2: Refactor imports**

Import shared `build_episode_record`, `summarize_episode_records`, `save_results_json`, and `format_summary_lines`. In the sanity runner, read `info["crashed"]` and `info["arrived"]` from the adapter step info, then pass those facts into `build_episode_record`. Remove local duplicate implementations from the sanity runner and CLI summary formatter.

- [ ] **Step 3: Verify sanity path**

Run: `uv run pytest tests/test_highway_metrics.py tests/test_highway_intersection_adapter.py -q`
Expected: pass.

### Task 4: XuanCe Metrics Callback And Evaluator

**Files:**
- Create: `ca_commappo/evaluation/mappo_highway_metrics.py`
- Test: `tests/test_mappo_highway_metrics.py`

- [ ] **Step 1: Write fake-env evaluator tests**

Test that `evaluate_highway_policy` returns episode records, summary, and scores from a fake vector env and fake agent.

Run: `uv run pytest tests/test_mappo_highway_metrics.py -q`
Expected: fail because module does not exist.

- [ ] **Step 2: Implement evaluator**

Implement a policy loop using `agents.action(...)`, `envs.step(...)`, `envs.reset()`, `envs.buf_state`, `envs.buf_avail_actions`, and `envs.buf_obs`, matching the XuanCe on-policy test loop enough for non-RNN current configs.

- [ ] **Step 3: Implement callback**

Implement `HighwayMetricsCallback.on_train_episode_info` to read terminal `infos[env_id]`, build an episode record, summarize the single record, and log scalars through a supplied logger callback or `agents.log_infos` wrapper.

- [ ] **Step 4: Verify evaluator/callback tests**

Run: `uv run pytest tests/test_mappo_highway_metrics.py -q`
Expected: pass.

### Task 5: Training Entrypoint Integration

**Files:**
- Modify: `ca_commappo/training/mappo_highway_intersection.py`
- Test: `tests/test_mappo_highway_metrics.py`

- [ ] **Step 1: Add integration-facing tests**

Test best-summary comparison helper: higher arrival rate wins, lower collision rate breaks ties, higher reward then shorter length break remaining ties.

Run: `uv run pytest tests/test_mappo_highway_metrics.py -q`
Expected: fail until helper exists.

- [ ] **Step 2: Wire callback and evaluator**

Instantiate `HighwayMetricsCallback` before `MAPPO_Agents`, pass it to `MAPPO_Agents(config=configs, envs=envs, callback=callback)`, set the callback logger to `agents.log_infos`, replace `agents.test()` calls in `test()` and `benchmark()` with `evaluate_highway_policy(...)`, and use the task-aware comparison helper for best model selection.

- [ ] **Step 3: Verify smoke config**

Run: `uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode benchmark --no-save`
Expected: completes and prints reward plus highway metric summary.

### Task 6: Final Verification

**Files:**
- All touched files

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_highway_metrics.py tests/test_mappo_highway_metrics.py tests/test_highway_intersection_adapter.py -q`
Expected: pass.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -q`
Expected: pass.

- [ ] **Step 3: Review diff**

Run: `git diff -- ca_commappo tests docs configs`
Expected: diff only contains shared metrics, adapter signal, training integration, tests, and docs for this feature.

## Self-Review

The plan covers adapter facts, shared metrics, sanity reuse, XuanCe callback telemetry, formal learned-policy evaluation, training entrypoint integration, and verification. It avoids changing vendored examples and keeps XuanCe internals unmodified.

# MAPPO Evaluation JSONL Output Design

Date: 2026-07-04

## Goal

Persist formal MAPPO test and benchmark evaluation results next to the main TensorBoard event file. TensorBoard remains the scalar-monitoring surface; JSON/JSONL becomes the structured experiment artifact for episode records, summaries, and later post-processing.

## Files

For each MAPPO run, write two files under `Path(agents.log_dir)`:

- `eval_metadata.json`
- `eval_records.jsonl`

For TensorBoard runs, `agents.log_dir` is the run directory created by XuanCe, for example `logs/mappo_highway/seed_1_2026_0704_190228/`. Using `agents.log_dir` avoids reconstructing the seed/timestamp path in project code.

## Metadata JSON

`eval_metadata.json` is written once after `MAPPO_Agents` is created and before the first formal evaluation is recorded. It contains run-level context:

```json
{
  "schema_version": 1,
  "created_at": "2026-07-04T19:02:28+08:00",
  "mode": "benchmark",
  "log_dir": "logs/mappo_highway/seed_1_2026_0704_190228",
  "env_name": "HighwayIntersection",
  "env_id": "intersection-multi-agent-v1",
  "seed": 1,
  "logger": "tensorboard",
  "config": {
    "env_name": "HighwayIntersection",
    "env_id": "intersection-multi-agent-v1",
    "seed": 1,
    "running_steps": 16,
    "eval_interval": 16,
    "test_episode": 1,
    "highway_config": {}
  }
}
```

The `config` object should be the full JSON-safe snapshot of `vars(configs)` after YAML loading and CLI overrides. Common fields are duplicated at the top level only for convenient post-processing.

## Evaluation JSONL

`eval_records.jsonl` is append-only. Each completed formal evaluation appends one JSON object on one line:

```json
{"schema_version":1,"mode":"benchmark","phase":"benchmark","epoch":0,"step":0,"is_initial_eval":true,"is_best":true,"scores":[2.4],"summary":{"episodes":1,"collision_rate":0.0,"arrival_rate":0.0},"episodes":[]}
```

Required fields per line:

- `schema_version`: integer, currently `1`.
- `mode`: training entrypoint mode, usually `benchmark` or `test`.
- `phase`: evaluator phase passed to `evaluate_highway_policy`.
- `epoch`: integer epoch index for benchmark; `null` for test mode.
- `step`: `agents.current_step`, matching TensorBoard scalar step.
- `is_initial_eval`: true only for benchmark evaluation before the first training epoch.
- `is_best`: true when this evaluation becomes the current best according to `is_better_highway_summary`.
- `scores`: evaluator score list.
- `summary`: aggregated highway metrics.
- `episodes`: complete per-episode records returned by `evaluate_highway_policy`.

## Write Timing

`test()` writes metadata once, runs one `evaluate_highway_policy(...)`, then appends one JSONL record.

`benchmark()` writes metadata once, appends the initial evaluation at step 0, then appends one record after each evaluation epoch. The JSONL file is updated immediately after each evaluation so interrupted long training keeps completed evaluation records.

Pure `train()` does not write these files because it does not run formal evaluation. Training episode telemetry remains in TensorBoard through `Train-Highway/*`.

## Serialization

Before writing JSON, convert values into JSON-safe Python types:

- `Path` to string.
- NumPy scalar and array values to Python scalars/lists.
- Non-serializable objects to strings only as a last resort.

The existing `save_results_json` helper can remain for whole-object JSON files, but JSONL appending should use a dedicated helper to avoid rewriting the full record history.

## Error Handling

If `agents.log_dir` is missing, fail with a clear error because colocating records with the main run directory is part of the contract.

Writing one JSONL line should create the parent directory if needed, append a trailing newline, and flush by closing the file. A failed append should surface as an exception rather than silently dropping evaluation artifacts.

## Testing

Add focused tests for:

- Metadata construction includes top-level fields and full `config`.
- JSON-safe conversion handles `Path`, NumPy scalar, and nested containers.
- JSONL append writes one valid JSON object per line and preserves previous lines.
- Benchmark recording marks the initial evaluation and best evaluation correctly.
- Training entrypoint helper chooses `Path(agents.log_dir) / "eval_records.jsonl"`.

The smoke benchmark should then produce both files beside the TensorBoard event file:

```powershell
uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode benchmark --no-save
Get-ChildItem logs\mappo_highway\seed_*\eval_metadata.json
Get-ChildItem logs\mappo_highway\seed_*\eval_records.jsonl
```

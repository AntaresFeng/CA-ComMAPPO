# Highway Metrics Training Integration Design

Date: 2026-07-04

## Goal

Add highway-intersection task metrics to MAPPO training, benchmark, and test flows while keeping sanity baselines and learned-policy evaluation on one metric definition.

## Current State

`ca_commappo/envs/highway_intersection_wrapper.py` is the active adapter. It already exposes adapter-facing masked rewards, raw highway rewards, `info["global_terminated"]`, and per-agent `info["crashed"]`. The training entrypoint still delegates `train`, `test`, and `benchmark` to XuanCe score-only methods. `ca_commappo/evaluation/sanity_baseline_runner.py` already computes collision, arrival, truncation, reward, and episode length metrics, but those definitions are local to the sanity runner.

## Design

Create a shared `ca_commappo/evaluation/highway_metrics.py` module for outcome classification, episode records, summary aggregation, log formatting, and JSON persistence. Sanity baselines, MAPPO evaluation, and training callbacks should consume this module rather than duplicating metric logic.

The adapter should expose `info["arrived"]` as a per-agent tuple alongside the already fixed `info["crashed"]`. This keeps subprocess vectorization and XuanCe callbacks from having to reach through the local Python wrapper to inspect `controlled_vehicles`.

Adapter `info` is the only runtime episode-fact boundary for training and evaluation. The shared metrics module should accept `crashed` / `arrived` facts that were already emitted by the adapter; it should not inspect `controlled_vehicles` itself.

Add `ca_commappo/evaluation/mappo_highway_metrics.py` for XuanCe-specific integration. It should include a `HighwayMetricsCallback` for training-time episode monitoring and an `evaluate_highway_policy(...)` loop for formal benchmark/test evaluation. The callback is for live training telemetry; the explicit evaluator is the source of structured episode records and summaries.

Update `ca_commappo/training/mappo_highway_intersection.py` so `MAPPO_Agents` receives the callback, `test()` uses the explicit evaluator, and `benchmark()` selects best models using task-aware metrics in addition to score reporting.

## Metric Semantics

Episode outcome precedence is collision, then arrival, then truncation. Collision is true if any controlled vehicle crashed. Arrival is true only when every controlled vehicle arrived and no collision occurred. Truncation is true only when the time limit ended the episode without collision or full arrival.

Summary metrics include `episodes`, `mean_episode_reward`, `mean_agent_reward`, `mean_episode_length`, `collision_rate`, `arrival_rate`, `truncation_rate`, `mean_agent_collision_fraction`, and `mean_agent_arrival_fraction` when episode records contain those fields.

## Error Handling

Metric functions should validate that episode records are non-empty. The evaluator should reject empty `test_episode` values through existing config validation paths or by failing with a clear `ValueError` when summary input is empty.

## Testing

Add focused tests for outcome precedence, summary aggregation, log key formatting, adapter `info["arrived"]`, sanity-runner reuse of the shared summary, and the MAPPO evaluation loop with a lightweight fake agent/vector environment. Run focused tests first, then `uv run pytest -q` if the focused suite passes.

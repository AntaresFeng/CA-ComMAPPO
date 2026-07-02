# Project Layout Reorganization Design

## Goal

Separate CA-ComMAPPO project code from upstream XuanCe reference examples without changing runtime behavior.

## Scope

This first phase only moves project-owned entrypoints and configs out of `examples/`. It does not rebuild the deleted `evaluate_*` workflow and does not add MAPPO task-metric evaluation.

## Layout

- `ca_commappo/envs/`: maintained highway-env to XuanCe environment adapters and wrapper debug tools.
- `ca_commappo/evaluation/`: maintained sanity-baseline execution and summary helpers.
- `ca_commappo/cli/`: project-owned command-line entrypoints that are not tied to one lower-level package, currently the sanity-baseline CLI.
- `ca_commappo/training/`: project-owned training entrypoints, starting with highway intersection MAPPO.
- `configs/sanity/`: reproducible sanity-baseline YAML configs.
- `configs/mappo/`: maintained highway MAPPO YAML configs.
- `examples/mappo/`: vendored upstream XuanCe MAPPO reference scripts and upstream example configs only.

## Decisions

`examples/random_highway_intersection.py`, `examples/debug_highway_env_episode.py`, and `examples/mappo/mappo_highway_intersection.py` are project code and move into the package. Their config defaults are updated to `configs/...`.

`examples/mappo/mappo_simple_spread.py`, `examples/mappo/mappo_football.py`, `examples/mappo/mappo_sc2.py`, and their upstream configs remain under `examples/mappo/` as references.

`main.py` remains a thin dispatcher. It should not own environment, sanity baseline, or training logic.

## Compatibility

The canonical commands become:

```powershell
uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy all
uv run python -m ca_commappo.envs.debug_highway_wrapper --target wrapper --seed 7
uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection_v1_smoke.yaml --mode train --no-save
```

Legacy direct paths under `examples/` are intentionally removed for project-owned code so the boundary stays visible.

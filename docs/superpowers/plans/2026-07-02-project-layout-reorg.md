# Project Layout Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CA-ComMAPPO-owned scripts and configs out of `examples/` while preserving their behavior.

**Architecture:** Project logic lives under `ca_commappo`, reusable configs live under `configs`, and `examples/mappo` is reserved for vendored upstream XuanCe references. `main.py` remains a small dispatcher into package modules.

**Tech Stack:** Python 3.10, argparse, pytest, uv, XuanCe, highway-env.

---

### Task 1: Move Project-Owned Files

**Files:**
- Move: `examples/random_highway_intersection.py` to `ca_commappo/cli/run_sanity_baseline.py`
- Move: `examples/debug_highway_env_episode.py` to `ca_commappo/envs/debug_highway_wrapper.py`
- Move: `examples/mappo/mappo_highway_intersection.py` to `ca_commappo/training/mappo_highway_intersection.py`
- Move: `examples/sanity/highway_intersection.yaml` to `configs/sanity/highway_intersection.yaml`
- Move: `examples/mappo/mappo_highway_configs/intersection_v1.yaml` to `configs/mappo/intersection_v1.yaml`
- Move: `examples/mappo/mappo_highway_configs/intersection_v1_smoke.yaml` to `configs/mappo/intersection_v1_smoke.yaml`

- [x] Create `ca_commappo/cli`, `ca_commappo/training`, `configs/sanity`, and `configs/mappo`.
- [x] Move the files with `Move-Item -LiteralPath ...`.
- [x] Add package marker files for the new Python packages.

### Task 2: Update Entrypoints

**Files:**
- Modify: `ca_commappo/cli/run_sanity_baseline.py`
- Modify: `ca_commappo/envs/debug_highway_wrapper.py`
- Modify: `ca_commappo/training/mappo_highway_intersection.py`
- Modify: `main.py`

- [x] Change sanity config default to `configs/sanity/highway_intersection.yaml`.
- [x] Change debug command examples to `python -m ca_commappo.envs.debug_highway_wrapper`.
- [x] Change MAPPO `CONFIG_DIR` to the repository `configs/mappo` directory.
- [x] Replace `main.py` with a thin dispatcher for `debug-wrapper`, `sanity`, and `mappo`.

### Task 3: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/sanity_baselines.md`
- Modify: `docs/source_code_analysis.md`

- [x] Document the new project layout.
- [x] Replace old `examples/...` project-owned commands with `python -m ca_commappo...`.
- [x] State that `examples/mappo` is reserved for vendored upstream reference code.
- [x] Note that deleted `evaluate_*` work remains out of scope for this phase.

### Task 4: Verify

**Files:**
- Run commands only.

- [x] Run `uv run pytest -q`.
- [x] Run `uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy all`.
- [x] Run `uv run python -m ca_commappo.envs.debug_highway_wrapper --target wrapper --seed 7 --max-steps 1`.
- [x] Run `uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection_v1_smoke.yaml --mode train --no-save`.

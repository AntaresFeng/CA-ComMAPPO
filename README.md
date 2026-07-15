# CA-ComMAPPO

用于把 `highway-env` 交叉路口环境接到 XuanCe 多智能体接口上的本地研究仓库。

## 本地运行

运行 sanity baseline：

```powershell
uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy all
```

调试单个 highway intersection episode：

```powershell
uv run python -m ca_commappo.envs.debug_highway_wrapper --target wrapper --seed 7 --max-steps 1
```

运行 highway intersection MAPPO 小参数烟测：

```powershell
uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode train --no-save
```

也可以通过顶层 dispatcher 调用：

```powershell
uv run python main.py sanity --config configs/sanity/highway_intersection.yaml --policy all
uv run python main.py debug-wrapper --target wrapper --seed 7 --max-steps 1
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode train --no-save
```

## 项目结构

- `ca_commappo/envs/`: highway-env 到 XuanCe 多智能体接口的环境适配器，以及 wrapper 端到端调试工具。
- `ca_commappo/evaluation/`: sanity baseline 执行和 summary 逻辑。
- `ca_commappo/cli/`: 本项目维护的普通命令行入口，例如 sanity baseline 运行入口。
- `ca_commappo/training/`: 本项目维护的训练入口。
- `configs/sanity/`: sanity baseline 配置。
- `configs/mappo/`: highway intersection MAPPO 配置。
- `examples/mappo/`: vendored XuanCe MAPPO 上游参考示例，仅作参考，不放本项目维护代码。
- `docs/`: 设计说明、排查记录和实验协议。
- `tests/`: pytest 测试。

## Highway `intersection-multi-agent-v1` 的 xuance 适配器

项目提供 `HighwayIntersectionMultiAgentEnv`，它默认基于 `gym.make("intersection-multi-agent-v1", ...)` 的 xuance `RawMultiAgentEnv` 适配器。所有 highway 环境参数都通过 `highway_config` 配置，包括 `controlled_vehicles`。

适配器会在调用 highway-env 前，把 `highway_config` 递归合并到一份完整的多智能体配置中。这是为了规避 `highway-env==1.11` 的浅合并问题：highway 内部使用顶层 `config.update(...)`，如果直接传入局部嵌套的 `observation` 或 `action` 配置，会覆盖掉必须保留的默认字段。

适配器的 `step()` 返回值采用 XuanCe 训练侧语义：返回的 `rewards` 字典和 `info["agents_rewards"]` 都是经过 active-agent mask 处理后的 adapter-facing reward。已经终止的 agent 在后续 transition 中 reward 记为 `0.0`，并且动作会被替换为 `IDLE_ACTION`。如果需要查看 highway-env 原始 per-agent reward，使用 `info["raw_agents_rewards"]`。环境级结束信号写在 `info["global_terminated"]`，碰撞状态写在 per-agent tuple 形式的 `info["crashed"]`。

MAPPO centralized critic 使用独立的全局场景观察，不再拼接各 agent 的局部 Kinematics observation。`global_observation()` 返回固定 agent 顺序的 `controlled: [N, 7]`、按路口中心距离确定性排序的 `npc: [K, 7]`，以及标记 NPC 有效行的 `npc_mask: [K]`。每车特征顺序为 `presence, x, y, vx, vy, cos_h, sin_h`，位置和速度使用全局绝对坐标，并复用 Kinematics 的归一化范围。终止 agent 的 controlled 行会清零，NPC 每步重新排序、截断并补零，不维护跨步车辆 ID。

XuanCe 的 `state()` 将上述三段按 `controlled.flatten() + npc.flatten() + npc_mask` 序列化为一维向量。可选的 wrapper 顶层配置 `global_npc_capacity` 可以显式设置 K；省略时根据 `max(1, initial_vehicle_count) + max(1, ceil(duration * policy_frequency))` 推导安全默认值。该字段与 `flatten_observations` 同级，不属于传给 highway-env 的 `highway_config`。正式配置默认 state shape 为 `(254,)`，smoke 配置默认为 `(205,)`。

highway-env 1.11 会在生成原始 step observation 后才删除和生成 NPC；适配器因此会在底层 step 完成后重新观察，使返回给 actor 的局部 observation 与 centralized state 基于同一 post-step 场景快照。这项 state 维度和 observation 时序变更不兼容旧 MAPPO checkpoint，旧模型应使用旧代码评估，新实验需要重新训练。

## MAPPO Video Evaluation

Each MAPPO command creates an isolated run directory under `runs/mappo_highway/`
by default. The resolved layout is:

```text
runs/mappo_highway/<timestamp>_<mode>_seed_<seed>/
|-- eval_metadata.json       # test/benchmark only
|-- eval_records.jsonl       # test/benchmark only
|-- wandb/                   # local W&B run files
|-- models/
|   |-- final_train_model.pth  # train
|   `-- best_model.pth         # benchmark
`-- videos/                  # test --record-video
```

Use `--output-dir` only when the run root itself needs to be redirected. The
per-run folder, log directory, model directory, and default video directory are
always derived automatically. The same run ID is also used as the W&B display
name and stored in the W&B config.

Record representative videos while testing a trained MAPPO checkpoint:

```powershell
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1.yaml --mode test --model-dir-load runs/mappo_highway/<benchmark_run>/models/best_model.pth --test-episode 32 --record-video
```

By default this records `min(test_episode, 6)` representative episodes under the run log directory's `videos/` folder and writes `video_eval_summary.json`, individual MP4s, a contact sheet, and a combined MP4.

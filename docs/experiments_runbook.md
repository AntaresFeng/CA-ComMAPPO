# 实验执行手册（Experiments Runbook）

> 覆盖执行提示词 `docs/ai-exec-thesis-prompt.md` 阶段 4/5 的实验矩阵与结果收集。
> 训练执行模式：AI 负责实现/smoke/归档；多 seed 正式训练交远端服务器（wandb）。

## 0. 通用约定（横向可比，不可违背）

- **始终显式传 `--config`**（`config_path_for` 大小写敏感，Linux 远端会报 `FileNotFoundError`，见 §陷阱）。
- 算法由 YAML 的 `agent` 字段选择（不是 `main.py` 子命令名；子命令 `mappo` 只是路由）。
- 横向对比固定：YAML（含 `env_seed`）、`test_episode`、`--seed`、`--video-seed`。
- **远端多 seed 必须用 `--mode benchmark`**（才会落盘 `eval_metadata.json` + `eval_records.jsonl` + `best_model.pth`；纯 `--mode train` 不产 eval 记录，阶段 5 无法聚合）。最终数字取自 `best_model.pth` + `--mode test`。
- 本地 smoke 用 `-smoke.yaml`（`logger: tensorboard`，无需 wandb）；远端正式配置 `logger: wandb`（远端需 `wandb login` 或 `WANDB_MODE=offline`）。
- 指标统一口径：`ca_commappo/evaluation/highway_metrics.py` 的 9-key summary + Phase 4 扩展的 `mean_travel_time`/`mean_throughput`/`mean_waiting_time`（条件出现）。best-model 选择键 `arrival↑→collision↓→reward↑→length↓`（**不改**）。

## 1. 算法清单与配置

| 算法 | `agent` | 主配置 | smoke 配置 | output_dir |
|---|---|---|---|---|
| IPPO | `IPPO` | `configs/ippo/intersection-multi-agent-v1.yaml` | `...-smoke.yaml` | `runs/ippo_highway/` |
| MAPPO-MLP | `MAPPO` | `configs/mappo/intersection-multi-agent-v1.yaml` | `...-smoke.yaml` | `runs/mappo_highway/` |
| Attention-MAPPO | `MAPPO` | `configs/attention_mappo/intersection-multi-agent-v1.yaml` | `...-smoke.yaml` | `runs/attention_mappo_highway/` |
| CommNet | `COMMNET` | `configs/commnet/intersection-multi-agent-v1.yaml` | `...-smoke.yaml` | `runs/commnet_highway/` |
| IC3Net | `IC3NET` | `configs/ic3net/intersection-multi-agent-v1.yaml` | `...-smoke.yaml` | `runs/ic3net_highway/` |
| CA-ComMAPPO | `CA_COMMAPPO` | `configs/ca_commappo/intersection-multi-agent-v1.yaml` | `...-smoke.yaml` | `runs/ca_commappo_highway/` |
| 通信消融 (c) 无冲突先验 | `CA_COMMAPPO` | `configs/ca_commappo/intersection-multi-agent-v1-no-prior.yaml` | `...-no-prior-smoke.yaml` | `runs/ca_commappo_no_prior_highway/` |

> **MAAC 与 MADDPG 按论文 `thesis-planning.md`「可选」标记有意暂缓**：MADDPG 不适合离散动作主线；MAAC attention critic 可后续补作对照。本 runbook 不含二者。

## 2. 本地 smoke 验证（每个算法、每次远端前）

```bash
uv run python main.py mappo --config <algo>/intersection-multi-agent-v1-smoke.yaml --mode train --no-save
```

对 CommNet/IC3Net/CA-ComMAPPO（任何 `action()` 依赖 `info` 或 RNN/通信的算法），**额外跑一次 test**（train smoke 不暴露 eval 路径问题）：

```bash
# 先用 benchmark 产 best_model，再 test
uv run python main.py mappo --config <algo>/<...>-smoke.yaml --mode benchmark
uv run python main.py mappo --config <algo>/<...>-smoke.yaml --mode test \
  --model-dir-load runs/<output_dir>/<benchmark_run>/models/best_model.pth --test-episode 1
```

门禁：smoke 不报维度/类型错、能完成构造与前向/反向。

## 3. 多 seed 正式训练（远端）

seed 列表 `[1, 2, 3, 4, 5]`。每个算法 × 5 seed：

```bash
# 远端（logger: wandb）。--mode benchmark 落盘 eval_records.jsonl + best_model.pth。
for SEED in 1 2 3 4 5; do
  uv run python main.py mappo \
    --config <algo>/intersection-multi-agent-v1.yaml \
    --mode benchmark --seed $SEED
done
```

每个 run 目录：`runs/<output_dir>/<timestamp>_benchmark_seed_<seed>/seed_<seed>_<ts>/`
含 `eval_metadata.json`、`eval_records.jsonl`、`models/best_model.pth`、`videos/`。

**最终验证数字**（取 best_model）：

```bash
uv run python main.py mappo \
  --config <algo>/intersection-multi-agent-v1.yaml \
  --mode test --seed $SEED \
  --model-dir-load runs/<output_dir>/<benchmark_run>/models/best_model.pth \
  --test-episode 100
```

收尾确认：每个远端 run 目录下 `eval_records.jsonl` 存在且非空。

## 4. 泛化实验（车数变化）

训练 `controlled_vehicles=4`，测试 `controlled_vehicles ∈ {6, 8, 10}`。

- 训练配置：复制各算法主配置 → `configs/<algo>/intersection-multi-agent-v1-gen4.yaml`，`highway_config.controlled_vehicles: 4`，`output_dir: runs/<algo>_gen4_highway/`。
- 测试配置：复制主配置，仅改 `highway_config.controlled_vehicles`（6/8/10），`--mode test` 用 best_model。
- **要点**：PI 编码器（Attention-MAPPO / CA-ComMAPPO）天然支持变车数（`n_views = input_dim // 105` 自适应）；MLP 基线（MAPPO/IPPO）输入维度固定 → 6/8/10 车时维度不匹配（论文要论证的不稳定性）。泛化测试仅对 PI 类算法可行；MLP 基线记录其维度不匹配失败作为对照。

```bash
# 训练（远端）
uv run python main.py mappo --config configs/ca_commappo/intersection-multi-agent-v1-gen4.yaml --mode benchmark --seed 1
# 测试变车数（用 gen4 的 best_model）
for N in 6 8 10; do
  uv run python main.py mappo --config configs/ca_commappo/intersection-multi-agent-v1-eval${N}.yaml \
    --mode test --model-dir-load runs/ca_commappo_gen4_highway/<run>/models/best_model.pth --test-episode 50
done
```

> gen4/evalN 配置在需要时按本节规则生成（复制主配置改 `controlled_vehicles` 与 `output_dir`）。

## 5. 顺序扰动实验

`ca_commappo/evaluation/order_perturbation.py`（见阶段 4 step 4）：在 eval rollout 时随机打乱观测中非 ego 车辆顺序（`obs[1:]`），统计指标波动。证明 MLP 基线波动大、PI 方法稳定。

```bash
# 用 best_model + 顺序扰动 eval（脚本封装 evaluate_highway_policy）
uv run python -m ca_commappo.evaluation.order_perturbation \
  --config <algo>/intersection-multi-agent-v1.yaml \
  --model-dir-load runs/<output_dir>/<run>/models/best_model.pth \
  --episodes 50 --seeds 1 2 3
```

## 6. 通信消融（论文 §4.5）

| 变体 | 配置 | 说明 |
|---|---|---|
| (a) 无通信 | `configs/mappo/` | MAPPO-MLP，无显式通信 |
| (b) 全连接平均通信 | `configs/commnet/` | CommNet mean 聚合 |
| (c) attention 通信，无冲突先验 | `configs/ca_commappo/*-no-prior.yaml` | `use_conflict_prior: false`（b_ij=0） |
| (d) 完整 CA-ComMAPPO | `configs/ca_commappo/` | 冲突感知 attention |
| (e) 去 PI 编码器 | `configs/mappo/` | 即 MAPPO-MLP，复用 (a) 结果 |

若评审要求区分「去 attention（求和/恒等）」与「CommNet mean」，在 (b) 之外再加 `(b')` 去 attention 变体；否则按 (b) 合并并在论文中说明等价。

## 7. 注意力可视化（可解释性图）

`ca_commappo/evaluation/attention_export.py`（阶段 4 step 6）：eval 时导出每个 agent 的 `α_ij` 矩阵（通信注意力）与编码器 attention 权重，存 JSON + 热力图（matplotlib），供论文可解释性图。

```bash
uv run python -m ca_commappo.evaluation.attention_export \
  --config configs/ca_commappo/intersection-multi-agent-v1.yaml \
  --model-dir-load runs/ca_commappo_highway/<run>/models/best_model.pth \
  --episodes 10 --output runs/ca_commappo_highway/attention/
```

## 8. 结果聚合与绘图（阶段 5）

```bash
# 跨 seed 聚合各算法的 eval_records.jsonl -> mean±std 表
uv run python -m ca_commappo.evaluation.aggregate_results \
  --runs-glob "runs/*_highway/*benchmark_seed_*/seed_*/eval_records.jsonl" \
  --output results/aggregated.csv

# 绘图（柱状图/泛化曲线/顺序扰动/消融/注意力热力图）
uv run python -m ca_commappo.evaluation.plot_results \
  --aggregated results/aggregated.csv --output-dir results/figures/
```

聚合/绘图脚本在至少一个真实 run 上跑通即满足阶段 5 门禁（可用本地 smoke run 验证）。

## 9. 已知陷阱

1. `config_path_for` 大小写敏感：Linux 远端 `configs/MAPPO/` 不存在（实际小写 `configs/mappo/`）→ 始终传 `--config`。
2. CommNet/IC3Net 必须 `use_rnn: true` + `Basic_RNN` + `comm_passes`；IC3Net `gate_hidden_size` 必须是 **list**（`[16]`，非标量）。
3. CommNet/IC3Net `action()` 需 `info`——本项目已运行时打补丁（`comm_runtime_patch.py`）；仍需额外跑 `--mode test` 验证。
4. CommNet `run_episodes()` 不调 `callback.on_train_episode_info`——训练期 highway 指标以 eval 期为准。
5. `eval_metadata.json` 只写一次——重跑同目录要先换 `output_dir` 或清空。
6. 远端必须 `--mode benchmark` 才产 eval 记录（`--mode train` 不产）。
7. 泛化实验中 MLP 基线输入维度固定——`controlled_vehicles` 变化时维度不匹配（预期，列为对照）。
8. CA-ComMAPPO 的 `b_ij` 用归一化空间 TTC 代理（kinematic proxy，局限见 `docs/ca_commappo_design.md` §9）。

## 10. 复现记录要求

每个实验记录：YAML 路径、命令、`git rev-parse HEAD`、seed 列表、产物路径、远端启动方式、结果表与图、结论。归档到 `docs/experiments_results.md`（阶段 5.3）。

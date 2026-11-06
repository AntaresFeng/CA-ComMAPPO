# CA-ComMAPPO

基于 highway-env 与 XuanCe 的无信号交叉路口多智能体强化学习研究仓库。当前实现包含
统一环境适配器、sanity baseline、IPPO/MAPPO、permutation-invariant
Attention-MAPPO、CommNet、IC3Net、CA-ComMAPPO，以及严格的评估、模型 provenance、
视频和结果聚合管线。

## 快速开始

```bash
# 环境与静态策略检查
uv run python main.py sanity \
  --config configs/sanity/highway_intersection.yaml --policy all

# MAPPO smoke
WANDB_MODE=offline uv run python main.py mappo \
  --config configs/mappo/ca-intersection-v0.yaml \
  --smoke --mode train --no-save

# 测试
uv run pytest -q
```

正式训练必须使用 `WANDB_MODE=offline`、`--mode benchmark`，先固化启动脚本，再通过
独立具名 tmux session 运行；CUDA 可用时显式传 `--device cuda:0`。
不要把 smoke 或 benchmark checkpoint-selection summary 当作论文最终结果。

## 配置结构

运行配置按以下顺序合并：

```text
configs/common.yaml
  + configs/highway/ca-intersection-v0.yaml
  + configs/<algorithm>/ca-intersection-v0.yaml
  + smoke / CLI overrides
```

所有 highway-env 参数通过 `highway_config` 传入。旧正式 MAPPO 基线来自已验证的
Iteration 5；其结果属于升级前任务。新环境使用独立 ID `ca-intersection-v0`（highway-env 1.12.1），
默认 4 个受控车辆、5 Hz、1M steps、`gamma=0.99`、
`gae_lambda=0.95`。

## 文档

从 [文档导航](docs/README.md) 开始：

- [当前实验全景](docs/current_experiment_status.md)
- [环境合同](docs/environment.md)
- [Canonical MAPPO baseline](docs/mappo_baseline.md)
- [CA-ComMAPPO 设计](docs/ca_commappo_design.md)
- [实验执行手册](docs/experiments_runbook.md)
- [论文实验主线](docs/thesis-planning.md)

原始 checkpoint、日志和视频位于被 Git 忽略的 `runs/`；经过审定的 manifest、聚合表
和小型 provenance 摘要位于受版本控制的 `results/`。

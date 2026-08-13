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
  --config configs/mappo/intersection-multi-agent-v1.yaml \
  --smoke --mode train --no-save

# 测试
uv run pytest -q
```

正式训练必须使用 `WANDB_MODE=offline`、`--mode benchmark` 和独立 tmux session。
不要把 smoke 或 benchmark checkpoint-selection summary 当作论文最终结果。

## 配置结构

运行配置按以下顺序合并：

```text
configs/common.yaml
  + configs/highway/intersection-multi-agent-v1.yaml
  + configs/<algorithm>/intersection-multi-agent-v1.yaml
  + smoke / CLI overrides
```

所有 highway-env 参数通过 `highway_config` 传入。正式 MAPPO 基线来自已验证的
Iteration 5，当前固定为 4 个受控车辆、5 Hz、1M steps、`gamma=0.99`、
`gae_lambda=0.95`。

## 文档

从 [文档导航](docs/README.md) 开始：

- [环境合同](docs/environment.md)
- [Canonical MAPPO baseline](docs/mappo_baseline.md)
- [CA-ComMAPPO 设计](docs/ca_commappo_design.md)
- [实验执行手册](docs/experiments_runbook.md)
- [论文实验主线](docs/thesis-planning.md)

原始 checkpoint、日志和视频位于被 Git 忽略的 `runs/`；经过审定的 manifest、聚合表
和小型 provenance 摘要位于受版本控制的 `results/`。

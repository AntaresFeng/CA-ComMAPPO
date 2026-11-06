# CA-ComMAPPO

基于 highway-env 与 XuanCe 的无信号路口多智能体强化学习研究仓库。当前使用项目环境
`ca-intersection-v0`（highway-env 1.12.1），支持逐车配置出生位置与路线、到达退出、
多智能体 mask，以及统一训练和评估。

旧论文计划及 intent/TTC 算法设计已作废。现有 CA-ComMAPPO 代码保留为遗留实验实现，
新算法和论文方向尚未确定；当前研究只考虑理想通信。

## 快速开始

```bash
uv run python main.py sanity --config configs/sanity/highway_intersection.yaml --policy all
uv run python main.py debug-wrapper --target wrapper --seed 7 --controlled-vehicles 8 --max-steps 5
WANDB_MODE=offline uv run python main.py mappo \
  --config configs/mappo/ca-intersection-v0.yaml --smoke --mode train --no-save
uv run pytest -q
```

配置按 `configs/common.yaml → configs/highway/ca-intersection-v0.yaml → 算法 YAML →
smoke/CLI` 合并；环境参数统一放在 `highway_config` 中。

## 文档与结果

- [文档导航](docs/README.md)：实现现状、环境说明和实验手册。
- [开发约定](AGENTS.md)：代码、测试和文档维护规则。
- [历史结果](results/README.md)：旧环境实验，不作为新环境的基线结果。

`runs/` 保存被 Git 忽略的模型、日志和视频；`results/` 保存审定的小型结果与证据。

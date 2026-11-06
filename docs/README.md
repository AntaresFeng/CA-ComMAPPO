# 文档导航

当前环境是 `ca-intersection-v0`（highway-env 1.12.1）。旧论文计划及基于 intent/TTC
代理量的算法设计已作废；遗留实现、历史实验不构成新的研究约束。

| 文档 | 内容 |
| --- | --- |
| [实现与研究现状](current_experiment_status.md) | 已有能力、遗留算法与尚未实现的问题 |
| [环境说明](environment.md) | 道路、出生、观测、奖励、终止、mask 和指标 |
| [实验手册](experiments_runbook.md) | 配置、sanity、训练、验证、视频与聚合 |
| [历史结果索引](../results/README.md) | 旧环境实验的证据入口与适用范围 |

开发约定见 [AGENTS.md](../AGENTS.md)。参数以 `configs/` 为准，运行行为以代码和测试
为准；文档出现冲突时核对并同步修正。失效计划和重复说明由 Git 历史保留，不再维护
第二套归档文档。

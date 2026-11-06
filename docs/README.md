# 文档导航

当前环境已迁移至 `ca-intersection-v0` / highway-env 1.12.1；
`environment.md` 定义新环境，既有实验矩阵和基线文档保留旧任务的历史证据。

本目录维护设计、合同和操作说明。每份文件负责一个主题；若文档、代码与
配置冲突，实验参数以 canonical YAML 为准，运行行为以代码和测试为准。

## 当前文档

| 文档 | 唯一职责 | 何时阅读 |
| --- | --- | --- |
| [`current_experiment_status.md`](current_experiment_status.md) | 当前实验全景、结果、证据链与 artifact 地图 | 接手项目或判断下一步前 |
| [`environment.md`](environment.md) | highway intersection 的环境、动作、终止和 reward 接口合同 | 修改适配器、环境配置或指标前 |
| [`mappo_baseline.md`](mappo_baseline.md) | 冻结 MAPPO 基线配置及 seed 1 参考结果 | 设计任何对比实验前 |
| [`ca_commappo_design.md`](ca_commappo_design.md) | CA-ComMAPPO 数学定义、实现映射和已知边界 | 修改创新算法前 |
| [`sanity_baselines.md`](sanity_baselines.md) | random / idle-only 的正式评估协议 | 验证环境或生成静态基线时 |
| [`experiments_runbook.md`](experiments_runbook.md) | smoke、正式训练、final test、视频和聚合的可执行流程 | 启动或归档实验时 |
| [`four_vehicle_experiment_plan.md`](four_vehicle_experiment_plan.md) | 四车论文实验的分阶段执行、冻结顺序与完成记录 | 追溯实验执行决策时 |
| [`thesis-planning.md`](thesis-planning.md) | 研究问题、算法矩阵、假设结论和论文边界 | 解释结果或安排论文工作时 |

## 权威顺序

1. `configs/common.yaml`、`configs/highway/*.yaml` 与算法 YAML 定义当前实验合同。
2. `current_experiment_status.md` 汇总已完成矩阵、审定结论和 artifact 位置。
3. `mappo_baseline.md` 解释冻结基线及已验证结果。
4. `experiments_runbook.md` 定义怎样产生可用于论文的 artifact。
5. 其他专题文档解释环境和算法设计，不覆盖上述合同。

正式训练必须使用 benchmark 模式，并设置 `WANDB_MODE=offline`。原始运行产物存放在
被 Git 忽略的 `runs/`；人工审定的 manifest、聚合表和小型 provenance 摘要存放在
受版本控制的 `results/`。

## 历史材料

[`archived/`](archived/) 中的文件仅用于追溯，不能复制其配置、命令或实验数字。
过期但无长期审计价值的教程和重复说明直接删除，由 Git 历史保留。

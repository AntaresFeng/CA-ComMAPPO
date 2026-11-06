# 历史结果索引

本目录现有实验均来自升级前的 `intersection-multi-agent-v1` / highway-env 1.11。
它们保留为历史证据，不代表 `ca-intersection-v0` 的性能；旧论文计划与 intent/TTC
设计已作废，旧假设编号、算法标签和图表不构成当前研究要求。

| 内容 | 入口 |
| --- | --- |
| 核心矩阵，冻结提交 `199ead5` | [逐 seed 结果](core_matrix_199ead5_final_test_results.md)、[manifest](core_matrix_199ead5_final_runs_manifest.json)、[聚合表](core_matrix_199ead5_aggregated.csv) |
| 扩展矩阵，冻结提交 `d50ed20` | [逐 seed 结果](extension_matrix_d50ed20_final_test_results.md)、[manifest](extension_matrix_d50ed20_final_runs_manifest.json)、[聚合表](extension_matrix_d50ed20_aggregated.csv) |
| 核心历史分析 | [观察与边界](core_matrix_199ead5_analysis_report.md)、[统计与图](core_matrix_199ead5_paper_analysis/)、[旧 CA trace](core_matrix_199ead5_ca_trace/) |
| 更早的 MAPPO seed 1 预实验 | [provenance](mappo_baseline_seed1_provenance.json)、[manifest](mappo_baseline_seed1_final_manifest.json)、[对比表](mappo_baseline_seed1_comparison.csv) |

核心矩阵含 MAPPO、Attention-MAPPO、CA full/no-prior；扩展矩阵含 IPPO、CommNet、IC3Net。
两者均采用训练 seeds 1/2/3 和 final cases 20000..20099，但冻结提交不同，应分别读取与
聚合。旧表的 `travel` / `mean_travel_time` 表示回合持续时间，不是逐车成功通行时间。

原始 checkpoint、JSONL、日志和视频路径由 manifest 记录，位于被 Git 忽略的 `runs/`。
历史复现需要匹配的代码、依赖与模型 provenance；不能把旧 YAML 路径换成当前路径后
直接复用结果。机器可读数据与图保持原样，新实验使用独立文件。

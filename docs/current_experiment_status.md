# 当前实验全景

本文是截至 2026-10-10 的实验状态入口，回答“现在完成了什么、各结果怎样关联、哪些
结论成立、继续工作时从哪里接手”。任务参数仍以 canonical YAML 为准；算法语义见专题
设计文档；最终数字以 `results/` 中的严格 manifest 和聚合产物为准。

## 一句话状态

四车主线的实现、质量门禁、7 个学习算法 × 3 个训练 seed、固定 100-case final test、
视频、核心配对统计、顺序扰动和 CA attention/TTC trace 均已完成。当前没有运行中的
正式实验；CUDA 已恢复并通过 CA-ComMAPPO action/update smoke，但既有正式矩阵保持其
原始 CPU provenance，不做设备变化导致的无理由重跑。

## 任务合同

```text
highway-env intersection
├── controlled agents: 4
├── horizon: 20 s × 5 Hz = 100 decisions
├── action: Discrete(3) = SLOWER / IDLE / FASTER
├── observation: 15 vehicle rows × 7 kinematic features per agent
├── reward: collision -40, arrival +10, speed shaping 0.01
└── outcome priority: collision > arrival > truncation
```

训练侧 reward 会按 agent 存活状态屏蔽；原始 per-agent reward 只保存在
`info["raw_agents_rewards"]`。所有 evaluator 统一使用 adapter 暴露的 crashed/arrived、
waiting steps 和 episode 完成语义，不能绕过 wrapper 重建指标。

## 算法关系

```text
static lower bounds: random, idle-only

IPPO ─────────────── decentralized actor + decentralized critic
MAPPO ────────────── decentralized actor + centralized critic
  └─ Attention-MAPPO adds permutation-invariant social encoder
       └─ CA-ComMAPPO adds conflict-aware inter-agent attention communication
            └─ no-prior removes only the kinematic bias b_ij

system communication baselines: CommNet, IC3Net (RNN backbone; not single-variable ablations)
```

主因果链是 `MAPPO → Attention-MAPPO → CA-ComMAPPO → no-prior ablation`。CommNet 和
IC3Net 与 CA-ComMAPPO 的 backbone 不同，只能作为系统级补充比较，不能把全部差异归因
于通信聚合方式。

## 证据生成链

```text
canonical YAML + frozen source
        │
        ├─ smoke: construction / action / update only, never a paper result
        │
        └─ benchmark: 1M steps, evaluate every 10k on cases 10000..10049
                │
                └─ best_model.pth
                   sort = arrival↑ → collision↓ → reward↑ → length↓
                        │
                        ├─ final test: cases 20000..20099, deterministic argmax
                        ├─ videos: cases 20000..20005
                        ├─ order-eval: same 100 cases × permutations 30000..30002
                        └─ CA trace: cases 20000..20009
                                │
                                └─ explicit manifest → strict aggregate → paper-analysis
```

训练 seed `N` 的训练环境 block 从 `N × 1,000,000` 开始，与 checkpoint selection 和
final-test cases 隔离。provenance v2 绑定 checkpoint SHA、训练配置、source commit、
runtime source manifest 和训练设备；正式 test 只接受严格匹配的 `best_model.pth`。

## 为什么有两套正式矩阵

| 矩阵 | 冻结提交 | 算法 | 用途 |
| --- | --- | --- | --- |
| 核心 | `199ead5` | MAPPO、Attention-MAPPO、CA full、CA no-prior | 主结论、配对统计、顺序扰动、trace |
| 扩展 | `d50ed20` | IPPO、CommNet、IC3Net | 修复 IPPO env-seed 污染后的补充比较 |

两套矩阵的任务、预算和 final cases 相同，但 source commit 不同。严格聚合器要求同一
manifest 中所有学习策略共享 source commit/runtime manifest，因此它们必须分别聚合；
扩展结果可以并列描述，不能混入核心配对检验。

## 已审定结果

数值为 3 个训练 seed 的 `mean ± sample std`；静态策略是固定策略点估计。

| 算法 | arrival | collision | reward | 解释 |
| --- | ---: | ---: | ---: | --- |
| random | 10.0% | 75.0% | -12.457 | 环境下界 |
| idle-only | 17.0% | 83.0% | -16.129 | 环境下界 |
| IPPO | 50.3% ± 12.3% | 20.3% ± 11.2% | 3.664 ± 3.504 | 扩展矩阵 |
| MAPPO | 36.0% ± 12.1% | 12.3% ± 17.9% | 5.197 ± 5.174 | 核心基线，seed 方差较大 |
| Attention-MAPPO | 75.3% ± 12.7% | 1.3% ± 1.5% | 9.582 ± 0.289 | 最稳定的核心强基线 |
| CommNet | 22.0% ± 15.5% | 51.7% ± 13.7% | -5.869 ± 4.128 | 扩展系统基线 |
| IC3Net | 9.0% ± 13.9% | 44.3% ± 45.1% | -6.293 ± 10.987 | 扩展系统基线，方差很大 |
| CA-ComMAPPO | 82.7% ± 20.0% | 3.3% ± 2.3% | 9.322 ± 1.184 | 最高平均 arrival，但非全面优于 Attention |
| CA no-prior | 39.3% ± 9.0% | 12.0% ± 7.8% | 5.852 ± 1.892 | 关键消融 |

## 假设结论

- **H1 支持：** MAPPO 在相同 100 cases 上明显优于 random/idle-only。
- **H2 支持：** MAPPO 在三组邻车行置换下 arrival 下降 `8–11` 个百分点、collision
  增加 `69–72` 个百分点；Attention-MAPPO 和 CA-ComMAPPO 的逐 case 指标变化全为零。
- **H3 不明确：** CA 相对 Attention 的 arrival 为 `+0.073 ± 0.140`，但 collision 为
  `+0.020 ± 0.030`、reward 为 `-0.261 ± 0.949`，不能声称显式通信全面更优。
- **H4 方向一致：** full 相对 no-prior 的 arrival 为 `+0.433 ± 0.240`、collision 为
  `-0.087 ± 0.091`、reward 为 `+3.470 ± 2.620`。三 seed 精确 sign-flip `p=0.25` 是
  当前样本量下同向差值的最小值，只能作探索性辅助。
- **H5 已完成：** full/no-prior 均有严格 provenance 的 attention/TTC trace、总览
  heatmap 和原始记录 SHA；10-case outcome 只用于解释，不替代多 seed 性能。
- **H6 不在范围内：** 本论文固定四车，不声明 4→6/8/10 checkpoint 泛化。

## Artifact 地图

| 需要查看的内容 | 位置 |
| --- | --- |
| 核心严格输入与聚合 | `results/core_matrix_199ead5_final_runs_manifest.json`、`core_matrix_199ead5_aggregated.csv` |
| 扩展严格输入与聚合 | `results/extension_matrix_d50ed20_final_runs_manifest.json`、`extension_matrix_d50ed20_aggregated.csv` |
| 核心逐 seed 人工报告 | `results/core_matrix_199ead5_final_test_results.md` |
| 扩展逐 seed 人工报告 | `results/extension_matrix_d50ed20_final_test_results.md` |
| 核心结论与边界 | `results/core_matrix_199ead5_analysis_report.md` |
| 主表、配对统计和论文图 | `results/core_matrix_199ead5_paper_analysis/` |
| CA trace 摘要与总览图 | `results/core_matrix_199ead5_ca_trace/` |
| checkpoint、原始 JSONL、日志和视频 | `runs/`（Git 忽略） |

## 当前操作状态

- 工作计划内的实验和归档已经结束；没有需要续跑的 tmux session。
- 当前主机 CUDA 可用。后续新长任务适用时显式使用 `--device cuda:0`，先写入
  `runs/launch_scripts/`，通过 `bash -n` 后再用具名 tmux session 后台启动。
- 不覆盖现有 eval metadata，不把 smoke 结果加入 manifest，不把两个冻结提交的学习
  策略混在同一严格聚合中。
- 论文正文暂不编写；若增加新实验，必须先明确研究问题、对照角色、seed 集和是否需要
  新冻结提交，不能把探索性 run 事后包装成正式证据。

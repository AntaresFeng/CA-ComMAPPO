# 扩展矩阵 d50ed20 正式 final test 结果

> 历史记录：以下内容来自 highway-env 1.11 旧任务，不代表当前环境性能。
> 旧论文计划及 intent/TTC 设计已作废；历史范围与指标解释见 [结果索引](README.md)。

> 本报告只统计**冻结提交 `d50ed20`** 上已完成并通过 artifact 审计的扩展算法：IPPO、CommNet、IC3Net，各训练 seeds 1/2/3。
>
> - 冻结 commit：`d50ed20a9b5bcef3a34f26533e77bc7d2d8439c3`
> - runtime source SHA：`191e071d2966410511a97794eacd33025db3582b164c4235734d6c98365ff52a`
> - 数据来源：每个算法的 100-case 正式 final test（`eval_records.jsonl`，mode=test/phase=test，episode seeds `20000..20099`）
> - 审计状态：全部 9 个 checkpoint 均为 provenance v2，`sidecar_sha256_verified`；每个 final test 都包含 6 个对应 cases 的视频、合并视频和 contact sheet。

## 1. 适用范围与可比性

本矩阵使用与核心矩阵相同的四车任务、训练预算、checkpoint selection 和 final-test cases，但它是在后续提交 `d50ed20` 上完成的。该提交将 IPPO 的训练向量环境由 `DummyVecMultiAgentEnv` 更正为 `SubprocVecMultiAgentEnv`，以保证训练环境 seed 不会被原地改写；CommNet 和 IC3Net 也在该提交上运行。

因此，本文件是扩展算法的独立正式结果集：可与 `results/core_matrix_199ead5_final_test_results.md` 并列作描述性参考，但不得将两个提交的学习算法放入同一严格 manifest、配对检验或核心结论统计。CommNet 和 IC3Net 还使用 RNN backbone/通信特有配置，不能把与 MAPPO 的全部差异归因于通信机制。

## 2. 逐 seed 结果（含 run 目录）

### 2.1 IPPO

| seed | mean reward | arrival | collision | trunc | agent arr frac | agent coll frac | length | throughput | wait | travel | run 目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.373 | 47.0% | 23.0% | 30.0% | 0.668 | 0.115 | 77.260 | 0.147 | 2.721 | 15.452 | `runs/ippo_highway/20260817_233925_097773_test_seed_1` |
| 2 | 7.630 | 64.0% | 8.0% | 28.0% | 0.868 | 0.040 | 86.730 | 0.198 | 2.991 | 17.346 | `runs/ippo_highway/20260818_010657_468452_test_seed_2` |
| 3 | 0.988 | 40.0% | 30.0% | 30.0% | 0.648 | 0.150 | 73.130 | 0.153 | 2.844 | 14.626 | `runs/ippo_highway/20260818_023459_863420_test_seed_3` |

benchmark run（best checkpoint 来源）：

- seed 1: `runs/ippo_highway/20260817_221322_866719_benchmark_seed_1`
- seed 2: `runs/ippo_highway/20260817_234055_336066_benchmark_seed_2`
- seed 3: `runs/ippo_highway/20260818_010827_724632_benchmark_seed_3`

### 2.2 CommNet

| seed | mean reward | arrival | collision | trunc | agent arr frac | agent coll frac | length | throughput | wait | travel | run 目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -9.946 | 6.0% | 64.0% | 30.0% | 0.243 | 0.323 | 57.310 | 0.054 | 2.915 | 11.462 | `runs/commnet_highway/20260819_222225_532667_test_seed_1` |
| 2 | -1.692 | 23.0% | 37.0% | 40.0% | 0.543 | 0.185 | 72.050 | 0.117 | 2.658 | 14.410 | `runs/commnet_highway/20260820_001757_878848_test_seed_2` |
| 3 | -5.969 | 37.0% | 54.0% | 9.0% | 0.448 | 0.273 | 48.480 | 0.130 | 0.510 | 9.696 | `runs/commnet_highway/20260820_021330_332839_test_seed_3` |

benchmark run（best checkpoint 来源）：

- seed 1: `runs/commnet_highway/20260819_204323_215093_benchmark_seed_1`
- seed 2: `runs/commnet_highway/20260819_222355_781279_benchmark_seed_2`
- seed 3: `runs/commnet_highway/20260820_001928_117480_benchmark_seed_3`

### 2.3 IC3Net

| seed | mean reward | arrival | collision | trunc | agent arr frac | agent coll frac | length | throughput | wait | travel | run 目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -2.180 | 25.0% | 41.0% | 34.0% | 0.525 | 0.205 | 67.890 | 0.122 | 2.071 | 13.578 | `runs/ic3net_highway/20260820_040102_779672_test_seed_1` |
| 2 | -18.743 | 2.0% | 91.0% | 7.0% | 0.070 | 0.493 | 31.050 | 0.016 | 0.274 | 6.210 | `runs/ic3net_highway/20260820_055135_409770_test_seed_2` |
| 3 | 2.044 | 0.0% | 1.0% | 99.0% | 0.210 | 0.005 | 99.980 | 0.042 | 12.970 | 19.996 | `runs/ic3net_highway/20260820_072937_962292_test_seed_3` |

benchmark run（best checkpoint 来源）：

- seed 1: `runs/ic3net_highway/20260820_021500_585329_benchmark_seed_1`
- seed 2: `runs/ic3net_highway/20260820_040233_025472_benchmark_seed_2`
- seed 3: `runs/ic3net_highway/20260820_055235_728611_benchmark_seed_3`

## 3. 三 seed 聚合（mean ± sample std）

| 算法 | mean reward | arrival | collision | agent arr frac | length | throughput | wait | travel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IPPO | 3.664 ± 3.504 | 50.3% ± 12.3% | 20.3% ± 11.2% | 0.728 ± 0.122 | 79.040 ± 6.973 | 0.166 ± 0.028 | 2.852 ± 0.135 | 15.808 ± 1.395 |
| CommNet | -5.869 ± 4.128 | 22.0% ± 15.5% | 51.7% ± 13.7% | 0.411 ± 0.153 | 59.280 ± 11.908 | 0.101 ± 0.040 | 2.028 ± 1.321 | 11.856 ± 2.382 |
| IC3Net | -6.293 ± 10.987 | 9.0% ± 13.9% | 44.3% ± 45.1% | 0.268 ± 0.233 | 66.307 ± 34.492 | 0.060 ± 0.055 | 5.105 ± 6.870 | 13.261 ± 6.898 |

## 4. 说明

- 以上均为正式 final test：100 episodes、seeds `20000..20099`、deterministic argmax；checkpoint selection 使用独立的 50 cases（seeds `10000..10049`）和固定排序 `arrival↑ → collision↓ → reward↑ → length↓`。
- “agent arr frac / agent coll frac” 分别为 `mean_agent_arrival_fraction` / `mean_agent_collision_fraction`；`trunc` 是 episode-level truncation rate。
- 聚合标准差为 3 个训练 seed 的 sample std。样本量为 3，数值仅作描述性参考；尚未为这一独立扩展矩阵生成跨算法正式配对统计。
- 早于 `d50ed20` 的 IPPO 失败 benchmark 使用了会污染 `env_seed` 的 `DummyVecMultiAgentEnv`，不属于本报告，也不得用于正式结果。

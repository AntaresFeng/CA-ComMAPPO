# 核心矩阵 199ead5 正式 final test 结果

> 本报告只统计**冻结提交 `199ead5`** 上已完成并通过自动队列审计的核心矩阵：
> MAPPO、Attention-MAPPO、CA-ComMAPPO (full)、CA-ComMAPPO no-prior，各训练 seeds 1/2/3。
>
> - 冻结 commit：`199ead5d8b2fababa649d3aa94c1c025eb08d682`
> - runtime source SHA：`191e071d2966410511a97794eacd33025db3582b164c4235734d6c98365ff52a`
> - 数据来源：每个算法的 100-case 正式 final test（`eval_records.jsonl`，mode=test/phase=test，episode seeds 20000..20099）
> - 对应审计日志：`runs/matrix_state/four_vehicle_matrix_events.log`（2026-08-16T14:30:50 启动）

## 1. 逐 seed 结果（含 run 目录）

### 1.1 MAPPO

| seed | mean reward | arrival | collision | trunc | agent arr frac | agent coll frac | length | throughput | wait | travel | run 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.176 | 38.0% | 1.0% | 61.0% | 0.802 | 0.005 | 97.140 | 0.165 | 4.245 | 19.428 | `runs/mappo_highway/20260816_011800_070557_test_seed_1` |
| 2 | 8.193 | 47.0% | 3.0% | 50.0% | 0.845 | 0.015 | 89.760 | 0.190 | 2.839 | 17.952 | `runs/mappo_highway/20260816_143054_207489_test_seed_2` |
| 3 | -0.777 | 23.0% | 33.0% | 44.0% | 0.555 | 0.165 | 72.170 | 0.137 | 4.178 | 14.434 | `runs/mappo_highway/20260816_155856_740519_test_seed_3` |

benchmark run（best checkpoint 来源）：

- seed 1: `runs/mappo_highway/20260815_220359_277463_benchmark_seed_1`
- seed 2: `runs/mappo_highway/20260816_011957_276420_benchmark_seed_2`
- seed 3: `runs/mappo_highway/20260816_143224_461321_benchmark_seed_3`

### 1.2 Attention-MAPPO

| seed | mean reward | arrival | collision | trunc | agent arr frac | agent coll frac | length | throughput | wait | travel | run 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9.612 | 69.0% | 0.0% | 31.0% | 0.922 | 0.000 | 93.420 | 0.200 | 3.194 | 18.684 | `runs/attention_mappo_highway/20260816_174629_193484_test_seed_1` |
| 2 | 9.856 | 90.0% | 3.0% | 7.0% | 0.968 | 0.015 | 89.770 | 0.215 | 2.752 | 17.954 | `runs/attention_mappo_highway/20260816_193401_647822_test_seed_2` |
| 3 | 9.280 | 67.0% | 1.0% | 32.0% | 0.910 | 0.005 | 92.820 | 0.197 | 3.340 | 18.564 | `runs/attention_mappo_highway/20260816_212134_159876_test_seed_3` |

benchmark run（best checkpoint 来源）：

- seed 1: `runs/attention_mappo_highway/20260816_160026_998303_benchmark_seed_1`
- seed 2: `runs/attention_mappo_highway/20260816_174759_412481_benchmark_seed_2`
- seed 3: `runs/attention_mappo_highway/20260816_193531_985592_benchmark_seed_3`

### 1.3 CA-ComMAPPO (full)

| seed | mean reward | arrival | collision | trunc | agent arr frac | agent coll frac | length | throughput | wait | travel | run 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10.129 | 90.0% | 2.0% | 8.0% | 0.975 | 0.010 | 92.990 | 0.210 | 2.842 | 18.598 | `runs/ca_commappo_highway/20260816_231036_701312_test_seed_1` |
| 2 | 9.875 | 98.0% | 2.0% | 0.0% | 0.990 | 0.010 | 85.170 | 0.233 | 2.068 | 17.034 | `runs/ca_commappo_highway/20260817_005939_237023_test_seed_2` |
| 3 | 7.962 | 60.0% | 6.0% | 34.0% | 0.880 | 0.030 | 93.250 | 0.189 | 3.514 | 18.650 | `runs/ca_commappo_highway/20260817_024841_685156_test_seed_3` |

benchmark run（best checkpoint 来源）：

- seed 1: `runs/ca_commappo_highway/20260816_212304_493522_benchmark_seed_1`
- seed 2: `runs/ca_commappo_highway/20260816_231206_972070_benchmark_seed_2`
- seed 3: `runs/ca_commappo_highway/20260817_010109_476036_benchmark_seed_3`

### 1.4 CA-ComMAPPO no-prior（消融）

| seed | mean reward | arrival | collision | trunc | agent arr frac | agent coll frac | length | throughput | wait | travel | run 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.913 | 29.0% | 7.0% | 64.0% | 0.797 | 0.035 | 94.390 | 0.171 | 4.482 | 18.878 | `runs/ca_commappo_no_prior_highway/20260817_043744_129516_test_seed_1` |
| 2 | 3.667 | 45.0% | 21.0% | 34.0% | 0.752 | 0.105 | 85.630 | 0.169 | 3.591 | 17.126 | `runs/ca_commappo_no_prior_highway/20260817_062646_653418_test_seed_2` |
| 3 | 6.976 | 44.0% | 8.0% | 48.0% | 0.823 | 0.040 | 93.880 | 0.174 | 3.891 | 18.776 | `runs/ca_commappo_no_prior_highway/20260817_081549_072925_test_seed_3` |

benchmark run（best checkpoint 来源）：

- seed 1: `runs/ca_commappo_no_prior_highway/20260817_025011_889545_benchmark_seed_1`
- seed 2: `runs/ca_commappo_no_prior_highway/20260817_043914_389233_benchmark_seed_2`
- seed 3: `runs/ca_commappo_no_prior_highway/20260817_062816_898151_benchmark_seed_3`

## 2. 三 seed 聚合（mean ± sample std）

| 算法 | mean reward | arrival | collision | agent arr frac | length | throughput | wait | travel |
|---|---|---|---|---|---|---|---|---|
| MAPPO | 5.197 ± 5.174 | 36.0% ± 12.1% | 12.3% ± 17.9% | 0.734 ± 0.157 | 86.357 ± 12.828 | 0.164 ± 0.027 | 3.754 ± 0.794 | 17.271 ± 2.566 |

| Attention-MAPPO | 9.582 ± 0.289 | 75.3% ± 12.7% | 1.3% ± 1.5% | 0.933 ± 0.030 | 92.003 ± 1.957 | 0.204 ± 0.010 | 3.095 ± 0.306 | 18.401 ± 0.391 |

| CA-ComMAPPO (full) | 9.322 ± 1.184 | 82.7% ± 20.0% | 3.3% ± 2.3% | 0.948 ± 0.060 | 90.470 ± 4.592 | 0.211 ± 0.022 | 2.808 ± 0.724 | 18.094 ± 0.918 |

| CA-ComMAPPO no-prior | 5.852 ± 1.892 | 39.3% ± 9.0% | 12.0% ± 7.8% | 0.791 ± 0.035 | 91.300 ± 4.917 | 0.171 ± 0.002 | 3.988 ± 0.453 | 18.260 ± 0.983 |

## 3. 说明

- 以上均为 **正式 final test**（100 episodes，seeds `20000..20099`，deterministic argmax）结果，落盘文件：`<test_run>/eval_records.jsonl`。
- 每个 test 已包含 6 个视频，路径见对应 `videos/` 目录。
- “agent arr frac / agent coll frac” 为 `mean_agent_arrival_fraction` / `mean_agent_collision_fraction`。
- 聚合标准差为三 seed 的 **sample std**；三 seed 情况下仅作描述性参考，正式统计请以阶段五的 `paper-analysis`（含配对差与 sign-flip 检验）为准。
- 核心矩阵只来自提交 `199ead5`；IPPO/CommNet/IC3Net 属于后续扩展（另一提交 `d50ed20`），不在本表。


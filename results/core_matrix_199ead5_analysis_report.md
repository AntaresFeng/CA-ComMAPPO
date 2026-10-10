# 核心矩阵 199ead5 论文分析结论

本报告汇总冻结提交 `199ead5d8b2fababa649d3aa94c1c025eb08d682` 的正式核心矩阵、
配对统计、观测顺序稳健性和 CA attention/TTC trace。主统计单位是训练 seed；所有
学习算法使用 seeds `[1, 2, 3]`，每个 seed 的最终指标来自固定 cases
`20000..20099` 的独立 100-episode test。

机器可读结果与图位于 [`core_matrix_199ead5_paper_analysis/`](core_matrix_199ead5_paper_analysis/)，
严格输入清单为 [`core_matrix_199ead5_final_runs_manifest.json`](core_matrix_199ead5_final_runs_manifest.json)。

## 结论

- **H1 得到支持。** MAPPO 的 arrival 为 `36.0% ± 12.1%`、collision 为
  `12.3% ± 17.9%`；random 分别为 `10.0%`、`75.0%`，idle-only 分别为
  `17.0%`、`83.0%`。静态策略是固定策略点估计，不把它们的 episode 波动解释为
  训练 seed 方差。
- **H2 得到直接支持。** 对 seed 1 checkpoint 的 100 个固定 cases，MAPPO 在三组
  邻车行置换下的 arrival 配对均值分别变化 `-0.11/-0.11/-0.08`，collision 分别变化
  `+0.72/+0.72/+0.69`。Attention-MAPPO 和辅助 CA-ComMAPPO 的所有配对指标在三组
  置换下均为 `0`，与其 permutation-invariant encoder 合同一致。
- **H3 未得到明确支持。** CA-ComMAPPO 相对 Attention-MAPPO 的三 seed arrival 差为
  `+0.073 ± 0.140`，但 collision 差为 `+0.020 ± 0.030`，reward 差为
  `-0.261 ± 0.949`。当前结果不能声称显式通信整体优于 Attention-MAPPO。
- **H4 得到方向一致但样本量有限的支持。** full 相对 no-prior 的 arrival 差为
  `+0.433 ± 0.240`、collision 差为 `-0.087 ± 0.091`、reward 差为
  `+3.470 ± 2.620`。三项精确双侧 sign-flip `p=0.25`；三个训练 seed 下这是同向差值
  能达到的最小 p 值，只能作为探索性辅助，结论应以效应量和逐 seed 原始值为主。
- **H5 已形成可审计证据。** full/no-prior 均完成 seed 1、cases `20000..20009` 的只读
  attention/TTC trace。总览热力图、摘要、metadata 和 raw-record SHA 归档在
  [`core_matrix_199ead5_ca_trace/`](core_matrix_199ead5_ca_trace/)；10-case outcome 只用于
  解释具体案例，不替代多 seed 性能比较。

## 顺序扰动 artifact

| 算法 | run 目录 | cases | permutation seeds |
| --- | --- | ---: | --- |
| MAPPO | `runs/mappo_highway/20261010_163415_404143_order-eval_seed_1` | 100 | 30000/30001/30002 |
| Attention-MAPPO | `runs/attention_mappo_highway/20261010_164825_144894_order-eval_seed_1` | 100 | 30000/30001/30002 |
| CA-ComMAPPO | `runs/ca_commappo_highway/20261010_172830_981347_order-eval_seed_1` | 100 | 30000/30001/30002 |

三项均使用 provenance v2 严格验证的 seed 1 `best_model.pth`，source commit 与 runtime
manifest 均和核心 final-test 矩阵一致。

## CA trace artifact

| 配置 | run 目录 | trace steps | effective lambda | 10-case outcome |
| --- | --- | ---: | --- | --- |
| full | `runs/ca_commappo_highway/20261010_173213_438857_ca-trace_seed_1` | 923 | 1.067926 | arrival 100%，collision 0% |
| no-prior | `runs/ca_commappo_no_prior_highway/20261010_173237_329004_ca-trace_seed_1` | 935 | 1.000000 | arrival 40%，collision 10% |

原始 trace 保留在被 Git 忽略的 `runs/`，不提交大体积 JSONL。归档 manifest 保存其
SHA-256，因此仍可检查原始记录是否发生变化。

## 执行与边界

- order-eval 与 ca-trace 均从冻结提交 `199ead5` 的干净隔离 checkout 执行，设备为 CPU。
  执行当时 PyTorch CUDA 初始化报 `CUDA unknown error`，因而未强制使用不可用的 CUDA
  上下文。主机重启后，RTX 3080 已于 2026-10-10 通过 PyTorch 张量探测和 CA-ComMAPPO
  真实 action/update smoke；这不改变既有 CPU artifact 的 provenance，也不触发重跑。
- 后台 CA 队列通过 `runs/launch_scripts/run_posthoc_analysis_199ead5.sh` 在 tmux session
  `paper-posthoc-199ead5` 中串行执行，event log 以
  `QUEUE COMPLETE all posthoc artifacts passed` 收尾。
- 扩展矩阵来自后续提交 `d50ed20`，只作补充描述性比较，不进入本报告的严格 manifest
  或配对统计。
- 三 seed sign-flip 检验分辨率很粗；报告不把 `p=0.25` 解释为传统显著性结论。

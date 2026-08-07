# MAPPO baseline 迭代日志

本文记录 MAPPO baseline 的审计、假设、单变量迭代、正式训练和独立验证。正式训练统一使用 `--mode benchmark`、`WANDB_MODE=offline`，长任务在具名 tmux session 中运行。最终结果只接受固定显式 seed、确定性动作、每个训练 seed 至少 100 episodes 的独立 `test`；benchmark 仅用于选择 checkpoint。

## 成功标准

- 同一配置和预算训练 3 个 seed；每个 seed 的 `best_model.pth` 独立验证 100 episodes。
- 3-seed 平均 `arrival_rate >= 0.60` 且 `collision_rate <= 0.30`。
- 至少 2/3 seed 分别满足 `arrival_rate >= 0.50` 且 `collision_rate <= 0.40`。
- 明显优于 random / idle-only sanity baseline。
- checkpoint 选择不能再出现 5 局 `1.00`、独立验证约 `0.15` 的极端落差。

## 保留的原始实验（只读基准）

以下目录不得覆盖或删除：

- `runs/mappo_highway/20260807_191658_348474_benchmark_seed_1/`
- `runs/mappo_highway/20260807_210338_460237_test_seed_1/`

关键 SHA-256：

- 原始 `best_model.pth`: `899e2ed8e4da120fcac4bcd93acccf869d2bec0379c6917eff6373eb847b48a7`
- 原始 benchmark `eval_records.jsonl`: `46a9c04ea9cdc0b9e6334f46eddeb3e712138469e61908c7d8652cde10bc0965`
- 原始 test `eval_records.jsonl`: `f78e09510753172ebe8c80d239e70b0606cc02300c998814660267ec5189aa19`
- 原始视频 summary: `82140f1bb5d510cebf80d364bc1fac2e8d64f35345cbb0dc68ce60c7c3562867`
- 严格确定性诊断 `eval_records.jsonl`: `64cbb6e41231a82bd70bf27c2502fc8490d867c0172961411d115f97338d930d`
- 严格确定性诊断视频 summary: `cbf023f2e54df1c30c23323db086de68f6136c8b9d84af63946026f1a477a3fb`
- 严格确定性合并视频: `9dcdfecd01b38aa9643f753c930b29dd3fe772597ed896932ed7ba51ff5614c5`

原始结果：

| 来源 | episodes | arrival | collision | mean reward | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| benchmark epoch 28 / 280k | 5 | 1.00 | 0.00 | 25.875 | 被选为 best |
| benchmark 全部 101 次评估合计 | 505 | 0.0871 | 0.9069 | — | 反映 5 局极值选择风险 |
| legacy test | 100 | 0.15 | 0.85 | 9.510 | stochastic、seed 与训练流重叠，不作为严格最终结果 |
| 原 best_model 严格复测 | 100 | 0.17 | 0.83 | 9.851 | fixed seed 20000..20099、deterministic；确认不达标；旧 checkpoint 无 provenance，故只作诊断 |
| legacy video | 6 | 0.333 | 0.667 | 13.009 | 已保存 6 个 episode 视频及合并视频 |

## Iteration 0：框架和数据流审计

日期：2026-08-07。

结论：训练预算和 checkpoint 保存/加载路径本身正确；环境适配器、离散动作空间、joint state、全局终止、masked reward、terminal/reset state 分离以及 GAE 总体数据流未发现会直接颠倒任务目标的问题。主要问题集中在评估协议和 inactive 样本统计：

1. benchmark 每 10k steps 用 5 个 episode 评估，共做 101 次极值搜索；只有 epoch 28 得到 5/5 到达。
2. 旧评估实际仍使用 stochastic sample，且消耗全局 Torch RNG，因而 benchmark 会影响后续训练随机流。
3. 旧并行评估只限制总完成 episode 数，快环境可重复贡献、慢环境可能一次也未计入。
4. benchmark 的评估环境持续复用，每个 checkpoint 看到不同 RNG 流，checkpoint 间并非同题比较。
5. legacy test 使用 `env_seed=1`，与训练 seed 流重叠，不是预先保留的独立测试集。
6. XuanCe 1.4.3 的 advantage normalization 和 ValueNorm update 在 loss mask 之前包含 inactive agent，可能污染 active 样本尺度。
7. XuanCe 只返回每个 rollout 最后一个 minibatch 的日志，缺少 KL、clip fraction、ratio 分位数等 PPO 诊断。
8. 旧聚合器从 benchmark 记录中选最好 summary，会把 5 局极值误当最终结果，且默认 glob 与真实 run layout 不符。
9. CLI 过去只覆盖 `seed` 而不覆盖 `env_seed`；若直接跑 seed 2/3，它们会共享
   seed 1 的环境随机流。简单令 `env_seed=seed` 仍会让 16 个并行 worker 的
   seed ranges `1..16 / 2..17 / 3..18` 大量重叠。正式运行前已冻结为
   `env_seed=training_seed*1,000,000`，并由 checkpoint provenance 强制校验。

原始 W&B learner 日志复核：无 NaN/Inf；gradient norm 中位数约 `0.753`、p95 约 `2.463`、最大约 `4.054`，低于 clip 10；entropy 从约 `0.708` 降至 262k 附近约 `0.323`，末段恢复至约 `0.646`，不是永久熵塌缩。训练 episode 的前 100k 到达率约 `0.169`、后 100k 约 `0.179`，没有可靠学习趋势。成功 episode 平均 reward 约 `25.71`，碰撞约 `6.64`，reward 排序没有反转。

审计前全量测试：`UV_CACHE_DIR=/tmp/ca-commappo-uv-cache uv run pytest -q`，结果 `128 passed, 2 warnings`。

## Iteration 1：framework-corrected control

假设：先保持原 MAPPO 网络、学习率、PPO epoch/minibatch 和 1M step 预算不变，只消除评估偏差与 inactive 统计污染。若表现仍差，再依据新增的 KL/clip/ratio 诊断做单变量超参实验，避免同时改动造成不可归因。

配置：`configs/mappo/intersection-multi-agent-v1-robust-eval.yaml`。

协议：

- checkpoint selection：固定 seed `10000..10049`，50 episodes，确定性动作，每个 checkpoint 同一题集，每环境只记录一次。
- final test：固定且不相交 seed `20000..20099`，100 episodes，确定性动作。
- checkpoint selection 和 final test 的 seed 均不参与训练或调参。
- 显式 `--seed N` 设置训练全局 seed；配置加载器固定派生
  `env_seed=N*1,000,000`，XuanCe worker 使用其后的连续 seed。不同训练 seed 的
  环境 block 互不重叠，且不接触 checkpoint/final 的 `10000/20000` 保留集。
  `env_seed` 及映射方案作为 seed identity 单独写入 provenance，不混入跨 seed
  应保持一致的超参数合同。
- benchmark 按 exact vector-step schedule 执行，最后不足一个 interval 时保留精确 tail，不静默少训/多训。
- final 聚合器拒绝 benchmark、少于 100 episodes、缺失显式确定性 seed 协议、重复训练 seed 和跨 run 协议不一致。
- learned-policy final 聚合还要求可验证的 `best_model_provenance.json`，复算全部
  episode 指标并校验 score/reward/agent mean；同时拒绝 runtime/config/checkpoint
  hash 不匹配、同一 checkpoint 冒充多个 seed、训练设备或训练合同不一致。
- 最终表由显式 runs manifest 选取。random / idle-only 各只有一个固定策略点估计，
  不复制成伪造的多训练 seed，跨 seed std 记为 N/A。
- advantage normalization 与 ValueNorm 仅统计 active agent。
- 新增 rollout 全 minibatch 聚合的 KL、clip fraction、ratio p05/p95、entropy、loss、gradient norm 和 active-mask fraction。
- 数值验证和录像均使用同一确定性动作路径。

当前验证：provenance/config/严格聚合/eval 定向测试 `71 passed, 2 warnings`；冻结后的
完整回归 `194 passed, 2 warnings`。保存型 benchmark smoke 与随后严格 sidecar 加载
test 均已通过；事件文件包含 KL、clip fraction、ratio p05/p95、entropy、loss、
gradient norm、invalid-ratio/nonfinite 计数和 active-mask fraction。原模型严格复测
及对应视频已完成；正式 seed 1 benchmark 尚未启动。

### 运行记录

| 阶段 | tmux / run | 状态 | 结果 |
| --- | --- | --- | --- |
| full pytest | — | passed | `194 passed, 2 warnings` |
| smoke benchmark | `runs/mappo_robust_eval_highway/20260807_214028_754748_benchmark_seed_1/` | passed | exact 16 steps；固定 seed 10000；训练前后各 1 episode；诊断成功落盘 |
| 保存型 provenance smoke benchmark | `runs/mappo_robust_eval_highway/20260807_222511_321788_benchmark_seed_1/` | passed | best/final checkpoint 与 sidecar 全部落盘；best SHA `f920585de965735fcf472791827cda3b5332cd4cce33c62b01f1e486d628f509`；runtime source SHA `1707c568e2c7f1f00f2e09469dc7fac260aadf2c79406054e9907309a9e784f9`；记录 `training_env_seed=1000000` |
| provenance smoke test | `runs/mappo_robust_eval_highway/20260807_222533_600764_test_seed_1/` | passed, smoke-only | sidecar/config/source/checkpoint hash 全部验证后成功加载；fixed final seed 20000 跑完 1 episode；少于 100 episodes，按设计不能进入最终聚合 |
| sandbox W&B test 尝试 | `runs/mappo_robust_eval_highway/20260807_214157_502452_test_seed_1/` | infra failure, preserved | sandbox 禁止本地 W&B socket；目录为空，未产出模型或 eval artifact；随后诊断 test 改用 TensorBoard |
| 原 best_model 严格 final test | `runs/mappo_robust_eval_highway/20260807_214301_472770_test_seed_1/` | complete, diagnostic-only | 100 episodes：arrival `0.17`、collision `0.83`、reward `9.851`；协议严格，但原 checkpoint 无 sidecar，需 `--allow-unverified-model`，最终聚合器按设计拒绝 |
| formal benchmark seed 1 | tmux `mappo-i1-robust-seed1`; `runs/mappo_robust_eval_highway/20260807_224810_780662_benchmark_seed_1/` | complete | 1M steps、101 次固定评估；best=epoch 1 / 10k：arrival `0.20`、collision `0.74`、truncation `0.06`、reward `9.705`；epoch 7–100 summary 逐项相同为 `0.20/0.80/0.00/10.400` |
| 原 best_model video | 同上 `videos/` | complete | seed 20000..20005 共 6 段；arrival `0.167`、collision `0.833`；合并视频 23.2 s；逐局指标与数值验证完全一致 |
| 新 best_model final test + video | tmux `mappo-i1-robust-seed1-final`; `runs/mappo_robust_eval_highway/20260808_025334_661367_test_seed_1/` | complete, below target | fixed seed 20000..20099：arrival `0.24`、collision `0.70`、truncation `0.06`、reward `10.240`；6 段视频、合并视频和 contact sheet 完整 |

用户明确批准长期 CPU/内存占用后，正式 benchmark 和 final test 分别在上述两个
tmux session 中完成。两次运行均由日志确认 `WANDB_MODE=offline`，未使用后台绕过
方案；benchmark W&B offline run 正常收尾，日志错误扫描为 0。

### 原策略行为诊断

严格 100-episode rollout 的 active-step 动作分布显示，原模型不是动态让行策略，
而是接近按 agent 身份固定速度：agent 0/1 的 IDLE 比例分别为 `88.9%/85.1%`，
agent 2/3 的 FASTER 比例分别为 `99.4%/97.2%`，SLOWER 几乎从不出现。全部 83 个
碰撞 episode 的首个 joint action 都是 `(IDLE, IDLE, FASTER, FASTER)`；该 joint
action 占碰撞 episode 全员 active steps 的 `88.53%`。碰撞对分散于不同车辆组合，
说明策略没有形成稳健路权协调。严格确定性 `0.17/0.83` 与 legacy stochastic
`0.15/0.85` 接近，失败不是单纯由随机评估造成。

### 固定策略 sanity reference

严格 case 集合 `20000..20099` 的可聚合 v3 artifacts：

| policy | artifact | arrival | collision | truncation | reward |
| --- | --- | ---: | ---: | ---: | ---: |
| random | `runs/random_highway/20260807_220901_727594_test_seed_1/` | 0.09 | 0.75 | 0.16 | 5.8179 |
| idle-only | `runs/idle_only_highway/20260807_220901_727594_test_seed_1/` | 0.17 | 0.83 | 0.00 | 9.8589 |

原 MAPPO 严格诊断几乎等同 idle-only，不能作为有效学习 baseline。更早的 sanity
v1/v2 artifacts 均保留但标为 superseded，不进入最终 manifest。

### 下一步判据

- 若新增日志显示 KL 或 clip fraction 持续偏高并伴随 entropy 快速下降，下一轮只把 learning rate 从 `3e-4` 改为 `1e-4`。
- 若 KL 与 clip fraction 很小、ratio 长期贴近 1，则下一轮只把 `n_epochs` 从 `2` 改为 `5`。
- 不先调整 entropy coefficient 或 gradient clip；不根据 final seeds 选择参数。

### Iteration 1 live PPO diagnostics（126,960 steps 快照）

只读解析 active W&B offline protobuf，共 62 个完整 rollout 更新；每条
`update_count=4`，与 `2 epochs × 2 minibatches` 一致。固定评估从 20k 到 120k
平台在 arrival `0.20`、collision `0.80`，best 暂为 10k 的 `0.20/0.74`。

| metric | first 10 mean | last 10 mean | all q05 / q50 / q95 |
| --- | ---: | ---: | ---: |
| approx KL | 0.00481 | 0.00091 | 0.00036 / 0.00125 / 0.00509 |
| clip fraction | 0.0644 | 0.0101 | 0.00007 / 0.0125 / 0.0805 |
| entropy | 0.726 | 0.455 | 0.432 / 0.493 / 0.776 |
| actor loss | -0.00265 | -0.00063 | -0.00305 / -0.00068 / -0.00006 |
| critic loss | 0.572 | 0.340 | 0.310 / 0.358 / 0.588 |
| gradient norm | 1.913 | 1.295 | 0.579 / 1.449 / 2.614 |
| active-mask fraction | 0.917 | 0.970 | 0.909 / 0.971 / 0.980 |

最大 minibatch KL `0.02658 << target_kl 0.25`；最大 gradient norm
`6.774 < clip 10`；critic loss 无发散，invalid/nonfinite 全为 0。当前证据排除
更新过强、梯度爆炸和 value 数值失稳，更符合 actor 更新逐渐变弱且 entropy 从
早期约 `0.73` 收缩到近期约 `0.46`，过早集中在高碰撞局部最优。该判断只是
中期可证伪假设，仍等待完整 1M benchmark 和独立 final test；不据此查看或调节
final seeds。

219,120-step 复查（107 个 rollout）进一步支持该假设：最近 10 条 KL 均值
`0.000924`、clip fraction `0.00727`、entropy `0.457`，最新 entropy `0.4115`；
相比 126,960-step 快照，KL 下降约 36%、clip 下降约 60%，而 60k–220k 的
deterministic evaluation 逐项相同。critic loss 近期 `0.310`、gradient norm
`1.195`，nonfinite 仍为 0。即策略概率继续变尖锐，但 argmax 行为没有改善。

关键同位置对照：旧 benchmark 的 epoch 28 / 280k 用 5 局 stochastic 评估得到
arrival `1.00`、collision `0.00` 并选为 best；Iteration 1 在 epoch 28 用固定 50 局
deterministic 评估得到 arrival `0.20`、collision `0.80`，且没有成为 best。这直接
证明旧 epoch-28 峰值是 checkpoint 小样本与随机评估造成的极值失真，不是该训练
时刻真实学会了协调。

### Iteration 1 完整结果与复盘

benchmark 于 2026-08-08 02:50 完成。共 101 条评估记录，覆盖 epoch `0..100`、
step `0..1,000,000`；`final_model.pth` 正常生成，训练日志无
Traceback/RuntimeError/OOM/NaN/Inf。best checkpoint 的实际 SHA-256
`cc0b605e07e5cd2691f0211d8664e3c0826c350866c39e26a6aa37378fd76835`
与 provenance sidecar 一致；sidecar 记录 training seed `1`、env seed `1,000,000`、
CPU、runtime source SHA 和 training contract SHA，选中 epoch 1 / step 10,000。

独立 final test 严格加载该 best checkpoint，在未参与选模的固定 seeds
`20000..20099` 上运行 100 episodes。数值结果和 checkpoint 选择集对比如下：

| 数据集 | episodes | arrival | collision | truncation | reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint selection，10000..10049 | 50 | 0.20 | 0.74 | 0.06 | 9.7047 |
| independent final，20000..20099 | 100 | 0.24 | 0.70 | 0.06 | 10.2399 |

选择集与 final 的 arrival/collision 差均为 `0.04`，方向还是 final 略好；新版协议
没有重现旧版 `1.00` 对约 `0.15` 的严重乐观失真。它证明选模协议修复有效，但不
证明策略已达标。final artifacts：

- `eval_records.jsonl` SHA-256：
  `93842354e98ec672af67864f4bf3ecdfc71aa13d90f0bde96fe88a1070e65ee9`
- 6 个视频对应固定 seeds `20000..20005`，逐局数值除 phase/env_id 等记录元数据外
  与 100 局记录的前 6 局一致；合并视频 29.3 秒。
- 视频 summary SHA-256：
  `b100279bec221bdcdd67976ec9756e93ce319dfa079d05a8b0601bc374036155`
- 合并视频 SHA-256：
  `bddcdc760116e82bd4bbe43525a598fc4ed41f2e930f069eb8558ccffb0a197a`

严格聚合器使用显式
`results/mappo_iteration1_seed1_final_manifest.json` 成功重算并接受所有记录：

| policy | training seeds | arrival | collision | truncation | reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iteration 1 MAPPO | 1 | 0.24 | 0.70 | 0.06 | 10.2399 |
| random | 固定策略点估计 | 0.09 | 0.75 | 0.16 | 5.8179 |
| idle-only | 固定策略点估计 | 0.17 | 0.83 | 0.00 | 9.8589 |

MAPPO 比 idle-only 的 arrival 高 0.07、collision 低 0.13，且 reward 仅高 0.381；
相对 random 的 collision 只低 0.05。单 seed 下这些差距不能称为“明显优于”，更
远未达到 arrival `>=0.60` / collision `<=0.30` 的三 seed 标准。

完整 W&B offline history 含 488 个 rollout diagnostics，覆盖 step
`2,032..999,408`。早期与最后 10 次更新的对比：

| metric | first 10 mean | last 10 mean | full q05 / q50 / q95 |
| --- | ---: | ---: | ---: |
| approx KL | 0.004809 | 0.000389 | 0.000079 / 0.000594 / 0.002431 |
| clip fraction | 0.06438 | 0.000013 | 0 / 0.000157 / 0.02962 |
| ratio p05 | 0.8694 | 0.9689 | 0.8936 / 0.9581 / 0.9863 |
| ratio p95 | 1.1253 | 1.0229 | 1.0135 / 1.0442 / 1.1138 |
| entropy | 0.7257 | 0.6273 | 0.4538 / 0.5503 / 0.6560 |
| actor loss | -0.002648 | -0.000194 | -0.001131 / -0.000315 / 0.000129 |
| critic loss | 0.5721 | 0.2801 | 0.2541 / 0.2986 / 0.3765 |
| gradient norm | 1.9130 | 0.6311 | 0.4032 / 0.6748 / 1.7299 |

最大 minibatch KL `0.02658 << target_kl 0.25`，最大 gradient norm `6.774 < 10`，
nonfinite 与 invalid-ratio 均为 0。entropy 在中期最低到约 `0.407` 后末段回升，
因此完整曲线不支持“永久熵塌缩”；critic 也稳定。最强证据是 PPO 更新越来越接近
identity：末段 clip fraction 几乎为 0、ratio 分位数贴近 1、actor loss 很小，且
固定 argmax 行为在 epoch 7–100 完全不变。

Iteration 1 结论：框架/选模修复成功，性能假设失败；当前控制配置陷入稳定的高碰撞
局部最优。按预注册判据，Iteration 2 只把 `n_epochs: 2 -> 5`，其可证伪预测是
KL/clip/ratio 的有效更新幅度增加，固定评估在相同 1M 预算内脱离 `0.20/0.80`
平台。learning rate、entropy coefficient、网络、buffer、minibatch、环境与奖励均
保持不变；先 smoke，通过后才启动正式 benchmark。

## Iteration 2：increase PPO epochs

假设：Iteration 1 的末段 clip fraction 几乎为 0、ratio 贴近 1、actor loss 很小，
说明每批 rollout 的 PPO 更新不足。只把 `n_epochs` 从 2 增加到 5，应提高有效更新
幅度并使固定 argmax 行为脱离长期 `0.20/0.80` 平台；若完整训练仍无改善，或出现
KL/clip 过高，则该假设被证伪。

配置：`configs/mappo/intersection-multi-agent-v1-epochs5.yaml`。解析后与 Iteration 1
只有 `n_epochs: 2 -> 5`、隔离产物所需的 `output_dir` 和 `config_path` 不同。

save-enabled smoke：
`runs/mappo_epochs5_highway/20260808_030102_264033_benchmark_seed_1/`。命令显式使用
`--smoke --n-epochs 5`，避免 smoke 默认 epoch=1 掩盖目标改动。16 steps 完成，
best/final model 和 provenance sidecar 均生成；sidecar 记录 n_epochs=5、
training env seed=1,000,000、CPU，best checkpoint SHA
`f888df3932602bd7616d9f4435f3ca0eb59e3676e52c513476cf8846c8ce008d`
校验一致。TensorBoard diagnostics 的两次 rollout 均为 `update_count=5`，nonfinite=0。

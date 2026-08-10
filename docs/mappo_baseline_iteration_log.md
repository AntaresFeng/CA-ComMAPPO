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

### Iteration 2 完整结果与复盘

正式 benchmark：tmux `mappo-i2-epochs5-seed1`，
`runs/mappo_epochs5_highway/20260808_030423_818708_benchmark_seed_1/`。
`WANDB_MODE=offline`，1M steps，101 条固定评估，日志错误扫描为 0；best/final
checkpoint 均正常生成。best 为 epoch 11 / 110k，SHA-256
`511ecfe1b2b1171d55a09b27966f4fc91dc13761b0cd19a61fde6c1412381779`：

| checkpoint | episodes | arrival | collision | truncation | reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iteration 1 best（epoch 1） | 50 | 0.20 | 0.74 | 0.06 | 9.7047 |
| Iteration 2 best（epoch 11） | 50 | 0.30 | 0.70 | 0.00 | 11.9676 |
| Iteration 2 final epoch 100 | 50 | 0.24 | 0.76 | 0.00 | 10.9938 |

Iteration 2 的固定曲线确实脱离控制组 epoch 7–100 完全不变的平台；epoch 20–100
有 65 个不同 summary。但 best 在 110k 后没有再刷新，末段退化到约 `0.24/0.76`。

完整 W&B offline history 含 488 个 rollout diagnostics：

| metric | first 10 mean | last 10 mean | full q05 / q50 / q95 |
| --- | ---: | ---: | ---: |
| approx KL | 0.007550 | 0.001625 | 0.000986 / 0.001802 / 0.003695 |
| clip fraction | 0.10667 | 0.01956 | 0.01142 / 0.02142 / 0.04680 |
| ratio p05 | 0.8279 | 0.9364 | 0.8659 / 0.9172 / 0.9614 |
| ratio p95 | 1.1891 | 1.0745 | 1.0418 / 1.0848 / 1.1385 |
| entropy | 0.8163 | 0.2760 | 0.2643 / 0.3288 / 0.5878 |
| actor loss | -0.008218 | -0.001488 | -0.004088 / -0.001770 / -0.001023 |
| critic loss | 0.5045 | 0.2672 | 0.2384 / 0.2848 / 0.3571 |
| gradient norm | 1.4475 | 0.5812 | 0.4312 / 0.6490 / 1.2597 |

相对 Iteration 1，5 epochs 成功提高 KL、clip fraction、actor-loss 幅度并让策略持续
移动；最大 minibatch KL 仍为 `0.02658`、最大 gradient norm `6.774`、nonfinite 与
invalid ratio 均为 0。然而 entropy 从早期 `0.816` 收缩到末段 `0.276`，明显低于
控制组末段 `0.627`，与 110k 后的性能退化同时出现。

独立 final test：tmux `mappo-i2-epochs5-seed1-final`，
`runs/mappo_epochs5_highway/20260808_210848_991096_test_seed_1/`。严格加载 best_model，
固定 seeds `20000..20099`，100 episodes：

| iteration | selection arrival/collision | final arrival | final collision | final truncation | final reward |
| --- | --- | ---: | ---: | ---: | ---: |
| 1：n_epochs=2 | 0.20 / 0.74 | 0.24 | 0.70 | 0.06 | 10.2399 |
| 2：n_epochs=5 | 0.30 / 0.70 | 0.25 | 0.75 | 0.00 | 11.1848 |

Iteration 2 final 的 arrival 仅 `+0.01`、collision 反而 `+0.05`，故“只增加 PPO
epochs 会显著改善最终策略”被否证。selection 与 final 的 arrival/collision 差均为
0.05，仍没有旧版严重选模失真。6 个视频固定对应 seeds `20000..20005`，逐局记录
与数值验证一致；合并视频 29.6 秒。关键 SHA-256：

- final `eval_records.jsonl`：
  `07bb009f48d5fe675c781c0152a6b9bc18db22676e6dc655a18a0859e0353c7f`
- video summary：
  `baf15203f4d36c51b33e653ea24b1817c53059056b09e27b21d45549491db626`
- combined video：
  `69ea15b2352de5d875c2cfe2aa4ab4c2a46b69eb9d8a48d69179f77c13e2cc89`

显式 `results/mappo_iteration2_seed1_final_manifest.json` 已通过严格聚合。Iteration 2
仍不满足单 seed 的 `arrival>=0.50 / collision<=0.40`，也没有明显优于 Iteration 1。

下一轮不依据 final seeds 的具体 episode 调参；依据训练期 entropy 曲线提出
Iteration 3：保留 n_epochs=5，只将 `ent_coef: 0.01 -> 0.03`。可证伪预测是末段
entropy 不再降到约 0.28，checkpoint 曲线不会在 110k 后持续退化，并在独立协议上
显著降低 collision；若只维持随机性却不改善固定 argmax 协调，该假设失败。

## Iteration 3：entropy regularization

假设：Iteration 2 的有效 PPO 更新已增强，但 entropy 从约 0.82 降到末段 0.28，且
best 在 110k 后退化。保持 n_epochs=5，只将 `ent_coef: 0.01 -> 0.03`；若训练期
entropy 得以维持且固定 argmax 的 collision 显著下降，则支持“过早策略集中阻碍
协调”的解释。若 entropy 上升但 final 协调不改善，则该解释被否证。

配置：`configs/mappo/intersection-multi-agent-v1-epochs5-ent03.yaml`。解析差异测试
证明相对 Iteration 2 仅有 `ent_coef`、隔离产物的 `output_dir` 和 `config_path` 不同。

save-enabled smoke：
`runs/mappo_epochs5_ent03_highway/20260808_211313_119098_benchmark_seed_1/`。
命令显式使用 `--smoke --n-epochs 5`；16 steps 完成，best/final checkpoint 与
sidecar 完整。sidecar 记录 n_epochs=5、ent_coef=0.03，best SHA
`c5564342bd82dbea3ff500ea2892e934b4cc0ab77e82ed7207fcefda1856e977`
校验一致；两次 rollout 的 update_count 均为 5、nonfinite=0。

### Iteration 3 完整结果与复盘

正式 benchmark：tmux `mappo-i3-epochs5-ent03-seed1`，
`runs/mappo_epochs5_ent03_highway/20260808_211505_584710_benchmark_seed_1/`。
1M steps、101 条固定评估、W&B offline 正常结束，日志错误扫描为 0。best 在
epoch 98 / 980k，selection 为 arrival `0.36`、collision `0.64`、truncation `0`、
reward `12.9011`；best SHA
`934f6fa8d8e524492bc1de85da97e89c6f4289562407fd3e6bd835e58b0e8b57`。
最后 10 次固定评估均值为 arrival `0.298`、collision `0.686`，改善持续到训练末段，
与 Iteration 2 在 110k 后不再刷新形成对照。

完整 488 次 rollout diagnostics：

| metric | first 10 mean | last 10 mean | full q05 / q50 / q95 |
| --- | ---: | ---: | ---: |
| approx KL | 0.007664 | 0.002103 | 0.001625 / 0.003383 / 0.005099 |
| clip fraction | 0.10043 | 0.01824 | 0.01102 / 0.03326 / 0.06136 |
| ratio p05 | 0.8346 | 0.9243 | 0.8527 / 0.8894 / 0.9390 |
| ratio p95 | 1.1881 | 1.0808 | 1.0670 / 1.1239 / 1.1592 |
| entropy | 0.8157 | 0.6814 | 0.6397 / 0.6813 / 0.7628 |
| actor loss | -0.008188 | -0.001968 | -0.005180 / -0.003002 / -0.001555 |
| critic loss | 0.5046 | 0.2756 | 0.2648 / 0.2979 / 0.3528 |
| gradient norm | 1.4003 | 0.6729 | 0.4841 / 0.6915 / 1.3850 |

entropy 全程最低约 `0.600`，末段 `0.681`，显著高于 Iteration 2 的 `0.276`；同时
KL/clip 保持有效，最大 minibatch KL `0.02774`、最大 gradient norm `6.776`，
nonfinite/invalid ratio 均为 0。因此“ent_coef=0.03 可防止策略过早集中”得到支持。

独立 final test：tmux `mappo-i3-epochs5-ent03-seed1-final`，
`runs/mappo_epochs5_ent03_highway/20260808_234624_477249_test_seed_1/`。固定 seeds
`20000..20099` 的 100 episodes 为 arrival `0.34`、collision `0.66`、truncation `0`、
reward `12.5375`。相对 Iteration 2 final：arrival `+0.09`、collision `-0.09`、reward
`+1.353`；selection 与 final 的 arrival/collision 仅各差 0.02，稳健选模继续成立。

6 个视频对应 seeds `20000..20005`，逐局记录与数值验证一致；合并视频 31.9 秒。
关键 SHA-256：

- final `eval_records.jsonl`：
  `1e839d85ddc1ccebac2f1ab0435d482aba7e5b5d5396d06ecf13a4a68cc40a3c`
- video summary：
  `e72c9904a6bf8ddb5a2cb61c25e1a66c1571407fde22859575a6f9f470f04999`
- combined video：
  `095c2b58426b7281fbe5695269753d630afacd703674bc243f82918705f25936`

显式 `results/mappo_iteration3_seed1_final_manifest.json` 已通过严格聚合。虽然这是
目前最佳迭代，但仍低于单 seed `0.50/0.40` 门槛，不能进入三 seed 验收阶段。

Iteration 4 的依据来自训练曲线而非 final seed 个案：Iteration 3 的最佳点直到
980k 才出现，且没有像 Iteration 2 那样早期退化，说明 1M budget 可能截断了仍在
进展的学习。下一轮保持全部超参和协议不变，只把 `running_steps: 1M -> 2M`。
可证伪预测是 1M 之后固定 selection 继续改善并达到至少 `0.50/0.40` 附近；若
1M–2M 平台化或退化，则“预算不足”假设失败。

## Iteration 4：extend training budget

假设：Iteration 3 的 best 直到 980k 才出现，且高 entropy 避免了早期策略冻结；
1M budget 可能截断仍在进展的学习。保持网络、优化器、n_epochs=5、ent_coef=0.03、
rollout 和评估协议不变，只将 `running_steps: 1,000,000 -> 2,000,000`。

配置：`configs/mappo/intersection-multi-agent-v1-epochs5-ent03-2m.yaml`。解析差异
测试证明相对 Iteration 3 仅 `running_steps`、隔离产物的 `output_dir/config_path`
不同。

save-enabled smoke：
`runs/mappo_epochs5_ent03_2m_highway/20260808_235007_515289_benchmark_seed_1/`。
显式 `--smoke --n-epochs 5 --running-steps 32 --eval-interval 16`，完成两个 exact
16-step epoch 和 3 次固定评估；best/final/sidecar 完整，best SHA
`4a336318aa723726d6b1153ac61ec3213c4c81abb7c80df70a76798993a801c3`。
4 次 rollout 的 update_count 均为 5，nonfinite=0。

### Iteration 4 完整结果与复盘

正式 benchmark：tmux `mappo-i4-epochs5-ent03-2m-seed1`，
`runs/mappo_epochs5_ent03_2m_highway/20260808_235213_111023_benchmark_seed_1/`。
2M steps、201 条固定评估、W&B offline 正常结束，日志错误扫描为 0。前 1M 的全部
评估记录（含 episode payload）与 Iteration 3 逐值相同，确认这是可复现的单变量
预算实验。best 在 epoch 168 / 1.68M，selection 为 arrival `0.36`、collision `0.64`、
truncation `0`、reward `12.9919`；best SHA
`44002ff6cf594105b3dbdff38923609debbed47132b1bb6f9867fff0fa723864`。

额外 1M 并未改善任务成功率：Iteration 3 的 best 也是 `0.36/0.64/0`，只有 selection
reward 增加约 `0.091`。1M 后固定评估均值为 arrival `0.2616`、collision `0.7048`、
truncation `0.0336`；最终 epoch 200 为 `0.20/0.68/0.12`。分段结果也显示延长阶段
不是仍在单调学习：

| step window | mean arrival | mean collision | mean truncation | mean reward |
| --- | ---: | ---: | ---: | ---: |
| 0.9M–1.0M | 0.2980 | 0.6860 | 0.0160 | 12.197 |
| 1.0M–1.5M | 0.2596 | 0.7184 | 0.0220 | 11.683 |
| 1.5M–2.0M | 0.2636 | 0.6912 | 0.0452 | 12.091 |

完整 976 次 rollout diagnostics 均为有限值。第二个 1M 中，前/后 10 次的 approximate
KL 为 `0.001952 -> 0.001512`，clip fraction 为 `0.01658 -> 0.01091`，entropy 为
`0.6578 -> 0.6844`，actor-loss 幅度为 `0.001889 -> 0.001439`；最大 minibatch KL
`0.01346`，最大 gradient norm `1.935`，nonfinite/invalid ratio 均为 0。训练数值健康，
但额外更新逐渐变弱，不能用崩溃、NaN 或梯度爆炸解释性能停滞。

独立 final test：tmux `mappo-i4-epochs5-ent03-2m-seed1-final`，
`runs/mappo_epochs5_ent03_2m_highway/20260810_213146_520005_test_seed_1/`。严格加载
best_model，固定 seeds `20000..20099` 的 100 episodes 为 arrival `0.31`、collision
`0.68`、truncation `0.01`、reward `12.3011`：

| iteration | training budget | selection arrival/collision | final arrival | final collision | final truncation | final reward |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 1M | 0.36 / 0.64 | 0.34 | 0.66 | 0.00 | 12.5375 |
| 4 | 2M | 0.36 / 0.64 | 0.31 | 0.68 | 0.01 | 12.3011 |

6 个视频对应 seeds `20000..20005`，去除 phase/env_id/policy 记录元数据后与数值验证
前 6 局逐字段一致；合并视频 31.3 秒。关键 SHA-256：

- final `eval_records.jsonl`：
  `4408e2ee653fd6462063b781682b6bf9bba9e938f3eb7dccaf1d1965e0a8668f`
- video summary：
  `6fbe131e7a61c658499d79b82c0b2769ea8e81401a17e180e3517c2f4f5dedd4`
- combined video：
  `0ad47be42c214612c12faf399a0f477928e812dd7701131b75007828984b458a`

显式 `results/mappo_iteration4_seed1_final_manifest.json` 已通过严格聚合。Iteration 4
final 相对 Iteration 3 为 arrival `-0.03`、collision `+0.02`、reward `-0.2365`；因此
“Iteration 3 只是被 1M 预算截断，延长到 2M 会继续改善”的假设被否证。继续增加相同
训练预算没有证据基础，且当前仍远低于单 seed `0.50/0.40` 门槛。

### 奖励排序诊断与 Iteration 5 决策

Iteration 1–4 的共同失败形态是 collision 长期在 `0.64–0.75`，而 reward 随策略改善
仍可升到约 `12–13`。源码核验确认当前每个 active agent 的普通步 reward 为
`-10 * crashed + 0.5 * normalized_speed`；到达步覆盖为 `+5`，环境在任一受控车辆
碰撞时全局终止。适配器训练返回的是 masked per-agent reward，并非 highway-env 的
标量团队均值；因此“其余 agent 稀释碰撞”不是环境直接先求平均，但 MAPPO 最终会在
agent/batch 维聚合损失，这个风险方向仍成立。

在 100 个决策步、`gamma=0.99` 下，按全速上界计算：满时限 discounted speed return
约 `31.70`，第 100 步到达约 `33.36`，第 100 步碰撞仍约 `+28.00`。这比仅看未折扣
上界 `50/+5/-10` 更直接地说明当前优化目标排序错误：晚碰撞仍可获得显著正回报，
而延迟到达也可高于立即到达。

候选奖励合同为 collision `-40`、high-speed `0.01`、arrival `+10`、
`reward_speed_range=[5,9]`、`normalize_reward=false`。即使第一轮保持 `gamma=0.99`，
100 步上界也变为 timeout `+0.634`、末步 arrival `+4.328`、末步 collision
`-14.155`，已经恢复 timeout < arrival 且 collision < 0 的排序。

`gamma=0.998` 与 `gae_lambda=0.99` 也有独立的时间尺度依据：5 Hz 下若希望保留每秒
`0.99/0.95` 的折扣，逐决策步应分别为 `0.99^(1/5)=0.997992` 与
`0.95^(1/5)=0.989794`。但把奖励、gamma、GAE 同时修改会混淆归因。为遵守单轮最小
可证伪改动，Iteration 5 把三项相互依赖的奖励系数作为一个
“奖励排序合同”整体变更，保持 Iteration 3 的 1M、n_epochs=5、ent_coef=0.03、
gamma=0.99、GAE=0.95 和所有评估协议不变。可证伪预测：collision 明显下降；若
主要转为 truncation，则下一轮再单独校正 5 Hz 的 gamma/GAE 时间尺度。

奖励改动会改变任务定义；2026-08-10 已获得明确批准后才开始实施。

## Iteration 5：correct reward ordering

假设：Iteration 1–4 的高碰撞局部最优主要来自任务奖励排序错误，而不是训练预算、
PPO 更新强度或数值稳定性。把全时限速度 shaping 压到 arrival bonus 的 10% 以内，
并使一次 collision penalty 大于其余三个 agent 的 arrival bonus 总和，应显著降低
固定 argmax 策略的 collision。若 collision 没有明显下降，则该奖励解释被否证；若
collision 下降但 truncation 大幅上升，则说明奖励排序已修复、后续瓶颈转为长时程
信用分配。

配置：
`configs/mappo/intersection-multi-agent-v1-epochs5-ent03-reward-ordering.yaml`。
相对 Iteration 3，只通过算法配置的 `highway_config` 层覆盖：

- `collision_reward: -10 -> -40`
- `high_speed_reward: 0.5 -> 0.01`
- `arrived_reward: 5 -> 10`

三项被预注册为一个不可拆分的奖励排序合同；共享 highway 配置未修改，历史实验仍可
按原任务配置解析。解析差异测试确认除此合同与隔离产物所需的
`config_path/output_dir` 外无其他变化；`running_steps=1M`、n_epochs=5、
ent_coef=0.03、gamma=0.99、GAE=0.95、网络、buffer、seeds 和评估协议全部冻结。

save-enabled smoke：
`runs/mappo_epochs5_ent03_reward_ordering_highway/20260810_214328_630594_benchmark_seed_1/`。
显式 `--smoke --n-epochs 5 --running-steps 32 --eval-interval 16`，完成两个 exact
16-step epoch 和 3 次固定评估；best/final/sidecar 完整。sidecar 记录 CPU、training
seed 1、env seed 1,000,000 及新奖励合同，best SHA
`3b63be082e809ee6ecef2be6eb3851c3dcc7aacbdf0e1a65fcdb10dd8a100180` 校验一致。
4 次 rollout 的 update_count 均为 5，nonfinite=0。初始碰撞 episode 的 mean reward
为 `-19.797`，说明新合同已在真实环境 rollout 生效，碰撞不再像旧任务那样获得高额
正回报。全量测试为 `198 passed, 2 warnings`。

正式 benchmark 于 2026-08-10 21:45 启动：tmux
`mappo-i5-reward-ordering-seed1`，
`runs/mappo_epochs5_ent03_reward_ordering_highway/20260810_214549_194215_benchmark_seed_1/`。
命令使用 `--mode benchmark --seed 1` 与 `WANDB_MODE=offline`。epoch 0 的 50 个固定
seed 在终局类别、步数和逐 agent 到达/碰撞状态上与 Iteration 3 初始策略全部一致，
只因奖励合同不同而从 reward `8.4516` 重算为 `-6.9040`，排除了初始策略或环境漂移。
epoch 1 / 10k 首次训练后为 arrival `0`、collision `0.04`、truncation `0.96`、reward
`-0.7015`；奖励排序已迅速抑制碰撞，但是否能进一步学会通行必须由完整 1M 曲线和
独立 final test 决定，不能用这个早期 checkpoint 宣称成功。

该 run 在 epoch 8 / 80k 后主动中止并完整保留，不能用于最终聚合。奖励重标定暴露出
一个此前被高碰撞终局掩盖的 adapter contract 缺陷：agent 首次进入到达区域时收到
arrival reward 且被永久 mask，但 `info["arrived"]` 与 `global_terminated` 每步重新
读取瞬时 `has_arrived`。车辆继续仿真并离开判定区域后，已终止 agent 的 arrived 会
从 true 退回 false。epoch 8 的记录中，终局 mean agent arrival 仅 `0.295`，而利用
新合同“非碰撞下累计 reward > 2 必然领取过 +10 arrival bonus”可反推出 ever-arrived
比例为 `0.55`；50 局中 48 局丢失至少一个历史到达标志。best 当时为
`0.02/0.04/0.94`，但由于终止/指标口径不可靠且没有 final_model，不得解释为正式
Iteration 5 结果。下一步先把 per-agent arrival 与终止状态改为 episode 内 sticky，
补充“先后到达也触发全局成功”的回归测试，再从 0 重新 smoke 和正式训练。

框架修复：adapter 现在分别保存 upstream 的瞬时 `current_arrived` / raw terminated，
以及 episode 内 sticky 的 adapter-facing `arrived` / `agents_terminated`。已经到达的 agent
持续保持 terminated；任一碰撞仍全局失败，所有 agent 先后到达则立即全局成功。新增
回归测试模拟 agent 0 在 step 1 到达后离开几何判定区、agent 1 在 step 2 到达，验证
step 2 正确返回 `(True, True)` 与 global termination。全量测试为
`199 passed, 2 warnings`。

修复后的 save-enabled 50-episode smoke：
`runs/mappo_epochs5_ent03_reward_ordering_highway/20260810_215758_263999_benchmark_seed_1/`。
完成两个 exact epoch、3 次各 50 局固定评估、best/final/sidecar；4 个 rollout 的
update_count 均为 5、nonfinite=0，best SHA
`d9120ccaba58707c063398346741d737d821a83bc12e9ceed5c73348eae0275a`。所有非碰撞
episode 中，由 arrival bonus 反推的历史到达 agent 与记录的 sticky arrived 完全一致，
丢失标志计数从中止 run 的 48/50 降为 0。

这项修复改变 runtime source contract。Iteration 1–4 的产物继续保留为修复前历史
证据，但其 per-agent arrival 指标和先后到达终止口径不能与修复后正式结果直接混用；
严格聚合器也会通过 runtime source hash 拒绝把旧 learned artifact 与新模型并表。
修复后将先重跑 random/idle-only 固定 100 局基线，再从零重启正式 Iteration 5。

修复后、奖励合同一致的 sanity baseline 已按固定 seeds `20000..20099` 重跑：

| policy | arrival | collision | truncation | mean reward | mean agent arrival |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.10 | 0.75 | 0.15 | -12.4574 | 0.2375 |
| idle-only | 0.17 | 0.83 | 0.00 | -16.1293 | 0.1700 |

artifact 分别为
`runs/random_highway/20260810_220643_345839_test_seed_1/` 与
`runs/idle_only_highway/20260810_220643_345839_test_seed_1/`。相对修复前同 seed，random
有一局从 truncation 改为 arrival（`0.09/0.75/0.16 -> 0.10/0.75/0.15`），直接证明
sticky arrival 会影响全局成功口径；idle-only 仍为 `0.17/0.83/0`。reward 因任务合同
改变而转为负值，不应与旧 reward 横向比较。两个 artifact 均包含 100 个连续 episode
payload、正确奖励快照和固定协议，并通过显式
`results/sanity_reward_ordering_final_manifest.json` 的严格重算。eval records SHA-256：

- random：`8e0295571b93c8d3a4ead5bc8d138d6002a2e402157e225dcbe02138eec3097a`
- idle-only：`6c066c45c660ed07e9c1955caa0130b6e55ce9340a51700e0c5e13972393c4ae`

修复后的正式 Iteration 5 于 2026-08-10 22:09 从零重启：tmux
`mappo-i5-reward-ordering-sticky-seed1`，
`runs/mappo_epochs5_ent03_reward_ordering_highway/20260810_220913_630764_benchmark_seed_1/`。
exact-source smoke、正式 run 与 checkpoint provenance 的 runtime source SHA 均为
`dda9ae90dc7e4138194f9520013350e35909870289f61049c32c4e5ffcc49728`。
epoch 0 的 global arrival/collision/truncation、reward 和长度与中止 run 完全相同，
但 sticky mean agent arrival 从错误的 `0.07` 修正为 `0.17`；epoch 1 仍为
`0/0.04/0.96`，证明重启保持了奖励训练轨迹的可复现起点，同时修正了报告口径。

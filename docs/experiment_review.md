# 实验设计评审报告（Experiment Design Review）

> **历史快照**：本文记录 2026-07-23 的设计状态，其中奖励、车辆数和 PPO 超参已经
> 被后续实验替换。不得从本文复制当前实验配置；后续实验和论文统一以
> `docs/mappo_baseline.md` 及 canonical YAML 为准。其余算法设计发现保留作审计参考。

> 归档日期：2026-07-23。分支 `nice-try`。
> 评审对象：CA-ComMAPPO（Conflict-Aware Communication MAPPO）创新算法及其基线/消融矩阵。
> 评审方法：7 个维度 reviewer + 逐条对抗验证（adversarial verify）+ 完整性补充，共 63 个 agent、51 条确认发现、4 条被否决。关键结论均经独立读码与实跑环境验证。
> 评审时点：仅设计/实现阶段，**尚未跑多 seed 正式训练**；因此针对设计层的潜在失效模式，而非训练曲线。
> 关联文档：`docs/ca_commappo_design.md`（冻结设计）、`docs/experiments_runbook.md`（实验矩阵）、`docs/thesis-planning.md`（论文主线）。

## 0. 实验核心目标

论文主线（`docs/thesis-planning.md`）：**MAPPO → Attention-MAPPO（PI 编码器）→ CA-ComMAPPO（冲突感知显式通信）** 三级递进，在 highway-env 无信号路口证明：用「带冲突先验的选择性注意力通信」替代 CommNet 的全连接平均通信，能提升通过效率与安全性，且对车辆顺序/数量变化不变。

落地为本分支的代码、配置与部分单元测试。本评审按用户指定的五个维度 + 一条贯穿全局的根因展开。

---

## §A. 贯穿全局的 Critical 根因：主配置下冲突先验完全失效

这是算法/训练/退化深挖/评估四个独立 reviewer **分别确认**、并经对抗验证保留的最重要的结论。

**事实链**（`ca_commappo/algorithms/conflict_aware_comm.py:118-142` + `configs/ca_commappo/intersection-multi-agent-v1.yaml:82`）：

1. 主配置 `controlled_vehicles: 2`，每车自屏蔽（`j≠i`）后**恰好剩 1 个通信邻居**。
2. `alive=torch.ones(...)`（`ca_commappo/algorithms/ca_commappo_agents.py:147`）恒全 1，所以只有自屏蔽在删项。
3. 对单元素 softmax 恒为 `1.0`，且 `d(softmax)/d(logit) = 1·(1−1) = 0`。

**直接后果**：

| # | 后果 |
|---|---|
| 1 | 聚合 `c_i = 1.0 · W_v m_other`——**与 `b_ij` 无关**，冲突先验对前向零影响。 |
| 2 | 可学标量 `λ` **无梯度、恒停在初值 1.0**；`W_q/W_k` 也无梯度（`W_v` 仍训练，因它经 `v` 直达 `c_i`）。 |
| 3 | 消融 **(c) no-prior 与 (d) full 在 n=2 下 bit-identical**（`max|c_full−c_no|=0`、`max|α_full−α_no|=0`）——主消融**无法检测冲突先验的贡献**。 |
| 4 | 注意力可视化（论文 §7 卖点）在主配置下 `α` 恒为 `[[0,1],[1,0]]`——热力图无信息。 |

**为何旧版 smoke 没暴露**：旧 `*-smoke.yaml` 用 `controlled_vehicles: 3`（先验此时有效），而正式配置用 `2`——**smoke 对、正式错**，门禁为错误的原因放行。现在两者已统一读取 highway 环境配置。

**为何泛化实验没救它**：runbook §4 泛化训练在 `controlled_vehicles=4`（先验有效），但 (a) 该 gen4 配置**尚未生成**；(b) CommNet/IC3Net/MAPPO-MLP 输入维度固定，无法在 6/8/10 车运行——**没有任何一个"先验有效 且 通信 baseline 可比"的设置**。即论文核心卖点"冲突感知 > 平均通信"在整个矩阵里**无处可证**。

**修复（已落实到配置层）**：共享 highway 配置使用 `controlled_vehicles: 3`，正式和 smoke 不再各自定义车数；消融 (c)/(d) 必须在 ≥3 下重做。

---

## 1. 环境建模（状态/动作/奖励/随机性）

### 1.1 奖励设计（Critical — 已实证量化）

`highway_config` 在六个算法配置中**逐字节相同**（横向公平前提，正确）。配置（`configs/ca_commappo/intersection-multi-agent-v1.yaml:82-89`，全算法相同）：

```yaml
collision_reward: -10   high_speed_reward: 0.5   arrived_reward: 5
reward_speed_range: [5.0, 9.0]   normalize_reward: false
```

奖励函数（`highway_env/envs/intersection_env.py:78-92`，每受控车）：

```python
reward = -10*crashed + 0.5*clip(speed∈[5,9]→[0,1]) + 5*arrived
reward = 5 if arrived else reward     # 到达步用 arrived_reward 覆盖
```

枚举各结局的**单 agent 累计奖励**（drive 9 m/s）：

| 结局 | 累计奖励 |
|---|---|
| step 10 到达 | **+10.00** |
| step 18 到达 | **+14.00** |
| step 5 到达 | **+7.50** |
| **超时（开满 20 步，不达、不撞）** | **+10.00** |
| 超时（全程怠速 0 m/s） | +0.00 |
| step 18 撞车 | **−0.50** |
| step 10 撞车 | −4.50 |
| step 5 撞车 | −7.00 |

**三个被量化证实的缺陷**：

- **(a) 到达与"超时狂飙"奖励完全相等（+10.00 = +10.00）。** 到达 bonus（+5）正好被"少走 10 步 × 0.5"的速度奖励抵消。永远不到达、开满全程超时拿 +10，与 step 10 高效到达**无差**。`arrival_rate` 作为学习信号几乎无梯度。
- **(b) 晚到达 > 早到达（+14 > +10 > +7.5）。** 到达 bonus 固定 +5 不随时间衰减——**奖励主动惩罚高效通过路口**，与 throughput/travel_time 指标矛盾。双 agent 下先到达者被 mask（reward=0、动作 IDLE），团队奖励由慢 agent 主导 → 鼓励"尽量拖"。
- **(c) 晚撞车几乎无惩罚（step 18 撞 = −0.50）。** `collision_reward` 固定 −10、不随回合长度缩放，速度奖励持续累积 → 越晚撞净惩罚越小，鼓励"最后时刻抢行"。

**根因**：`high_speed_reward`（0.5/step，满回合 +10）量级 ≥ `arrived_reward`（+5）。highway-env 官方默认 `high_speed_reward:1, arrived_reward:1, collision:-5, duration:13`（`intersection_env.py:42-54`），三者比例自洽；本项目改 `arrived=5, collision=−10` 却把 `high_speed` 砍到 0.5 并把 `duration` 拉到 20，破坏比例。

**建议**：`high_speed_reward` 降到 **0.1**；或 `arrived_reward` 提到 **≥ +10**；给到达加**时间衰减**（如 `5·(1−t/duration)`）；或开 `normalize_reward: true`（`_agent_reward` 已支持 lmap，`intersection_env.py:86-91`）。

### 1.2 决策频率（Critical — 实测）

六配置均**未覆盖 `policy_frequency`**，继承 highway-env 默认 `1`（`highway_env/envs/common/abstract.py:97-98`，`simulation_frequency:15`）。每次决策间 15 个仿真子步、1 整秒物理时间。

**1 Hz vs 5 Hz 实测**（同随机策略、同种子、`duration=20`）：

| | policy_frequency=1（当前） | policy_frequency=5 |
|---|---|---|
| 每决策间隔 | 1.0 s（15 仿真子步） | 0.2 s（3 仿真子步） |
| 9 m/s 车每决策移动 | 9.0 m | 1.8 m |
| **TTC<1s 冲突反应决策点** | **1 个** | **5 个** |
| 10 随机 episode 碰撞数 | 7 | 3 |

1 Hz 下 TTC<1s 冲突只剩 1 个决策点，错过即撞，**通信来不及起作用**。对 CA-ComMAPPO 尤其致命：冲突感知通信的全部价值在"提前几步协调"，但 1 Hz 决策间隔比典型路口 TTC 还长——`b_ij` 传递的 ttc/intent 要到下一个决策点（1 s 后）才被使用，那时冲突可能已发生。**5 Hz 是从"不可协调"到"可协调"的质变**。

⚠️ **强制联动**：`high_speed_reward:0.5` 是 **per decision step**（非 per second）。提频后 `max_episode_steps = duration × policy_frequency`：

| | 超时狂飙累计 | 到达累计 | 失衡比 |
|---|---|---|---|
| 1 Hz（当前） | +10 | +10 | 1:1 |
| **5 Hz 但不改奖励** | **+50** | **+10** | **5:1（恶化 5 倍）** |

**只提频率不改奖励会让 §1.1 失衡从 1:1 恶化到 5:1**。同理 `collision_reward=−10` 固定值在 5 Hz 下相对 per-step 累计更微不足道。**必须把所有 per-step 奖励除以新频率（恢复 per-second 语义），并把固定量级 collision 按比例上调**。

**环境参数可执行配置**（六配置同步，原子提交）：

```yaml
highway_config:
  controlled_vehicles: 3      # §A: ≥3 才激活冲突先验
  policy_frequency: 5         # Critical: 反应窗口 ×5
  simulation_frequency: 15    # 保持
  high_speed_reward: 0.1      # 0.5/5≈0.1，恢复 per-second 语义
  collision_reward: -25        # 5 Hz 下相对 per-step 累计补强
  arrived_reward: 10           # 使到达明确优于超时
  destination: null            # 见 §1.3
  ...
```

其他联动：`running_steps` 是 decision-step 数（5 Hz 下需 ~5M 才等价物理经验）；`buffer_size:2048` 在 5 Hz 下 episode 从 20→100 步，rolllow episode 数下降，建议提到 8192~10240。

### 1.3 目的地路由（High — 实测确认）

配置未覆盖 `destination`，继承 highway-env 单 agent 默认 `"o1"`（`intersection_env.py:43`；`MultiAgentIntersectionEnv.default_config:372-393` 未覆盖）。受控车放置（`intersection_env.py:285-299`）：两车目的地都是 o1（西出口）。实测路由：

| 配置 | agent 0 路由 | agent 1 路由 |
|---|---|---|
| `destination: "o1"`（当前默认） | `o0→ir0→il1→o1`（3 段，直行） | `o1→ir1→il0→o0→ir0→il1→o1`（**6 段，绕行**） |
| `destination: null`（每车随机） | `o0→ir0→il1→o1`（3 段） | `o1→ir1→il3→o3`（**3 段，直行**） |

两个后果：
- **agent 1 路由系统性 2× 长**（绕行 o0 再回 o1），可达性与 policy 无关地更难 → `arrival_rate = all(arrived)`（`highway_metrics.py:12`）被路由难度偏置。
- 两车**汇聚到同一西出口**，产生**第二个冲突点**，而 conflict prior 只建模单点 C=(0,0)（`docs/ca_commappo_design.md:22`）。`destination: null` 时两车出口分离，结构上才是"单中心冲突点"，匹配 conflict prior 假设。

**建议**：所有配置 `highway_config` 加 `destination: null`。一行修复，但改变所有 baseline 可比基准，需六配置同步修改并重训。

### 1.4 其余环境项

| 严重度 | 问题 | 证据 |
|---|---|---|
| Medium | 通信图**只含受控车**，不含 guaranteed challenger 与 IDM NPC——而它们才是主要碰撞源。`b_ij` 只能协调受控车汇聚，无法对真实威胁选边。 | `ca_commappo_agents.py:147-148`；`intersection_env.py:276-283` |
| Medium | `arrival_rate = all(arrived) and not collision`：要求**全部**到达，高车数（6/8/10）下趋近 0，best-model 排序键退化为噪声 `collision_rate`。 | `highway_metrics.py:12`；`aggregate_results.py:42-47` |
| Low | `env_seed:1` 跨 5 算法 seed 固定 → 每 seed 首 episode 初始态相同（仅 RNG 演化分叉）。`--mode test` 复用训练 env（`env_seed=1`），`--mode benchmark` 才 `seed+10000` → 最终 100-episode 验证有 ~16% episode 跨 seed 完全相关，低估方差。 | `train_highway_intersection.py:247-253,304-306`；`configs:16` |

### 1.5 环境建模正面结论

- `highway_config` 六配置逐字节一致（横向公平）。
- `absolute:false` 下 ego 行仍为绝对量（设计文档 §2 已核对 highway-env 源码确认），CA-ComMAPPO 计算 `ttc_i/intent_i` 的前提成立。
- `target_speeds:[0,4.5,9]` + `Discrete(3)`，reset 即最高挡（speed=10、index=2、target=9），动作空间合理。
- guaranteed challenger（8 m/s、speed_deviation=0）保证存在一辆冲突车，对 conflict prior 可训性是正面设计。

---

## 2. 算法选择与超参数

### 2.1 算法选择结论（合理）

MAPPO 底座、`Categorical_MAAC_Policy` 离散策略、PI 编码器、CTDE 架构**都适配**此任务。`use_parameter_sharing:true` 下 `MAAC_Policy.forward` **确实一次性传全 `(n_env·n_agents)` batch**（`xuance/.../categorical_marl.py:150-155`，已验证非 bug），通信集成正确。问题集中在"先验的参数化健壮性"与"车数配置"。

### 2.2 超参数问题

| 严重度 | 问题 | 证据/排查 |
|---|---|---|
| **Critical** | 见 §A：n=2 下 `λ/W_q/W_k` 死参数、消融退化。 | `conflict_aware_comm.py:118-142` |
| **High** | `λ` 是**无约束** `nn.Parameter`（初值 1.0，`weight_decay:0`）。在泛化 n≥3（唯一梯度来源）下 PPO 可把 `λ` 推负 → `−λ·|Δttc|` 变正 → **吸引 ttc 最不相似车对**（先验倒置）。设计文档 §3:48 明确"不做 sigmoid 限幅"但**未记录符号翻转风险**。建议 `λ = softplus(raw)` 或 `clamp(min=0)`。 | `conflict_aware_comm.py:84,109`；`docs/ca_commappo_design.md:44-48` |
| Medium | `μ=10`（固定直觉值）vs `λ`（可学）相对量纲失控：`recede_bias∈[0,−20]`、`ttc_bias∈[0,−50·λ]`、学到 `q·k~O(√128)≈11`。三项各自漂移，无 sweep。建议扫 `recede_penalty∈{2,5,10,20}` 并固定 `λ_init` 做敏感性。 | `conflict_aware_comm.py:66,77,85,109-112` |
| Medium | `alive_ally` 恒全 1 → 不活跃 agent 陈旧消息仍进活跃 agent 的 `c_i`（n≥3 下污染）。设计文档 §5:77 称"无害（因动作被丢弃）"，但**丢弃的是动作不是消息**。 | `ca_commappo_agents.py:147`；`docs/ca_commappo_design.md:77` |
| Low | `intent_i` 代码为**硬二值** `(approach>0).float()`，文档 §3:35 规定 `sigmoid(approach_i)∈(0,1)`。边界附近一步跳变，`R_i` 退化为 3 级离散。**注意**：文档 §3:47 自身又把 recede 描述为 3 级——文档自相矛盾，需定一份口径并回改文档。 | `ca_commappo_agents.py:115,120`；`docs/ca_commappo_design.md:35,47` |
| Low（澄清） | `target_kl=0.25` **是惰性参数**——clip 版 `MAPPO_Learner.update` 从不读它（只 `clip_range=0.2` 生效）。不引发失稳，但配置易误导读者以为有 KL trust region。 | `configs:56`；`mappo_learner.py:92` |
| Low（澄清） | `gain:0.01` **不生效**——`mlp_block` 调 `orthogonal_` 时未传 gain，实际用默认 gain=1.0。对训练无偏（全算法同），仅文档误导。 | `configs:31`；`xuance/torch/utils/layers.py` |
| Low | `n_epochs=2 × n_minibatch=2` / `buffer=2048` → 每 rollout 仅 4 次梯度更新，update-to-data 偏低，1M 步预算下可能欠拟合，压窄算法间差异。建议适度上调 `n_epochs` 或 `n_minibatch`。 | `configs:49-50`；`on_policy_marl.py:69-72` |

---

## 3. 训练稳定性与收敛性

| 严重度 | 问题 | 证据 |
|---|---|---|
| **Critical** | 见 §A：`λ` 在主配置下死锁于 1.0，梯度恒 0；n=2 softmax 饱和使 `d(c)/d(λ)=0`。 | `conflict_aware_comm.py:138-142` |
| Medium | 消息携带**原始未归一化 ttc ∈ [0,50]**（设计文档 §3:50 规定归一化到 [0,1]）。`ttc` 量级 ~50 而 `z` 特征 O(1)，`W_q/W_k/W_v`（默认 init ~0.088，且 `ConflictAwareComm` **从未接收 config 的 `initialize/gain`**）被 ttc 列主导；同时 `b_ij=−λ·|Δttc|` 可达 `−50·λ`，softmax 近 one-hot → `d(α)/d(scores)≈0` → **λ 梯度再次消失**（即便 n≥3）。建议按文档 `ttc/(ttc+1)` 归一化。 | `ca_commappo_agents.py:114-119,143-145`；`conflict_aware_comm.py:81-83`；`docs:50` |
| Medium | `λ` 无界（n≥3 下真正训练时）：可负（倒置先验）、可大（saturation 成硬路由）。见 §2.2。 | `conflict_aware_comm.py:84` |
| Low | 中心化 critic `PermutationInvariantAttention` 对全局 state 做**逐 agent 视图 mean-pool**（`representations.py:146`）：`V([s0,s1])=V([s1,s0])`，无法表达"哪辆车在哪个状态"的联合交互项。MAPPO 的 `V(完整联合态)` 前提被削弱。但这是**文档化的刻意选择**（支持变车数泛化），且被 value_norm/advnorm 缓解，故 Low。 | `representations.py:146`；`ca_commappo_agents.py:231-242` |

### 3.1 稳定性正面结论（已验证非问题）

- masked-softmax 用 `finfo.min` + `any_valid` 守卫，**无 NaN 路径**（`conflict_aware_comm.py:123,130-141`，空行 → 置零后 softmax）。
- `normalize_reward:false` 的奖励尺度失衡被 `use_value_norm` + `use_advnorm` + `use_huber(delta=10)` 缓解，TD 误差 O(1)，huber 不饱和。

---

## 4. 评估指标与基准

| 严重度 | 问题 | 证据 |
|---|---|---|
| **Critical** | 见 §A：消融 (c)/(d) 在主配置 n=2 下数值等价，主消融失效。 | `...-no-prior.yaml:25`；`conflict_aware_comm.py:108-116` |
| **High** | **无"先验有效 + 通信 baseline 可比"的设置**：主比较 n=2（先验失效），泛化 6/8/10 排除 CommNet/IC3Net/MLP（维度固定）。"冲突感知 > 平均通信"这一核心声明**无处可证**。 | `runbook §4:80` |
| **High** | `best_model.pth` 在 **5-episode** benchmark eval 上选（`arrival_rate/collision_rate` 量化到 6 级 `{0,0.2,...,1.0}`），常选出 5-episode 侥幸点；后续 100-episode test 只是**重测一个已错选的模型**，系统性偏置。全算法对称故不制造假阳性，但压低所有方法数字、放大噪声。 | `train_highway_intersection.py:314-320,345-357`；所有 `configs:77` `test_episode:5` |
| **High** | `aggregate_results` 用 `best_summary=max(summaries,...)`（best-of-5-episode 检查点）且 glob 只匹配 `*benchmark_seed_*` 目录 → **从不使用 runbook 说的 100-episode 最终数字**。聚合表 = "每 seed 最侥幸检查点"的 mean±std。 | `aggregate_results.py:50-55,119`；`runbook §8:134` |
| **High** | `lr_decay_steps:5000` **只在 CommNet/IC3Net 配置出现**（MAPPO/IPPO/Attention/CA 无）。虽 `use_linear_lr_decay:false` 下对 CommNet 是 `LinearLR(total_iters=5000)` 仍被构造 → **CommNet/IC3Net 的 LR 实际会衰减，其余不变**，跨算法 LR 行为不一致。 | `configs/commnet:61-62`、`configs/ic3net:61-62`（其余无） |
| Medium | CommNet/IC3Net 用 `Basic_RNN + use_rnn:true`，CA-ComMAPPO 用 `PermutationInvariantAttention + use_rnn:false` → "CA vs CommNet"**混淆通信聚合类型与 backbone**（RNN vs PI-MLP）。但**不影响核心声明**：Attention-MAPPO（同 PI backbone、无通信）是干净的 PI 对照。需在论文中明确"CA-ComMAPPO 的对照是 Attention-MAPPO（消融去通信），而非 CommNet（不同 backbone）"。 | `configs/commnet:20,23` vs `configs/ca_commappo:20,23` |
| Medium | `arrival_rate` 全到达要求（见 §1.4），高车数退化。 | `highway_metrics.py:12` |
| Medium（补充） | **无显著性检验/置信区间/功效分析**：`aggregate_results` 只产 `mean ± pstdev`，5 seed 无配对检验。"CA > baseline" 无法统计支撑。建议加 paired bootstrap/permutation test，或至少报 sample std。 | `aggregate_results.py:86-89`；`runbook §3:48` |
| Low | `order_perturbation.py`、`attention_export.py`、`plot_results.py` **均不存在**（runbook §5/§7/§8 引用为已有）→ 顺序不变性证据、可解释性图、绘图全部无法产出；gen4/evalN 配置未生成。 | Glob `**/{order_perturbation,attention_export,plot_results}.py` = No files |
| Low | `pstdev`（除以 N）而非样本 std（除以 N−1），5 seed 下报 ~0.894× 样本 std，低估不确定性。 | `aggregate_results.py:88` |
| Low | 缺失/空 jsonl 被 `if not records: continue` **静默跳过**，`seeds` 列只报存活数，无预期校验 → 3/5 seed 发散也能悄无声息出表。 | `aggregate_results.py:75-76,82` |

---

## 5. 工程实现细节

| 严重度 | 问题 | 证据 |
|---|---|---|
| Medium | 消息 ttc 原始 [0,50] 未归一化（见 §3）。 | `ca_commappo_agents.py:143` |
| Medium | `alive_ally` 恒全 1（见 §2.2/§3）。 | `ca_commappo_agents.py:147` |
| Medium | **无端到端集成测试**：`ConflictAwareComm` 与 `CAComMAPPOActorRep` 仅单元测试（手造 obs），没有任何测试构造 `CA_ComMAPPO_Agents` 并跑一次 `action()`，设计文档 §8 验收要求未覆盖。`xuance` batch 布局若变动会静默打乱"谁聚合谁的消息"。 | `tests/test_ca_commappo_actor_rep.py`；`docs/ca_commappo_design.md §8` |
| Medium（文档一致性） | `q_i = W_q(m_send)`（用**完整消息**，含 ttc/intent），但文档 §4:61 规定 `q_i = W_q z_i`（只编码观测）。即 query 也携带自身冲突态，与 `b_ij` 双重计入。 | `conflict_aware_comm.py:81,103`；`docs:61` |
| Low（文档） | 代码 `dot/sqrt(comm_dim)` 缩放（`:107,85`），文档 §4:57 公式 `softmax(q^T k + b)` 无该因子 → 偏置项相对 dot 比文档描述强 ~√128 倍。 | `conflict_aware_comm.py:85,107`；`docs:57` |
| Low | `forward` 硬假设 batch = `(n_env·n_agents)` 并 `% n_agents==0`，无 `use_parameter_sharing` 守卫。当前所有配置都 `true` 故不触发；设 `false` 会静默错位。 | `ca_commappo_agents.py:127-133` |
| Low | agent 恰在 C=(0,0) 时 `d≈0 → dir_to_c≈0 → approach≈0 → not_approaching`（取 `≤0`）→ 该最高风险点被报为 `ttc=TTC_MAX, intent=0`（零冲突）。设计文档未覆盖此退化。 | `ca_commappo_agents.py:110-115` |

### 5.1 工程正面结论（已验证正确）

- masked-softmax 数值安全。
- `MAAC_Policy` 在 `use_parameter_sharing:true` 下正确传全 batch（candidate 经对抗验证**否决**为非 bug）。
- `comm_runtime_patch` 不为 CA-ComMAPPO 所需（只补 CommNet/IC3Net）。
- `registration` 幂等（identity 检查）。
- 常量与 `b_ij/R_i` 公式与文档一致（`TTC_MAX=50, μ=10, λ_init=1, R_i=1−intent`，两项式）。

---

## 6. 被对抗验证否决的 4 项（透明列出）

1. **global_terminated 让非撞车 agent bootstrap=0 损坏 GAE** → 否决：任意碰撞即终止全回合是合作 MARL 标准做法，非缺陷。
2. **arrival_rate 全到达在高车数退化** → 部分否决：代码事实成立，但某下游失效机制表述不准（有效部分已在 §1.4/§4 保留）。
3. **无任何多 seed 结果，聚合只在合成数据上验证** → 否决：`runs/` 下已有 5 个真实 `eval_records.jsonl`（ca_commappo/commnet/...）。
4. **registration 非幂等** → 否决：有 identity 检查，幂等。

---

## 7. 优先级修复清单

### P0（开远端训练前必须，否则实验无意义）

1. 正式配置 `controlled_vehicles: ≥3`，smoke 与正式统一车数（§A）。
2. `policy_frequency: 5` + 奖励 per-second 归一化（`high_speed_reward:0.1`）+ `collision_reward` 量级上调 + `destination: null`（§1）。六配置同步。
3. `λ` 加 `softplus/clamp(min=0)` 防符号翻转（§2.2）。
4. ttc 消息按文档归一化到 [0,1]（§3）——否则即便 n≥3，λ 梯度仍因 softmax 饱和而消失。
5. 消融 (c)/(d) 在 ≥3 下重做；确认 `λ` 在泛化训练下确实更新（加 `λ` 值监控）。

### P1（强烈建议）

6. `best_model` 选择改用更大 eval episode（如 50）或滑动平均；`aggregate_results` 改用 100-episode test 记录而非 best-of-5-episode（§4）。
7. 补 `order_perturbation.py`/`attention_export.py`/`plot_results.py` 与 gen4/evalN 配置（否则论文卖点无法产出）。
8. `intent` 改 `sigmoid` 或回改文档定一份口径（§2.2）；`q_i` 改用 `z_i` 或回改文档（§5）。
9. `alive_ally` 注入真实 agent mask（§2.2/§3）；CommNet/IC3Net 的 `lr_decay_steps` 一致化或记录为已知差异（§4）。
10. 加配对显著性检验 + 样本 std + 缺失 seed 告警（§4）。

### P2（打磨）

11. `arrival_rate` 改 per-agent 分数作主排序键。
12. `env_seed` 随 `--seed` 派生。
13. 补端到端集成测试。
14. 文档补 `/sqrt(comm_dim)` 与符号翻转风险说明。

---

## 附录：评审元数据

- 评审方法：7 维度 reviewer → 逐条对抗验证 → 完整性 critic。
- 规模：63 agent，2,668,072 tokens，833 次工具调用。
- 结果：51 条确认（6 Critical / 6 High / 18 Medium / 16 Low / 5 not-a-bug），4 条否决，6 条补充。
- 验证：关键结论均经独立读码（含被否决项）与实跑 highway-env 环境验证。
- 评审时点分支 HEAD：见提交 `docs: add experiment design review report`。

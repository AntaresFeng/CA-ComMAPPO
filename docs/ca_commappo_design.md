# CA-ComMAPPO 算法设计

本文只定义 CA-ComMAPPO 的算法语义、当前代码映射和已知边界。共享任务与训练
超参由 [`mappo_baseline.md`](mappo_baseline.md) 冻结，实验操作见
[`experiments_runbook.md`](experiments_runbook.md)。

## 研究定位

CA-ComMAPPO 在 MAPPO 上增加两层结构：

1. `PermutationInvariantAttention` 编码每个 agent 的 ego 与邻车观测，降低车辆列表
   排序的影响。
2. `ConflictAwareComm` 在受控 agent 之间做带运动学冲突先验的注意力通信。

actor 依据本地编码 `z_i` 和通信上下文 `c_i` 选动作，critic 使用全局状态，保持
centralized training / decentralized execution。该通信只覆盖受控 agent，不覆盖 IDM
背景车，因此不能声称建模了场景中的全部冲突参与者。

## 当前数学定义

每个 agent 从观测第 0 行的 ego 状态构造路口中心 `C=(0,0)` 的代理量：

```text
d_i        = ||pos_i||
approach_i = heading_i · (-pos_i / (d_i + eps))
tau_i      = clamp(d_i / (speed_i + eps), 0, TTC_MAX)
ttc_i      = tau_i / (tau_i + 1)
intent_i   = 1[approach_i > 0]
```

不朝向中心的 agent 令 `tau_i=TTC_MAX`，再按同一公式缩放。消息和冲突偏置只使用
`ttc_i`，因此运动学头有界于 `[0,1)`，不会把原始 `[0,50]` 尺度直接送入线性层或
softmax。

```text
m_i = concat(z_i, ttc_i, intent_i)
```

通信注意力与聚合为：

```text
b_ij     = -lambda * |ttc_i - ttc_j| - mu * ((1-intent_i) + (1-intent_j))
score_ij = (W_q m_i)^T (W_k m_j) / sqrt(d_comm) + b_ij
alpha_ij = masked_softmax_j(score_ij), j != i
c_i      = sum_j alpha_ij * W_v m_j
```

有效 `lambda=softplus(raw_lambda)`，初始有效值由 `conflict_lambda_init=1.0` 决定，
因此训练中不会变负并反转 TTC 相似性。`mu` 对背离车辆施加固定惩罚；
`use_conflict_prior=false` 时令 `b_ij=0`，对应“有
attention、无冲突先验”的消融。masked softmax 排除自身和失活发送方，在没有有效
发送方时返回零上下文。

上述公式描述的是当前代码，而不是理想化版本：query 和 key 都读取完整消息，intent
是硬二值。

通信合同随 checkpoint 持久化为版本 `2`。版本 2 表示有界 TTC、非负 lambda 和真实
agent mask 通信；缺少
版本字段、包含旧 `comm.lam` 参数或版本不匹配的 CA-ComMAPPO checkpoint 会被明确
拒绝。旧模型只允许保留为历史证据，不能迁移或用于正式对比。

## 实现映射

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `PermutationInvariantAttention` | `ca_commappo/algorithms/representations.py` | ego/邻车共享编码与 masked attention pooling |
| `ConflictAwareComm` | `ca_commappo/algorithms/conflict_aware_comm.py` | 冲突先验、通信 attention、消息聚合 |
| `CAComMAPPOActorRep` | `ca_commappo/algorithms/ca_commappo_agents.py` | 计算 TTC/intent，将 `z_i` 与 `c_i` 拼接 |
| `CA_ComMAPPO_Policy` | 同上 | 复用 categorical MAAC policy |
| `CA_ComMAPPO_Learner` | 同上 | 复用 MAPPO clipped PPO learner |
| `CA_ComMAPPO_Agents` | 同上 | 构造 actor/critic representation 和 policy |
| `register_components()` | `ca_commappo/algorithms/registration.py` | 运行时注册 XuanCe 组件 |

通信折叠在 actor representation 中。自定义 policy 用同步 context 把 rollout 或
minibatch 的 agent mask 传入 representation；自定义 learner 只建立 mask context 后调用
原 MAPPO update，不复制 PPO loss。critic 使用独立的 permutation-invariant
representation，不接收通信上下文。

actor representation 还提供异常安全的只读 trace context。仅在 `ca-trace` 评估模式
显式启用时，它复制导出 active mask、scaled TTC、intent、effective lambda 和
`alpha_ij`；正常训练与推理不保存 trace，开启 trace 也不改变 policy 输出。trace
evaluator 另外记录动作、normalized ego 轨迹、reward 和结局，并只在 sender/receiver
都有效的时刻计算平均 attention heatmap。

## 观测与可比性前提

- 当前观测特征为 `presence,x,y,vx,vy,cos_h,sin_h`，第 0 行是 ego。
- `absolute: false` 时 ego 行仍表示归一化世界坐标，其他车辆相对 ego；TTC 代理依赖
  这一 highway-env 1.11 行为。
- 冲突点只取路口中心，不能表达出口汇流、车道级交叉点或真实 route priority。
- 正式配置使用 4 个受控 agent。至少 3 个 agent 才能让每个接收方在两个候选发送方
  之间产生有信息量的 softmax；2-agent 设置下冲突先验无法改变唯一邻居的权重。
- 完整方法、no-prior、Attention-MAPPO 和 MAPPO 必须复用同一环境、训练预算、seed
  block 与 final-test protocol。

## 已知边界

车辆数量变化会同时改变 agent identity 和 critic 输入形状；当前 checkpoint 没有
4→6/8/10 辆的兼容协议。因此正式结果只支持冻结的四车任务，不支持规模泛化声明。
规模泛化已明确排除在本论文实验范围外，所以它不是现有四车正式结果的阻断项；若未来
重新纳入，属于算法与实验合同变更，必须使用新提交、新 checkpoint 和新对照矩阵。

## 论文可支持的声明

当前正式证据支持以下有限结论；数值和统计边界见
`results/core_matrix_199ead5_analysis_report.md`：

- 顺序扰动实验直接支持 PI encoder 的排列稳定性。
- Attention-MAPPO 对比 MAPPO 支持 encoder 结构贡献。
- full 对比 no-prior 给出冲突先验的方向一致正效应，但仅有三个训练 seed。
- full 相对 Attention-MAPPO 没有形成“显式通信全面更优”的明确证据。
- full 对比 CommNet / IC3Net 只能作为系统级通信基线比较；其 RNN backbone 不同，
  不能单独归因于聚合方式。
- attention heatmap 只提供可解释性案例，不能替代成功率、安全性和显著性检验。

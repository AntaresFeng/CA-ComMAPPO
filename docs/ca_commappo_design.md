# CA-ComMAPPO 设计文档（冲突感知 + 顺序无关 + 显式通信）

> 本文档在写代码之前冻结 CA-ComMAPPO 的数学定义与实现口径（论文 `docs/thesis-planning.md` §3、执行提示词阶段 3）。任何实现偏离都应先回改本文档。

## 1. 目标与定位

CA-ComMAPPO = **C**onflict-**A**ware **Com**munication **MAPPO**。以 MAPPO 为训练底座：

1. **Permutation-invariant social encoder**（复用阶段 1 的 `PermutationInvariantAttention`）：处理多车观测，对车辆顺序不敏感。
2. **Conflict-aware communication module**（`ConflictAwareComm`，新建）：把 CommNet 的「全连接平均聚合」换成「带冲突先验的注意力聚合」。
3. **MAPPO centralized critic**：复用 MAPPO 的 clipped PPO loss 与中心化 critic（先复用，不重写）。

执行为去中心化（CTDE）：actor 只用自身观测 `z_i` 与接收消息聚合 `c_i`；critic 用全局状态 `s`。

## 2. 观测前提（关键，决定 `b_ij` 可计算性）

配置沿用 MAPPO（`absolute: false`，`features: [presence, x, y, vx, vy, cos_h, sin_h]`，`vehicles_count: 15`）。highway-env `KinematicObservation` 的契约（`observation.py:233-275`、`kinematics.py` `to_dict`）：

- **ego（受控车，观测 index 0）行是绝对量**：`to_dict()` 不传 origin，故 ego 的 `x,y,vx,vy` 是**世界系绝对**位置/速度；只有其它车（index 1..14）经 `to_dict(origin=ego)` 变为相对 ego。
- 因此**每个 agent 能从自己的 ego 行观测到自身绝对位置与速度**——`ttc_i = dist(i, C)/speed_i` 可计算。这正是本设计成立的前提（此前曾误以为 ego 行也是相对量，经核对源码已澄清）。
- 观测默认 `normalize: true`，ego 的 `x,y,vx,vy` 被 `lmap` 到 `[-1,1]`：`norm_x = x/(5·MAX_SPEED)`、`norm_y = y/(DEFAULT_WIDTH·n_side_lanes)`、`norm_vx = vx/(2·MAX_SPEED)`、`norm_vy = vy/(2·MAX_SPEED)`。
- **冲突点 C = (0, 0)**：4-way 交叉路口对称于原点（`intersection_env.py _make_road` 四角 90° 旋转），路口中心即世界系原点；`x=0→norm_x=0`，故 C 在归一化空间仍是 (0,0)。

## 3. 冲突先验 `b_ij` 公式（冻结）

**口径选择**：在**归一化观测空间**计算 `ttc` 代理量。理由：`b_ij = -λ·|ttc_i - ttc_j|` 作为注意力偏置只需 agent 间的**相对**排序；所有 agent 共享同一归一化，故代理量与真实 ttc 仅差一个常数（被可学 `λ` 吸收）。`x/y` 归一化范围不对称是次要近似，列为局限。

每个 agent `i` 从自身 ego 行（归一化）取：

- 位置 `pos_i = (norm_x_i, norm_y_i)`，到冲突点距离 `d_i = ||pos_i||`。
- 速度 `speed_i = sqrt(norm_vx_i^2 + norm_vy_i^2)`。
- 航向 `h_i = (cos_h_i, sin_h_i)`。
- 朝向冲突点的接近度（投影）：`approach_i = -h_i · (pos_i / max(d_i, eps))`。`approach_i > 0` 表示朝 C 行驶（有冲突意图）；`≤ 0` 表示背离/静止。
- **TTC 代理**：`ttc_i = d_i / max(speed_i, eps)`；若 `approach_i ≤ 0`（不朝 C 行驶）则置 `ttc_i = TTC_MAX`（不构成冲突）。`TTC_MAX` 为超参（默认 `50.0`，归一化空间）。
- **intent_i = sigmoid(approach_i)** ∈ (0,1)：朝 C 的意图强度。

**冲突先验**（`i` 对 `j` 的消息注意力偏置）：

```
b_ij = -λ · |ttc_i - ttc_j|  -  μ · (R_i + R_j)
R_i  = 1 - intent_i          # recede 指标：背离/静止(intent≈0)→R≈1；朝 C 行驶(intent≈1)→R≈0
```

- `λ` 为**可学标量**（`nn.Parameter`，初值 `1.0`），吸收归一化常数与量纲。
- `μ` 为**固定超参**（默认 `10.0`），对涉及背离 agent 的通信对施加强负偏置。
- 第一项 `-λ·|ttc_i - ttc_j|`：到达冲突点时间差越小，偏置越大（负得越少）→ 两车同时到达 C（典型冲突）注意力高。
- 第二项 `-μ·(R_i + R_j)`：**关键修正**——纯 ttc 相似度会让两辆都背离的车（`ttc=TTC_MAX`，`|diff|=0`）得到 `b_ij≈0`（误判为高冲突）。引入 `R` 惩罚背离对：两辆都背离 → `b_ij = -2μ`（强负，低注意力）；一辆朝向一辆背离 → `-μ`；两辆都朝向 → `0`（由 ttc 项决定）。这是 TDD 暴露后补的设计修正。
- 不做 sigmoid 限幅（softmax 已限幅；`λ` 可学、`μ` 固定即可）。若数值不稳，再改为 `tanh` 限幅。

**消息携带**：agent `j` 发送 `m_j = f_m(z_j) ⊕ [ttc_j_scaled, intent_j]`，即在编码 `z_j` 后接一个小固定维度头（2 维 kinematic head：`ttc_j` 归一化到 `[0,1]` via `ttc_j/(ttc_j+1)` 或 `min(ttc_j,TTC_MAX)/TTC_MAX`，加 `intent_j`）。`ConflictAwareComm` 从 `m_j` 末尾切片读出 `ttc_j`/`intent_j` 算 `b_ij`；`ttc_i`/`intent_i` 由 `i` 自身计算（在 policy forward 内，不依赖 `j` 的消息）。

**CTDE 合规**：`b_ij` 只用 `i` 自身 ego 行（去中心化可观测）与 `j` 发送的消息（`m_j` 携带 `ttc_j`/`intent_j`）——不读全局状态、不读 env 内部。执行时每车只用自身观测 + 接收消息。

## 4. 通信聚合公式

```
α_ij = softmax_{j≠i}( q_i^T k_j + b_ij )       # 软冲突先验注意力（非硬离散边）
c_i  = Σ_{j≠i} α_ij · W_v m_j
```

- `q_i = W_q z_i`、`k_j = W_k m_j`、`v_j = W_v m_j`（`m_j` 含 kinematic head，故 `k_j` 也含冲突信息）。
- 对 `alive_ally` 掩码（非活跃 agent 不参与）、排除自身（`j≠i`）。
- **实现口径**：论文称「conflict-aware communication graph」，但用**软注意力** `softmax(q^T k + b)` 实现，而非硬离散边。命名保留「graph」，实现为 soft attention，本文档与代码一致。

## 5. 模块结构（实现口径）

| 组件 | 基类 | 职责 |
|---|---|---|
| `ConflictAwareComm(Module)` | — | 模仿 `CommNet.forward` 的 drop-in：`forward(m_send, ttc_i, alive_ally) -> c_i`，把 mean 聚合换成 `c_i = Σ_{j≠i} α_ij W_v m_j`，`α_ij = softmax(q_i^T k_j + b_ij)`；从 `m_j` 末尾切片读 `ttc_j/intent_j` 算 `b_ij`。参数共享时按 `(n_env, n_agents)` reshape。 |
| `CAComMAPPOActorRep(Module)` | `Module`（签名对齐 `Basic_MLP`） | 自定义 actor representation：`forward` 内做 `z_i = PI_encoder(o_i)` → 算 `ttc_i/intent_i`（从 ego 行）→ `m_i = head([z_i, ttc_i, intent_i])` → `c_i = ConflictAwareComm(m, ttc_i, alive_ally=全1)` → 输出 `{'state': [z_i, c_i]}`，`output_shapes={'state': (H+comm_dim,)}`。**关键实现选择**：把 comm 折进 actor representation（而非 policy forward），使 `MAAC_Policy.forward` 不改即可工作——actor 输入维 `dim_actor_in = output_shapes['state'][0] + n_agents` 自动含 `comm_dim`，避免重写 actor 维构造。critic 用 `PermutationInvariantAttention`（无 comm）。 |
| `CA_ComMAPPO_Policy` | `MAAC_Policy`（`Categorical_MAAC_Policy`） | 薄子类（主要为 config 命名 `policy: "CA_ComMAPPO_Policy"`；forward 继承 MAAC，调 actor representation 即完成 enc+comm+actor）。 |
| `CA_ComMAPPO_Learner` | `MAPPO_Learner` | 薄子类，直接复用 MAPPO clipped PPO loss（先不把通信纳入 critic）。 |
| `CA_ComMAPPO_Agents` | `MAPPO_Agents` | 重写 `_build_policy`：**手动构造** `A_representation = CAComMAPPOActorRep`（显式传 `n_agents`，因 `_build_representation` 不传 n_agents；仿 CommNet `_build_communicator` 手动建通信器）+ `C_representation = PermutationInvariantAttention`（经 `_build_representation("PermutationInvariantAttention", space_critic_in, ...)`）+ `policy = CA_ComMAPPO_Policy`。绕开 `MAPPO_Agents._build_policy` 的 policy 字符串白名单。`use_rnn: false`（与 MAPPO 一致，非 CommNet 的 RNN 路径）。 |

**实现选择说明（comm 折进 representation）**：论文 §3 的数学（`z_i=enc`、`m_i=head`、`c_i=ΣαW_v m_j`、`π_i(a|z_i,c_i)`）与把 comm 放 policy forward 等价；折进 representation 的好处是 `MAAC_Policy.forward` 与 `MAPPO_Learner` 完全复用（actor 维自动匹配、buffer/动作路径不变），把全部新逻辑隔离在 `ConflictAwareComm` + `CAComMAPPOActorRep` 内，便于 TDD 与 smoke 验证。命名上仍保留 `CA_ComMAPPO_Policy`/`CA_ComMAPPO_Learner` 以对齐论文与 config 字符串。

**`alive_ally` 处理**：CA-ComMAPPO 继承 `MAPPO_Agents.action`（不取 `info`）。`ConflictAwareComm` 内部 `alive_ally` 取全 1（所有受控 agent 活跃）；非活跃 agent 的 IDLE 动作由 adapter 处理，policy 输出被丢弃，故全 1 无害（与 CommNet 训练期一致）。如需精细化，未来可扩展 `action` 注入真实 `agent_mask`（同 `comm_runtime_patch` 思路）。

## 6. 运行时注册

`register_components()`（`ca_commappo/algorithms/registration.py`）追加：
- `REGISTRY_Policy["CA_ComMAPPO_Policy"] = CA_ComMAPPO_Policy`
- `REGISTRY_Learners["CA_ComMAPPO_Learner"] = CA_ComMAPPO_Learner`
- `REGISTRY_Representation["PermutationInvariantAttention"]`（阶段 1 已注册，幂等）

`AGENT_REGISTRY["CA_COMMAPPO"] = CA_ComMAPPO_Agents`（`train_highway_intersection.py`）。`run()` 在构造 agent 前调用 `register_components()`（阶段 1 已接入）。

## 7. 配置（`configs/ca_commappo/`）

`agent: "CA_COMMAPPO"`、`learner: "CA_ComMAPPO_Learner"`、`policy: "CA_ComMAPPO_Policy"`、`representation: "PermutationInvariantAttention"`、`use_rnn: false`、`output_dir: "runs/ca_commappo_highway/"`。其余超参与 MAPPO 对齐（可比性）。smoke 同结构小参数。视频前缀 `ca_commappo_test`（阶段 3.6 参数化）。

## 8. 验收测试（TDD，先写）

- `ConflictAwareComm`：attention 权重对 `alive_ally` mask 正确归一化（被 mask 的 j 权重为 0，其余和为 1）；排除自身（`j≠i`）；`b_ij` 对 `|ttc_i - ttc_j|` 单调递减；forward 形状正确。
- 冲突场景：构造两车朝 C 对向接近（`approach_i, approach_j > 0` 且 `ttc_i ≈ ttc_j`）→ 其 `α_ij` 大于两车背离场景（`approach ≤ 0` → `ttc=TTC_MAX` → `b_ij` 更负 → `α` 小）。
- `CA_ComMAPPO_Agents`：在 smoke 配置下完成构造与一步前向（action 输出形状对）。

## 9. 局限与未来工作（写入论文）

- **kinematic proxy**：`b_ij` 用归一化空间 TTC 代理，非真实 route/优先级；无车道级冲突点集（单点 C 简化）；`x/y` 归一化范围不对称是次要近似。
- **alive_ally**：当前全 1，未利用真实 inactive 掩码（adapter 已处理 IDLE，故无害）。
- **critic**：暂复用 MAPPO critic，未把通信纳入中心化 critic（`V({z_i, m_i, a_i})`）——列为未来工作。
- 未来可扩展：各入口车道多冲突点集、真实 route/意图观测（需 `observe_intentions`）、硬冲突图阈值化。

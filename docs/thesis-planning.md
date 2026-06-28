我的建议是：**主线不要从 MADDPG 起步，而是从 MAPPO 起步；创新算法构建在“Permutation-invariant Attention-MAPPO + 显式通信模块”之上。**

原因很直接：`highway-env` 的 `intersection-v0` 本身是无信号路口密集交通任务，默认动作是 `DiscreteMetaAction`，多智能体版本也可以通过 `controlled_vehicles`、`MultiAgentAction`、`MultiAgentObservation` 配置出来；这类离散决策、同质车辆、合作通过路口的任务，更适合先用 MAPPO/IPPO 体系做稳定基线。([highway-env.farama.org][1]) MAPPO 论文也说明，PPO 系列在合作多智能体任务中是很强的实用基线。([arXiv][2])

## 1. 基线怎么选

我建议分成三组，而不是只选一个“普通 MAPPO”。

**第一组：必要 sanity baseline**

| Baseline                       | 作用                              |
| ------------------------------ | ------------------------------- |
| Random / Rule-based / IDM-like | 检查环境、奖励、碰撞统计是否正常                |
| IPPO                           | 每车独立策略，无 centralized critic，无通信 |
| MAPPO                          | CTDE，无显式通信，是你的主算法底座             |

**第二组：强结构基线**

| Baseline                             | 作用                          |
| ------------------------------------ | --------------------------- |
| MAPPO-MLP                            | 普通 MAPPO，作为最低 MARL baseline |
| MAPPO + DeepSets / Attention Encoder | 解决车辆顺序敏感问题，但不引入通信           |
| MAPPO + Social Attention             | 针对路口多车交互，是非常重要的强基线          |

这一组很关键。`highway-env` 文档明确提到，MLP 对车辆列表顺序敏感，同一场景如果车辆排序变化，模型会把它当成新状态；官方建议使用 permutation-invariant 架构，或者换成不依赖排序的观测表示。([highway-env.farama.org][3]) Social Attention 也正是为“车辆数量变化、顺序无关、交通参与者交互建模”设计的。([Eleurent][4])

**第三组：通信 MARL baseline**

| Baseline |      是否建议做 | 说明                                                 |
| -------- | ---------: | -------------------------------------------------- |
| CommNet  |         建议 | 经典连续通信基线，适合作为“平均通信/全局广播”对照                         |
| IC3Net   |         建议 | CommNet 的门控通信扩展，可比较“是否学习何时通信”                      |
| MAAC     |         可选 | 注意力 critic，不一定是显式通信，但可以作为 attention-MARL 对照        |
| MADDPG   | 可选，不建议作为主线 | 更适合连续动作或混合竞争任务；你当前 `DiscreteMetaAction` 场景下不是最自然选择 |

CommNet 的核心是让多智能体通过可微连续消息一起学习通信协议，主要面向完全合作任务；IC3Net 进一步加入 gate，让智能体学习“何时通信”。([arXiv][5]) MADDPG 的价值在于 centralized critic、decentralized actor，并考虑其他智能体策略，适合 mixed cooperative-competitive 场景，但你的研究点1更偏“合作车辆通过路口 + 显式通信机制”，所以它不是最优主线。([arXiv][6]) MAAC 的 attention critic 可以动态选择关注哪些智能体，适合做 attention 对照，但它的重点是 critic 选择信息，不等同于车辆间通信。([arXiv][7])

## 2. 你的创新算法从什么基础构建

推荐主线：

**MAPPO → Attention-MAPPO → Conflict-aware Communication MAPPO**

也就是：

> 以 MAPPO 为训练框架，以 permutation-invariant attention encoder 处理路口多车观测，再加入面向无信号路口冲突关系的显式通信模块。

这样论文逻辑最清楚：

普通 MAPPO 解决多车合作；
Attention-MAPPO 解决“车辆顺序无关、多车交互建模”；
你的方法进一步解决“车辆之间应该交换什么信息、与谁通信、如何利用通信提高通过效率和安全性”。

## 3. 具体如何构建

可以把每辆受控车作为一个 agent。每个 agent 的输入包括自身状态、周围车辆状态、目标出口或行驶意图。不要直接把车辆列表 flatten 后送 MLP，而是先用共享 encoder 编码每辆邻车：

[
e_{ij} = \phi(o_i, o_j)
]

其中 (o_i) 是自车状态，(o_j) 是邻车状态，最好使用相对位置、相对速度、航向、距离冲突点时间等特征。然后用 attention 或 DeepSets 聚合：

[
z_i = \text{AttnPool}({e_{ij}})
]

这一步得到每个 agent 的局部交通理解，且对车辆顺序不敏感。

接着加入通信模块：

[
m_i = f_m(z_i, intent_i, ttc_i)
]

每辆车发送消息 (m_i)。接收端不是简单平均所有消息，而是做**冲突感知 attention**：

[
c_i = \sum_{j \neq i} \alpha_{ij} W_v m_j
]

其中 (\alpha_{ij}) 可以由以下信息决定：

[
\alpha_{ij} = \text{softmax}(q_i^\top k_j + b_{ij})
]

(b_{ij}) 可以设计成路口冲突先验，例如两车路径是否冲突、到达冲突点时间差、距离冲突区域远近、相对优先级等。这样你的创新点就不是“我也用了 attention”，而是：

**通信不是全连接平均通信，而是围绕无信号路口冲突关系的选择性通信。**

最终 actor：

[
\pi_i(a_i | z_i, c_i)
]

centralized critic：

[
V(s) \quad \text{或} \quad V({z_i, m_i, a_i}_{i=1}^N)
]

训练用 MAPPO 的 clipped PPO loss，执行时每辆车只使用自己的观测和接收到的消息，符合 CTDE 思路。

## 4. 你的算法可以怎么命名和包装

可以考虑类似：

**CA-ComMAPPO：Conflict-Aware Communication MAPPO**

核心模块：

1. **Permutation-invariant social encoder**：解决车辆顺序敏感问题。
2. **Conflict-aware communication graph**：根据路口冲突关系构建通信边。
3. **Attention message aggregation**：学习不同车辆消息的重要性。
4. **MAPPO centralized critic**：保证训练稳定。
5. **Ablation**：去掉通信、去掉冲突先验、去掉 attention、换成 CommNet 平均通信。

这个设计比“普通 MAPPO 加 attention”更像论文创新，因为它把通信机制和无信号路口场景强绑定。

## 5. 推荐实验顺序

先不要一上来做复杂现实通信。研究点1建议这样推进：

1. 跑通 `intersection-v0` 多智能体配置：`controlled_vehicles > 1`，`MultiAgentAction`，`MultiAgentObservation`。
2. 先做 IPPO、MAPPO-MLP。
3. 加 DeepSets / Attention Encoder，验证 permutation-invariant 结构是否提升。
4. 加 CommNet-MAPPO、IC3Net-MAPPO。
5. 加你的 CA-ComMAPPO。
6. 做泛化实验：训练 4 辆车，测试 6、8、10 辆车。
7. 做顺序扰动实验：随机打乱观测中车辆顺序，证明普通 MLP 不稳定，而你的方法稳定。
8. 做通信消融：无通信、全连接平均通信、attention 通信、conflict-aware attention 通信。

评价指标建议至少包括：

| 指标                                    | 含义        |
| ------------------------------------- | --------- |
| collision rate                        | 安全性       |
| success / pass rate                   | 是否成功通过路口  |
| average travel time                   | 通行效率      |
| waiting time                          | 路口等待代价    |
| throughput                            | 单位时间通过车辆数 |
| average reward                        | RL 通用指标   |
| generalization to vehicle number      | 多车扩展能力    |
| communication attention visualization | 可解释性      |

## 6. 最终推荐结论

你的基线组合可以写成：

> 本文首先采用 IPPO 和 MAPPO 作为无通信多智能体强化学习基线；考虑到 highway-env 中基于车辆列表的运动学观测存在顺序敏感问题，进一步引入 permutation-invariant attention encoder 构建 Attention-MAPPO 作为强结构基线；随后选取 CommNet 和 IC3Net 作为经典通信 MARL 基线；在此基础上，提出面向无信号路口冲突关系的 Conflict-aware Communication MAPPO，通过冲突图和注意力消息聚合实现理想通信条件下的协同驾驶决策。

一句话：**主算法底座选 MAPPO，强基线必须包含 Attention-MAPPO，通信基线选 CommNet/IC3Net，你的创新算法做“冲突感知 + 顺序无关 + 显式通信”的 MAPPO 扩展。**

[1]: https://highway-env.farama.org/environments/intersection/ "Intersection - HighwayEnv Documentation"
[2]: https://arxiv.org/abs/2103.01955 "[2103.01955] The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"
[3]: https://highway-env.farama.org/faq/ "Frequently Asked Questions - HighwayEnv Documentation"
[4]: https://eleurent.github.io/social-attention/ "Abstract | Social Attention for Autonomous Decision-Making in Dense Traffic"
[5]: https://arxiv.org/abs/1605.07736 "[1605.07736] Learning Multiagent Communication with Backpropagation"
[6]: https://arxiv.org/abs/1706.02275 "[1706.02275] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
[7]: https://arxiv.org/abs/1810.02912 "[1810.02912] Actor-Attention-Critic for Multi-Agent Reinforcement Learning"

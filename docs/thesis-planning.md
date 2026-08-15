# 论文实验主线

本文只回答“论文要比较什么、验证什么、还缺什么”。固定 MAPPO 任务合同见
[`mappo_baseline.md`](mappo_baseline.md)，执行方式见
[`experiments_runbook.md`](experiments_runbook.md)。

## 研究问题

在 highway-env 无信号路口的合作式离散控制中，验证三层递进：

```text
MAPPO → permutation-invariant Attention-MAPPO → CA-ComMAPPO
```

核心问题是：顺序无关的社会编码是否改善多车状态建模，以及带运动学冲突先验的选择性
通信是否比无通信或通用通信更安全、更高效。

## 算法矩阵

| 组别 | 方法 | 控制变量 |
| --- | --- | --- |
| 环境下界 | random、idle-only | 验证任务和指标是否合理 |
| 无通信 | IPPO、MAPPO | centralized critic 的贡献 |
| 结构基线 | Attention-MAPPO | PI social encoder 的贡献 |
| 通信基线 | CommNet、IC3Net | 平均通信与门控通信 |
| 完整方法 | CA-ComMAPPO | PI encoder + conflict-aware attention communication |
| 关键消融 | CA-ComMAPPO no-prior | 移除 `b_ij`，保留 attention communication |

Attention-MAPPO 同时承担 permutation-invariant / social-attention 强基线。CommNet 和
IC3Net 使用 RNN backbone，适合作为系统级通信基线，但不是严格的单变量消融。

## 假设与证据

| 假设 | 必需证据 | 不能替代它的证据 |
| --- | --- | --- |
| H1：MAPPO 学到优于静态策略的协作 | 相同 100 cases 上的 arrival/collision/reward | 训练曲线或少量视频 |
| H2：PI encoder 降低顺序敏感性 | MAPPO 与 Attention-MAPPO 的原序/扰动配对测试 | 只做普通 final test |
| H3：显式通信改善协作 | Attention-MAPPO 与 CA-ComMAPPO 多 seed 对比 | CA 与 CommNet 单独比较 |
| H4：冲突先验有贡献 | full 与 no-prior 的配对多 seed 消融 | attention heatmap |
| H5：方法可解释 | 固定案例的 attention/TTC 导出与轨迹视频 | 单张挑选的成功截图 |
| H6：方法对规模变化稳健 | 预注册的跨车辆数协议和兼容 checkpoint | 仅声称 PI 理论上不变 |

## 主要指标

- 安全：episode collision rate、agent collision fraction。
- 成功：episode arrival rate、agent arrival fraction。
- 效率：travel time、waiting time、throughput。
- 优化参考：episode/agent reward；reward 不作为唯一论文结论。
- 稳健性：跨训练 seed 的 sample std，以及配对置信区间或显著性检验。

所有指标由共享 evaluator 产生。best checkpoint 排序固定为
`arrival↑ → collision↓ → reward↑ → length↓`；最终表只使用独立 100-episode test。

## 当前完成度

| 项目 | 状态 |
| --- | --- |
| 环境适配、统一指标、固定评估 seed、provenance v2 | 已完成 |
| MAPPO canonical 配置与 seed 1 benchmark/final/video | 已完成 |
| random / idle-only 同 case 正式结果 | 已完成 |
| IPPO、Attention-MAPPO、CommNet、IC3Net、CA-ComMAPPO 代码与主配置 | 已有，需逐项重新 smoke |
| MAPPO seeds 2/3 | 未运行 |
| 其他学习算法多 seed 正式训练 | 未运行 |
| 顺序扰动 evaluator | 已实现，待最终 checkpoint 生成正式 artifact |
| attention/TTC 导出和论文图 | 未实现 |
| 跨车辆数 checkpoint 兼容协议 | 未设计完成 |
| 最终统计检验与论文结果归档 | 未完成 |

当前只能把历史 MAPPO/CA-ComMAPPO seed 1 作为预实验参考；provenance v1 结果不进入
新的正式聚合，不能形成跨算法论文结论。

## 实施顺序

1. 实现顺序扰动、attention/TTC 导出和统计绘图工具，并先冻结评估协议。
2. 在同一提交上 smoke 全算法，冻结算法矩阵和训练 seed 集 `[1,2,3]`。
3. 在最终冻结提交上依次重跑 MAPPO、Attention-MAPPO、CA-ComMAPPO/no-prior，随后
   运行 IPPO、CommNet、IC3Net。
4. 用显式 manifest 聚合，完成统计检验、图表和最终结果文档。

本论文固定四车任务，不开展 4→6/8/10 checkpoint 泛化实验，也不声明规模泛化能力。

## 论文边界

- 研究假设尚未由多 seed 实验验证，文中应区分“设计目标”和“实证结论”。
- 冲突先验是单中心 kinematic proxy，不包含真实路线优先级或所有背景车辆。
- 理想通信不考虑延迟、丢包、带宽和隐私；这些属于后续工作。
- MAAC 与 MADDPG 暂不进入主矩阵：前者关注 attention critic，后者不适合当前离散
  动作主线；如后续加入，必须预注册其角色和比较口径。

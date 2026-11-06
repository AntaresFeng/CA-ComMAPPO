# 实现与研究现状

## 研究状态

2026-11-07：旧论文计划、H1–H6 假设、固定四车论文范围以及基于中心距离/航向构造
intent/TTC 的算法设计全部作废。新的研究问题、意图表示、通信机制和实验矩阵尚未
确定；当前只考虑理想通信。已有代码、默认参数和历史结果是可修改的研究起点。

## 已有实现

| 部分 | 代码入口 | 当前能力 |
| --- | --- | --- |
| 路口环境 | `ca_commappo/envs/intersection.py` | 自定义道路连接、出生与路线、奖励、终止和到达退出 |
| 多智能体适配 | `ca_commappo/envs/highway_intersection_wrapper.py` | XuanCe 字典接口、存活 mask、奖励屏蔽和等待计时 |
| 训练入口 | `ca_commappo/training/train_highway_intersection.py` | train、benchmark、test、视频与诊断模式 |
| 基础算法 | `configs/{ippo,mappo,attention_mappo}/` | IPPO、MAPPO、带排列不变编码器的 MAPPO |
| 通信基线 | `configs/{commnet,ic3net}/` | XuanCe 的 CommNet、IC3Net，当前配置使用 RNN |
| 遗留 CA 实现 | `ca_commappo/algorithms/ca_commappo_agents.py`、`conflict_aware_comm.py` | 旧运动学代理与通信注意力；full/no-prior 配置仍可运行 |
| 共享评估 | `ca_commappo/evaluation/` | 逐局指标、provenance、视频和严格聚合 |

环境升级提交 `77453a5` 通过 263 项测试、MAPPO train 与 CA benchmark 烟测，以及
8 车短运行检查。新环境尚无正式训练结果；环境支持多车不等于旧 checkpoint 能跨车数
加载或泛化。历史矩阵入口见 [results/README.md](../results/README.md)。

## 尚未解决的问题

- 当前观测仍为 `15×7` 运动学特征，默认感知半径 200 m，没有建筑遮挡。局部可见范围
  和信息权限尚未完成设计，通信收益不能由管线可运行推断。
- 每车有明确路线，但导航只通过 `info["agents_navigation"]` 提供诊断，尚未接入 actor。
  真实转向/让行意图的表示、识别与共享尚未实现。
- 旧 CA 的 `intent` 只是朝向中心的二值代理，`TTC` 也不是真实车对碰撞时间。保留这些
  变量名仅为维护旧实现，不将它们重新作为新算法定义。
- PI/CA 编码器依赖固定观测布局；改变特征或车辆数时需检查 actor、critic、agent
  identity 和模型加载兼容性。CommNet/IC3Net 的 RNN 结构也使跨算法差异不止通信方式。
- `ca-trace` 和 `paper-analysis` 仍包含旧 CA 字段与固定配对，属于历史分析工具；
  新研究需要先调整工具，不能沿用其默认输出作为新论文方案。

后续先明确场景与信息权限，再验证环境和无通信基线，随后依据可测的协作问题设计
算法与对照。具体研究方案另行制定，不从旧计划继承结论或限制。

# Highway intersection 环境合同

本文定义项目适配器对 XuanCe 暴露的环境语义。当前任务参数位于
`configs/highway/intersection-multi-agent-v1.yaml`；本文解释接口，不复制整份配置。

## 环境与配置加载

- 正式环境是 `intersection-multi-agent-v1`，由
  `HighwayIntersectionMultiAgentEnv` 适配为 XuanCe `RawMultiAgentEnv`。
- 所有 highway-env 参数必须放在 `highway_config` 中。加载顺序为
  `common → highway → algorithm → smoke → CLI`，后者覆盖前者。
- highway-env 1.11 对嵌套字典只做浅更新。适配器会在 `gym.make(...)` 前深度合并
  `observation` 和 `action`，不得绕过该入口或直接编辑 `.venv`。
- 裸 `intersection-v1` 是单智能体连续动作环境，仅用于明确的兼容性排查。

## 动作与初始速度

每个受控 agent 的动作空间为 `Discrete(3)`：

| 动作 | 语义 |
| ---: | --- |
| `0` | `SLOWER`，目标速度降一档 |
| `1` | `IDLE`，保持当前目标速度 |
| `2` | `FASTER`，目标速度升一档 |

`target_speeds: [0, 4.5, 9]` 是速度挡位，不是动作表。受控车通常以入口限速
`10 m/s` reset，并映射到最高目标挡；所以 `IDLE` 不等于停车。

## Reward 与存活掩码

`step()` 返回训练侧 masked reward：

- 返回的 `rewards` 与 `info["agents_rewards"]` 含义相同。
- agent 到达或终止后的后续 reward 为 `0.0`，动作由适配器替换为 `IDLE`。
- `info["raw_agents_rewards"]` 才是 highway-env 原始 per-agent reward。
- 不得穿透 wrapper 读取 `controlled_vehicles` 来重建 episode 事实。

当前 reward 排序由 canonical YAML 冻结：全程速度 shaping 小于 arrival bonus，碰撞
应明显劣于其他 agent 到达；`normalize_reward` 必须保持关闭。

## 终止与指标事实源

- `info["crashed"]` 和 `info["arrived"]` 是 per-agent tuple，也是所有 evaluator 的
  统一事实源。
- `info["global_terminated"]` 表示环境级终止；完整 episode 结束条件为
  `global_terminated or truncated`。
- episode 结局优先级是 `collision > arrival > truncation`。因此发生任一受控车碰撞
  的 episode 不能同时计为成功到达。
- `waiting_steps` 在 adapter 内按 active agent 累积，`dt=1/policy_frequency`；
  travel time、throughput 和 waiting time 都由共享 metrics 模块计算。

## 观测约定

运动学特征为 `presence, x, y, vx, vy, cos_h, sin_h`。每个 agent 观测的第 0 行是
ego；在当前 `absolute: false` 配置下，ego 行仍是归一化世界坐标，其他车辆行相对
ego。CA-ComMAPPO 依赖这一约定从 ego 行构造冲突点 TTC 代理。

## XuanCe 集成限制

- 离散策略使用 `Categorical_MAAC_Policy`。
- CommNet / IC3Net 必须使用 `Basic_RNN`、`use_rnn: true` 和 `comm_passes`；IC3Net
  的 `gate_hidden_size` 必须是列表。
- 两者的 `action()` 缺失 `info` 时由运行时补丁合成全存活 mask，不修改 site-packages。
- CommNet / IC3Net 的训练循环不触发项目的 episode callback，训练期 highway 指标
  以 benchmark eval 为准。

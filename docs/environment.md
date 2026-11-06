# 项目路口环境

当前环境为 `ca-intersection-v0`，基于 **highway-env 1.12.1** 的
`ConnectedLaneMultiAgentIntersectionEnv`。实现分为：

- `ca_commappo/envs/intersection.py`：道路、场景生成、奖励、终止、车辆退出与首次到达时间。
- `ca_commappo/envs/highway_intersection_wrapper.py`：XuanCe 字典接口、存活 mask、奖励屏蔽和等待时间。
- `configs/highway/ca-intersection-v0.yaml`：当前共享任务配置。

旧 `intersection-multi-agent-v1` 实验属于 highway-env 1.11 的历史任务，结果和 checkpoint
不能直接作为新环境的对照。升级依赖、环境 ID、配置和 runtime source 均进入现有
provenance 校验；历史结果保留原样，不覆盖 metadata。

## 配置与上游修复

所有环境选项通过 `highway_config` 传入，加载顺序仍为
`common → highway → algorithm → smoke → CLI`。

1.12.1 已修复多智能体 `default_config()` 缺失嵌套字段，以及出生清理误删受控车的问题。
项目已删除旧的单车/多车默认值拼接实现，不再维护这些上游问题的补丁。

但 **1.12.1 的 `AbstractEnv.configure()` 仍使用浅层 `dict.update`**；上游
`utils.update_config()` 也只校验并覆盖嵌套字典，并不是递归合并。项目子类复用 XuanCe
的 `recursive_dict_update` 支持局部配置覆盖，例如只改 `target_speeds` 时保留动作类型。
这项配置语义仍然需要，不能与已修复的默认配置问题一起删除。

## 道路与路线

复用上游四向路口几何、转弯轨迹与底层车辆控制；入口节点为 `o0..o3`，出口改为
独立的 `exit0..exit3` 终点，禁止通过路网连接从出口返回入口。编号为
`0=南、1=西、2=北、3=东`。

每车显式使用三段路线：入口 → 路口 movement → 目的出口。目的地不能与入口相同。
相对于入口，`(destination-entry) % 4` 为 `1/2/3` 时分别表示左转/直行/右转。
`access_length` 控制进出口道路长度，`speed_limit` 控制车道限速。
保留上游道路优先权；规则背景车遵守 `RegulatedRoad`，离散受控车是 `MDPVehicle`，
不由该规则层自动替代 RL 做让行决策。启用上游 connected-lane 邻车检测修复。

## 出生与背景车

默认生成器先随机排列四个入口，再按每入口队列放车。`spawn_distance` 是队首距入口
车道末端的距离，后车增加 `spawn_spacing`，各车叠加有界的 `spawn_jitter`。
目的地均匀选择其他三个出口。场景只使用环境 RNG，同一 seed 可重放。

也可在 `highway_config` 中指定完整车辆列表，列表长度必须等于 `controlled_vehicles`：

```yaml
controlled_vehicles: 2
controlled_vehicle_specs:
  - {entry: 0, destination: 2, longitudinal: 60.0, speed: 9.0}
  - {entry: 0, destination: 1, longitudinal: 35.0, speed: 4.5}
```

`longitudinal` 是从入口车道起点计量的米数，不能越出车道。初始速度范围为
`[0, speed_limit]`。生成器检查 `minimum_spawn_gap`，非法/重叠的显式场景直接报错，
不会删车、移车或静默减少 agent 数量。车辆数受实际道路容量限制；队列超出道路时需要
增加 `access_length` 或调整出生距离，不能仅增加计数。

`initial_vehicle_count` 精确指定初始背景车数，`spawn_probability` 指定每个决策步新增
一辆背景车的尝试概率。没有隐式 challenger，也没有 reset 前三秒预演。
初始背景车放置失败会报错；运行中的入口被占据时跳过本次出生尝试。
两项都为零时，不论受控车数量多少，整个回合都不会产生背景车。

## 动作与观测

离散动作保持 `0=SLOWER, 1=IDLE, 2=FASTER`，默认目标速度档为 `[0, 4.5, 9]`。
`IDLE` 保持目标速度，不代表停车。`initial_speed` 控制初始实际速度。

本阶段保留现有算法使用的 `15×7` 观测布局：
`presence, x, y, vx, vy, cos_h, sin_h`。第 0 行是 ego；`absolute: false` 时 ego 行仍是
归一化世界坐标，其他车辆行为相对坐标。CA 的现有运动学代理仍依赖这一布局。

`perception_distance` 可配置观测选车半径，默认 200 m；目前没有建筑遮挡。
`info["agents_navigation"]` 暴露逐车的入口、目的地和出生参数，用于场景审计与后续导航
输入设计。它是集中诊断数据，不能未经信息权限设计就把所有车辆的私有路线喂给 actor。
本轮没有改变 actor 输入维度，也没有实现意图预测/通信算法。

原生 Kinematics 支持 `cos_d/sin_d` 导航方向特征，但切换特征列表前还需同步修改项目中
固定布局的 PI/CA representation；不能仅改 YAML 就宣称现有算法已使用导航信息。

## Reward、退出与终止

保留 canonical reward 数值，明确优先级：

1. 碰撞车获得 `collision_reward`；即使同时位于到达区域，也不获得到达奖励。
2. 首次到达指定出口 `arrival_distance` 后，获得一次 `arrived_reward`。
3. 已到达车后续 reward 为零；其余在道路上的车获得速度 shaping。

到达车从物理道路退出，避免继续行驶产生二次碰撞；固定 `controlled_vehicles` / agent
槽位保留，供 XuanCe 的定长观测、终止和 mask 使用。其后续动作被替换为 `IDLE`。
`info["raw_agents_rewards"]` 保存子类原始 reward，`step()` 的 reward 与
`info["agents_rewards"]` 是 adapter masked reward；两者在已到达车上均为零。

任一受控车碰撞或所有车成功到达时全局终止；启用 `offroad_terminal` 时检查所有仍活跃
受控车，而不是只检查第 0 辆。到达状态与终止状态保持 sticky。
使用整数决策步限制时间，默认 `20 s × 5 Hz = 100`，避免浮点时间累积多走一步。
`simulation_frequency` 必须是 `policy_frequency` 的整数倍。

`agent_mask()` 对应刚执行 transition 的动作前 mask；`info["next_agent_mask"]` 对应
下一动作的 mask。到达 transition 仍参与学习，之后不再参与奖励、等待计时或通信。
XuanCe 的 masked advantage / ValueNorm 兼容修复继续保留。

## 指标与接口事实源

- `crashed`、`arrived`、`current_arrived` 是逐车 tuple。
- `global_terminated` 是环境级信号；完整回合结束为它或 `truncated`。
- `arrival_times` 记录每车从 reset 到首次成功到达的秒数，分辨率为一个决策步；未完成、
  碰撞车为 `None`，不能把第一次 terminal 时间当成成功通行时间。
- `waiting_steps` 只在动作前活跃且速度低于 1 m/s 的车上累加，`dt=1/policy_frequency`。

共享 evaluator 将 `arrival_times` 写入 episode payload，新增 `completed_travel_time`
（该局已完成车辆的平均通行时间）以及跨 episode 按车辆汇总的
`mean_completed_travel_time` / `completed_agents`。没有成功车辆时，episode 值为 null，
summary 不产生虚假的零通行时间，使用 `completed_agents=0` 表明无样本。

为保持历史 artifact 读取兼容，旧 `travel_time` / `mean_travel_time` 字段仍表示**回合
持续时间**；新记录同时使用准确命名 `episode_duration` / `mean_episode_duration`。
旧聚合表里的 travel 列不可解释为逐车通行时间。
`throughput` 仍为到达数量/该回合时长，仅作为有限批次的完成速率，不代表稳态交通容量。

episode 结局优先级保持 `collision > arrival > truncation`，best checkpoint 排序继续是
`arrival↑ → collision↓ → reward↑ → length↓`。评估器只读取 adapter info，不自行遍历
底层车辆重建结果。

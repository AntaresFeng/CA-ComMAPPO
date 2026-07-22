# Highway Intersection Notes

## MAPPO action semantics

当前 `HighwayWrapper` 使用 `intersection-multi-agent-v1`。每个 agent 的动作空间是
`Discrete(3)`，底层来自 highway-env 的 `DiscreteMetaAction`：

- `0`: `SLOWER`
- `1`: `IDLE`
- `2`: `FASTER`

`[0, 4.5, 9]` 不是动作表，而是 `MDPVehicle.target_speeds` 目标速度挡位表。
`SLOWER` / `FASTER` 会先把当前实际速度映射到最近的速度挡位，再降一档或升一档并裁剪到合法范围。
`IDLE` 保持当前目标速度。

## Reset-time speeds

只看 `intersection-multi-agent-v1`：

- 背景 IDM 车辆由 `_spawn_vehicle()` 生成，初始速度是
  `8.0 + normal() * speed_deviation`。普通背景车使用 `speed_deviation=1.0`。
- 普通背景车生成后，环境会先 warm up 3 秒，因此 reset 返回时它们的速度已经可能偏离初始采样值。
- warm up 之后会生成一辆 guaranteed challenger，`speed_deviation=0.0`，所以生成速度是
  `8.0 m/s`，但如果离受控车过近，后续防碰撞逻辑可能移除它。
- 受控车最后生成在入口车道，初始实际速度使用 `ego_lane.speed_limit`。当前 intersection 入口车道限速是
  `10.0 m/s`。
- 受控车是 `MDPVehicle`，目标速度挡位为 `[0, 4.5, 9]`。因此 reset 后通常是
  `speed=10.0`、`speed_index=2`、`target_speed=9.0`。

运行时检查也符合这个结论：受控车 reset 在最高目标速度挡位，背景车速度随 seed、生成位置和 warm up 仿真变化。

## Adapter reward contract

`HighwayIntersectionMultiAgentEnv.step()` 面向 XuanCe 训练语义返回 reward：

- 返回值里的 `rewards` 字典是 masked reward。
- `info["agents_rewards"]` 与返回的 `rewards` 保持一致，也是 masked reward tuple。
- `info["raw_agents_rewards"]` 保存 highway-env 原始 per-agent reward tuple。
- agent 到达或终止后的下一步开始会被视为 inactive：adapter 会用 `IDLE_ACTION` 代替其动作，并把该 agent 的 adapter-facing reward 置为 `0.0`。
- `info["global_terminated"]` 是环境级结束信号；`info["crashed"]` 是 per-agent crash tuple，不使用 highway-env 默认的单车 `info["crashed"]` 语义。

## CommNet / IC3Net 集成注意事项

CommNet / IC3Net（`xuance.torch.agents`）作为通信基线接入本项目时，有三处与 highway 适配器/评估器接口的偏差，已用运行时补丁和配置解决，记录如下。

### 1. `action()` 需要 `info` 参数（已运行时打补丁）

`CommNet_Agents.action()` / `IC3Net_Agents.action()` 都通过 `for data in info` 取每个 env 的 `agent_mask` 来构造 `alive_ally`，`info=None` 时立即 `TypeError`。但本项目评估器 `evaluate_highway_policy()`（`mappo_highway_metrics.py`）和视频录制器 `_agent_action_for_single_env`（`mappo_video_recorder.py`）调 `agents.action(...)` 时**不传 `info`**。因此 `--mode test` / `--mode benchmark` 会崩；而 `--mode train` smoke 走 CommNet 自己的 `run_episodes()`（内部会构造并传入 `info`），**不暴露此问题**。

补丁在 `ca_commappo/algorithms/comm_runtime_patch.py` 的 `patch_comm_agents_info()`：运行时 monkey-patch 两个 agent 类的 `action()`，当 `info is None` 时合成 `info = [{'agent_mask': {k: 1 for k in agent_keys}} for _ in range(n_env)]`（全 1，即所有 agent alive）。这与 CommNet `run_episodes()` 训练期构造 `info` 的方式（`commnet_agents.py` `info = [{'agent_mask': {k: True ...}} ...]`）完全一致，故 eval 行为与训练一致。补丁幂等、不改 `.venv`。`run()` 在构造 agent 前调用。

### 2. CommNet 的 `run_episodes()` 不调训练期 callback（已知限制）

CommNet 重写的 `run_episodes()`（`commnet_agents.py:155-252`）在 episode 结束时只调 `self.log_infos(step_info, ...)`，**不调 `callback.on_train_episode_info`**。IC3Net 继承该方法，行为相同。因此 `HighwayMetricsCallback` 在 CommNet/IC3Net 训练期收不到任何 per-episode 记录，训练期 highway 指标为空——**训练期指标以 eval 期（`evaluate_highway_policy`）为准**。`--mode benchmark` 的 eval 期记录正常落盘到 `eval_records.jsonl`。

### 3. CommNet/IC3Net 配置必填项与类型

- `use_rnn: true` + `representation: "Basic_RNN"`（必需，非可选）。CommNet/IC3Net 的 policy `forward` 依赖 RNN 路径；`use_rnn: false` 会在训练时崩。
- `comm_passes`（CommNet/IC3Net 必填，本项目用 `1`）。
- `gate_hidden_size`（IC3Net 必填）**必须是 list**，如 `[16]`，**不是标量 `16`**。原因：`IC3Net_Policy.__init__` 调 `communicator.create_mlp(recurrent_hidden_size, gate_hidden_size, 2, ...)`，而 `CommNet.create_mlp` 的第二个参数 `layers` 期望 list（内部 `len(layers)`）；传标量会 `TypeError: object of type 'int' has no len()`。canonical xuance IC3Net 配置用 `[64,]`。
- `lr_decay_steps`（CommNet/IC3Net 必填）：`CommNet_Learner.build_optimizer` 一定会建 `LinearLR` 调度器并 `total_iters=lr_decay_steps`（即使 `use_linear_lr_decay: false`）。本项目用 `5000`（与 canonical xuance CommNet mpe 配置一致）。注意这是 CommNet/IC3Net 相对 MAPPO baseline 的一个行为差异（CommNet LR 会衰减），归档时记录。


# Sanity Baselines And Reproducibility Protocol

Date: 2026-06-29

## Purpose

在训练 MAPPO 之前，先运行非学习型 sanity baseline，确认 highway intersection 环境、奖励、碰撞/到达统计和 seed 复现逻辑是正常的。

第一版支持两种策略：

- `random`: 每个 agent 每步从离散动作空间均匀采样。
- `idle-only`: 每个 agent 每步固定执行动作 `1`，即 highway-env 的 `IDLE`。

## Default Command

运行配置中所有策略：

```powershell
uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy all
```

只运行随机策略：

```powershell
uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy random
```

只运行 idle-only：

```powershell
uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy idle-only
```

保存完整 episode 记录和 summary：

```powershell
uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy all --output results/sanity_highway.json
```

## YAML Protocol

默认配置：

`configs/sanity/highway_intersection.yaml`

字段含义：

- `env_id`: highway-env 环境 id，当前为 `intersection-multi-agent-v1`。
- `env_seed`: 未显式提供 `seeds` 时使用的默认 seed。
- `episodes`: 每个 base seed 运行的 episode 数。
- `seeds`: base seed 列表。
- `policies`: 配置文件默认要运行的策略列表。
- `highway_config`: 传入适配器的 highway-env 配置，支持嵌套覆盖。

当 `episodes > 1` 时，脚本会从每个 base seed 派生实际 episode seed，避免重复运行完全相同的 episode。

## Metrics

summary 输出：

- `episodes`: episode 总数。
- `mean_episode_reward`: 每个 episode 的平均 agent 累计回报，再对 episode 取均值。
- `mean_agent_reward`: 所有 episode、所有 agent 的累计回报均值。
- `mean_episode_length`: episode 步数均值。
- `collision_rate`: 发生任意受控车辆碰撞的 episode 比例。
- `arrival_rate`: 所有受控车辆到达且没有碰撞的 episode 比例。
- `truncation_rate`: 没有碰撞、没有全员到达、因时间限制结束的 episode 比例。

每个 episode 记录还包含：

- `crashed_agents`
- `arrived_agents`
- `agent_collision_fraction`
- `agent_arrival_fraction`

## Collision And Arrival Source

raw highway-env 的 `info["crashed"]` 不作为主源。

原因：raw highway-env 的该字段来自 abstract env 的单车字段，在多受控车辆场景下可能不能代表所有 agent。实际探查中出现过某个受控车辆已经 `vehicle.crashed == True`，但 raw `info["crashed"] == False` 的情况。

本项目坚持由 `HighwayIntersectionMultiAgentEnv` adapter 作为 episode fact 的统一出口。adapter 会将 raw highway-env 的车辆状态转换为 per-agent tuple，并通过 `info["crashed"]` 和 `info["arrived"]` 暴露给上层。sanity baseline、MAPPO callback 和 MAPPO evaluator 都消费这两个 adapter-facing 字段，不再穿透 env 或通过共享 helper 读取 `controlled_vehicles`。

正式统计入口使用：

```python
crashed = info["crashed"]
arrived = info["arrived"]
```

episode 结束语义与 highway-env intersection 保持一致：

```python
any(crashed) or all(arrived) or truncated
```

## Reproducibility Rules

- 同一个 YAML、同一个 policy、同一个 seed 列表，应产生相同结果。
- `random` 策略使用 `numpy.random.default_rng(seed)`，不依赖 Gymnasium space 内部 RNG。
- 实验记录应保存 YAML 文件、命令、git commit 或工作树说明、输出 JSON。
- 后续 MAPPO/IPPO 评估应复用同一套 metric 定义，避免学习型和非学习型 baseline 使用不同统计口径。

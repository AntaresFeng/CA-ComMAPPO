# Sanity Baselines Design

Date: 2026-06-29

## Goal

为 highway intersection 多智能体环境提供可复现实验前置 sanity baseline，第一版只支持 `random` 和 `idle-only` 两种策略，并通过 YAML 配置导入环境和评估参数。

## Scope

本设计只覆盖非学习型 sanity baseline，不实现 MAPPO/IPPO 训练，也不加载模型。

第一版输出基础评估指标：

- episode 数量
- mean episode reward
- mean per-agent reward
- mean episode length
- collision rate
- arrival rate
- truncation rate

## Architecture

实现分为可复用模块和命令行脚本两层：

- `ca_commappo/evaluation/sanity_baselines.py`
  - 负责 YAML 加载、策略动作选择、episode 运行、指标汇总和 JSON 写出。
  - 该模块可被未来 MAPPO/IPPO 评估脚本复用。
- `examples/random_highway_intersection.py`
  - 负责命令行参数解析和打印结果。
  - 支持 `--config`、`--policy`、`--output`。
- `examples/sanity/highway_intersection.yaml`
  - 默认 sanity baseline 配置。
- `tests/test_sanity_baselines.py`
  - 覆盖配置加载、动作选择、指标汇总和真实 highway 环境的轻量 episode。

## YAML Configuration

默认配置文件建议放在：

`examples/sanity/highway_intersection.yaml`

配置结构：

```yaml
env_id: "intersection-v1"
env_seed: 1
episodes: 5
seeds: [1, 2, 3]
policies: ["random", "idle-only"]

highway_config:
  controlled_vehicles: 3
  duration: 13
  initial_vehicle_count: 10
  spawn_probability: 0.6
  normalize_reward: false
```

命令行可以覆盖 policy：

```powershell
uv run python examples/random_highway_intersection.py --config examples/sanity/highway_intersection.yaml --policy random
uv run python examples/random_highway_intersection.py --config examples/sanity/highway_intersection.yaml --policy idle-only
uv run python examples/random_highway_intersection.py --config examples/sanity/highway_intersection.yaml --policy all
```

## Policy Semantics

`idle-only`：

- 所有 agent 固定执行动作 `1`。
- 在 highway-env `IntersectionEnv.ACTIONS` 中，动作 `1` 是 `IDLE`。

`random`：

- 每一步对每个 agent 从其离散动作空间均匀采样。
- 不使用 `action_space.sample()`，而是使用 `numpy.random.default_rng(seed)` 对 `Discrete.n` 采样，保证同一 seed 下动作序列可复现。

## Episode Termination

sanity baseline 直接使用 `HighwayIntersectionMultiAgentEnv`，不使用 XuanCe vector wrapper。

episode 停止条件按 highway-env intersection 的环境级终止语义定义：

```python
any(vehicle.crashed for vehicle in controlled_vehicles)
or all(base_env.has_arrived(vehicle) for vehicle in controlled_vehicles)
or truncated
```

原因：

- highway-env `IntersectionEnv._is_terminated()` 的语义是任一受控车辆碰撞，或所有受控车辆到达。
- 当前适配器返回的是 per-agent `terminated` 字典，即每辆车 `crashed or has_arrived`。
- 如果只使用 `all(terminated.values())`，部分车辆碰撞后 episode 会继续滚动，和 highway-env 的环境级终止不一致。

## Collision And Arrival Signals

探查结论：

- `env.env.unwrapped.controlled_vehicles[i].crashed` 是逐车碰撞状态主源。
- `env.env.unwrapped.has_arrived(vehicle)` 是逐车到达状态主源。
- `info["rewards"]["collision_reward"]` 是受控车辆碰撞比例，可用于交叉校验。
- `info["rewards"]["arrived_reward"]` 是受控车辆到达比例，可用于交叉校验。
- `info["agents_terminated"]` 是逐车 `crashed or has_arrived`，不能区分碰撞和到达。
- `info["crashed"]` 不能作为多智能体碰撞率主源，因为它来自 abstract env 的单车字段；探查中出现过 `info["crashed"] == False` 但第二辆受控车 `vehicle.crashed == True` 的情况。

因此正式实现必须基于 `controlled_vehicles` 逐车读取：

```python
crashed = [bool(vehicle.crashed) for vehicle in base_env.controlled_vehicles]
arrived = [bool(base_env.has_arrived(vehicle)) for vehicle in base_env.controlled_vehicles]
```

episode 级指标：

- `collision = any(crashed)`
- `arrival = all(arrived) and not collision`
- `truncated = bool(truncated) and not collision and not arrival`

逐车比例指标：

- `agent_collision_fraction = mean(crashed)`
- `agent_arrival_fraction = mean(arrived)`

第一版 summary 输出 episode 级 `collision_rate`、`arrival_rate`、`truncation_rate`。逐车比例可保留在每个 episode record 中，后续再决定是否进入 summary。

## Output Shape

`run_sanity_baseline()` 返回 dict：

```python
{
    "config": {
        "env_id": "intersection-v1",
        "episodes": 5,
        "seeds": [1, 2, 3],
    },
    "policies": {
        "random": {
            "episodes": [...],
            "summary": {
                "episodes": 15,
                "mean_episode_reward": 1.23,
                "mean_agent_reward": 0.41,
                "mean_episode_length": 9.8,
                "collision_rate": 0.2,
                "arrival_rate": 0.6,
                "truncation_rate": 0.2,
            },
        },
        "idle-only": {
            "episodes": [...],
            "summary": {...},
        },
    },
}
```

每个 episode record 包含：

- `policy`
- `seed`
- `episode_index`
- `steps`
- `episode_reward`
- `agent_rewards`
- `collision`
- `arrival`
- `truncated`
- `crashed_agents`
- `arrived_agents`
- `agent_collision_fraction`
- `agent_arrival_fraction`

## Error Handling

- YAML 文件不存在：抛出 `FileNotFoundError`，CLI 打印错误并返回非零退出码。
- YAML 缺少 `highway_config`：使用空 dict，依赖适配器默认配置。
- `episodes <= 0`：抛出 `ValueError`。
- `seeds` 为空：使用 `[env_seed]`。
- 未知 policy：抛出 `ValueError`，错误信息列出合法值：`random`、`idle-only`、`all`。
- 非离散 action space 用于 `random`：抛出 `TypeError`，因为第一版只支持 highway intersection 当前的 discrete meta action。

## Testing

采用 TDD。

测试覆盖：

1. YAML 配置加载。
2. `idle-only` 对任意 agent 返回动作 `1`。
3. `random` 同一 seed 下动作序列可复现。
4. summary 聚合均值和 rate。
5. 真实 highway intersection 环境运行 1 个短 episode，不发生接口错误，并在 `finally` 中关闭环境。

## Out Of Scope

- MAPPO 训练和模型评估。
- rule-based/cautious 策略。
- CSV 输出。
- TensorBoard/WandB 日志。
- 轨迹视频录制。

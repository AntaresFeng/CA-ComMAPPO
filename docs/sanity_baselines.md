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

## Strict Final-Test Protocol

与 MAPPO `best_model.pth` 的正式 100-episode test 横向对比时，不使用上面的
legacy 小样本配置。统一运行：

```bash
uv run python -m ca_commappo.cli.run_sanity_baseline \
  --config configs/sanity/highway_intersection_final.yaml \
  --policy all \
  --artifact-root runs \
  --output results/sanity_highway_final.json
```

该配置把 `20000..20099` 写成显式、连续且有序的 episode seed 列表。random
与 idle-only 按完全相同的顺序运行这 100 个 case。环境直接使用原始 episode
seed；random 动作 RNG 使用 `SeedSequence(episode_seed).spawn(1)[0]` 派生的独立
child stream，避免环境 RNG 和动作 RNG 使用相同的随机流，同时仍可精确重放。
sanity runner 当前串行执行，所以元数据如实记录 `parallels: 1`；并行度只是
执行细节，不改变 case 集合。

evaluation protocol 区分“策略确定性”和“运行可复现性”：

- random: `deterministic_actions=false`、`reproducible_actions=true`、
  `action_selection_mode=seeded_random`。
- idle-only: `deterministic_actions=true`、`reproducible_actions=true`、
  `action_selection_mode=fixed_policy`。

因此不能因为 seeded-random 可重放，就把它错误标记成 deterministic policy。

`--artifact-root runs` 会分别创建：

```text
runs/random_highway/<timestamp>_test_seed_1/eval_metadata.json
runs/random_highway/<timestamp>_test_seed_1/eval_records.jsonl
runs/idle_only_highway/<timestamp>_test_seed_1/eval_metadata.json
runs/idle_only_highway/<timestamp>_test_seed_1/eval_records.jsonl
```

每个目录只有一个 `mode=test, phase=test, epoch=null` 的最终记录，包含 100 条
逐 episode 记录、显式 seed、summary、score 和完整 evaluation protocol。环境
配置直接复用 `configs/highway/intersection-multi-agent-v1.yaml`，避免 sanity 与
MAPPO 的 task snapshot 漂移。`run_seed` 只是静态 baseline 的 artifact identity；
它不会替代或改变 `20000..20099` 的 episode seeds。

需要另一个预先声明的 case 集合时，可以改 YAML，或同时覆盖 base/count：

```bash
uv run python -m ca_commappo.cli.run_sanity_baseline \
  --config configs/sanity/highway_intersection_final.yaml \
  --episode-seed-base 30000 --episode-count 100 \
  --artifact-root runs
```

不能只覆盖 base 或 count 其中一个。正式横向对比也不得在看到结果后换 seed
集合。

### Canonical Reference Run

使用上述原样命令和 `20000..20099` 得到：

| policy | episodes | arrival | collision | truncation | mean reward |
|---|---:|---:|---:|---:|---:|
| random | 100 | 0.10 | 0.75 | 0.15 | -12.4574 |
| idle-only | 100 | 0.17 | 0.83 | 0.00 | -16.1293 |

可聚合产物位于：

- `runs/random_highway/20260810_220643_345839_test_seed_1/`
- `runs/idle_only_highway/20260810_220643_345839_test_seed_1/`
- 严格对比表：`results/mappo_baseline_seed1_comparison.csv`

这两个 run 使用 canonical MAPPO baseline 的奖励合同，并与 learned policy 共用
`20000..20099`。更早、协议不完整或旧奖励口径的 sanity runs 已清理，不得用于论文。

## YAML Protocol

默认配置：

`configs/sanity/highway_intersection.yaml`

正式对比配置：`configs/sanity/highway_intersection_final.yaml`。

字段含义：

- `env_id`: highway-env 环境 id，当前为 `intersection-multi-agent-v1`。
- `env_seed`: 未显式提供 `seeds` 时使用的默认 seed。
- `episodes`: 每个 base seed 运行的 episode 数。
- `seeds`: base seed 列表。
- `policies`: 配置文件默认要运行的策略列表。
- `highway_config`: 传入适配器的 highway-env 配置，支持嵌套覆盖。
- `highway_config_path`: 可复用另一个 YAML 中的完整 highway task 配置；同名
  `highway_config` 内联项会作为显式覆盖。
- `episode_seed_base` / `episode_count`: 两者必须同时提供，生成正式验证用的
  显式连续 seed 列表。
- `eval_parallels`: sanity 当前只支持真实串行值 `1`。
- `run_seed`: 标准 artifact 中的 run identity seed，不参与 episode 或动作采样。

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
- `random` 策略使用 episode seed 的独立 `SeedSequence` child 创建
  `numpy.random.Generator`，不复用环境 seed stream，也不依赖 Gymnasium space
  内部 RNG。
- 实验记录应保存 YAML 文件、命令、git commit 或工作树说明、输出 JSON。
- 后续 MAPPO/IPPO 评估应复用同一套 metric 定义，避免学习型和非学习型 baseline 使用不同统计口径。

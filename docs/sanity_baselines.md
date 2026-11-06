# Sanity baseline 协议

> 环境迁移说明：当前命令使用 `ca-intersection-v0` / highway-env 1.12.1。
> 本文引用的已有训练数字与 checkpoint 均属于旧 1.11 四车任务，不能直接迁移；
> 新环境语义见 [environment.md](environment.md)。

本文定义 random 与 idle-only 两个非学习策略的用途、复现方式和正式结果。它们用于
验证环境和指标下界，不是训练 seed，也不应复制成伪造的跨 seed 样本。

## 策略

- `random`：每步为每个 active agent 均匀采样 `Discrete(3)` 动作。
- `idle-only`：始终选择动作 `1=IDLE`。IDLE 保持目标速度，不代表停车。

## 快速检查

```bash
uv run python -m ca_commappo.cli.run_sanity_baseline \
  --config configs/sanity/highway_intersection.yaml \
  --policy all
```

该配置用于开发期小样本诊断。需要保存便于人工查看的完整 JSON 时加：

```bash
--output results/sanity_highway.json
```

## 正式对比

```bash
WANDB_MODE=offline uv run python -m ca_commappo.cli.run_sanity_baseline \
  --config configs/sanity/highway_intersection_final.yaml \
  --policy all \
  --artifact-root runs \
  --output results/sanity_highway_final.json
```

正式配置直接引用 canonical highway task，并固定 episode seeds `20000..20099`，与
学习策略 final test 完全一致。runner 当前串行执行，所以 artifact 如实记录
`parallels=1`。

Random 的动作 RNG 从每个 episode seed 派生独立 `SeedSequence` child，不与环境共用
同一随机流：它是可复现的随机策略，而不是确定性策略。协议分别记录：

| policy | action selection | deterministic | reproducible |
| --- | --- | --- | --- |
| random | `seeded_random` | false | true |
| idle-only | `fixed_policy` | true | true |

## Artifact

使用 `--artifact-root runs` 后，每个策略生成一个独立 final-test run：

```text
runs/<policy>_highway/<timestamp>_test_seed_1/
├── eval_metadata.json
└── eval_records.jsonl
```

记录必须包含 100 个逐 episode payload、显式 seed 列表、summary 和 evaluation
protocol。`run_seed` 仅作为 artifact identity，不参与环境或动作采样。

## Canonical 参考结果

在当前 MAPPO baseline 的任务合同和 `20000..20099` 上：

| policy | episodes | arrival | collision | truncation | mean reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 100 | 0.10 | 0.75 | 0.15 | -12.4574 |
| idle-only | 100 | 0.17 | 0.83 | 0.00 | -16.1293 |

机器可读对比见 `results/mappo_baseline_seed1_comparison.csv`。保留的原始 runs 位于：

- `runs/random_highway/20260810_220643_345839_test_seed_1/`
- `runs/idle_only_highway/20260810_220643_345839_test_seed_1/`

## 指标和事实源

sanity 与学习策略共用 `ca_commappo/evaluation/highway_metrics.py`。主要指标为 episode
reward/length、collision/arrival/truncation rate、per-agent collision/arrival
fraction，以及条件性的 travel time、throughput、waiting time。

碰撞和到达只读取 adapter 提供的 `info["crashed"]` 与 `info["arrived"]` per-agent
tuple，不读取 highway-env 的单车字段，也不穿透 wrapper。结局优先级固定为
`collision > arrival > truncation`。

正式对比若要更换 case 集合，必须在看结果前同时声明 `episode_seed_base` 和
`episode_count`，并对所有策略使用同一集合。

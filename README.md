# CA-ComMAPPO

用于把 `highway-env` 交叉路口环境接到 XuanCe 多智能体接口上的本地研究仓库。

当前仓库已经具备：

- `intersection-v1` 的 XuanCe `RawMultiAgentEnv` 适配器。
- XuanCe 环境注册和 `make_envs(...)` 基础兼容。
- 适配器 reset / step / state 测试。
- random / idle-only sanity baseline 和基础评估指标。
- 本地 vendored XuanCe `examples/mappo` 参考脚本。

当前还不等价于完整 MAPPO baseline。缺口主要是 highway 专用 MAPPO
训练入口、YAML 配置、小步端到端训练烟测、训练后评估脚本，以及 terminal
agent 后续 transition 的 mask / reward 语义处理。细节见
[`docs/highway_intersection_mappo_baseline_gap_analysis.md`](docs/highway_intersection_mappo_baseline_gap_analysis.md)。

## 本地运行

日常使用 `uv run` 运行脚本：

```powershell
uv run python main.py --help
uv run python main.py smoke
uv run python main.py debug --target wrapper --actions 1 --duration 2
uv run python main.py sanity --policy random
```

也可以直接运行 sanity baseline：

```powershell
uv run python examples/random_highway_intersection.py --config examples/sanity/highway_intersection.yaml --policy all
uv run python examples/random_highway_intersection.py --config examples/sanity/highway_intersection.yaml --policy all --output results/sanity_highway.json
```

只在修改适配器核心逻辑、排查回归或准备提交时运行测试：

```powershell
uv run pytest -q
```

这是本地实验研究仓库，不把发布构建作为常规步骤；除非明确需要产物，不要
运行 `uv build`。

## 项目结构

- `ca_commappo/envs/highway_intersection.py`: highway intersection XuanCe 适配器。
- `ca_commappo/evaluation/sanity_baselines.py`: random / idle-only sanity baseline 逻辑。
- `examples/random_highway_intersection.py`: sanity baseline 命令入口。
- `examples/debug_highway_env_episode.py`: raw / wrapper 环境调试入口。
- `examples/sanity/highway_intersection.yaml`: 默认 sanity baseline 配置。
- `examples/mappo/`: 本地保存的 XuanCe MAPPO 参考示例；当前不是 highway baseline。
- `tests/`: 适配器、入口和 sanity baseline 测试。
- `docs/`: 设计说明、排查记录和 baseline 缺口分析。

## Highway `intersection-v1` 的 xuance 适配器

项目提供 `HighwayIntersectionMultiAgentEnv`，它是基于
`gym.make("intersection-v1", ...)` 的 xuance `RawMultiAgentEnv` 适配器。
所有 highway 环境参数都通过 `highway_config` 配置，包括
`controlled_vehicles`。

```python
from argparse import Namespace

from ca_commappo.envs.highway_intersection import HighwayIntersectionMultiAgentEnv

env = HighwayIntersectionMultiAgentEnv(
    Namespace(
        env_id="intersection-v1",
        highway_config={
            "controlled_vehicles": 4,
            "duration": 20,
            "initial_vehicle_count": 12,
            "spawn_probability": 0.4,
            "observation": {
                "observation_config": {
                    "vehicles_count": 9,
                    "features": ["presence", "x", "y", "vx", "vy"],
                }
            },
            "action": {
                "action_config": {
                    "target_speeds": [0, 4.5, 9],
                }
            },
        },
    )
)
```

适配器会在调用 highway-env 前，把 `highway_config` 递归合并到一份完整的
多智能体配置中。这是为了规避 `highway-env==1.11` 的浅合并问题：
highway 内部使用顶层 `config.update(...)`，如果直接传入局部嵌套的
`observation` 或 `action` 配置，会覆盖掉必须保留的默认字段。

在 xuance 中使用：

```python
from argparse import Namespace

from xuance.environment import make_envs
from ca_commappo.envs.highway_intersection import register_highway_intersection_env

register_highway_intersection_env()

envs = make_envs(
    Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-v1",
        env_seed=1,
        parallels=1,
        vectorize="DummyVecMultiAgentEnv",
        distributed_training=False,
        highway_config={
            "controlled_vehicles": 4,
            "duration": 20,
        },
    )
)
```

## Sanity Baseline

默认配置会运行 5 个 base seed，每个 seed 一个 episode，并比较两种非学习策略：

- `random`: 每个 agent 独立均匀采样离散动作。
- `idle-only`: 所有 agent 固定执行动作 `1`，即 highway-env 的 `IDLE`。

summary 指标包括：

- mean episode reward
- mean per-agent reward
- mean episode length
- collision rate
- arrival rate
- truncation rate

协议说明见 [`docs/sanity_baselines.md`](docs/sanity_baselines.md)。

## 关键文档

- [`docs/highway_intersection_mappo_baseline_gap_analysis.md`](docs/highway_intersection_mappo_baseline_gap_analysis.md):
  当前 highway intersection MAPPO baseline 的完成度和缺口。
- [`docs/sanity_baselines.md`](docs/sanity_baselines.md):
  random / idle-only sanity baseline 协议、指标和复现规则。
- [`docs/highway_config_update_bug.md`](docs/highway_config_update_bug.md):
  `highway-env==1.11` 嵌套配置浅合并问题和可工作配置模式。
- [`docs/highway_intersection_notes.md`](docs/highway_intersection_notes.md):
  intersection 动作语义和 reset 初始速度观察记录。
- [`docs/thesis-planning.md`](docs/thesis-planning.md):
  论文和实验主线规划，仅作全局参考。

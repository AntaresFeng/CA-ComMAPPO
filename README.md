# CA-ComMAPPO

用于让 `xuance` 适配 `highway-env` 交叉路口环境的工具项目。

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

运行快速检查：

```powershell
uv run pytest -q
uv run python main.py
```

# CA-ComMAPPO

用于把 `highway-env` 交叉路口环境接到 XuanCe 多智能体接口上的本地研究仓库。

## 本地运行

运行 sanity baseline：

```powershell
uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy all
```

调试单个 highway intersection episode：

```powershell
uv run python -m ca_commappo.envs.debug_highway_wrapper --target wrapper --seed 7 --max-steps 1
```

运行 highway intersection MAPPO 小参数烟测：

```powershell
uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection_v1_smoke.yaml --mode train --no-save
```

也可以通过顶层 dispatcher 调用：

```powershell
uv run python main.py sanity --config configs/sanity/highway_intersection.yaml --policy all
uv run python main.py debug-wrapper --target wrapper --seed 7 --max-steps 1
uv run python main.py mappo --config configs/mappo/intersection_v1_smoke.yaml --mode train --no-save
```

## 项目结构

- `ca_commappo/envs/`: highway-env 到 XuanCe 多智能体接口的环境适配器，以及 wrapper 端到端调试工具。
- `ca_commappo/evaluation/`: sanity baseline 执行和 summary 逻辑。
- `ca_commappo/cli/`: 本项目维护的普通命令行入口，例如 sanity baseline 运行入口。
- `ca_commappo/training/`: 本项目维护的训练入口。
- `configs/sanity/`: sanity baseline 配置。
- `configs/mappo/`: highway intersection MAPPO 配置。
- `examples/mappo/`: vendored XuanCe MAPPO 上游参考示例，仅作参考，不放本项目维护代码。
- `docs/`: 设计说明、排查记录和实验协议。
- `tests/`: pytest 测试。

## Highway `intersection-v1` 的 xuance 适配器

项目提供 `HighwayIntersectionMultiAgentEnv`，它是基于 `gym.make("intersection-v1", ...)` 的 xuance `RawMultiAgentEnv` 适配器。 所有 highway 环境参数都通过 `highway_config` 配置，包括 `controlled_vehicles`。

适配器会在调用 highway-env 前，把 `highway_config` 递归合并到一份完整的多智能体配置中。这是为了规避 `highway-env==1.11` 的浅合并问题：highway 内部使用顶层 `config.update(...)`，如果直接传入局部嵌套的 `observation` 或 `action` 配置，会覆盖掉必须保留的默认字段。

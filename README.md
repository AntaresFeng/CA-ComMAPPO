# CA-ComMAPPO

用于把 `highway-env` 交叉路口环境接到 XuanCe 多智能体接口上的本地研究仓库。

## 本地运行

（项目重构中，等稳定后写）

## 项目结构

（项目重构中，等稳定后写）

## Highway `intersection-v1` 的 xuance 适配器

项目提供 `HighwayIntersectionMultiAgentEnv`，它是基于 `gym.make("intersection-v1", ...)` 的 xuance `RawMultiAgentEnv` 适配器。 所有 highway 环境参数都通过 `highway_config` 配置，包括 `controlled_vehicles`。

适配器会在调用 highway-env 前，把 `highway_config` 递归合并到一份完整的多智能体配置中。这是为了规避 `highway-env==1.11` 的浅合并问题：highway 内部使用顶层 `config.update(...)`，如果直接传入局部嵌套的 `observation` 或 `action` 配置，会覆盖掉必须保留的默认字段。

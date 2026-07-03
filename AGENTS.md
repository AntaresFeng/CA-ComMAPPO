# 仓库指南

## 项目结构与模块组织

用于将 `highway-env` 交叉路口环境适配到 XuanCe 多智能体接口的包。核心代码位于 `ca_commappo/`，当前适配器在 `ca_commappo/envs/highway_intersection.py`。直接调试 highway wrapper 的端到端工具也放在 `ca_commappo/envs/`，当前为 `ca_commappo/envs/debug_highway_wrapper.py`。普通命令行入口放在 `ca_commappo/cli/`，训练入口放在 `ca_commappo/training/`。实验配置放在 `configs/`，其中 `configs/sanity/` 存放 sanity baseline 配置，`configs/mappo/` 存放 highway MAPPO 配置。测试放在 `tests/`；测试夹具和示例配置也应放在这里，例如 `tests/default_config.json`。设计说明、排查记录和实施计划放在 `docs/`。

`examples/mappo/` 保存的是本地 vendored XuanCe MAPPO 参考示例，只做上游参考。不要把本项目实际维护的脚本或配置放回 `examples/`。

当前仓库已经具备：

- `intersection-multi-agent-v1` 的 XuanCe `RawMultiAgentEnv` 适配器。
- XuanCe 环境注册和 `make_envs(...)` 基础兼容。
- 适配器 reset / step / state 测试。
- random / idle-only sanity baseline 和基础评估指标。
- highway intersection 单 episode 调试工具。
- 本地 vendored XuanCe `examples/mappo` 参考脚本。
- highway intersection MAPPO 训练入口、正式 YAML 配置和小参数烟测配置。

当前还不等价于完整 MAPPO baseline。缺口主要是正式多 seed 训练、训练后评估结果归档和实验说明。

## 本地运行方式

这是本地实验研究仓库，代码和配置会频繁变动，不按完整项目发布流程管理。日常运行脚本时使用 `uv run` 即可，例如 `uv run python <your_script>.py`。不要 `uv build`，不要把发布构建作为常规步骤。`uv run pytest -q` 只在重要代码改动、准备提交或需要验证适配器行为时运行。

## 测试指南

测试框架为 `pytest`。测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。普通参数实验不要求每次运行测试；修改适配器核心逻辑、准备提交或排查回归时，再运行 `uv run pytest -q`。新增测试应覆盖配置深度合并、校验错误、智能体数量、动作/观测空间形状以及 reset/step 兼容性。环境测试优先使用确定性 seed，并在 `finally` 块中关闭环境。

新增或修改 sanity baseline 时，优先覆盖配置加载、策略动作选择、summary 指标和最小真实环境 episode。默认命令是 `uv run python -m ca_commappo.cli.run_sanity_baseline --config configs/sanity/highway_intersection.yaml --policy all`，需要持久化结果时加 `--output results/<name>.json`。

## 提交与 Pull Request 指南

提交前 `ruff check --fix; ruff format`。当前历史使用 Conventional Commit 风格，例如 `feat: add highway intersection xuance adapter`。如果提交代码，继续使用简洁前缀，如 `feat:`、`fix:`、`docs:`、`test:`。本仓库以本地研究迭代为主，提交说明只需清楚描述本次实验或行为变化；有运行测试时再记录对应命令。

## 配置提示

所有 `highway-env` 选项应通过 `highway_config` 传入。处理嵌套的 `observation` 和 `action` 字典时要谨慎：本项目会在调用 `gym.make` 前进行深度合并，以保留必要默认值。

`intersection-multi-agent-v1` 当前多智能体动作空间是离散的 `Discrete(3)`，动作语义为 `0=SLOWER`、`1=IDLE`、`2=FASTER`。配置 MAPPO 时应使用离散动作 policy，例如 `Categorical_MAAC_Policy`。裸 `intersection-v1` 仍是单智能体连续动作环境；只在显式兼容测试或排查 highway-env 配置行为时使用。

## 资料源

- [xuance GitHub 仓库](https://github.com/agi-brain/xuance): 最新最全，仓库里的 examples/mappo 已经放在本地，可参考
- [xuance 中文文档](https://cn.xuance.org/): 可以参考，但更新不及时
- [论文计划](./docs/thesis-planning.md): 论文和实验主线规划，仅作全局参考。
- [Sanity baseline 协议](./docs/sanity_baselines.md): random / idle-only baseline 的命令、指标和复现规则

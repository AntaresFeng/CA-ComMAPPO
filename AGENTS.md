# 仓库指南

## 项目结构与模块组织

这是一个用于将 `highway-env` 交叉路口环境适配到 XuanCe 多智能体接口的小型 Python 包。核心代码位于 `ca_commappo/`，当前适配器在 `ca_commappo/envs/highway_intersection.py`。sanity baseline 逻辑位于 `ca_commappo/evaluation/sanity_baselines.py`。测试放在 `tests/`；测试夹具和示例配置也应放在这里，例如 `tests/default_config.json`。`main.py` 是快速冒烟检查和演示入口，当前支持 `debug`、`sanity` 和 `smoke` 子命令。设计说明、排查记录和实施计划放在 `docs/`。

`examples/mappo/` 保存的是本地 vendored XuanCe MAPPO 参考示例，可参考 runner 和配置形状，但不要把它当成已经完成的 highway intersection MAPPO baseline。当前可直接运行的非学习 baseline 入口是 `examples/random_highway_intersection.py`，默认配置在 `examples/sanity/highway_intersection.yaml`。

## 本地运行方式

这是本地实验研究仓库，代码和配置会频繁变动，不按完整项目发布流程管理。日常运行脚本时使用 `uv run` 即可，例如 `uv run python main.py --help`、`uv run python main.py smoke`、`uv run python main.py sanity --policy random`，或 `uv run python <your_script>.py`。不要 `uv build`，不要把发布构建作为常规步骤，只有明确需要产物时才考虑。`uv run pytest -q` 只在重要代码改动、准备提交或需要验证适配器行为时运行。

## 代码风格与命名约定

使用 Python 3.10+ 语法、4 空格缩进，并遵循 PEP 8 命名：函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用大写形式，例如 `DEFAULT_ENV_NAME`。公共辅助函数和适配器方法优先添加类型标注。适配器方法应保持短小明确，尤其是 `highway-env` 的元组空间与 XuanCe 的智能体字典之间的转换逻辑。不要提交 `.venv/`、`.pytest_cache/`、`.ruff_cache/`、`dist/` 或 `*.egg-info/` 等生成内容。

## 测试指南

测试框架为 `pytest`，`pyproject.toml` 中已配置 `pythonpath = ["."]`。测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。普通参数实验不要求每次运行测试；修改适配器核心逻辑、准备提交或排查回归时，再运行 `uv run pytest -q`。新增测试应覆盖配置深度合并、校验错误、智能体数量、动作/观测空间形状以及 reset/step 兼容性。环境测试优先使用确定性 seed，并在 `finally` 块中关闭环境。

新增或修改 sanity baseline 时，优先覆盖配置加载、策略动作选择、summary 指标和最小真实环境 episode。默认命令是 `uv run python examples/random_highway_intersection.py --config examples/sanity/highway_intersection.yaml --policy all`，需要持久化结果时加 `--output results/<name>.json`。

## Baseline 状态约定

当前仓库不能宣称已经完成 highway intersection MAPPO baseline。已经完成的是环境适配、XuanCe 注册、基础兼容测试，以及 random / idle-only sanity baseline。称为完整 MAPPO baseline 前，至少需要 highway 专用 MAPPO runner、YAML 配置、小步训练烟测、训练后评估和 README 复现实验命令。

评估 MAPPO 或后续 CA-ComMAPPO 时，不要只报告 reward。至少同时记录 mean episode reward、mean per-agent reward、collision rate、arrival/pass rate、episode length 和 truncation/timeout rate。正式结果应保存运行命令、配置、seed 列表、git commit 或工作树说明。

## 提交与 Pull Request 指南

当前历史使用 Conventional Commit 风格，例如 `feat: add highway intersection xuance adapter`。如果提交代码，继续使用简洁前缀，如 `feat:`、`fix:`、`docs:`、`test:`。本仓库以本地研究迭代为主，提交说明只需清楚描述本次实验或行为变化；有运行测试时再记录对应命令。

## 配置提示

所有 `highway-env` 选项应通过 `highway_config` 传入。处理嵌套的 `observation` 和 `action` 字典时要谨慎：本项目会在调用 `gym.make` 前进行深度合并，以保留必要默认值。

`intersection-v1` 当前多智能体动作空间是离散的 `Discrete(3)`，动作语义为 `0=SLOWER`、`1=IDLE`、`2=FASTER`。配置 MAPPO 时应使用离散动作 policy，例如 `Categorical_MAAC_Policy`，不要直接沿用 MPE 连续动作示例里的 Gaussian policy。

已到达或已终止 agent 的后续 transition 语义仍是重要未完成点。后续修改适配器时，要明确处理 terminal agent 的 `agent_mask` 和 reward，避免把无效后续样本继续送入 MAPPO loss。

## 资料源

- [xuance GitHub 仓库](https://github.com/agi-brain/xuance): 最新最全，仓库里的 examples/mappo 已经放在本地，可参考

- [xuance 中文文档](https://cn.xuance.org/): 可以参考，但更新不及时

- [论文计划](./docs/thesis-planning.md): 全局规划，仅参考

- [Baseline 缺口分析](./docs/highway_intersection_mappo_baseline_gap_analysis.md): 判断当前工作是否已经达到可训练、可复现、可评估 baseline 时先读这里

- [Sanity baseline 协议](./docs/sanity_baselines.md): random / idle-only baseline 的命令、指标和复现规则

# 仓库指南

## 项目结构与模块组织

这是一个用于将 `highway-env` 交叉路口环境适配到 XuanCe 多智能体接口的小型 Python 包。核心代码位于 `ca_commappo/`，当前适配器在 `ca_commappo/envs/highway_intersection.py`。测试放在 `tests/`；测试夹具和示例配置也应放在这里，例如 `tests/default_config.json`。`main.py` 是快速冒烟检查和演示入口。设计说明、排查记录和实施计划放在 `docs/`。

## 本地运行方式

这是本地实验研究仓库，代码和配置会频繁变动，不按完整项目发布流程管理。日常运行脚本时使用 `uv run` 即可，例如 `uv run python main.py`，或 `uv run python <your_script>.py`。不要把发布构建作为常规步骤；只有明确需要产物时才考虑。`uv run pytest -q` 只在重要代码改动、准备提交或需要验证适配器行为时运行。

## 代码风格与命名约定

使用 Python 3.10+ 语法、4 空格缩进，并遵循 PEP 8 命名：函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用大写形式，例如 `DEFAULT_ENV_NAME`。公共辅助函数和适配器方法优先添加类型标注。适配器方法应保持短小明确，尤其是 `highway-env` 的元组空间与 XuanCe 的智能体字典之间的转换逻辑。不要提交 `.venv/`、`.pytest_cache/`、`.ruff_cache/`、`dist/` 或 `*.egg-info/` 等生成内容。

## 测试指南

测试框架为 `pytest`，`pyproject.toml` 中已配置 `pythonpath = ["."]`。测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。普通参数实验不要求每次运行测试；修改适配器核心逻辑、准备提交或排查回归时，再运行 `uv run pytest -q`。新增测试应覆盖配置深度合并、校验错误、智能体数量、动作/观测空间形状以及 reset/step 兼容性。环境测试优先使用确定性 seed，并在 `finally` 块中关闭环境。

## 提交与 Pull Request 指南

当前历史使用 Conventional Commit 风格，例如 `feat: add highway intersection xuance adapter`。如果提交代码，继续使用简洁前缀，如 `feat:`、`fix:`、`docs:`、`test:`。本仓库以本地研究迭代为主，提交说明只需清楚描述本次实验或行为变化；有运行测试时再记录对应命令。

## 配置提示

所有 `highway-env` 选项应通过 `highway_config` 传入。处理嵌套的 `observation` 和 `action` 字典时要谨慎：本项目会在调用 `gym.make` 前进行深度合并，以保留必要默认值。

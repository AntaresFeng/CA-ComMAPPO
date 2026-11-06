# CA-ComMAPPO 仓库指南

## 项目目标

本仓库将 highway-env 无信号交叉路口适配到 XuanCe 多智能体接口，并实现
IPPO、MAPPO、Attention-MAPPO、CommNet、IC3Net 与 CA-ComMAPPO 的可复现实验管线。
它是本地研究仓库，不按 Python 发布包流程管理；不要执行 `uv build`。

## 权威资料

开始任务前先从 [`docs/README.md`](docs/README.md) 选择相关专题。重要口径：

- [`docs/environment.md`](docs/environment.md)：环境、动作、reward、终止和指标事实源。
- [`docs/mappo_baseline.md`](docs/mappo_baseline.md)：后续对比统一使用的任务与 MAPPO 合同。
- [`docs/ca_commappo_design.md`](docs/ca_commappo_design.md)：创新算法定义及正式训练阻断项。
- [`docs/experiments_runbook.md`](docs/experiments_runbook.md)：smoke、正式训练、验证、视频与聚合。
- [`docs/thesis-planning.md`](docs/thesis-planning.md)：论文实验矩阵和剩余证据。

`docs/archived/` 只用于历史追溯，不得从中复制当前配置、命令或实验数字。若文档与
canonical YAML 冲突，参数以 YAML 为准；若文档与实现冲突，先核对代码和测试，并在
同一变更中修正文档。

## 目录职责

- `ca_commappo/envs/`：highway-env 适配器与环境调试入口。
- `ca_commappo/algorithms/`：自定义 representation、通信模块和运行时注册/兼容补丁。
- `ca_commappo/training/`：统一学习算法训练、benchmark 和 test 入口。
- `ca_commappo/evaluation/`：共享指标、artifact、provenance、视频与结果聚合。
- `ca_commappo/cli/`：sanity baseline 等普通命令行入口。
- `configs/common.yaml`：共享训练合同；`configs/highway/`：共享任务合同；
  `configs/<algorithm>/`：只保留算法差异。
- `tests/`：pytest 行为和回归测试。
- `results/`：受 Git 跟踪的审定 manifest、聚合表和小型 provenance 摘要。
- `runs/`：被 Git 忽略的 checkpoint、日志、视频和原始评估 artifact。
- `examples/mappo/`：vendored XuanCe 上游参考，禁止放项目维护代码。

## 开发命令

日常使用 `uv run`：

```bash
uv run python main.py sanity --config configs/sanity/highway_intersection.yaml --policy all
uv run python main.py debug-wrapper --target wrapper --seed 7 --max-steps 1
uv run python main.py mappo \
  --config configs/mappo/ca-intersection-v0.yaml \
  --smoke --mode train --no-save
uv run pytest -q
```

提交前运行：

```bash
ruff check --fix
ruff format
uv run pytest -q
```

若工具在当前环境缺失，应如实记录，不要修改 `.venv` 或依赖锁来绕过一次文档任务。

## 实现原则

- 用户意图清楚时直接实现，保持改动小、可读且符合现有结构。
- 不编辑 `.venv` 或 XuanCe site-packages；自定义组件通过仓库代码运行时注册或补丁接入。
- 所有 highway-env 选项通过 `highway_config` 传入；嵌套 `observation` / `action`
  由项目环境子类递归合并；不再重建上游默认配置。
- `ca-intersection-v0` 使用 `Discrete(3)`：`0=SLOWER`、`1=IDLE`、
  `2=FASTER`。离散 MARL policy 使用 categorical action distribution。
- `step()` 的 `rewards` 与 `info["agents_rewards"]` 是 masked adapter-facing reward；
  原始 per-agent reward 只在 `info["raw_agents_rewards"]`。
- best checkpoint 排序固定为 `arrival↑ → collision↓ → reward↑ → length↓`。改变该键
  会破坏跨实验可比性，除非用户明确批准新协议并重跑全部对照。
- 公共算法/环境合同改变时，同步更新 canonical 配置、相关文档和 provenance 校验。

## 测试与注释

- 普通参数实验无需新增测试。修改核心适配器、算法语义、评估协议或准备提交时运行完整
  `uv run pytest -q`。
- 测试只保护可观察行为、非平凡不变量、边界或真实回归。优先扩展现有行为边界，避免
  只镜像常量、实现细节或已删除功能的 tombstone 测试。
- 环境测试使用确定性 seed，并在 `finally` 中关闭环境。涉及通信算法时至少覆盖真实
  action/eval 路径，而不只验证张量形状。
- 注释解释不明显的原因、外部兼容约束或安全不变量；不要复述符号名和直观控制流。

## 正式实验规范

- 正式训练必须设置 `WANDB_MODE=offline`、使用 `--mode benchmark`，并在独立具名 tmux
  session 中运行。烟测和临时调试不是正式实验。
- 每个算法/seed 使用独立 run；训练开始后冻结代码、配置和依赖锁，直到 final test
  完成。不要覆盖已有 `eval_metadata.json`。
- 最终数字只取带严格 provenance 的 `best_model.pth`，在固定 100 cases 上执行独立
  `--mode test`。benchmark eval 只用于 checkpoint selection。
- `--allow-unverified-model` 只用于诊断旧模型，不得生成论文结果。
- 不提交 `runs/`。提交 `results/` 前先确认文件是小型、可重算且经过人工审定的结果，
  不要用 `git add -f` 绕过忽略规则。
- 每完成一个实现、实验或文档整理阶段就创建一次聚焦的 Git 提交；使用 Conventional
  Commit，例如 `feat:`、`fix:`、`experiment:`、`docs:`、`test:`、`refactor:`、`chore:`。

## 工作树与文档纪律

- 用户已有改动属于用户；保留无关修改，不做 destructive reset/checkout。
- 当前规范放在 `docs/` 根目录，每个文件只解释一个主题。历史决策放
  `docs/archived/`，普通重复材料直接删除并交由 Git 历史追溯。
- 新命令只有在代码已经支持且至少完成 smoke 后才能写入 runbook。未实现的研究工作只
  写在 thesis planning，不能伪装成可执行流程。
- 实验文档记录 YAML、完整命令、Git commit、seed、设备、artifact 路径和偏离协议之处。

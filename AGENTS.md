# CA-ComMAPPO 仓库指南

## 当前方向与事实源

本仓库研究无信号路口多智能体协作，使用 highway-env 1.12.1、项目环境
`ca-intersection-v0` 与 XuanCe。当前只考虑理想通信。

**旧论文计划、固定四车论文范围、旧 H1–H6 假设及 intent/TTC 算法设计已作废。**
CA-ComMAPPO full/no-prior 代码保留为遗留实现，不是新算法的规定方案；不得从历史
报告、提示词或现有变量名恢复这些设计约束。新论文问题与算法尚未确定。

开始任务先读 [文档导航](docs/README.md)，按需查看：

- [实现现状](docs/current_experiment_status.md)：已有能力、遗留代码、未解决问题。
- [环境说明](docs/environment.md)：观测、动作、奖励、终止、mask 与指标。
- [实验手册](docs/experiments_runbook.md)：当前命令、评估与 provenance 行为。
- [历史结果](results/README.md)：旧任务证据，不作为新环境的基线成绩。

参数以 YAML 为准，行为以代码和测试为准。发现冲突时核对并同步修正文档。
默认参数和已有评估协议可以随研究调整；变更时说明可比性影响并建立新实验版本，
不要把旧论文范围或文档中的门禁当作研究设计限制。

## 目录与实现

- `ca_commappo/envs/` 管理道路、车辆生命周期和 XuanCe 适配。
- `ca_commappo/algorithms/` 管理 representation、通信模块与运行时注册/补丁。
- `ca_commappo/training/`、`evaluation/`、`cli/` 管理训练、共享评估与命令入口。
- `configs/common.yaml` 是共享训练默认值；`configs/highway/` 是环境参数；算法目录
  只存差异。所有环境选项通过 `highway_config` 传入，嵌套配置由环境子类递归合并。
- `tests/` 保存行为与回归测试；`examples/mappo/` 是 vendored 上游参考，不放项目代码。
- 不编辑 `.venv` 或 site-packages；通过仓库子类、注册或兼容补丁接入上游。
  本仓库不是发布包，不执行 `uv build`。
- 保留到达后的物理退出、固定 agent 槽位与 mask 语义。评估读取 adapter 的逐车
  `info`；masked reward 与 `raw_agents_rewards` 的区别见环境说明。
- 用户意图明确时直接实现，改动保持聚焦；保护用户已有修改，不做 destructive reset。

## 验证

常用短检查见根目录 README。提交前执行：

```bash
ruff check --fix
ruff format
uv run pytest -q
```

测试保护可观察行为、非平凡不变量和真实回归，优先扩展已有测试；不要为普通参数、
字面值、删除项或覆盖率单独增加测试。环境测试使用固定 seed，并在 `finally` 中关闭。
通信改动应覆盖真实 action/eval 路径。注释解释原因与边界，不复述代码。
工具缺失时如实记录，不为文档任务修改依赖或虚拟环境。

## 实验与文档维护

- 正式训练使用 `WANDB_MODE=offline`、`--mode benchmark`，长任务通过具名 tmux
  session 启动；每个算法/seed 独立 run，并记录配置、commit、设备、命令和产物路径。
- 当前正式加载与聚合有源码、训练合同和 checkpoint provenance 校验，具体见实验手册。
  同一轮 benchmark/final test 使用匹配版本；调整协议时同步配置、校验和对照。
- smoke、模型选择期评估和 `--allow-unverified-model` 诊断不能冒充正式 final test。
  不覆盖已有 metadata，不把新旧任务结果混为同一实验。
- `runs/` 不提交；`results/` 仅保存人工审定的小型可追溯结果，不用 `git add -f`
  绕过忽略规则。每个实现、实验或文档阶段创建聚焦的 Conventional Commit。
- 当前文档按主题维护，避免重复参数表和多套状态清单。失效计划、提示词与重复教程
  直接删除，使用 Git 历史追溯；历史结果统一从 `results/README.md` 进入。
- 只把代码已支持且经过烟测的命令写入操作手册。尚未实现的研究想法明确标为待设计，
  不包装成既定算法、完成能力或已证实的论文结论。

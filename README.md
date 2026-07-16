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
uv run python -m ca_commappo.training.mappo_highway_intersection --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode train --no-save
```

也可以通过顶层 dispatcher 调用：

```powershell
uv run python main.py sanity --config configs/sanity/highway_intersection.yaml --policy all
uv run python main.py debug-wrapper --target wrapper --seed 7 --max-steps 1
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode train --no-save
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

## Highway `intersection-multi-agent-v1` 的 xuance 适配器

项目提供 `HighwayIntersectionMultiAgentEnv`，它默认基于 `gym.make("intersection-multi-agent-v1", ...)` 的 xuance `RawMultiAgentEnv` 适配器。所有 highway 环境参数都通过 `highway_config` 配置，包括 `controlled_vehicles`。

适配器会在调用 highway-env 前，把 `highway_config` 递归合并到一份完整的多智能体配置中。这是为了规避 `highway-env==1.11` 的浅合并问题：highway 内部使用顶层 `config.update(...)`，如果直接传入局部嵌套的 `observation` 或 `action` 配置，会覆盖掉必须保留的默认字段。

适配器的 `step()` 返回值采用 XuanCe 训练侧语义：返回的 `rewards` 字典和 `info["agents_rewards"]` 都是经过 active-agent mask 处理后的 adapter-facing reward。已经终止的 agent 在后续 transition 中 reward 记为 `0.0`，并且动作会被替换为 `IDLE_ACTION`。如果需要查看 highway-env 原始 per-agent reward，使用 `info["raw_agents_rewards"]`。环境级结束信号写在 `info["global_terminated"]`，碰撞状态写在 per-agent tuple 形式的 `info["crashed"]`。

## MAPPO 模型验证与视频录制

完整使用说明见
[MAPPO 模型验证与视频录制](docs/mappo_model_evaluation_and_video.md)，包括
checkpoint 与配置匹配、正式验证指标、视频参数、产物目录和复现规则。

每次 MAPPO 命令默认都会在 `runs/mappo_highway/` 下创建独立的运行目录：

```text
runs/mappo_highway/<timestamp>_<mode>_seed_<seed>/
|-- eval_metadata.json       # test/benchmark only
|-- eval_records.jsonl       # test/benchmark only
|-- wandb/                   # local W&B run files
|-- models/
|   |-- final_train_model.pth  # train
|   `-- best_model.pth         # benchmark
`-- videos/                  # test --record-video
```

仅在需要修改运行根目录时使用 `--output-dir`。每次运行的子目录、log 目录、
模型目录和默认视频目录都会自动生成。同一个 run ID 也会作为 W&B 显示名称并写入
W&B 配置。

验证训练完成的 checkpoint 并录制代表性视频：

```powershell
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1.yaml --mode test --model-dir-load runs/mappo_highway/<benchmark_run>/models/best_model.pth --test-episode 100 --record-video --video-episodes 6
```

默认视频数量为 `min(test_episode, 6)`。视频保存在本次 test run 的 `videos/`
目录，同时生成 `video_eval_summary.json`、逐 episode MP4、缩略图总览和合并
MP4。正式评估结果保存在 run 根目录的 `eval_metadata.json` 和
`eval_records.jsonl`。

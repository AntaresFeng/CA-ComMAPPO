# MAPPO 模型验证与视频录制

本文说明如何加载训练完成的 MAPPO checkpoint，在 highway intersection
环境中执行正式验证，并录制策略运行视频。

## 入口

推荐通过项目顶层命令运行：

```bash
uv run python main.py mappo [参数]
```

它会将参数转发给
`ca_commappo.training.mappo_highway_intersection`。模型验证使用
`--mode test`；视频录制是在正式验证结束后执行的可选步骤，只能与
`--mode test` 一起使用。

## 选择 checkpoint 和配置

验证时应使用训练该 checkpoint 时的同一份 YAML 配置。网络结构、观测配置、
智能体数量或动作空间不一致时，模型可能无法加载，或者验证结果不再具有可比性。

可以从训练运行目录的 `eval_metadata.json` 中查看：

- `config.config_path`：训练使用的 YAML。
- `config.seed`：训练 seed。
- `config.highway_config`：环境参数快照。
- `config.policy`、网络隐藏层和观测配置：模型结构信息。

仓库中的模型通常有两种：

- `benchmark` 产生的 `models/best_model.pth`：按照 highway 任务指标选出的最佳
  checkpoint，推荐用于最终验证。
- `train` 产生的 `models/final_train_model.pth`：训练最后一步保存的 checkpoint，
  不保证是训练过程中的最佳模型。

## 只执行正式验证

```bash
uv run python main.py mappo \
  --config configs/mappo/intersection-multi-agent-v1.yaml \
  --mode test \
  --model-dir-load runs/mappo_highway/<benchmark_run>/models/best_model.pth \
  --test-episode 100
```

`--test-episode` 指定正式验证的 episode 数量。程序会加载模型，以
`test_mode=True` 运行确定性策略，并在终端打印平均分数和 highway 任务指标。

每次命令都会自动创建独立目录：

```text
runs/mappo_highway/<timestamp>_test_seed_<seed>/
|-- eval_metadata.json
`-- eval_records.jsonl
```

其中：

- `eval_metadata.json` 保存配置、模型和环境相关的运行快照。
- `eval_records.jsonl` 每行是一次完整评估记录，包含 summary 和逐 episode
  结果，适合后续脚本分析。

正式验证 summary 包括：

- `mean_episode_reward`：平均 episode reward。
- `mean_agent_reward`：所有 agent 的平均累计 reward。
- `mean_episode_length`：平均 episode 长度。
- `collision_rate`：发生任一受控车辆碰撞的 episode 比例。
- `arrival_rate`：到达目标的 episode 比例。
- `truncation_rate`：达到时间限制而截断的 episode 比例。
- `mean_agent_collision_fraction`：每个 episode 中发生碰撞的 agent 平均占比。
- `mean_agent_arrival_fraction`：每个 episode 中到达目标的 agent 平均占比。

## 正式验证并录制视频

在验证命令后添加 `--record-video`：

```bash
uv run python main.py mappo \
  --config configs/mappo/intersection-multi-agent-v1.yaml \
  --mode test \
  --model-dir-load runs/mappo_highway/<benchmark_run>/models/best_model.pth \
  --test-episode 100 \
  --record-video \
  --video-episodes 6 \
  --video-seed 10042
```

执行顺序是：

1. 使用向量化环境完成 `--test-episode` 指定的正式验证。
2. 创建单环境 `rgb_array` 录制环境。
3. 从 `--video-seed` 开始，每个录制 episode 的 seed 依次加一。
4. 保存逐 episode MP4、视频指标、缩略图总览和合并视频。

录制视频是正式验证之外的独立 rollout。因此上例正式验证 100 个 episode，
随后额外运行 6 个用于视频展示的 episode；视频 summary 不应替代 100 轮正式验证
结果。

默认视频目录是本次 test run 下的 `videos/`：

```text
runs/mappo_highway/<timestamp>_test_seed_<seed>/
|-- eval_metadata.json
|-- eval_records.jsonl
`-- videos/
    |-- mappo_test-episode-0.mp4
    |-- mappo_test-episode-1.mp4
    |-- ...
    |-- mappo_test_contact_sheet.jpg
    |-- mappo_test_all_episodes.mp4
    `-- video_eval_summary.json
```

`video_eval_summary.json` 包含视频 rollout 的 seed、逐 episode 指标、汇总指标、
checkpoint 路径和全部视频文件路径。

## 视频参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--record-video` | 在正式 test 后录制视频 | 不录制 |
| `--video-episodes N` | 录制 N 个 episode，最多不超过 `test_episode` | `min(test_episode, 6)` |
| `--video-episodes all` | 录制与 `test_episode` 相同数量的视频 | - |
| `--video-seed N` | 视频 rollout 的起始 seed | 配置中的 `seed` |
| `--video-dir PATH` | 自定义视频和视频 summary 输出目录 | 当前 test run 的 `videos/` |
| `--no-video-contact-sheet` | 不生成 JPEG 缩略图总览 | 生成 |
| `--no-combined-video` | 不生成拼接后的总 MP4 | 生成 |

例如，只保存两个独立视频，不生成额外可视化产物：

```bash
uv run python main.py mappo \
  --config <训练时配置.yaml> \
  --mode test \
  --model-dir-load <checkpoint.pth> \
  --test-episode 20 \
  --record-video \
  --video-episodes 2 \
  --no-video-contact-sheet \
  --no-combined-video
```

## Seed 与复现

- `--seed` 控制本次 test 的全局随机种子，并用于 run ID；正式向量评估环境的
  `env_seed` 来自 YAML 配置。
- `--video-seed` 单独控制录制 rollout，便于不同模型在完全相同的交通场景下进行
  视频对比。
- 第 `i` 个视频使用 `video_seed + i`，从 0 开始计数。
- 做模型横向对比时，应固定 YAML（包括 `env_seed`）、`test_episode`、`--seed` 和
  `--video-seed`。

训练 seed 不会因为加载 checkpoint 而自动恢复。若希望 test run 的命名和随机设置
也对应训练 seed，应显式传入 `--seed <训练 seed>`。

## 常见问题

### 模型加载时报维度或参数不匹配

优先检查 checkpoint 运行目录的 `eval_metadata.json`，确认验证命令使用了相同
YAML。尤其检查 `controlled_vehicles`、observation features、隐藏层大小、
parameter sharing 和 global state 设置。

### 为什么测试 100 轮却只有 6 个视频

`--test-episode 100` 控制正式统计样本数；视频默认最多录制 6 轮。使用
`--video-episodes all` 才会录制全部 100 轮，但通常会消耗较多时间和磁盘空间。

### 视频速度不合适

配置中的 `fps` 控制输出视频帧率。highway-env 会通过 `RecordVideo` wrapper
提供中间仿真帧，避免只按低频策略 step 生成画面。

### 可以在 train 或 benchmark 模式直接录制吗

不可以。`--record-video` 当前仅支持 `--mode test`。先完成训练或 benchmark，
再用保存的 checkpoint 单独执行 test。

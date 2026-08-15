# 实验执行手册

本文只描述当前代码已经支持的可复现实验流程。任务与 MAPPO 超参以
[`mappo_baseline.md`](mappo_baseline.md) 为准；研究矩阵和尚未实现的实验见
[`thesis-planning.md`](thesis-planning.md)。

## 不可变规则

- 正式训练必须使用 `--mode benchmark`，并通过环境变量设置
  `WANDB_MODE=offline`。`--mode train` 不能产生 checkpoint-selection 证据。
- 长训练在独立具名 tmux session 中运行；一个 session 只跑一个算法和 seed。
- 显式传 `--config` 和 `--seed`。算法由 YAML 的 `agent` 字段决定，`main.py mappo`
  只是统一路由入口。
- 训练 seed `N` 的环境 seed block 固定为 `N*1,000,000 + worker_rank`；checkpoint
  与 final evaluation 分别保留 `10000..10049` 和 `20000..20099`。
- 正式训练启动后冻结代码、配置、`pyproject.toml` 和 `uv.lock`，直至对应 final test
  完成。每完成一个实现或实验阶段创建一次 Git 提交。
- 非 smoke 的 benchmark/test 在创建 run 前检查 Git tracked tree；存在 staged 或
  unstaged 修改时直接拒绝运行。被忽略的 `runs/` 产物不影响该门禁。
- `runs/` 保存原始大体积 artifact 并被 Git 忽略；`results/` 只保存人工审定的
  manifest、聚合 CSV 和小型 provenance 摘要，受 Git 跟踪。

建议为所有命令设置可写缓存：

```bash
export WANDB_MODE=offline
export UV_CACHE_DIR=/tmp/ca-commappo-uv-cache
export MPLCONFIGDIR=/tmp/ca-commappo-mpl-cache
export PYTHONUNBUFFERED=1
```

## 配置矩阵

| 论文名称 | 配置 | 说明 |
| --- | --- | --- |
| IPPO | `configs/ippo/intersection-multi-agent-v1.yaml` | 无 centralized critic |
| MAPPO | `configs/mappo/intersection-multi-agent-v1.yaml` | canonical MLP baseline |
| Attention-MAPPO | `configs/attention_mappo/intersection-multi-agent-v1.yaml` | PI social encoder |
| CommNet | `configs/commnet/intersection-multi-agent-v1.yaml` | 平均通信、RNN backbone |
| IC3Net | `configs/ic3net/intersection-multi-agent-v1.yaml` | 门控通信、RNN backbone |
| CA-ComMAPPO | `configs/ca_commappo/intersection-multi-agent-v1.yaml` | 完整方法；当前有设计债务 |
| CA-ComMAPPO no-prior | `configs/ca_commappo/intersection-multi-agent-v1-no-prior.yaml` | `b_ij=0` 消融 |

CA-ComMAPPO 正式训练前必须先处理其设计文档列出的阻断项。CommNet / IC3Net 的
backbone 与 CA-ComMAPPO 不同，比较时不得把全部差异归因于通信聚合。

## 1. Smoke 门禁

所有算法先跑构造、rollout 和反向更新：

```bash
WANDB_MODE=offline uv run python main.py mappo \
  --config configs/<algorithm>/intersection-multi-agent-v1.yaml \
  --smoke --mode train --no-save
```

对 CommNet、IC3Net 和 CA-ComMAPPO，再用 smoke benchmark 产 checkpoint，并完成
一次 test，以覆盖通信 action/eval 路径：

```bash
WANDB_MODE=offline uv run python main.py mappo \
  --config configs/<algorithm>/intersection-multi-agent-v1.yaml \
  --smoke --mode benchmark

WANDB_MODE=offline uv run python main.py mappo \
  --config configs/<algorithm>/intersection-multi-agent-v1.yaml \
  --smoke --mode test --seed 1 \
  --model-dir-load runs/<output>/<benchmark-run>/models/best_model.pth \
  --test-episode 1
```

smoke 只验证管线，不是正式结果，不进入论文 manifest。

## 2. 正式 benchmark

论文最低训练 seed 集为 `[1,2,3]`；若预算允许扩展到 5 个，必须预先声明并对所有
学习算法使用同一集合。单 seed 的 tmux 模板：

```bash
tmux new-session -d -s <algo>-seed1 -c /home/fzx/CA-ComMAPPO \
  'env WANDB_MODE=offline UV_CACHE_DIR=/tmp/ca-commappo-uv-cache \
   MPLCONFIGDIR=/tmp/ca-commappo-mpl-cache PYTHONUNBUFFERED=1 \
   uv run python main.py mappo \
   --config configs/<algorithm>/intersection-multi-agent-v1.yaml \
   --mode benchmark --seed 1 \
   2>&1 | tee runs/launch_logs/<algo>-seed1.log'
```

benchmark 每 10k steps 在固定 50 cases 上确定性评估，以
`arrival↑ → collision↓ → reward↑ → length↓` 选择 `best_model.pth`。完整 run 应包含：

```text
runs/<algorithm>_highway/<timestamp>_benchmark_seed_<seed>/
├── eval_metadata.json
├── eval_records.jsonl
└── models/
    ├── best_model.pth
    ├── best_model_provenance.json
    └── final_model.pth
```

监控时检查日志持续推进、无 traceback/NaN/OOM，并确认 best checkpoint 与 provenance
同时更新。benchmark summary 只用于选模型，不是论文最终数字。

## 3. 独立 final test 与视频

每个训练 seed 使用 best checkpoint 在固定 100 cases 上独立验证：

```bash
WANDB_MODE=offline uv run python main.py mappo \
  --config configs/<algorithm>/intersection-multi-agent-v1.yaml \
  --mode test --seed <training-seed> \
  --model-dir-load runs/<algorithm>_highway/<benchmark-run>/models/best_model.pth \
  --test-episode 100 \
  --record-video --video-episodes 6
```

test 默认使用 deterministic argmax 和 seeds `20000..20099`。视频复用其中前 6 个
seeds，输出逐 episode MP4、合并视频、contact sheet 与 `video_eval_summary.json`；
视频 summary 不能替代 100-episode 数值结果。

provenance v2 sidecar 记录 checkpoint SHA、训练 seed/env seed、训练配置、设备、Git
commit、tracked-tree clean 状态，以及 `main.py`、`pyproject.toml`、`uv.lock` 和
`ca_commappo/**/*.py` 的逐文件 SHA manifest。严格模型加载要求当前 commit、runtime
manifest 和训练合同与 checkpoint 完全一致。

缺少 sidecar 或只有 provenance v1 的旧模型只能显式使用
`--allow-unverified-model` 做诊断；其 test artifact 会标记为 unverified，并被正式
聚合器拒绝。

## 4. 结果审定与聚合

不要用宽泛 glob 直接生成论文表。先人工检查每个 final run，再建立显式 manifest：

```json
{
  "schema_version": 1,
  "eval_records": [
    "runs/mappo_highway/<seed1-test>/eval_records.jsonl",
    "runs/mappo_highway/<seed2-test>/eval_records.jsonl",
    "runs/mappo_highway/<seed3-test>/eval_records.jsonl",
    "runs/random_highway/<final-test>/eval_records.jsonl",
    "runs/idle_only_highway/<final-test>/eval_records.jsonl"
  ],
  "expected_seeds": {
    "mappo": [1, 2, 3],
    "random": [1],
    "idle_only": [1]
  }
}
```

```bash
uv run python -m ca_commappo.evaluation.aggregate_results \
  --runs-manifest results/final_runs_manifest.json \
  --output results/aggregated.csv
```

聚合器只接受不少于 100 episodes 的 final test，重算 episode summary，并校验协议、
seed 集、保留的 checkpoint/sidecar、训练合同和重复 checkpoint。它使用 artifact 内的
v2 manifest 完成历史聚合，不要求聚合命令运行在训练时 checkout；但同一 manifest 中
所有学习算法仍必须来自同一 Git commit 和 runtime manifest。学习策略报告 sample
std；静态 baseline 是固定策略点估计，std 留空。

## 5. 阶段收尾

每一阶段至少记录：

- Git commit、工作树状态和准确命令；
- 算法 YAML、训练 seed 与运行设备；
- benchmark/final run 路径和模型 SHA；
- final summary、视频可播放性和聚合器验证结果；
- 异常、提前终止或偏离协议的说明。

提交前执行适用的格式检查和 `uv run pytest -q`。任何改变任务、奖励、训练合同或评估
协议的实验都必须使用新提交和新 run，不得覆盖既有 metadata。

# 实验手册

本文描述当前工具的行为，不规定新论文的算法矩阵。环境见 [environment.md](environment.md)，
实现状态见 [current_experiment_status.md](current_experiment_status.md)。

## 配置与运行准备

配置按 `configs/common.yaml → configs/highway/ca-intersection-v0.yaml → 算法 YAML →
smoke/CLI` 合并。`main.py mappo` 是统一训练入口，算法由 YAML 的 `agent` 决定。
可用算法目录为 `ippo`、`mappo`、`attention_mappo`、`commnet`、`ic3net`、`ca_commappo`；
CA 配置属于遗留实现。默认超参是研究起点，不是新论文的冻结基线。

```bash
export WANDB_MODE=offline
export UV_CACHE_DIR=/tmp/ca-commappo-uv-cache
export MPLCONFIGDIR=/tmp/ca-commappo-mpl-cache
```

当前默认训练为 1M steps、16 workers；具体 PPO 参数以 `configs/common.yaml` 为准。
设备按运行时实际可用性选择，正式比较记录并统一训练设备，不沿用旧机器状态说明。

## 环境检查与 sanity

```bash
uv run python main.py debug-wrapper --target wrapper --seed 7 --controlled-vehicles 8 --max-steps 5
uv run python main.py sanity --config configs/sanity/highway_intersection.yaml --policy all
```

`random` 为每个 active agent 均匀采样动作；`idle-only` 始终选择 `1=IDLE`（保持目标
速度，不是停车）。random 的动作 RNG 独立于环境 RNG，可重放但不是确定性策略。

需要与学习算法使用相同的 final cases 时：

```bash
uv run python main.py sanity --config configs/sanity/highway_intersection_final.yaml \
  --policy all --artifact-root runs
```

该配置串行评估 seeds `20000..20099`，每个策略生成独立的 metadata 和逐局记录。
静态策略报告点估计，不复制成多个训练 seed 的样本。

## 训练与独立验证

先验证构造、rollout 和 update：

```bash
uv run python main.py mappo --config configs/mappo/ca-intersection-v0.yaml \
  --smoke --mode train --no-save
```

通信算法还需覆盖 benchmark 保存和 test 加载路径；例如旧 CA 的 benchmark 烟测：

```bash
uv run python main.py mappo --config configs/ca_commappo/ca-intersection-v0.yaml \
  --smoke --mode benchmark
```

smoke checkpoint 用相同配置的 `--smoke --mode test --test-episode 1` 加
`--model-dir-load` 验证。smoke 产物不进入正式结果。

正式训练使用 benchmark。以下为 MAPPO seed 1 的命令，设备按实际情况设置：

```bash
uv run python main.py mappo --config configs/mappo/ca-intersection-v0.yaml \
  --mode benchmark --seed 1 --device cpu
```

长任务将完整命令、环境变量、日志路径和退出状态写入 `runs/launch_scripts/` 的脚本，
检查 `bash -n` 后通过独立具名 tmux session 启动。每个算法/seed 独立 run；同一轮训练
与 final test 使用同一冻结代码、配置和依赖。训练 seed 集由新实验预先确定。

当前默认评估协议：

| 阶段 | 环境 seed / 行为 |
| --- | --- |
| 训练 | 训练 seed N 对应 `N × 1,000,000 + worker_rank` |
| 模型选择 | 每 10k steps，50 cases：`10000..10049` |
| 独立 final test | 100 cases：`20000..20099`，deterministic argmax |
| 视频 | 默认复用 final cases 的前 6 个 |

best 排序为 `arrival↑ → collision↓ → reward↑ → length↓`，由共享评估器实现。
benchmark 结果用于选模型，最终成绩来自 best checkpoint 的独立 test。
从运行输出取得真实模型路径，再执行以下模板：

```bash
uv run python main.py mappo --config configs/mappo/ca-intersection-v0.yaml \
  --mode test --seed 1 --device cpu \
  --model-dir-load '<best_model.pth 的实际路径>' \
  --test-episode 100 --record-video --video-episodes 6
```

运行目录保存 `eval_metadata.json`、`eval_records.jsonl`、模型及 provenance sidecar；
视频包含逐局 MP4、合并视频、contact sheet 和视频 summary。以实际输出路径为准；
视频 summary 不替代 100-case 数值结果。

## Provenance 与聚合

当前非 smoke 的 benchmark/test/order-eval/ca-trace 要求 tracked Git 工作树干净。
严格模型加载校验 checkpoint SHA、训练合同、provenance v2、Git commit 和 runtime
source manifest；即使只新增文档提交，也不能在新 commit 直接严格加载旧模型。
历史复现使用匹配的 checkout。`--allow-unverified-model` 仅供诊断，产物不进入正式聚合。

人工检查 final runs 后建立显式 manifest，避免宽泛 glob 混入诊断结果。示例结构：

```json
{
  "schema_version": 1,
  "eval_records": [
    "runs/mappo_highway/<seed1-test>/eval_records.jsonl",
    "runs/mappo_highway/<seed2-test>/eval_records.jsonl",
    "runs/mappo_highway/<seed3-test>/eval_records.jsonl"
  ],
  "expected_seeds": {"mappo": [1, 2, 3]}
}
```

替换实际路径和预先声明的 seeds 后运行：

```bash
uv run python -m ca_commappo.evaluation.aggregate_results \
  --runs-manifest results/final_runs_manifest.json --output results/aggregated.csv
```

聚合器重算逐局 summary，并检查不少于 100 episodes 的 final test、协议、seed 集、
checkpoint/sidecar 与重复模型。同一严格 manifest 的学习算法需来自同一源码提交和
runtime manifest、兼容的任务与评估协议；聚合命令自身不必在历史 checkout 中执行。
学习策略报告训练 seed 间 sample std。通行时间与回合时长的区别见环境说明；当前
聚合表的旧 `travel` 字段仍表示回合时长。

`order-eval` 可诊断观测行顺序敏感性。`ca-trace` 和 `main.py paper-analysis` 保留了
旧 CA 字段、三 seed 和固定算法配对，当前只用于旧研究追溯，不作为新论文必做步骤。
其参数可用对应 `--help` 查看；重新用于新研究前需先调整这些假设。

阶段结束记录配置、commit、seed、设备、命令、模型与产物路径、验证结果和协议偏离。
历史结果入口见 [results/README.md](../results/README.md)；不覆盖既有 metadata 或将
旧环境成绩移植到新环境。

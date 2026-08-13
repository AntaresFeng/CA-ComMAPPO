# Canonical MAPPO baseline

本文档定义后续实验和论文统一使用的 MAPPO 基线。该配置来自已验证的 Iteration 5，
不再以迭代配置维护。

## 配置合同

正式入口：`configs/mappo/intersection-multi-agent-v1.yaml`。

- 环境与奖励：`configs/highway/intersection-multi-agent-v1.yaml`
- 共享训练超参：`configs/common.yaml`
- MAPPO 算法身份：`configs/mappo/intersection-multi-agent-v1.yaml`

关键环境参数：4 个 controlled vehicles、5 Hz、20 秒、100 个决策步，collision reward
`-40`、high-speed shaping `0.01`、arrival reward `+10`、不做 reward normalization。
关键训练参数：1M steps、16 parallels、buffer 2048、5 PPO epochs、2 minibatches、
learning rate `3e-4`、entropy coefficient `0.03`、`gamma=0.99`、
`gae_lambda=0.95`，启用 ValueNorm、advantage normalization、GAE 和 gradient clipping。

`gamma=0.998 / gae_lambda=0.99` 的受控实验在独立 100 局验证中得到 arrival `0.23`、
collision `0.36`，弱于本基线的 `0.38/0.01`，因此已回退且不属于正式配置。

## Seed 1 参考结果

正式 benchmark artifact：
`runs/mappo_epochs5_ent03_reward_ordering_highway/20260810_220913_630764_benchmark_seed_1/`。
目录名保留历史名称以保持 checkpoint provenance 不变；它在论文和后续实验中统一标记为
“MAPPO baseline”。best checkpoint 为 epoch 89 / 890k，由固定 seeds
`10000..10049` 的 50 局 deterministic argmax 选择：

| protocol | episodes | arrival | collision | truncation | mean reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint selection | 50 | 0.28 | 0.04 | 0.68 | 7.0388 |
| independent final | 100 | 0.38 | 0.01 | 0.61 | 8.1758 |
| random | 100 | 0.10 | 0.75 | 0.15 | -12.4574 |
| idle-only | 100 | 0.17 | 0.83 | 0.00 | -16.1293 |

独立 final artifact：
`runs/mappo_epochs5_ent03_reward_ordering_highway/20260811_195920_893803_test_seed_1/`，
固定 seeds `20000..20099`，deterministic argmax，并保留 6 个对应 seeds
`20000..20005` 的视频、合并视频和 contact sheet。best checkpoint SHA-256：
`b1ee3493b7d34a35960e96d5c12791a70e975cea923d5cc6e95da3452cf5697e`；正式训练
contract SHA-256：
`d942c76c83d9a8b58a3100fbf3034182f4b25b3ec705ccd7241c00ca4256d699`。

当前只有 seed 1，尚不满足三 seed 的最终论文验收条件；后续算法对比必须使用相同任务
配置、预算、训练 seed block 和固定选择/final 协议。

机器可读证据集中在：

- `results/mappo_baseline_seed1_provenance.json`
- `results/mappo_baseline_seed1_final_manifest.json`
- `results/mappo_baseline_seed1_comparison.csv`

## 复现命令

正式训练必须在独立 tmux 中运行 benchmark，并保持 W&B 离线：

```bash
tmux new-session -d -s mappo-baseline-seed1 -c /home/fzx/CA-ComMAPPO \
  'env WANDB_MODE=offline UV_CACHE_DIR=/tmp/ca-commappo-uv-cache \
   MPLCONFIGDIR=/tmp/ca-commappo-mpl-cache PYTHONUNBUFFERED=1 \
   uv run python main.py mappo \
   --config configs/mappo/intersection-multi-agent-v1.yaml \
   --mode benchmark --seed 1'
```

最终结果只取 best model 的独立 100 局验证：

```bash
WANDB_MODE=offline UV_CACHE_DIR=/tmp/ca-commappo-uv-cache \
MPLCONFIGDIR=/tmp/ca-commappo-mpl-cache \
uv run python main.py mappo \
  --config configs/mappo/intersection-multi-agent-v1.yaml \
  --mode test --seed 1 \
  --model-dir-load runs/mappo_highway/<benchmark-run>/models/best_model.pth \
  --test-episode 100 --record-video --video-episodes 6
```

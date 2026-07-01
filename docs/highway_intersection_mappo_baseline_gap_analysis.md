# Highway Intersection MAPPO Baseline Gap Analysis

Date: 2026-06-29
Updated: 2026-07-01

## Summary

当前仓库已经完成了 `highway-env` 交叉路口环境到 XuanCe 多智能体接口的适配层，并新增了 highway intersection MAPPO 训练入口、正式 YAML 配置和小参数烟测配置，但还没有形成一个可复现、可评估并带结果归档的正式 highway intersection MAPPO baseline。

也就是说，当前状态更接近：

- 已有：环境适配器、XuanCe 注册、reset/step/state 基础兼容性测试、terminal agent 后续 transition 的 mask/reward 处理、random / idle-only sanity baseline 基础版，以及 highway MAPPO runner、正式配置和 smoke 配置。
- 缺少：训练后评估指标脚本、正式多 seed 训练结果归档和实验说明。

## Current State

### 已具备

1. 环境适配器

   文件：`ca_commappo/envs/highway_intersection.py`

   已提供：

   - `build_intersection_config()`
   - `HighwayIntersectionMultiAgentEnv`
   - `register_highway_intersection_env()`

   适配器负责：

   - 使用 `gym.make("intersection-v1", ...)` 创建环境。
   - 将 highway-env 的 tuple observation/action 空间映射到 XuanCe 的 agent dict。
   - 将 `info["agents_rewards"]` 和 `info["agents_terminated"]` 转成 per-agent dict。
   - 提供 `state()`、`agent_mask()`、`avail_actions()`、`render()`、`close()`。
   - 在调用 highway-env 前深度合并 `highway_config`，避免 highway-env 1.11 的浅合并覆盖嵌套配置。
   - 已到达或已终止 agent 的后续 transition 会通过 `agent_mask=False` 排除，并将后续 adapter-facing reward 清零。

2. 适配器测试

   文件：`tests/test_highway_intersection_adapter.py`

   已覆盖：

   - 默认 highway config 与运行时 config 一致。
   - `highway_config` 嵌套字段深度合并。
   - `controlled_vehicles` 控制智能体数量。
   - reset/step/state 的 XuanCe 兼容性。
   - `register_highway_intersection_env()` 可被 `xuance.environment.make_envs()` 使用。

3. 冒烟和诊断入口

   文件：`main.py`

   当前能创建 3-agent intersection 环境，reset 一次，采样动作 step 一次，并打印基础信息。

   文件：`examples/debug_highway_env_episode.py`

   当前可用于对比 raw highway-env 与 adapter wrapper 行为，并打印 wrapper 侧的 `agent_mask`、adapter-facing `agents_rewards` 和 highway-env 原始 `raw_agents_rewards`。

4. 参考示例

   文件：

   - `examples/mappo/mappo_simple_spread.py`
   - `examples/mappo/mappo_football.py`
   - `examples/mappo/mappo_mpe_configs/simple_spread_v3.yaml`
   - `examples/mappo/mappo_football_configs/3v1.yaml`

   这些是 XuanCe MAPPO 示例，但还不是 highway intersection baseline。

5. Random / idle-only sanity baseline

   文件：

   - `examples/random_highway_intersection.py`
   - `examples/sanity/highway_intersection.yaml`
   - `ca_commappo/evaluation/sanity_baselines.py`
   - `tests/test_sanity_baselines.py`

   已支持：

   - `random` 策略：每个 agent 独立采样离散动作。
   - `idle-only` 策略：所有 agent 固定动作 `1`，即 `IDLE`。
   - 多 seed 配置。
   - 输出 mean episode reward、mean per-agent reward、episode length、collision rate、arrival rate、truncation rate。
   - 可选输出完整 JSON 结果。

6. Highway MAPPO 训练入口和配置

   文件：

   - `examples/mappo/mappo_highway_intersection.py`
   - `examples/mappo/mappo_highway_configs/intersection_v1.yaml`
   - `examples/mappo/mappo_highway_configs/intersection_v1_smoke.yaml`

   已支持：

   - 注册 `HighwayIntersection` 并通过 `make_envs(configs)` 创建 `DummyVecMultiAgentEnv`。
   - 使用 `Categorical_MAAC_Policy` 和 `Basic_MLP` 初始化 `MAPPO_Agents`。
   - `intersection_v1.yaml` 提供正式训练配置；`intersection_v1_smoke.yaml` 保留 CPU、小并行数、小 buffer、短 `running_steps`，用于快速训练链路烟测。
   - 通过 `flatten_observations: true` 将二维 kinematics observation 展平成一维向量，匹配 XuanCe `Basic_MLP`。
   - 对 XuanCe 1.4.3 MARL buffer 的 `n_envs` / `obs_space` / `act_space` 旧属性读取做 runner 内兼容补丁，不修改 `.venv`。

## Missing Pieces

### 1. Highway 专用 MAPPO 训练入口（基础版已完成）

已新增训练脚本：

`examples/mappo/mappo_highway_intersection.py`

已具备：

- 读取 highway intersection 专用 YAML 配置。
- 调用 `register_highway_intersection_env()`。
- 用 `make_envs(configs)` 创建训练环境。
- 创建 `MAPPO_Agents(config=configs, envs=envs)`。
- 支持 `train`、`test`、`benchmark` 三种模式。
- 默认保存最终模型，benchmark 模式保存最佳模型；`--no-save` 可用于烟测。
- 结束时调用 `Agents.finish()` 和环境关闭逻辑。

当前 `main.py` 仍只适合适配器冒烟；MAPPO 训练使用 `examples/mappo/mappo_highway_intersection.py`。

### 2. Highway 专用 MAPPO YAML 配置（正式配置和 smoke 配置已完成基础版）

已新增配置文件：

`examples/mappo/mappo_highway_configs/intersection_v1.yaml`

这是参考 `examples/mappo/mappo_mpe_configs/simple_spread_v3.yaml` 补齐字段后的正式训练配置。它保留 highway wrapper 需要的离散动作 policy、`flatten_observations: true` 和 `use_global_state: true`，并使用更长训练步数和更大的 buffer。

关键设置：

- `policy: "Categorical_MAAC_Policy"`，匹配 `intersection-v1` 的 `Discrete(3)` 动作空间。
- `vectorize: "DummyVecMultiAgentEnv"`，优先保持 Windows 本地实验稳定；需要更高吞吐时再测试 `SubprocVecMultiAgentEnv`。
- `parallels: 4`、`buffer_size: 1024`、`n_epochs: 4`、`n_minibatch: 4`、`running_steps: 1000000`、`eval_interval: 10000`、`test_episode: 5`。
- `highway_config` 显式记录 `MultiAgentObservation` / `MultiAgentAction`、3 个受控车辆、40 秒 episode、奖励和离散目标速度。

烟测配置文件：

`examples/mappo/mappo_highway_configs/intersection_v1_smoke.yaml`

该配置使用小参数，便于快速迭代：

```yaml
dl_toolbox: "torch"
project_name: "CA-ComMAPPO"
logger: "tensorboard"
render: false
render_mode: "rgb_array"
fps: 15
device: "cpu"
distributed_training: false

agent: "MAPPO"
env_name: "HighwayIntersection"
env_id: "intersection-v1"
env_seed: 1
vectorize: "DummyVecMultiAgentEnv"

learner: "MAPPO_Learner"
policy: "Categorical_MAAC_Policy"
representation: "Basic_MLP"
use_rnn: false
use_parameter_sharing: true
use_actions_mask: false
use_global_state: true

flatten_observations: true
representation_hidden_size: [64, 64]
actor_hidden_size: [64, 64]
critic_hidden_size: [64, 64]
activation: "relu"
normalize: "LayerNorm"

seed: 1
parallels: 1
buffer_size: 8
n_epochs: 1
n_minibatch: 1
learning_rate: 0.0003
weight_decay: 0
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
vf_coef: 0.5
ent_coef: 0.01

use_linear_lr_decay: false
end_factor_lr_decay: 0.5
use_value_clip: true
value_clip_range: 0.2
use_value_norm: true
use_huber_loss: true
huber_delta: 10.0
use_advnorm: true
use_gae: true
use_grad_clip: true
grad_clip_norm: 10.0

running_steps: 16
eval_interval: 16
test_episode: 1

log_dir: "logs/mappo_highway/"
model_dir: "models/mappo_highway/"

highway_config:
  controlled_vehicles: 3
  duration: 13
  initial_vehicle_count: 10
  spawn_probability: 0.6
  collision_reward: -5
  high_speed_reward: 1
  arrived_reward: 1
  normalize_reward: false
  observation:
    observation_config:
      vehicles_count: 15
      features: ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
  action:
    action_config:
      target_speeds: [0, 4.5, 9]
```

注意点：

- intersection 当前动作空间是 `Discrete(3)`，所以 MAPPO policy 应使用 `Categorical_MAAC_Policy`，不是 MPE 示例里的 `Gaussian_MAAC_Policy`。
- 初期建议使用 `DummyVecMultiAgentEnv`，先降低 Windows 子进程和调试复杂度。
- `use_global_state: true` 可以利用适配器的 `state()` 作为 centralized critic 输入，更符合 MAPPO 的 CTDE 设定。
- `flatten_observations: true` 是当前 XuanCe `Basic_MLP` 跑 highway kinematics observation 的必要兼容选项；否则二维 observation space 和 XuanCe 内部展平后的 batch shape 不一致。

### 3. 端到端训练烟测

状态：基础版已完成。当前最小训练命令已经证明 XuanCe 的 `MAPPO_Agents`
可以完成初始化、采样、写入 on-policy buffer，并执行至少一次网络更新。

已验证命令：

```bash
uv run python examples/mappo/mappo_highway_intersection.py --env-id intersection_v1_smoke --mode train --running-steps 8 --buffer-size 4 --parallels 1 --n-epochs 1 --n-minibatch 1 --no-save
```

该命令完成后会打印 `Finish training.`。这仍然只是接口和训练链路烟测，不用于评价算法性能。

这个很小的端到端检查覆盖：

- 创建 1-2 个并行环境。
- `buffer_size` 设置得很小，例如 8 或 16。
- `running_steps` 设置得很小，例如 16 或 32。
- 初始化 `MAPPO_Agents`。
- 调用一次 `Agents.train(...)` 或 `Agents.run_episodes(...)` + `Agents.train_epochs(...)`。
- 确认没有 shape、dtype、action 类型、state 类型错误。

配置层面还通过 `tests/test_mappo_highway_intersection.py` 检查了 `weight_decay`、
`use_linear_lr_decay` 和 `end_factor_lr_decay` 等 XuanCe learner 初始化必需字段，
避免再次出现训练入口启动后才暴露的配置缺口。

### 3.5 Terminal Agent 后续 Transition 的 Mask / Reward 处理（已完成）

状态：已在 adapter 层完成，并通过 adapter 测试和 wrapper debug episode 验证。`reset()` 时所有 agent active；每个 step 使用 `active_before_step` 作为本 transition 的 `agent_mask`；首次 terminal transition 保留 mask 和 reward；后续 step 中该 agent 的 mask 为 `False`，adapter-facing reward 为 `0.0`。highway-env 原始重复到达奖励保留在 `info["raw_agents_rewards"]` 中用于诊断。

当前 highway intersection 存在一个需要在 adapter 层明确处理的多智能体生命周期问题：

- highway-env 的全局终止条件是任一受控车碰撞、所有受控车到达，或 `offroad_terminal=true` 时 ego 离路。
- 单个受控车到达后，只要全局 episode 还没有结束，highway-env 仍会在后续 step 中为该车生成 observation、`agents_rewards` 和 `agents_terminated`。
- 对已到达车辆，`IntersectionEnv._agent_reward()` 会持续返回 `arrived_reward`，默认值为 `1`；这不是一次性到达奖励，而是到达后每个后续 step 都可能继续得到 `1`。
- XuanCe MAPPO 会把 `info["agent_mask"]` 存入 buffer，并在 actor、entropy、critic loss 中使用该 mask。`terminated` 主要用于 return / GAE 的 bootstrap 截断，不会自动把该 agent 后续 transition 从 loss 中排除。

因此，terminal agent 后续 transition 应视为无效训练样本。旧 adapter 的 `agent_mask()` 一直返回全 True，会让已到达 agent 的后续动作、奖励和观测继续进入 MAPPO 训练，可能放大 arrived agent 的回报并污染 credit assignment；当前实现已修正这一点。

建议处理语义：

- reset 时所有 agent active。
- 每个 step 先记录 `active_before_step`，这个 mask 对应该 step 的 `obs_t/action_t/reward_t/terminated_t`，用于 XuanCe 的 `agent_mask`。
- 如果某个 agent 在本 step 首次 `terminated=True`，保留这个 terminal transition 的奖励和 mask；从下一步开始该 agent 的 `agent_mask=False`。
- terminal 后的 agent 后续 reward 应清零，或至少必须通过 `agent_mask=False` 从 MAPPO loss 中排除。
- terminal 后动作可以忽略或替换为安全动作，例如 `IDLE`；不要简单把 `avail_actions` 全置 False，因为它是动作可用性 mask，不是 alive mask，且全 False 可能破坏离散策略分布。

这部分已有 adapter 测试覆盖：单个 agent 到达但全局未结束时，到达当步仍保留有效 terminal transition，下一步开始 mask 为 False，后续 reward 不再累计到训练信号。

端到端诊断命令：

```bash
uv run python examples/debug_highway_env_episode.py --target wrapper --render-mode rgb_array
```

在 `arrived_reward=5`、2-agent wrapper 诊断 episode 中，已观测到以下关键语义：

- 首次到达步：terminal agent 的 adapter-facing reward 保留为 `5`，`raw_agents_rewards` 也为 `5`，`agent_mask` 仍为 `True`，用于保留该 terminal transition。
- 到达后的后续步：terminal agent 的 adapter-facing reward 为 `0.0`，`agent_mask=False`，而 `raw_agents_rewards` 仍显示 highway-env 原始重复到达奖励 `5`。
- terminal 后如果上层仍传入该 agent 的动作，adapter 会在传给 highway-env 前替换为安全动作 `IDLE=1`；`avail_actions` 不被置空。

### 4. 评估指标脚本

只看 average reward 不够支撑 highway intersection baseline。

建议新增：

`examples/evaluate_highway_intersection.py`

或：

`ca_commappo/evaluation/highway_metrics.py`

至少统计：

- mean episode reward
- collision rate
- arrival/pass rate
- mean episode length
- mean per-agent reward
- timeout/truncation rate

进一步可统计：

- average travel time
- waiting time
- throughput
- average speed

这些指标需要从 highway-env 的 `info`、车辆状态和 episode 结束原因中整理。第一版可以先只做 reward、collision、arrival、episode length。

### 5. Random / Rule-based Sanity Baseline（基础版已完成）

基础版已完成，不再是 MAPPO baseline 的阻塞缺口。

已具备：

- `examples/random_highway_intersection.py`
- `examples/sanity/highway_intersection.yaml`
- `ca_commappo/evaluation/sanity_baselines.py`
- `tests/test_sanity_baselines.py`

已支持策略：

- random：每个 agent 独立采样离散动作。
- idle-only：所有 agent 固定动作 `1`，即 `IDLE`。

已支持统计：

- mean episode reward
- mean per-agent reward
- mean episode length
- collision rate
- arrival rate
- truncation rate

后续仍可扩展：

- cautious：接近冲突区时倾向 `SLOWER`，否则 `IDLE` 或 `FASTER`。
- 将正式 random / idle-only 多 seed 结果归档为 CSV 或 JSON，作为 MAPPO 对照基线。

### 6. 实验协议和 README 说明

README 当前主要说明适配器用法，缺少 baseline 运行说明。

需要补充：

- 如何运行随机 baseline。
- 如何运行 MAPPO 小步烟测。
- 如何运行正式训练。
- 如何测试已保存模型。
- 日志目录和模型目录在哪里。
- 推荐 seed 列表，例如 `1, 2, 3, 4, 5`。
- 常见问题：Windows 下优先用 `DummyVecMultiAgentEnv`，确认稳定后再尝试 `SubprocVecMultiAgentEnv`。

## Recommended Execution Order

1. 新增 highway MAPPO 配置文件。（已完成基础版）

   先用 `DummyVecMultiAgentEnv`、小并行数、短训练步数，保证能快速迭代。

2. 新增 highway MAPPO 训练脚本。（已完成基础版）

   直接参考 `examples/mappo/mappo_simple_spread.py`，但必须注册 `HighwayIntersection`，并使用离散动作对应的 `Categorical_MAAC_Policy`。

3. 跑端到端小步训练烟测。（已完成基础版）

   目标不是性能，而是确认 `MAPPO_Agents` 初始化、采样、存 buffer、更新网络都不报错。

4. 复跑并归档 random / idle-only baseline。

   基础脚本已完成；下一步是用固定配置输出多 seed JSON 或 CSV，建立指标下限，并确认碰撞率和回报统计合理。

5. 新增评估指标汇总脚本。

   输出 CSV 或 JSON，便于后续多 seed 对比。

6. 更新 README。

   把训练、评估、日志和模型目录写成可直接复制运行的命令。

7. 扩展正式实验。

   在基础 MAPPO 跑通后，再进入 IPPO、Attention-MAPPO、CommNet/IC3Net、CA-ComMAPPO。

## Minimal Definition Of Done

要称为完成 highway intersection MAPPO baseline，至少需要满足：

- `uv run python examples/mappo/mappo_highway_intersection.py --env-id intersection_v1` 能启动训练。
- 小步训练烟测能完成至少一次 policy update。
- 有一份固定配置文件记录 highway 参数和 MAPPO 超参数。
- 能通过 `examples/random_highway_intersection.py` 产出 random / idle-only baseline 结果，并将正式对照结果归档。
- 有 MAPPO 训练后评估结果。
- 评估结果至少包含平均回报、碰撞率、通过率、episode length。
- README 记录可复现实验命令。

## Notes

- 当前适配器测试通过并不等价于 MAPPO baseline 完成；它只说明环境接口基本可用。
- terminal agent 后续 transition 的 `agent_mask` 和 reward 语义已在 adapter 层处理；后续 MAPPO runner 应继续确认 XuanCe buffer 中使用的是 adapter 提供的 mask/reward。
- highway-env 的 kinematics observation 对车辆顺序敏感，普通 MLP-MAPPO 应作为基础 baseline，但不是最终强 baseline。
- 后续论文主线建议从 MAPPO-MLP 逐步扩展到 Attention-MAPPO，再扩展到显式通信与 conflict-aware communication。

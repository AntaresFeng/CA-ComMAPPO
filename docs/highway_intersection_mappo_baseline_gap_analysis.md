# Highway Intersection MAPPO Baseline Gap Analysis

Date: 2026-06-29

## Summary

当前仓库已经完成了 `highway-env` 交叉路口环境到 XuanCe 多智能体接口的适配层，但还没有形成一个可复现、可训练、可评估的 highway intersection MAPPO baseline。

也就是说，当前状态更接近：

- 已有：环境适配器、XuanCe 注册、reset/step/state 基础兼容性测试。
- 缺少：MAPPO 训练入口、专用配置、端到端训练烟测、评估指标脚本、随机/规则 sanity baseline、实验说明。

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

2. 适配器测试

   文件：`tests/test_highway_intersection_adapter.py`

   已覆盖：

   - 默认 highway config 与运行时 config 一致。
   - `highway_config` 嵌套字段深度合并。
   - `controlled_vehicles` 控制智能体数量。
   - reset/step/state 的 XuanCe 兼容性。
   - `register_highway_intersection_env()` 可被 `xuance.environment.make_envs()` 使用。

3. 冒烟入口

   文件：`main.py`

   当前能创建 3-agent intersection 环境，reset 一次，采样动作 step 一次，并打印基础信息。

4. 参考示例

   文件：

   - `examples/mappo/mappo_simple_spread.py`
   - `examples/mappo/mappo_football.py`
   - `examples/mappo/mappo_mpe_configs/simple_spread_v3.yaml`
   - `examples/mappo/mappo_football_configs/3v1.yaml`

   这些是 XuanCe MAPPO 示例，但还不是 highway intersection baseline。

## Missing Pieces

### 1. Highway 专用 MAPPO 训练入口

需要新增一个训练脚本，例如：

`examples/mappo/mappo_highway_intersection.py`

职责：

- 读取 highway intersection 专用 YAML 配置。
- 调用 `register_highway_intersection_env()`。
- 用 `make_envs(configs)` 创建训练环境。
- 创建 `MAPPO_Agents(config=configs, envs=envs)`。
- 支持训练、测试、benchmark 三种常用模式。
- 保存最佳模型和最终模型。
- 结束时调用 `Agents.finish()` 和环境关闭逻辑。

当前 `main.py` 只适合适配器冒烟，不适合作为训练 baseline。

### 2. Highway 专用 MAPPO YAML 配置

需要新增配置文件，例如：

`examples/mappo/mappo_highway_configs/intersection_v1.yaml`

关键字段建议：

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

representation_hidden_size: [128, 128]
actor_hidden_size: [128, 128]
critic_hidden_size: [128, 128]
activation: "relu"
normalize: "LayerNorm"

seed: 1
parallels: 4
buffer_size: 512
n_epochs: 5
n_minibatch: 4
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
vf_coef: 0.5
ent_coef: 0.01

use_value_clip: true
value_clip_range: 0.2
use_value_norm: true
use_huber_loss: true
huber_delta: 10.0
use_advnorm: true
use_gae: true
use_grad_clip: true
grad_clip_norm: 10.0

running_steps: 1000000
eval_interval: 20000
test_episode: 20

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

### 3. 端到端训练烟测

当前测试只证明环境能被 `make_envs()` 创建，还没有证明 XuanCe 的 `MAPPO_Agents` 能完成初始化和训练循环。

需要一个很小的端到端检查，可以是脚本或测试：

- 创建 1-2 个并行环境。
- `buffer_size` 设置得很小，例如 8 或 16。
- `running_steps` 设置得很小，例如 16 或 32。
- 初始化 `MAPPO_Agents`。
- 调用一次 `Agents.train(...)` 或 `Agents.run_episodes(...)` + `Agents.train_epochs(...)`。
- 确认没有 shape、dtype、action 类型、state 类型错误。

这个检查不用于评估算法性能，只用于防止训练入口在真实长跑时才暴露接口问题。

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

### 5. Random / Rule-based Sanity Baseline

MAPPO baseline 之前应先有一个随机策略或简单规则策略，用来确认环境、奖励和指标统计是正常的。

建议新增：

`examples/random_highway_intersection.py`

或集成到评估脚本中：

- random：每个 agent 从 `env.action_space[agent].sample()` 采样。
- idle-only：所有 agent 固定动作 `1`，即 `IDLE`。
- cautious：接近冲突区时倾向 `SLOWER`，否则 `IDLE` 或 `FASTER`。这可以后续再做。

第一版至少保留 random 和 idle-only。

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

1. 新增 highway MAPPO 配置文件。

   先用 `DummyVecMultiAgentEnv`、小并行数、短训练步数，保证能快速迭代。

2. 新增 highway MAPPO 训练脚本。

   直接参考 `examples/mappo/mappo_simple_spread.py`，但必须注册 `HighwayIntersection`，并使用离散动作对应的 `Categorical_MAAC_Policy`。

3. 跑端到端小步训练烟测。

   目标不是性能，而是确认 `MAPPO_Agents` 初始化、采样、存 buffer、更新网络都不报错。

4. 新增 random / idle-only baseline。

   先建立指标下限，确认碰撞率和回报统计合理。

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
- 有 random 或 idle-only baseline 结果。
- 有 MAPPO 训练后评估结果。
- 评估结果至少包含平均回报、碰撞率、通过率、episode length。
- README 记录可复现实验命令。

## Notes

- 当前适配器测试通过并不等价于 MAPPO baseline 完成；它只说明环境接口基本可用。
- highway-env 的 kinematics observation 对车辆顺序敏感，普通 MLP-MAPPO 应作为基础 baseline，但不是最终强 baseline。
- 后续论文主线建议从 MAPPO-MLP 逐步扩展到 Attention-MAPPO，再扩展到显式通信与 conflict-aware communication。

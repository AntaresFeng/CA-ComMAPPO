# AI 执行提示词：完成 CA-ComMAPPO 论文的算法与实验

> **这份文档是什么**：一份分阶段的、仓库感知的执行提示词。把它整体粘贴进本仓库的一个新 Claude Code 会话，即可驱动该会话实现论文计划 `docs/thesis-planning.md` 中的全部算法并跑完实验矩阵。
>
> **范围**：只覆盖「算法实现 + 实验执行 + 结果归档」，**不含论文撰写**。
> **执行对象**：本仓库的 Claude Code 会话（可用 `uv run`、可读写仓库文件、可运行测试）。
> **训练执行模式**：AI 负责实现代码/配置/脚本，用 smoke 配置本地验证可用性，产出可复现 runbook；多 seed 正式训练与全量实验矩阵交由远端服务器（wandb）执行，AI 负责结果收集与归档。
>
> **前置假设**：执行者已读过 `AGENTS.md`。本提示词中所有文件路径、类名、行号都基于当前仓库与已安装的 `xuance==1.4.3`（`.venv`），但代码会变动，**执行前务必自行核对关键符号是否仍存在**。

---

## 0. 全局硬性规则（不可妥协）

执行任何阶段前，先内化以下规则。违反任何一条都会让下游结果不可比或不可复现。

### 0.1 仓库工作方式（来自 AGENTS.md）

- 日常运行用 `uv run python <script>.py` 或 `uv run python -m <module>`。**不要** `uv build`，不要把发布构建当常规步骤。
- `uv run pytest -q` 只在改动适配器核心逻辑、准备提交或排查回归时运行；普通参数实验不强制跑测试。
- 提交前执行 `ruff check --fix; ruff format`。提交信息用 Conventional Commit 简洁前缀：`feat:` / `fix:` / `docs:` / `test:` / `refactor:` / `chore:`。
- 本仓库以本地研究迭代为主，提交说明只需清楚描述本次实验或行为变化；跑了测试就记录对应命令。
- 实验记录应保存：YAML 文件、命令、git commit 或工作树说明、输出 JSON/JSONL。

### 0.2 不得修改 `.venv`；自定义类一律运行时注册

- XuanCe 安装在 `.venv\Lib\site-packages\xuance\`。**严禁**直接编辑 site-packages 里的文件。
- 本项目已有的兼容模式是 `ca_commappo/xuance_compat.py` 的 `patch_xuance_marl_buffer_aliases()`——在运行时 monkey-patch 已安装的 `BaseBuffer`，而不是改源码。所有自定义 XuanCe 组件（自定义 Representation / Policy / Learner / 通信模块）都沿用这一思路：**把自定义类放进仓库 `ca_commappo/algorithms/`，并在运行时把类注册进 XuanCe 的对应注册表**，再构造 agent。
- 仓库自有 agent 选择走 `train_highway_intersection.py` 里的 `AGENT_REGISTRY`（见下文），不走 XuanCe 的 `REGISTRY_Agents`。

### 0.3 Reward 契约（来自 `docs/highway_intersection_notes.md` 与适配器实测）

`HighwayIntersectionMultiAgentEnv.step()`（`ca_commappo/envs/highway_intersection_wrapper.py:209-247`）的契约：

- 返回的 `rewards` 与 `info["agents_rewards"]` 都是 **masked adapter-facing reward**；非活跃 agent（已到达/已碰撞的后续步）reward 为 `0.0`。
- `info["raw_agents_rewards"]` 保存 highway-env 原始 per-agent reward，**未掩码**。不要把它当作主 reward 使用。
- `info["crashed"]`、`info["arrived"]` 都是 **per-agent tuple**（不是 highway-env 默认的单车标量）。所有统计必须从这两个字段取。
- `info["global_terminated"] = any(crashed) or all(terminated.values())`（`terminated` 是 highway-env per-agent 的 crashed-or-arrived 标志；对非碰撞车 `terminated` ⇔ `arrived`，故与 `all(arrived)` 在无碰撞时等价，但代码用 `terminated.values()`，符号以此为准）。episode 真正结束 = `global_terminated or truncated`。
- 非活跃 agent 的动作由适配器用 `IDLE_ACTION = 1`（IDLE）代替；动作语义：`0=SLOWER`、`1=IDLE`、`2=FASTER`（`Discrete(3)`，底层 `DiscreteMetaAction`，`target_speeds=[0,4.5,9]`）。
- 新算法的 callback / evaluator **必须**消费适配器面向字段，不得穿透 env 读 `controlled_vehicles` 或读 highway-env 原始 `info["crashed"]`。

### 0.4 指标与评估产物（不可另立口径）

- 指标统一用 `ca_commappo/evaluation/highway_metrics.py`：`episode_outcome`（优先级 collision > arrival > truncation）、`build_episode_record`、`summarize_episode_records`。summary 的 9 个 key：`episodes, mean_episode_reward, mean_agent_reward, mean_episode_length, collision_rate, arrival_rate, truncation_rate, mean_agent_collision_fraction, mean_agent_arrival_fraction`。
- 学习型策略评估统一用 `evaluate_highway_policy()`（`ca_commappo/evaluation/mappo_highway_metrics.py:40-121`），它要求 `info["crashed"]`/`info["arrived"]` 为 per-agent tuple，否则在第 141-144 行抛 `KeyError`。**注意**：该评估器调 `agents.action(...)` 时**不传 `info`**；CommNet/IC3Net（以及若 CA-ComMAPPO 沿用 CommNet 的 `alive_ally` 接口）的 `action()` 需要 `info`——必须运行时打补丁，详见 §1.4 与阶段 2。
- best-model 选择键（`is_better_highway_summary` → `_summary_sort_key`，`mappo_highway_metrics.py:161-169`）：`arrival_rate↑ → collision_rate↓ → mean_episode_reward↑ → mean_episode_length↓`（先到达率，再少碰撞，再高回报，再短 episode）。
- 评估产物写两份：`eval_metadata.json`（`build_eval_metadata`/`write_eval_metadata`，`"x"` 模式，**每次 run 只写一次**，已存在则抛 `FileExistsError`）和 `eval_records.jsonl`（`append_eval_record_jsonl`，`"a"` 模式追加，每行一条）。`SCHEMA_VERSION = 1`。
- run 目录由 `prepare_run_directory()`（`ca_commappo/training/mappo_run_layout.py:10-45`）生成：`runs/<output_dir>/<timestamp>_<mode>_seed_<seed>/`，其下 collocate `models/`、`eval_records.jsonl`、`eval_metadata.json`、`videos/`。`output_dir` 由 YAML 决定（IPPO 用 `runs/ippo_highway/`、MAPPO 用 `runs/mappo_highway/`）。

### 0.5 配置规则

- 所有 highway-env 选项通过 `highway_config` 传入；`observation` / `action` 嵌套字典由适配器 `build_intersection_config()` 与默认值**深度合并**（highway-env 1.11 是浅 update，见 `docs/highway_config_update_bug.md`）。
- 配置由 `configs/common.yaml`、`configs/highway/<env-id>.yaml` 和 `configs/<agent>/<env-id>.yaml` 合并；smoke 通过同一配置加 `--smoke` 动态覆盖运行规模。
- 算法由 YAML 的 `agent` 字段选择（**不是** `main.py` 的子命令名）。`main.py` 的子命令只负责路由到训练模块；即使子命令叫 `mappo`，只要 `--config` 指向的 YAML 里 `agent: "CA_COMMAPPO"`，就会构造 CA-ComMAPPO（前提是 `AGENT_REGISTRY` 里有该键）。

### 0.6 可复现性

- 同一份 YAML、同一个 agent、同一个 seed 列表必须产生相同结果。确定性优先。
- **`eval_metadata.json` / `eval_records.jsonl` 只在 `--mode test` 与 `--mode benchmark` 写**；`--mode train` 不写任何 eval 产物（`train()` 只调 `agents.train()` + 可选存模型）。`eval_metadata.json` 含 `config.config_path`、`config.seed`、`config.highway_config` 快照，`"x"` 模式只写一次；`eval_records.jsonl` 每次 eval 追加一条。归档时要确保 git commit 已记录。

---

## 1. 技术背景速查（关键文件 / 类 / 接口 + 行号）

> 这是给执行者的「地图」，避免重新摸索。行号会随代码变动漂移，执行前以实际为准。

### 1.1 训练入口 `ca_commappo/training/train_highway_intersection.py`

- `AGENT_REGISTRY`（第 46-49 行）：`{"IPPO": IPPO_Agents, "MAPPO": MAPPO_Agents}`，二者从 `xuance.torch.agents` 导入（第 10 行）。**新增算法 = 往这里加一行键值**。`get_agent_class()`（52-55）查表，未知键抛 `ValueError`。
- `run()`（384-406）执行顺序：`prepare_run_directory`（385）→ `patch_xuance_marl_buffer_aliases()`（386）→ `set_seed`（387）→ `make_envs`（388）→ `HighwayMetricsCallback()`（391）→ `get_agent_class(configs.agent)`（392）→ `agent_class(config, envs, callback)`（393）。**自定义 XuanCe 组件的运行时注册必须在 392 行之前完成**——建议在 `patch_xuance_marl_buffer_aliases()` 旁边加一个 `register_ca_commappo_components()` 调用。
- `main()` 默认加载 MAPPO；显式 `--config` 用来选择其他算法或消融配置。通用加载逻辑位于 `ca_commappo/config.py`。
- 模式：`--mode {train,test,benchmark}`。`benchmark` 会用 `EVAL_ENV_SEED_OFFSET = seed + 10000`（第 44 行）建独立 test envs 并交替 train/eval、按 best-key 存 `best_model.pth`。`test` 加载模型后跑 `evaluate_highway_policy` 并可选 `--record-video`。
- `cli_overrides()` 构建 CLI 覆盖字典（仅含非 None 项）；合并顺序为 common → highway → algorithm → smoke → CLI，因此显式 CLI 参数优先级最高。

### 1.2 XuanCe 四个 torch 注册表（均由 `xuance/torch/__init__.py` 重导出）

| 注册表 | 文件 | 行 | 由哪个 YAML 字段驱动 |
|---|---|---|---|
| `REGISTRY_Representation` | `xuance/torch/representations/__init__.py` | 6-13 | `representation` |
| `REGISTRY_Policy` | `xuance/torch/policies/__init__.py` | 59-106 | `policy` |
| `REGISTRY_Learners` | `xuance/torch/learners/__init__.py` | 40-98 | `learner` |
| `REGISTRY_Agents` | `xuance/torch/agents/__init__.py` | 59-116 | `agent`（本项目走自有 `AGENT_REGISTRY`，不用这个） |

- agent 构造链：`MAPPO_Agents.__init__` → `IPPO_Agents.__init__`（建 policy/memory/learner）→ `OnPolicyMARLAgents`/`MARLAgents`。
- `_build_representation(representation_key, input_space, config)`（`xuance/torch/agents/base/agents_marl.py:243-282`）在第 279 行做 `REGISTRY_Representation[representation_key](**input_representations)`。`input_representations` 是一个很大的 kwargs 字典（265-278 行）：`input_shape, hidden_sizes, normalize, initialize, activation, kernels, strides, filters, fc_hidden_sizes, N_recurrent_layers, rnn, dropout, device`。**自定义 Representation 必须用 `**kwargs` 吞掉不相关的字段**。
- `_build_learner(*args)`（288-289）做 `REGISTRY_Learners[self.config.learner](*args)`。
- `MAPPO_Agents._build_policy`（`xuance/torch/agents/multi_agent_rl/mappo_agents.py`）把 policy 限制在 `Categorical_MAAC_Policy` / `Gaussian_MAAC_Policy`；其他字符串会抛 `AttributeError`。所以 CA-ComMAPPO 若要自定义 policy，需要**继承 `MAPPO_Agents` 并重写 `_build_policy`**，或注册一个新 policy 字符串并放宽该检查。

### 1.3 自定义 Representation 的接口模板（来自 `Basic_MLP`，`xuance/torch/representations/mlp.py:24-54`）

```python
class Basic_MLP(Module):
    def __init__(self, input_shape, hidden_sizes, normalize=None, initialize=None,
                 activation=None, device=None, **kwargs):
        super().__init__()
        self.input_shape = input_shape          # space2shape(obs_space) -> 形如 (105,)
        self.output_shapes = {'state': (hidden_sizes[-1],)}   # 必须有
        ...
    def forward(self, observations):           # np.ndarray 进
        return {'state': tensor}                # dict 出，键固定 'state'
```

- `input_shape` 对 flatten 后的 `Box(105,)` 是 `(105,)`。Permutation-invariant 编码器需要在 `forward` 内部把它 reshape 成 `(vehicles_count=15, n_features=7)`，对 `presence==0` 的车辆做 mask，再做 φ/attention 聚合。
- 注册：`REGISTRY_Representation["PermutationInvariantAttention"] = PermutationInvariantAttention`，在 agent 构造前执行。

### 1.4 XuanCe 已内置的相关算法（**可直接复用，不必重写**）

均在 `xuance/torch/agents/multi_agent_rl/` 与 `learners/multi_agent_rl/` 下，policy 在 `xuance/torch/policies/categorical_marl.py`：

| 算法 | Agent | Policy | Learner | 通信模块 |
|---|---|---|---|---|
| MAPPO | `MAPPO_Agents` | `Categorical_MAAC_Policy` | `MAPPO_Learner` | — |
| IPPO | `IPPO_Agents` | 同上 | `IPPO_Learner` | — |
| CommNet | `CommNet_Agents` | `CommNet_Policy`（categorical_marl.py:360，继承 `MAAC_Policy`） | `CommNet_Learner` | `communications/comm_net.py` 的 `CommNet` |
| IC3Net | `IC3Net_Agents` | `IC3Net_Policy`（:432，继承 `CommNet_Policy`） | `IC3Net_Learner` | `IC3NetComm` |
| TarMAC | `TarMAC_Agents` | `TarMAC_Policy`（:510） | `TarMAC_Learner` | `communications/attention_comm.py` |
| MADDPG | `MADDPG_Agents` | `MADDPG_Policy` | `MADDPG_Learner` | — |

- **关键发现**：XuanCe **没有** permutation-invariant / DeepSets / social-attention 编码器。Representation 只有 `Basic_Identical / Basic_MLP / Basic_CNN / AC_CNN_Atari / Basic_RNN / Basic_ViT`。所以 Attention-MAPPO 的编码器**必须自建并注册**。
- `CommNet` 模块（`communications/comm_net.py`）的接口是 `forward(obs, msg_send, alive_ally) -> msg_receive`：对其他 agent 的消息做 `mean` 聚合（按 `alive_ally` 掩码、按 `alive_agent_num` 归一化），依赖 `use_parameter_sharing`、`n_agents`、`model_keys`、`agent_keys`。**CA-ComMAPPO 的通信模块应模仿这个接口，但把 mean 聚合换成 conflict-aware attention**——这是一个干净的 drop-in 替换。
- **CommNet/IC3Net 必须用 RNN 路径**：`use_rnn: true` + `representation: "Basic_RNN"` 是**必需**，不是可选。当 `use_rnn: false` 时，基类 `OnPolicyMARLAgents.train()`（`xuance/torch/agents/core/on_policy_marl.py`）走自带 rollout，调用 `self.action(obs_dict=..., state=..., avail_actions_dict=..., test_mode=False)` 但**不传 `info`**，而 CommNet/IC3Net 的 `action()` 必须 `for data in info` 构建 `alive_ally` → 立即 `TypeError`。只有 `use_rnn: true` 才走 CommNet 自己重写的 `run_episodes()`（内部会构建并传入 `info`）。**MAPPO baseline 的 `use_rnn: false` 与 CommNet/IC3Net 不兼容，不能照抄。** 此外 `comm_passes` 是 CommNet 必填字段（`commnet_agents.py` 读 `self.config.comm_passes`），`gate_hidden_size` 是 IC3Net 必填字段（`IC3Net_Policy.__init__` 读 `self.config.gate_hidden_size`）——MAPPO 配置里都没有，必须补（建议 `comm_passes: 1`、`gate_hidden_size: 16`）。
- **CommNet/IC3Net 的 `action()` 需要 `info` 参数，与项目评估器不兼容**：`CommNet_Agents.action()`（`commnet_agents.py:91-109`）/`IC3Net_Agents.action()`（`ic3net_agents.py:116-134`）都 `for data in info` 取 `agent_mask` 构建 `alive_ally`，`info=None` 时 `TypeError`。但项目评估器 `evaluate_highway_policy()`（`mappo_highway_metrics.py:40-121`）和视频录制器 `_agent_action_for_single_env`（`mappo_video_recorder.py:54-71`）调 `agents.action(...)` 时**都不传 `info`**。因此 `--mode test`/`--mode benchmark` 会崩；而 `--mode train` smoke **不会暴露此问题**（CommNet 自己的 `run_episodes` 会传 `info`）——smoke 通过不等于 test/benchmark 可用。必须运行时打补丁（放 `ca_commappo/algorithms/comm_runtime_patch.py` 或 `xuance_compat.py` 同款，**不改 `.venv`**）：(a) monkey-patch `CommNet/IC3Net_Agents.action` 使 `info is None` 时 `alive_ally` 全 1；或 (b) 扩展 `evaluate_highway_policy`/video recorder 按需构建 `info=[{'agent_mask': envs.agent_mask()...}]` 并传入。详见阶段 2。

### 1.5 现有 MAPPO/IPPO 配置要点（`configs/mappo/intersection-multi-agent-v1.yaml`）

- 算法选择：`agent: "MAPPO"`、`learner: "MAPPO_Learner"`、`policy: "Categorical_MAAC_Policy"`、`representation: "Basic_MLP"`、`dl_toolbox: "torch"`、`vectorize: "SubprocVecMultiAgentEnv"`（IPPO 用 `DummyVecMultiAgentEnv`）。
- 共享/中心化：`use_parameter_sharing: true`、`use_actions_mask: false`、`use_global_state: true`（MAPPO）/`false`（IPPO）。
- 网络尺寸：`representation_hidden_size: [128,128]`、`actor_hidden_size: [128,128]`、`critic_hidden_size: [128,128]`、`activation: "relu"`、`normalize: "LayerNorm"`、`initialize: "orthogonal"`、`fc_hidden_sizes: [64,64,64]`、`recurrent_hidden_size: 64`、`use_rnn: false`、`flatten_observations: true`。
- 优化/rollout：`seed: 1`、`parallels: 16`、`buffer_size: 2048`、`n_epochs: 2`、`n_minibatch: 2`、`learning_rate: 0.0003`、`running_steps: 1000000`、`eval_interval: 10000`、`test_episode: 5`、`gamma: 0.99`、`gae_lambda: 0.95`、`clip_range: 0.2`、`vf_coef: 0.5`、`ent_coef: 0.01`、`use_value_norm: true`、`use_gae: true`、`use_grad_clip: true`、`grad_clip_norm: 10.0`。
- `highway_config` 单独位于 `configs/highway/intersection-multi-agent-v1.yaml`：`controlled_vehicles: 3`、`duration: 20`、`initial_vehicle_count: 10`、`spawn_probability: 0.6`、`collision_reward: -10`、`high_speed_reward: 0.5`、`reward_speed_range: [5.0,9.0]`、`arrived_reward: 5`。
- 派生形状：每 agent flatten obs = `15 * 7 = 105` 维；`state_space` = `3 * 105 = 315` 维（3 个受控车）。
- smoke 通过 `--smoke` 动态生成：切换 tensorboard / dummy vectorizer，并缩小隐藏层、并行数、buffer、epoch 和步数；highway 环境配置与正式实验完全相同。
- 训练/验证入口见 `docs/mappo_model_evaluation_and_video.md`：`uv run python main.py mappo --config <cfg> --mode {train,test,benchmark}`；直接调模块 `uv run python -m ca_commappo.training.train_highway_intersection --config <cfg> --mode test`。

### 1.6 视频录制（对新算法的已知限制）

- `record_mappo_policy_videos()`（`ca_commappo/evaluation/mappo_video_recorder.py:167-299`）里 `mappo_test` 字面量散布在多处：`CONTACT_SHEET_FILENAME`（:23）、`COMBINED_VIDEO_FILENAME`（:24）、`_expected_mappo_video_path`（:93）、`RecordVideo(name_prefix=...)`（:198）、`build_episode_record(policy=...)`（:245）、校验 glob `mappo_test-episode-*.mp4`（:254）。**新算法录视频需把所有这些站点统一参数化**（由 `configs.agent` 派生单一 `name_prefix`，如 `ca_commappo_test`）；MAPPO 必须保持 `mappo_test` 以兼容现有测试 `tests/test_mappo_video_recorder.py`（该测试断言 `mappo_test` 名，需与参数化同步更新）。这是阶段 3 的明确任务。

---

## 2. 算法实现策略总表（reuse vs build）

| 算法 | 策略 | 关键动作 |
|---|---|---|
| Random / Idle-only sanity | **已完成**，复用 | `configs/sanity/highway_intersection.yaml`（仅 `random`/`idle-only`；论文 Group1 的 IDM-like 按备选暂缓，idle-only 充当简单规则基线） |
| IPPO / MAPPO-MLP | **已完成**，作为基线复用 | 现有配置 |
| **Attention-MAPPO** | 新建 Representation，复用 `MAPPO_Agents` | 自建 `PermutationInvariantAttention`，运行时注册进 `REGISTRY_Representation`；YAML 改 `representation` 字段，`agent` 仍为 `"MAPPO"` |
| **CommNet-MAPPO** | **复用 XuanCe `CommNet_Agents`** | 往 `AGENT_REGISTRY` 加 `"COMMNET": CommNet_Agents`；YAML **必须** `use_rnn: true` + `representation: "Basic_RNN"` + `comm_passes: 1`；运行时打 `action()` 的 `info` 补丁（见 §1.4 与阶段 2） |
| **IC3Net-MAPPO** | **复用 XuanCe `IC3Net_Agents`** | 同 CommNet，另需 `gate_hidden_size: 16`；`"IC3NET": IC3Net_Agents` |
| **CA-ComMAPPO**（创新） | 新建 agent/policy/learner/通信模块 + 复用 PI 编码器 | `CA_ComMAPPO_Agents` 继承 `MAPPO_Agents`，重写 `_build_policy`；`ConflictAwareComm` 模块（模仿 `CommNet` 接口，mean→conflict attention）；运行时注册 policy/learner/representation；`AGENT_REGISTRY` 加 `"CA_COMMAPPO"` |

---

## 3. 分阶段任务

> 每个阶段都以**验收门禁**结尾（见 §4）。未过门禁不进入下一阶段。

### 阶段 0 · 定向（不写代码）

1. 通读：`AGENTS.md`、`docs/thesis-planning.md`、`docs/highway_intersection_notes.md`、`docs/sanity_baselines.md`、`docs/mappo_model_evaluation_and_video.md`、`docs/highway_config_update_bug.md`。
2. 通读本提示词 §1 的所有文件与行号，确认符号仍在。
3. 复述 reward 契约、9-key 指标、best-key、eval 产物 schema、run 目录布局——确认理解一致后再开工。
4. 跑一次现有 smoke 确认基线可用：`uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1.yaml --smoke --mode train --no-save`。

### 阶段 1 · Attention-MAPPO（强结构基线，解决车辆顺序敏感）

目标：MAPPO + permutation-invariant social encoder，对观测中车辆顺序不敏感。

> **与论文基线的映射（重要，勿漏）**：论文 `thesis-planning.md` 第二组列了三行——`MAPPO-MLP`、`MAPPO + DeepSets / Attention Encoder`（解决顺序敏感）、`MAPPO + Social Attention`（路口多车交互，标注「非常重要的强基线」，引文 [4] Leurent）。本阶段实现的 `PermutationInvariantAttention`（pairwise `e_ij = φ(o_i, o_j)` + attention pooling）**同时充当 (a) permutation-invariant 结构基线 和 (b) Social Attention 基线**——其 pairwise φ + attention 池化与 Leurent social attention 同构，论文表中可标为 **Social Attention** 并引 [4]。DeepSets 作为备选被有意并入/暂缓（attention 已覆盖顺序无关性）；如评审要求再补 DeepSets 变体。**不要让论文缺一个显式标注的 Social Attention 对照行。**

1. **新建模块** `ca_commappo/algorithms/`（含 `__init__.py`、`representations.py`、`registration.py`）。
2. **实现 `PermutationInvariantAttention(Module)`**，签名对齐 `Basic_MLP`：
   - `__init__(self, input_shape, hidden_sizes, normalize=None, initialize=None, activation=None, device=None, **kwargs)`。
   - 内部从 `input_shape=(105,)` 推断 `vehicles_count=15`、`n_features=7`（从 YAML 读取更稳，但接口不直接给 config，故可约定常量并加 assert）。
   - `forward(observations)`：reshape 成 `(B, V, F)` → 用 `presence` 通道构造 mask → 对每对 `(o_i, o_j)` 算 `e_ij = φ(o_i, o_j)`（相对位置/速度/航向特征）→ `z_i = AttnPool({e_ij})`（对 mask 做 softmax 归一化）→ 输出 `{'state': z_i}`。
   - `self.output_shapes = {'state': (hidden_sizes[-1],)}`。
   - 论文公式（见附录 A）：`e_ij = φ(o_i, o_j)`、`z_i = AttnPool({e_ij})`。
3. **运行时注册**：在 `registration.py` 写 `register_components()`，内含 `REGISTRY_Representation["PermutationInvariantAttention"] = PermutationInvariantAttention`（从 `xuance.torch.representations` 导入注册表）。在 `train_highway_intersection.py` 的 `run()` 里、`get_agent_class` 之前调用（与 `patch_xuance_marl_buffer_aliases()` 并列）。
4. **配置**：新增极小算法 YAML，只写 `representation`、`output_dir` 等算法差异；通用、环境和 smoke 参数由加载器提供。
5. **测试（TDD）**：在 `tests/` 新增 `test_permutation_invariant_representation.py`，断言：(a) 同一场景打乱车辆顺序，`forward` 输出不变（permutation-invariance，数值容差内）；(b) `output_shapes` 正确；(c) `presence==0` 的车被 mask。用确定性 seed，`finally` 关闭资源。
6. **冒烟**：`uv run python main.py mappo --config configs/attention_mappo/intersection-multi-agent-v1.yaml --smoke --mode train --no-save` 跑通且不报维度错。
7. **门禁**：`uv run pytest -q tests/test_permutation_invariant_representation.py -q` 全绿 + smoke 跑通。

### 阶段 2 · CommNet / IC3Net 通信基线（复用 XuanCe）

目标：经典通信 MARL 基线，作为「平均通信 / 门控通信」对照。

1. 在 `train_highway_intersection.py` 顶部 import `CommNet_Agents, IC3Net_Agents`（从 `xuance.torch.agents`），在 `AGENT_REGISTRY` 加 `"COMMNET": CommNet_Agents`、`"IC3NET": IC3Net_Agents`。
2. **先读** `CommNet_Agents._build_policy` 与 `IC3Net_Agents._build_policy`，确认硬性要求：`use_rnn: true` + `representation: "Basic_RNN"`（必需，见 §1.4）、`comm_passes`（CommNet 必填，建议 1）、`gate_hidden_size`（IC3Net 必填，建议 16）、`use_parameter_sharing`、`use_global_state`，以及它们如何与 highway 的 dict-of-agents 接口对接。
3. **配置**：建 `configs/commnet/intersection-multi-agent-v1.yaml`（+ smoke），字段对齐 MAPPO，但 `agent: "COMMNET"`、`learner: "CommNet_Learner"`、`policy: "CommNet_Policy"`、`use_rnn: true`、`representation: "Basic_RNN"`、`comm_passes: 1`。`controlled_vehicles` 与 MAPPO 一致（可比性）。IC3Net 同理建 `configs/ic3net/`，加 `gate_hidden_size: 16`。
4. **运行时补丁 `action()` 的 `info` 不兼容**（关键，不做则 test/benchmark 必崩）：CommNet/IC3Net 的 `action()` 需要 `info`（含 per-env `agent_mask`）构建 `alive_ally`，但项目评估器 `evaluate_highway_policy()`（`mappo_highway_metrics.py:65-72`）和视频录制器 `_agent_action_for_single_env`（`mappo_video_recorder.py:54-71`）调 `agents.action(...)` 时**不传 `info`**。二选一打补丁（放 `ca_commappo/algorithms/comm_runtime_patch.py` 或 `xuance_compat.py` 同款，**不改 `.venv`**）：(a) monkey-patch `CommNet/IC3Net_Agents.action` 使 `info is None` 时 `alive_ally` 全 1；(b) 扩展 `evaluate_highway_policy` 与 video recorder 按需构建 `info=[{'agent_mask': envs.agent_mask()...}]` 并传入。另外 CommNet 的 `run_episodes()`（`commnet_agents.py:155-252`）**不调 `callback.on_train_episode_info`**，故 `HighwayMetricsCallback` 收不到训练期 episode 记录——已知限制，写进 `docs/highway_intersection_notes.md`，训练期指标以 eval 期为准。
5. **冒烟（train + test 都要跑）**：先 `--mode train --no-save` 跑通；**再 `--mode test --test-episode 1`**（必须，因为 train smoke 走 `run_episodes` 不暴露 `action()` 的 `info` 问题，只有 test/benchmark 走 `evaluate_highway_policy` 才暴露）。若仍有 buffer/mask/agent 数兼容问题，按 `xuance_compat.py` 同款运行时补丁修复。
6. **门禁**：两个算法的 train + test smoke 都跑通 + 现有 `uv run pytest -q` 仍全绿。

### 阶段 3 · CA-ComMAPPO（创新算法：冲突感知 + 顺序无关 + 显式通信）

目标：以 MAPPO 为底座，PI 编码器处理多车观测，加入面向无信号路口冲突关系的显式通信模块。

1. **模块实现**（放 `ca_commappo/algorithms/`）：
   - `ConflictAwareComm(Module)`：模仿 `CommNet` 接口 `forward(obs_or_hidden, msg_send, alive_ally) -> msg_receive`，但聚合改成 `c_i = Σ_{j≠i} α_ij W_v m_j`，`α_ij = softmax(q_i^T k_j + b_ij)`。`b_ij` 是冲突先验（见下条）。对 `alive_ally` 掩码、对自身消息排除。
   - `CA_ComMAPPO_Policy`（继承 `Categorical_MAAC_Policy` 或 `CommNet_Policy`）：把 `ConflictAwareComm` 接入前向，actor 输入 `π_i(a_i | z_i, c_i)`。
   - `CA_ComMAPPO_Learner`（继承 `MAPPO_Learner`）：通常可直接复用 MAPPO 的 clipped PPO loss；只在需要把通信纳入 critic 时扩展。critic 用 `V(s)` 或 `V({z_i, m_i, a_i})`。
   - `CA_ComMAPPO_Agents`（继承 `MAPPO_Agents`）：重写 `_build_policy` 以使用自定义 policy/representation（绕开 `MAPPO_Agents._build_policy` 的 policy 字符串白名单）。
2. **`b_ij` 冲突先验的具体设计**（必须可从去中心化观测/消息推导，避免破坏 CTDE）。观测只有 `[presence,x,y,vx,vy,cos_h,sin_h]`（相对 ego，无 route），故用 kinematic proxy——**编码前先在 `docs/ca_commappo_design.md`（新建）冻结以下公式再写代码**：
   - **冲突点**：取路口中心为单一冲突点 `C`（简化；若需要可扩展为各入口车道冲突点集）。
   - **TTC proxy**：`ttc_i = dist(o_i, C) / max(speed_i, eps)`，`speed_i = sqrt(vx_i^2 + vy_i^2)`；符号由航向是否朝 `C` 决定（背离则 `ttc_i = +inf`）。`intent_i` 用航向 + 所在车道方向近似（朝 `C` = 有冲突意图）。
   - **`b_ij` 公式**：`b_ij = -λ · |ttc_i - ttc_j|`（到达冲突点时间差越小先验越大），可再过 `sigmoid` 限幅；`λ` 为可学标量或超参。
   - **消息携带**：`m_j = f_m(z_j, ttc_j, intent_j)` —— 在 `z_j` 后接一个小固定维度头 `[ttc_j, intent_j]`，`ConflictAwareComm` 从 `m_j` 末尾切片读出 `ttc_j`/`intent_j` 算 `b_ij`。
   - **「通信图」实现口径**：论文模块名是「conflict-aware communication graph」，但按论文 §3 公式用**软冲突先验 attention** `α_ij = softmax(q_i^T k_j + b_ij)` 实现，而非硬离散边——在 design doc 里写明，保持命名与实现一致。
   - **验收测试**：构造两车朝 `C` 对向接近的场景，断言其 `α_ij` 大于两车背离场景；`b_ij` 对 `|ttc_i - ttc_j|` 单调递减。
   - **局限**：这是 kinematic proxy（无真实 route/优先级），写进 design doc，列为未来工作。
3. **运行时注册**：在 `registration.register_components()` 里把 `CA_ComMAPPO_Policy` 注册进 `REGISTRY_Policy`（如键 `"CA_ComMAPPO_Policy"`）、`CA_ComMAPPO_Learner` 进 `REGISTRY_Learners`（`"CA_ComMAPPO_Learner"`）、PI 编码器进 `REGISTRY_Representation`。确认在 `run()` 构造 agent 前执行。
4. **`AGENT_REGISTRY`** 加 `"CA_COMMAPPO": CA_ComMAPPO_Agents`。
5. **配置** `configs/ca_commappo/intersection-multi-agent-v1.yaml`（+ smoke）：`agent: "CA_COMMAPPO"`、`learner: "CA_ComMAPPO_Learner"`、`policy: "CA_ComMAPPO_Policy"`、`representation: "PermutationInvariantAttention"`、`output_dir: "runs/ca_commappo_highway/"`。其余超参与 MAPPO 对齐。
6. **视频前缀参数化**：按 §1.6 列举的所有 `mappo_test` 站点（`:23,24,93,198,245,254`）统一由 `configs.agent` 派生单一 `name_prefix`（CA-ComMAPPO 用 `ca_commappo_test`）；MAPPO 保持 `mappo_test`，并同步更新 `tests/test_mappo_video_recorder.py`。
7. **测试（TDD）**：新增 `tests/test_ca_commappo_*.py`：通信模块的 attention 权重对 mask 正确归一化、`b_ij` 单调性、forward 形状；agent 能在 highway smoke 配置下完成构造与一步前向。
8. **冒烟**：`uv run python main.py mappo --config configs/ca_commappo/intersection-multi-agent-v1.yaml --smoke --mode train --no-save` 跑通。
9. **门禁**：新测试全绿 + smoke 跑通 + 现有测试仍全绿。

### 阶段 4 · 实验矩阵 + runbook（多 seed、泛化、顺序扰动、通信消融、可视化）

目标：产出可复现的实验配置与启动脚本，重度训练交远端，AI 负责 smoke 验证与归档。

1. **指标扩展**：论文要求的 `average travel time`、`waiting time`、`throughput`、`success/pass rate`、`communication attention visualization` 当前不在 9-key summary。
   - `success/pass rate` ≈ `arrival_rate`（已覆盖）。
   - `average travel time` ≈ `mean_episode_length`（步数 × dt）——明确换算口径并记录（`dt = 1 / policy_frequency`）。
   - `throughput` = 到达车辆数 / episode 时长——在 `highway_metrics.py` 加 `throughput` 字段（从 `arrived_agents` 计数与 episode 长度推导）。
   - `waiting time` = 车速低于阈值的累计时间——**只往 adapter `info` 加 `agents_speeds` 不够**：向量化环境 `DummyVecMultiAgentEnv.step_wait`（`xuance/environment/vector_envs/dummy/dummy_vec_maenv.py:74`）每步用当前 info 覆盖 `buf_info`，episode 结束时只剩最后一步；`XuanCeMultiAgentEnvWrapper.step`（`xuance/environment/utils/wrapper.py:169-180`）只累计 `episode_step`/`episode_score`，其余 key 透传不累计。因此累计必须落在：(a) 扩展 `XuanCeMultiAgentEnvWrapper`（或项目子类）在 episode 内累计 `waiting_steps`（speed < 阈值），仿 `wrapper.py:172-176` 的 `episode_score` 累计；或 (b) 在 `evaluate_highway_policy` rollout 循环里每步读 `infos[i_env]['agents_speeds']` 累计。**推荐 (a)**：adapter `step()` 用 `self.env.unwrapped.controlled_vehicles` 在 `info` 加 `agents_speeds`（每车 `sqrt(vx^2+vy^2)`），再在 wrapper 累计低速步数。新增 `mean_waiting_time`。加验收测试：已知低速 episode 的 `mean_waiting_time > 0`。
   - 把新指标加进 `summarize_episode_records`（向后兼容：保留原 9 key，追加 `throughput`/`mean_waiting_time`/`mean_travel_time`）。**更新这两个测试**：`tests/test_highway_metrics.py`、`tests/test_mappo_highway_metrics.py`（它们按 exact-key 断言 summary）。**新指标仅用于报告，绝不改 `_summary_sort_key`**（best-model 选择仍是 arrival↑→collision↓→reward↑→length↓）；加测试断言 sort key 不变。**所有算法共用同一套指标**。
2. **多 seed 配置**：为每个算法（IPPO、MAPPO-MLP、Attention-MAPPO、CommNet、IC3Net、CA-ComMAPPO）准备 seed 列表 `[1,2,3,4,5]` 的训练配置（通过 `--seed` 覆盖或写多份 YAML）。固定 `env_seed`、`test_episode`、`--seed`、`--video-seed` 以保证横向可比。**MAAC 与 MADDPG 按论文 `thesis-planning.md` 的「可选」标记有意暂缓**（MADDPG 不适合离散动作主线；MAAC attention critic 可后续补作对照），在 runbook 记录该决策。**远端多 seed 必须用 `--mode benchmark`**（才会落盘 `eval_records.jsonl` + `best_model.pth`，见 §0.6）；纯 `--mode train` 不产 eval 记录，阶段 5 无法聚合。最终数字取自 best_model 的 `--mode test`。
3. **泛化实验**：训练 `controlled_vehicles=4`，测试 `controlled_vehicles ∈ {6,8,10}`。建 `configs/<algo>/intersection-multi-agent-v1-gen4.yaml`（训练）与 eval 配置（测试时改 `controlled_vehicles`）。注意 `state_space`/obs 形状随车数变化——PI 编码器天然支持变车数，MLP 基线需要固定输入维度（论文要点：PI 方法稳、MLP 不稳）。
4. **顺序扰动实验**：写一个 eval 包装器（`ca_commappo/evaluation/order_perturbation.py`），在 rollout 时随机打乱观测中车辆顺序，统计指标变化。证明 MLP 基线指标波动大、PI 方法稳定。
5. **通信消融**（config 开关控制，每个变体一份 config；与论文 §4.5 对齐）：(a) 无通信；(b) 全连接平均通信（退化为 mean，即 CommNet 对照）；(c) attention 通信但 `b_ij=0`（去冲突先验，保留 attention）；(d) conflict-aware attention 通信（完整 CA-ComMAPPO）。另外 (e) 去 PI 编码器（换 `Basic_MLP`）即等价于 MAPPO-MLP 基线，可直接复用阶段 1 的 MAPPO 结果，不必重复跑。**注意论文 §4.5 把「去掉 attention」与「换成 CommNet 平均通信」列成两项**：若评审需要区分「去 attention（求和/恒等）」与「CommNet mean」，则在 (b) 之外再加 `(b')` 去 attention 变体；否则按 (b) 合并并在论文中说明等价。在 runbook 明确每个变体的开关语义。
6. **注意力可视化导出器**：`ca_commappo/evaluation/attention_export.py`，在 eval 时导出每个 agent 的 `α_ij` 矩阵（通信注意力）与编码器 attention 权重，存 JSON/热力图（matplotlib），供论文可解释性图。
7. **runbook**：写 `docs/experiments_runbook.md`，列出每个实验的：配置路径、命令、seed 列表、预期产物路径、远端启动方式（wandb）、结果收集命令。每个实验都先 smoke 验证再交远端。
8. **门禁**：每个新配置的 smoke 跑通；指标扩展测试全绿；runbook 命令可复制粘贴执行。

### 阶段 5 · 结果归档 + 绘图 + 可复现文档

目标：把远端训练产出的 `eval_records.jsonl` 聚合成论文级表格与图。

1. **聚合脚本** `ca_commappo/evaluation/aggregate_results.py`：跨 seed 读取各算法的 `eval_records.jsonl`，输出 mean±std 表（collision rate、arrival rate、throughput、travel time、waiting time、mean reward）。
2. **绘图脚本** `ca_commappo/evaluation/plot_results.py`：生成 (a) 各算法主指标柱状图（mean±std）、(b) 泛化曲线（车数 vs 指标）、(c) 顺序扰动稳定性对比、(d) 通信消融对比、(e) 注意力热力图。遵循 `dataviz` 规范（如有）。
3. **可复现文档** `docs/experiments_results.md`：记录每个实验的 git commit、配置、命令、产物路径、结果表与图、结论。
4. **门禁**：聚合脚本在至少一个真实 run 上跑通；图表生成无报错；文档完整。

---

## 4. 验收门禁（每阶段必做）

每个阶段结束时，**必须**执行并展示输出（不得仅声称通过）：

- 代码风格：`ruff check --fix; ruff format`。
- 测试：`uv run pytest -q`（改动核心逻辑或准备提交时）；新测试单独 `uv run pytest -q tests/<新测试>.py`。
- smoke 训练：用该阶段正式 YAML 加 `--smoke --mode train --no-save`，确认不报错、能完成构造与前向/反向。
- 仅当门禁全绿才提交。提交信息用 Conventional Commit；提交说明里记录跑了哪些命令。

---

## 5. 远端训练交接（runbook 约定）

- 重度多 seed 训练在远端服务器跑（正式配置 `logger: "wandb"`，远端需 `wandb login` 或 `WANDB_MODE=offline`）。本地只做实现 + smoke（tensorboard）+ 归档。
- **远端 run 必须用 `--mode benchmark`**：只有 benchmark/test 模式才落盘 `eval_metadata.json` + `eval_records.jsonl`（含 `config_path`、`seed`、`highway_config` 快照）；`--mode train` 不产 eval 记录（见 §0.6）。
- 横向对比固定：YAML（含 `env_seed`）、`test_episode`、`--seed`、`--video-seed`。
- `--mode benchmark` 会自动按 best-key 存 `best_model.pth`；最终验证用 `best_model.pth` + `--mode test`。
- 视频对比固定 `--video-seed`，第 i 个视频用 `video_seed + i`。
- 收尾前确认每个远端 run 目录下 `eval_records.jsonl` 存在且非空，否则阶段 5 无法聚合。

---

## 6. 已知陷阱清单

1. **不要编辑 `.venv` 里的 xuance**——用运行时注册（`xuance_compat.py` 同款）。
2. 省略 `--config` 只选择默认 MAPPO；其他算法必须显式传对应 YAML。
3. **CommNet/IC3Net 必须 `use_rnn: true` + `Basic_RNN`**（非可选），且 `action()` 需 `info`——见 §1.4 与阶段 2 的运行时补丁。`--mode train` smoke 不暴露此问题，必须额外跑 `--mode test`。
4. **`MAPPO_Agents._build_policy` 限制 policy 字符串**——CA-ComMAPPO 要继承并重写，或注册新 policy 并放宽检查。
5. **视频前缀 `mappo_test` 硬编码**——新算法需参数化，否则误导。
6. **`highway_config` 的 deep-merge 由适配器负责**——highway-env 1.11 自身是浅 update，但本项目适配器 `build_intersection_config()` 已用 `_deep_merge` 把 `highway_config` 深度合并进默认值（见 §0.5 与 `highway_intersection_wrapper.py:17-24,48-61`），故 `observation`/`action` 可给**部分覆盖**（现有 smoke 仅给 `observation_config.vehicles_count/features` 与 `action_config.target_speeds` 即可工作）。无需给完整结构。
7. **`info["crashed"]` 必须用 per-agent tuple**，不要用 highway-env 原始单车字段。
8. **masked reward vs raw reward**——训练用 masked，分析可看 raw，但统计口径统一用 masked。
9. **泛化实验中 MLP 基线输入维度固定**——`controlled_vehicles` 变化时 MLP 会维度不匹配，这是论文要论证的点（PI 方法天然变长）。
10. **自定义 Representation 要 `**kwargs`**——`_build_representation` 会传一堆无关 kwargs（kernels/strides/filters 等），不吞会 `TypeError`。
11. **运行时注册必须在 `run()` 构造 agent（第 393 行）之前**——否则 `KeyError`。
12. **`eval_metadata.json` 只写一次**——重跑同目录要先换 `output_dir` 或清空，否则 `FileExistsError`。

---

## 附录 A：CA-ComMAPPO 数学定义与实现映射

来自 `docs/thesis-planning.md` §3，对应实现位置：

| 论文公式 | 含义 | 实现 |
|---|---|---|
| `e_ij = φ(o_i, o_j)` | 自车 o_i 与邻车 o_j 的配对编码 | `PermutationInvariantAttention.forward` 内 |
| `z_i = AttnPool({e_ij})` | 顺序无关聚合 | 同上，masked softmax |
| `m_i = f_m(z_i, intent_i, ttc_i)` | agent i 发送的消息 | `CA_ComMAPPO_Policy` 消息头 |
| `c_i = Σ_{j≠i} α_ij W_v m_j` | 冲突感知消息聚合 | `ConflictAwareComm.forward` |
| `α_ij = softmax(q_i^T k_j + b_ij)` | 注意力权重 + 冲突先验 | 同上，`b_ij` 为 kinematic 冲突 proxy |
| `π_i(a_i \| z_i, c_i)` | 去中心化 actor | `CA_ComMAPPO_Policy` |
| `V(s)` 或 `V({z_i,m_i,a_i})` | 中心化 critic | `CA_ComMAPPO_Learner`（先复用 MAPPO critic） |

消融（论文 §4 第 5 点，权威清单见阶段 4.5）：(a) 去通信；(b) 换 CommNet 平均通信（去 attention 退化为 mean）；(c) 去冲突先验（`b_ij=0`，保留 attention）；(d) 完整 CA-ComMAPPO；(e) 去 PI 编码器（换 `Basic_MLP`，即 MAPPO-MLP 基线，可复用阶段 1 结果）。若评审要求区分「去 attention」与「CommNet mean」，见阶段 4.5 的 `(b')` 变体。

---

## 附录 B：评价指标映射（论文 §5）

| 论文指标 | 实现口径 |
|---|---|
| collision rate | `collision_rate`（9-key 之一） |
| success / pass rate | `arrival_rate` |
| average travel time | `mean_episode_length` × dt（明确换算） |
| waiting time | 新增 `mean_waiting_time`（速度低于阈值累计） |
| throughput | 新增 `throughput`（到达数 / episode 时长） |
| average reward | `mean_episode_reward` / `mean_agent_reward` |
| generalization to vehicle number | 泛化实验曲线 |
| communication attention visualization | `attention_export.py` 热力图 |

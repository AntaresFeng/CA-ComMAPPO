# CA-ComMAPPO 源代码独立分析报告

> 分析范围：所有被 `git` 追踪的 `.py` 源文件（排除 `tests/`）加上未追踪的新增文件。不信任任何文档/注释，仅依据源代码。
>
> 分析日期：2026-07-02 | 方法：四个子代理并行阅读源码后人工综合

> 结构迁移更新：2026-07-02 已将本项目维护的 CLI、训练入口和 highway 配置从 `examples/` 迁出。当前项目代码位于 `ca_commappo/cli/`、`ca_commappo/training/` 和 `configs/`；`examples/mappo/` 只保留 vendored XuanCe 上游参考。已删除的 `evaluate_*` 系列不纳入本轮重构，等待后续重新设计。

---

## 目录

1. [完整文件清单](#1-完整文件清单)
2. [各脚本作用详解](#2-各脚本作用详解)
3. [功能重叠与重复代码](#3-功能重叠与重复代码)
4. [职责不清与命名误导](#4-职责不清与命名误导)
5. [MAPPO Baseline 训练-测试-验证流程完备性评估](#5-mappo-baseline-训练-测试-验证流程完备性评估)
6. [综合建议](#6-综合建议)

---

## 1. 完整文件清单

| 路径 | 行数 | 角色 |
|---|---|---|
| `ca_commappo/__init__.py` | 2 | 包标记 |
| `ca_commappo/envs/__init__.py` | 14 | 环境模块导出 |
| `ca_commappo/envs/highway_intersection.py` | 254 | **核心：highway-env → XuanCe 适配器** |
| `ca_commappo/evaluation/__init__.py` | 1 | 包标记 |
| `ca_commappo/xuance_compat.py` | 27 | **XuanCe 版本兼容 shim** |
| `ca_commappo/evaluation/sanity_baseline_runner.py` | 199 | **sanity baseline 执行器** |
| `main.py` | 34 | 顶层薄 dispatcher |
| `ca_commappo/cli/run_sanity_baseline.py` | 68 | 兜底策略评估执行器 |
| `ca_commappo/envs/debug_highway_wrapper.py` | 598 | 单 episode 逐帧调试工具 |
| `examples/evaluate_highway_intersection.py` | 已删除 | 旧 JSON 后处理器，等待后续评估链路重构 |
| `ca_commappo/training/mappo_highway_intersection.py` | 242 | **唯一可运行的 MAPPO 训练入口** |
| `examples/mappo/mappo_simple_spread.py` | 97 | 含 bug，不可运行 |
| `examples/mappo/mappo_football.py` | 228 | 含 bug，不可运行 |
| `examples/mappo/mappo_sc2.py` | 251 | 含 bug，不可运行 |

---

## 2. 各脚本作用详解

### 2.1 `ca_commappo/envs/highway_intersection.py` — 环境适配器

默认将 highway-env 的 `intersection-multi-agent-v1` 包装为 XuanCe `RawMultiAgentEnv`；显式传入 `intersection-v1` 时仍保留兼容路径，但主线配置不再使用它。

- **`HighwayIntersectionMultiAgentEnv`**（L63-249）
  - 读 `env_config`（duck-typed Namespace）的 `env_id` / `render_mode` / `flatten_observations` / `highway_config`。
  - `build_intersection_config()` 组装嵌套 config → `gym.make()` 造底层 env。
  - `controlled_vehicles` 决定 agent 数，命名 `agent_0..agent_{n-1}`，单组 CTDE。
  - **step**：调底层 step → 取 per-agent reward → `_mask_rewards` 对 inactive 置 0 → 返回的 `rewards` 和 `info["agents_rewards"]` 都使用 masked adapter-facing reward → 原始 highway reward 放 `info["raw_agents_rewards"]` → `terminated` 取 per-agent terminated → `info["global_terminated"]` 表示环境级结束 → `info["crashed"]` / `info["arrived"]` 暴露 per-agent episode facts → `truncated` 透传标量 → 更新 `_active_agents` 状态机。
  - `state()`（L225-230）= 各 agent obs 拼接（**不是真正全局状态，不含非受控车**）。
  - `IDLE_ACTION = 1`（L13）硬编码，假设 highway 动作表 `{0:SLOWER,1:IDLE,2:FASTER}`。
- **`build_intersection_config()`**（L47-60）：合并默认 config + 用户 override，校验 `controlled_vehicles > 0`。
- **`register_highway_intersection_env()`**：注册到 XuanCe `REGISTRY_MULTI_AGENT_ENV`，键默认 `"HighwayIntersection"`；模块导入时会自动注册默认键。

### 2.2 `ca_commappo/xuance_compat.py` — XuanCe 兼容 shim

- **`patch_xuance_marl_buffer_aliases()`**：集中处理 XuanCe 1.4.3 `BaseBuffer` 字段名漂移，把 `num_envs` / `per_env_buffer_size` / `observation_space` / `action_space` 暴露为旧版 `n_envs` / `n_size` / `obs_space` / `act_space`。训练入口复用该函数，不再在训练脚本内联 monkey patch。

### 2.3 `ca_commappo/evaluation/sanity_baseline_runner.py` — Sanity baseline 执行器

- **`SUPPORTED_POLICIES = ("random", "idle-only")`**。`idle-only` 使用从 adapter 导入的 `IDLE_ACTION`。
- **`run_episode()`**（L73-119）：每步从 adapter step info 读取 `info["crashed"]` / `info["arrived"]`，命中 collision/arrival/truncated 即 break。episode record 和 summary 交给共享 `highway_metrics` helper。
- **`run_sanity_baseline()`**（L122-153）：三层循环（policy × seed × episode），用 `summarize_episode_records` 聚合。`episode_seed = base_seed + episode_index * len(config.seeds)`（L140），与 record 的 `episode_index` 字段（L145）两套编号。
- 旧的 `_controlled_vehicle_flags` / `controlled_vehicle_flags` sanity 路径已移除；sanity runner 不再穿透底层 env 读取车辆状态。
- **`_make_env`**（L165-168）：用 `argparse.Namespace` 手搓 env_config，**不传 `flatten_observations` / `render_mode` / `env_seed`**，adapter 大半开关未用。

### 2.4 `main.py` — 顶层 CLI

- 当前是薄 dispatcher，不再内联环境或训练逻辑。
- `debug-wrapper` → 转发 `ca_commappo.envs.debug_highway_wrapper`
- `sanity` → 转发 `ca_commappo.cli.run_sanity_baseline`
- `mappo` → 转发 `ca_commappo.training.mappo_highway_intersection`

### 2.5 `ca_commappo/cli/run_sanity_baseline.py` — 兜底评估 CLI

- 委托 `sanity_baseline_runner.run_sanity_baseline()` 跑环境 + 聚合。
- `--policy` choices 复用 `SUPPORTED_POLICIES` 并追加 `all`。
- `print_summary()` 本地拼接（L39-49），**不复用 `format_summary_lines`，缺 `mean_per_agent_reward` 字段**。

### 2.6 `ca_commappo/envs/debug_highway_wrapper.py` — 调试工具

- 单 episode 逐帧打印，支持 `--target raw / wrapper / both`。
- `raw` 用 `gym.make`；`wrapper` 用 `HighwayIntersectionMultiAgentEnv`，当前 argparse 描述已改为 CA-ComMAPPO adapter。
- raw debug 目标仍可直接打印底层车辆状态；wrapper/sanity/MAPPO 评估路径应消费 adapter 暴露的 `info["crashed"]` / `info["arrived"]`，指标汇总逻辑由共享 `highway_metrics` 模块负责。
- 默认 `controlled_vehicles=2`、`duration=15`（L21-27），与 sanity yaml（3、13）不一致。

### 2.7 `examples/evaluate_highway_intersection.py` — 已删除，待后续重构

- 旧版本是纯 JSON 后处理器，不开 env、不加载模型、不跑 rollout。
- 本轮结构迁移不恢复它；评估链路和 MAPPO rollout JSON 输出等待后续重构。

### 2.8 `ca_commappo/training/mappo_highway_intersection.py` — 唯一的 MAPPO 训练入口

- **`run()`**：调用 `patch_xuance_marl_buffer_aliases()`（来自 `ca_commappo.xuance_compat`）→ `make_envs` → `MAPPO_Agents` → 模式分支。环境注册由 adapter 模块导入时自动完成。
- **`train()`**（L139-144）：只调 `agents.train()` + 可选 `save_model("final_train_model.pth")`，**不评估**。
- **`benchmark()`**：周期性 `agents.test()`，但**只用 episode reward mean/std 选 best model，不计算 collision/arrival/episode length 等任务指标**。
- **`test()`**（L147-156）：加载模型，只打印 `Mean Score / Std`。
- `--env-id` 默认 `intersection-multi-agent-v1`，内置配置文件名和 YAML 里的 highway `env_id` 已对齐；`env_name: HighwayIntersection` 仍是 XuanCe 注册名。

### 2.9 三个不可运行的 mappo example

- **`mappo_simple_spread.py`**（L20-22）：`parser.env_id` — AttributeError，`parser` 是 ArgumentParser 实例，无 `.env_id` 属性。
- **`mappo_football.py`**（L218-221）：同上 bug，且 `runner.run(mode="benchmark")` 但方法签名 `def run(self)`（L69）不接受 `mode` → TypeError。
- **`mappo_sc2.py`**（L241-244）：同上两个 bug。与 `football` 几乎逐字复制，仅指标字段数不同。
- **结论：这三个是上游 XuanCe 模板，CA-ComMAPPO 未实际维护，无法直接运行。**

---

## 3. 功能重叠与重复代码

### 3.1 完全逐字复制的代码

| 代码片段 | 位置 1 | 位置 2 | 差异 |
|---|---|---|---|
| 车辆状态读取 | adapter `step()` / debug raw 目标 | sanity/MAPPO 消费 adapter `info` | 训练评估不再重复穿透底层 env；debug raw 仍用于人工探查 |

### 3.2 语义重复 / 未共享常量的代码

| 语义 | 定义处 | 硬编码处 | 风险 |
|---|---|---|---|
| IDLE 动作 = 1 | `highway_intersection.py`（`IDLE_ACTION`） | `sanity_baseline_runner.py` import 复用 | 已共享 |
| highway-env 默认 id | `DEFAULT_HIGHWAY_ENV_ID` | adapter、sanity runner、debug wrapper、MAPPO 训练入口复用 | 已有共享默认常量 |

### 3.3 三处分散的打印格式

- `run_sanity_baseline.py:39-49`：本地拼接 summary 输出。
- 当前没有共享 summary formatter；后续若恢复 MAPPO 评估后处理，应统一打印和 JSON 字段口径。

### 3.4 四份 mappo example 的大量样板重复

- argparse 框架（`--env-id` / `--test` / `--benchmark`）：四份各写一套
- config 加载三段式（`load_yaml` → `recursive_dict_update` → `argparse.Namespace`）：四份各写一套
- benchmark 循环（复制 config → create test_envs → 初始 test → train/test 交替 → 比较 mean → 保存 best）：`mappo_highway_intersection.py` 与 `mappo_simple_spread.py` 几乎一致
- `mappo_football.py` 与 `mappo_sc2.py`：Runner 类的 `__init__`、`run`、`benchmark`、`test_episodes`、`get_battles_info`、`get_battles_result`、`time_estimate` **几乎逐字复制**，仅指标字段数不同

---

## 4. 职责不清与命名误导

### 4.1 严重命名误导

| 文件名 | 实际行为 | 应为 |
|---|---|---|
| `examples/evaluate_highway_intersection.py` | 已删除 | 待后续重新设计评估链路 |
| `ca_commappo/cli/run_sanity_baseline.py` | 跑 random + idle-only 进行兜底评估 | 已从 `examples/` 迁出 |
| `ca_commappo/envs/debug_highway_wrapper.py` | 直接 new `HighwayIntersectionMultiAgentEnv`，无 XuanCe | 描述已改为 CA-ComMAPPO adapter |

### 4.2 Episode Fact 边界已收敛到 adapter

```
环境 adapter (highway_intersection.py):
    step() → info["agents_terminated"] / info["global_terminated"] / info["crashed"] / info["arrived"] / _active_agents 状态机

评估层 (sanity/MAPPO evaluator/callback):
    只消费 adapter step info 里的 per-agent episode facts，再交给 highway_metrics 聚合
```
- adapter 对 `terminated` 加了"全局 terminated 时强制全员 True"的语义（L213-214）
- `crashed` / `arrived` 的 raw highway-env 状态读取集中在 adapter，sanity 和 MAPPO 评估不再自行穿透 env。
- 这样可以让 Dummy/Subproc vector env、XuanCe callback、sanity baseline 使用同一份 episode fact contract。

### 4.3 `info["agents_rewards"]` 语义已定稿

`highway_intersection.py` step：
- 返回值 `rewards` = **掩码后的** adapter-facing rewards（inactive agent 置 0）。
- `info["agents_rewards"]` 与返回的 `rewards` 保持一致，也是 masked reward tuple。
- 原始 highway per-agent reward 保存在 `info["raw_agents_rewards"]`。
- 该 contract 已写入 README、AGENTS 和 `docs/highway_intersection_notes.md`。

### 4.4 `register_highway_intersection_env` 已自动注册默认键

- `highway_intersection.py` 模块底部自动调用 `register_highway_intersection_env()`。
- 训练入口和测试不再需要手动注册默认 `"HighwayIntersection"`。
- `register_highway_intersection_env(env_name=...)` 仍保留给自定义注册名使用。

### 4.5 默认配置三套不共享真源

| 配置来源 | `controlled_vehicles` | `duration` | `arrived_reward` | `collision_reward` |
|---|---|---|---|---|
| `run_sanity_baseline.py` → yaml | 3 | 13 | 默认 | 默认 |
| `debug_highway_wrapper.py` → `DEFAULT_HIGHWAY_CONFIG` | 2 | 15 | 5 | -10 |
| `mappo_highway_intersection.py` → yaml | 2 | 不同 | 默认 | 默认 |

无共享默认常量的真源，同一语义不同默认值。

### 4.6 env_id 命名已收敛，仍需区分 XuanCe 注册名

| 出现位置 | 字符串 | 用途 |
|---|---|---|
| `mappo_highway_intersection.py` argparse `--env-id` 默认值 | `intersection-multi-agent-v1` | 同时用于找同名内置 config 文件 |
| mappo yaml `env_id` 字段 | `intersection-multi-agent-v1` | 传给 `gym.make` |
| mappo yaml `env_name` 字段 | `HighwayIntersection` | XuanCe 注册名 |
| `register_highway_intersection_env()` 默认参数 | `HighwayIntersection` | 注册表键 |

highway-env id 和内置配置文件名已经对齐；仍需在文档和代码里明确区分 highway-env id 与 XuanCe 注册名。

### 4.7 mappo example 职责越界

`mappo_highway_intersection.py`：
- XuanCe buffer alias patch 已移到 `ca_commappo/xuance_compat.py`。
- 环境注册已由 adapter 模块导入时自动完成。
- 训练入口仍负责训练/测试/benchmark 流程和调用 XuanCe 兼容 shim。

---

## 5. MAPPO Baseline 训练-测试-验证流程完备性评估

### 5.1 当前架构图

```
训练端:                       评估端:
mappo_highway_intersection.py  待重建的 MAPPO 评估入口
  │                              │
  │ train(): agents.train()      │ 只读 JSON, 不跑 env, 不加载模型
  │  只保存 .pth                 │ 期望 {policies:{...}} 或 {episodes:[...]}
  │  REWARD ONLY, 无碰撞率等     │ 评估后处理入口待重建
  │                              │
  │ benchmark(): agents.test()   │ 当前唯一 JSON 来源:
  │  也只输出 reward mean/std    │ run_sanity_baseline.py
  │                              │  仅 random/idle-only 策略
  │ 不产出任务指标               │
  │ 不产出结构化 episode JSON     │
  │                              │
  └─── 无连接 ──────────────────┘

对照侧:
sanity_baseline_runner.py
  SUPPORTED_POLICIES = ("random", "idle-only")
  度量的是环境本身的碰撞/到达/超时几何属性, 不是策略好坏
```

### 5.2 完备性判定

| 环节 | 状态 | 问题 |
|---|---|---|
| **① MAPPO 训练** | ✅ 基本可用 | `train`/`benchmark`/`test` 三种模式，有 best-model 选择 |
| **② 训练阶段输出任务指标** | ❌ 缺失 | train/benchmark/test **均不产出** collision rate / arrival rate / episode length |
| **③ MAPPO rollout → 结构化 JSON** | ❌ 缺失 | 训练脚本不 dump 每个 episode 的含指标记录 |
| **④ MAPPO 评估 CLI** | ❌ 缺失 | 旧 `evaluate_highway_intersection.py` 已删除，后续需重建 |
| **⑤ Baseline 对照** | ❌ 缺失 | sanity baseline 只量环境属性（random/idle 的几何投影），不构成 MAPPO 性能标尺 |
| **⑥ Baseline vs MAPPO 口径对齐** | ❌ 缺失 | `episode_reward=sum/num_agents` vs `mean_agent_reward` 全局池均值口径不同；无对照表聚合 |
| **⑦ 注册与环境联动** | ✅ 已自动注册 | 默认 `HighwayIntersection` 在 adapter 模块导入时注册 |
| **⑧ 非 highway example** | ❌ 不可运行 | 三个含 bug 无法直接运行，不能作为 baseline 参照 |

### 5.3 严重程度排序

1. **P0 — 训练管道与评估管道完全断开**
   - 训练侧不出结构化 episode JSON（仅 reward）
   - 评估侧的共享指标模块和后处理入口尚未重建
   - **无法回答"MAPPO 比随机 / idle 基线好多少"**

2. **P0 — Sanity baseline 不构成 MAPPO 性能标尺**
   - `random` 和 `idle-only` 的 collision/arrival/truncation 反映的是环境几何结构和动力学约束
   - mean_episode_reward 口径与训练侧不对齐
   - baseline vs MAPPO 无对照表设计

3. **P1 — 终止语义已收敛到 adapter fact contract**
   - adapter `_active_agents` 状态机、`info["global_terminated"]`、`info["crashed"]`、`info["arrived"]` 是 sanity/MAPPO 评估共同依赖的 episode 边界。
   - 后续改 adapter 终止语义时，应同步保持这些 `info` 字段和测试断言。

4. **P1 — `info["agents_rewards"]` 语义已定稿**
   - 当前 contract：返回 `rewards` 和 `info["agents_rewards"]` 都是 masked，`info["raw_agents_rewards"]` 保存原始 highway reward。

5. **P2 — 大量重复代码 + 配置不共享真源**
   - 三个打印格式、两个 `save_results_json`；训练/评估指标汇总正在向共享 `highway_metrics` 收敛
   - 三套默认配置数值不一致

6. **P2 — env_name/env_id 双层概念仍需理解**
   - 自动注册已解决默认 XuanCe 注册问题。
   - highway-env id 与 XuanCe 注册名仍是两层概念。

7. **P3 — 三个非 highway example 含 bug 不可运行**
   - `parser.env_id` AttributeError
   - `runner.run(mode=...)` TypeError

---

## 6. 综合建议

### 短期（文件级修复）

1. **`sanity_baseline_runner.py`**
   - `idle-only` 动作 1 改为 import `IDLE_ACTION` 引用
   - 已删除冗余 `_controlled_vehicle_flags` 包装，改为消费 adapter step info
   - break 条件中的 `truncated` 与 `episode_outcome` 对齐语义

2. **评估后处理链路**
   - 旧 `evaluate_highway_intersection.py` 已删除。
   - 后续应重新设计 MAPPO rollout JSON 输出和汇总入口。

3. **`ca_commappo/envs/debug_highway_wrapper.py`**
   - raw 目标保留底层车辆探查逻辑；wrapper 目标应以 adapter `info["crashed"]` / `info["arrived"]` 为准
   - argparse description 中 "XuanCe wrapper" 措辞已在结构迁移中修正

4. **共享指标模块**
   - 抽取 collision/arrival/outcome/summary 逻辑，统一 sanity、debug 和后续 MAPPO 评估口径

### 中期（架构调整）

5. **打通训练 → 评估链路**
   - 在 `mappo_highway_intersection.py` 的 `benchmark()` / `test()` 中劫持 episode 回调，记录 `collision`/`arrival`/`episode_length` 等，dump 成后续评估入口可消费的 JSON 格式
   - 或在 xuance 的 `test` 循环后调用共享指标模块计算任务指标

6. **统一配置真源**
   - `controlled_vehicles`、`duration`、`DEFAULT_ENV_NAME`、`IDLE_ACTION` 等常量从一处 (`highway_intersection.py` 或新建 config 模块) 导出，下游 import

7. **注册自动化**
   - 已完成：`highway_intersection.py` 导入时自动注册默认 `HighwayIntersection`。

8. **抽取公共训练模板**
   - 建议 `examples/mappo/_mappo_common.py`：`build_mappo_parser()`、`load_mappo_config()`、`print_train_information()`、`benchmark_loop()`、`test_loop()`
   - 建议 `examples/mappo/_teambattle_runner.py`：football / sc2 共用 Runner 基类

### 长期（流程闭环）

9. **建立 Baseline vs MAPPO 对照表**
   - `summarize_evaluation_results` 增加 `baseline vs trained` 对照聚合入口
   - 统一 reward 口径（统一用 `sum(agent_rewards)/num_agents` per episode then average）

10. **终止语义统一**
    - 已选方向：adapter 是 episode fact 的统一出口，evaluation 消费 adapter 暴露的字段，不再穿透底层 env。
    - 添加或保留断言，确保 `info["crashed"]` / `info["arrived"]` 与 `terminated`、`global_terminated`、mask 行为一致。

---

## 附录：各子代理报告的原始关键引用

| 文件 | 关键行 | 引用概要 |
|---|---|---|
| `highway_intersection.py` | L13 | `IDLE_ACTION = 1` 硬编码 |
| `highway_intersection.py` | L205 | `_reward` 丢弃，不用底层 reward |
| `highway_intersection.py` | L213-214 | 全局 terminated 时强制全员 True |
| `highway_intersection.py` | step | `rewards` 与 `info["agents_rewards"]` 均为 masked reward |
| `highway_intersection.py` | step | `raw_agents_rewards` 存原始 highway reward |
| `highway_intersection.py` | module import | 默认 `HighwayIntersection` 自动注册 |
| `sanity_baseline_runner.py` | L61 | `{agent: 1}` 独立硬编码 IDLE 动作 |
| `sanity_baseline_runner.py` | 旧 L171-174 | 冗余 `_controlled_vehicle_flags` 已移除；当前消费 adapter info |
| `sanity_baseline_runner.py` | L98 | break 用原始 `truncated`，与 outcome 语义不同 |
| `mappo_highway_intersection.py` | L111-136 | monkey-patch 藏在 example 里 |
| `mappo_highway_intersection.py` | L139-144 | train() 不评估，只 `.pth` |
| `mappo_highway_intersection.py` | L147-156 | test() 只打 Mean Score/Std |
| `mappo_highway_intersection.py` | L159-208 | benchmark() 只用 reward mean/std 选 best |
| `mappo_highway_intersection.py` | L212 | 注册需训练脚本显式调用 |
| `mappo_simple_spread.py` | L21-22 | `parser.env_id` AttributeError bug |
| `mappo_football.py` | L219-227 | 同上 bug + `run(mode=...)` TypeError |
| `mappo_sc2.py` | L242-250 | 同上两个 bug |

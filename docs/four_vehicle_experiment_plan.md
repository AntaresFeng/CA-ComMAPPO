# 四车论文实验执行计划

## 目标与当前定位

本计划记录四车论文实验从质量门禁到最终归档的执行顺序和完成状态，并以当前仓库、
canonical YAML 和 `docs/` 根目录文档为准。论文主线固定为四个 controlled vehicles，
不开展 4→6/8/10 车辆泛化，也不据此声明规模泛化能力。

历史 MAPPO I5 seed 1 与早期 CA-ComMAPPO seed 1 仅保留为预实验参考。核心正式结果
已在提交 `199ead5` 上按训练 seeds `[1, 2, 3]` 重新生成；IPPO 修复后形成独立的
`d50ed20` 扩展矩阵，两者按 provenance 规则分别聚合。

## 阶段一：质量门禁与 provenance v2

1. 清理全仓库 Ruff/format 存量问题，运行完整 pytest，形成独立质量门禁提交。
2. 将模型 provenance 升级为 snapshot-stable v2：
   - sidecar 记录 Git commit、tracked-tree clean 状态和逐文件 SHA manifest；
   - 正式 benchmark/test 拒绝脏工作树；
   - final test 仍要求当前 runtime 与 checkpoint 完全一致；
   - 聚合时校验 checkpoint、sidecar 和 final artifact，而不依赖聚合时工作树；
   - 同一正式 manifest 中的学习算法必须来自同一 Git commit/runtime SHA；
   - v1 artifact 只作历史参考，不进入新的正式结果表。
3. 覆盖篡改、脏树、跨提交混用、历史聚合和旧 schema 拒绝等回归测试。

## 阶段二：论文评估工具

### 顺序扰动

- 环境 cases 固定为 seeds `20000..20099`。
- 只打乱每个 agent 观测中 ego 之后的车辆行，整行特征一起移动；ego、全局状态、
  agent mask 和环境协议保持不变。
- permutation seeds 固定为 `[30000, 30001, 30002]`。
- 输出原始/扰动逐 case 配对记录和指标差值。
- 正式用于 MAPPO 与 Attention-MAPPO，CA-ComMAPPO 作为辅助稳定性检查。

### Attention/TTC 导出

- 使用不会改变正常 policy 输出的只读 trace context。
- 导出 active mask、scaled TTC、intent、effective lambda、`alpha_ij`、动作和轨迹。
- 固定使用 full/no-prior 的训练 seed 1 checkpoint 和 cases `20000..20009`。
- 输出机器可读 JSON 和 PNG/PDF 热力图。

### 统计与绘图

- 主统计单位为训练 seed，报告三 seed mean、sample std 和逐 seed原始值。
- MAPPO→Attention、Attention→CA、CA→no-prior 使用相同训练 seed 的配对差。
- 固定 case 的配对统计只作辅助，不替代训练 seed 方差。
- 生成主结果表、顺序扰动图、消融图和解释性图。

`main.py paper-analysis` 复用严格 final-test 聚合门禁，默认要求核心三组
配对齐全，输出逐 seed 原始值、sample std、方向明确的配对效应量与精确双侧 sign-flip
检验。三个 seed 的 p 值仅作探索性辅助。主结果、配对消融与顺序扰动图输出 PNG/PDF；
CA trace 工具生成的 attention/TTC heatmap 构成解释性图。正式输出已归档到
`results/core_matrix_199ead5_paper_analysis/` 和 `results/core_matrix_199ead5_ca_trace/`。

## 阶段三：全算法 smoke 与源码冻结

在最终源码上依次验证 MAPPO、Attention-MAPPO、CA-ComMAPPO full、CA-ComMAPPO
no-prior、IPPO、CommNet 和 IC3Net。所有算法必须通过真实 action/update smoke；通信
算法还要通过 benchmark checkpoint、严格 test 和单集录像。最终执行 Ruff、format、
完整 pytest 和 NaN/Inf 检查，记录冻结 commit 与 runtime SHA。

冻结后不得修改代码、配置或依赖锁，直至全部正式 final test 完成。若发现必须修复的
实现缺陷，应停止实验、修复并产生新冻结提交；此前该轮 runs 全部降级为预实验。

## 阶段四：正式多 seed 实验

正式任务统一设置 `WANDB_MODE=offline`，使用 `--mode benchmark`。长任务先固化为
`runs/launch_scripts/` 脚本，再通过独立具名 tmux session 启动；本机一次只运行一个
长任务。既有正式矩阵使用 CPU，2026-10-10 重启后 CUDA 已通过真实 action/update
smoke，后续新任务可在同一比较矩阵统一使用 `--device cuda:0`。

核心矩阵优先：

1. MAPPO seeds 1/2/3
2. Attention-MAPPO seeds 1/2/3
3. CA-ComMAPPO full seeds 1/2/3
4. CA-ComMAPPO no-prior seeds 1/2/3

随后扩展：IPPO、CommNet、IC3Net，各运行 seeds 1/2/3。

核心矩阵已在冻结提交 `199ead5d8b2fababa649d3aa94c1c025eb08d682` 完成：MAPPO、
Attention-MAPPO、CA-ComMAPPO full 和 no-prior 各有 seeds 1/2/3 的正式 benchmark、
100-case final test 与 6-case 视频。IPPO 的首个正式 run 暴露了 XuanCe
`DummyVecMultiAgentEnv` 会原地改变共享 `env_seed` 的问题，因此该失败 run 不属于正式
结果。IPPO 改用 `SubprocVecMultiAgentEnv` 后，与 CommNet、IC3Net 一起在后续修复提交上
作为扩展矩阵重新运行，并使用独立 manifest；论文主结论只使用 `199ead5` 的核心矩阵，
不得把两个提交的学习算法混入同一严格聚合。

每个 benchmark 完成后立即用 best checkpoint 执行固定 100-case final test，并录制
seeds `20000..20005` 的 6 个视频。逐 run 审计 checkpoint/sidecar SHA、配置和 runtime
合同、日志异常、视频可播放性；CA 额外检查 lambda 非负且梯度有限。

## 阶段五：聚合与归档

截至 2026-10-10，本阶段已完成。核心矩阵与扩展矩阵分别建立了严格 manifest 和聚合
CSV；核心矩阵已完成 MAPPO、Attention-MAPPO、辅助 CA-ComMAPPO 的正式顺序扰动，
full/no-prior 的 attention/TTC trace，以及包含三组预注册配对的 `paper-analysis`。
审定表、图、trace 摘要和结论报告均已归档到 `results/`，当前没有运行中的实验。

- 顺序扰动、attention/TTC 导出、核心配对统计和绘图均在匹配冻结提交的严格环境完成。
- 每个矩阵已有显式 seeds 1/2/3 manifest 和严格聚合 CSV。
- 核心四算法与扩展三算法保持独立 manifest；扩展结果只作补充比较。
- `runs/` 保留 Git 忽略的原始大 artifact；`results/` 保存审定 manifest、表、图和小型
  provenance 摘要。
- 实现、实验、聚合、分析和文档阶段均形成了聚焦 Conventional Commit。

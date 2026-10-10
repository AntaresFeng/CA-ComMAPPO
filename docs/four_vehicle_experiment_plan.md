# 四车论文实验执行计划

## 目标与当前定位

本计划接续历史交接文档中未完成的实验阶段，并以当前仓库、canonical YAML 和
`docs/` 根目录文档为准。论文主线固定为四个 controlled vehicles，不开展
4→6/8/10 车辆泛化，也不据此声明规模泛化能力。

现有历史 MAPPO I5 seed 1 与 CA-ComMAPPO seed 1 benchmark/final/video 保留为预实验
参考。后续评估工具和 provenance 合同完成后，所有论文正式结果都必须在同一个最终
源码提交上使用训练 seeds `[1, 2, 3]` 重新生成。

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

已实现 `main.py paper-analysis`：它复用严格 final-test 聚合门禁，默认要求核心三组
配对齐全，输出逐 seed 原始值、sample std、方向明确的配对效应量与精确双侧 sign-flip
检验。三个 seed 的 p 值仅作探索性辅助。主结果、配对消融与顺序扰动图输出 PNG/PDF；
CA trace 工具生成的 attention/TTC heatmap 构成解释性图。正式 artifact 尚待最终冻结
提交上的多 seed 训练完成后生成。

## 阶段三：全算法 smoke 与源码冻结

在最终源码上依次验证 MAPPO、Attention-MAPPO、CA-ComMAPPO full、CA-ComMAPPO
no-prior、IPPO、CommNet 和 IC3Net。所有算法必须通过真实 action/update smoke；通信
算法还要通过 benchmark checkpoint、严格 test 和单集录像。最终执行 Ruff、format、
完整 pytest 和 NaN/Inf 检查，记录冻结 commit 与 runtime SHA。

冻结后不得修改代码、配置或依赖锁，直至全部正式 final test 完成。若发现必须修复的
实现缺陷，应停止实验、修复并产生新冻结提交；此前该轮 runs 全部降级为预实验。

## 阶段四：正式多 seed 实验

正式任务统一设置 `WANDB_MODE=offline`，使用 `--mode benchmark` 和独立具名 tmux。
本机 20 核、31 GiB，长任务一次只运行一个，避免 CPU/内存竞争。

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

截至 2026-10-10，核心矩阵和扩展矩阵的全部正式 benchmark、final test 与视频均已
完成，当前没有运行中的实验。扩展矩阵已有位于 `runs/matrix_state/` 的严格 manifest
与聚合 CSV，但尚未归档到 `results/`；核心矩阵尚未建立正式 manifest。顺序扰动、
attention/TTC trace 和核心 `paper-analysis` 仍待执行。

- 在冻结提交上完成顺序扰动与 attention/TTC 导出。
- 每个算法建立显式 seeds 1/2/3 manifest，通过严格聚合器后生成审定 CSV。
- 核心四算法与后续修复提交上的扩展三算法分别建立 manifest；扩展结果在论文中明确
  标注来源提交，只作补充比较，不进入核心主结论。
- `runs/` 始终保持 Git 忽略；只将人工审定的 manifest、聚合表和小型 provenance
  摘要写入 `results/`。
- 每完成一个实现、实验或文档阶段创建聚焦 Conventional Commit。
- 最终更新论文实验结果文档，记录 YAML、完整命令、Git commit、seed、设备、artifact
  路径、偏离协议之处、结果表和图。

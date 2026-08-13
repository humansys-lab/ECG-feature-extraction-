# ECGAgent 确定性重构方案

分析日期：2026-08-03
分析对象：`ecgagent/agent/diagnostic_pathways.py`（v5）、`ecgagent/agent/diagnostic.py`、
`ecgagent/evidence/model_view.py`、`ecgagent/evidence/diagnostic_contract.py`
证据来源：全 123 个 `DIAGNOSIS_CATALOG` 诊断码的路径静态展开；
`qwen_agent_output/deepseek_v37_diverse50`（50 条 PTB-XL，DeepSeek）的逐节点审计与
`record_results.csv`；`ptbxl_09000_ecgfeat/features/09000_hr_features.json` 上的工具/视图实跑。

> **版本注记**：静态普查基于 `ecgagent.diagnostic-pathways.v5`（2026-08-03 04:56 的代码状态，
> 284 个节点实例）。逐节点运行统计来自 v37 那次批跑，当时是 v4（274 个实例，461 个节点被实际评估）。
> 两者比例一致、结论同向，但**不要把两组绝对数混在同一张表里引用**。
> v5 相对 v4 新增 3 个节点：`qt_threshold`、`alternative_qrs_cause_excluded`、
> `sinus_candidate_stream_reconciled`——其中两个属于本文档所说的"确定性问题交给模型"，
> 说明这个模式仍在继续产生。

---

## 实施进度（2026-08-03 更新）

| 步骤 | 状态 | 落地内容 |
|---|---|---|
| 第 0 步 影子模式 | ✅ 已实施 | 审计每个节点同时记 `model_status` / `program_status`；`ECG_AGENT_DETERMINISTIC_NODES_SHADOW=1` 一键回滚（同时翻转视图路由与裁决归属） |
| 第 1 步 路径语法 | ✅ 已实施 | 新增 `invalidator` 门类型；`_step()` 拒绝未知 gate；LVH 加两个程序化前置条件 |
| 第 2 步 节点确定化 | ✅ 大部已实施 | `ecgagent/agent/deterministic_pathways.py`，程序占比 5.6% → **47.3%** |
| 第 3 步 放开候选上限 | ⬜ 未做 | `MERGED_CANDIDATE_MAX = 5` 未动 |
| 第 4 步 H 类处置 | ⬜ 未做 | 81 个实例仍由模型判定 |

**实跑结果（`deepseek_v38_determinism50`，50 条，DeepSeek，与 v37 同记录同特征，50/50 verified）**

对 PTB-XL 标签的一致性**下降**了：confirmed F1 **0.359 → 0.267**（TP 14→10，FP 18→19）。
但拆开看，这个下降不能直接当成质量下降：

| 变化 | 数量 | 真阳性 | 假阳性 |
|---|---:|---:|---:|
| 丢失的确认 | 22 | 3 | 19 |
| 新增的确认 | 16 | 1 | 15 |

丢失的 22 个里 19 个是假阳性——**这一半是健康的**。问题在新增的 15 个"假阳性"，
而其中 **7 个经测量核对是客观正确、只是 PTB-XL 未标注**：09887 PR=217ms、09913 PR=213ms、
09933 HR=50.3bpm、09166 QTc=321ms、09557 QTc=485ms、09249 胸导全部<1.00mV、09756 肢导全部<0.50mV。
确定化让 agent 开始输出测量派生结论（PR/心率/QTc/低电压），而这些恰是 PTB-XL 系统性漏标的类别。
**记录级标签已经无法为这一层打分**（batch 自己的 RESULTS.md 也写着"未标注异常不自动等于假阳性"）。

机制层面确定化是成功的：程序节点占比 15% → **67.8%**，程序节点 unknown 率 21.5%
对模型节点 52.5%；`invalidator` 在 6 个节点开火，4 条 LVH 假阳性（09025 IVCD、09070 CLBBB、
09129 PACE、09557 PACE）被否决，三条真 LVH 全部保留。

**暴露出的三个真实缺陷，均已修复**（`deepseek_v39_fixes50` 验证中）：

1. **09699 在 WPW 记录上确认 RBBB**（QRS 152ms，参考 SBRAD|WPW）。根因不是缺节点，而是
   `preexcitation_excluded` 只读 `short_pr_interval`——该记录 PR 测不出来所以为 false，
   但 `short_pr_segment=true` 且 9 个导联有 delta 波，节点因此返回 unknown，什么都没排除。
   已修为与 `preexcitation_components` 一致（两个构成任一为真即算短 PR），
   并把该节点以 `gate=invalidator` 加入束支阻滞路径。**必须是 invalidator**：真 RBBB 记录
   09545 在该节点返回 unknown，若用 required 会连真阳性一起卡死。
2. **`alternative_qrs_cause_excluded` 被提升为 `required`**。"是否未显示替代解释"是开放式否定命题，
   单一视图无法证明，实测 0 fail / 4 unknown——只能 stall 不能否决，独立阻断 3 个确认。
   已改为 `invalidator`。
3. **孤立室早结构性不可诊断**。`_pvc_morphology_fact` 要求非主导宽 QRS 组 `member_count >= 2`，
   而 09388 是单个 218ms 的组。10 秒记录里的单个室早因此永远无法确认。已加
   `pvc_single_wide_qrs_ms = 140.0`：孤立但显著增宽的组可单独 pass，120–140ms 的孤立组仍为 unknown。

**未修（刻意）**：`right_precordial_voltage` 在 09275、09620 返回 unknown，丢掉两个 RVH 真阳性。
两条记录参考诊断都含 CRBBB，而右束支阻滞本身就会在 V1 产生高 R'，使右胸电压标准失效——
在 n=2 上放宽这个判据去凑标签，正是本文档反对的做法。正确方向是给 RVH 也加一个
CRBBB 失效前置条件（会变成 reject 而非 confirm），留待判据层决策。

## v39 验证结果（`deepseek_v39_fixes50`，50/50 verified）

| 层级 | v37 基线 | v38 确定化 | v39 修复后 |
|---|---:|---:|---:|
| confirmed | **0.359** | 0.267 | **0.308** |
| confirmed + 鉴别 | 0.421 | 0.392 | 0.404 |
| 全部考虑过（含 rejected） | 0.414 | 0.397 | 0.394 |
| 可测量核对的确认 / 其中错误 | 22 / **0** | 27 / **0** | 27 / **0** |

`invalidator` 否决 8 次（v38 为 6 次），新增的两次是 09699 与 09275 的 `preexcitation_excluded`。
其中 **09275 是误否决**——该记录参考诊断含 LAFB。根因：它的 PR 是实测的 141ms（未缩短），
而我把 `short_pr_segment=true` 当成了短 PR 证据。PR 段等于 PR 间期减 P 波时限，
P 波宽而 PR 正常时段也会短，这不是预激。已收紧为**只在 PR 间期测不出来时**才用段作替代
（判据用 `/rhythm_inputs/record/availability/pr_available`，该指针在 `atrial_signal` 段，
两处 step 参数已改为 `["preexcitation", "atrial_signal"]`）。修正后 09699 仍 `fail`、
09275/09545 均为 `unknown`，不再误伤。

**剩余 5 个相对 v37 丢失的真阳性，逐条归因后没有一个由确定化造成：**

| 记录 | 丢失项 | 原因 |
|---|---|---|
| 09275 / 09620 | RVH | 刻意未修（`right_precordial_voltage` 在 CRBBB 下本就失效） |
| 09114 | 房颤 | 模型改报 `atrial_fibrillation_flutter_indeterminate`，该码无别名也不在房颤类别的 `prediction_codes` 里——**打分策略问题**，非能力问题 |
| 09388 | 窦速 | 模型改提名 `atrial_tachycardia`，被两个模型节点判 unknown |
| 09136 | 右束支 | `required_lead_pattern` 由模型判 unknown |

即：**3 个是模型逐轮提名/判定的随机波动，2 个是刻意保留的判据问题。**
这给出一个重要的方法论边界——50 条记录 + 随机模型的测量精度，与本方案追逐的效应量同一量级。
**任何单轮 F1 差异小于约 0.05 都应视为噪声**，结论必须靠归因而不是靠头条数字。

## 标签已经无法为这一层打分——新增测量核对量尺

`score_measurement_verifiable.py`：对"定义本身就是测量比较"的家族（PR、心率、QTc、低电压、电轴），
直接用记录自己的 `*_features.json` 核对，而不是查标签。结果：

| | 可核对的确认 | 其中错误 | 正确但未标注 |
|---|---:|---:|---:|
| v37 | 22 | **0** | 19 |
| v38 | 27 | **0** | 25 |

**两轮在这些家族上都是 100% 正确**，确定化只是让 agent 多输出了 5 个（22→27）。
标签打分却把 v38 的 27 个里 25 个记成假阳性。所以 v37→v38 的 F1 下降必须这样读：
真正的损失是形态类家族的 4 个真阳性（其中 3 个已由上述修复找回 2 个），
而不是"新增了假阳性"。`measurement-contradicted` 这一列是唯一不能用标签不完整解释的数字，
应作为主指标之一驱动到 0。

**剩余的确定性节点**：31 个实例（10.6%），从 133 个降下来。构成是
`st_territorial_confirmation` 10、`tu_lead_confirmation` 6、`qrs_duration_support` 6（分支阻滞类，
非完全性束支的那部分）、`cross_lead_voltage_criterion` 3（非 `lvh_voltage_criteria` 的 LVH 码）、
`multilead_flutter_support` 2、`av_relation_characterized` 2、`q_wave_morphology` 2。
这些**故意保留**：它们的判据需要相邻导联拓扑或分支阻滞的分型语义，盲写风险高于收益，
应在影子模式积累一轮分歧数据后再迁移。

全量测试 1435 通过；新增 8 条针对 `invalidator` 门、LVH 前置条件与影子模式的测试。

---

## 0. 结论摘要

**问题不是模型不够聪明，是模型被放在了错误的岗位上。**

三个互相独立的测量指向同一件事：

1. **诊断路径的 284 个决策节点里，133 个（46.8%）是纯阈值比较、计数或布尔与运算**，
   现在全部由 LLM 打 `pass/fail/unknown` 标签。程序自己动手的只有 16 个（5.6%）。
2. **在程序与模型都给出答案的 69 个节点上，两者分歧率 28%**；凡能裁定的 12 例，
   程序全对、模型全错。此外有 7 例是模型对**它从未看到的证据**给出了确定标签。
3. **模型提名的候选集 F1 0.459（召回 0.609）跑赢纯规则基线 F1 0.426（召回 0.500）**，
   但最终输出只有 F1 0.351（召回 0.283）——**28 个正确提名里 15 个死在节点判定阶段**
   （8 个 rejected、7 个 unresolved）。

也就是说：**"中枢大脑"那一段（形成假设、决定看什么）已经是全系统最强的一环，
而它的成果被一层它不胜任的自我验证吃掉了。**

重构的直接目标：把节点判定从模型手里收回程序，让 28 个正确提名活着走到输出，
即 shipped F1 从 0.351 提升到提名层的 0.459，一步超过规则基线。

---

## 1. 审计方法与可复现性

四条独立证据链，全部可复现：

| 方法 | 做了什么 | 得到什么 |
|---|---|---|
| 路径静态展开 | 对 123 个诊断码调用 `build_diagnostic_pathway`，枚举全部节点 | 60 个不同节点、284 个（码×节点）实例 |
| 天然对照实验 | 审计同时记录 `requested_status`（模型）与 `effective_status`（程序），确定性门开火时两者并存 | 69 个节点上的模型-程序分歧率 |
| 视图实跑 | 对每个节点的 tool+arguments 在真实记录上执行，再走 `build_model_evidence_view` | 模型实际看到多少证据 |
| 端到端归因 | 用项目自己的 `CATEGORIES` 映射，分别对基线 / 提名集 / 最终输出打分 | 损失发生在哪一段 |

关键陷阱（后续复现必须避开）：`record_results.csv` 的 `candidate_agent_codes`
**不是**提名集——在 `verified=True` 时它等于最终输出。真正的提名集是审计中每条
`decision_audit` 的 `code` 字段，不论 `final_placement`。

---

## 2. 现状测量

### 2.1 节点普查（v5，284 个实例）

| 类 | 含义 | 实例 | 占比 | 现状 |
|---|---|---:|---:|---|
| P | 程序已确定 | 16 | 5.6% | ✅ 已由 `_compact_deterministic_pathway_step` 处理 |
| T | 纯阈值比较 | 43 | 15.1% | ❌ 交给模型 |
| C | 计数 / 游程 / 比例 | 66 | 23.2% | ❌ 交给模型 |
| R | 可靠性布尔聚合 | 24 | 8.5% | ❌ 交给模型 |
| H | 硬标准 + 形态分布 | 93 | 32.7% | ❌ 交给模型；`clinical_rules/` 里几乎都有对应实现 |
| J | 真正开放判断 | 42 | 14.8% | ✅ 适合模型 |

**T + C + R = 133 个实例（46.8%）是应当由程序回答却交给了模型的。**
全表见附录 A。

确定性处理器 `_compact_deterministic_pathway_step`（`ecgagent/agent/diagnostic.py:736`）
只覆盖 7 个 step_id：`rate_threshold`、`axis_threshold`、`preexcitation_components`、
`preexcitation_excluded`、`dominant_qrs_wide`、`wide_qrs_representative`、
`cross_lead_voltage_criterion`（且仅对 `lvh_voltage_criteria` 生效）。函数末尾 `return None`
表示其余一律回落给模型。

三个最刺眼的具体例子：

- **同一逻辑写了两遍。** `dominant_qrs_wide` 用 `mean_qrs_ms >= 120 → pass, < 110 → fail`
  程序化处理 IVCD 的 QRS 增宽；`qrs_duration_support`（13 个诊断码使用）读同一个字段，
  却交给模型。`pattern_representative`（13 个）与 `wide_qrs_representative` 里的
  `member_pct >= 50.0` 同理。这 26 个实例可以零新逻辑迁移。
- **文档里已经是布尔值，还要问模型。** `short_pr_criterion` 问"PR 是否满足短 PR 定义"，
  而 `/rhythm_inputs/preexcitation/short_pr_interval` 本身就是导出好的布尔。
  `interval_reportable`（6 个）、`component_endpoint_support`（6 个）同理。
- **阈值常数只活在 Python 里。** `pr_criterion` 问 PR 阈值，
  `ecgagent/agent/safety_policy.py:26` 有 `first_degree_av_block_pr_lower_exclusive_ms = 200.0`，
  `feature_extraction/ecgfeat/interpret.py:725` 还有按年龄×心率的 DXL 二维表。
  模型看不到这些数字，只能猜定义边界；阈值仅在事后由 `semantic_guard` 否决。

### 2.2 模型 vs 程序：天然对照实验

在确定性门开火的 69 个节点上，模型的 `requested_status` 与程序的 `effective_status`
**分歧率 28%（19/69）**。逐条核对原始数值与 PTB-XL 参考诊断后：

| 节点 | 分歧数 | 裁定 | 典型例 |
|---|---:|---|---|
| `axis_threshold` | 4 | 程序 4/4 对 | **09883 `qrs_axis_deg = +2.64°`，参考 NORM，模型判"满足电轴左偏"**（定义 −90°～−30°）。不是边界舍入，是根本没做比较。另有 −29.71°、−27.93° 被判 pass |
| `cross_lead_voltage_criterion` | 8 | 程序 8/8 对 | 模型算了 Cornell 就收工，漏掉 Peguero 的"或"分支。09537 男性 Cornell 2.349 < 2.8 但 Peguero 2.672 ≥ 2.3；09070 女性 Cornell 4.376 远超 2.0 阈值，模型给 unknown |
| `dominant_qrs_wide` / `wide_qrs_representative` / `preexcitation_excluded` | 7 | 模型不该作答 | 模型给出确定 pass/fail，程序返回 unknown——因为**该指针从未进入模型的授权视图**（09144、09550 各 3 个节点） |

**结论：失败模式不是"LLM 不会算术"，而是四件不同的事：**

1. 阈值常数没有交付给模型，它在猜定义；
2. 多分支判据（Cornell **或** Peguero）只走了一条；
3. **看不到数据时不会弃权**，照样自信作答；
4. 计数类节点的数据本身被截断到无法回答（见 2.3）。

### 2.3 证据可见性

对 56 个可执行节点在真实记录上跑 tool + `build_model_evidence_view`：
**5855 条引用最终只有 1981 条进入模型上下文，66% 被丢弃**，17 个节点损失超过一半。

| 节点 | 引用数 → 可见原子 |
|---|---|
| `repolarization_context` | 612 → 40 |
| `alternative_qrs_cause` | 468 → 41 |
| `pr_criterion` | 298 → 48 |
| `t_wave_distribution` | 132 → 37 |

与计数类节点叠加时是致命的：`ecgagent/evidence/model_view.py:33` 的
`_INDEXED_GROUP_LIMIT = 8` 会把逐事件/逐搏表**等距抽样**到 8 组。实测 09000
这条记录有 28 个 P 事件（3 个 blocked），`get_atrial_event_table(limit=16)` 渲染 96 条引用，
模型最终看到的是**事件 [0, 2, 4, 6, 9, 11, 13, 15] 这 8 个不相邻的采样**，
3 个 blocked 事件里有 1 个根本不在其中。

所以"数一数有几个未下传 P""是否 1:1 关联""是否 ≥2 个相邻导联"这类问题，
在当前视图上**结构性不可能答对**，与模型能力无关。

### 2.4 端到端损失定位（v37，50 条，类别级 micro，`semantic_scope == "direct"`）

| | TP | FP | FN | 精确度 | 召回 | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 规则基线（无 LLM） | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
| **模型提名的全部候选** | **28** | 48 | 18 | 0.368 | **0.609** | **0.459** |
| Agent 最终输出（confirmed） | 13 | 15 | 33 | 0.464 | 0.283 | 0.351 |

**提名阶段跑赢规则基线**，找到了基线漏掉的 LVH、窦速等。
**28 个正确提名最终只有 13 个活着走出来，死掉 15 个：8 个 `rejected`、7 个 `unresolved`**，
涉及 LAFB、PVC、PAC、房颤、LBBB、IVCD、LPFB。杀死它们的机制就是 2.1 的那些算术节点——
某个 `required` 节点返回了 fail 或 unknown。

因果链闭合：
提名 28 个正确假设 → 每个须通过 2–4 个模型算不了的数值节点 →
36% 节点返回 unknown、28% 与程序判定相左 → 15 个正确假设死亡 → 召回 0.609 掉到 0.283。

---

## 3. 四层问题分解

现在这四件事被混在一起讨论，所以每次修改都只动了一层。**它们必须分开立项。**

| 层 | 问题 | 证据 | 本方案是否覆盖 |
|---|---|---|---|
| **L1 提名被截断** | `MERGED_CANDIDATE_MAX = 5`，每条记录丢弃约 1.6 个规则候选 | 提名 F1 0.459 本身就是在这个帽子下测出的，天花板更高 | ✅ 第 3 步 |
| **L2 路径语法缺否决类型** | 只有 `required`/`supporting`，supporting 结构上无法否决 | LVH 的 `qrs_morphology_compatible` 至今仍是 `supporting`，永远拦不住；6 个假阳性中 5 个带 QRS 增宽混杂因素 | ✅ 第 1 步 |
| **L3 节点判定交给模型** | 284 个节点里 133 个是算术 | 28% 分歧率，15 个正确诊断死于此 | ✅ 第 2 步 |
| **L4 特征/判据层缺陷** | infarction 0/16、ischemia 1/18 | 已证实**不是**闸门限制的 | ❌ **独立立项，不在本方案内** |

---

## 4. 目标架构：程序算事实，模型解读

数据流分六段，每段只做一件事：

```
① 事实层（程序）   ecgfeat 测量 + 新增确定性派生量
        ↓
② 判据层（程序）   284 个节点全部程序计算 → pass/fail/unknown + 依据指针 + 适用性前置条件
        ↓
③ 提名层（模型，盲态）  读概览，提出候选。不再受 5 个硬上限
        ↓
④ 否决层（模型，有据）  拿到填好的判据表，只回答"哪一条在本例不成立、为什么"。只能减不能加
        ↓
⑤ 叙述层（模型）   报告、未决问题、人该复核哪一段波形
        ↓
⑥ 裁决层（程序）   置信度 = 节点来源的函数；unknown 进鉴别而非丢弃
```

各层设计要点：

**① 事实层.** 在现有测量之外补齐确定性派生量：P/QRS 计数、连续未下传 P 游程、
房室传导比、PR 均值/标准差/线性斜率、阻滞前后 PR 差值、相邻导联计数、支持导联计数。
这是唯一需要动 `feature_extraction/ecgfeat` 导出层的地方。

顺带修复已知的"算了却丢掉"：`feature_extraction/ecgfeat/rhythm_rules.py:561-562` 的
`two_to_one_conduction_suspected`、`av_block_level_hint`，以及
`_classify_second_degree_type` 产出的 `second_degree_avb_basis`
（含 `preceding_pr_ms`/`following_pr_ms`/`pr_increment_ms`/`pr_spread_ms`，即 Mobitz I/II 的鉴别依据）
都不在 `export.py` 的 `av_block` 字典里，只存在于被 `_FORBIDDEN_KEYS` 剥离的
`metadata/rhythm_analysis` 下，agent 可证明地拿不到。

**② 判据层.** 硬约束：`unknown` 只允许有一个来源——**测量不可用**。
不允许再出现"没算出来"这种 unknown。当前 36% 的 unknown 里绝大部分是后者。
每个节点必须同时产出「结论 + 依据指针 + 该判据在本例是否适用」。

**③ 提名层.** **保留盲态设计**（`_FORBIDDEN_KEYS` 对 `clinical_interpretation` /
`interpretation` / `rhythm_analysis` 的剥离），因为它在工作——提名 F1 0.459 > 基线 0.426。
盲态只在提名阶段有效，提名冻结之后即可放开（"先盲后验"）。

**④ 否决层.** 这是模型真正的增量价值：规则引擎只能排除有人事先枚举过的东西。
LVH 在起搏记录上误报（09129）就是这个洞——`clinical_rules/` 里没有起搏排除节点。
该层只能减不能加，且必须给理由和引用，所以模型误判的代价是损失敏感度，
而且留下可审计记录；不像现在，09883 那种凭空造出的阳性是悄无声息的。

**⑥ 裁决层.** 置信度从 `HIGH if len(support_tools) >= 2 else MEDIUM`
（"有几个工具出过力"，与证据强度无关）改成节点来源的函数：
全部节点程序确定 → HIGH；含模型否决 → MEDIUM；含 unknown → 不确认但**进鉴别诊断**。
现在 7 个正确诊断就是卡在 unresolved 后消失的。

---

## 5. 实施方案

**顺序不可调整**，理由见每步的"为什么在这个位置"。

### 第 0 步：影子模式（非破坏性，必须最先）

让程序对全部 284 个节点都计算答案并写入审计，但**不改变任何行为**，模型标签照旧生效。

- 产出：每次真实运行都免费生成全节点的「程序 vs 模型」对照，即把 §2.2 那个
  只覆盖 69 个节点的天然实验扩展到 284 个。
- **为什么在这个位置**：没有这一步，后面每一步迁移都是盲改。有了它，
  每个节点的迁移都有数据支撑，而且可以按"分歧率高"排序决定迁移优先级。
- 验收：审计中每个 `pathway_steps` 条目同时含 `model_status` 与 `program_status`；
  跑一次 50 条，产出全节点分歧率表。

### 第 1 步：路径语法——引入前置条件/否决节点类型

- 在 `_step()` 的 `gate` 之外增加 `precondition` / `invalidator` 类型，
  使否决节点能真正阻断确认（当前 `supporting` 的状态根本不进 `required_statuses`）。
- 立即修复 LVH：把 `qrs_morphology_compatible` 提升为阻断型，或在电压节点前加
  QRS 宽度/起搏有效性前置条件。
- **为什么必须在第 2 步之前或同时**：电压节点程序算术上 8/8 正确，但 5 条通过的记录里
  只有 2 条是真 LVH。**模型算不出来这件事，正在意外掩盖一个过度敏感的判据。**
  先做确定化，LVH 假阳性会从 6 涨上去。前置条件节点是这个副作用的对冲。
- 验收：v37 同一批记录上 LVH 假阳性 6 → 1，真阳性 09089 保留。

### 第 2 步：节点确定化（分三批）

| 批 | 范围 | 实例 | 依赖 | 风险 |
|---|---|---:|---|---|
| 2a | 重复实现：`qrs_duration_support`、`pattern_representative` | 26 | 无，复用 `dominant_qrs_wide` 分支现有逻辑 | 极低 |
| 2b | 纯阈值 + 可靠性布尔（T + R） | 67 | 常数已在 `safety_policy.py` / `clinical_rules/config.py` | 低 |
| 2c | 计数 / 游程 / 比例（C） | 66 | 依赖第 ① 层新增导出字段 | 中 |

**附带收益**：节点归程序后直接读 `EvidenceStore`，不经过模型上下文预算，
§2.3 那 66% 的丢弃率对它们不再有意义。因此视图预算的修复范围大幅缩小，
只剩否决层和叙述层还需要看证据。

- 验收（每批）：跑同一 50 条，提名召回不变（0.609），shipped 召回单调上升，
  影子模式分歧率在已迁移节点上归零。

### 第 3 步：放开 `MERGED_CANDIDATE_MAX`

- **为什么在这个位置**：验证层可靠之后才敢放宽提名，否则更多候选只产生更多假阳性。
- 验收：提名召回 > 0.609，shipped 精确度不低于第 2 步结束时的水平。

### 第 4 步：H 类 93 个节点的处置（必须二选一）

模型看不到波形——14 个工具全是数值表，没有任何图像/绘图工具。
所以当前让它做"跨导联形态分布判断"，本质上仍是对数值表做算术，只是换了个名字。
这解释了 H 类 51% 的 unknown 率。两条路：

- **(a) 给信号**：接多模态，让模型真的看图（项目已有 `ecg_plots` 与 medgemma 底子）。
- **(b) 归程序**：接 `clinical_rules/` 已有的 `_rbbb`/`_lbbb`/ischemia 等实现，
  在提名冻结后放入（"先盲后验"）。

**建议先走 (b)**：复用现成代码，且不破坏盲态设计的价值。(a) 是新的研究问题，另起一轨。

---

## 6. 度量纪律

盯三个数，当前基线：

| 指标 | 当前 | 目标 |
|---|---:|---|
| 提名召回 | 0.609 | 不下降（第 3 步后上升） |
| shipped 召回 | 0.283 | → 0.609 |
| **两者差距** | **0.326** | **→ 0** |

北极星是**差距**，不是绝对 F1。差距归零意味着 shipped F1 0.351 → 0.459，一步超过规则基线 0.426。

**预期的正常现象：shipped 精确度会先掉**（判据的过度敏感被暴露出来），
**不要当成回归**。这正是把"判据错"和"计算错"解耦的目的——现在这两件事纠缠在一起，
连归因都做不了。

---

## 7. 已知边界与陷阱

1. **L4 架构救不了。** infarction 0/16、ischemia 1/18 是最大的两个窟窿，
   已证实不是闸门限制的。**不要和架构改造放在同一个迭代里做**，否则又会变成
   两件事纠缠、无法归因——这个项目已经踩过这个坑。应独立立项：先查那 16 条的特征层，
   判断是 ecgfeat 测不出来还是判据没写对。
2. **确定性 ≠ 正确。** 电压节点程序算术上 8/8 正确，但通过的 5 条里 3 条不是真 LVH。
   把节点收回程序会让判据本身的缺陷暴露出来——这是目的，不是副作用。
3. **提名精确度只有 0.368。** 48 个假阳性比基线的 39 还多。模型是个**宽提名器**：
   广撒网、召回高、精度低。宽提名 + 可靠验证 = 好系统；宽提名 + 不可靠验证 = 现在这样。
   确定化验证层是必要条件，不是充分条件。
4. **不要用 `candidate_agent_codes` 当提名集**（见 §1）。
5. **模式仍在扩散。** 本次审计期间（2026-08-03 04:56）新增的 3 个节点里，
   `qt_threshold`（纯阈值，`safety_policy` 已有常数）与
   `sinus_candidate_stream_reconciled`（读的全是布尔）两个都属于"确定性问题交给模型"。
   建议在评审清单里加一条：**新增 required 节点时必须说明它为什么不能由程序回答。**

---

## 附录 A：284 个节点实例的完整分类

类别含义：`P` 程序已确定 · `T` 纯阈值 · `C` 计数/游程/比例 · `R` 可靠性布尔聚合 ·
`H` 硬标准+形态分布 · `J` 真正开放判断。
「v37 实测 P/F/U」为 pass/fail/unknown 计数，来自 v4 那次批跑，v5 新增节点显示"未出现"。

| 类 | 节点 id | 实例 | gate | 工具 | v37 实测 P/F/U | 判定内容 · 现成实现位置 |
|---|---|---:|---|---|---|---|
| P | `rate_threshold` | 5 | required | `get_global_table` | 15/0/0 | 心率 vs 60/100 |
| P | `axis_threshold` | 3 | required | `get_global_table` | 10/13/0 | 电轴区间 |
| P | `dominant_qrs_wide` | 2 | required | `get_morphology_groups` | 1/2/2 | mean_qrs_ms vs 120/110 |
| P | `preexcitation_excluded` | 2 | required | `get_rhythm_profile` | 0/0/5 | 同上取反 |
| P | `wide_qrs_representative` | 2 | required | `get_morphology_groups` | 1/0/4 | QRS>=120 AND member_pct>=50 |
| P | `preexcitation_components` | 1 | required | `get_rhythm_profile` | 1/0/0 | 短PR AND delta导联数>=2 |
| P/T | `cross_lead_voltage_criterion` | 4 | required | `get_lead_table` | 15/3/2 | Cornell/Peguero；仅 lvh_voltage_criteria 接通程序 |
| T | `qrs_duration_support` | 13 | required | `get_morphology_groups` | 10/5/1 | 读同一个 mean_qrs_ms，与 dominant_qrs_wide 完全同逻辑 |
| T | `interval_reportable` | 6 | required | `get_interval_waveform_context` | 9/0/2 | qt_reliability / reliable_for_qt 已是布尔 |
| T | `qt_threshold` | 6 | required | `get_interval_waveform_context` | 未出现 | QTc 阈值；safety_policy 已有常数 ← 新增，未接程序 |
| T | `irregular_ventricular_response` | 4 | required | `get_rhythm_profile` | 4/18/0 | rr_cv / rr_rmssd / rr_entropy vs 阈值 |
| T | `territorial_qrs_voltage` | 4 | required | `get_lead_table` | 0/2/0 | 肢导<0.50mV / 胸导<1.00mV，常数在 clinical_rules/config.py |
| T | `pr_criterion` | 3 | required | `get_interval_waveform_context` | 2/4/4 | PR vs 200ms；safety_policy 与 DXL 年龄×心率表均已存在 |
| T | `right_precordial_voltage` | 3 | required | `get_lead_table` | 2/2/2 | V1 R/S 比值阈值 |
| T | `short_pr_criterion` | 1 | required | `get_interval_waveform_context` | 未出现 | short_pr_interval 在文档里已经是布尔 |
| C | `pattern_representative` | 13 | required | `get_morphology_groups` | 12/1/3 | member_pct>=50，wide_qrs_representative 分支里已实现 |
| C | `st_territorial_confirmation` | 10 | required | `get_lead_table` | 3/3/2 | >=2 相邻导联，ischemia.py 已实现 |
| C | `tu_lead_confirmation` | 6 | required | `get_lead_table` | 0/1/2 | 一致导联计数 |
| C | `capture_relation_support` | 5 | required | `get_pacing_profile` | 未出现 | 夺获搏动比例 |
| C | `pacing_marker_support` | 5 | required | `get_pacing_profile` | 未出现 | 起搏标志计数/置信度 |
| C | `multilead_atrial_support` | 4 | supporting | `get_rhythm_profile` | 5/5/12 | 支持导联计数 |
| C | `repeated_blocked_atrial_events` | 4 | required | `get_atrial_event_table` | 7/2/0 | 统计 association_type=='blocked' 个数 |
| C | `sequential_av_pattern` | 4 | required | `get_rhythm_profile` | 2/2/5 | 传导比与连续游程 |
| C | `one_to_one_av` | 3 | required | `get_atrial_event_table` | 6/4/0 | 每个 QRS 对应的下传 P 计数 |
| C | `premature_timing` | 3 | required | `get_beat_table` | 4/1/4 | rr_prev vs 基线比例 |
| C | `representative_event` | 3 | required | `get_morphology_groups` | 1/1/7 | member_count / longest_run 阈值 |
| C | `av_relation_characterized` | 2 | required | `get_atrial_event_table` | 0/1/2 | 传导比 |
| C | `multilead_flutter_support` | 2 | required | `get_atrial_event_table` | 0/0/3 | 支持导联计数 |
| C | `q_wave_morphology` | 2 | required | `get_lead_table` | 0/1/3 | q时限>=阈值 AND q/r>=阈值，ischemia.py 已实现 |
| R | `component_endpoint_support` | 6 | required | `get_interval_waveform_context` | 4/1/6 | t_fusion_reliable / qt_reliability 布尔 |
| R | `p_measurement_reliability` | 5 | required | `get_p_assessment_table` | 1/0/1 | accepted 计数 vs 最小值 |
| R | `organized_p_absent` | 4 | required | `get_p_assessment_table` | 4/15/3 | accepted 布尔 |
| R | `sinus_p_support` | 4 | required | `get_p_assessment_table` | 4/1/1 | accepted / ta_ambiguous 布尔 |
| R | `organized_atrial_activity` | 2 | required | `get_rhythm_profile` | 4/2/0 | availability 布尔 |
| R | `sinus_candidate_stream_reconciled` | 2 | required | `get_rhythm_profile` | 未出现 | localized_atrial_event_excess 等布尔 ← 新增，未接程序 |
| R | `pr_component` | 1 | supporting | `get_interval_waveform_context` | 0/0/1 | reliable_for_pr 布尔 |
| H | `required_lead_pattern` | 13 | required | `get_morphology_map` | 8/3/5 | conduction.py 的 _rbbb/_lbbb |
| H | `st_change_distribution` | 10 | required | `get_morphology_map` | 3/3/2 | ischemia.py ST 阈值+相邻性 |
| H | `repolarization_context` | 8 | required | `get_native_beat_profile` | 7/3/6 | repolarization.py |
| H | `t_wave_distribution` | 8 | required | `get_morphology_map` | 8/2/6 | repolarization.py / t_morphology.py |
| H | `atrial_mechanism` | 6 | required | `get_rhythm_profile` | 1/2/5 | atrial_rhythm.py |
| H | `p_morphology_support` | 6 | required | `get_p_assessment_table` | 1/2/5 | atrial_rhythm.py |
| H | `precordial_progression` | 6 | required | `get_lead_table` | 4/0/3 | r_progression.py 移行区规则 |
| H | `tu_morphology_distribution` | 6 | required | `get_morphology_map` | 0/1/2 | u_wave.py / t_morphology.py |
| H | `p_wave_distribution` | 5 | required | `get_lead_table` | 1/0/1 | atrial_abnormality 规则 |
| H | `qrs_morphology_compatible` | 4 | supporting | `get_morphology_map` | 6/0/14 | hypertrophy.py（**LVH 的否决节点，结构上无法阻断**） |
| H | `ectopic_morphology` | 3 | required | `get_morphology_groups` | 4/0/5 | ectopy.py |
| H | `rvh_qrs_distribution` | 3 | required | `get_morphology_map` | 2/0/4 | hypertrophy.py |
| H | `atrial_relation` | 2 | required | `get_rhythm_profile` | 0/0/1 | av_block.py |
| H | `organized_flutter_activity` | 2 | required | `get_rhythm_profile` | 0/1/2 | rhythm.py 房扑规则 |
| H | `q_wave_distribution` | 2 | required | `get_native_beat_profile` | 0/0/4 | ischemia.py |
| H | `sinus_mechanism_support` | 2 | required | `get_global_table` | 6/3/4 | basic_rhythm.py |
| H | `specific_bbb_excluded` | 2 | required | `get_morphology_map` | 1/0/4 | 同上取反 |
| H | `ventricular_morphology` | 2 | required | `get_morphology_groups` | 0/0/1 | wide_tachycardia.py |
| H | `wide_complex_sequence` | 2 | required | `get_morphology_groups` | 0/0/1 | wide_tachycardia.py |
| H | `qrs_onset_support` | 1 | supporting | `get_morphology_map` | 1/0/0 | preexcitation.py |
| J | `direct_measurement_support` | 13 | required | `get_native_beat_profile` | 1/0/1 | 开放式判断，无对应规则实现 |
| J | `discriminative_countercheck` | 13 | required | `get_native_beat_profile` | 0/0/2 | 开放式判断，无对应规则实现 |
| J | `secondary_qrs_context` | 10 | supporting | `get_native_beat_profile` | 3/0/5 | 开放式判断，无对应规则实现 |
| J | `alternative_qrs_cause_excluded` | 6 | required | `get_native_beat_profile` | 未出现 | 开放式排除 ← 新增，gate=required |

---

## 附录 B：复现要点

- **节点普查**：对 `DIAGNOSIS_CATALOG` 的每个 code 调用 `build_diagnostic_pathway(code)`，
  统计 `steps[].id`。
- **模型 vs 程序**：遍历 `diagnoses/*.json`，找出所有含 `pathway_steps` 的对象，
  筛 `resolution_owner == "deterministic_measurement_gate"`，比较 `requested_status`
  与 `effective_status`。
- **证据可见性**：对每个节点用**新建的** `build_default_registry(store)`（预算按工具计，
  复用同一个 registry 会从第 N 个节点起全部返回 `ok=False`），
  再把结果喂给 `build_model_evidence_view`，比较 `len(citations)` 与 `atom_count`。
- **端到端打分**：用 `evaluate_target_ecgfeat_diagnosis.CATEGORIES` 做映射
  （参考侧用 `spec.ptbxl_refs`，预测侧用 `spec.prediction_codes`），
  只统计 `semantic_scope == "direct"` 的类别。提名集取审计中每条 `decision_audit`
  的 `code`，不论 `final_placement`。

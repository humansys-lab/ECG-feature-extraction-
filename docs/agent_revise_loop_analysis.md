# Agent 修订循环（revise）问题分析与改进清单

分析日期：2026-08-01
分析对象：`ecgagent/agent/loop.py` 的确定性校验—修订循环
证据来源：`ptbxl_09000_medgemma27b_v16/09000_hr`（协议 v16，MedGemma-27B）、
`ptbxl_09000_qwen36_27b/09001_hr`（协议 v14，Qwen3.6-27B），以及全部
1138 份已存 verdict 的汇总统计。

---

## 0. 结论摘要

修订循环**要么在 3 轮内收敛，要么基本不会收敛**。继续往 4 轮及以上跑，
收益极低，却消耗了整条记录 40% 以上的墙钟时间。

| 进入修订的轮数 | 记录数 | 最终 verified | 说明 |
|---|---:|---:|---|
| 1–3 轮 | 372 | **363（97.6%）** | 修订机制在这里是有效的，不应削弱 |
| ≥4 轮（触顶） | 239 | **47（19.7%）** | 触顶后基本救不回来 |

触顶的 239 条里：

| 结局 | 条数 | 占比 |
|---|---:|---:|
| 产出报告但未通过证据校验（`ok=True`，无 error） | 136 | 56.9% |
| 用尽预算，契约仍不通过 | 99 | 41.4% |
| 修订把 JSON 改坏了 | 4 | 1.7% |

> 汇总口径说明：统计排除了**未进入修订就失败**的记录（synthesis 输出不可解析、
> API 402/400 等），这类共 527 条，与修订循环无关，早期版本的统计把它们混入
> `revisions=0` 会严重压低基线。

**核心判断：问题不在"修订轮数不够"，而在于第 1 轮反馈的质量。**
把预算从 4 提到 6 不会有帮助；把第 1 轮反馈做得可执行才会。

---

## 1. 两个案例的逐轮解剖

### 1.1 MedGemma / 09000_hr —— 三轮输出逐字节相同

阶段时间线（修订部分）：

| 轮 | 模型回合 | 成功工具调用 | 耗时 | 停止原因 |
|---|---:|---:|---:|---|
| revise 1 | 3 | 1 | 280.32 s | end_turn |
| revise 2 | 1 | 0 | 141.84 s | end_turn |
| revise 3 | 1 | 0 | 142.46 s | end_turn |

修订合计 564.6 s，占整条记录 1404.88 s 的 **40.2%**。

对三轮输出做 md5 比对：

```
synthesize   md5=d1d5271450c5
revise 1     md5=e9f7725d0113   ← 有变化
revise 2     md5=e9f7725d0113   ← 与上一轮完全相同
revise 3     md5=e9f7725d0113   ← 与上一轮完全相同
```

最终以 `contract revision stalled: the model returned an unchanged invalid
verdict twice` 失败。

**revise 1 的变化本身也是有问题的。** 合成阶段输出了
`nonspecific_ivcd`，理由写的是「QRS时限80 ms在正常范围内（<120 ms）」——
自证其否。IVCD 守卫正确拦截。模型的修法是删掉 `nonspecific_ivcd`，换上
`st_depression`，理由是「ST段在大部分导联呈水平，**符合正常变异**」——
同一类自我否定错误，只是换了个诊断码；而 `st_depression` 没有守卫，于是通过了。

模型没有理解守卫背后的**错误类别**，只做了局部替换。

复现该记录的契约问题（5 条）：

```
1. `summary` 含未落地数量 (191 ms, 80 ms, 497 ms)
2. ranked_complete_interpretations[0].complete_diagnosis 含未落地数量 (同上)
3. ranked_complete_interpretations[1].complete_diagnosis 含未落地数量 (同上)
4. diagnoses[2].reasoning 含未落地数量 (100 bpm)
5. interval_measurement_contexts[0].residual_evidence 未引用任何 T/U 分量波观察
```

第 4 条**无法满足**：原文是「心率113 bpm高于100 bpm，符合心动过速的定义」，
其中 100 bpm 是**诊断定义阈值**，不是患者测量。要通过校验只能删掉定义、
或伪造一条 `value=100` 的证据项（后者被其他规则禁止）。模型两条路都没选，
于是原样重发。

同时第 1–3 条也存在结构性困难：PR 191 ms / QRS 80 ms 都是**正常值**，
而唯一的落地机制是 `diagnoses[].evidence`，只为阳性发现而存在。
正常测量在当前 schema 里无处安放。

### 1.2 Qwen / 09001_hr —— 四轮修订只改了一个字符

| 轮 | 模型回合 | 成功工具调用 | 拒绝调用 | 耗时 | 停止原因 |
|---|---:|---:|---:|---:|---|
| revise 1 | 7 | 3 | **11** | 34.09 s | **turn_limit** |
| revise 2 | 4 | 3 | 0 | 103.48 s | end_turn |
| revise 3 | 7 | 3 | 7 | 123.37 s | end_turn |
| revise 4 | 7 | 3 | 5 | 118.54 s | end_turn |

修订合计 379.5 s，占整条 863.43 s 的 **44.0%**。

```
synthesize   len=4794  md5=19a161c255
revise 1     len=3     md5=27c4e28d48   ← 输出只有 3 个字符，turn_limit
revise 2     len=4795  md5=05ca36a583
revise 3     len=4795  md5=05ca36a583   ← 与上一轮完全相同
revise 4     len=4795  md5=05ca36a583   ← 与上一轮完全相同
```

synthesize 与最终版的**唯一**差异：

```diff
- "citations": ["E2", "E3", "E5"]
+ "citations": ["E2", "E3", "E15"]
```

一个字符。12 条校验问题一条未处理，包括：缺失顶层键
`ranked_complete_interpretations`、`normal_ecg` 与 `left_atrial_abnormality`
互斥、LAA 证据门槛不足、7 处未落地数量、1 条算术冲突。

同一次运行的另一条记录 `09000_hr` 更糟：
`contract revisions did not return parseable JSON` —— 修订把输出彻底改坏了。

---

## 2. 根因

### R1. 反馈是一次性平铺的问题清单，没有优先级也没有可执行修法

[`loop.py:589-602`](../ecgagent/agent/loop.py#L589-L602)，非 adjudicate 模式下
`_contract_feedback` 只做两件事：把最多 16 条问题拼成 bullet 列表，
再加一句通用提示。

```python
lines = [
    "The verdict violates the deterministic output contract. Fix every item:",
    *[f"- {problem}" for problem in structural[:16]],
]
```

模型收到的是 12 条并列要求，其中混杂着：

- **机械类**（删掉叙述里的一个数字）
- **结构类**（补一个缺失的顶层键）
- **实质类**（移除一个够不上门槛的诊断）
- **不可满足类**（把定义阈值当患者测量去落地）

没有任何信号区分它们。"Fix every item" 面对一条做不到的要求时，
最省事的策略就是什么都不改 —— 两个案例的行为完全一致。

### R2. 不可满足的问题会毒化整批修订

只要清单里有 1 条无解，模型就倾向于放弃全部 12 条，而不是修掉能修的 11 条。
当前循环没有"部分修复"的概念：`structural` 非空即失败。

两个已确认的不可满足/误报规则：

- [`diagnostic_prompts.py:466-505`](../ecgagent/agent/diagnostic_prompts.py#L466-L505)
  `_reject_quantities` 对**定义阈值**误报。它只豁免区间写法
  （`60-100 bpm` 通过），不豁免单个阈值（`高于100 bpm` 被拦）。
- [`semantic_guard.py:21-31`](../ecgagent/agent/semantic_guard.py#L21-L31)
  `_COMPARISON_RE` 的 negation 盲区，把正确的「未达到阈值」判成算术矛盾。
  详见 §5 的独立条目。

### R3. 修订预算是全局共享的，一次解析失败就吃掉 1/4

[`loop.py:1200`](../ecgagent/agent/loop.py#L1200)，`_request_revision` 内部
`while result.revisions < self.max_revisions` 与外层结构/证据修订**共用同一个
计数器**，默认 `max_revisions = 4`（[`loop.py:353`](../ecgagent/agent/loop.py#L353)）。

Qwen 案例里 revise 1 返回 3 个字符的非 JSON，直接消耗掉 1 次预算，
留给实质修复的只剩 3 次。解析失败这种**纯格式事故**不应与语义修订抢配额。

### R4. 修订阶段允许工具调用，模型会把回合花在工具上而不是改文本

Qwen 的 revise 1：7 个回合、3 次成功调用、**11 次被拒调用**，撞上
`turn_limit` 后只吐出 3 个字符。模型在修订阶段陷入了工具调用的重试循环，
根本没走到编辑 JSON 那一步。

多数契约问题（删数字、补键、去诊断）**不需要任何新证据**，纯粹是文本编辑。
当前设计却把完整工具集连同修订反馈一起递给模型。

### R5. 停机判据滞后，触顶前会白烧两轮

[`loop.py:475-492`](../ecgagent/agent/loop.py#L475-L492) 的 stall 检测要求
连续两轮**签名完全相同**才停。也就是说第一次原样重发不会停，
要等第二次才停 —— MedGemma 案例白烧了 142 s。

而 [`loop.py:452-458`](../ecgagent/agent/loop.py#L452-L458) 的 `max_revisions`
判定在请求下一轮**之前**，于是两条停机路径会互相抢先：Qwen 案例先撞
`max_revisions` 报 `failed deterministic contract validation`，
MedGemma 案例先撞 stall 报 `revision stalled`。同一种"模型改不动了"的现象，
产生两种不同的错误信息，给后续统计和排障增加噪音。

### R6. 全有全无，实质正确的报告被整条丢弃

Qwen/09001 的报告在临床上其实相当好：正确判定窦性心律、正确识别并否决了
房颤检测器假阳性、正确标注 T 轴可靠性存疑。它唯一的实质错误是多报了一个
左房异常。但因为契约未通过，整条记录判 `failed`，报告顶部标注
「证据与语义校验：未执行或状态未知」。

触顶记录里 136 条（56.9%）实际上产出了报告，只是没通过校验 —— 当前
没有机制把"实质可用但未完全合规"与"彻底失败"区分开。

### R7. 被拒工具调用完全不可观测

[`trace.py:77`](../ecgagent/trace.py#L77) 只记录 `rejected_calls` 的**计数**，
不记录工具名、参数和拒绝理由。Qwen/09001 全程 27 次被拒调用，
排障时完全是黑盒；这也是它烧掉 1,325,708 prompt token
（MedGemma 同类任务 566,952）的直接原因。

---

## 3. 改进建议（按性价比排序）

### P1 — 反馈分层下发，机械类问题直接给替换文本

改 `_contract_feedback`，按可修复难度分组，并对机械类问题给出**确切的修改动作**：

```
必须修复（结构）：
- 缺失顶层键 `ranked_complete_interpretations`，补一个数组
- `normal_ecg` 与 `left_atrial_abnormality` 互斥，二选一

必须修复（证据门槛）：
- diagnoses[1] `left_atrial_abnormality` 证据不足，移入 differential_diagnoses

机械修改（无需调用工具，直接改文本）：
- `summary` 删除 "92ms"、"434ms" 两处数字，或把它们加进已引用的证据项
- diagnoses[1].evidence[0].claim 拆成两条，每条只写一个数量
```

配套让校验器为每条问题带上 `severity` 与 `mechanical: bool` 字段，
而不是现在的纯字符串。这是本清单里收益最大的一项 ——
1–3 轮的收敛率已有 97.6%，把更多记录推进这个区间即可。

### P2 — 修订阶段默认关闭工具，按需再开

绝大多数契约问题是纯文本编辑。建议：

- 第 1 轮修订**不给工具**，只给反馈和当前 verdict，要求返回完整 JSON
- 只有当反馈里含"需要重新读取测量"类问题（例如
  `[diagnosis_evidence_gate]`、值不匹配）时，才在第 2 轮开放工具

直接消除 R4，顺带大幅压低 token 消耗。

### P3 — 拆分修订预算

把 `max_revisions` 拆成两个独立配额：

- `max_format_retries`（解析失败重试，建议 2）
- `max_semantic_revisions`（实质修订，建议 3）

解析事故不再吃掉语义修订的额度（R3）。同时把语义修订上限从 4 降到 3 ——
数据显示第 4 轮几乎没有收益，砍掉可以为每条触顶记录省下约 120 s。

### P4 — 统一并提前停机判据

- stall 检测改为**一轮**未变化即停（当前要两轮），并把 `max_revisions`
  触顶与 stall 合并成同一个错误码，附带"最后一轮未修复的问题清单"
- 停机时把未修复问题写进 `result.verification`，让轨迹里不再出现
  「没有确定性验证报告」这种断档 —— 两个案例的轨迹目前都查不到校验器到底说了什么

### P5 — 引入"部分通过"状态

在 `ok` / `verified` 之外增加 `contract_incomplete`：契约未完全通过但
verdict 结构完整、无实质性矛盾时，保留报告并在报告头部列出**具体**未通过项，
而不是笼统的「未执行或状态未知」。对应 R6 里那 136 条。

### P6 — 轨迹记录被拒调用明细

在 `trace.py` 里为每次被拒调用记录工具名、参数摘要、拒绝原因。
几行代码，但目前 27 次失败调用完全无法排查（R7）。

### P7 — 修掉两条会误伤正确推理的校验规则

这两条会制造无解反馈，直接触发 R2：

1. `_reject_quantities` 对定义阈值放行 —— 加一个指南常量白名单
   （60/100/120/200 bpm、120 ms、200 ms、440/460/500 ms 等），
   或把该检查限定在 `summary` / `complete_diagnosis` 这类叙述字段，
   不再作用于 `reasoning`
2. `_arithmetic_problems` 的 negation 盲区 —— 见 §5

---

## 4. 验证方式

改动后应能观察到：

- 触顶（≥4 轮）记录占比从 21.0%（239/1138）下降
- 进入修订的记录整体 verified 率从 67.1% 上升
- 连续两轮输出 md5 相同的情况消失（可在 `trace` 里加一个断言）
- 修订阶段的 `rejected_calls` 总数显著下降（P2 直接作用）

回归样本固定用这两条：

| 记录 | 期望行为 |
|---|---|
| `09000_hr`（MedGemma v16） | 不再因 `100 bpm` 定义阈值卡死；自我否定型诊断在第 1 轮就被完整移除，而不是换一个码 |
| `09001_hr`（Qwen v14） | 4 轮至少修掉机械类的 7 条未落地数量与缺失顶层键；`0.0039 mV*s 未达到 >0.04 mV*s` 不再被判算术矛盾 |

---

## 5. 附：算术守卫的 negation 盲区

独立于修订循环，但因为它会制造无解反馈而直接加剧 R2，一并记录。

[`semantic_guard.py`](../ecgagent/agent/semantic_guard.py) 的
`_arithmetic_problems()` 把正确的否定句判成算术矛盾。最小复现：

```
FLAG  V1导联P波终末力约为0.0039 mV*s，未达到左房异常的经典阈值（>0.04 mV*s）
        → "0.0039 > 0.04 is false"          【句子本身完全正确】
FLAG  QRS时限92 ms，未超过120 ms
        → "92 超过 120 is false"             【句子本身完全正确】
ok    PTF为0.0039 mV*s，低于0.04 mV*s的阈值
ok    心率62 bpm，未达到100 bpm的心动过速标准
```

两个独立缺陷：

1. **negation 盲区。** `_COMPARISON_RE` 的 `context` 组
   `[^\d]{0,48}?` 会把两个数字之间的否定词整个吞掉，而
   `_arithmetic_problems` 从不调用 `_NEGATION_RE`（该正则只被
   `_positive_narrative_match` 使用）。`未超过` 不在算符表里，
   正则于是匹配裸的 `超过` 并丢掉 `未`；而 `不超过` 整词在 `_LESS_EQUAL` 中
   所以能通过 —— 纯粹是词表不一致。`_NEGATION_RE` 本身也缺
   未达到 / 未超过 / 低于 / 不足 / 远低于。
2. **单位前缀匹配。** `_UNIT` 没有右边界，`mV*s` 会当作 `mV` 匹配，
   于是 mV 与 mV·s 被当成同量纲比较。

修法：报错前对整个匹配区间跑一次 `_NEGATION_RE`；补全否定词表；
给 `_UNIT` 加右边界，使 `mV*s` / `mV·ms` 不再塌缩成 `mV`。

**不要直接删掉这个检查。** 扫描其中含完整 verdict 对象的 632 份，11 份触发，
其中 **10 份是真实的模型错误**（如「145 ms，略高于正常上限（200 ms）」、
「63 bpm，符合心动过速的定义（>100 bpm）」），只有 Qwen 这条是误报。
但会触发误报的句式恰恰是**排除诊断时的标准写法**，
所以受影响最大的是正常心电图。

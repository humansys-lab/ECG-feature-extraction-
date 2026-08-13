<!-- i18n-nav -->
[中文](agent_revise_loop_analysis.md) | [English](agent_revise_loop_analysis.en.md) | [日本語](agent_revise_loop_analysis.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="agent-修订循环revise问题分析与改进清单"></a>
# Agent revision cycle (revise) problem analysis and improvement list

Analysis date: 2026-08-01
Analysis object: Deterministic verification-revision cycle of `ecgagent/agent/loop.py`
Source of evidence: `ptbxl_09000_medgemma27b_v16/09000_hr` (protocol v16, MedGemma-27B),
`ptbxl_09000_qwen36_27b/09001_hr` (protocol v14, Qwen3.6-27B), and all
Summary statistics for 1138 existing verdicts.

---

<a id="0-结论摘要"></a>
## 0. Conclusion Summary

The revision cycle either converges within 3 rounds or barely converges at all. Continue to run to 4 rounds and above,
The gain is very low, but it consumes more than 40% of the wall clock time of the entire record.

| Number of rounds entering revision | Number of records | Final verified | Description |
|---|---:|---:|---|
| Rounds 1–3 | 372 | **363 (97.6%)** | The revision mechanic is effective here and should not be weakened |
| ≥4 rounds (topped) | 239 | **47 (19.7%)** | It is basically impossible to recover after hitting the top |

Among the 239 items that hit the top:

| Ending | Number of lines | Proportion |
|---|---:|---:|
| Output report but failed evidence verification (`ok=True`, no error) | 136 | 56.9% |
| Exhausted budget, contract still not passed | 99 | 41.4% |
| The revision changed JSON | 4 | 1.7% |

> Description of summary caliber: Statistics exclude records that **failed before entering revision** (synthesis output cannot be parsed,
> API 402/400, etc.), there are 527 entries in this category, which have nothing to do with the revision cycle. The statistics of earlier versions mix them into
> `revisions=0` will seriously suppress the baseline.

**Core judgment: The problem is not "not enough revision rounds", but the quality of the first round of feedback. **
Raising the budget from 4 to 6 won't help; making Round 1 feedback actionable will.

---

<a id="1-两个案例的逐轮解剖"></a>
## 1. Round-by-round anatomy of two cases

<a id="11-medgemma-09000hr-三轮输出逐字节相同"></a>
### 1.1 MedGemma / 09000_hr ——Three rounds of output are the same byte by byte

Phase timeline (revised part):

| Round | Model Round | Successful Tool Call | Time Elapsed | Stop Reason |
|---|---:|---:|---:|---|
| revise 1 | 3 | 1 | 280.32 s | end_turn |
| revise 2 | 1 | 0 | 141.84 s | end_turn |
| revise 3 | 1 | 0 | 142.46 s | end_turn |

Revisions totaled 564.6 s, accounting for **40.2%** of the entire record's 1404.88 s.

Do md5 comparison of the three rounds of output:

```
synthesize   md5=d1d5271450c5
revise 1     md5=e9f7725d0113   ← 有变化
revise 2     md5=e9f7725d0113   ← 与上一轮完全相同
revise 3     md5=e9f7725d0113   ← 与上一轮完全相同
```

Finally ended up with `contract revision stalled: the model returned an unchanged invalid
verdict twice` failed.

The change in **revise 1 is also problematic in itself. ** The synthesis stage outputs
`nonspecific_ivcd`, the reason is "QRS time limit 80 ms is within the normal range (<120 ms)"——
Prove it yourself. IVCD guard intercepts correctly. The method to modify the model is to delete `nonspecific_ivcd` and replace it with
`st_depression`, the reason is that "the ST segment is horizontal in most leads, **consistent with normal variation**"——
The same type of self-denial error, but with a different diagnostic code; `st_depression` has no guard, so it passes.

The model does not understand the **error class** behind the guard and only makes partial replacements.

Contract issues to reproduce this record (5 items):

```
1. `summary` 含未落地数量 (191 ms, 80 ms, 497 ms)
2. ranked_complete_interpretations[0].complete_diagnosis 含未落地数量 (同上)
3. ranked_complete_interpretations[1].complete_diagnosis 含未落地数量 (同上)
4. diagnoses[2].reasoning 含未落地数量 (100 bpm)
5. interval_measurement_contexts[0].residual_evidence 未引用任何 T/U 分量波观察
```

Article 4 **Unsatisfied**: The original text is "The heart rate is 113 bpm higher than 100 bpm, which meets the definition of tachycardia."
Where 100 bpm is a **diagnostic definition threshold**, not a patient measurement. To pass the verification, you can only delete the definition,
Or forge an evidence item of `value=100` (the latter is prohibited by other rules). The model did not choose either path;
So repost it as it is.

At the same time, there are also structural difficulties in Articles 1–3: PR 191 ms / QRS 80 ms are all **normal values**,
The only landing mechanism is `diagnoses[].evidence`, which only exists for positive findings.
Normal measurements have no place in the current schema.

<a id="12-qwen-09001hr-四轮修订只改了一个字符"></a>
### 1.2 Qwen / 09001_hr —— Only one character was changed in four rounds of revisions

| Round | Model Round | Successful Tool Call | Rejected Call | Time Elapsed | Stop Reason |
|---|---:|---:|---:|---:|---|
| revise 1 | 7 | 3 | **11** | 34.09 s | **turn_limit** |
| revise 2 | 4 | 3 | 0 | 103.48 s | end_turn |
| revise 3 | 7 | 3 | 7 | 123.37 s | end_turn |
| revise 4 | 7 | 3 | 5 | 118.54 s | end_turn |

The total revisions were 379.5 s, accounting for **44.0%** of the entire 863.43 s.

```
synthesize   len=4794  md5=19a161c255
revise 1     len=3     md5=27c4e28d48   ← 输出只有 3 个字符，turn_limit
revise 2     len=4795  md5=05ca36a583
revise 3     len=4795  md5=05ca36a583   ← 与上一轮完全相同
revise 4     len=4795  md5=05ca36a583   ← 与上一轮完全相同
```

The **only** difference between synthesize and the final version:

```diff
- "citations": ["E2", "E3", "E5"]
+ "citations": ["E2", "E3", "E15"]
```

a character. One of the 12 verification issues has not been resolved, including: missing top-level key
`ranked_complete_interpretations`, `normal_ecg` and `left_atrial_abnormality`
Mutually exclusive, insufficient LAA evidence threshold, 7 ungrounded quantities, 1 arithmetic conflict.

Another record from the same run, `09000_hr`, is even worse:
`contract revisions did not return parseable JSON` - The revision completely changed the output.

---

<a id="2-根因"></a>
## 2. Root cause

<a id="r1-反馈是一次性平铺的问题清单没有优先级也没有可执行修法"></a>
### R1. Feedback is a one-time flat list of issues, with no priority and no executable fixes.

[`loop.py:589-602`](../ecgagent/agent/loop.py#L589-L602), in non-adjudicate mode
`_contract_feedback` only does two things: spell up to 16 questions into a bullet list,
One more general tip.

```python
lines = [
    "The verdict violates the deterministic output contract. Fix every item:",
    *[f"- {problem}" for problem in structural[:16]],
]
```

The model received 12 parallel requests, mixed with:

- **Mechanical** (delete a number in the description)
- **Structure class** (fill in a missing top-level key)
- **Substantial** (remove a diagnosis that does not meet the threshold)
- **Unsatisfiable class** (use the defined threshold as a patient measurement to implement it)

There is no signal to distinguish them. "Fix every item" When faced with a request that cannot be achieved,
The cheapest strategy is to change nothing - the behavior in both cases is exactly the same.

<a id="r2-不可满足的问题会毒化整批修订"></a>
### R2. Unsatisfactory problems will poison the entire revision

As long as there is one unsolvable item in the list, the model will tend to abandon all 12 items rather than fix the 11 items that can be fixed.
The current loop has no concept of "partial repair": `structural` is either empty or failed.

Two confirmed unsatisfiable/false positive rules:

- [`diagnostic_prompts.py:466-505`](../ecgagent/agent/diagnostic_prompts.py#L466-L505)
  `_reject_quantities` False positive for **define threshold**. It only exempts interval writing
  (`60-100 bpm` passes), single threshold is not exempted (`高于100 bpm` is blocked).
- [`semantic_guard.py:21-31`](../ecgagent/agent/semantic_guard.py#L21-L31)
  The negation blind spot of `_COMPARISON_RE` treats the correct "not reaching the threshold" as an arithmetic contradiction.
  See separate entry in §5 for details.

<a id="r3-修订预算是全局共享的一次解析失败就吃掉-14"></a>
### R3. The revision budget is shared globally, and 1/4 will be eaten once a parsing failure occurs.

[`loop.py:1200`](../ecgagent/agent/loop.py#L1200), `_request_revision` internal
`while result.revisions < self.max_revisions` shares the same structure/evidence revision**
Counter**, default `max_revisions = 4` ([`loop.py:353`](../ecgagent/agent/loop.py#L353)).

In the Qwen case, revise 1 returns 3 characters of non-JSON, which directly consumes the budget once.
There are only 3 real repairs left. Parse failures such as pure formatting accidents should not compete with semantic revision quotas.

<a id="r4-修订阶段允许工具调用模型会把回合花在工具上而不是改文本"></a>
### R4. The revision phase allows tool calls, and the model will spend turns on the tool rather than modifying the text.

Qwen's revise 1: 7 rounds, 3 successful calls, **11 rejected calls**, bumped
Only 3 characters will be output after `turn_limit`. The model is stuck in a retry loop of tool calls during the revision phase,
Didn't get to the point of editing JSON at all.

Most contract issues (deleting numbers, filling in keys, dediagnosing) **do not require any new evidence** and are purely text editing.
Current designs, however, hand the model the full toolset along with revision feedback.

<a id="r5-停机判据滞后触顶前会白烧两轮"></a>
### R5. The shutdown criterion lags behind, and two rounds will be wasted before reaching the top.

Stall detection requirements for [`loop.py:475-492`](../ecgagent/agent/loop.py#L475-L492)
Stop only if the signatures are exactly the same for two consecutive rounds. In other words, the first re-send will not stop.
Have to wait until the second time to stop - the MedGemma case burned in vain for 142 seconds.

And [`loop.py:452-458`](../ecgagent/agent/loop.py#L452-L458)’s `max_revisions`
It is determined that ** occurs before requesting the next round of **, so the two shutdown paths will compete with each other: the Qwen case collides first
`max_revisions` reports `failed deterministic contract validation`,
The MedGemma case hit stall first and reported `revision stalled`. The same phenomenon of "the model cannot be changed",
Two different error messages are generated, adding noise to subsequent statistics and troubleshooting.

<a id="r6-全有全无实质正确的报告被整条丢弃"></a>
### R6. All and nothing, substantively correct reports are discarded in their entirety.

The report of Qwen/09001 is actually quite good clinically: it correctly determined sinus rhythm, correctly identified and rejected
The atrial fibrillation detector has false positives and the reliability of correctly labeling the T-axis is questionable. Its only substantive error was that it overreported one
Left atrial abnormality. However, because the contract was not passed, the entire record was judged `failed`, and the top of the report was marked
"Evidence and semantic verification: not executed or status unknown".

136 (56.9%) of the peaking records actually produced reports, but failed the verification - currently
There is no mechanism to distinguish between "substantially usable but not fully compliant" and "outright failure".

<a id="r7-被拒工具调用完全不可观测"></a>
### R7. Rejected tool calls are completely unobservable

[`trace.py:77`](../ecgagent/trace.py#L77) only records the **count** of `rejected_calls`,
Tool names, parameters and rejection reasons are not recorded. Qwen/09001 was rejected 27 times throughout the process.
Completely black box when troubleshooting; this is why it burned 1,325,708 prompt tokens
(MedGemma similar task 566,952).

---

<a id="3-改进建议按性价比排序"></a>
## 3. Improvement suggestions (sorted by cost-effectiveness)

<a id="p1-反馈分层下发机械类问题直接给替换文本"></a>
### P1 — Feedback is distributed in layers, and mechanical issues are directly replaced with text.

Modify `_contract_feedback`, group them by repairability difficulty, and provide **exact modification actions** for mechanical problems:

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

The package allows the validator to bring the `severity` and `mechanical: bool` fields to each question.
Instead of the current plain string. This is the most profitable item on this list -
The convergence rate in rounds 1–3 is already 97.6%. Just push more records into this range.

<a id="p2-修订阶段默认关闭工具按需再开"></a>
### P2 — The tool is closed by default during the revision phase and reopened as needed.

The vast majority of contract issues are pure text editing. Suggestions:

- The first round of revision **does not give tools**, only gives feedback and current verdict, and requires a complete return JSON
- Only if the feedback contains "measurements need to be re-read" type issues (e.g.
  `[diagnosis_evidence_gate]`, value does not match), the tool will be opened in the second round

Directly eliminate R4 and significantly reduce token consumption.

<a id="p3-拆分修订预算"></a>
### P3 — Split revised budget

Split `max_revisions` into two independent quotas:

- `max_format_retries` (retry if parsing fails, recommendation 2)
- `max_semantic_revisions` (substantial revision, recommendation 3)

Parsing incidents no longer eat into semantic revision credits (R3). Also lowered the semantic revision limit from 4 to 3 -
The data shows that there is almost no gain in round 4, and cutting off can save about 120 s for each topping record.

<a id="p4-统一并提前停机判据"></a>
### P4 — Unify and advance shutdown criteria

- The stall detection is changed to **one round** and stops if there is no change (currently it requires two rounds), and `max_revisions`
  Top and stall are merged into the same error code, with a "last round of unfixed issue list"
- Write unfixed problems into `result.verification` during shutdown so that they no longer appear in the track
  There is a gap like "there is no deterministic verification report" - the tracks of the two cases cannot be found at present. What exactly did the verifier say?

<a id="p5-引入部分通过状态"></a>
### P5 — Introducing "Partial Pass" status

Add `contract_incomplete` in addition to `ok` / `verified`: the contract is not fully passed but
When the verdict structure is complete and there are no substantive contradictions, keep the report and list the **specific** failed items at the head of the report.
Rather than a general "not executed or status unknown". Corresponds to 136 items in R6.

<a id="p6-轨迹记录被拒调用明细"></a>
### P6 — Track Record Rejected Call Details

Record the tool name, parameter summary, and rejection reason for each rejected call in `trace.py`.
A few lines of code, but the current 27 failed calls are completely undetectable (R7).

<a id="p7-修掉两条会误伤正确推理的校验规则"></a>
### P7 — Remove two validation rules that may harm correct reasoning

These two items will create unsolvable feedback and directly trigger R2:

1. `_reject_quantities` Release the defined threshold - add a guide constant whitelist
   (60/100/120/200 bpm, 120 ms, 200 ms, 440/460/500 ms, etc.),
   Or limit the check to narrative fields such as `summary` / `complete_diagnosis`,
   No longer affects `reasoning`
2. Negation blind area of `_arithmetic_problems` - see §5

---

<a id="4-验证方式"></a>
## 4. Verification method

After the change you should be able to observe:

- The proportion of peaking (≥4 rounds) records decreased from 21.0% (239/1138)
- The overall verification rate of records entering revision increased from 67.1%
- The situation of outputting the same md5 in two consecutive rounds disappears (an assertion can be added to `trace`)
- The total number of `rejected_calls` in the revision stage dropped significantly (direct effect of P2)

The regression samples are fixed with these two items:

| Logging | Expected Behavior |
|---|---|
| `09000_hr` (MedGemma v16) | No longer stuck due to `100 bpm` definition threshold; self-negative diagnosis is completely removed in the first round instead of changing a code |
| `09001_hr` (Qwen v14) | Repair at least 7 unlanded numbers and missing top-level keys in the mechanical category in 4 rounds; `0.0039 mV*s 未达到 >0.04 mV*s` will no longer be sentenced to an arithmetic contradiction |

---

<a id="5-附算术守卫的-negation-盲区"></a>
## 5. Attachment: Negation blind area of arithmetic guard

Independent of the revision loop, but directly exacerbating R2 because it creates unanswerable feedback, note this as well.

[`semantic_guard.py`](../ecgagent/agent/semantic_guard.py)
`_arithmetic_problems()` Judge correct negative sentences as arithmetic contradictions. Minimum recurrence:

```
FLAG  V1导联P波终末力约为0.0039 mV*s，未达到左房异常的经典阈值（>0.04 mV*s）
        → "0.0039 > 0.04 is false"          【句子本身完全正确】
FLAG  QRS时限92 ms，未超过120 ms
        → "92 超过 120 is false"             【句子本身完全正确】
ok    PTF为0.0039 mV*s，低于0.04 mV*s的阈值
ok    心率62 bpm，未达到100 bpm的心动过速标准
```

Two independent flaws:

1. **negation blind spot. ** `context` group of `_COMPARISON_RE`
   `[^\d]{0,48}?` will swallow the negative word between the two numbers whole, and
   `_arithmetic_problems` never calls `_NEGATION_RE` (this regex is only
   `_positive_narrative_match` uses). `未超过` is not in the operator table,
   The regex then matches the bare `超过` and discards `未`; and the whole word `不超过` is in `_LESS_EQUAL`
   So it passes - purely a vocabulary inconsistency. `_NEGATION_RE` itself also lacks
   Not met/not exceeded/below/insufficient/far below.
2. **Unit prefix matching. ** `_UNIT` has no right boundary, `mV*s` will be matched as `mV`,
   Therefore, mV and mV·s are compared as having the same dimensions.

Fix: Run `_NEGATION_RE` for the entire matching range before reporting an error; complete the negative word list;
Add a right boundary to `_UNIT` so that `mV*s` / `mV·ms` no longer collapses into `mV`.

**Do not delete this check directly. ** Scanned 632 copies with complete verdict objects, 11 triggers,
**10 of them are real model errors** (such as "145 ms, slightly higher than the normal upper limit (200 ms)",
"63 bpm, consistent with the definition of tachycardia (>100 bpm)"), only Qwen is a false alarm.
But the sentence pattern that triggers false positives is exactly the standard way of writing when ruling out a diagnosis**.
So the most affected is the normal electrocardiogram.

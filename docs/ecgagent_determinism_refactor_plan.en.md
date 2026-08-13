<!-- i18n-nav -->
[中文](ecgagent_determinism_refactor_plan.md) | [English](ecgagent_determinism_refactor_plan.en.md) | [日本語](ecgagent_determinism_refactor_plan.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecgagent-确定性重构方案"></a>
# ECGAgent Deterministic Reconstruction Plan

Analysis Date: 2026-08-03
Analysis Targets: `ecgagent/agent/diagnostic_pathways.py` (v5), `ecgagent/agent/diagnostic.py`,
`ecgagent/evidence/model_view.py`, `ecgagent/evidence/diagnostic_contract.py`
Evidence Sources: Static path expansion of all 123 `DIAGNOSIS_CATALOG` diagnostic codes;
node-by-node audit of `qwen_agent_output/deepseek_v37_diverse50` (50 PTB-XL, DeepSeek) and
`record_results.csv`; tool/view live runs on `ptbxl_09000_ecgfeat/features/09000_hr_features.json`.

> **Version Note**: The static census is based on `ecgagent.diagnostic-pathways.v5` (code state as of 2026-08-03 04:56,
> 284 node instances). Node-by-node run statistics come from the v37 batch run, which was v4 (274 instances, 461 nodes actually evaluated).
> The ratios are consistent and the conclusions align, but **do not mix the absolute numbers from the two groups in the same table**.
> v5 adds 3 nodes relative to v4: `qt_threshold`, `alternative_qrs_cause_excluded`,
> `sinus_candidate_stream_reconciled`—two of which fall under "deterministic issues handed to the model" as described in this document,
> indicating that this pattern continues to emerge.

---

<a id="实施进度2026-08-03-更新"></a>
## Implementation Progress (Updated 2026-08-03)

| Step | Status | Implementation Details |
|---|---|---|
| Step 0 Shadow Mode | ✅ Implemented | Audit each node while logging `model_status` / `program_status`; `ECG_AGENT_DETERMINISTIC_NODES_SHADOW=1` one-click rollback (simultaneously flipping view routing and decision ownership) |
| Step 1 Path Syntax | ✅ Implemented | Added `invalidator` gate type; `_step()` rejects unknown gates; LVH adds two programmatic preconditions |
| Step 2 Node Determinization | ✅ Mostly Implemented | `ecgagent/agent/deterministic_pathways.py`, programmatic proportion 5.6% → **47.3%** |
| Step 3 Lift Candidate Cap | ⬜ Not Done | `MERGED_CANDIDATE_MAX = 5` unchanged |
| Step 4 H-Class Handling | ⬜ Not Done | 81 instances still determined by the model |

**Live Run Results (`deepseek_v38_determinism50`, 50 items, DeepSeek, same records and features as v37, 50/50 verified)**

Consistency with PTB-XL labels **decreased**: confirmed F1 **0.359 → 0.267** (TP 14→10, FP 18→19).
However, breaking it down, this decrease cannot be directly interpreted as a quality drop:

| Change | Count | True Positives | False Positives |
|---|---:|---:|---:|
| Lost Confirmations | 22 | 3 | 19 |
| New Confirmations | 16 | 1 | 15 |

Of the 22 lost, 19 were false positives—**this half is healthy**. The issue lies in the 15 new "false positives",
of which **7 were objectively correct upon measurement verification, but simply unannotated in PTB-XL**: 09887 PR=217ms, 09913 PR=213ms,
09933 HR=50.3bpm, 09166 QTc=321ms, 09557 QTc=485ms, 09249 all chest leads <1.00mV, 09756 all limb leads <0.50mV.
Determinization caused the agent to start outputting measurement-derived conclusions (PR/Heart Rate/QTc/Low Voltage), which are precisely the categories systematically missed by PTB-XL annotations.
**Record-level labels can no longer score this layer** (the batch's own RESULTS.md also states "unannotated abnormalities do not automatically equal false positives").

Mechanistically, determinization was successful: programmatic node proportion 15% → **67.8%**, programmatic node unknown rate 21.5%
vs. model node 52.5%; `invalidator` fired on 6 nodes, 4 LVH false positives (09025 IVCD, 09070 CLBBB,
09129 PACE, 09557 PACE) were rejected, and all 3 true LVH cases were retained.

**Three real defects exposed, all fixed** (`deepseek_v39_fixes50` verification in progress):

1. **09699 confirmed RBBB on a WPW record** (QRS 152ms, reference SBRAD|WPW). The root cause was not a missing node, but
   `preexcitation_excluded` only reading `short_pr_interval`—the PR interval was unmeasurable in this record, so it was false,
   but `short_pr_segment=true` and 9 leads had delta waves, causing the node to return unknown, excluding nothing.
   Fixed to align with `preexcitation_components` (either condition being true counts as short PR),
   and added this node to the bundle branch block path as `gate=invalidator`. **It must be an invalidator**: true RBBB record
   09545 returns unknown at this node; if used as required, it would block true positives as well.
2. **`alternative_qrs_cause_excluded` promoted to `required`**. "Whether no alternative explanation is shown" is an open-ended negative proposition,
   which a single view cannot prove; actual test results were 0 fail / 4 unknown—it can only stall, not reject, independently blocking 3 confirmations.
   Changed to `invalidator`.
3. **Isolated PVCs are structurally undiagnosable**. `_pvc_morphology_fact` requires non-dominant wide QRS group `member_count >= 2`,
   but 09388 is a single 218ms group. A single PVC in a 10-second record can therefore never be confirmed. Added
   `pvc_single_wide_qrs_ms = 140.0`: isolated but significantly widened groups can pass individually, while isolated groups of 120–140ms remain unknown.

**Not Fixed (Intentional)**: `right_precordial_voltage` returned unknown for 09275 and 09620, losing two true RVH positives.
Both records' reference diagnoses include CRBBB, and right bundle branch block itself produces a high R' in V1, invalidating right chest voltage criteria—
loosening this criterion for n=2 to match labels is exactly the practice this document opposes. The correct direction is to add a
CRBBB invalidation precondition for RVH as well (which would become a reject rather than confirm), left for decision-layer resolution.

<a id="v39-验证结果deepseekv39fixes505050-verified"></a>
## v39 Verification Results (`deepseek_v39_fixes50`, 50/50 verified)

| Level | v37 Baseline | v38 Determinization | v39 Post-Fix |
|---|---:|---:|---:|
| confirmed | **0.359** | 0.267 | **0.308** |
| confirmed + differential | 0.421 | 0.392 | 0.404 |
| All considered (including rejected) | 0.414 | 0.397 | 0.394 |
| Measurable confirmations / errors among them | 22 / **0** | 27 / **0** | 27 / **0** |

`invalidator` rejected 8 times (6 times in v38), the two new rejections being `preexcitation_excluded` for 09699 and 09275.
Among them, **09275 was a false rejection**—the record's reference diagnosis includes LAFB. Root cause: its PR was measured at 141ms (not shortened),
but I treated `short_pr_segment=true` as evidence of short PR. The PR segment equals the PR interval minus the P-wave duration,
The P wave is wide and the PR normal period will be short, which is not a pre-excitation. has been tightened to **only use segments as a replacement when the PR interval cannot be measured**
(The criterion is `/rhythm_inputs/record/availability/pr_available`. The pointer is in the `atrial_signal` section.
The two step parameters have been changed to `["preexcitation", "atrial_signal"]`). After correction, 09699 is still `fail`,
09275/09545 are all `unknown`, no more accidental damage.

**The remaining 5 true positives lost relative to v37, none of which were due to finalization after item-by-item attribution: **

| Records | Missing items | Reason |
|---|---|---|
| 09275 / 09620 | RVH | Deliberately not repaired (`right_precordial_voltage` is invalid under CRBBB) |
| 09114 | Atrial fibrillation | The model is changed to `atrial_fibrillation_flutter_indeterminate`. This code has no alias and is not in the atrial fibrillation category `prediction_codes` - **scoring strategy issue**, non-ability issue |
| 09388 | Dou Su | Model change nomination `atrial_tachycardia`, judged unknown by two model nodes |
| 09136 | Right bundle branch | `required_lead_pattern` determined by the model unknown |

That is: **3 are random fluctuations in the model's nomination/judgment round by round, and 2 are criterion issues that are deliberately retained. **
This gives an important methodological bound - 50 records + the measurement accuracy of the random model, which is of the same order as the effect size pursued by this protocol.
**Any single-round F1 difference less than about 0.05 should be considered noise** and conclusions must be drawn by attribution rather than headline numbers.

<a id="标签已经无法为这一层打分新增测量核对量尺"></a>
## Tags can no longer be used to score this layer - add a new measurement and verification ruler

`score_measurement_verifiable.py`: For the family "the definition itself is the measurement comparison" (PR, heart rate, QTc, low voltage, electric axis),
Check directly with `*_features.json` that records yourself instead of checking the label. Result:

| | verifiable confirmation | which is wrong | correct but not marked |
|---|---:|---:|---:|
| v37 | 22 | **0** | 19 |
| v38 | 27 | **0** | 25 |

**Both rounds were 100% correct on these families**, determinism only made the agent output 5 more (22 → 27).
Label scoring recorded 25 out of 27 v38 as false positives. So the F1 drop of v37→v38 must be read like this:
The real loss is 4 true positives for the morphological class family (3 of which 2 have been recovered by the above repair),
Rather than "added false positives". The column `measurement-contradicted` is the only number that cannot be fully explained by the labels,
Should be driven to 0 as one of the main indicators.

**Deterministic nodes remaining**: 31 instances (10.6%), down from 133. The composition is
`st_territorial_confirmation` 10, `tu_lead_confirmation` 6, `qrs_duration_support` 6 (branch block type,
The part of the incomplete bundle branch), `cross_lead_voltage_criterion` 3 (LVH code other than `lvh_voltage_criteria`),
`multilead_flutter_support` 2, `av_relation_characterized` 2, `q_wave_morphology` 2.
These are **intentionally reserved**: their criteria require adjacent lead topology or typing semantics of branch block, and the risk of blind writing outweighs the benefit,
The shadow mode should accumulate a round of different data before migrating.

1435 of all tests passed; 8 new tests for `invalidator` gate, LVH preconditions and shadow mode were added.

---

<a id="0-结论摘要"></a>
## 0. Conclusion Summary

**The problem is not that the model is not smart enough, it is that the model is placed in the wrong position. **

Three independent measurements pointing to the same thing:

1. **Of the 284 decision nodes in the diagnostic path, 133 (46.8%) are pure threshold comparisons, counts, or Boolean AND operations**,
   All are now labeled `pass/fail/unknown` by LLM. Only 16 (5.6%) programmers did it themselves.
2. **On the 69 nodes where both the program and the model gave answers, the disagreement rate between the two was 28%**; in the 12 cases that could be adjudicated,
   The program is all right and the model is all wrong. There were also 7 cases where the model gave a definite label to **evidence it had never seen**.
3. **The model-nominated candidate set F1 0.459 (recall 0.609) outperforms the pure rule baseline F1 0.426 (recall 0.500)**,
   But the final output is only F1 0.351 (recall 0.283) - **15 of the 28 correct nominations died in the node determination stage**
   (8 rejected, 7 unresolved).

In other words: **The "central brain" section (forming hypotheses, deciding what to watch) is already the strongest link in the entire system.
And its gains are eaten away by a layer of self-validation that it's incapable of doing. **

The direct goal of the reconstruction is to take back the node determination program from the model and allow the 28 correct nominations to reach the output alive.
That is, shipped F1 improves from 0.351 to 0.459 at the nomination level, one step above the rule baseline.

---

<a id="1-审计方法与可复现性"></a>
## 1. Audit methods and reproducibility

Four independent evidence chains, all reproducible:

| Method | What is done | What is obtained |
|---|---|---|
| Path static expansion | Call `build_diagnostic_pathway` for 123 diagnostic codes, enumerate all nodes | 60 different nodes, 284 (code × node) instances |
| Natural control experiment | Audit simultaneously records `requested_status` (model) and `effective_status` (program), both coexist when the deterministic gate is fired | Model-program divergence rate on 69 nodes |
| View actual running | Execute the tool+arguments of each node on the real record, and then run `build_model_evidence_view` | How much evidence does the model actually see |
| End-to-end attribution | Use the project's own `CATEGORIES` mapping to score the baseline/nomination set/final output respectively | In which segment the loss occurred |

Key trap (must be avoided in subsequent recurrences): `candidate_agent_codes` of `record_results.csv`
**Not** the nomination set - at `verified=True` it is equal to the final output. The real nomination set is each item in the audit
`code` field of `decision_audit`, regardless of `final_placement`.

---

<a id="2-现状测量"></a>
## 2. Current Measurement

<a id="21-节点普查v5284-个实例"></a>
### 2.1 Node census (v5, 284 instances)

| Class | Meaning | Instance | Proportion | Current Situation |
|---|---|---:|---:|---|
| P | Program confirmed | 16 | 5.6% | ✅ Processed by `_compact_deterministic_pathway_step` |
| T | Pure threshold comparison | 43 | 15.1% | ❌ Leave it to the model |
| C | count / run / scale | 66 | 23.2% | ❌ handed to model |
| R | Reliability boolean aggregation | 24 | 8.5% | ❌ handed to model |
| H | Hard standard + morphological distribution | 93 | 32.7% | ❌ Handed to the model; almost all corresponding implementations are in `clinical_rules/` |
| J | True Open to Judgment | 42 | 14.8% | ✅ Fit Model |

**T + C + R = 133 instances (46.8%) that should have been answered by the program but were given to the model. **
See Appendix A for the full table.

Deterministic processor `_compact_deterministic_pathway_step` (`ecgagent/agent/diagnostic.py:736`)
Only 7 step_ids are covered: `rate_threshold`, `axis_threshold`, `preexcitation_components`,
`preexcitation_excluded`, `dominant_qrs_wide`, `wide_qrs_representative`,
`cross_lead_voltage_criterion` (and only valid for `lvh_voltage_criteria`). End of function `return None`
Indicates that the rest will fall back to the model.

Three most glaring specific examples:

- **Written the same logic twice. ** `dominant_qrs_wide` uses `mean_qrs_ms >= 120 → pass, < 110 → fail`
  Programmed processing of IVCD's QRS widening; `qrs_duration_support` (used by 13 diagnostic codes) reads the same field,
  But leave it to the model. `pattern_representative` (13) and `wide_qrs_representative`
  `member_pct >= 50.0` Same reason. These 26 instances can be migrated with zero new logic.
- **The document already contains Boolean values, so we still need to ask the model. ** `short_pr_criterion` asks "whether PR satisfies the short PR definition",
  And `/rhythm_inputs/preexcitation/short_pr_interval` itself is an exported Boolean.
  The same applies to `interval_reportable` (6 pieces) and `component_endpoint_support` (6 pieces).
- **Threshold constants only live in Python. ** `pr_criterion` asks PR threshold,
  `ecgagent/agent/safety_policy.py:26` has `first_degree_av_block_pr_lower_exclusive_ms = 200.0`,
  `feature_extraction/ecgfeat/interpret.py:725` There is also a two-dimensional table DXL by age × heart rate.
  The model cannot see these numbers and can only guess at the defined boundaries; the thresholds are only overruled by `semantic_guard` after the fact.

<a id="22-模型-vs-程序天然对照实验"></a>
### 2.2 Model vs. Program: Natural Control Experiment

At the 69 nodes where the deterministic gate fires, the model's `requested_status` is the same as the program's `effective_status`
**Disagreement rate 28% (19/69)**. After checking the original values and PTB-XL reference diagnosis one by one:

| Node | Number of disagreements | Ruling | Typical examples |
|---|---:|---|---|
| `axis_threshold` | 4 | Program 4/4 Pairs | **09883 `qrs_axis_deg = +2.64°`, refer to NORM, the model judgment "satisfies the left deviation of the electrical axis"** (definition −90°～−30°). It's not boundary rounding, it's just no comparison at all. In addition, −29.71° and −27.93° were sentenced to pass |
| `cross_lead_voltage_criterion` | 8 | Program 8/8 Pair | Forget about the model, Cornell calls it a day, leaving out Peguero's "or" branch. 09537 Male Cornell 2.349 < 2.8 but Peguero 2.672 ≥ 2.3; 09070 Female Cornell 4.376 far exceeds the 2.0 threshold and the model gives unknown |
| `dominant_qrs_wide` / `wide_qrs_representative` / `preexcitation_excluded` | 7 | The model should not answer | The model gives OK pass/fail, and the program returns unknown - because **This pointer has never entered the authorized view of the model** (09144, 09550 3 nodes each) |

**Conclusion: The failure mode is not "LLM can't do arithmetic", but four different things:**

1. The threshold constant is not delivered to the model, it is guessing the definition;
2. Only one of the multi-branch criteria (Cornell ** or ** Peguero) goes;
3. **Don’t give up when you can’t see the data** and still answer confidently;
4. The data of the counting node itself is truncated to the point that it cannot be answered (see 2.3).

<a id="23-证据可见性"></a>
### 2.3 Visibility of evidence

Run tool + `build_model_evidence_view` on real records for 56 executable nodes:
**Only 1981 of the 5855 citations eventually entered the model context, 66% were discarded**, and 17 nodes lost more than half.

| node | reference count → visible atoms |
|---|---|
| `repolarization_context` | 612 → 40 |
| `alternative_qrs_cause` | 468 → 41 |
| `pr_criterion` | 298 → 48 |
| `t_wave_distribution` | 132 → 37 |

It is fatal when superimposed with counting nodes: `ecgagent/evidence/model_view.py:33`
`_INDEXED_GROUP_LIMIT = 8` will equidistantly sample event-by-event/beat-by-beat tables into 8 groups. Actual measurement 09000
This record has 28 P events (3 blocked), `get_atrial_event_table(limit=16)` renders 96 citations,
What the model finally sees is the event [0, 2, 4, 6, 9, 11, 13, 15], these 8 non-adjacent samples**,
1 out of 3 blocked events is not included at all.

Therefore, questions such as "count how many undownloaded P's", "whether there is a 1:1 correlation" and "whether there are ≥2 adjacent leads",
It is **Structurally impossible** to answer correctly on the current view, regardless of model capabilities.

<a id="24-端到端损失定位v3750-条类别级-microsemanticscope-direct"></a>
### 2.4 End-to-end loss location (v37, 50 items, category-level micro, `semantic_scope == "direct"`)

| | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Regular baseline (no LLM) | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
| **All candidates for model nomination** | **28** | 48 | 18 | 0.368 | **0.609** | **0.459** |
| Agent final output (confirmed) | 13 | 15 | 33 | 0.464 | 0.283 | 0.351 |

**Outperformed the rule baseline during the nomination stage** and found LVH, sinus velocity, etc. that were missed by the baseline.
**Of the 28 correct nominations, only 13 came out alive and 15 died: 8 `rejected`, 7 `unresolved`**,
Involving LAFB, PVC, PAC, AF, LBBB, IVCD, LPFB. The mechanism that kills them is the arithmetic nodes of 2.1——
A `required` node returned fail or unknown.

Causal chain closure:
Nominate 28 correct hypotheses → Each must pass 2–4 numerical nodes that cannot be calculated by the model →
36% of the nodes returned unknown, 28% were inconsistent with the program judgment → 15 correct hypotheses died → the recall dropped from 0.609 to 0.283.

---

<a id="3-四层问题分解"></a>
## 3. Four-level problem decomposition

Now these four things are discussed together, so each modification only moves one layer. **They must be established separately. **

| Layer | Question | Evidence | Is this plan covered |
|---|---|---|---|
| **L1 nominations truncated** | `MERGED_CANDIDATE_MAX = 5`, discarding ~1.6 rule candidates per record | Nomination F1 0.459 itself measured under this hat, higher ceiling | ✅ Step 3 |
| **L2 path syntax lacks veto type** | Only `required`/`supporting`, the supporting structure cannot be vetoed | LVH's `qrs_morphology_compatible` is still `supporting` and can never be stopped; 5 of the 6 false positives have QRS widening confounding factors | ✅ Step 1 |
| **L3 node determination is left to the model** | 133 of the 284 nodes are arithmetic | 28% disagreement rate, 15 correct diagnoses died from this | ✅ Step 2 |
| **L4 feature/criteria layer defect** | infarction 0/16, ischemia 1/18 | It has been confirmed that ** is not** restricted by the gate | ❌ **Independent project, not included in this plan** |

---

<a id="4-目标架构程序算事实模型解读"></a>
## 4. Target architecture: program to calculate facts, model interpretation

The data flow is divided into six segments, each segment only does one thing:

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

Key points in the design of each layer:

**① Fact layer.** Complement the deterministic derived quantities in addition to existing measurements: P/QRS count, continuous undownloaded P runs,
Atrioventricular conduction ratio, PR mean/standard deviation/linear slope, PR difference before and after block, adjacent lead count, supporting lead count.
This is the only place where the `feature_extraction/ecgfeat` export layer needs to be touched.

By the way, fix the known "forget it but throw it away": `feature_extraction/ecgfeat/rhythm_rules.py:561-562`
`two_to_one_conduction_suspected`, `av_block_level_hint`, and
`second_degree_avb_basis` produced by `_classify_second_degree_type`
(Including `preceding_pr_ms`/`following_pr_ms`/`pr_increment_ms`/`pr_spread_ms`, which is the identification basis of Mobitz I/II)
None of them are in the `av_block` dictionary of `export.py`, they only exist in those stripped by `_FORBIDDEN_KEYS`
Under `metadata/rhythm_analysis`, the agent is provably unavailable.

**② Criterion layer.** Hard constraint: `unknown` only allows one source - **Measurement not available**.
The unknown "not calculated" is not allowed to appear again. Most of the current 36% unknowns are the latter.
Each node must simultaneously produce "conclusion + basis pointer + whether the criterion is applicable in this case."

**③ Nomination layer.** **Keep blind design** (`_FORBIDDEN_KEYS` vs. `clinical_interpretation` /
Stripping of `interpretation` / `rhythm_analysis`) as it works - Nominated F1 0.459 > Baseline 0.426.
The blind state is only valid during the nomination stage, and can be released after the nomination is frozen ("blind first and then test").

**④ Veto layer.** This is the real incremental value of the model: the rules engine can only exclude things that someone has enumerated beforehand.
The LVH false positive (09129) on the pacing record is this hole - there is no pacing exclusion node in `clinical_rules/`.
This layer can only subtract but not add, and reasons and references must be given, so the cost of misjudgment by the model is the loss of sensitivity.
And it leaves an auditable record; unlike now, the positive results created out of thin air by 09883 are silent.

**⑥Judgment layer.** Confidence level from `HIGH if len(support_tools) >= 2 else MEDIUM`
("Several tools have contributed", regardless of the strength of the evidence) Change it to a function of the node source:
All node programs are confirmed → HIGH; including model rejection → MEDIUM; including unknown → not confirmed but **progress to differential diagnosis**.
Now the 7 correct diagnoses disappear after being stuck in unresolved.

---

<a id="5-实施方案"></a>
## 5. Implementation plan

**The order cannot be adjusted**. For the reason, see "Why are you at this position" in each step?

<a id="第-0-步影子模式非破坏性必须最先"></a>
### Step 0: Shadow Mode (non-destructive, must be first)

Let the program calculate the answer for all 284 nodes and write the audit, but without changing any behavior, the model label will still be in effect.

- Output: A "program vs model" comparison of the full node is generated free of charge for each real run, that is, the one in §2.2
  The natural experiment, which only covered 69 nodes, was expanded to 284.
- **Why in this position**: Without this step, every subsequent migration step will be a blind modification. With it,
  The migration of each node is supported by data, and the migration priority can be determined by sorting "high divergence rate".
- Acceptance: Each `pathway_steps` entry in the audit contains both `model_status` and `program_status`;
  Run 50 records at a time to produce a full node divergence rate table.

<a id="第-1-步路径语法引入前置条件否决节点类型"></a>
### Step 1: Path Syntax - Introducing Preconditions/Deprecating Node Types

- Add `precondition` / `invalidator` types in addition to `gate` of `_step()`,
  Enable the veto node to truly block confirmation (the current status of `supporting` does not enter `required_statuses` at all).
- Fix LVH immediately: upgrade `qrs_morphology_compatible` to blocking type, or add
  QRS Width/pacing validity precondition.
- **Why must be before or at the same time as step 2**: The voltage node program is arithmetically 8/8 correct, but out of 5 records passed
  Only 2 are true LVH. **The model cannot calculate this and is accidentally covering up an overly sensitive criterion. **
  Do the determination first, the LVH false positive will increase from 6. Precondition nodes are a hedge against this side effect.
- Acceptance: v37 LVH false positive 6 → 1 on the same batch of records, true positive 09089 reserved.

<a id="第-2-步节点确定化分三批"></a>
### Step 2: Node determination (in three batches)

| Batch | Scope | Instance | Dependency | Risk |
|---|---|---:|---|---|
| 2a | Duplicate implementation: `qrs_duration_support`, `pattern_representative` | 26 | None, reuse `dominant_qrs_wide` branch existing logic | Very low |
| 2b | Pure Threshold + Reliability Boolean (T + R) | 67 | Constant already in `safety_policy.py` / `clinical_rules/config.py` | Low |
| 2c | Count/Run/Proportion (C) | 66 | Depend on the new export field in layer ① | Medium |

**Collateral benefits**: `EvidenceStore` is read directly after the node is returned to the program, without going through the model context budget.
§2.3 That 66% discard rate no longer makes sense to them. Therefore, the repair scope of the view budget is greatly reduced,
Only the veto layer and the narrative layer still need to see evidence.

- Acceptance (each batch): Run the same 50 items, the nomination recall remains unchanged (0.609), and the shipped recall increases monotonically.
  Shadow mode divergence rates are reset to zero on migrated nodes.

<a id="第-3-步放开-mergedcandidatemax"></a>
### Step 3: Let go of `MERGED_CANDIDATE_MAX`

- **Why in this position**: Do not dare to relax nominations until the verification layer is reliable, otherwise more candidates will only produce more false positives.
- Acceptance: Nominated recall > 0.609, shipped accuracy no less than at the end of step 2.

<a id="第-4-步h-类-93-个节点的处置必须二选一"></a>
### Step 4: Disposal of 93 nodes of Class H (must choose one of the two)

The model has no visible waveforms - the 14 tools are all tables of values, without any image/drawing tools.
So currently it is asked to do "cross-lead shape distribution judgment", which is essentially still doing arithmetic on the value table, but with a different name.
This explains the 51% unknown rate for class H. Two paths:

- **(a) Give signal**: Connect to multi-modality and let the model really see the picture (the project already has the foundation of `ecg_plots` and medgemma).
- **(b) Return to program**: Connect to the existing `_rbbb`/`_lbbb`/ischemia implementation of `clinical_rules/`,
  Put in after nomination freeze ("blind first").

**It is recommended to go first (b)**: Reuse existing code without destroying the value of blind design. (a) is a new research question, starting from a different track.

---

<a id="6-度量纪律"></a>
## 6. Measurement Discipline

Staring at three numbers, the current baseline:

| Metrics | Current | Target |
|---|---:|---|
| Nominated Recall | 0.609 | Not falling (rising after step 3) |
| shipped recalled | 0.283 | → 0.609 |
| **The difference** | **0.326** | **→ 0** |

Polaris is **Gap**, not absolute F1. Zeroing the gap means shipped F1 0.351 → 0.459, a step above the regular baseline of 0.426.

**Expected normal phenomenon: shipped accuracy will drop first** (the over-sensitivity of the criterion is exposed),
**Don’t think of it as a regression**. This is exactly the purpose of decoupling "criteria errors" and "calculation errors" - now these two things are entangled,
Can't even make an attribution.

---

<a id="7-已知边界与陷阱"></a>
## 7. Known boundaries and pitfalls

1. **L4 architecture cannot be saved. ** infarction 0/16 and ischemia 1/18 are the two largest holes.
   Confirmed not to be gate restricted. **Don’t do it in the same iteration as the architecture transformation**, otherwise it will become
   Two things are entangled and cannot be attributed - this project has already stepped on this pit. An independent project should be established: first check the feature layer of those 16 items,
   It is judged whether ecgfeat cannot be measured or the criterion is not written correctly.
2. **Certainty ≠ Correct. ** The voltage node program is arithmetically 8/8 correct, but 3 of the 5 that pass are not true LVH.
   Removing nodes from the program will expose flaws in the criterion itself - this is the purpose, not a side effect.
3. **Nomination accuracy is only 0.368. ** 48 false positives are more than the baseline’s 39. The model is a **wide nominator**:
   Casting a wide net results in high recall and low precision. Wide nominations + reliable verification = good system; wide nominations + unreliable verification = now.
   Determining the validation layer is a necessary condition, but not a sufficient condition.
4. **Do not use `candidate_agent_codes` as the nomination set** (see §1).
5. The ** model is still spreading. ** Among the 3 new nodes added during this audit period (2026-08-03 04:56),
   `qt_threshold` (pure threshold, `safety_policy` already has constants) and
   `sinus_candidate_stream_reconciled` (all read as Boolean) both belong to the category of "leaving deterministic problems to the model".
   It is recommended to add an item to the review list: **When adding a required node, you must explain why it cannot be answered by the program. **

---

<a id="附录-a284-个节点实例的完整分类"></a>
## Appendix A: Complete classification of 284 node instances

Category meaning: `P` Program Determined · `T` Pure Threshold · `C` Count/Run/Scale · `R` Reliability Boolean Aggregation ·
`H` Hard standard + morphological distribution · `J` Really open to judgment.
"V37 measured P/F/U" is the count of pass/fail/unknown, which comes from the batch run of v4. The new node of v5 shows "not appearing".

| Class | Node id | Instance | gate | Tool | v37 measured P/F/U | Determination content · Ready-made implementation location |
|---|---|---:|---|---|---|---|
| P | `rate_threshold` | 5 | required | `get_global_table` | 15/0/0 | Heart rate vs 60/100 |
| P | `axis_threshold` | 3 | required | `get_global_table` | 10/13/0 | Electric axis interval |
| P | `dominant_qrs_wide` | 2 | required | `get_morphology_groups` | 1/2/2 | mean_qrs_ms vs 120/110 |
| P | `preexcitation_excluded` | 2 | required | `get_rhythm_profile` | 0/0/5 | Same as above |
| P | `wide_qrs_representative` | 2 | required | `get_morphology_groups` | 1/0/4 | QRS>=120 AND member_pct>=50 |
| P | `preexcitation_components` | 1 | required | `get_rhythm_profile` | 1/0/0 | Short PR AND delta lead number >=2 |
| P/T | `cross_lead_voltage_criterion` | 4 | required | `get_lead_table` | 15/3/2 | Cornell/Peguero; only lvh_voltage_criteria switch-on procedure |
| T | `qrs_duration_support` | 13 | required | `get_morphology_groups` | 10/5/1 | Read the same mean_qrs_ms, which is exactly the same logic as dominant_qrs_wide |
| T | `interval_reportable` | 6 | required | `get_interval_waveform_context` | 9/0/2 | qt_reliability / reliable_for_qt is already boolean |
| T | `qt_threshold` | 6 | required | `get_interval_waveform_context` | Did not appear | QTc threshold; safety_policy already has a constant ← New, missing program |
| T | `irregular_ventricular_response` | 4 | required | `get_rhythm_profile` | 4/18/0 | rr_cv / rr_rmssd / rr_entropy vs threshold |
| T | `territorial_qrs_voltage` | 4 | required | `get_lead_table` | 0/2/0 | Limb conduction <0.50mV / Chest conduction <1.00mV, constant at clinical_rules/config.py |
| T | `pr_criterion` | 3 | required | `get_interval_waveform_context` | 2/4/4 | PR vs 200ms; safety_policy and DXL age × heart rate monitor already exist |
| T | `right_precordial_voltage` | 3 | required | `get_lead_table` | 2/2/2 | V1 R/S ratio threshold |
| T | `short_pr_criterion` | 1 | required | `get_interval_waveform_context` | not present | short_pr_interval is already Boolean in the document |
| C | `pattern_representative` | 13 | required | `get_morphology_groups` | 12/1/3 | member_pct>=50, wide_qrs_representative branch has been implemented |
| C | `st_territorial_confirmation` | 10 | required | `get_lead_table` | 3/3/2 | >=2 adjacent leads, ischemia.py implemented |
| C | `tu_lead_confirmation` | 6 | required | `get_lead_table` | 0/1/2 | consistent lead count |
| C | `capture_relation_support` | 5 | required | `get_pacing_profile` | Not Appeared | Capture Pulse Ratio |
| C | `pacing_marker_support` | 5 | required | `get_pacing_profile` | not present | pacing flag count/confidence |
| C | `multilead_atrial_support` | 4 | supporting | `get_rhythm_profile` | 5/5/12 | Supporting lead counting |
| C | `repeated_blocked_atrial_events` | 4 | required | `get_atrial_event_table` | 7/2/0 | Statistics association_type=='blocked' Number |
| C | `sequential_av_pattern` | 4 | required | `get_rhythm_profile` | 2/2/5 | Conductivity ratio and continuous run |
| C | `one_to_one_av` | 3 | required | `get_atrial_event_table` | 6/4/0 | The download P count corresponding to each QRS |
| C | `premature_timing` | 3 | required | `get_beat_table` | 4/1/4 | rr_prev vs baseline ratio |
| C | `representative_event` | 3 | required | `get_morphology_groups` | 1/1/7 | member_count / longest_run threshold |
| C | `av_relation_characterized` | 2 | required | `get_atrial_event_table` | 0/1/2 | Conductivity ratio |
| C | `multilead_flutter_support` | 2 | required | `get_atrial_event_table` | 0/0/3 | Support lead counting |
| C | `q_wave_morphology` | 2 | required | `get_lead_table` | 0/1/3 | q time limit >= threshold AND q/r >= threshold, ischemia.py has been implemented |
| R | `component_endpoint_support` | 6 | required | `get_interval_waveform_context` | 4/1/6 | t_fusion_reliable / qt_reliability Boolean |
| R | `p_measurement_reliability` | 5 | required | `get_p_assessment_table` | 1/0/1 | accepted count vs minimum value |
| R | `organized_p_absent` | 4 | required | `get_p_assessment_table` | 4/15/3 | accepted Boolean |
| R | `sinus_p_support` | 4 | required | `get_p_assessment_table` | 4/1/1 | accepted / ta_ambiguous Boolean |
| R | `organized_atrial_activity` | 2 | required | `get_rhythm_profile` | 4/2/0 | availability Boolean |
| R | `sinus_candidate_stream_reconciled` | 2 | required | `get_rhythm_profile` | not present | localized_atrial_event_excess etc. Boolean ← New, missed program |
| R | `pr_component` | 1 | supporting | `get_interval_waveform_context` | 0/0/1 | reliable_for_pr Boolean |
| H | `required_lead_pattern` | 13 | required | `get_morphology_map` | 8/3/5 | conduction.py's _rbbb/_lbbb |
| H | `st_change_distribution` | 10 | required | `get_morphology_map` | 3/3/2 | ischemia.py ST threshold + adjacency |
| H | `repolarization_context` | 8 | required | `get_native_beat_profile` | 7/3/6 | repolarization.py |
| H | `t_wave_distribution` | 8 | required | `get_morphology_map` | 8/2/6 | repolarization.py / t_morphology.py |
| H | `atrial_mechanism` | 6 | required | `get_rhythm_profile` | 1/2/5 | atrial_rhythm.py |
| H | `p_morphology_support` | 6 | required | `get_p_assessment_table` | 1/2/5 | atrial_rhythm.py |
| H | `precordial_progression` | 6 | required | `get_lead_table` | 4/0/3 | r_progression.py Transition zone rules |
| H | `tu_morphology_distribution` | 6 | required | `get_morphology_map` | 0/1/2 | u_wave.py / t_morphology.py |
| H | `p_wave_distribution` | 5 | required | `get_lead_table` | 1/0/1 | atrial_abnormality rules |
| H | `qrs_morphology_compatible` | 4 | supporting | `get_morphology_map` | 6/0/14 | hypertrophy.py (**LVH's veto node, structurally unblockable**) |
| H | `ectopic_morphology` | 3 | required | `get_morphology_groups` | 4/0/5 | ectopy.py |
| H | `rvh_qrs_distribution` | 3 | required | `get_morphology_map` | 2/0/4 | hypertrophy.py |
| H | `atrial_relation` | 2 | required | `get_rhythm_profile` | 0/0/1 | av_block.py |
| H | `organized_flutter_activity` | 2 | required | `get_rhythm_profile` | 0/1/2 | rhythm.py Room flutter rules |
| H | `q_wave_distribution` | 2 | required | `get_native_beat_profile` | 0/0/4 | ischemia.py |
| H | `sinus_mechanism_support` | 2 | required | `get_global_table` | 6/3/4 | basic_rhythm.py |
| H | `specific_bbb_excluded` | 2 | required | `get_morphology_map` | 1/0/4 | Same as above |
| H | `ventricular_morphology` | 2 | required | `get_morphology_groups` | 0/0/1 | wide_tachycardia.py |
| H | `wide_complex_sequence` | 2 | required | `get_morphology_groups` | 0/0/1 | wide_tachycardia.py |
| H | `qrs_onset_support` | 1 | supporting | `get_morphology_map` | 1/0/0 | preexcitation.py |
| J | `direct_measurement_support` | 13 | required | `get_native_beat_profile` | 1/0/1 | Open judgment, no corresponding rule implementation |
| J | `discriminative_countercheck` | 13 | required | `get_native_beat_profile` | 0/0/2 | Open judgment, no corresponding rule implementation |
| J | `secondary_qrs_context` | 10 | supporting | `get_native_beat_profile` | 3/0/5 | Open judgment, no corresponding rule implementation |
| J | `alternative_qrs_cause_excluded` | 6 | required | `get_native_beat_profile` | not present | open exclusion ← new, gate=required |

---

<a id="附录-b复现要点"></a>
## Appendix B: Reproduction Points

- **Node Census**: Call `build_diagnostic_pathway(code)` for each code of `DIAGNOSIS_CATALOG`,
  Statistics `steps[].id`.
- **Model vs Program**: Traverse `diagnoses/*.json` and find all objects containing `pathway_steps`,
  Sieve `resolution_owner == "deterministic_measurement_gate"`, compare `requested_status`
  with `effective_status`.
- **Evidence Visibility**: Use **New** `build_default_registry(store)` for each node (budget per tool,
  Reusing the same registry will return `ok=False` starting from the Nth node),
  Then feed the result to `build_model_evidence_view` and compare `len(citations)` with `atom_count`.
- **End-to-end scoring**: Use `evaluate_target_ecgfeat_diagnosis.CATEGORIES` for mapping
  (`spec.ptbxl_refs` is used for the reference side and `spec.prediction_codes` is used for the prediction side),
  Only the category `semantic_scope == "direct"` is counted. Each item in the nomination collection audit is `decision_audit`
  of `code`, regardless of `final_placement`.

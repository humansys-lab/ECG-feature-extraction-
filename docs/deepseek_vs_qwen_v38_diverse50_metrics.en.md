<!-- i18n-nav -->
[中文](deepseek_vs_qwen_v38_diverse50_metrics.md) | [English](deepseek_vs_qwen_v38_diverse50_metrics.en.md) | [日本語](deepseek_vs_qwen_v38_diverse50_metrics.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="deepseek-vs-qwenv38-diverse50指标计算方法与结果全记录"></a>
# DeepSeek vs. Qwen (v38 diverse50) index calculation method and full record of results

Record the **precise definition, calculation formula, and complete results** of each indicator used in the analysis on 2026-08-03 for subsequent recalculation, review, or reference when writing reporting materials to avoid re-derivation.

<a id="0-数据来源"></a>
## 0. Data source

- `qwen_agent_output/deepseek_v38_diverse50/`: 50 records of DeepSeek running as backbone
- `qwen_agent_output/qwen_v38_diverse50/`: 50 records of Qwen running as backbone
- `dataset_dir` in the two `record_manifest.json` are exactly the same (both point to `qwen_v36_diverse50/wfdb_headers`). It is confirmed that they are the same batch of 50 PTB-XL records and can be directly compared head-to-head.
- Scripts involved: `evaluate_target_ecgfeat_diagnosis.py` (`CATEGORIES` definition) and `ecgagent/batch.py` (`_category_set`/`_metric_rows`) that come with the warehouse generate `category_metrics.csv`/`record_results.csv` official logic), `score_measurement_verifiable.py` (measurement review); the part marked "temporary script" in this document was written during this analysis and is not in the warehouse. The formulas are listed below for reproduction.

---

<a id="1-第一层逐记录-逐诊断家族的四格表"></a>
## 1. The first level: record by record × four grid table by diagnosis family

<a id="11-诊断家族定义"></a>
### 1.1 Diagnostic family definition

`CATEGORIES: tuple[CategorySpec, ...]` is predefined in `evaluate_target_ecgfeat_diagnosis.py`, and there is one `CategorySpec` for each family:

```python
@dataclass
class CategorySpec:
    key: str                          # 家族 id，如 "atrial_fibrillation"
    display_name_zh: str              # 中文名
    ptbxl_refs: frozenset[str]        # 参考标签集合，如 {"AFIB"}
    local_refs: frozenset[str]
    prediction_codes: frozenset[str]  # 模型/规则引擎输出中，命中该家族的代码集合
    evaluates_codes: frozenset[str] = frozenset()
    rule_ids: frozenset[str] = frozenset()
    semantic_scope: str = "direct"    # "direct" | "broad" | "screening"
```

Among the 50 records used in this analysis, there are **17** families of `semantic_scope="direct"` (see Appendix A for the complete list). The headline indicator only summarizes the `direct` family. The definition of the `broad` (such as T wave abnormality, atrial abnormality, QRS low voltage) and `screening` (ischemia/ST screening, myocardial infarction/Q wave screening) families is ambiguous and the criterion is not a single numerical comparison, so they are excluded for separate discussion.

<a id="12-四格表判定"></a>
### 1.2 Judgment of four-square table

For each record `i`, each family `spec`:

```
reference_positive = (记录 i 的 PTB-XL 标签集合) ∩ spec.ptbxl_refs ≠ ∅
predicted_positive  = (模型/规则引擎输出的代码集合) ∩ spec.prediction_codes ≠ ∅
```

| | reference_positive | ¬reference_positive |
|---------------------|--------------------------|----------------------|
| **predicted_positive** | TP | FP |
| **¬predicted_positive**| FN | TN (not participating in F1) |

Accumulate record by record and family by family to obtain the TP/FP/FN of each family in `category_metrics.csv` (`ecgagent/batch.py::_metric_rows`).

---

<a id="2-第二层precision-recall-f1及五种平均口径"></a>
## 2. Second level: Precision/Recall/F1, and five average calibers

Standard formula:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

For the same four-grid table data, there are many recognized methods of "how to average", and the conclusions will be different. This time I calculated them all:

<a id="21-micro-f1汇总后算一次"></a>
### 2.1 Micro-F1 (counted once after summarizing)

First add the TP/FP/FN of all `direct` families and all 50 records into a total table, and then calculate P/R/F1. **The default caliber of this report** is also the first number given.

<a id="22-macro-f1label-centric逐家族先各算-f1-再平均"></a>
### 2.2 Macro-F1 (label-centric, calculate F1 for each family first and then average)

```
Macro-F1 = (1/|F|) · Σ_{f∈F} F1_f
```
`F` is a set of 17 direct families, and each family has equal weight, regardless of whether it appears 1 time or 6 times in 50 records. This is the core caliber selection of [PTB-XL official benchmark paper](https://arxiv.org/abs/2004.13701) (Strodthoff et al. 2020) §II.C (the original text is used for AUC, this time our output is a binary judgment, without probability score, so the same "class-by-class equal weight average" idea is applied to F1): Original words of the paper - "macro-averaging is preferred, since we expect class imbalance and do not want the score to be dominated by a few large classes".

<a id="23-sample-centric-f1record-centric逐记录先各算准召再平均"></a>
### 2.3 Sample-centric F1 (record-centric, record-by-record calculation and then average)

Also from [PTB-XL paper](https://arxiv.org/abs/2004.13701) §II.C, formula derived from CAFA Protein Function Prediction Challenge:

```
Precision = (1/N_P) · Σ_{i: |P_i|>0}  TP_i / |P_i|      # 只在"该记录至少预测了1个家族"的记录上取平均
Recall    = (1/N_T) · Σ_{i: |T_i|>0}  TP_i / |T_i|      # 只在"该记录真实至少有1个家族"的记录上取平均
F1        = 2 · Precision · Recall / (Precision + Recall)
```

Where `P_i`/`T_i` is the predicted/real direct family set of record `i`, and `N_P`/`N_T` is the number of records with non-zero denominator. The denominator processing of the prediction side in the original paper is "only average on the samples with predictions". This time, the recall side is also symmetrically limited to "average on the samples with real labels" (because a large number of records in this data set have empty real labels of the direct family, which is different from the CAFA scenario where each protein must have a functional label. This symmetrical processing is required to prevent division by zero). **The Fmax of the original paper will also take the maximum value of the prediction threshold τ scan**; the output of this model is a hard binary judgment (there is no probability score to scan), which is equivalent to only counting the F1 of the only point on the threshold scan curve, not the real Fmax.

<a id="24-subset-accuracy子集精确匹配率"></a>
### 2.4 Subset Accuracy (subset exact matching rate)

```
Subset Accuracy = (1/N) · Σ_i 𝟙[P_i = T_i]
```
Determine "whether the predicted family set is completely consistent with the real label set" record by record, yes/no binary, one of the classic indicators in the multi-label classification literature.

<a id="25-平均-jaccard-相似度"></a>
### 2.5 average Jaccard similarity

```
Jaccard_i = |P_i ∩ T_i| / |P_i ∪ T_i|     （P_i、T_i 都为空时记为 1）
Mean Jaccard = (1/N) · Σ_i Jaccard_i
```

---

<a id="3-第三层-a测量复核scoremeasurementverifiablepy"></a>
## 3. The third level A: Measurement review (`score_measurement_verifiable.py`)

The first and second layers only look at whether "model code string vs label code string" matches, regardless of whether the values are correct or not. For families whose definition itself is a threshold comparison (heart rate, PR/QT interval, QRS voltage, spindle), the script bypasses the label and recalculates directly with the `*_features.json` record itself:

```python
CHECKS = {
    "sinus_bradycardia": lambda r: r.g("heart_rate_bpm") < 60.0,
    "first_degree_av_block": lambda r: r.g("pr_ms") > 200.0,
    "prolonged_qt": lambda r: r.qtc() > (460.0 if r.female else 450.0),
    "left_axis_deviation": lambda r: -90.0 < r.g("qrs_axis_deg") < -30.0,
    "low_qrs_voltage_limb_leads": lambda r: all(amp < 0.50 for amp in 肢导振幅),
    # ... 完整列表见 score_measurement_verifiable.py
}
```

For the diagnosis given by each model, if the code is in `CHECKS`, the true value will be recalculated and divided into four buckets:

| Bucket | Meaning |
|---|---|
| `measurement-confirmed and labelled` | It is calculated to be true and the label is also written |
| `unlabelled-but-correct` | The calculation is true, but the label did not mention it (PTB-XL reported that the doctor did not mention it, so it is not considered a model error) |
| `measurement-contradicted` | Calculated to be false——**True mistake** |
| `not-checkable` | There is no corresponding `CHECKS` function (morphological families such as LVH/RVH/WPW), or the record lacks characteristic data |

Usage: `python3 score_measurement_verifiable.py qwen_agent_output/deepseek_v38_diverse50 qwen_agent_output/qwen_v38_diverse50`

---

<a id="4-第三层-b乐观口径两步修正临时脚本本次现算未入库"></a>
## 4. The third layer B: Optimistic two-step correction (temporary script, calculated this time, not included in the database)

Both steps are based on the first-level four-grid table, with the purpose of peeling off two types of known accounting noise from the "surface label-F1" and obtaining an optimistic estimate of the "real misjudgment rate".

<a id="41-第一步验证过的同义代码合并"></a>
### 4.1 Step 1: Merge verified synonymous codes

Three groups of naked word variants were manually added to the `prediction_codes` collection and verified one by one for semantic equivalence:

```python
ALIAS_ADD = {
    "atrial_fibrillation":     {"atrial_fibrillation"},
    "first_degree_av_block":   {"first_degree_av_block", "possible_first_degree_av_delay"},
    "complete_av_block":       {"complete_av_block"},
}
```

**Method to decide whether to merge**: Add the candidate aliases separately, recalculate the four-cell table for the rule baseline (non-LLM, stable behavior, suitable for A/B comparison), and see the TP/FP changes:

| Candidate aliases | Baseline ΔTP | Baseline ΔFP | Conclusion |
|---|---|---|---|
| `bradycardia` → `sinus_bradycardia` | +1 | **+5** | Rejected - refers to the broader "slow heart rate", not an alternative word for the same finding |
| `tachycardia` → `sinus_tachycardia` | +3 | **+4** | Rejected, same as above |
| `possible_lvh_voltage` → `lvh_voltage_criteria` | +0 | **+3** | Rejected, not semantically equivalent |
| `atrial_fibrillation` → `atrial_fibrillation_pattern` family | 0 | 0 | **Adopted** |
| `first_degree_av_block`/`possible_first_degree_av_delay` → `first_degree_av_delay` family | 0 | +1 | **Adopted** (FP increment is negligible and can be further cleaned up by the second step of measurement verification) |
| `complete_av_block` → `complete_av_block_pattern` | 0 | 0 | **Adopted** |

Principle: **Before merging any candidate aliases, do an A/B test on the rule baseline**. If the FP increment is significantly larger than the TP increment, it means that the two codes do not refer to the same thing and cannot be merged.

<a id="42-第二步测量复核剔除假-fp"></a>
### 4.2 Step 2: Measure and review to eliminate false FPs

For each newly generated FP after the first step of merging, throw the code that triggered it back to the `CHECKS` function of the third layer A, and use it to record its own features for re-verification: if the verification is true (confirmed by measurement, but the label is not written), the FP will be directly deleted from the statistics (not counted as TP or FP); if the verification is false or the code cannot be verified, it will be retained as FP.

pseudocode:

```python
for record in records:
    for spec in direct_scope_categories:
        ref_pos = bool(record.ref_codes & spec.ptbxl_refs)
        pred_codes = spec.prediction_codes | ALIAS_ADD.get(spec.key, set())
        matched = record.agent_codes & pred_codes
        pred_pos = bool(matched)
        if ref_pos and pred_pos:
            TP += 1
        elif pred_pos:
            if any(CHECKS.get(c) and CHECKS[c](record.features) is True for c in matched):
                pass  # 剔除，不计入 FP
            else:
                FP += 1
        elif ref_pos:
            FN += 1
```

---

<a id="5-结果总表"></a>
## 5. Summary table of results

<a id="51-micro-f1原始-vs-乐观修正"></a>
### 5.1 Micro-F1 (original vs optimistically corrected)

| Source | Caliber | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Rule Baseline | Original | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
| Regular Baseline | Optimism Modifier | 23 | 28 | 23 | 0.451 | 0.500 | **0.474** |
| DeepSeek | Original | 12 | 17 | 34 | 0.414 | 0.261 | 0.320 |
| DeepSeek | Optimism Correction | 12 | 9 | 34 | 0.571 | 0.261 | **0.358** |
| Qwen | Original | 9 | 19 | 37 | 0.321 | 0.196 | 0.243 |
| Qwen | Optimism Modification | 11 | 9 | 35 | 0.550 | 0.239 | **0.333** |

> After Qwen's optimistic correction, TP changed from 9→11 (not just FP reduction): `first_degree_av_block`/`atrial_fibrillation` After the bare words were merged, 2 records changed from FN to real TP (the original model was correct, but the code string was not recognized).

<a id="52-五种平均口径对照direct-家族"></a>
### 5.2 Comparison of five average calibers (direct family)

| Caliber | Rule baseline | DeepSeek | Qwen | Rule baseline (optimistic) | DeepSeek (optimistic) | Qwen (optimistic) |
|---|---:|---:|---:|---:|---:|---:|
| Micro-F1 | 0.426 | 0.320 | 0.243 | 0.474 | 0.358 | 0.333 |
| Macro-F1 (17 families with equal rights) | 0.452 | 0.188 | 0.154 | 0.471 | 0.211 | 0.197 |
| Macro-F1 (only 16 families with true positive support) | 0.480 | 0.200 | 0.164 | — | — | — |
| Sample-centric F1 (PTB-XL/CAFA formula) | 0.425 | 0.347 | 0.261 | 0.469 | 0.389 | 0.352 |
| Subset Accuracy | 0.280 | **0.320** | 0.300 | — | — | — |
| Average Jaccard | 0.412 | 0.387 | 0.370 | — | — | — |
| Our ad hoc sample macro F1 (0/0→0, conservative) | 0.277 | 0.190 | 0.133 | — | — | — |
| Our ad hoc sample macro F1 (0/0→1, loose) | 0.457 | 0.410 | 0.393 | — | — | — |

Supplementary statistics (original caliber, used to understand the denominator in the above table): **21** of the 50 records have empty real tags on the direct family (no direct family hits the reference tag); among them, the number of records with "prediction is also empty, real is also empty" (trivial match): 9 rule baselines, 11 DeepSeek, and 13 Qwen.

**Key Points**: DeepSeek is consistently better than Qwen in all calibers. The leading amplitude of the baseline relative to the two models is very sensitive to the caliber - the baseline under Macro-F1 is about 2.2~3 times that of the two models (both models have 0% recall for the 6 conduction block families, and are equally amplified), and is compressed to 1.2~1.6 times under Micro/Sample-centric (these families have small samples and are diluted on average); under Subset Accuracy DeepSeek On the contrary, it exceeds the baseline (0.320 vs 0.280), because the rule baseline is more "talkative" (especially the LVH family has a large number of false positives). When looking at "whether the family set is completely hit" record by record, it is easy to miss the full score because of one or two more words.

<a id="53-测量复核结果"></a>
### 5.3 Measurement review results

| Model | Total number of confirmed diagnoses | not-checkable | unlabelled-but-correct | confirmed-and-labelled | Number of verifiable diagnoses | Number of verified errors |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 59 | 34 (57.6%) | 23 (39.0%) | 2 (3.4%) | 25 | **0 (0.0%)** |
| Qwen | 74 | 44 (59.5%) | 28 (37.8%) | 2 (2.7%) | 30 | **0 (0.0%)** |

<a id="54-分诊断家族召回率13-个有信号的-direct-家族原始口径"></a>
### 5.4-point diagnostic family recall (13 signaled direct families, original caliber)

| family | n (number of true positives) | rule baseline | DeepSeek | Qwen |
|---|---:|---:|---:|---:|
| Sinus bradycardia | 2 | 50% | 50% | 0% |
| Sinus tachycardia | 4 | 25% | 25% | 25% |
| Atrial Fibrillation | 4 | 75% | 50% | 0%* |
| Atrial premature beats | 2 | 100% | 0% | 0% |
| Premature ventricular contractions | 5 | 60% | 60% | 60% |
| Right bundle branch block RBBB | 6 | 33% | 0% | 0% |
| Left bundle branch block LBBB | 2 | 50% | 0% | 0% |
| Left Anterior Branch Block LAFB | 2 | 100% | 0% | 0% |
| Left posterior fascicular block LPFB | 1 | 100% | 0% | 0% |
| Nonspecific intraventricular conduction delay | 4 | 25% | 0% | 0% |
| Ventricular Preexcitation WPW | 1 | 100% | 100%* | 100% |
| Left ventricular hypertrophy voltage standard | 4 | 50% | 100% | 100% |
| Right Ventricular Hypertrophy RVH | 3 | 100% | 0% | 0% |

0% of `*` atrial fibrillation Qwen is accounting noise (see §6.2); WPW DeepSeek has a precision rate of only 0.5 under 100% recall (see §6.1 case).

**Structural conclusion**: RBBB/LBBB/LAFB/LPFB/non-specific IVCD/RVH/premature atrial contractions, a total of 7 families, a total of 20/46 (about 43%) true positives, the recall rate of both models is 0% - it is not a misjudgment, it is because these diagnostic codes do not appear at all in the current agent output. No accounting/labeling fixes can change this part, the pipeline itself needs to be fixed (see §7 known bug list).

---

<a id="6-案例分析逐条核实"></a>
## 6. Case analysis (verify item by item)

<a id="61-09275hr-deepseek-独有-wpw-假阳性"></a>
### 6.1 09275_hr — DeepSeek Exclusive WPW False Positive

- Reference label: `CRBBB|LAFB|RVH|SR` (complete right bundle branch block + left anterior fascicular block + right ventricular hypertrophy, bifascicular block), without WPW.
- Rule baseline: `lafb_pattern|rvh_pattern|secondary_t_wave_abnormality`, also without WPW.
- DeepSeek Unique diagnosis: `ventricular_preexcitation_pattern`, HIGH confidence, references `ev:/rhythm_inputs/preexcitation/short_pr_interval` and `delta_lead_count=3`, global `pr_ms=141`.
- Record itself `metadata.rhythm_analysis.rule_summary.preexcitation` (`ptbxl_09000_ecgfeat/features/09275_hr_features.json`): `wpw_pattern: false`, `short_pr_interval: false` (141ms is in the normal range 120–200ms, not a short PR), `delta_lead_count: 3` < `delta_threshold: 8`. Substitute `feature_extraction/ecgfeat/clinical_rules/preexcitation.py::evaluate_preexcitation()` to get `status="not_matched"`.
- **The program's own certainty criterion has been calculated as false, and the field is in the evidence path referenced by the model. The model still gives the opposite conclusion and HIGH confidence. **
- Control: Qwen gave `left_axis_deviation` on the same record and was not misled; both models confirmed correctly on the real WPW case `09699_hr` (accuracy rate 1.0/1.0).

<a id="62-09114hr-09592hr-09913hr-09570hr-09136hr-09550hr-诊断代码同义词分裂"></a>
### 6.2 09114_hr / 09592_hr / 09913_hr / 09570_hr / 09136_hr / 09550_hr - Diagnostic code synonym split

Only one variant of `prediction_codes` from some families in `evaluate_target_ecgfeat_diagnosis.py` is registered. If the model uses an unregistered synonymous variant, it will be recorded as a missed detection:

| Record | Model | Actual output code | Set of codes identified by scoring | Consequences |
|---|---|---|---|---|
| 09114_hr | Qwen | `atrial_fibrillation` | `atrial_fibrillation_pattern`/`probable_*` | Atrial fibrillation determined correctly but counted as 0 match |
| 09592_hr | Qwen | `atrial_fibrillation` | Same as above | Same as above, directly lowering the AFIB family recall |
| 09913_hr | Qwen | `first_degree_av_block` | `first_degree_av_delay` | The first-degree room delay judgment was correct but not scored |
| 09570_hr | DeepSeek | `first_degree_av_block` | `first_degree_av_delay` | Same as above |
| 09136_hr | DeepSeek | `t_wave_abnormality` | `primary_/secondary_t_wave_abnormality` | T wave abnormality not scored (broad family, does not affect headline F1) |
| 09550_hr | Qwen | `possible_lvh_voltage` | `lvh_voltage_criteria` | LVH prompt not scored |

This sample: Qwen hit 4 times, DeepSeek hit 2 times - both sides were affected, Qwen hit more this time, 2 of which directly cut off the recall rate that the AFIB family should have.

<a id="63-09114hr-两模型都成功剪除规则引擎历史噪声"></a>
### 6.3 09114_hr — Both models successfully trimmed the rule engine historical noise

The rule baseline gives 9 codes, among which `prior_infarct_q_wave_pattern` + `rvh_pattern` + `lvh_voltage_criteria` appear at the same time (the previously recorded combination of "high amplitude QRS inducing pathological Q wave misjudgment"). Both DeepSeek and Qwen cut the 9 codes to only 2 (atrial fibrillation + LVH), which are highly consistent with the reference label `AFIB|INVT|NDT|PVC|STD_`. The sample size is limited (n=1) and does not represent systematic repair of defects.

<a id="64-09570hr-deepseek-识别陈旧心梗-q-波qwen-漏检"></a>
### 6.4 09570_hr — DeepSeek Identify Q wave of old myocardial infarction, Qwen missed detection

The reference label contains `ASMI|ILMI` (anteroseptal + inferior lateral wall old myocardial infarction). DeepSeek outputs `prior_infarct_q_wave_pattern`, which matches the label directly; Qwen does not give this code.

---

<a id="7-已知缺陷复查状态"></a>
## 7. Known defect review status

| Known issues | Status | This sample evidence |
|---|---|---|
| Total missed detections of conduction block families (RBBB/LBBB/LAFB/LPFB/IVCD/RVH) | Still exists | All 6 families in both models have 0% recall, in contrast to the regular baseline of 25%~100%, and the nature is consistent with previous positioning |
| Ischemia/myocardial infarction screening recall rate is extremely low | Still exists | DeepSeek ischemia 0/18, myocardial infarction 1/16; Qwen ischemia 1/18, myocardial infarction 0/16; regular baseline 7/18, 4/16 |
| Diagnosis code synonym splitting | still exists | 6 new specific instances confirmed this time (§6.2) |
| LVH gate coverage is narrower than the old rules engine | Partial improvement | F1 increased from 0.222 to 0.533, the two models are completely consistent (programmed), and the 7 FPs have not been verified one by one |
| Q wave false positive/high amplitude QRS induces pathological Q wave misjudgment | No new occurrences were found this time | 09114_hr Both models actively cut out this combination of noise, and the sample is limited (n=1) |
| WPW/Pre-excitation path lacks model - program consistency check | **New discovery** | 09275_hr: DeepSeek still gives HIGH confidence when the program criterion is clearly false WPW |
| sinus_rhythm gate stagnation/AF model-override | This sample is not triggered | No relevant evidence is observed, the sample size is small, it does not mean that it has been repaired |

---

<a id="8-运行效率对比"></a>
## 8. Comparison of operating efficiency

| Indicators | DeepSeek | Qwen |
|---|---:|---:|
| Median time / P90 | 22.058s / 28.316s | 87.734s / 109.671s |
| Total tool calls / record average | 495 / 9.9 | 507 / 10.14 |
| Token: prompt / completion / cached | 617,993 / 60,756 / 286,336 | 427,957 / 71,790 / 0 |
| Verification pass rate / revision round | 50/50, 0 | 50/50, 0 |
| record_grade（Q0/Q1） | 46/4 | 46/4 |
| Record-by-record relative to baseline: improved/unchanged/worsened | 23/21/6 | 22/23/5 |

DeepSeek is about 4 times faster and hits a large number of prompt caches (no cache hits on the Qwen side).

---

<a id="9-跨论文数字对比的坑重要避免误用"></a>
## 9. Pitfalls in comparing numbers across papers (important, avoid misuse)

PTB-XL F1 reported in published ECG-LLM/agent papers (e.g., CARE-ECG, ECG-Chat), **most likely contains NORM (normal)/ "Normal" labels such as SR (sinus rhythm) are used as scoreable categories**, and the recall rate of such labels is naturally high, which will significantly increase the aggregation index:

- ECG-Chat Original words of the paper (Fig.4 description): "Some common labels, such as 'Normal (NORM)', 'Sinus rhythm (SR)', have high F1 scores... Many labels have an F1 score of 0."
- ECG-Chat reported CE-Disease (including NORM) F1 = 22.33% in his paper; CARE-ECG used the same "ECG-Chat" model as the baseline and the same PTB-XL task in the paper, but reported F1 = 0.70 - Same model, same data set, different scoring protocols resulting in numbers that are more than 3 times different.
- Our own `evaluate_target_ecgfeat_diagnosis.CATEGORIES` **Without NORM/Normal Category**, a model that says "everything is normal" will never get TP, only points will be deducted for missing real anomalies.

**Conclusion**: All the numbers in §5 of this document are only horizontally comparable "between our own three sources (baseline/DeepSeek/Qwen)"; do not directly compare them with the F1/Accuracy numbers of external papers unless you first confirm whether the other party's scoring category set contains NORM/SR, and which average caliber of macro/micro/sample-centric is used.

---

<a id="附录-a本次涉及的-17-个-direct-scope-家族完整列表"></a>
## Appendix A: Complete list of 17 direct-scope families involved in this article

```
sinus_bradycardia, sinus_tachycardia, atrial_fibrillation, atrial_flutter,
premature_atrial_complexes, premature_ventricular_complexes,
first_degree_av_block, complete_av_block,
right_bundle_branch_block, left_bundle_branch_block,
left_anterior_fascicular_block, left_posterior_fascicular_block,
nonspecific_ivcd, ventricular_preexcitation,
left_ventricular_hypertrophy, right_ventricular_hypertrophy,
prolonged_qt
```

(`unspecified_ectopy_pattern`, `ambiguous_left_conduction_family`, `atrial_abnormality`, `low_qrs_voltage`, `t_wave_abnormality` are `broad` scope; `ischemia_or_st_abnormality`, `infarction_or_q_wave` are `screening` scope; are not included in the headline metric in §5 of this document).

<a id="附录-b复现命令"></a>
## Appendix B: Reproduction Command

```bash
# 测量复核（§3）
python3 score_measurement_verifiable.py \
  qwen_agent_output/deepseek_v38_diverse50 \
  qwen_agent_output/qwen_v38_diverse50

# 第一/二层四格表与各平均口径、第三层B乐观修正：
# 均为本次分析中现写的一次性脚本，未入库；核心逻辑见本文档 §1、§2、§4 的伪代码，
# 依赖 evaluate_target_ecgfeat_diagnosis.CATEGORIES 与 score_measurement_verifiable.CHECKS，
# 输入取自各 run 目录下的 record_results.csv 与 features/*.json。
```

<a id="附录-c相关项目记忆"></a>
## Appendix C: Related Project Memory

- `project_v38_diverse50_deepseek_vs_qwen.md`
- `project_deepseek_wpw_gate_override.md`
- `project_diagnosis_catalog_synonym_split.md`
- `project_measurement_verifiable_scoring.md`
- `project_ecg_llm_literature_f1_not_comparable.md`
- `project_conduction_block_total_miss.md`

<!-- i18n-nav -->
[中文](deepseek_vs_qwen_v38_diverse50_results_tables.md) | [English](deepseek_vs_qwen_v38_diverse50_results_tables.en.md) | [日本語](deepseek_vs_qwen_v38_diverse50_results_tables.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="deepseek-vs-qwenv38-diverse50结果表格"></a>
# DeepSeek vs Qwen (v38 diverse50) results table

Pure table version, no text description; formula definitions and case-by-case analysis can be found in [`deepseek_vs_qwen_v38_diverse50_metrics.md`](deepseek_vs_qwen_v38_diverse50_metrics.en.md); Excel multi-sheet version of the same data can be found in [`deepseek_vs_qwen_v38_diverse50_results.xlsx`](deepseek_vs_qwen_v38_diverse50_results.xlsx).

<a id="1-核心指标总览"></a>
## 1. Overview of core indicators

| Indicators | Rule Baseline | DeepSeek | Qwen | Remarks |
|---|---|---|---|---|
| Surface Label-F1 (direct family, micro) | 0.426 | 0.320 | 0.243 | All lower than the baseline, containing a large amount of label missing noise |
| Optimistic caliber F1 (synonym merging + measurement to eliminate false FP) | 0.474 | 0.358 | 0.333 | Conservative estimate after two-step correction |
| Measurement of verifiable diagnoses / errors | — | 25 / 0 | 30 / 0 | 0 arithmetic errors for both models |
| Measure the proportion of zero errors in verifiable diagnoses | — | 100% | 100% | |
| The proportion of false positives with "missing tags but actually correct" | — | 92% (23/25) | 93% (28/30) | |
| Median single line time | — | 22.1s | 87.7s | DeepSeek is about 4 times faster |
| P90 single line time | — | 28.3s | 109.7s | |
| Total tool calls / record average | — | 495 / 9.9 | 507 / 10.1 | Similar path lengths |
| Prompt / Completion / Cached tokens | — | 617,993 / 60,756 / 286,336 | 427,957 / 71,790 / 0 | DeepSeek hit large cache |
| Verification pass rate | — | 50/50 | 50/50 | 0 execution errors, 0 revision rounds |
| record_grade distribution (Q0/Q1) | 46/4 | 46/4 | 46/4 | Data quality grade is exactly the same |
| Record-by-record relative to baseline: improved/unchanged/worsened | — | 23/21/6 | 22/23/5 | |

<a id="2-五种平均口径对照direct-家族"></a>
## 2. Five average caliber comparisons (direct family)

| Caliber | Rule Baseline (Original) | DeepSeek (Original) | Qwen (Original) | Rule Baseline (Optimistic) | DeepSeek (Optimistic) | Qwen (Optimistic) | Definition Source |
|---|---|---|---|---|---|---|---|
| Micro-F1 (calculated once after summarizing the four-grid table) | 0.426 | 0.320 | 0.243 | 0.474 | 0.358 | 0.333 | The default caliber of this report |
| Macro-F1 (17 families with full equal rights) | 0.452 | 0.188 | 0.154 | 0.471 | 0.211 | 0.197 | PTB-XL paper term-centric ideas |
| Macro-F1 (16 families with true positive support) | 0.480 | 0.200 | 0.164 | — | — | — | Same as above, excluding 0 supported families |
| Sample-centric F1 (PTB-XL/CAFA formula) | 0.425 | 0.347 | 0.261 | 0.469 | 0.389 | 0.352 | PTB-XL paper §II.C, derived from CAFA Challenge |
| Subset Accuracy (family set exact match) | 0.280 | 0.320 | 0.300 | — | — | — | Multi-label classification classic indicators |
| Average Jaccard similarity | 0.412 | 0.387 | 0.370 | — | — | — | Classic indicator for multi-label classification |
| Sample macro F1 (0/0→0, conservative) | 0.277 | 0.190 | 0.133 | — | — | — | This ad hoc, sklearn-style division-by-zero processing |
| Sample macro F1 (0/0→1, loose) | 0.457 | 0.410 | 0.393 | — | — | — | This ad hoc, trivial-match convention |

<a id="3-micro-f1-底层混淆矩阵明细"></a>
## 3. Micro-F1 underlying confusion matrix details

| Source | Caliber | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Rule Baseline | Original | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
| Regular Baseline | Optimism Modifier | 23 | 28 | 23 | 0.451 | 0.500 | 0.474 |
| DeepSeek | Original | 12 | 17 | 34 | 0.414 | 0.261 | 0.320 |
| DeepSeek | Optimistic Correction | 12 | 9 | 34 | 0.571 | 0.261 | 0.358 |
| Qwen | Original | 9 | 19 | 37 | 0.321 | 0.196 | 0.243 |
| Qwen | Optimism Modification | 11 | 9 | 35 | 0.550 | 0.239 | 0.333 |

<a id="4-测量复核结果"></a>
## 4. Measurement review results

| Model | Total number of confirmed diagnoses | not-checkable | unlabelled-but-correct | confirmed-and-labelled | Number of verifiable diagnoses | Number of verification errors | Error rate |
|---|---:|---|---|---|---:|---:|---:|
| DeepSeek | 59 | 34 (57.6%) | 23 (39.0%) | 2 (3.4%) | 25 | 0 | 0.0% |
| Qwen | 74 | 44 (59.5%) | 28 (37.8%) | 2 (2.7%) | 30 | 0 | 0.0% |

<a id="5-分诊断家族召回率13个有信号的-direct-家族原始口径"></a>
## 5. Diagnostic family recall rate (13 direct families with signals, original caliber)

| Diagnostic family | True positive number n | Rule baseline | DeepSeek | Qwen | Remarks |
|---|---:|---|---|---|---|
| Sinus bradycardia | 2 | 50% | 50% | 0% | |
| Sinus tachycardia | 4 | 25% | 25% | 25% | |
| Atrial fibrillation | 4 | 75% | 50% | 0% | Qwen is accounting noise, see §7 |
| Atrial premature beats | 2 | 100% | 0% | 0% | Total missed detections in both models |
| Premature ventricular contractions | 5 | 60% | 60% | 60% | |
| Right bundle branch block RBBB | 6 | 33% | 0% | 0% | Total missed detections in both models |
| Left bundle branch block LBBB | 2 | 50% | 0% | 0% | Total missed detections in both models |
| Left anterior fascicular block LAFB | 2 | 100% | 0% | 0% | Total missed detections in both models |
| Left posterior branch block LPFB | 1 | 100% | 0% | 0% | Total missed detections in both models |
| Non-specific indoor conduction delay | 4 | 25% | 0% | 0% | Total missed detections for both models |
| Ventricular preexcitation WPW | 1 | 100% | 100% | 100% | DeepSeek accuracy is only 0.5, see §6 |
| Left ventricular hypertrophy voltage standard | 4 | 50% | 100% | 100% | The two models are completely consistent (programmed) |
| Right ventricular hypertrophy RVH | 3 | 100% | 0% | 0% | Total missed detections in both models |

<a id="6-关键分歧记录案例分析"></a>
## 6. Case analysis of key disagreement records

| Record ID | Case Type | Reference Tag | DeepSeek Output | Qwen Output | Conclusion |
|---|---|---|---|---|---|
| 09275_hr | DeepSeek unique WPW false positive (new discovery) | CRBBB\|LAFB\|RVH\|SR | ventricular_preexcitation_pattern (HIGH) | left_axis_deviation | The program's own wpw_pattern criterion = false, and in the model reference evidence path, the model still gives the opposite conclusion |
| 09699_hr | Control: real WPW cases, both models are correct | Contains WPW | ventricular_preexcitation_pattern ✓ | ventricular_preexcitation_pattern ✓ | Prove that DeepSeek is not a systematically biased false positive WPW |
| 09114_hr | Both models cut out the rule engine historical noise | AFIB\|INVT\|NDT\|PVC\|STD_ | AF+LVH (cut from baseline 9 code to 2) | AF+LVH (also cut to 2) | Q-wave/RVH false alarm combinations that do not follow the baseline |
| 09570_hr | DeepSeek Advantage Case | ASMI\|ILMI\|IRBBB\|SVARR\|ABQRS | Contains prior_infarct_q_wave_pattern ✓ | The code is not given ✗ | DeepSeek correctly identifies old myocardial infarction Q wave, Qwen misses detection |

<a id="7-诊断代码同义词分裂-确认实例"></a>
## 7. Diagnostic code synonym splitting - confirmed example

| Record ID | Model | Actual output code | Set of codes identified by the score | Consequences |
|---|---|---|---|---|
| 09114_hr | Qwen | atrial_fibrillation | atrial_fibrillation_pattern / probable_* | Atrial fibrillation is correctly determined but counted as 0 matches |
| 09592_hr | Qwen | atrial_fibrillation | atrial_fibrillation_pattern / probable_* | Same as above, directly lowering the AFIB family recall |
| 09913_hr | Qwen | first_degree_av_block | first_degree_av_delay | The first degree room delay was judged correctly but not scored |
| 09570_hr | DeepSeek | first_degree_av_block | first_degree_av_delay | Same as above |
| 09136_hr | DeepSeek | t_wave_abnormality | primary_/secondary_t_wave_abnormality | T wave abnormality is not scored (broad family, does not affect headline F1) |
| 09550_hr | Qwen | possible_lvh_voltage | lvh_voltage_criteria | LVH prompt not scored |

This sample: Qwen hit 4 times, DeepSeek hit 2 times - both sides were affected, Qwen hit more this time, 2 of which directly cut off the recall rate that the AFIB family should have.

<a id="8-已知缺陷复查状态"></a>
## 8. Known defect review status

| Known issues | Status | This sample evidence |
|---|---|---|
| Total missed detections of conduction block families (RBBB/LBBB/LAFB/LPFB/IVCD/RVH) | 🔴 Still exists | All 6 families in both models have 0% recall, which is in contrast to the regular baseline of 25% to 100%, and the nature is consistent with previous positioning |
| The recall rate of ischemia/myocardial infarction screening is extremely low | 🔴 Still exists | DeepSeek ischemia 0/18, myocardial infarction 1/16; Qwen ischemia 1/18, myocardial infarction 0/16; regular baseline 7/18, 4/16 |
| Diagnosis code synonym split | 🔴 still exists | 6 new specific instances confirmed this time (§7) |
| LVH gate coverage is narrower than the old rules engine | 🟡 Partial improvement | F1 increased from 0.222 to 0.533, the two models are completely consistent (programmed), and the 7 FPs have not been verified one by one |
| Q wave false positive/high amplitude QRS induces pathological Q wave misjudgment | 🟢 No new occurrences were found this time | 09114_hr Both models actively cut out this combination of noise, and the sample is limited (n=1) |
| WPW/pre-excitation path lacks model - program consistency check | 🔴 New discovery | 09275_hr: DeepSeek still gives HIGH confidence level WPW when the program criterion is clearly false |
| sinus_rhythm gate stagnation/AF model-override | ⚪ This sample was not triggered | No relevant evidence was observed, and the sample size is small, which does not mean that it has been repaired |

<a id="9-运行效率对比"></a>
## 9. Comparison of operating efficiency

| Indicators | DeepSeek | Qwen |
|---|---|---|
| Median time / P90 | 22.058s / 28.316s | 87.734s / 109.671s |
| Total tool calls / record average | 495 / 9.9 | 507 / 10.14 |
| Token: prompt / completion / cached | 617,993 / 60,756 / 286,336 | 427,957 / 71,790 / 0 |
| Verification pass rate / revision round | 50/50, 0 | 50/50, 0 |
| record_grade（Q0/Q1） | 46/4 | 46/4 |
| Record-by-record relative to baseline: improved/unchanged/worsened | 23/21/6 | 22/23/5 |

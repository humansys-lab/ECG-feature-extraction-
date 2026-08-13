<!-- i18n-nav -->
[中文](ludb_full_compare_result_analysis.md) | [English](ludb_full_compare_result_analysis.en.md) | [日本語](ludb_full_compare_result_analysis.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ludb-full-compare-结果分析与改进建议"></a>
# LUDB Full Compare Result Analysis and Improvement Suggestions

Analysis object: `/home/chtmedgemma/projects/ecg_gemma/ludb_full_compare`

Analysis time: 2026-07-05

<a id="1-数据概况"></a>
## 1. Data overview

- `summary.csv` 200 records in total.
- Each record contains 12 leads representing measurement contrast, global interval/axis contrast, beat/missing/mass counts.
- The per-lead comparison text parsed to a total of 16,800 lines of indicators, including `PR/QRS/QT/QTcB/R/T/ST-J`.

Overall, the R peak/HR basic capabilities of the current algorithm are relatively stable, and the main errors are concentrated in:

1. PR/P wave boundary and PR availability.
2. QRS onset/offset, especially wide QRS, pacing, terminal slur.
3. T end/QT rescue or low-support path.
4. QT dispersion is significantly suppressed.
5. There is an evaluation caliber problem with axis and per-lead/global comparison.

<a id="2-全局指标表现"></a>
## 2. Global indicator performance

| Indicators | Number of valid records | Bias | MAE | Median abs | P75 abs | Max abs | Main conclusions |
|---|---:|---:|---:|---:|---:|---:|---|
| HR | 200 | -0.21 bpm | 0.66 bpm | 0.22 bpm | 0.59 bpm | 29.84 bpm | Most are good, a few have serious missed detection/half frequency problems |
| PR | 151 | -11.77 ms | 16.85 ms | 12 ms | 22.75 ms | 88 ms | Systematically short, and 48 algorithm PRs are missing |
| QRS | 200 | +5.73 ms | 9.88 ms | 8.75 ms | 14 ms | 46 ms | The overall width is wider, but the outlier is wider or narrower |
| QT | 200 | -1.09 ms | 13.58 ms | 8 ms | 15 ms | 91.5 ms | Median performance is acceptable, rescue/low-support outlier is obvious |
| QTcB | 200 | -1.88 ms | 14.99 ms | 8.78 ms | 15.97 ms | 117.74 ms | Affected by both QT and HR outlier |
| QTcF | 200 | -1.64 ms | 14.41 ms | 8.35 ms | 15.48 ms | 100.83 ms | Same as above |
| P axis | 172 | -10.12 deg | 24.25 deg | 8.88 deg | 21.48 deg | 228.83 deg | Part of it is a real error, part of it may be an angle wrap comparison problem |
| QRS axis | 199 | +0.78 deg | 12.90 deg | 8.21 deg | 16.48 deg | 88.37 deg | Overall acceptable, few leads/low amplitude abnormalities |
| T axis | 150 | -1.13 deg | 16.60 deg | 6.73 deg | 20.50 deg | 352.21 deg | 50 algorithms T axis is missing; max is mostly a 360-degree surround problem |
| QT dispersion | 190 | -25.35 ms | 27.83 ms | 20.5 ms | 42 ms | 112 ms | Clearly underestimated and one of the most stable systemic problems currently |

Number of records exceeding threshold:

| Metrics | Thresholds | Beyond Thresholds |
|---|---:|---:|
| HR | > 2 bpm | 9/200 |
| PR | > 20 ms | 44/151 |
| QRS | > 20 ms | 17/200 |
| QT | > 30 ms | 25/200 |
| QTcB | > 30 ms | 28/200 |
| QTcF | > 30 ms | 26/200 |
| P axis | > 30 deg | 36/172 |
| QRS axis | > 30 deg | 21/199 |
| T axis | > 30 deg | 15/150 |
| QT dispersion | > 40 ms | 49/190 |

<a id="3-beat-检测与质量计数"></a>
## 3. Beat detection and quality counting

Algorithms usually have more beats than GT:

- Average `algorithm_beats - ground_truth_beats = +1.645`.
- The median is `+2`.
- 138/200 records are exactly 2 beats more than GT.
- This is most likely due to the different beat inclusion rules at the beginning and end: the algorithm also includes beats at the edge of the 10-second record, and the GT report/summary may only count complete measurable beats.

This usually has little impact on HR, because HR is mostly determined by the RR median/internal beat; but it will contaminate:

- `algorithm_total` vs `ground_truth_total`
- missing count
- unreliable count
- per-record quality percentage

It is recommended to add beat matching/window alignment to the evaluation script first:

- Global HR can continue to be evaluated independently.
- beat-level missing/unreliable should only compare beats that match GT.
- Incomplete beats at the beginning and end are either excluded from both sides or listed separately as edge beats.

<a id="4-p-波与-pr需要优先改"></a>
## 4. P wave and PR: need to be modified first

Phenomenon:

- Global PR bias = `-11.77 ms`, MAE = `16.85 ms`.
- 44/151 PRs have an error of more than 20 ms.
- per-lead PR is more obvious: bias = `-21.72 ms`, MAE = `28.31 ms`.
- The leads with the largest PR errors are concentrated in V3/V4/V5, III, aVR, etc., indicating that the onset of low-amplitude/complex P waves is more likely to be delayed by the algorithm.
- There are 48 missing PRs and 25 GTs in the algorithm. In other words, although the algorithm has less beat-level `P missing` than GT, its PR availability is worse, mainly because P is marked as unreliable or atrial measurement is suppressed.

Typical outlier:

- Record 39: PR `-88 ms`
- Record 57: PR `+86 ms`
- Record 130: PR `-84.5 ms`
- Record 120: PR `-61 ms`

Root cause judgment:

1. P onset is mostly late, resulting in systematically short PR.
2. When long PR/low amplitude P/P is confused with T or flutter-like residual, the current reliability rules easily make the global PR become N/A.
3. Multi-lead P anchor does not protect PR summarization enough, especially when the P wave in some leads is clear but other leads are unreliable, the global PR will be overly suppressed.

Suggested improvements:

1. Enhance the early rescue of P onset: For beats that already have stable P peak, use multi-lead atrial anchor to look forward for consistent onset, instead of relying only on single-lead tangent.
2. PR summary does not require too many leads to be reliable at the same time; robust PR with limb-lead priority and V1 assistance can be used.
3. Relax the P lookback/early-onset restrictions for long PR scenarios, and change the trigger condition of long-PR rescue from single-lead to multi-lead consistency.
4. The evaluation is divided into three levels: `P detected`, `P reliable`, and `PR globally reportable`.

<a id="5-qrs-onsetoffset宽-qrs起搏和-terminal-slur-是主要风险"></a>
## 5. QRS onset/offset: wide QRS, pacing and terminal slur are major risks

Global QRS:

- Bias = `+5.73 ms`
- MAE = `9.88 ms`
- 17/200 errors exceeded 20 ms.

Typical outlier:

- Record 108: QRS `-46 ms`
- Record 90: QRS `+44.5 ms`
- Record 99: QRS `+35.5 ms`
- Record 5: QRS `+32 ms`
- Record 8: QRS `+28.9 ms`

Visual confirmation:

- Record 108: Algorithm QRS offset is significantly earlier than GT in some leads, cutting off terminal QRS/slur, resulting in ST-J deviation, for example, V1 ST-J diff reaches `+0.59 mV`.
- Record 8: QRS detector confidence is extremely low, with a median of about `0.049`, and suspected pacing spike metadata; the global route enters `wide_qrs_jt`, and QRS/QT is affected.
- Record 74: Paced/low-amplitude QRS scene, the algorithm only detects 5 beats, GT is 9, and HR changes from about 60 bpm to about 30 bpm.

Root cause judgment:

1. QRS terminal boundary is sensitive to wide QRS, low-amplitude terminal slur, and ST/T junction.
2. Under pacing and low amplitude QRS, when R detector confidence is very low, subsequent measurements are still entered, which may easily produce half-frequency or misgrouping.
3. The QRS offset error will directly pollute the ST-J, JT, QT path, wide_qrs_jt paths.

Suggested improvements:

1. Add multi-lead fallback for records with low QRS detector confidence: As long as multiple leads have synchronized deflection near the expected RR, beats should not be missed.
2. Added spike-anchored capture search to the pacing scene: Find and capture QRS from the fixed physiological window after the pacing spike, instead of just relying on ordinary detectors.
3. QRS offset consensus needs to introduce terminal slur/ST-J guard: If `ST40/ST80` is very different from point J after offset, you should backtrack to determine whether the offset is too early or too late.
4. The wide QRS route should not only affect QT/JT, but also reversely constrain the QRS offset credibility.

<a id="6-t-endqt主路径可用rescue-和-low-support-路径需要收紧"></a>
## 6. T end/QT: The main path is available, the rescue and low-support paths need to be tightened

Overall QT:

- Bias = `-1.09 ms`
- MAE = `13.58 ms`
- 25/200 errors exceed 30 ms.

Grouped by QT source:

| QT source | Number of records | Bias | MAE | P75 abs | Max abs |
|---|---:|---:|---:|---:|---:|
| reliable_lead_median | 125 | +5.1 ms | 9.6 ms | 11.5 ms | 63 ms |
| low_qt_support_fallback | 36 | -5.9 ms | 20.9 ms | 28.4 ms | 81 ms |
| reliable_raw_qt_cluster_rescue | 25 | -13.8 ms | 16.1 ms | 10.5 ms | 91.5 ms |
| raw_qt_core_rescue | 8 | -27.3 ms | 29.3 ms | 47 ms | 58 ms |
| wide_qrs_low_support_qt_fallback | 4 | -20.6 ms | 28.6 ms | 30.6 ms | 73 ms |

The conclusion is clear:

- `reliable_lead_median` The main path performs best.
- `low_qt_support` and various rescue paths are significantly worse and overall shorter.
- Large QT outliers mostly come with `st_t_confusion`, `t_end_fallback`, `beat_unreliable` or rescue source.

Typical outlier:

- Record 57: QT `-91.5 ms`. It can be seen from the figure that T offset is truncated early by the algorithm.
- Record 73: QT `-81 ms`.
- Record 125: QT `-73 ms`.
- Record 81: QT `-72.5 ms`.
- Record 133: QT `-68.5 ms`.
- Record 24: QT `+63 ms`.

Suggested improvements:

1. Add "do not end prematurely" constraints in the rescue/low-support QT path, such as the shortest tail after T peak, T area tail, and cross-lead late-tail support.
2. When `st_t_confusion` appears, do not directly regard the early trough as the T end; first look for the subsequent homopolar or biphasic T wave tail.
3. For the beat of `t_end_fallback`, reduce the weight of participating in the global QT, instead of entering the global median as long as it passes the basic range.
4. In the wide QRS/JT path, QT cannot only rely on the rationality of JT, but also checks whether the QRS offset is reliable.

<a id="7-qt-dispersion当前明显低估"></a>
## 7. QT dispersion: Currently obviously underestimated

QT dispersion：

- Bias = `-25.35 ms`
- MAE = `27.83 ms`
- 49/190 items exceeded 40 ms.

This is a very stable systemic problem. The reason is not a single T end outlier, but the current representative lead/consensus strategy tends to pull the QT of multiple leads to similar values, compressing the differences between leads.

Typical outlier:

- Record 132: `-112 ms`
- Record 29: `-101 ms`
- Record 21: `-99 ms`
- Record 84: `-97 ms`
- Record 52: `-94 ms`

Suggested improvements:

1. QT dispersion should be calculated using independent per-lead QT instead of global `qt_consensus_ms` after filling in each lead.
2. Keep three sets of values ​​for each lead: independent QT, consensus QT, and used-for-global-QT.
3. When calculating dispersion, only filter obviously invalid leads, and do not force all leads to be aligned to the global T end.
4. Clearly write the dispersion source in the report: `independent_lead_qt` or `consensus_qt`.

<a id="8-axis算法问题和评估问题混在一起"></a>
## 8. Axis: Algorithm problems and evaluation problems are mixed

Axis indicators need to be interpreted with caution:

- P axis MAE = `24.25 deg`, max diff = `228.83 deg`.
- T axis MAE = `16.60 deg`, max diff = `352.21 deg`.
- These maxima most likely include angle wraparound issues, e.g. direct subtraction of 179 deg from -179 deg gives 358 deg, but the clinical angle difference should be 2 deg.

At the same time, the algorithm T axis is missing 50 items, while the GT T axis is not missing. This is an algorithm usability issue.

Suggested improvements:

1. The comparison script must use circular angular difference when calculating axis diff: `min(abs(a-b), 360-abs(a-b))`.
2. T axis aggregation should not be overly reliant on all leads T being reliable; as long as the main limb leads have available T amplitude, a low confidence but reportable T axis should be given.
3. If P axis is large outlier, it is necessary to reorder after correcting the circular diff, and then determine whether the true P wave direction is wrong or the comparative aperture is wrong.

## 9. ST-J/T/R amplitude

The overall performance of per-lead amplitude is better:

| Indicators | Bias | MAE | P75 abs | Max abs |
|---|---:|---:|---:|---:|
| R amplitude | -0.005 mV | 0.023 mV | 0.03 mV | 0.51 mV |
| T amplitude | +0.003 mV | 0.035 mV | 0.03 mV | 0.81 mV |
| ST-J | -0.004 mV | 0.029 mV | 0.03 mV | 0.68 mV |

The median performance of these metrics is OK, but significantly outlier:

- Record 116: Multiple limb leads exhibiting large deviations in T/ST-J polarity or baseline levels.
- Record 108: ST-J large deviation from QRS offset truncation.
- Record 35/116: Large T amplitude outlier is mostly caused by T polarity/selection error.

Suggested improvements:

1. ST-J outlier gives priority to modifying from QRS offset and baseline instead of adjusting ST-J sampling alone.
2. T amplitude outlier should be modified together with T polarity/T candidate selection.
3. For beats whose ST-J exceeds the physiological range and QRS terminal confidence is low, the weight will be automatically downgraded or marked as measurement unreliable.

<a id="10-报告比较脚本本身需要改"></a>
## 10. The report/comparison script itself needs to be modified

The current comparison text has several problems with mixed calibers, which will affect manual judgment:

1. Many QRS/QT values in the per-lead table are exactly the same in lead 12, indicating that it may be comparing consensus/global values rather than truly independent per-lead values.
2. Some records global measurement and per-lead comparison are inconsistent:
   - Record 8: global QRS is `144.4 ms`, and QRS in the per-lead table is `204 ms`.
   - Record 109: The global QT is `305 ms`, and the QT in the per-lead table is `492 ms`.
3. Axis diff does not perform circular normalization, causing some outliers to be exaggerated.

It is recommended to repair the comparison/report harness first:

1. per-lead indicates clear separation:
   - `raw_independent`
   - `lead_representative`
   - `consensus`
   - `global_used`
2. The global indicator in summary is only compared with the GT global indicator from the same source.
3. The per-lead indicator is only compared with the per-lead GT. Do not copy the global consensus to each lead and use it as a per-lead measurement.
4. axis diff uses circular distance.

<a id="11-改进优先级"></a>
## 11. Improve priority

<a id="p0先修评估口径"></a>
### P0: Prerequisite assessment caliber

If these are not repaired, subsequent algorithm parameter adjustments will be misled:

1. axis diff uses circular difference.
2. Beat-level comparison is used for beat matching, and the first and last edge beats are processed separately.
3. Per-lead/global source separation to avoid repeated comparison of consensus values ​​to each lead.
4. QT dispersion is re-evaluated using independent per-lead QT.

<a id="p1修起搏低幅-qrs-detector"></a>
### P1: Repair pacing/low amplitude QRS detector

Target cases:

- 74
- 8
- 83
- 99
- 110

Goal:

- Eliminate HR half frequency/missed detection.
- QRS Enter multi-lead fallback when detector confidence is extremely low.
- Create capture QRS search window after pacing spike.

<a id="p2修-qrs-terminal-boundary"></a>
### P2: Repair QRS terminal boundary

Target cases:

- 108
- 90
- 5
- 23
- 170

Goal:

- Reduce QRS >20 ms outlier.
- Reduce ST-J's large deviation caused by QRS offset error.
- Make wide_qrs_jt path rely on reliable QRS offset.

<a id="p3修-t-end-rescuelow-support-qt"></a>
### P3: Fix T end rescue/low-support QT

Target cases:

- 57
- 73
- 125
- 81
- 133
- 24

Goal:

- Lowered low-support/rescue QT MAE from ~20-29 ms to 10-12 ms closer to main path.
- Reduce QT >30 ms outlier.

<a id="p4修-p-onsetpr-可用性"></a>
### P4: Fix P onset/PR availability

Target cases:

- 39
- 57
- 130
- 120
- 64

Goal:

- PR bias is pulled back close to 0 from about `-12 ms`.
- Reduce algorithm PR missing records.
- Improve the onset stability of long PR / low amplitude P.

<a id="12-建议的回归评估清单"></a>
## 12. Recommended regression evaluation checklist

After each algorithm change, it is recommended to track:

1. `summary.csv` global MAE/Bias.
2. PR >20 ms record number.
3. QRS >20 ms record number.
4. Number of QT >30 ms records.
5. QT dispersion bias.
6. HR >2 bpm record number, especially record 74.
7. MAE for low-support/rescue QT source.
8. `algorithm_pr`, `algorithm_t_axis` missing quantity.
9. `st_t_confusion`, `t_end_fallback`, `qrs_terminal_classification_only` flag count.
10. Manual review of the comparison chart of Record 57/74/108/8/109/116.

<a id="13-当前结论"></a>
## 13. Current conclusion

The backbone of the current algorithm can stably complete most common LUDB recorded HR/QRS/QT measurements. The most worthwhile investment first is to evaluate the caliber and boundary outliers, rather than a full rewrite.

The most cost-effective route in the short term is:

1. Learn the comparison/report harness first.
2. Repair the missing detection of pacing/low amplitude QRS.
3. Then repair QRS terminal boundary.
4. Then fix T end rescue/low-support.
5. Finally, system adjust P onset/PR and axis availability.


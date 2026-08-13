<!-- i18n-nav -->
[中文](lead_beat_representative_consensus_audit.md) | [English](lead_beat_representative_consensus_audit.en.md) | [日本語](lead_beat_representative_consensus_audit.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="per-leadper-beat代表波形与多导联共识利用审计"></a>
# Per-Lead/Per-Beat, Representative Waveforms and Multi-Lead Consensus Utilization Audit

Analysis objects:

- Current code: `feature_extraction/ecgfeat`
- Key example: `ludb_full_compare/5/5_features.json`
- Target question: Whether each waveform, each segment, and each feature correctly follows the measurement strategy of "first each lead and each beat, then synthesize the representative waveform, and then use multi-lead consensus".

<a id="1-总体结论"></a>
## 1. Overall conclusion

The current architecture already has three layers of information:

1. `beat_features`: P/QRS/T/ST/interval/morphology of each `lead × beat` independently measured.
2. `representative_leads`: Representative lead parameters summarized by lead.
3. `*_consensus_ms` and `global_features`: global interval after cross-lead consensus.

But the utilization methods are not exactly the same:

- QRS/QT global interval makes full use of multi-lead consensus.
- The P/PR section uses per-lead/per-beat and limb-lead summaries, but represents an underutilization of waveform and cross-lead P onset consensus.
- ST segment has added a raw remeasurement pass: QRS offset raw repair. After the beat-level J consensus, the risky ST-J/ST40/ST80 will be retested; however, there is still room for improvement in the representative-template ST measurement.
- QT dispersion Although independent per-lead QT is used, core cluster is selected, which easily reduces the dispersion between leads.
- The representative waveform is currently often a medoid single shot, not necessarily a multi-shot average/median template, and therefore has limited contribution to noise reduction and boundary refinement.
- Report/comparison contains mixed use of raw, representative, and consensus, which makes it easy to misjudge the algorithm itself.

In other words: **The framework direction is correct, but "representative waveforms" and "multi-lead consensus" are not yet fully utilized by all waveforms/segments in a consistent, traceable manner. **

<a id="2-当前流水线是否符合目标结构"></a>
## 2. Whether the current pipeline conforms to the target structure

<a id="21-已经做到了的部分"></a>
### 2.1 What has been done

The main process is at `feature_extraction/ecgfeat/api.py`:

1. First perform multi-lead QRS/R peak detection.
2. Then press R peak to create the beat.
3. Perform morphology grouping on the beat.
4. Build representative beat for each group.
5. `delineate_beats()` independently extracts boundaries and features for each beat × lead.
6. `delineate_beats()` executes `_apply_multilead_consensus()` internally and writes `qrs_consensus_ms`, `pr_consensus_ms`, and `qt_consensus_ms`.
7. Construct measurement representative beat again, and perform delineation again on this representative beat.
8. `build_representative_lead_features()` summarizes selected beats and representative beat feature into representative parameters for each lead.
9. `compute_global_features()` summarizes global PR/QRS/QT/axis/dispersion from representative leads and beat features.

Corresponding code location:

- `api.py:1303-1420`: QRS detection, grouping, representative waveform, beat×lead delineation.
- `api.py:1474-1506`: measurement representative beat reconstructed and representative lead features.
- `delineate.py:1964`: P/QRS/T/ST extraction for each beat × lead.
- `delineate.py:1596`: multi-lead consensus.
- `features.py:348`: Build representative lead features.
- `features.py:2001`: Calculate global features.

<a id="22-关键限制"></a>
### 2.2 Key limitations

Currently representative beat is not always a "multi-shot average template". `build_representative_beats_with_meta()` will do alignment, family clustering, outlier exclusion, and then give priority to family medoid. If medoid exists, `used_count=1`.

In record 5:

- `member_count = 9`
- `outlier_count = 1`
- `used_count = 1`

This shows that the representative waveform is actually a "representative single beat" and not a median/average template of 8 clean beats. This design can retain the true shape, but the noise reduction ability is not as good as the median template.

It is recommended to retain two types of representatives in the future:

- `medoid_template`: used for morphological classification, Q/R/S/notch/slur, etc.
- `median_template`: For boundary and low-noise measurements such as P onset, QRS onset/offset, T end, ST segment, etc.

<a id="3-各波形segment-的利用情况"></a>
## 3. Utilization of each waveform/segment

<a id="31-p-波与-pr-segmentpr-interval"></a>
## 3.1 P wave and PR segment/PR interval

Current practice:

- Each lead × beat independently detects P onset/peak/offset.
- P morphology, such as notched, biphasic, terminal negative, PTF-V1, measured on each beat×lead.
- When summarizing representative lead, P amplitude/area/morphology is given priority to representative beat feature; if not, pooled beats median is taken.
- PR interval preferentially takes pooled median in representative lead instead of representative waveform.
- global PR uses limb leads priority median, and has physiologic PR core rescue.

Advantages:

- P morphology has utilized representative waveforms that help stabilize P wave morphology.
- Global PR does not blindly use all prerecordial leads to avoid V leads and false Ps that shorten PR.
- In record 5, global PR is 130 ms, GT is 138 ms, and the error is only -8 ms, indicating that core rescue is effective in this example.

Disadvantages:

- PR/P onset is still at risk of being systematically short. The PR bias in the previous summary was about -11.77 ms.
- Indicates that the waveform is not used as the main path of the PR interval. The interval first uses pooled median.
- Multi-lead P onset consensus only writes `pr_consensus_ms`, but the global PR mainly still uses the per-lead PR of `_global_pr_median()`, and does not give priority to `pr_consensus_ms`.
- At low amplitude P or long PR, the SNR advantage of the representative waveform is not fully released.

Suggestions:

1. Reserve three sets of landmarks `p_on_independent`, `p_on_template`, and `p_on_consensus` for P waves.
2. Change global PR to take priority:
   - Multi-lead P onset consensus.
   -limb-lead raw PR core.
   - representative-template PR.
3. For long PR/low amplitude P, use the median representative template to retest P onset.
4. PR segment/baseline Don’t just rely on the local PR segment of a single beat. You can use the median of the PR segment representing the beat in the same lead as a baseline prior.

<a id="32-qrs-波群"></a>
## 3.2 QRS wave group

Current practice:

- Each lead × beat independently finds QRS onset/offset, Q/R/S, R'/S', notch, slur, VAT, QRS area.
- representative beat is first used to construct group-level boundary prior to help single-shot local correction.
- `_apply_multilead_consensus()` uses multi-lead QRS onset/offset to get `qrs_consensus_ms` and `qrs_wide_ms`.
- global QRS uses consensus first and has raw core fallback.

Advantages:

- QRS is currently the part that makes the most use of multi-lead consensus.
- per-lead WaveBounds are not covered by consensus and lead morphological differences are retained.
- Late terminal QRS has classification-only / measurement shunt to avoid treating all ST/T tail as QRS.

Problems exposed by record 5:

- Global QRS = 112 ms, GT = 80 ms, error +32 ms.
- independent per-lead QRS The difference is huge:
  - I: 70 ms
  - II: 80 ms
  - III: 191 ms
  - aVF: 128 ms
  - V1: 66 ms
  - V5: 82 ms
- `qrs_consensus_ms = 112 ms`, `qrs_wide_ms = 128 ms`.

This shows that the current consensus has indeed suppressed the extreme value of 191 ms of III, but it is still widened by several late offsets, causing global QRS to be wider.

Disadvantages:

- QRS consensus is not robust enough to late offsets.
- The representative template does not serve as a strong constraint for the "standard QRS end point" consensus.
- QRS offset error will be passed to ST-J, JT, QT path.

Suggestions:

1. QRS global measurement should add `representative_median_template_qrs` as third-party evidence.
2. If the per-lead raw QRS core is between 70-90 ms and the consensus is >110 ms, late offset should require at least multi-lead morphological evidence.
3. QRS offset consensus should output:
   - measurement offset
   - classification/wide offset
   - raw core offset
   -template offset
4. ST-J/ST40/ST80 should be based on measurement offset; BBB/IVCD classification can be based on classification offset.

### QRS offset raw repair update

QRS offset raw repair pass is currently added. It uses the measurement terminal consensus of reliable QRS leads in each beat to identify the obviously late/early and low-confidence raw `qrs.offset` outliers, and only repairs the lead×beat that is judged to be a measurement error, without forcibly pulling all leads to the same offset.

After correction, `LeadBeatFeatures.qrs.offset` will be written back and recalculated. `qrs_ms`, `qrs_area`, `qrs_signed_area`, `j_index`, `st_on_mv`, `st_mid_mv`, `st_80ms_mv`, `jt_ms`, `qt_ms` and a terminal QRS morphology field that can be safely recalculated. The wide QRS classification still retains the `qrs_wide_ms` path and is not mixed with the measurement offset.

In order to avoid excessively shortening the reasonable QRS, the repair gate will also protect lead×beat whose original width is reasonable and there is no evidence of ST/T confusion or beat-unreliable; only obvious long-tail, low-confidence or unreliable contexts are allowed to be corrected by consensus feedback.

LUDB full-batch results:

- QRS MAE: 9.88 -> 9.88
- QRS bias: 5.727 -> 5.727
- `|QRS diff| > 20 ms` / `> 40 ms`: 17/2 -> 17/2
- QT MAE: 13.4625 -> 13.4625
- QRS/P/T missing: 0/2669/258 -> 0/2669/258
- raw QRS any-boundary missing: 0 / 25860
- `qrs_offset_repaired_by_consensus`: 7 rows across 6 records

## 3.3 ST segment / J point

Current practice:

- ST-J, ST40, ST80 measured on every lead × beat.
- representative lead's `st_on_mv` aggregated by `_representative_st_j_params()`.
- ST-J unreliable is marked if QRS tail guard is found.
- 12SL profile also has STJ/STM/STE.
- After QRS offset raw repair, add ST/J raw remeasurement pass: for lead×beat with evidence of `st_j_unreliable`, tail jump, J offset outlier or beat-unreliable, use the J-point consensus of reliable QRS leads in the same beat to resample ST-J/ST40/ST80 on the original lead waveform.

Advantages:

- ST is a per-lead feature and is not simply covered by the global consensus.
- There is a QRS tail guard to prevent some QRS ends from contaminating the ST.
- ST/J raw remeasurement does not change `qrs.onset`, `qrs.offset` or compatible `j_index`, only writes the actual ST sample anchor to `st_j_remeasured_index` and recalculates the ST value, so the real inter-lead QRS difference can be preserved.

Disadvantages:

- ST-J without triggering raw remeasurement is still primarily based on per-lead local QRS offset rather than representative-template J point.
- `_apply_multilead_consensus()` still only sets ST-J to unreliable when `qrs_wide_ms` conflicts with ST-J/ST40/ST80; the real remeasurement occurs in the earlier ST/J raw remeasurement pass.
- representative beat The use of ST segment is not stable enough: ST-J sometimes takes rep_feature, sometimes takes pooled median, but does not explicitly use consensus J point resampling.

### ST/J raw remeasurement update

Currently a new ST/J raw remeasurement pass is added. After QRS offset raw repair, it uses the J-point consensus of reliable QRS leads in the same beat as the sampling anchor, and only retests the lead×beat ST segment with `st_j_unreliable`, tail jump, offset outlier or beat-unreliable evidence.

This pass does not change `qrs.onset`, `qrs.offset` or the compatible `j_index`, but records the ST sample anchor to `st_j_remeasured_index`. The correction will recalculate `st_on_mv`, `st_mid_mv`, `st_80ms_mv`, `st_slope_mv_per_ms` and `st_morphology`. If J/ST40/ST80 are already consistent, the true ST elevation/depression will not be smoothed out.

LUDB full-batch results:

- PR MAE: 9.5372 -> 9.5372; QRS MAE: 9.8800 -> 9.8800; QT MAE: 12.8575 -> 12.8575
- PR/QRS/QT `|diff| > 20 ms`: 17/17/33 -> 17/17/33
- QRS/P/T missing: 0/2669/258 -> 0/2669/258
- `st_j_unreliable`: 305
- `st_j_remeasured_by_consensus`: 1724 rows across 164 records
- reason split: `offset_outlier=1092`, `tail_guard=417`, `beat_unreliable=215`

In record 5:

- I lead representative ST-J = -0.104 mV, but pooled median ST-J is about -0.069 mV.
- aVL representative ST-J = +0.042 mV, but pooled median ST-J is about -0.061 mV.

This means that the ST level representing single beats may not be consistent with the pooled beats. If the representative waveform is a medoid single beat, this ST-J difference may come from the single beat baseline/offset rather than the stable ST segment.

Suggestions:

1. Establish an independent representative-template measurement for the ST segment:
   - `st_j_raw_local`
   - `st_j_consensus_j`
   - `st_j_template`
   - `st_j_12sl`
2. For diagnosis/reporting, `st_j_template_consensus_j` or reliable pooled median is preferred instead of single medoid ST-J.
3. If the difference between representative ST-J and pooled median is >0.05-0.08 mV, it should be marked as unstable or fall back to pooled median.
4. ST segment should be bound to QRS measurement offset instead of wide/classification offset.

<a id="34-t-波与-qt"></a>
## 3.4 T wave and QT

Current practice:

- Each lead × beat independently detects T peak/T end.
- T candidate selection takes into account polarity, area, width, ST-T confusion.
- Multi-lead consensus uses T-end median of reliable QT leads to obtain `qt_consensus_ms`.
- Independent `qt_ms` and consensus `qt_consensus_ms` are retained in representative lead.
- global QT gives priority to reliable leads median and records `consensus_vs_independent_per_lead`.

Advantages:

- QT is currently the feature that makes the most use of multi-lead consensus except QRS.
- III independent QT = 504 ms, V1 = 328 ms in record 5; global QT is 402 ms after using consensus and is not dragged away directly by single-lead extreme values.
- `global_qt` metadata retains independent vs consensus QT for each lead for easy auditing.

Disadvantages:

- T wave shape, T amplitude, T area may still come from representative beat or pooled raw, and may not necessarily be reintegrated using consensus T-end.
- `qt_consensus_ms` has high priority in global QT, possibly at the expense of true lead differences.
- The low-support/rescue path error is significantly larger than the main path.

In record 5:

- Global QT = 402 ms, GT = 382.5 ms, error +19.5 ms.
- independent QT:
  - V1: 328 ms
  - V2/V3: 368 ms
  - I/II/V5: 384 ms
  - aVF: 416 ms
  - III: 504 ms
- consensus QT = 402 ms.

This shows that the consensus suppresses the outlier, but the overall T end or QRS onset is still slightly later/longer.

Suggestions:

1. T morphology should be distinguished:
   -raw local T bounds
   - consensus T end
   - representative-template T bounds
2. If T area/T duration/Tpe is used for diagnosis, it should be recalculated according to consensus T-end on the representative template.
3. The lead of `st_t_confusion` should not directly participate in the T morphology master table unless the representative waveform and adjacent leads are unanimously supported.
4. Global QT can continue to prioritize consensus, but the report must display both independent QT and consensus QT.

### T-end raw repair update

Currently a new T-end raw repair pass is added. It uses the T-end measurement consensus of reliable QT leads in each beat to identify raw `t.offset` outliers that are obviously early/late and have low confidence or risk evidence. It only repairs the leads × beats that are judged to have measurement errors, and does not force all leads to the same T-end.

After correction, `LeadBeatFeatures.t.offset` will be written back and `qt_ms`, `jt_ms`, `t_dur_ms`, `t_area`, `t_signed_area` and `tpe_ms` will be recalculated. Real long T with high confidence and reasonable form will not be forcibly pulled back to the consensus; if the repair candidate will cause QT `<280 ms`, the repair will be refused to avoid QRS tail/ST-T confusion mistakenly regarded as T-end.

Before the consensus raw repair, a more conservative T-end dual-method raw rescue pass has been added. It only processes raw `t.offset` that are "obviously early-truncated, low-confidence, or fallback/risky," first establishing a beat-level T-end consensus using reliable QT leads, and then simultaneously calculating three types of T offset evidence—chord, slope-return, and tangent—on the original waveform of the same lead. The raw `t.offset` is written back only when the spread of the three endpoints `<=30 ms`, the candidate does not cross the next QRS, it does not cause QT `<280 ms`, and the extension does not exceed `160 ms`.

The goal of this pass is not to improve the final report aggregation, but to make the raw `lead × beat` T offset itself more credible. It separately records `t_offset_dual_original_index`, `t_offset_dual_rescued_index`, `t_offset_dual_delta_ms`, `t_offset_dual_consensus_index`, `t_offset_dual_support`, `t_offset_dual_used_leads`, `t_offset_dual_chord_index`, `t_offset_dual_slope_index`, `t_offset_dual_tangent_index`, and `t_offset_dual_method_spread_ms`, to avoid confusion with subsequent consensus repair provenance.

LUDB full-batch results:

- QT MAE: 13.4625 -> 12.8575
- QT bias: -1.1875 -> -1.2125
- `|QT diff| > 20 ms` / `> 40 ms`: 34/17 -> 33/14
- QRS MAE: 9.88 -> 9.88
- PR MAE: 9.537236842105264 -> 9.537236842105264
- QRS/P/T missing: 0/2669/258 -> 0/2669/258
- raw T peak/offset missing: 378 / 25860
- `t_offset_dual_method_rescued`: 1 row across 1 record
- dual rescue case: record 54, lead V1, beat 5, `3522 -> 3534` samples (`+24 ms`), support 8, method spread `28 ms`
- `t_offset_repaired_by_consensus`: 1201 rows across 167 records
- consensus repair reason split: 1001 `late_outlier`, 200 `early_truncation`

<a id="35-u-波"></a>
## 3.5 U Wave

Current approach:

- The U wave is primarily used as a secondary peak flag after the T-end.
- Prominent U waves affect the T-U nadir and T-end clamping.
- There are no independent representative waveform measurements for U onset/peak/offset.

Shortcomings:

- There is no complete per-lead/per-beat U wave feature pipeline.
- There is no representative U morphology.
- The impact of prominent U waves on QT/T-end can be flagged, but it is not sufficiently interpretable.

Recommendations:

1. To systematically utilize the U wave, new `u` bounds and `u_amp_mv/u_area/u_confidence` should be added.
2. QT and QU should be output separately: `qt_ms`, `qu_ms`.
3. When the T-end is corrected by the U wave, retain the original T-end and T-U nadir in the metadata.

## 3.6 QT Dispersion

Current approach:

- `_raw_qt_dispersion()` uses independent `qt_ms` from representative leads, not consensus QT.
- However, it filters out leads with low confidence, high SD, invisible T waves, or unreasonable paths.
- If the raw dispersion is too large, it selects the core cluster and returns the max-min of the core lead cluster.

In record 5:

- Algorithm QT dispersion = 22 ms.
- GT QT dispersion = 64 ms.
- The independent QTs actually show significant differences: V1 328 ms, III 504 ms, aVF 416 ms.
- The current logic filters/clusters these outlier values, retaining only the core cluster, so the dispersion is compressed to 22 ms.

Conclusion:

- The current approach is not entirely wrong in using consensus QT to calculate dispersion, but the "core cluster" strategy systematically underestimates the clinically significant lead dispersion.

Recommendations:

1. Output simultaneously:
   - `qt_dispersion_raw_all_reliable`
   - `qt_dispersion_core_cluster`
   - `qt_dispersion_reported`
2. When comparing with LUDB GT, prioritize the raw independent lead dispersion that is closer to the annotation definition.
3. Do not display only the core dispersion in the report, as this would hide inter-lead differences.

## 3.7 Axis

Current approach:

- P/QRS/T axis is calculated from the signed area or net amplitude of representative leads.
- Confidence is used as a weight/filter.

Advantages:

- Axis inherently requires cross-lead combination; the current approach does not rely on a single lead.
- Using signed area is more stable than using a single-point peak.

Shortcomings:

- If the P/T area of the representative leads comes from a single medoid beat, noise and ST baseline will affect the axis.
- T axis is missing frequently, indicating that T amplitude/confidence gating is too conservative.
- The axis difference in the comparison does not use circular distance, which amplifies some errors.

Recommendations:

1. Use the signed area of the median representative template for the axis, rather than a single medoid beat.
2. Output reliability and used leads for the T axis.
3. Use circular difference for the comparison.

<a id="4-report-comparison-的混用问题"></a>
## 4. Mixing Issues in Report / Comparison

In the current `compare_annotations.py`, the per-lead QRS/QT comparison prioritizes consensus:

- `qrs_ms -> qrs_consensus_ms`
- `qt_ms -> qt_consensus_ms`

This causes the per-lead QRS/QT of `*_comparison.txt` to appear identical across all 12 leads.

In record 5:

- The per-lead table in the algorithm report shows raw representative values:
  - I QRS 70, II QRS 80, III QRS 191, aVF QRS 128
  - I QT 384, III QT 504, V1 QT 328
- The comparison table, however, shows QRS = 112 and QT = 402 for all leads.

Therefore, the per-lead QRS/QT of comparison is not a pure per-lead representative measurement, but a consensus measurement. This caliber will mask:

- Which leads independently measure poorly.
- Whether the representative waveform improves the lead.
- consensus whether excessive flattening of lead differences.

It is recommended to modify comparison/report:

1. Column breakdown in per-lead comparison:
   - `raw_independent`
   - `representative_template`
   - `pooled_median`
   - `consensus`
2. Use consensus for global QRS/QT in summary.
3. Do not use consensus instead of raw by default for per-lead tables.
4. For QT dispersion, please indicate whether to use raw or core cluster.

<a id="5-特征级审计表"></a>
## 5. Feature-level audit table

| Features | per lead × beat | representative waveform | multi-lead consensus | Current rating |
|---|---|---|---|---|
| P onset/offset | Yes | Yes, but interval is not preferred | Yes `pr_consensus_ms`, but global PR is not preferred | Partially sufficient |
| P morphology | Yes | Priority representative/pooled | No need for strong consensus | Basically reasonable |
| PR interval | Yes | Not the main path | Yes but not fully used for global PR | Needs enhancement |
| Baseline/PR segment | There is a partial baseline | The representative template baseline is not fully used | No clear consensus | Needs to be strengthened |
| QRS onset/offset | Yes | Used as prior | Strong consensus | Basically sufficient, but terminal offset needs to be changed |
| Q/R/S/R'/S' | Yes | Priority representative/pooled | Should not be globally overridden | Reasonable |
| ST-J/ST40/ST80 | Yes | Yes but medoid may be unstable | Only guard, no retest | Need to strengthen |
| ST morphology/slope | Yes | Yes | Weak | Needs strengthening |
| T peak/T end | Yes | Yes | T-end consensus strong | Basically sufficient |
| T amplitude/area | Yes | Priority representative/pooled | Not re-integrated according to consensus T-end | Need to strengthen |
| QT interval | Yes | pooled/representative are all retained | global priority consensus | basically sufficient, but outlier/rescue needs to be changed |
| JT interval | Yes | Yes | Indirectly affected by QRS/QT consensus | Partially sufficient |
| Tpe | yes | pooled interval priority | weak | needs to be strengthened |
| U wave | flag level | No complete representative measurements | None | Inadequate |
| QT dispersion | Use independent QT | Use representative leads | Avoid using consensus directly, but the core cluster will suppress it | Definitions need to be split |
| Axis | representative leads | Yes | Achieved through lead combination | Basically reasonable |

<a id="6-推荐的目标架构"></a>
## 6. Recommended target architecture

It is recommended to explicitly separate each feature into four levels instead of mixing it into one field:

1. `raw_lead_beat`: Single-lead single-beat raw measurement.
2. `template_lead`: Same lead represents waveform measurement.
3. `beat_multilead_consensus`: Cross-lead fiducial consensus of the same beat.
4. `global_reported`: Stable global value used for final reporting.

Field example:

```text
qrs_ms_raw_lead
qrs_ms_template_lead
qrs_ms_consensus_beat
qrs_ms_global_reported

qt_ms_raw_lead
qt_ms_template_lead
qt_ms_consensus_beat
qt_ms_global_reported

st_j_raw_local
st_j_template_consensus_j
st_j_reported
```

<a id="7-优先修改建议"></a>
## 7. Prioritize modification suggestions

<a id="p0先改报告比较口径"></a>
### P0: Change the report/comparison caliber first

First, display raw, representative, and consensus separately. Otherwise it is difficult to tell whether the algorithm actually utilizes the representative waveform.

Key documents:

- `compare_annotations.py`
- report/export related table generation

<a id="p1代表波形同时保留-medoid-与-median-template"></a>
### P1: Represents the waveform while retaining medoid and median template.

The medoid is suitable for shapes, and the medial template is suitable for borders and segments.

Suggestions:

- `representative_medoid`
- `representative_median`
- `representative_mean_or_trimmed_mean`

<a id="p2st-segment-重新绑定-consensus-j-point"></a>
### P2: ST segment rebind consensus J point

For each lead's representative median template, retest using consensus QRS offset:

-ST-J
- ST40
- ST80
- ST slope

<a id="p3qrs-terminal-consensus-加-template-guard"></a>
### P3: QRS terminal consensus plus template guard

Late offset multi-lead terminal morphology support is required when global QRS is significantly wider than raw/template core.

<a id="p4qt-dispersion-拆成-rawcorereported"></a>
### P4: QT dispersion is split into raw/core/reported

Don't just print core cluster dispersion. The LUDB comparison should use a version closer to the independent lead annotation.

<a id="p5p-onsetpr-使用-representative-consensus-主路径"></a>
### P5: P onset/PR uses representative + consensus main path

Especially for long PR, low amplitude P, LAE/RAE, and noise scenes.

<a id="8-对-record-5-的具体判断"></a>
## 8. Specific judgment on record 5

record 5 shows that the current process is useful, but not sufficient:

- HR and PR are acceptable: HR error -0.2 bpm, PR error -8 ms.
- QRS is wider: global QRS 112 ms vs GT 80 ms, indicating that the QRS consensus is widened by late offsets.
- QT is longer: global QT 402 ms vs GT 382.5 ms, partly from QRS/global fiducial bias.
- QT dispersion underestimation: 22 ms vs GT 64 ms, from core cluster filtering.
- The difference between representative and pooled median in some leads of ST-J is obvious, indicating that medoid single beat is not necessarily suitable for ST segment.

Therefore, the focus of improvement in record 5 is not "whether there is representative/consensus", but:

1. QRS terminal consensus should be more constrained by raw/template core.
2. ST segment should be retested using representative median template + consensus J point.
3. QT dispersion needs to be redefined with independent lead values.
4. The comparison table should stop replacing per-lead QRS/QT with consensus by default.

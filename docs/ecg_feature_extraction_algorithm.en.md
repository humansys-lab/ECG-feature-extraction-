<!-- i18n-nav -->
[中文](ecg_feature_extraction_algorithm.md) | [English](ecg_feature_extraction_algorithm.en.md) | [日本語](ecg_feature_extraction_algorithm.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecg-特征提取算法逻辑与流程说明"></a>
# ECG Feature extraction algorithm logic and process description

This document is based on the current warehouse code and is dated 2026-07-06. The core implementation is
`ECGFeatureExtractor.extract()` under `feature_extraction/ecgfeat`, the overall
DXL-inspired 12-lead ECG feature extraction process, not Philips DXL proprietary algorithm
Line-by-line reproduction is not a proven medical device software.

<a id="1-入口与整体数据流"></a>
## 1. Entry and overall data flow

<a id="11-主要入口"></a>
### 1.1 Main entrance

| Scenario | File/Function | Function |
| --- | --- | --- |
| Single record demonstration | `demo_feature_extraction.py` | Read ECG and patient information from local `dataset/*.mat/.hea`, call the extractor, generate `*_features.json`, reports and visualizations. |
| Batch `.pt` data | `batch_extract_ecgfeat.py` | Normalize the batch input shape, call the extractor sample by sample, and save JSON/PT/report/manifest. |
| Algorithm main entrance | `feature_extraction/ecgfeat/api.py::ECGFeatureExtractor.extract` | Complete feature extraction, pacing/rhythm strategy, interpretation layer and metadata assembly. |
| Export | `feature_extraction/ecgfeat/export.py::to_dict` | Expand the dataclass result into a JSON friendly structure, and add `rhythm_inputs`, `morphology_inputs`, `statement_engine`. |

<a id="12-输入约定"></a>
### 1.2 Input convention

The core input is:

- `ecg_12lead`: `numpy.ndarray`, expected shape is `[12, n_samples]`, unit is mV.
- `fs`: Original sampling rate.
- `meta`: Optional `PatientMeta(age, sex, meds)` for adult/child rule routing, gender related thresholds and reporting context.

If the ECG input is `[n_samples, 12]`, the extractor is automatically transposed. If there are neither 12 rows nor 12 columns, `ValueError` will be thrown.

The standard lead sequence is fixed to:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

<a id="13-主流程总览"></a>
### 1.3 Overview of the main process

```text
输入 ECG
  -> 采样率统一和双路预处理
  -> 导联质量评估、导联反接检测、起搏 spike 检测
  -> 多导联 QRS/R 峰检测
  -> 起搏 spike 与 QRS 校验，必要时去 spike 后重跑 QRS
  -> beat grouping 和 measurement group 选择
  -> representative beat 构建
  -> 逐搏逐导联 P/QRS/T 边界定位和基础测量
  -> 根据定位结果重新评估 measurement group
  -> atrial event、AF/AFL、起搏/AV block/preexcitation 规则输入
  -> measurement representative 和 representative lead 特征
  -> group/global 特征聚合，QT/PR/QRS rescue，axis 计算
  -> 12SL-style 并行 measurement profile
  -> rhythm availability 和 statement evidence
  -> clinical interpretation
  -> JSON 导出
```

<a id="2-预处理"></a>
## 2. Preprocessing

Implementation location: `preprocess.py`

<a id="21-采样率"></a>
### 2.1 Sampling rate

`fs_run` is controlled by construction parameters:

- `fs_internal is None`: Use the rounded integer of input `fs`.
- `fs_internal` When specified: Resample to this internal sample rate.

Resampling uses `scipy.signal.resample_poly` to simplify the upper and lower sampling ratio according to the greatest common divisor of the input/output sampling rate.

<a id="22-两路信号"></a>
### 2.2 Two signals

The extractor retains two signals for different purposes:

| Signal | How to build | Purpose |
| --- | --- | --- |
| `ecg_an` / measurement signal | Two-stage median filtering to remove baseline drift, default 200 ms and 600 ms windows; then do power frequency notch. No low pass. | All amplitude, boundary, ST/T/P/QRS measurements, try to retain peak values. |
| `ecg_det` / detection signal | `analysis_signal` and then set the default 40 Hz fourth-order low pass. | Only used for QRS/R peak detection to reduce EMG noise. |

<a id="3-质量评估与反接检测"></a>
## 3. Quality assessment and reverse connection detection

Implementation location: `quality.py`

<a id="31-每导联质量"></a>
### 3.1 Quality per lead

`compute_quality()` Calculates for each standard lead:

- baseline wander: `0.5 Hz` The following trends are relative to the overall standard deviation.
- muscle noise: `35-100 Hz` frequency band power ratio.
- powerline noise: Power frequency `mains_freq +/- 1 Hz` power ratio.
- clipping: the ratio of adjacent spreads to 0.
- flatline/missing: standard deviation `< 0.005 mV`.

Output `LeadQuality`:

- Overall `reliable`
- Waveform-specific reliability: `reliable_for_p`, `reliable_for_qrs`, `reliable_for_t`, `reliable_for_qt`
- `grade`: `Q0/Q1/Q2/Q3`
- `flags` and `reason_codes`

Main thresholds:

- P wave is stricter than QRS, baseline wander `< 0.30`, muscle `< 0.20`, powerline `< 0.10`.
- QRS tolerates higher myoelectric noise, muscle `< 0.40`, powerline `< 0.20`.
- QT requires both T and QRS to be reliable.

`summarize_record_quality()` summarizes the number of reliable leads in all records and gives:

- `record_grade`
- `rejected_functions`: For example, `p_measurement`, `qt_measurement`, `record`
- global reason codes

<a id="32-导联反接"></a>
### 3.2 Lead reverse connection

Limb lead reverse connection uses heuristic:

- Einthoven relationship error: `II - I - III`
- Lead correlation, such as `I` and `-II`, `aVR` and `aVL/aVF`
- Output `probable_ra_la`, `probable_ra_ll`, `probable_la_ll`, `probable_extremity_reversal`

Precordial lead reverse connection is detected in the representative lead phase based on the progression of the V1-V6 R wave:

- If the R-wave progression score is low and significantly improved after swapping a pair of adjacent chest leads, mark `probable_precordial_reversal`.
- This judgment will be skipped for low voltage chest leads.

<a id="4-起搏-spike-检测与去-spike"></a>
## 4. Pacing spike detection and spike removal

Implementation location: `quality.py`, `api.py`

The pacing logic is an important bypass in the main flow as it changes the measurement signal, beat flag, QRS width and PR availability.

<a id="41-spike-候选"></a>
### 4.1 spike candidates

Default detector:

- Differential enhancement for each lead.
- Estimating local noise with rolling MAD.
- Candidates need to cross the `4 * sigma` dynamic threshold.
- The spike width requirement is about `<= 6 ms`.
- The default absolute prominence threshold is `250 uV`, and the high-amplitude spike reference threshold is `1000 uV`.

There is also legacy high-pass RMS detection as a strong evidence fallback.

<a id="42-多导联共识"></a>
### 4.2 Multi-lead consensus

Candidates are merged into events by about `5 ms` windows. The event needs to meet:

- Default to at least 4 lead votes, or weak vote weighted score high enough.
- If there are more than 3 spikes, neighbor artifacts with intervals less than half the median interval will be removed.
- The rhythm is considered paced when the spike interval is robust CV `< 0.15`.

<a id="43-qrs-校验与去除伪影"></a>
### 4.3 QRS verification and artifact removal

After the initial QRS detection, `validate_pacing_spikes_against_qrs()` will eliminate the situation where the QRS edge is mistakenly recognized as a spike:

- Calculate the offset of the spike from the nearest R peak.
- Compare spike high frequency peak with QRS high frequency peak.
- Reserved when the width is narrow and relatively QRS high-frequency energy dominates.
- If a large number of spikes are fixedly synchronized with QRS and resemble the QRS edge artifact, clear the spike and mark `qrs_edge_artifact`.
- Possible atrial/non-ventricular pacing is handled conservatively and does not directly affect ventricular paced measurements.

If the spike is confirmed, `remove_pacing_spikes()` will default to the `+-4 ms` window linear interpolation near each spike, and then use the signal after the spike to reconstruct the detection signal and rerun QRS.

### 4.4 attenuated pacing rescue

If the initial pacing evidence is insufficient, `api.py` will try to lower the prominence to the rescue path of `150/100/25 uV`. This path will only affect the downstream after re-spike, re-run QRS, and pass QRS capture verification.

<a id="45-beat-级起搏状态"></a>
### 4.5 beat level pacing status

If the spike falls within the range of `R - 80 ms` to `R + 20 ms` and the capture alignment is established, the beat is marked as paced. It will be generated later:

- `paced_beat_ids`
- `pacing_spike_beat_ids`
- `pacing_spike_offsets_ms`
- `pacing_capture_confirmed`
- `pacing_detection_state`
- `measurement_pacing_state`
- `pacing_measurement_effect`
- `pacing_segmentation_effect`

<a id="5-多导联-qrsr-峰检测"></a>
## 5. Multi-lead QRS/R peak detection

Implementation location: `qrs.py`

<a id="51-检测信号"></a>
### 5.1 Detection signal

By default, lead `I, II, V2, V3, V4, V5, V6` is used to construct vector magnitude:

```text
VM = sqrt(mean(zscore(lead)^2))
```

After:

- bandpass `5-25 Hz`
- The energy is obtained by squaring the first difference
- `120 ms` Moving Average

<a id="52-阈值和-r-峰精修"></a>
### 5.2 Threshold and R-Peak Refinement

Main threshold:

```text
max(P95(energy) * 0.30, mean + 0.5 * std)
```

The minimum distance between peaks is approximately `220 ms`. Each energy peak uses the lead II local waveform to refine the R position within `+-50 ms`; if it is a negative QS dominant waveform, the fiducial will be placed at the beginning of the steep negative direction instead of the late S/QS valley.

Edge protection: Candidates with a distance of `<160 ms` from the recording edge will be discarded to avoid incomplete measurement.

### 5.3 robust fallback

If the main detection is too sparse, a robust threshold based on median/MAD will be tried. Replace the main result only if the result is robust by at least 3 beats, has a reasonable detection rate, and significantly complements the main detection.

<a id="6-beat-grouping-与-measurement-group"></a>
## 6. Beat grouping and measurement group

Implementation location: `grouping.py`, `api.py`

### 6.1 beat template

Each beat takes a 12-lead segment from `R - 80 ms` to `R + 120 ms`, interpolates each lead to 32 points, removes the mean and normalizes, and then splices it into a template vector.

<a id="62-两阶段聚类"></a>
### 6.2 Two-stage clustering

`cluster_beats()`：

1. Roughly estimate the width of QRS based on vector magnitude, and divide it into narrow/wide categories according to `<120 ms` and `>=120 ms`.
2. Various internal online morphology clustering, cosine similarity `>=0.88` are classified into existing groups.
3. Maximum 5 groups, maximum 2 groups for wide QRS branch.
4. Sort by group size, `group 1` is the dominant group.
5. The paced beats will be reorganized individually to avoid contaminating the native measurement family.

<a id="63-measurement-group-选择"></a>
### 6.3 measurement group selection

`api.py::_select_measurement_group()` Select the measurable group according to the proportion of paced beat:

- If the global paced majority is used, paced beat will be given priority.
- Otherwise native/non-paced beat takes precedence.
- If subsequent delineation finds that the paced group is unstable and there is a stable native family, the measurement group will be re-selected.

The final result is written as:

- `representative_group_id`
- `measurement_beat_ids`
- `measurement_group_reselected`
- `measurement_group_reselect_reason`

<a id="7-representative-beat-构建"></a>
## 7. Representative beat build

Implementation location: `representative.py`

Each group construct represents a beat:

- Window: `R - 300 ms` to `R + 500 ms`
- Use vector magnitude for cross-correlation alignment, maximum lag `+-30 ms`
- For irregular rhythms, if the number of beats in the group is `>=6` and RR CV `>0.15`, the beats with RR within the group median `+-20%` will be retained first
- Select dominant family using morphology family assignment
- Use medoid to represent the dominant family, and use the correlation coefficient threshold `0.70` to exclude outlier beats if necessary
- fallback is per-sample median

Representing pulsation is mainly used for:

- Provide P/QRS/T boundary prior
- Build measurement representative beat
- Stable representative lead parameter

<a id="8-逐搏逐导联-pqrst-边界定位"></a>
## 8. Beat-by-lead P/QRS/T boundary positioning

Implementation location: `delineate.py`, `repolarization.py`, `p_morphology.py`

<a id="81-beat-窗口和搜索范围"></a>
### 8.1 beat window and search range

Default analysis window for each beat:

```text
beat_start = R - 350 ms
beat_end   = R + 550 ms
```

T-end cannot cross the protection zone before the next QRS:

```text
t_cap = next_R - max(150 ms, 28% * RR)
```

P wave lookback adaptive:

- First beat: `400 ms`
- Other beats: `min(450 ms, 65% * previous_RR)`

<a id="82-representative-prior-和多导联-pt-anchor"></a>
### 8.2 representative prior and multi-lead P/T anchor

If there are representative beats, a rough delineation will be done on the representative beats first to obtain the boundary offset prior of each group/lead.

Beat-level P/T anchor fusion is also performed during beat-to-beat measurement:

- Collect P/T peak candidate triplets from all leads.
- Combine quality markers and representative prior.
- P anchor cluster radius is about `12 ms`.
- T anchor cluster radius is about `20 ms`.
- The fused anchor will narrow the local search window of each lead, but will not hard cover all boundaries.

<a id="83-qrs-边界"></a>
### 8.3 QRS Boundary

The core of `_qrs_bounds()`:

- Bandpass `5-30 Hz` for single-lead signals.
- Smoothing with first derivative squared energy, `12 ms`.
- The threshold is `max(4% * QRS peak energy, 1.5 * noise_floor)`.
- If the QRS detector uses a low-slope fallback, the low-slope fiducial guard will be enabled.
- Use second derivative zero-crossing to refine onset/offset.
- Additional rescue for sustained low slope QRS foot.
- Returns onset/offset, notch count, onset/off confidence.

Special treatment for pacing beat:

- If the spike is near the beat, QRS onset is not allowed to be earlier than the spike.
- If capture is confirmed and QRS is too narrow, the minimum QRS width `120 ms` floor will be applied to the paced beat.
- If intrinsic QRS is significantly narrow, the floor will be suppressed to prevent QRS edge artifact from being mistaken for paced QRS.

QRS offset raw repair second pass:

- After completing the initial QRS measurement for each lead × beat, first use reliable QRS leads to establish beat-level measurement terminal consensus.
- For raw `qrs.offset` that obviously deviates from the consensus, has low offset confidence, and is judged as late/early outlier by the multi-lead consensus, partial review and correction of the original `WaveBounds` is allowed.
- Immediately recalculate the lead×beat’s `qrs_ms`, QRS area/signed area, `j_index`, ST-J/ST40/ST80, JT/QT and terminal morphology that can be safely recalculated after correction.
- This pass does not change the QRS onset, and does not force all leads to the same offset; the wide QRS/BBB/pacing classification still retains the `qrs_wide_ms` path.
- Over-shortening fixes will be rejected if the original QRS width itself is reasonable and there is no evidence of ST/T confusion or beat-unreliable.

ST/J raw remeasurement second pass:

- Use reliable QRS leads to establish beat-level J-point measurement consensus after QRS offset raw repair.
- For `st_j_unreliable`, where the J point and ST40/ST80 jump are too large, the local J point obviously deviates from the consensus, or the beat is unreliable, resample ST-J/ST40/ST80 on the original waveform of the same lead.
- Corrected recalculation of `st_on_mv`, `st_mid_mv`, `st_80ms_mv`, `st_slope_mv_per_ms` and `st_morphology`.
- This pass does not change `qrs.onset`, `qrs.offset`, or `j_index` for compatibility; the actual ST sample anchor is written to `st_j_remeasured_index`, and the original index, delta, cause, supporting leads, and excluded leads are logged.
- If J/ST40/ST80 are internally consistent, it means it may be true ST elevation/depression, and no retest is required.
- The report-level PR/QRS/QT and missing indicators in LUDB full-batch remain unchanged, and 1724 raw ST/J lead×beat features are retested, covering 164 records.

<a id="84-p-波"></a>
### 8.4 P wave

native P wave path:

- The search window is before QRS and is limited by fused P anchor or prior.
- Candidate peaks must fall within the approximate physiological PR range, the current code uses approximately `60-350 ms` pre-range verification.
- Amplitude needs to be at least `max(8 uV, 1.5% * R amplitude)`.
- onset uses the ascending tangent, and offset uses the descending tangent.
- After final candidate re-selection, additionally generate zero-phase `35 Hz` low-pass P
  Detection view; the original measurement signal is still used for amplitude, area and form measurements.
- Subsequent beats are only estimated from the verified TP quiet window between the previous beat T offset and the current beat P onset
  Local noise; the PR/PQ section contains Ta and is no longer used as a noise reference. The first shot can only be used if it is explicitly marked as
  Unvalidated pre-P fallback for `previous_t_out_of_context`. output
  `p_local_noise_rms_mv`, `p_local_snr` and `p_informative`. Less than `2 sigma`
  Candidates are retained for low-confidence coverage fallback, but do not participate in high-confidence boundary order statistics.
- Output `p_tp_gap_ms`, `p_quiet_window_available`, `p_on_t_overlap_risk`,
  `p_ta_overlap_risk`, `p_baseline_method` and `p_baseline_confidence`.
  When TP disappears, only continuous robust quadratic trends are fitted on both sides of the P candidate for boundary stability diagnosis,
  Original boundaries are not overwritten; boundary confidence is capped and multi-lead fusion is forced to fall back to cluster median.
- Measured with tangent on low-pass P-view with `4%-12%` threshold perturbation
  `p_onset_sigma_ms` / `p_offset_sigma_ms`, separate P-wave existence confidence from boundary stability.
- Long PR can rescue earlier P onset.
- If the geometric relationship of P/QRS is unreasonable, discard P or only keep the information that cannot be used for PR.
- P candidate context calculates template similarity, multilead support, PR consistency and PP consistency for each original candidate. If a short PR candidate has weak template and PP timing at the same time and is only supported by chest lead/non-limb lead, even if multiple leads are misdetected at the same time, it will be suppressed as a noise-like P candidate to avoid the final PR report from covering up raw P onset/offset errors.
- representative P axis uses `p_area` combined with `p_amp_mv` symbol to get signed P area, and combined with P candidate context to make limb-lead polarity consensus. If high-context P waves such as I/II/aVF form a stable atrial vector, aVR/aVL/III contributors with low confidence/low context and conflicting projection directions will be excluded to avoid QRS front Na/Q trough or baseline offset from flipping the P axis to the reverse direction.
- If the overall P vector is obviously inverted, but the PR support is in a stable sinus-like range, and I/II/aVF/aVR give enough inversion votes, the entire group of P-axis contributors will be flipped back to fix the polarity problem of "the real P is suppressed by the negative valley before QRS". Negative P vectors that are truly supported by multiple high-context leads but have no evidence of sinus-like PR will not be forced back.
- P axis also requires minimum frontal support: if there are less than 2 limb leads, the axis will not be reported; when there are only 2 leads, there must be stable PR support, otherwise N/A will be reported. This avoids hard-calculating false axes such as 180 degrees with single-lead or dual-lead noise without PR support.

paced beat P wave path:

- Search for retrograde P after QRS offset `40-250 ms`.
- The amplitude needs to be about `>=0.03 mV`.
- Width required: `<=120 ms`.
- If it overlaps with QRS/T, discard it.

<a id="85-t-波和-t-end"></a>
### 8.5 T-wave and T-end

T-wave search starts after QRS:

- paced beat: at least QRS offset followed by `40 ms`.
- Width QRS: QRS offset rear `40 ms`.
- Narrow QRS: QRS offset rear `20 ms`.
- At the same time, there are protections such as R+40/R+80 ms to avoid falling on the QRS tail/J wave.

`detect_t_wave()` will combine:

- Lead expected polarity, e.g. I, II, aVF, V5, V6 expected positive.
- Multi-lead cluster polarity.
- ST-T confusion mark.
- Significant components segmented by baseline crossing in the T search window; only when the component area reaches `160 uV*ms` will it be regarded as T/T' morphological evidence.
- The significant component supported by area and width will feed back the T peak selection, suppressing the narrow early ST spike to seize the real T peak.
- If there is a reverse, baseline-crossing, significant area T' component after T peak, the initial T offset can be extended to the end of the last significant component, and `t_prime_peak` is passed into the 12SL-style profile.

T-end preferentially uses the chord/geometric method:

- Draw a chord from T peak to the search end point.
- Take the maximum chord distance position in the polar direction as T-end.
- Use threshold fallback for low amplitude or short arms.

Extra protection:

- Biphasic/notch T will try to extend to the second component.
- If there is a significant U wave, the T-end can clip the T-U nadir.
- representative prior only allows extension of T-end under reasonable circumstances to avoid incorrect shortening of QT.
- If QT `<200 ms`, T-end is discarded.
- During the global QT aggregation stage, if the multi-lead cluster of reliable raw QT is stable and significantly later than the consensus QT, it means that the consensus T-end may be premature, and the algorithm will use late raw QT cluster rescue. This is not a special judgment based on record, but requires normal QRS, regular RR, sufficient lead support, T visible and sufficient T-end confidence.

T-end raw repair second pass:

- After completing the initial T peak/T-end measurement for each lead × beat, first use reliable QT leads to establish beat-level T-end measurement consensus.
- Before general consensus repair, run a more conservative dual-method raw rescue: only process raw `t.offset` with obvious early truncation, low confidence or fallback/risky, and require three T-end endpoints of chord, slope-return and tangent spread `<=30 ms`.
- dual rescue will record independent provenance: `t_offset_dual_original_index`, `t_offset_dual_rescued_index`, `t_offset_dual_consensus_index`, `t_offset_dual_support`, `t_offset_dual_chord_index`, `t_offset_dual_slope_index`, `t_offset_dual_tangent_index` and `t_offset_dual_method_spread_ms`.
- For raw `t.offset` that is significantly earlier or later than the consensus and has low confidence or has `t_end_fallback` / `st_t_confusion` / `beat_unreliable` evidence, it is allowed to re-find the T-end on the same lead signal.
- Recalculate `qt_ms`, `jt_ms`, `t_dur_ms`, `t_area`, `t_signed_area` and `tpe_ms` immediately after correction.
- Real long T with high confidence and reasonable form will not be forcibly pulled back to consensus; if the repair candidate will cause QT `<280 ms`, cross the next QRS guard, or the dual-method divergence is too large, the repair will be refused.
- This pass does not change QRS, nor does it change the QT dispersion/report comparison caliber.
- T-end dual-method raw rescue and T-end consensus raw repair are run after QRS offset raw repair and ST/J raw remeasurement, but before final `_apply_multilead_consensus()`.

<a id="86-每搏每导联输出"></a>
### 8.6 Beat per lead output

`LeadBeatFeatures` will save:

- `p/qrs/t` Category III `WaveBounds(onset, peak, offset)`
- `pr_ms/qrs_ms/qt_ms/jt_ms`
- `p_amp_mv/q_amp_mv/r_amp_mv/s_amp_mv/t_amp_mv`
- `qrs_area/qrs_signed_area`
- `st_on_mv/st_mid_mv/st_80ms_mv`
- Q wave duration/area/ratio, initial QRS area/net
- R'/S' amplitude and duration, QRS notch/peak count
- VAT, P/T duration, P/T area, P terminal components, PTF-V1
- Tpe, ST morphology, fragmented QRS score, U-wave flag
- 12SL-style per-beat ST/QRS/T profile field
- beat-level noise/baseline/template correlation/reliability
- P/QRS/QT confidence and flags

<a id="87-多导联-consensus"></a>
### 8.7 Multi-lead consensus

`_apply_multilead_consensus()` Fusion of reliable lead boundaries in beats:

| Global Boundaries | Source |
| --- | --- |
| QRS onset | reliable_for_qrs lead P10; a more central onset may be used when the RR is long and the onset is spread out. |
| QRS offset | reliable_for_qrs The terminal offset of the lead, measurement usually uses P75; median can be used when paced or low support. |
| QRS wide offset | Use P90 for classification and P75 for paced. |
| P onset | Physiological onset cluster; take the protected second earliest onset when there is sufficient evidence of local SNR/boundary stability, otherwise use cluster median/P10 fallback. |
| P offset | Offset cluster supported by reliable leads; take protected second-night offset when diagnostic evidence is sufficient, and retain paired-duration guard. |
| T end | reliable_for_qt Median of the lead. |

P onset/offset uses joint geometric fallback: if the second earliest onset causes all supported offsets
If the relaxed P-duration constraint is violated, the beat falls back to the onset cluster median instead of being lost.
Complete P bound.

Late QRS terminal offset is protected:

- If some leads are late offset like ST/T tail and do not have enough terminal morphology support, they will be marked as `classification_only` and will not enter measurement QRS.
- If supported enough, late offset can be used for measurement or wide classification.
- QRS offset raw repair will be run before the final `_apply_multilead_consensus()`, so that the obviously wrong lead×beat raw offset will be corrected before entering consensus/report.

Consensus writes back each beat/lead:

- `qrs_consensus_ms`
- `qrs_wide_ms`
- `pr_consensus_ms`
- `qt_consensus_ms`

QT also has per-lead and global sanity: if QTcB `>590 ms` is implied, the related T-end will be filtered or the global T-end will be empty.

<a id="9-representative-lead-特征"></a>
## 9. Representative lead characteristics

Implementation location: `features.py::build_representative_lead_features`

At this stage, selected measurement beats are summarized into representative parameters of each lead:

- Priority is given to the measurement of measurement representative beat.
- fallback uses the median/mean/mode of the selected beat pool.
- Summarize beat-to-beat variance, for example `pr_ms_sd/qrs_ms_sd/qt_ms_sd/jt_ms_sd/st_on_mv_sd/t_amp_mv_sd`.
- Write reliability, confidence, and measurement sources for each lead.
- If P-wave support is insufficient, `pr_ms/pr_consensus_ms` will be left blank and `p_measurement_suppressed=low_p_support` will be marked.
- Perform precordial reversal testing based on V1-V6 R wave progression.

<a id="10-group-和-global-特征"></a>
## 10. Group and global features

Implementation location: `features.py`

### 10.1 group features

`compute_group_features()` Statistics by beat group:

- member count and percentage
- longest run
- mean RR/PR/QRS/QT
-mean ventricular rate
- `dominant_group`
- `wide_qrs`

<a id="102-global-hr-和-pr"></a>
### 10.2 global HR and PR

HR uses R-R median:

```text
HR = 60000 / median(RR_ms)
```

PR initial global PR median from representative leads; if atrial measurement invalid, such as AF/AFL or P support is insufficient, it can be left blank.

`api.py` and physiological PR core rescue:

- Extract core values from beat-to-beat PR.
- If the rhythm is allowed, QRS is narrow, RR rules and the PR core is at `120-240 ms`, obviously abnormal or missing global PRs will be replaced.

### 10.3 global QRS

global QRS takes priority from consensus measurement QRS:

- `qrs_consensus_ms` for measurement
- wide classification can refer to `qrs_wide_ms`
- If raw/consensus is inconsistent under pacing or pacing-like context, `api.py` will apply multiple QRS override paths to prevent paced or wide QRS from being underestimated or over-stretched by ST/T tail.

<a id="104-global-qt-和-qtc"></a>
### 10.4 global QT and QTc

QT is the most complex aggregation path in the current implementation.

Reliable QT leads require:

- `reliable_for_qt`
- T wave visible, default `|T_amp| >= 0.05 mV`
- QT beat-to-beat SD `<25 ms`
- T-end confidence `>0.20`
- QT is within physiological range

If the strict path has no leads, the rescue pass selects the best lead:

- QT SD `<30 ms`
- confidence `>0.10`

Wide QRS/QT path during pacing Use JT Reasonability:

```text
JT = QT - QRS
允许范围约 160-550 ms
```

QT aggregation priority is roughly:

1. Consensus QT median of reliable leads.
2. Wide QRS raw QT rescue.
3. raw QT core rescue.
4. Reliable raw QT cluster rescue.
5. Late raw QT cluster rescue: Under normal QRS/regular consensus RR, if the stable raw QT cluster is at least about `20 ms` later than QT, use raw cluster to prevent short QT outlier.
6. split raw QT rescue.
7. The amplitude-weighted median of reliable leads.
8. global reliable lead fallback.
9. irregular beat QT core.
10. raw QT late core rescue.
11. preferred lead fallback: `II, V5, V4, V6, III, aVF`.
12. low-support late raw QT cluster rescue: When the `low_qt_support` path can only give a short QT, but the narrow QRS, RR rules, and multiple reliable/global lead raw QTs form a tight late cluster, use the raw cluster to correct the global QT; this path does not write back each lead raw QT, so it will not forcefully flatten the QT dispersion.

QT path decision will mark:

- `normal_qrs`
- `wide_qrs_jt`
- `paced_jt`
- `low_qt_support`

Final output:

- `qt_ms`
- `qtc_bazett_ms = QT / sqrt(RR_sec)`
- `qtc_fridericia_ms = QT / cbrt(RR_sec)`
- `qt_dispersion_ms`
- `qt_source`
- `qt_used_leads`
- `qt_reliability`
- `qt_path`
- per-lead consensus vs independent comparison

If `qt_reliability == low_confidence` or there is no reliable QT lead, the interpretation layer will inhibit QTc prolongation interpretation.

### 10.5 Axis

The frontal axis uses signed area or amplitude estimates of the six limb leads:

- P axis: P signed area/P amplitude, P is required to be reliable.
- QRS axis: QRS net area/amplitude, which can be corrected with peak-net or initial-core QRS axis if necessary.
- T axis: T signed area/amp, multi-layered by ST-corrected, cluster, suppression and fallback.
- ST axis: ST mid value.

T axis additional protection:

- Width QRS, pacing, T amplitude is too low, polarity conflict, ST-T confusion, RR may be left blank when there is obvious irregularity.
- If I/II/aVF form a strong and co-directional limb-seed T consensus, values ​​in aVR/aVL/III that obviously conflict with the seed projection and lack sufficient confidence support will be excluded to avoid misdetection of a single limb lead polarity and bias the T axis.
- ST-corrected T-axis only uses ST samples with sufficient ST confidence; when QRS offset confidence, QRS offset repair or ST samples are unstable, STJ/STM/STE still outputs, but does not dominate the T-axis.
- If low QT support causes regular T axis to be suppressed, but QRS is narrow, RR rules, multiple limb leads have stable T amplitude/signed area and the axis is not extreme, then low-support T-axis coverage fallback can be used. After the 12SL-style profile writes back STJ/STM/STE, special T and ST confidence, the API layer will also do a narrow range T-axis backfill, only filling in the original N/A stable low support T-axis. This fallback/backfill only restores records with "morphological evidence is stable but QT confidence gate is too conservative" and is not used for wide QRS/ST plateau or low amplitude unstable T's.
- `t_axis_reliable` requires at least 2 limb leads `|T_amp| > 0.15 mV`.

The axis summary of LUDB comparison preserves both signed raw diff and circular absolute diff:

- `diff_p_axis/diff_qrs_axis/diff_t_axis` is still the direct difference of algorithm - ground truth to see the direction.
- `abs_p_axis_circular_diff/abs_qrs_axis_circular_diff/abs_t_axis_circular_diff` uses the circular angle difference to avoid the big mistake of `176°` vs `-175°` being miscalculated as `351°+`.

<a id="11-atrialafafl-和-rhythm-availability"></a>
## 11. Atrial, AF/AFL and rhythm availability

Implementation location: `atrial.py`, `rhythm_rules.py`, `api.py`

### 11.1 atrial event extraction

`extract_atrial_events()` Sources include:

- P peak per lead, requires `p_confidence >=0.30` and lead `reliable_for_p`.
- inter-beat atrial scan.
- If raw ECG is available, prioritize composite raw P detection; otherwise fallback to coarse per-lead scan.

Candidates are clustered by time, deduplicated, and associated with the nearest QRS:

- `conducted`
- `blocked`
- `retrograde`
- `unknown`

<a id="112-qrst-residual-与-afafl"></a>
### 11.2 QRST residual and AF/AFL

`build_qrst_subtracted_residual()` builds QRST window residual summary:

- Try template subtraction first, but it only serves AF/AFL residual and is not considered
  Universal solution for P-on-T. QRS keeps R aligned; ST-T translates by beat-by-beat derivative correlation and
  Huber-style iteratively reweighted fit amplitude/bias; QRS→ST-T and uses cosine gradients at both ends of the template window.
- Only template correlation `>=0.85`, the number of beats is sufficient, and the residual RMS ratio is qualified.
  And only when the second-order differential high-frequency energy ratio is `<=1.25`, it is marked validated QRST subtraction.
- When the median RR is `<520 ms`, the template support window may invade the next P area, and the continuous QRST
  The residual path is not available directly.
- Otherwise the scaffold summary is still exposed for rule input use.

`_classify_af_afl()`：

- When RR CV `>=0.15` and organized P ratio `<0.25` and residual repetitiveness are not strong, it tends to be probable AF.
- flutter uses residual dominant cycle `120-400 ms` and spectral evidence.
- mark probable flutter when flutter confidence `>=0.60`, and avoid judging AF at the same time.

### 11.3 rhythm rules

`rhythm_rules.py` provides:

- pacing context: continuous/intermittent/ventricular/atrial/dual chamber pacing.
- Pacing failure: When there is no QRS in `160 ms` after the spike, it is suspected of capture failure.
- preexcitation/WPW: short PR, short PR segment, delta leads, QRS `>=100 ms`, initial QRS axis.
- pauses and secondary AV block: RR pause, atrial events per RR, PR series.
- post-pause escape/interpolated beat candidates.
- measurement availability.

### 11.4 measurement availability

The following scenarios will make atrial rhythm, PR, P-axis and other measurements no longer suitable for downstream rhythm interpretation:

-continuous pacing
- wide QRS pacing-like context
-probable AF
-probable flutter
- complete AV block
- AV dissociation
- atrial measurements unavailable

`api.py::_apply_measurement_availability_to_representatives()` will leave the global PR and representative PR fields blank in a strong unavailability situation.

<a id="12-12sl-style-并行-measurement-profile"></a>
## 12. 12SL-style parallel measurement profile

Implementation location: `twelve_sl.py`

This module generates parallel profiles that do not directly cover the native DXL-style main measurements.

Calculated per beat/lead:

- STJ, STM, STE
- STM offset = RR / 16
- STE offset = RR / 8
- ST/T amplitude reference = QRS onset voltage
- QRS area, signed area, balance, reflection
-minimum ST
- T/T': T' is inferred from the significant baseline-crossing component after T peak, and the significant area threshold is `160 uV*ms`
- ST confidence: Low QRS offset confidence, QRS offset repair, STJ/STM/STE instability or missing ST sample will reduce `twelve_sl_st_confidence` and record `twelve_sl_st_confidence_reason`.
- special T amplitude: The basic rule is `min(T, T - STE)`; but when ST confidence is low, STE does not participate in the special T branch to avoid ST tail/QRS offset errors from contaminating T morphology. Significant negative T' overrides special T; small negative T' is ignored when positive T is at least 4 times it and `|T'| < 70 uV`; T offset voltage branches are also compared when negative T and no T' are present.
- QRS significance, threshold `160 uV*ms`

record-level profiles include:

- first-last QRS heart rate
- multilead global fiducials: P/QRS/T offset median summary relative to R
- constants and profile version

The export layer provides three sets of measurement profiles:

- `native`
- `12sl`
- `hybrid`

The hybrid strategy is that the ST/QRS/T profile value takes precedence over the 12SL available values, while the global axis and QT dispersion remain native.

<a id="13-clinical-interpretation-层"></a>
## 13. Clinical interpretation layer

Implementation location: `interpret.py`, `mi.py`, `pediatric_rules.py`, `statement_engine.py`

The interpretation layer does not reprocess the original waveform, but consumes:

- `global_features`
- `representative_leads`
- `beats`
- `beat_features`
- `metadata["rhythm_analysis"]`

<a id="131-判读流程"></a>
### 13.1 Interpretation process

The code is organized hierarchically similar to clinical chart reading:

1. Technical quality and lead reverse connection.
2. Rhythm and rate.
3. Axis and intervals.
4. P/QRS/T/ST morphology.
5. Anatomical positioning, MI, hypertrophy, reciprocal changes.

### 13.2 rhythm/rate

- HR `<45 bpm`: `extreme_bradycardia`
- HR `<50 bpm`: `bradycardia`
- HR `<100 bpm`: `normal`
- HR `>=100 bpm`: `tachycardia`
- RR CV `>0.15`: `irregular`
- RR CV `>0.08`: `mildly_irregular`
- Probable AF when irregular and P confidence is low.

### 13.3 axis/conduction/QTc

- QRS axis: normal `[-30, 90]`, LAD/LAFB/RAD/LPFB/ERAD layered.
- T axis: normal `[-10, 100]`.
- PR: short `<120 ms`; one-degree AVB threshold using age and HR dynamic tables.
- QRS:
  - `<100 ms`: normal
  - `100-110 ms`: borderline IVCD
  - `110-120 ms`: nonspecific IVCD
  - `>=120 ms`: BBB/IVCD, and judge RBBB/LBBB based on V1 R', lateral S, V1/V6/I amplitude.
- QTc:
  - `<340 ms`: short
  - `>465 ms`: borderline prolonged
  - `>485 ms`: prolonged
  - `>520 ms`: significantly prolonged
  - BBB, RVH/LVH, etc. can inhibit QTc prolongation statement.

<a id="134-morphology-和解剖定位"></a>
### 13.4 Morphology and Anatomical Positioning

Interpretation layer coverage:

- WPW/preexcitation
- RAE/LAE/BAE, including P duration, P amplitude, V1 terminal component and PTF-V1
- pathological Q waves and territory: inferior/anterior/lateral
- R-wave progression and R/S transition
- ST elevation/depression territories, STEMI-style codes, reciprocal changes
- LVH voltage/score, including Cornell/Sokolow/aVL, etc.
- low voltage
- RVH graded score
- tall T
- dextrocardia
- COPD pattern
- posterior MI and culprit artery evidence
- statement candidates and suppression/bypass

LBBB inhibits judgments related to secondary repolarization:

- ST elevation/depression
-T-wave abnormality
- Q-wave infarct signs
- reciprocal changes

<a id="135-儿童规则"></a>
### 13.5 Children’s Rules

Take the pediatric morphology route at age `0 <= age < 16`:

- Children QRS axis and QRS duration use age binning thresholds.
- pediatric QTc Thresholds vary by age and gender.
- RVH/LVH/LSH/BVH use age bin voltage thresholds from `pediatric_rules.py`.
- RBBB/LBBB can bypass part of the hypertrophy interpretation.
- Management of pericarditis and early repolarization within specific age ranges.

<a id="14-输出结构"></a>
## 14. Output structure

`*_features.json` saved in the current batch and demo uses `export.to_dict()`, and the top level contains:

```text
fs
quality
beats
beat_features
representative_leads
groups
global_features
metadata
interpretation
rhythm_inputs
morphology_inputs
statement_engine
clinical_interpretation
reference_metadata
```

<a id="141-主要分区"></a>
### 14.1 Primary Partition

| Partition | Content |
| --- | --- |
| `quality` | Quality of each lead, flags, Q0-Q3, reliability of each waveform. |
| `beats` | R index, paced, group id, and front and rear RR of each beat. |
| `beat_features` | P/QRS/T bounds, intervals, amplitudes, ST, morphology, confidence, flags per beat x lead. |
| `representative_leads` | Per-lead representative parameters and variance. |
| `groups` | beat group summary. |
| `global_features` | HR, PR, QRS, QT/QTc, axis, QT provenance, pacing status, etc. |
| `metadata` | Input/internal sampling rate, QRS detector, record quality, lead reversal, pacing, measurement group, rhythm analysis. |
| `interpretation` | Clinically derived categories and flags. |
| `rhythm_inputs` | Normalized input for rhythm rule layer. |
| `morphology_inputs` | Normalized input of morphology rule layer, including native/12SL/hybrid profile. |
| `statement_engine` | statement candidate, final/suppressed/bypassed results. |
| `clinical_interpretation` | Current authoritative clinical rule results, coverage, suppression and audit evidence. |
| `reference_metadata` | The old interpretation layer remains and its reference-only/supersession status. |

There is also `export.build_structured_payload(features)` that can generate stable partitioned payload:

```text
record / signal / quality / beats / groups / global / pacing /
rhythm_inputs / morphology_inputs / statement_engine / provenance /
clinical_interpretation / reference_metadata /
schema_version
```

The current structured schema is `ecgfeat_structured_payload.v3`. v3 no longer includes Glasgow
Explain the block.

NOTE: The current `to_dict()` exported sample JSON does not automatically include a top-level field named `structured_payload` unless the caller explicitly calls `build_structured_payload()`.

<a id="15-批处理流程"></a>
## 15. Batch processing process

Implementation location: `batch_extract_ecgfeat.py`

Batch input support:

- Single `[12, n_points]`
- Single `[n_points, 12]`
- batch `[n_samples, 12, n_points]`
- batch `[n_samples, n_points, 12]`

Unified and standardized to:

```text
[n_samples, 12, n_points]
```

Each sample output:

- `{sample_id}_features.json`
- `{sample_id}_features.pt`
- `{sample_id}_report.txt`
- Optional `{sample_id}_ecg_annotated.png`
- `manifest.json`

<a id="16-关键设计约束和已知限制"></a>
## 16. Key design constraints and known limitations

1. This project is a DXL-inspired research scaffold and is not regulatory-verified diagnostic software.
2. Quality, pacing, AF/AFL, width QRS, LBBB and other scenarios will trigger multiple suppression/rescues, and you cannot just look at a single raw interval.
3. `beat_features` is the beat-by-lead original measurement perspective, `representative_leads` is the measurement group summary, and `global_features` is the global aggregation perspective for downstream interpretation.
4. 12SL profile is a parallel output surface and is not equivalent to replacing native measurement.
5. QRST subtraction is validated only when it meets template validation; otherwise it is just scaffold residual summary.
6. The dedicated delineation of paced beat is still engineering implementation and protection logic, and complex pacing types still require more verification.
7. The child threshold depends on age/sex; missing or illegal age will default to the adult route.
8. Lead reversal detection is a heuristic that affects axis, precordial progression and interpretation, but is not a final medical confirmation.

<a id="161-已退役的-glasgow-解释层"></a>
## 16.1 Retired Glasgow interpretation layer

The Glasgow rule framework has been removed from the current runtime interpretation chain:

- `ECGFeatureExtractor.extract()` Glasgow analysis is no longer performed or cached;
- `to_dict()` and structured v3 no longer export top-level `glasgow`;
- Clinical Conflict Resolution, Text/Graphic Reporting and MedGemma Context No Longer Consumption Glasgow Conclusions;
- The exporter will clean up `metadata.glasgow_analysis` in old objects;
- History Glasgow appendix is filtered before old reports enter MedGemma.

Historical `glasgow_rules/` files are temporarily retained for old experiments and results traceability, but do not belong to the current
Interpretation layer. `glasgow_measurements` used in delineation is the underlying numerical measurement profile.
It is not a diagnostic explanation, so it is not deleted this time to avoid changing the basic measurement of P/QRS/T.

<a id="17-推荐阅读顺序"></a>
## 17. Recommended reading order

If you need to go deeper from the code, it is recommended to read in the following order:

1.`feature_extraction/ecgfeat/api.py::ECGFeatureExtractor.extract`
2. `preprocess.py`, `quality.py`, `qrs.py`
3. `grouping.py`, `representative.py`
4. `delineate.py`
5. `features.py`
6. `atrial.py`, `rhythm_rules.py`
7. `twelve_sl.py`
8. `interpret.py`
9. `export.py`

<!-- i18n-nav -->
[中文](ecgfeat_neurokit_hybrid_improvement.md) | [English](ecgfeat_neurokit_hybrid_improvement.en.md) | [日本語](ecgfeat_neurokit_hybrid_improvement.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecgfeat-吸收-neurokit2-优点的波形检测改进方案"></a>
# ecgfeat Waveform detection improvement scheme that absorbs the advantages of NeuroKit2

> Status: R/P/T/S positioning and robust ST/J measurement bypass implemented, original measurement and interpretation layers remain unchanged
> Dataset: LUDB 1.0.1, 500 Hz, 12 leads, 10 seconds
> Scope of application: R, P, T, S wave positioning, as well as ST/J point, amplitude, trend and concave and convex morphology measurement
> Core constraints: The new positioning results are first used as bypass fields and do not directly cover the existing feature calculation and interpretation layer inputs.

<a id="1-目标"></a>
## 1. Goal

The advantages of ecgfeat are multi-lead heartbeat consensus, complete QRS/P/T boundaries, representative waveforms and clinical
feature layer. Some algorithms of NeuroKit2 are more accurate in local peak positioning of single leads, especially
R peak and prominence based T peak localization.

This solution is not to replace ecgfeat with NeuroKit2, but to combine the advantages of both:

1. ecgfeat continues to determine whether the heartbeat exists, beat grouping, QRS approximate range and final feature level;
2. Use positive and negative bipolar prominence to refine peaks within a restricted physiological window for each lead;
3. Save "morphological peak position" and "position for amplitude measurement" separately;
4. Use multi-lead time consensus to filter single-lead noise;
5. Do not change intervals, amplitudes, representative leads, and interpretation layer results until fully verified.

<a id="2-当前对比结论"></a>
## 2. Current comparison conclusion

<a id="21-r-峰"></a>
### 2.1 R peak

LUDB First 10 records, all 12 leads:

| Method or marker source | QRS F1 | R peak MAE | Median absolute error | P95 absolute error |
|---|---:|---:|---:|---:|
| ecgfeat Original multi-lead global R | 0.8852 | 13.42 ms | 6 ms | 44 ms |
| ecgfeat original lead-by-lead fiducial | 0.8917 | 11.83 ms | 4 ms | 46 ms |
| ecgfeat mixed prominence | **0.8917** | **9.71 ms** | **2 ms** | 50 ms |
| NeuroKit2 | About 0.819 | **4.57 ms** | 2 ms | 14 ms |
| BioSPPy | About 0.873 | 3.9 ms | — | — |

Conclusion:

- ecgfeat multi-lead strategy has higher sensitivity, but the shared global position is not suitable for directly evaluating the R peak of each lead;
- NeuroKit2’s lead-by-lead local correction has better R-peak positioning accuracy;
- The implemented hybrid scheme reduces the ecgfeat R peak MAE without changing the number of beats.
  from 13.42 ms to 9.71 ms;
- A small number of negative or complex QRS still have long-tail errors, and you cannot just look at the average MAE.

<a id="22-t-波"></a>
### 2.2 T wave

| Method | T wave F1 | T peak MAE | T onset MAE | T offset MAE |
|---|---:|---:|---:|---:|
| ecgfeat | **0.8324** | 8.68 ms | 30.81 ms | 16.60 ms |
| ecgfeat Mixed T prominence | **0.8333** | **7.25 ms** | 30.91 ms | 16.79 ms |
| NeuroKit2 DWT | 0.6925 | 26.19 ms | 53.97 ms | 32.12 ms |
| NeuroKit2 prominence | 0.8076 | **8.13 ms** | 25.49 ms | 20.92 ms |

Conclusion:

- The NeuroKit2 T wave in the original comparison used DWT, which underestimated the ability of NeuroKit2 for abnormal T waves;
- prominence simultaneously searches for local maxima and local minima, and can handle negative T waves;
- The T wave detection F1 of ecgfeat is higher, and the T peak positioning of NeuroKit2 prominence is slightly accurate;
- Implemented polarity-aware prominence bypass reduces ecgfeat T-peak MAE from
  8.68 ms dropped to 7.25 ms, P95 dropped from 78.0 ms to 58.6 ms;
- T onset/offset still uses native bounds, not yet covered by bypass.

<a id="23-p-波"></a>
### 2.3 P wave

| NeuroKit2 Tracing method | P wave F1 | P peak MAE | Remarks |
|---|---:|---:|---|
| DWT | **0.7110** | 30.93 ms | F1 is the best among current working methods |
| prominence | 0.707 | **17.4 ms** | 1/120 lead run failed |
| peak | 0.677 | 18.0 ms | 13/120 lead run failed |

ecgfeat has a P-wave F1 of 0.7469 on the same first 10 records.

Conclusion:

- The T-wave prominence branch of NeuroKit2 uses both positive and negative peaks;
- But its P-wave prominence branch only searches for local maxima;
- `peak`, DWT, and CWT also have positive morphological preferences;
- Therefore, replacing the built-in tracing method of NeuroKit2 cannot completely solve inversion, biphasic, retrograde,
  Problems with low amplitude, missing, or multiple P waves;
- The original NeuroKit2 P-wave result of approximately 0.711 is not as significantly underestimated as the T-wave.

<a id="24-s-波"></a>
### 2.4 S wave

LUDB only provides the starting point, representative peak and end point of P, the entire QRS and T wave, and does not provide independent
Q peak or S peak expert annotation. Therefore, LUDB cannot be used to directly calculate S peak F1/MAE.
Nor can QRS offset be regarded as the true S-peak value.

The current algorithm semantics are different:

| Content | ecgfeat | NeuroKit2 |
|---|---|---|
| Search range | This lead QRS onset–offset | R back beat interval; prominence constrained to 180 ms after Q |
| Main selection rules | Deepest deflection after positive R | First negative local minimum after R |
| S amplitude | `s_amp_mv` | Not output by default |
| S peak position | Not saved independently | `ECG_S_Peaks` |
| R′/S′/Notch | Partially measured | No explicit classification |
| Multi-lead constraints | QRS range is affected by multi-lead results | Single lead independent |

These two definitions are each more suitable for different goals:

- "First effective negative deflection" is closer to the morphological S-peak;
- "The deepest negative deflection within QRS" is more suitable for clinical voltage measurement;
- In RSR′, QS, qR, wide QRS and pacing QRS, the two cannot be mixed into one position.

<a id="3-总体架构"></a>
## 3. Overall architecture

It is recommended to place enhanced positioning after the main tracing is completed and before the calculation of global features and interpretation layers:

```text
原始 12 导联 ECG
        │
        ▼
ecgfeat 预处理与多导联 QRS 检测
        │
        ▼
现有逐导联 P/QRS/T 描记与特征计算
        │
        ├── 原字段：继续供间期、幅值、代表波形、解释层使用
        │
        ▼
混合局部定位旁路
        ├── R：正向 prominence，必要时负向回退
        ├── T：正负双极性 prominence
        ├── P：正负双极性、多候选、允许缺失
        ├── S：形态学 S 与幅值 S 分离
        └── ST：PR 基线、J 点局部稳定、多导联共识和窗口中位数
        │
        ▼
独立定位、极性、形态和置信度字段
        │
        ▼
可视化与独立评价
```

Only when the bypass results pass acceptance on both the independent validation set and the test set will the specific fields be considered
Enter the measurement layer item by item; you cannot replace all original results at once.

<a id="4-r-峰混合定位"></a>
## 4. R peak mixing positioning

<a id="41-已实现"></a>
### 4.1 Implemented

Implementation location:

- [`r_localization.py`](../feature_extraction/ecgfeat/r_localization.py)
- [`models.py`](../feature_extraction/ecgfeat/models.py)
- [`api.py`](../feature_extraction/ecgfeat/api.py)

Current rules:

1. Multi-lead `global_r` is only responsible for determining whether a heartbeat exists;
2. Search each lead within `global_r ± 90 ms`;
3. Then intersect with the original lead-by-lead fiducial `±35 ms`;
4. Then limit it to the vicinity of the existing QRS onset/offset;
5. Select the positive local peak with the largest prominence;
6. Only use negative peaks when there are no positive local peaks;
7. If there is no valid candidate, fall back to the original `qrs.peak`.

Independent fields:

```text
r_localized_index
r_localization_method
r_localization_confidence
r_localization_prominence_mv
r_localization_polarity
```

<a id="42-后续改进"></a>
### 4.2 Subsequent improvements

The current P95 error is still 50 ms, which mainly needs to be processed:

- Negative dominance QRS in leads aVR, V1 and part III;
- QS-type and rS-type complex waves;
- Bundle branch block with R′ higher than R;
- Boundary beats and low quality leads;
- LUDB represents a case where the peak definition is inconsistent with the clinical positive R definition.

It is recommended to add QRS morphological prior:

```text
positive-dominant
negative-dominant
QS
rS
RS
RSR-prime
paced
uncertain
```

Different candidate scores are used for different modalities to avoid fixing and prioritizing positive peaks in all leads.

<a id="5-t-波混合定位与异常形态"></a>
## 5. T wave hybrid positioning and abnormal morphology

<a id="51-借鉴-neurokit2-的部分"></a>
### 5.1 Learn from NeuroKit2

Valid designs for NeuroKit2 prominence include:

1. Search for local maxima and local minima at the same time;
2. Calculate prominence for positive and negative peaks respectively;
3. If the T peak is a negative valley, invert the signal first and then calculate the onset/offset;
4. Use S-wave and RR intervals to limit T-wave candidate ranges.

<a id="52-已实现的旁路"></a>
### 5.2 Implemented Bypass

Add independent results:

```text
t_localized_index
t_localized_onset
t_localized_offset
t_localization_confidence
t_localization_prominence_mv
t_localization_polarity
t_localization_morphology
t_localization_secondary_index
```

Among them `t_localization_morphology` supports:

```text
positive
negative
biphasic-positive-negative
biphasic-negative-positive
uncertain
```

The candidate window should be determined by the following conditions:

- no earlier than the reliable QRS offset/ST segment;
- Do not exceed the safety margin before the next R peak;
- Dynamically adjust based on current RR instead of just using a fixed time;
- Priority falls within the existing ecgfeat T-wave range or its limited extension.

The current candidate scores include prominence, noise and dynamic range; it is still recommended to add in the future:

```text
score =
    prominence_score
  + timing_score
  + width_score
  + beat_template_score
  + multilead_time_consensus
  - noise_penalty
  - u_wave_penalty
```

Biphasic T waves cannot be preserved only with the largest lobe. It is necessary to save the main peak, secondary peak, two-petal sequence and both
Relative amplitude, avoid treating the T2 or U wave as a single T peak.

<a id="6-p-波双极性和异常节律处理"></a>
## 6. P wave bipolarity and abnormal rhythm processing

<a id="61-不能直接复制-neurokit2-p-波实现"></a>
### 6.1 Cannot directly copy NeuroKit2 P wave implementation

The P wave of NeuroKit2 prominence only uses positive local maxima and cannot be processed reliably:

- Negative P wave in leads such as aVR;
- P wave inversion due to ectopic atrial rhythm;
- Biphasic P wave;
- Low amplitude P wave;
- Absence of P wave in atrial fibrillation;
- Multiple atrial waves in atrial flutter or atrioventricular block;
- Retrograde P wave after QRS;
- Long PRs beyond the fixed PR search range.

<a id="62-已实现的安全旁路"></a>
### 6.2 Implemented Safe Bypass

New fields:

```text
p_localized_index
p_localized_onset
p_localized_offset
p_localization_confidence
p_localization_prominence_mv
p_localization_polarity
p_localization_morphology
p_localization_candidate_count
p_localization_absent_probability
p_localization_retrograde
```

Current detection logic:

1. Establish a normal P wave window before the current QRS onset;
2. For paced cardiac beats or suspected retrograde conduction, establish a second search window after QRS;
3. Search for positive and negative prominence at the same time;
4. Save significant candidate numbers, allowing clear no-candidate results;
5. When there is a native P peak, only the candidates within 40 ms nearby are analyzed and the position of the native peak is maintained;
6. Keep both lobes for biphasic candidates instead of just the highest positive peak.

The second phase yet to be implemented includes PR/PP consistency, template correlation, multi-lead timing support,
and one-to-one P-QRS hypothesis adjustments in atrial fibrillation/atrial flutter states.

LUDB experiments show that directly covering the original P peak with the maximum prominence significantly increases the peak value
MAE; therefore the current implementation retains the existing native P peaks and only annotates polarity and morphology with bipolar candidates.
Candidates in the absence of native P remain in the bypass field but have not yet entered the formal evaluation or interpretation layer.

Lead polarity alone cannot determine whether it is abnormal. For example, negative P in aVR is often a normal manifestation.
Judgments need to be combined with lead direction, multi-lead P-axis and rhythm context.

<a id="7-s-波位置幅值和形态分离"></a>
## 7. S wave position, amplitude and shape separation

<a id="71-当前-ecgfeat-需要明确的语义"></a>
### 7.1 Currently ecgfeat requires clear semantics

The current `s_amp_mv` is the lowest value after the true forward R in this lead QRS, suitable for voltage
measurement, but there are the following problems:

- The corresponding sample location is not saved;
- The deepest valley in RSR′ is not necessarily the first morphological S;
- A positive `s_amp_mv` may be obtained when there is no negative deflection after R;
- QS-type complexes should not be forced to split into R and S;
- The R split points used by the S magnitude path and the S duration path do not exactly match.

<a id="72-已实现的旁路字段"></a>
### 7.2 Implemented bypass fields

```text
s_peak_index
s_amplitude_index
s_localization_confidence
s_localization_prominence_mv
s_localization_morphology
s_prime_peak_index
qs_nadir_index
```

Field semantics:

- `s_peak_index`: The first negative deflection after R that meets the amplitude, prominence and width requirements;
- `s_amplitude_index`: Deepest negative deflection within QRS for S voltage measurement;
- `s_prime_peak_index`: subsequent negative deflection after R′;
- `qs_nadir_index`: QS lowest point without true forward R;
- The original `s_amp_mv` is currently not modified by bypass; if it is connected to the measurement layer in the future, there will be no negative S
  should be changed to `0` or `None` and cannot be a positive value.

<a id="73-当前检测流程"></a>
### 7.3 Current detection process

1. Use the existing lead-by-lead QRS onset/offset;
2. Use real `r_peak_index` instead of global R or negative dominant fiducial;
3. Find all local minima after R and calculate prominence;
4. Exclude notches that are smaller than the noise threshold;
5. The first valid negative peak is regarded as morphological S;
6. The deepest effective negative peak is used for S amplitude;
7. If R′ appears, mark the negative peak after R′ separately with S′;
8. Mark `no-S` when the baseline is not crossed;
9. Enter the `QS` branch when there is no valid forward R.

<a id="8-stj-点稳健测量"></a>
## 8. ST/J point robust measurement

<a id="81-原算法仍保留的路径"></a>
### 8.1 Paths still retained by the original algorithm

The native ST path continues to use lead-by-lead `qrs.offset` as the J point, and calculates `st_on_mv`,
`st_mid_mv`, `st_80ms_mv` and `st_slope_mv_per_ms`. When point J is equal to
When there is an obvious QRS tail jump in ST40/ST80, the original raw remeasurement pass
Multi-lead QRS offset median resampling for the same beat will be used.

This path remains the input for the current interpretation layer and compatibility exports.

<a id="82-新增的稳健旁路"></a>
### 8.2 New robust bypass

Implementation location:

- [`st_localization.py`](../feature_extraction/ecgfeat/st_localization.py)
- [`models.py`](../feature_extraction/ecgfeat/models.py)
- [`features.py`](../feature_extraction/ecgfeat/features.py)
- [`export.py`](../feature_extraction/ecgfeat/export.py)
- [`api.py`](../feature_extraction/ecgfeat/api.py)

New process:

1. The baseline first takes the real PR segment from `P offset + 8 ms` to `QRS onset - 8 ms`;
2. When the PR segment is insufficient, fall back to the QRS front quiet window and the old R front window in sequence;
3. Establish J-point median consensus using reliable QRS leads for each beat;
4. Each lead still retains the QRS offset of this lead, and searches within the local ±30 ms range.
   The stable point of "QRS high slope transition to ST low slope transition";
5. J, J+40 ms, J+80 ms no longer use a single sample, but use approximately ±6 ms
   window median;
6. Robust linear fit using J+12 ms to J+80 ms without crossing T onset
   Calculate ST slope;
7. The trends are divided into `upsloping`, `horizontal`, and `downsloping`;
8. Second-order amplitude changes are separately divided into `straight`, `concave-up`, and `concave-down`;
9. Values are retained when pacing, QRS is unreliable, T waves overlap prematurely, or the baseline or overall confidence is insufficient.
   But mark the bypass as unreliable.

Main fields:

```text
st_hybrid_j_index
st_hybrid_j_mv
st_hybrid_40ms_mv
st_hybrid_80ms_mv
st_hybrid_mean_mv
st_hybrid_area_mv_ms
st_hybrid_slope_mv_per_ms
st_hybrid_trend
st_hybrid_shape
st_hybrid_baseline_mv
st_hybrid_baseline_source
st_hybrid_baseline_confidence
st_hybrid_j_method
st_hybrid_j_confidence
st_hybrid_consensus_index
st_hybrid_consensus_support
st_hybrid_noise_mv
st_hybrid_reliable
st_hybrid_unreliable_reason
```

Robust ST represents a lead that preferentially aggregates the median of reliable beats in the same group, without preferentially relying on one
medoid heartbeat. Structured export is located at:

```text
morphology_inputs.leads.<lead>.st.hybrid_robust
```

Since LUDB does not have ST elevation/depression or J point expert truth value, it cannot be
Use LUDB to give the ST sensitivity, specificity, or MAE. New bypasses are enabled by default but will not be overridden
The native ST field also does not enter the ischemia, premature repolarization, or pericarditis interpretation rules.

<a id="83-ludb-前-10-条内部验证"></a>
### 8.3 LUDB top 10 internal verifications

A total of 1,188 leads × beats are obtained for all 12 leads:

| Check items | Results |
|---|---:|
| Passed Robust ST Reliability Gating | 1,044 / 1,188 (87.9%) |
| T wave prematurely overlaps J+80 window | 126 |
| This lead J is too different from the multi-lead consensus | 18 |
| True PR segment baseline using P | 759 |
| Fallback to baseline before using QRS | 429 |
| Median absolute offset of point J relative to the original effective anchor | 0 ms |
| J point absolute offset P95 | 10 ms |
| Maximum absolute offset of point J | 26 ms |

Turn `enable_hybrid_st_measurement` on and off for LUDB record 1
`global_features`, final `interpretation`, all native beat fields and all
The native representative field is consistent on a case-by-case basis. The new field itself and corresponding metadata
Of course only exists when the switch is on.

The offset distribution here is only used to check whether the algorithm has uncontrolled J-point jumps and cannot be interpreted as
J-point error relative to the expert's true value.

<a id="9-安全接入策略"></a>
## 9. Security access strategy

The first phase only allows bypass fields to be used for:

- Visualization;
- Independent evaluation of LUDB or other datasets;
- Debugging information;
- Confidence and error case analysis.

Metadata should be clearly documented:

```json
{
  "wave_localization": {
    "r": {"enabled": true, "affects_measurements": false},
    "p": {"enabled": false, "affects_measurements": false},
    "t": {"enabled": false, "affects_measurements": false},
    "s": {"enabled": false, "affects_measurements": false},
    "affects_interpretation": false
  },
  "st_measurement": {
    "enabled": true,
    "method": "robust_pr_baseline_local_settling_multilead_consensus",
    "affects_native_measurements": false,
    "affects_interpretation": false
  }
}
```

At this stage, the following must be ensured to be exactly the same as when the bypass is turned off:

- `global_features`
- `representative_leads`
- `beats` and `groups`
- Original `qrs.peak` and P/QRS/T boundaries
- Heart rate, PR, QRS, QT/QTc
- Axis, voltage and ST-T characteristics
- Final interpretation layer output

<a id="10-公平评价要求"></a>
## 10. Fair evaluation requirements

<a id="101-必须注明完整配置"></a>
### 10.1 The complete configuration must be specified

You cannot just write "NeuroKit2", you should at least indicate:

```text
NeuroKit2 (R detector: neurokit, delineation: dwt)
NeuroKit2 (R detector: neurokit, delineation: prominence)
ecgfeat (QRS marker: global)
ecgfeat (QRS marker: lead-fiducial)
ecgfeat (QRS marker: hybrid-prominence)
```

<a id="102-防止测试集挑选算法"></a>
### 10.2 Prevent test set selection algorithm

Different methods should be selected first on the validation set, then the parameters should be fixed and reported once on the independent test set.
After seeing the test set results, select the best method for each waveform as the only score.

<a id="103-指标"></a>
### 10.3 Indicators

Each waveform reports at least:

- sensitivity, precision, F1;
- Peak bias, MAE, median absolute error, P95;
- onset/offset MAE;
- Number of failed leads;
- Stratified results such as normal, inverted, biphasic, low amplitude, wide QRS, pacing, etc.;
- Lead-by-lead and record-by-record results.

S waves must be annotated using experts with independent S peak definitions. If there is no such truth value, we can only
Internal consistency, morphological cases and manual review results are reported, spurious S-peak F1 cannot be reported.

<a id="11-测试要求"></a>
## 11. Testing requirements

<a id="111-合成波形单元测试"></a>
### 11.1 Synthetic waveform unit test

Cover at least:

- Normal qRS;
- rS, RS, qR;
-QS;
- RSR′, rSr′;
- wide QRS and pacing QRS;
- Orthodromic, inverted and biphasic P/T;
- No P, no S;
- Low amplitude waves, baseline drift and high frequency noise;
- Recording of incomplete heartbeats at the beginning and end.

<a id="112-回归测试"></a>
### 11.2 Regression testing

Every bypass change must be verified:

1. When bypass is turned off, the result is consistent with the historical version;
2. When bypass is turned on, the original serialization results are consistent except for the new fields;
3. The interpretation layer results are consistent field by field;
4. The visualization uses the new position correctly, but the beat count still comes from multi-lead detection;
5. Can safely fall back when there are exceptions or missing candidates.

<a id="12-推荐实施顺序"></a>
## 12. Recommended implementation sequence

1. **R Peak: Completed. ** Continuing with negative dominance and long tail cases.
2. **T wave: bypass completed. ** The peak MAE of the first 10 records of LUDB has been improved and will be processed in the next step.
   U-wave confusion and the biphasic boundary.
3. **P wave: bipolar morphological bypass completed. ** The original peak remains unchanged and will be added in the next step.
   Revalidation of missing P rescue after multi-lead consensus.
4. **S wave: Morphological bypass completed. ** The first S and deepest amplitude point, S′, have been separated.
   QS and no-S; independent S peak artificial ground truth is still required.
5. **ST: Robust measurement bypass completed. ** Added PR baseline, local J settling,
   Multi-lead consensus, multi-point median, trend/bump pattern, and reliability gating; required next step
   Data set with J points and ST diagnostic ground truth.
6. **Interpretation layer access: final implementation. **Only after both the independent test set and regression tests pass,
   Only then is it allowed to replace the original measurement values item by item by specific fields.

<a id="13-当前建议"></a>
## 13. Current recommendations

The most valuable and lowest risk portfolio in the short term is:

```text
心搏存在与分组       → ecgfeat 多导联共识
R 峰显示与评价       → 已实现的逐导联 hybrid-prominence
T 峰位置与极性       → 已实现正负双极性 prominence 旁路
P 波                 → 原生峰不动，已增加双极性形态旁路
S 波电压             → 保留 ecgfeat 的 QRS 内最深负向测量
S 波形态位置         → 已增加第一个有效负峰和 QS/no-S 分类
ST/J 稳健测量        → 使用新旁路做研究导出，原生 ST 继续供解释层使用
最终临床解释         → 暂时完全保持现状
```

This design absorbs the advantages of NeuroKit2 in local prominence positioning while retaining
ecgfeat's structural advantages in multi-lead consistency, complex morphology, clinical measurement and interpretation layers.

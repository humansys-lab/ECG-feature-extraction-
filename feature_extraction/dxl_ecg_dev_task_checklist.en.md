<!-- i18n-nav -->
[中文](dxl_ecg_dev_task_checklist.md) | [English](dxl_ecg_dev_task_checklist.en.md) | [日本語](dxl_ecg_dev_task_checklist.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="dxl-inspired-ecg-边界定位开发任务清单"></a>
# DXL-inspired ECG Boundary Localization Development Task List

> Objective: Upgrade the current library's beat-by-beat, lead-by-lead P/QRS/T boundary localization module from the "heuristic first version" to a complete measurement kernel featuring **multi-lead + grouping + representative beat + geometric T-end + reliability scoring**.  
> The methodological mainline publicly described in the DXL manual is: first perform waveform quality assessment, then waveform recognition, subsequently form representative beat, generate comprehensive measurements, and finally enter the interpretation layer; within this, Group 1 performs outlier exclusion, alignment, and averaging; the difficulty with QT lies in the T-wave end, with global QT using the median of reliable leads.

---

<a id="0-文档使用说明"></a>
## 0. Document Usage Instructions

<a id="范围"></a>
### Scope
This checklist covers the following modules:

- `quality/`
- `qrs/`
- `grouping/`
- `representative/`
- `delineate/`
- `features/qt.py`
- `pacing/`
- `validate/`

<a id="不在本轮范围"></a>
### Out of Scope for This Round
The following are not included in this round:

- Complete interpretive statements
- Full set of clinical rule classifications
- 1:1 replication of DXL private thresholds

<a id="本轮交付定义"></a>
### Delivery Definition for This Round
Upon completion, the system should possess:

1. Multi-lead QRS anchor detection
2. Beat grouping
3. representative beat-driven boundary refinement
4. P/QRS/T single-beat local correction
5. Geometric T-end
6. Reliable lead / confidence system
7. Global QT calculation
8. Basic visualization and validation benchmarks

---

<a id="1-目录与代码结构重构"></a>
## 1. Directory and Code Structure Refactoring

<a id="任务-t001重构-delineation-目录"></a>
### Task T001: Refactor the delineation directory

**Objective**  
Split the existing `delineate.py` into layered modules to facilitate iteration and testing.

**Input**  
Existing:

- `ecgfeat/delineate.py`

**Output**  
New structure:

```text
ecgfeat/delineate/
├── __init__.py
├── rough.py
├── representative_refine.py
├── beat_local_refine.py
├── p_wave.py
├── qrs_bounds.py
├── t_wave.py
├── t_end_geometric.py
├── confidence.py
└── fusion.py
```

**Core Work**
- Split existing functions by responsibility
- Maintain compatible entry points
- Add module-level docstrings

**Acceptance Criteria**
- `from ecgfeat.delineate import delineate_beats` remains functional
- Unit tests cover both old and new API
- Old call chains do not raise errors

**Priority**  
P0

**Dependencies**  
None

---

<a id="任务-t002建立统一边界数据结构"></a>
### Task T002: Establish Unified Boundary Data Structure

**Objective**  
Define a unified schema for rough / refined / fused boundaries to avoid passing raw dicts between functions in the future.

**Input**  
`WaveBounds` within the current `models.py`

**Output**
New dataclass:

```python
@dataclass
class BoundaryEstimate:
    onset: Optional[int]
    peak1: Optional[int]
    peak2: Optional[int]
    offset: Optional[int]
    confidence: float
    method: str
    flags: list[str]
```

**Core Work**
- Add `BoundaryEstimate`
- Add `LeadBoundaryResult`
- Add `BeatBoundaryResult`
- Support three-stage results: raw candidate / refined / fused

**Acceptance Criteria**
- All delineation submodules return a unified structure
- `None` boundary scenarios do not crash
- Support JSON serialization

**Priority**  
P0

**Dependencies**  
T001

---

<a id="2-质量门控与基础可靠性"></a>
## 2. Quality Gating and Basic Reliability

<a id="任务-t003实现-lead-level-质量评分重构"></a>
### Task T003: Implement lead-level quality scoring refactoring

**Objective**  
Ensure that boundary detection no longer assumes equal weight for all leads by default, but instead implements lead quality gating first.

**Input**
- 12-lead ECG
- Original sampling rate
- Current `quality.py`

**Output**
Per lead output:

- `baseline_wander_score`
- `muscle_noise_score`
- `powerline_score`
- `flatline_score`
- `clipping_score`
- `lead_reliable_for_p`
- `lead_reliable_for_qrs`
- `lead_reliable_for_t`
- `lead_reliable_for_qt`

**Core Algorithms**
- Baseline wander: Low-frequency energy ratio + beat baseline shift
- Muscle noise: High-frequency energy outside QRS
- Powerline: 50/60 Hz narrowband peak
- Flatline: Proportion of near-zero derivatives
- Clipping: Saturation/repeated sample detection

**Implementation Key Points**
- Add `quality/metrics.py`
- Add `quality/gating.py`
- Support defining different reliability thresholds based on waveform type

**Acceptance Criteria**
- Unreliable leads in noisy ECG can be identified
- Most leads in clean ECG have stable scores
- Consistent with visualization results

**Priority**  
P0

**Dependencies**  
T002

---

<a id="任务-t004实现-beat-level-质量评分"></a>
### Task T004: Implement beat-level quality scoring

**Objective**  
Drill down from "whole lead quality" to "local quality of each beat."

**Input**
- Windowed signal for each beat
- Lead-level quality

**Output**
Per beat output:

- `beat_noise_score`
- `beat_baseline_shift`
- `beat_template_corr`
- `beat_measurement_reliable`

**Core Algorithms**
- Local SNR within the beat
- Correlation between the beat and the group template
- PR segment stability
- Noise level in the T region

**Acceptance Criteria**
- Confidence for ectopic/noisy beats is lower than for clean dominant beats
- Abnormal beats within a group can be excluded

**Priority**  
P0

**Dependencies**  
T003, T008

---

<a id="3-多导联-qrs-检测与锚点生成"></a>
## 3. Multi-lead QRS Detection and Anchor Generation

<a id="任务-t005升级多导联-qrs-detector"></a>
### Task T005: Upgrade multi-lead QRS detector

**Objective**  
Solidify the current QRS detector into a clear multi-lead fusion scheme, serving as the anchor layer for the entire delineation process.

**Input**
- 12-lead ECG
- `fs`

**Output**
- `r_locs`
- `qrs_candidate_windows`
- `qrs_detector_confidence`

**Core Algorithms**
1. Lead selection: `I, II, V1-V6`
2. Z-score normalization
3. Construct multi-lead vector magnitude
4. `5–30 Hz` bandpass
5. Derivative + squaring + MWI
6. Adaptive threshold + refractory period
7. Local refinement of R using reference leads

**Implementation Key Points**
- Retain current implementation logic
- Add parameter configuration class
- Add detector debug output

**Acceptance Criteria**
- Stable detection for normal sinus rhythm
- Acceptable recall rate for PVC / BBB / low voltage
- Does not rely solely on lead II

**Priority**  
P0

**Dependencies**  
None

---

<a id="任务-t006qrs-onsetoffset-两阶段精修"></a>
### Task T006: Two-stage refinement of QRS onset/offset

**Objective**  
QRS cannot rely solely on energy threshold expansion; representative-aware refinement must be added.

**Input**
- `r_locs`
- Group representative beats
- Lead quality

**Output**
- `qrs_on`
- `qrs_off`
- `qrs_on_confidence`
- `qrs_off_confidence`

**Core Algorithms**
- Coarse stage: Local energy rise/fall
- Fine stage:
  - Significant rise points in the first derivative
  - Zero crossings in the second derivative
  - Terminal force end deflection
  - Alignment correction with representative beat

**Implementation Key Points**
- Create new `delineate/qrs_bounds.py`
- Perform high-confidence localization on representative beat first
- Allow single-beat correction only within a small window

**Acceptance Criteria**
- Boundaries of wide QRS are more stable than the current version
- `qrs_off` does not truncate prematurely in multi-peak QRS
- Reduced jitter in QRS duration

**Priority**  
P0

**Dependencies**  
T005, T010

---

<a id="4-beat-grouping-与-representative-beat"></a>
## 4. Beat Grouping and representative beat

<a id="任务-t007重写-beat-grouping-描述子"></a>
### Task T007: Rewrite beat grouping descriptors

**Objective**  
Ensure that grouping serves not only rhythm analysis but also directly supports templated measurements.

**Input**
- `r_locs`
- Local waveform for each beat
- Lead quality

**Output**
Per beat descriptor:

- `rr_prev`
- `rr_next`
- `qrs_width_rough`
- `vector_area`
- `paced_flag`
- `template_embedding`
- `lead_corr_signature`

**Core Algorithms**
- Vector magnitude snippet
- PCA embedding
- Morphology cosine distance
- RR features

**Acceptance Criteria**
- Dominant beat family automatically forms Group 1
- PVCs and normal beats can be separated
- Paced beats are not grouped with non-paced beats

**Priority**  
P0

**Dependencies**  
T005

---

<a id="任务-t008实现两阶段分组器"></a>
### Task T008: Implement two-stage grouper

**Objective**  
Perform coarse grouping first, then fine grouping, to reduce the instability of pure clustering.

**Input**
- Beat descriptors

**Output**
- `group_id` for each beat
- `group_summary`

**Core Algorithms**
- Stage 1: Rule-based coarse grouping
  - Paced / non-paced
  - Narrow / wide
  - Early / non-early
- Stage 2: Intra-group morphology clustering
  - Hierarchical clustering or DBSCAN
  - Distance metric: Template cross-correlation

**Acceptance Criteria**
- Maximum of 5 groups
- Group 1 is the group with the most members
- Grouping results are stable across re-runs on the same ECG

**Priority**  
P0

**Dependencies**  
T007

---

<a id="任务-t009实现-representative-beat-生成增强版"></a>
### Task T009: Implement enhanced version of representative beat generation

**Objective**  
Make representative beat the primary input for subsequent boundary refinement.

**Input**
- Group member beat snippets
- Beat-level quality

**Output**
For each group, for each lead, output:

- representative beat
- `member_count`
- `mean_template_corr`
- `outlier_count`
- `beat_to_beat_onset_std`
- `beat_to_beat_offset_std`

**Core Algorithm**
- R alignment
- Local cross-correlation fine-tuning within QRS
- Outlier exclusion
- Mean beat formation
- Record intra-group variability

**Acceptance Criteria**
- representative beat is smoother than a single beat
- Significant reduction in intra-group EMG noise
- Preserve variability metadata

**Priority**  
P0

**Dependencies**  
T004, T008

---

<a id="5-rough-delineation先粗定位区域"></a>
## 5. Rough Delineation: Coarse Localization First

<a id="任务-t010实现-approximate-waveform-regions"></a>
### Task T010: Implement Approximate Waveform Regions

**Objective**  
Change from "direct peak finding" to "coarse region first, then fine boundaries."

**Input**
- `r_locs`
- representative beats
- RR information

**Output**
Coarse regions for each beat, each lead:

- `P_search_region`
- `QRS_search_region`
- `T_search_region`

**Core Algorithm**
- P region: `qrs_on - 260ms` to `qrs_on - 20ms`
- QRS region: `R ± 120ms`
- T region: `qrs_off + 20ms` to `min(qrs_off+500ms, next_expected_p-margin)`

**Acceptance Criteria**
- P/T search does not cross into wrong regions
- At high heart rates, the T region does not overlap with the next beat's P
- Search region is not excessively wide when RR is very long

**Priority**  
P0

**Dependencies**  
T006, T009

---

<a id="6-p-波检测改进"></a>
## 6. P Wave Detection Improvements

<a id="任务-t011实现-multilead-p-candidate-detector"></a>
### Task T011: Implement Multilead P Candidate Detector

**Objective**  
P wave detection cannot rely solely on single-lead abs-peak; multilead candidate generation is required.

**Input**
- `P_search_region`
- Lead quality
- representative beat

**Output**
P candidates per lead:

- `peak_candidates`
- `candidate_scores`

**Core Algorithm**
- Extrema points
- Derivative reversal points
- Local area peaks
- Wavelet scale peaks

**Implementation Notes**
- Allow 0~N candidates per lead
- Score based on amplitude, area, SNR, and intra-group consistency

**Acceptance Criteria**
- Low-amplitude P waves are not all missed
- Negative P waves still generate candidates
- Confidence decreases rather than false positives in atrial flutter samples

**Priority**  
P1

**Dependencies**  
T010

---

<a id="任务-t012实现-p-多导联融合与双分量模型"></a>
### Task T012: Implement P Multilead Fusion and Bicomponent Model

**Objective**  
Support monophasic P, notched P, and biphasic P.

**Input**
- P candidates from each lead
- Lead reliability
- Representative priors

**Output**
- `p_on`
- `p_peak1`
- `p_peak2`
- `p_off`
- `p_biphasic_flag`
- `p_notch_flag`
- `p_confidence`

**Core Algorithm**
- Candidate lead visibility scoring
- Top-K lead fusion
- Bicomponent model:
  - `P1`, `P2`, notch depth
- Boundaries determined jointly by three types of evidence:
  - Amplitude return to baseline
  - Cumulative area ratio
  - Derivative/curvature decay

**Acceptance Criteria**
- Stable results for normal sinus P waves
- Biphasic P can output two peaks
- `PR` jitter is less than current implementation

**Priority**  
P1

**Dependencies**  
T011

---

<a id="7-qrs-多峰形态与细节特征"></a>
## 7. QRS Multifocal Morphology and Detailed Features

<a id="任务-t013实现-qrrss-分量提取"></a>
### Task T013: Implement Q/R/R'/S/S' Component Extraction

**Objective**  
Enable QRS to output not only total duration but also components usable for morphology.

**Input**
- Refined QRS bounds
- representative beat

**Output**
- `Q_amp`
- `R_amp`
- `R_prime_amp`
- `S_amp`
- `S_prime_amp`
- `qrs_num_peaks`
- `qrs_notch_count`
- `qrs_slur_flag`

**Core Algorithm**
- Local extrema sequence in QRS region
- Peak-trough alternation constraints
- Small peak denoising threshold
- Terminal slur/notch identification

**Acceptance Criteria**
- More reasonable component features for RBBB/LBBB/fragmented QRS
- Normal narrow QRS is not overfitted into multifocal
- Stable notch counting

**Priority**  
P1

**Dependencies**  
T006

---

<a id="任务-t014实现-vat-j-point-精修"></a>
### Task T014: Implement VAT / J-point Refinement

**Objective**  
Support more complete morphology lead measurements.

**Input**
- Refined QRS
- representative beat

**Output**
- `vat_ms`
- `j_point_idx`
- `j_point_confidence`

**Core Algorithm**
- VAT: Time from onset to main R peak
- J-point: Local stable inflection point at the end of QRS

**Acceptance Criteria**
- Stable VAT logic when QRS is wide
- J-point is not significantly dragged by ST noise

**Priority**  
P2

**Dependencies**  
T013

---

<a id="8-t-波与几何-t-end"></a>
## 8. T Wave and Geometric T-end

<a id="任务-t015实现-multilead-t-candidate-detector"></a>
### Task T015: Implement Multilead T Candidate Detector

**Objective**  
T wave allows positive, inverted, biphasic, and notched morphologies, no longer taking only abs-peak.

**Input**
- `T_search_region`
- Lead quality
- representative beat

**Output**
- `T1_peak`
- `T2_peak`
- `t_main_polarity`
- `t_notch_flag`
- `t_candidate_score`

**Core Algorithm**
- Local extrema + waveform area + curvature
- Primary and secondary peak scoring
- Consistency constraint with group template

**Acceptance Criteria**
- Inverted T waves still output stably
- Notched T can distinguish primary and secondary peaks
- Candidate scores significantly reduced for noisy leads

**Priority**  
P1

**Dependencies**  
T010

---

<a id="任务-t016实现-dxl-like-几何-t-end-算法"></a>
### Task T016: Implement DXL-like Geometric T-end Algorithm

**Objective**  
Replace the current "threshold return to baseline" method with a geometric inflection point method.

**Input**
- T candidate peaks
- Representative T waveform
- Baseline trend

**Output**
- `t_end_candidate`
- `t_end_confidence`
- `t_flat_tail_score`
- `u_wave_present`
- `p_on_t_flag`

**Core Algorithm**
1. Take `T_peak`
2. Define right-side search region
3. Construct `peak -> search_start` or `peak -> search_end` reference line
4. Calculate perpendicular distance
5. Take the point of maximum perpendicular distance as the inflection candidate
6. Apply upper and lower bound constraints on slope
7. If a U wave exists, prioritize locating the T-U nadir

**Implementation Notes**
- Implement for single lead first
- Then perform cross-lead fusion
- Support confidence reduction for flat tails

**Acceptance Criteria**
- Flatter T tails are more stable than the current threshold method
- Large U wave samples no longer systematically prolong QT
- `T_end` intra-group variance decreases

**Priority**  
P0

**Dependencies**  
T015, T018

---

<a id="任务-t017实现-t-波多导联融合"></a>
### Task T017: Implement T Wave Multilead Fusion

**Objective**  
Prevent a single bad lead from determining the final T-end.

**Input**
- per-lead `t_end_candidate`
- lead reliability
- beat-to-beat variance

**Output**
- `lead_t_end_used`
- `t_end_fused`
- `qt_reliable_leads`

**Core Algorithm**
- Lead scoring:
  - `lead_quality`
  - `t_amp`
  - `t_end_var`
  - `rep_confidence`
- Fusion strategy:
  - Top-K leads
  - robust median / weighted median

**Acceptance Criteria**
- Noisy leads are automatically excluded
- Per-lead QT outliers do not affect global QT
- The fused result is more stable than any single lead

**Priority**  
P0

**Dependencies**  
T016, T019

---

<a id="9-单搏局部修正与代表搏回投"></a>
## 9. Single-beat Local Correction and Representative Beat Back-projection

<a id="任务-t018实现-representative-boundary-priors"></a>
### Task T018: Implement representative-boundary priors

**Objective**  
First establish high-confidence boundaries on representative beat, then back-project them to individual beats within the group.

**Input**
- representative beats
- rough search regions

**Output**
- `rep_p_prior`
- `rep_qrs_prior`
- `rep_t_prior`

**Core Algorithm**
- Perform high-precision delineation on representative beat first
- Output a prior window for each boundary

**Acceptance Criteria**
- Representative boundaries are reproducible
- Beats within the group share the same set of priors

**Priority**  
P0

**Dependencies**  
T009, T010, T012, T016

---

<a id="任务-t019实现-beat-wise-local-correction"></a>
### Task T019: Implement beat-wise local correction

**Objective**  
Ensure that the final boundary of each beat equals the representative prediction plus a small-window local correction.

**Input**
- representative priors
- single beat signal
- local quality

**Output**
- Final single-beat `p_on/off`, `qrs_on/off`, `t_on/off`

**Core Algorithm**
- First, perform template alignment
- Then, only within the `±20~30 ms` small window:
  - Local derivative correction
  - Local cross-correlation correction
  - Baseline regression correction

**Acceptance Criteria**
- Boundary jitter within the group is less than in the current version
- Noisy beats do not deviate excessively
- Beat-to-beat variance can be used for subsequent reliable lead determination

**Priority**  
P0

**Dependencies**  
T018

---

<a id="10-置信度与-reliable-lead-体系"></a>
## 10. Confidence and Reliable Lead System

<a id="任务-t020实现-boundary-confidence-引擎"></a>
### Task T020: Implement boundary confidence engine

**Objective**  
All boundaries must have confidence scores; otherwise, subsequent QT and rule layers cannot be tuned.

**Input**
- local noise
- derivative consistency
- template correlation
- lead quality
- beat variance

**Output**
- `p_confidence`
- `qrs_confidence`
- `t_confidence`
- `t_end_confidence`

**Core Algorithm**
Combined scoring:

```text
conf = f(local_snr, template_corr, baseline_stability, boundary_sharpness, group_consistency)
```

**Acceptance Criteria**
- Confidence for clean beats is significantly higher
- Confidence for flat T / noisy P is noticeably reduced
- Confidence trends are consistent with manual visual inspection

**Priority**  
P0

**Dependencies**  
T004, T019

---

<a id="任务-t021实现-reliable-lead-判定器"></a>
### Task T021: Implement a reliable lead detector

**Objective**  
Replicate the methodological spirit of DXL: use only reliable leads for global QT, prioritizing the median over extreme values.

**Input**
- Per-lead QT
- Per-lead onset/offset variance
- Lead quality
- Confidence scores

**Output**
- `reliable_for_qt`
- `reliable_for_axis`
- `reliable_for_global_measure`

**Core Algorithm**
Example of lead reliability conditions:

```python
reliable = (
    lead_quality < th1 and
    t_end_var < th2 and
    qrs_on_var < th3 and
    rep_confidence > th4 and
    t_end_confidence > th5
)
```

**Acceptance Criteria**
- Low-amplitude, high-noise leads are excluded
- Global QT is no longer skewed by a single bad lead
- The number of reliable leads is reasonable

**Priority**  
P0

**Dependencies**  
T020

---

<a id="11-global-qt-与-qtc"></a>
## 11. Global QT and QTc

<a id="任务-t022实现-per-lead-qt-jt-输出"></a>
### Task T022: Implement per-lead QT / JT output

**Objective**  
First, solidify the per-lead QT calculation, then compute the global QT.

**Input**
- `qrs_on`
- `t_end`
- confidence
- reliability

**Output**
- `qt_ms`
- `jt_ms`
- `qt_confidence`

**Core Algorithm**
- `QT = T_end - QRS_on`
- `JT = T_end - QRS_off`

**Acceptance Criteria**
- Each lead has an independent QT
- Unreliable leads retain their values but are marked as unusable for global calculation

**Priority**  
P0

**Dependencies**  
T017, T021

---

<a id="任务-t023实现-global-qt-qtc"></a>
### Task T023: Implement global QT / QTc

**Objective**  
Implement global QT selection in the style of DXL.

**Input**
- per-lead QT
- reliable lead mask
- RR

**Output**
- `global_qt_ms`
- `qtc_bazett_ms`
- `qtc_fridericia_ms`
- `qt_dispersion_ms`

**Core Algorithm**
- `global_qt = median(qt_candidates_from_reliable_leads)`
- `qt_dispersion = max - min`
- QTc outputs both Bazett and Fridericia

**Acceptance Criteria**
- Global QT is insensitive to outlier leads
- Better repeatability compared to the current simple threshold method
- QTc logic is correct when RR varies

**Priority**  
P0

**Dependencies**  
T022

---

<a id="12-pacing-专用分支"></a>
## 12. Pacing-Specific Branch

<a id="任务-t024实现-pacing-spike-detector"></a>
### Task T024: Implement pacing spike detector

**Objective**  
Independently identify pacing spikes as a prerequisite for paced beat delineation.

**Input**
- Raw high-fidelity ECG
- `fs`

**Output**
- `spike_events`
- `spike_confidence`
- `paced_status`

**Core Algorithm**
- High-frequency channel `>100 Hz`
- Ultra-narrow pulse width detection
- Differential peak + area constraints
- Multi-lead temporal clustering

**Acceptance Criteria**
- Pacing spikes are not confused with ordinary narrow QRS
- Obvious spikes can be detected
- The number of false spikes is controllable

**Priority**  
P1

**Dependencies**  
T005

---

<a id="任务-t025paced-beat-delineation-分支"></a>
### Task T025: Paced beat delineation branch

**Objective**  
Paced beats cannot directly use the standard QRS model.

**Input**
- paced beats
- spike events
- representative paced template

**Output**
- `qrs_on/off` for paced beats
- `stim_to_qrs_ms`
- `ventricular_paced_flag`

**Core Algorithm**
- Wide QRS pattern after spike
- QRS onset can be earlier
- Terminal offset is more conservative

**Acceptance Criteria**
- Paced QRS duration is more reasonable than the current version
- The relationship between spike and QRS is interpretable

**Priority**  
P2

**Dependencies**  
T024

---

<a id="13-morphology-字段补全"></a>
## 13. Morphology Field Completion

<a id="任务-t026补全-lead-level-morphology-measurements"></a>
### Task T026: Complete lead-level morphology measurements

**Objective**  
Make library output granularity close to that of the Extended Measurements report.

**Input**
- Final boundary results
- amplitude / area / slope calculators

**Output**
For each lead and each beat, or at least for the representative lead, complete the following:

- P: `on/peak1/peak2/off/amp/area/dur`
- QRS: `Q/R/R'/S/S'/dur/area/notch/slur/VAT`
- ST: `J_point/ST_on/ST_mid/ST_80/ST_end/ST_slope`
- T: `on/peak1/peak2/off/amp/area/dur/U_wave_flag`

**Acceptance Criteria**
- Unified field naming
- Complete JSON / dataframe export
- Missing fields have clear `None` semantics

**Priority**  
P1

**Dependencies**  
T012, T013, T014, T016

---

<a id="14-可视化与调试工具"></a>
## 14. Visualization and Debugging Tools

<a id="任务-t027实现边界叠加可视化"></a>
### Task T027: Implement boundary overlay visualization

**Objective**  
Without visualization, it is difficult to tune the boundary algorithm.

**Input**
- ECG
- boundary results
- quality/confidence

**Output**
Debug plots:

- Single-lead beat plot
- representative beat plot
- P/QRS/T boundary overlay
- Confidence annotations

**Core Features**
- matplotlib plotting
- Bad beat highlighting
- Reliable lead marking

**Acceptance Criteria**
- One line of code can plot the boundaries for a specific lead and beat
- Visualization can be directly used for error troubleshooting

**Priority**  
P0

**Dependencies**  
T019, T020

---

<a id="任务-t028实现回归快照工具"></a>
### Task T028: Implement regression snapshot tool

**Objective**  
Make algorithm changes traceable to avoid "fixing one thing and breaking another."

**Input**
- Fixed ECG sample set
- Current version output

**Output**
- JSON snapshots
- PNG overlays
- Diff report

**Acceptance Criteria**
- Automatically compare key boundary shifts after each change
- Provide a sample list when regression fails

**Priority**  
P1

**Dependencies**  
T027

---

<a id="15-验证与基准"></a>
## 15. Validation and Benchmarks

<a id="任务-t029建立-synthetic-signal-benchmark"></a>
### Task T029: Establish synthetic signal benchmark

**Objective**  
First validate onsets/offsets on controllable signals, then move to real data.

**Input**
- Synthetic ECG generator
- Controllable noise injector

**Output**
Evaluation metrics:
- onset error
- offset error
- QT error
- ST error
- Noise robustness curve

**Acceptance Criteria**
- Can inject baseline wander / muscle noise / powerline interference separately
- Error statistics are automatically generated

**Priority**  
P0

**Dependencies**  
T023

---

<a id="任务-t030建立专家标注数据评测"></a>
### Task T030: Establish expert-annotated data evaluation

**Objective**  
Validate boundary repeatability and error distribution on real ECG.

**Input**
- Annotated dataset
- Current algorithm output

**Output**
- `P_on/off error`
- `QRS_on/off error`
- `T_end error`
- `PR/QRS/QT error`

**Acceptance Criteria**
- Evaluation code is reproducible
- Output includes mean / SD / percentiles
- Supports scenario-specific statistics: sinus rhythm, wide QRS, low-amplitude T, noisy ECG

**Priority**  
P1

**Dependencies**  
T029

---

<a id="16-里程碑计划"></a>
## 16. Milestone Plan

<a id="milestone-m1可用的第二版测量内核"></a>
## Milestone M1: Usable Second-Version Measurement Kernel
**Includes**
- T001 ~ T010
- T016
- T018 ~ T023
- T027
- T029

**Capabilities After Completion**
- QRS and T-end are significantly stronger than the current version
- representative beat truly participates in boundary localization
- global QT uses reliable-lead median

---

<a id="milestone-m2形态特征增强"></a>
## Milestone M2: Morphological Feature Enhancement
**Includes**
- T011 ~ T015
- T026
- T028
- T030

**Capabilities upon completion**
- P/T dual components
- QRS multi-peak
- More complete morphology lead measurements

---

<a id="milestone-m3paced-高复杂度场景"></a>
## Milestone M3: Paced / High-Complexity Scenarios
**Includes**
- T024
- T025
- T014

**Capabilities upon completion**
- Dedicated branch for paced beats
- More complete VAT / J-point

---

<a id="17-优先级总表"></a>
## 17. Priority Summary Table

### P0
- T001
- T002
- T003
- T004
- T005
- T006
- T007
- T008
- T009
- T010
- T016
- T017
- T018
- T019
- T020
- T021
- T022
- T023
- T027
- T029

### P1
- T011
- T012
- T013
- T015
- T024
- T026
- T028
- T030

### P2
- T014
- T025

---

## 18. Definition of Done

A task is considered complete only if it meets all of the following 5 criteria:

- Code has been merged into the main branch
- Unit tests pass
- At least 1 visualization example has passed manual review
- No abnormal drift in regression snapshots
- Documentation completed: inputs, outputs, failure modes, parameter descriptions

---

<a id="19-建议的开发顺序"></a>
## 19. Recommended Development Order

Proceed in the order of highest cost-effectiveness:

1. `T001-T010`
2. `T016-T023`
3. `T027-T029`
4. `T011-T015, T026`
5. `T024-T025, T030`

The reason for this order is simple:  
First, establish the main chain of **QRS anchors, representative beat, geometric T-end, reliable leads, and global QT**. This yields the greatest benefit and best aligns with the methodological focus published in the DXL manual.

---

<a id="20-建议你立刻开工的首批-todo"></a>
## 20. Recommended Initial TODOs to Start Immediately

```text
[ ] T001 重构 delineate 目录
[ ] T002 建立统一边界数据结构
[ ] T003 lead-level 质量评分重构
[ ] T005 升级多导联 QRS detector
[ ] T007 重写 beat grouping 描述子
[ ] T008 实现两阶段分组器
[ ] T009 representative beat 增强版
[ ] T010 approximate waveform regions
[ ] T016 几何 T-end
[ ] T018 representative-boundary priors
[ ] T019 beat-wise local correction
[ ] T020 boundary confidence 引擎
[ ] T021 reliable lead 判定器
[ ] T022 per-lead QT / JT
[ ] T023 global QT / QTc
[ ] T027 边界叠加可视化
[ ] T029 synthetic benchmark
```

<!-- i18n-nav -->
[中文](p%20wave%20doc) | [English](p%20wave%20doc.en.md) | [日本語](p%20wave%20doc.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="多导联-ecg-p-波定界代码修改重点指南"></a>
# Multi-Lead ECG P-wave Delimitation: Key Guidelines for Code Modifications

> Scope of application: Multi-lead P wave onset/offset delimitation is achieved through deterministic signal processing without using machine learning.  
> Engineering goal: Prioritize ensuring that the results are reliable, interpretable, and calibrated, and clearly reject the signal when the signal is unidentifiable.

---

<a id="1-核心设计"></a>
## 1. Core design

It is recommended to adopt the following main process:

**Acquisition quality check → Dual-branch preprocessing → Atrial activity status judgment → Multi-lead coarse positioning → Single-lead precise delimitation → Reliability assessment → Robust fusion → Beat-to-beat verification**

Among the most important principles are:

1. **First use multi-lead information to determine the approximate range of the P wave, and then precisely position it lead by lead within a narrow window. **
2. **Isopotential or high-noise leads do not participate in delimitation and fusion. **
3. **Global onset/offset is the "earliest or latest credible boundary", not the average of all lead boundaries. **
4. **Ta wave is the real atrial repolarization activity and cannot be simply used as baseline or noise to remove. **
5. **When the TP segment disappears under high heart rate, the baseline estimation mode must be switched. **
6. **P-on-T scenes cannot be directly subtracted by hard subtraction using the fixed T template. **
7. **Rejection must be allowed when P, T, Ta or artifacts cannot be reliably distinguished. **

---

<a id="2-输出内容"></a>
## 2. Output content

The system should not output only an onset and offset. Output at least per beat:

- Strictly global onset/offset
- Robust support for onset/offset
- onset/offset confidence interval
- P wave state
- List of valid leads
- Each lead boundary, quality score and uncertainty
- Current baseline mode in use
- Whether it has undergone T-wave reconstruction
- Is there Ta wave ambiguity?
- Accept or reject status
- Reasons for rejection

<a id="两套全局边界"></a>
### Two sets of global boundaries

| Output | Meaning | Usage |
|---|---|---|
| Strict global boundaries | Among the leads that pass hard quality control, the earliest onset and latest offset | Close to the traditional multi-lead measurement caliber |
| Robust support boundaries | Earliest/latest credible boundaries supported by multiple relatively independent lead groups | Engineering master output |

When the difference between the two sets of results is too large, the confidence level should be lowered. Common causes include single-lead artifacts, channel desynchrony, T/Ta residuals, or misconnected leads.

<a id="建议的-p-波状态"></a>
### Suggested P-wave status

- `P_PRESENT`: Stable P wave detected
- `AF_LIKE`: No stable P template, performance close to atrial fibrillation
- `ORGANIZED_ATRIAL_ACTIVITY`: Atrial flutter or multiple organized atrial waves
- `OVERLAP_UNCERTAIN`: P-on-T, Ta interference or insufficient evidence

<a id="建议的拒判原因"></a>
### Suggested reasons for rejection

- `BASELINE_UNOBSERVABLE`
- `P_ON_T_UNRESOLVED`
- `P_OFFSET_TA_AMBIGUOUS`
- `INSUFFICIENT_INFORMATIVE_LEADS`
- `CROSS_LEAD_DISAGREEMENT`
- `CHANNEL_DESYNCHRONIZED`
- `LEAD_CONFIGURATION_INVALID`
- `MODEL_DISAGREEMENT`

---

<a id="3-采集链与通道质量检查"></a>
## 3. Acquisition chain and channel quality inspection

This step must precede all filtering and delimiting.

<a id="必查项目"></a>
### Must check items

- Are the sampling rate, timestamp and data length consistent?
- Are the signal units and gains correct?
- Is there saturation, flat lines, lead dropouts or large spikes?
- Whether each lead is synchronized
- Whether each channel is filtered or resampled differently
- Is there half-sample point or fixed channel delay
- Is the algebraic relationship between limb leads reasonable?
- Whether there is lead reversal, duplicate channels or abnormal gain

<a id="修改重点"></a>
### Key points of modification

- Added channel delay estimation and compensation.
- Check cross-lead time synchronization using QRS high slope region.
- Added limb lead relationship check to detect connection errors, gain abnormalities and out-of-synchrony.
- When an acquisition chain error is detected, multi-lead fusion should not proceed.

---

<a id="4-双分支预处理"></a>
## 4. Double branch preprocessing

Don't let the same set of strong low-pass signals handle both rough detection and final fine demarcation.

<a id="检测分支"></a>
### Detect branches

Used for:

- QRS/T detection
- P wave coarse positioning
- Space speed calculation
- Template related
- Judgment of atrial activity status

It is recommended to use a stronger low pass to improve overall stability.

<a id="精定界分支"></a>
### Finely defined branch

Used for:

- Final onset/offset positioning
- P inspection for small cuts and chips
- Local change point detection
- Cross-validation of broadband results

It is recommended to retain a wide diagnostic bandwidth to avoid strong low-pass obliteration of weak initial or terminal components.

<a id="上采样说明"></a>
### Upsampling instructions

Upsampling can:

- Reduce the problem of output being locked on the original sampling raster
- Unify time window configuration under different sampling rates
- Supports tangent intersection and continuous curve fitting

But it **cannot recover information that has been lost during the acquisition phase**.

---

<a id="5-基线估计普通模式与无-tp-模式"></a>
## 5. Baseline estimation: normal mode and no TP mode

<a id="普通模式"></a>
### Normal mode

Points in the PQ or TP segments cannot be unconditionally used as "zero potential nodes". Only when the following conditions are met at the same time can it be used as a baseline soft observation:

- Low activity in multiple leads
- Low local slope and fine-scale energy
- Not in P, QRS or T activity zone
- T/Ta residual is small
- No simultaneous changes in multiple leads

These points should be used as soft constraints with uncertainties, rather than as forced zero potentials.

<a id="高心率tp-消失模式"></a>
### High heart rate, TP disappearing mode

The disappearance of TP means that the baseline temporarily lacks direct observation and the beat-by-beat free spline fitting cannot continue.

Suggestions:

- Use a state-space baseline model across multiple shots.
- Update baseline when there are credible low activity samples.
- Only make predictions when there are no credible samples.
- Automatically increase the baseline uncertainty as the no-observation time increases.
- Bidirectional smoothing can be used for offline processing; fixed delay smoothing can be used for online processing.

<a id="多导联基线"></a>
### Multi-lead baseline

Prioritize dividing the baseline into:

- Slowly changing parts shared by multiple leads, such as respiration or body movement
- Each lead's own slow remnant

This is less likely to absorb the true P wave into the baseline than using high-DOF splines for each lead independently.

<a id="失败处理"></a>
### Failure handling

If there is no credible equipotential observation for a long time, and the baseline, T tail, Ta tail and P wave cannot be reliably separated, the output should be:

`BASELINE_UNOBSERVABLE`

---

<a id="6-ta-波感知"></a>
## 6. Ta wave perception

The Ta wave may affect the offset of the current P wave, or it may extend to near the P onset of the next beat when the room rate is high.

<a id="主要风险"></a>
### Main risks

- The tail of Ta on the previous beat is mistaken for the early activity of P on the next beat
- The initial part of the current Ta changes the P wave back to zero shape
- PQ/TP segments are no longer truly equipotential
- Area method or baseline intersection method produces systematic deviations
- Stable Ta tail is misjudged as high confidence terminal P

<a id="代码处理原则"></a>
### Code processing principles

- Ta waves are only treated as restricted slow-varying interference terms, and complete beat-by-beat reconstruction is not pursued.
- Allows a small number of smooth basis functions to be used to describe the Ta/T tails.
- The P wave core region prohibits free fitting of the Ta model.
- The reverse polarity of Ta and P can only be used as a soft tip, not as a hard rule.
- Perturb the intensity, width and position of Ta, accounting for boundary changes into uncertainty.

<a id="onset-与-offset-分开处理"></a>
### Onset and offset are handled separately

P onset can find local activity onset points on a slowly changing background, so it is usually easier to maintain stability than offset.

P offset is more susceptible to Ta. After entering Ta-aware mode:

- Area zeroing method is downgraded or disabled
- Baseline intersection method to reduce weight
- Increase the weight of fine-scale wavelets, derivative activity and spatial direction continuity
- When multiple methods clearly diverge, high confidence boundaries will not be output.

When terminal P and early Ta cannot be distinguished, the output is:

`P_OFFSET_TA_AMBIGUOUS`

---

<a id="7-p-on-t受约束的-t-波重建"></a>
## 7. P-on-T: Constrained T-wave reconstruction

<a id="禁止固定模板直接硬相减"></a>
### Disable direct hard subtraction of fixed templates

The main problems with direct subtraction include:

- T-wave amplitude changes beat by beat, leaving obvious low-frequency residue
- The T-wave is slightly time-displaced, producing sharp derivative-shaped residues
- Rectangular windows start and stop causing edge discontinuities and broadband artifacts
- Under a fixed PR rhythm, P wave may be written into the T/QRST template and subtracted together

<a id="推荐方式"></a>
### Recommended method

Change "Template Subtraction" to "Constrained Multi-Lead T-Wave Reconstruction":

- P candidate region does not participate in T wave parameter estimation
- T template from similar beats currently recorded
- Templates are filtered by QRS/T morphology, RR, respiratory phase and spatial direction
- Allows only limited amplitude scaling, time shifting, slight telescoping and spatial adjustment
- Disable unconstrained time warping
- The reconstruction results must remain smooth and continuous
- Prohibit hard splicing of rectangular windows

<a id="三分支交叉验证"></a>
### Three-branch cross validation

Keep three branches at the same time:

1. **Original Broadband Branch**: No subtraction, only processing slowly changing background.
2. **T wave reconstruction branch**: Use the residual signal after constrained reconstruction.
3. **T space subspace suppression branch**: Suppress T waves from multi-lead spatial directions.

Judgment rules:

- The three branches have consistent results: increase confidence
- Small divergence of results: expand confidence intervals
- The results are obviously divergent: output `P_ON_T_UNRESOLVED`

<a id="t-重建失败检查"></a>
### T Rebuild failure check

Must check:

- Whether new peaks appear at the edge of the reconstruction area
- Does the T-region residuals without P actually decrease?
- Is the boundary stable after replacing similar templates?
- Whether known P-wave patterns are attenuated after reconstruction
- whether to generate new boundaries that are not supported at all by the original wideband signal
- Whether the new boundary is supported by multiple independent lead groups

---

<a id="8-多导联粗定位"></a>
## 8. Multi-lead coarse positioning

<a id="主方法白化空间活动度"></a>
### Main method: whiten spatial activity

Ordinary multi-lead sum of squares assumes that the noise in each lead is independent and the variance is close, which is usually not true in practice.

It is recommended to first estimate the multi-lead noise covariance based on the low activity area, and then calculate the whitened spatial activity to reduce:

- Single noisy lead
- Common mode artifacts across leads
- Limb lead redundancy
- Differences in noise amplitude in different leads

The threshold is first obtained from the empirical distribution of the low activity area, and uses high and low dual thresholds:

- High threshold confirms the presence of activity
- Low threshold to backtrack to the true starting point or end point
- Require activities to last for a certain period of time to avoid single-point noise triggers

<a id="svd-互补分支"></a>
### SVD complementary branch

Don't always use only the first principal component.

Suggestions:

- Whiten by noise level first
- Weighted again by lead quality
- When the first and second space components are close, the two-dimensional subspace is retained
- Increase uncertainty when spatial direction is unstable
- Perform at most a small number of iterations to prevent false initial windows from self-reinforcement

SVD results can only be interpreted as favorable body surface spatial projections and not as anatomical activation origins.

---

<a id="9-单导联精定界"></a>
## 9. Single lead precise delimitation

No single method should independently determine the final boundary.

<a id="onset-主证据"></a>
### Onset Master Evidence

- Multi-scale wavelet
- local change points
- Least squares tangent
- Consistency of broadband and detection branches

<a id="offset-主证据"></a>
### Offset main evidence

- Multi-scale wavelet activity disappears
- Derivative activity dropped
- Conditional area method
- Constrained curve fitting
- Ta-aware results

<a id="关键修改"></a>
### Key changes

- Biphasic and triphasic P waves must be handled explicitly, especially V1.
- Do not use fixed rules such as "the first extreme value is onset".
- The tangent method should perform straight line fitting on a small section near the maximum slope to avoid using single point derivatives.
- The area method is only enabled when the baseline is reliable, there is no obvious Ta, and there is no serious P-on-T.
- Curve fitting is mainly used for subsampling estimation and cross-validation, and cannot forcefully explain all complex patterns.
- Multi-scale or multi-method disagreements must be entered into the uncertainty calculation.

---

<a id="10-导联可靠性"></a>
## 10. Lead reliability

<a id="硬排除"></a>
### Hard exclusion

- flat line
- saturated
- Lead loss
- Large spikes
- Severely out of sync
- Obvious gain anomaly
- Lead configuration error

<a id="软质量指标"></a>
### Soft quality indicators

- P-wave local signal-to-noise ratio
- P wave local energy
- Derivative energy
- Relevance to this record template
- Baseline stability
- Multi-scale consistency
- Multi-method consistency
- Threshold perturbation stability
- Beat boundary jitter
- T reconstruction sensitivity
- Ta model sensitivity

<a id="等电位导联"></a>
### Equipotential leads

You can't just look at a single peak. Should jointly check:

- Is the peak significant?
- Is the total energy significant?
- Is the derivative energy significant?
- Whether there is a stable template structure
- Is there consistent activity across multiple scales?

When multiple pieces of evidence indicate that there is no valid P information:

- Mark as uninformative leads
- onset/offset weight reset to zero
- Do not output this lead boundary

<a id="生理范围"></a>
### Physiological range

P wave duration or amplitude outside the common range may be a real pathology and should not be dismissed out of hand.

- Acquisition failure: hard troubleshooting
- Physiological range crossing: soft punishment
- Pathological morphology: allowed to remain, but with increased uncertainty

---

<a id="11-不确定度与置信区间"></a>
## 11. Uncertainty and confidence interval

Threshold perturbation can only reflect part of the instability and cannot represent the true confidence alone.

The uncertainty should cover at least:

- High frequency random noise
- Baseline estimation error
- Wavelet scale and threshold changes
- Derivative window changes
- Differences in different delimitation methods
-T template selection and reconstruction parameter changes
-Ta model changes
- Beat-to-beat jitter

<a id="实现要点"></a>
### Implementation points

- Make small-scale reasonable disturbances to relevant parameters and repeat delimitation.
- Noise resampling should preserve temporal correlation and cross-lead correlation.
- Set minimum uncertainty for each lead to avoid infinite weight.
- Set the maximum weight for single leads and lead groups to avoid a single path dominating the fusion.
- Final output confidence interval, not just an abstract quality score.

<a id="校准检查"></a>
### Calibration Check

Check on the validation set:

- Whether the nominal confidence interval reaches the corresponding actual coverage
- When the confidence interval is wider, is the true error generally larger?
- After the low-confidence sample is rejected, whether the accepted results are significantly improved

---

<a id="12-单侧稳健融合"></a>
## 12. Unilateral Robust Fusion

The weighted median of all leads is not suitable as the final global boundary because it tends to delay onset and advance offset, thereby shortening the P wave duration.

<a id="onset-融合"></a>
### Onset Fusion

- Sort lead onset by time
- Form candidate time clusters within a limited time range
- Requires support from at least two relatively independent lead groups
- Select the earliest time cluster that reaches the reliable weight threshold
- Take the early weighted position within the cluster

<a id="offset-融合"></a>
### Offset Fusion

- Use the same principle to find the latest credible time cluster
- Take a later weighted position within the cluster

<a id="导联分组"></a>
### Lead grouping

Distinguish at least:

- Limb lead group
- Right precordial lead group
- Left precordial lead group

Set an upper limit on the weight of each group to prevent relevant noise from being counted repeatedly.

<a id="拒判条件"></a>
### Conditions for rejection

After eliminating obvious outlier leads, if the boundaries are still highly dispersed, the output should be:

`CROSS_LEAD_DISAGREEMENT`

---

<a id="13-拍间一致性"></a>
## 13. Beat-to-beat consistency

Beat smoothing or superpositioning can only be done within the same morphological cluster.

The forms that need to be separated include at least:

- Sinus P
- Premature atrial contractions
- Retrograde P
- Other stable ectopic P

Cross-morphological clusters are prohibited:

- Stacked average
- PR interval smoothing
- P time limit smoothing
- Boundary Kalman smoothing

Mutations should first trigger a morphology or rhythm pattern switching check and should not be smoothed out directly.

---

<a id="14-建议的代码模块"></a>
## 14. Suggested code modules

| Module | Main Responsibilities |
|---|---|
| `acquisition_qc` | Channel synchronization, gain, lead relationship, saturation and dropout check |
| `preprocessing` | Detection and precise bounding dual-branch filtering and resampling |
| `baseline` | Normal baseline and no TP state space baseline |
| `rhythm_gate` | P, AF-like, atrial flutter and overlapping status judgment |
| `global_detector` | Whitening spatial activity and global coarse P-window |
| `svd_projection` | Weighted one-dimensional/two-dimensional space projection |
| `t_reconstruction` | Constrained T-wave reconstruction under P-zone mask |
| `ta_model` | Ta-aware judgment and offset ambiguity processing |
| `single_lead` | Wavelet, change point, tangent, area and fitting |
| `lead_quality` | Lead reliability and uninformative lead judgment |
| `uncertainty` | Parameter perturbation, noise resampling and confidence intervals |
| `fusion` | Earliest/Latest Trusted Time Cluster Fusion |
| `temporal_consistency` | Beat-to-beat verification within the same morphological cluster |
| `diagnostics` | Logs, diagnostic diagrams and reasons for rejection |

---

<a id="15-修改优先级"></a>
## 15. Modify priority

<a id="p0必须完成"></a>
### P0: Must be completed

- [ ] Add acquisition chain and channel synchronization check
- [ ] Create detection/precise bounding dual branch
- [ ] Reset equipotential lead weights to zero
- [ ] Use whitened multi-lead spatial activity
- [ ] Replace the overall weighted median with "earliest/latest credible clusters"
- [ ] Add clear rejection status and reasons
- [ ] Switch state space baseline when TP disappears
- [ ] Disable direct hard subtraction of fixed T templates

<a id="p1显著提升鲁棒性"></a>
### P1: Significantly improve robustness

- [ ] Implement constrained T-wave reconstruction under P-region mask
- [ ] Add Ta-aware offset logic
- [ ] Keep the three branches of original, T reconstruction and T subspace
- [ ] Add two-dimensional SVD subspace
- [ ] Increase the upper limit of lead group weight
- [ ] propagate baseline uncertainty to bounding confidence intervals
- [ ] Increase multi-scale, multi-method and parameter perturbation uncertainty

<a id="p2进一步优化"></a>
### P2: Further optimization

- [ ] Add local change point delimitation
- [ ] Implement multi-lead joint noise resampling
- [ ] Use respiratory phase to assist T-template selection
- [ ] Calibration confidence interval coverage
- [ ] Automatically generate diagnostic diagrams and failure cause reports
- [ ] Separate offline mode and online fixed delay mode

---

<a id="16-测试重点"></a>
## 16. Test focus

<a id="基线"></a>
### Baseline

- TP complete and TP completely gone
- No reliable baseline observation for a long time
- Breathing-like drift and non-linear drift
- P waves must not be absorbed by the baseline model

### Ta

- The tail of the previous beat Ta is superimposed on the next beat P
- Current Ta interferes with P offset
- Terminal P is indistinguishable from early Ta
- Whether `P_OFFSET_TA_AMBIGUOUS` is triggered correctly

<a id="t-波重建"></a>
### T wave reconstruction

- Amplitude changes
- Slight time offset
- Changes in spatial axis caused by breathing
- Whether the edges generate spikes
-Whether the P wave is wrongly attenuated
- Whether P is generated out of thin air on the T wave without P

<a id="多导联融合"></a>
### Multi-lead fusion

- A single early or late pseudo-boundary
- Only one lead group supported
- Two independent lead groups support
- Duplicate counting of votes for multiple related leads
- Changes in the number of effective leads

<a id="临床与噪声场景"></a>
### Clinical and Noise Scenarios

- Normal sinus rhythm
- Low amplitude P
- Biphasic and multi-notch P
- Shattered P
- High heart rate, no TP
-P-on-T
- Premature atrial contractions
- Atrial flutter and atrial fibrillation
- Baseline drift
- EMG noise
- Electrode movement
- Reverse lead connection
- Channel fixed delay

---

<a id="17-验收指标"></a>
## 17. Acceptance indicators

In addition to the mean error and standard deviation, report at least:

- Median absolute error
- 95% quantile absolute error
- Detection rate
-Rejection rate
- No false positive rate on P rhythm
- Errors stratified by signal-to-noise ratio, heart rate, morphology and overlap
- Offset error in Ta-aware mode
- actual coverage of confidence interval
- Difference between strict boundary and robust boundary
- Number of valid leads and number of valid lead groups
- Online handling of delays and worst-case uptime

Focus on observing the "acceptance rate-error" relationship:

> As the rejection threshold becomes stricter, whether the error of the accepted results decreases steadily.

This better reflects the actual reliability of the system than just reporting the average error across all samples.

---

<a id="18-参数冻结与外部验证"></a>
## 18. Parameter freezing and external validation

Deterministic algorithms can also overfit the database.

Suggestions:

- Simulation and a database for development and parameter selection
- Second database for independent verification
- A third database or factory locked set for final testing
- All threshold, window, bias correction and fusion parameters frozen before testing
- Simultaneous reporting of results before and after bias correction
- Statistical confidence intervals are sampled by records, not independently sampled by individual shots

Simulation should cover:

- Various P wave forms
- Real P/T template combination
- Different Ta intensity and time limit
- Different P/T overlap levels
- Axis and amplitude changes due to respiration
- Multi-lead correlated noise
- ADC quantization and different sampling rates
- Lead gain, reverse connection and channel delay

---

<a id="19-最终工程原则"></a>
## 19. Final Engineering Principles

1. **Upsampling is used to reduce raster effects and does not mean restoring the original information. **
2. ** Stronger low-pass is suitable for detection, and the final boundary should be confirmed by the wider bandwidth branch. **
3. **The global boundary is a one-sided credible boundary, not the center value of the lead result. **
4. **Ta wave belongs to real atrial activity and cannot be directly removed as the baseline. **
5. The disappearance of **TP means that the baseline observability is reduced. **
6. **Under a fixed PR rhythm, the unmasked QRST template may eliminate the target P wave together. **
7. **The goal of T wave processing is to keep the P boundary stable, not to pursue the cleanest residual. **
8. **Leads without information must not output boundaries or participate in fusion. **
9. **Baseline, T reconstruction and Ta ambiguity must be entered into uncertainty. **
10. **Explicitly reject the judgment when it cannot be identified, which is better than outputting wrong but high-confidence results. **

---

<a id="20-ecgfeat-实施状态2026-07-28"></a>
## 20. ecgfeat implementation status (2026-07-28)

The project items in this guide have completed the first round of implementation in `feature_extraction/ecgfeat`, including:

- Acquisition synchronization, duplicate channels, abnormal gain/filtering and lead configuration quality control
- Fixed channel delay compensation with rollback protection
- Offline two-way/online one-way state space baseline
- Wideband accuracy branch, 35 Hz detection branch, noise covariance whitening and 1–2D SVD
- Primitive, T-reconstruction and T-space subspace three-branch of P-on-T
- Multi-method boundaries such as amplitude, DoG/derivative, tangent, energy change point and conditional area
- Empirical confidence intervals generated for baseline perturbations, method perturbations and cross-branch differences
- Unchained cluster fusion of at least 3 valid leads and 2 independent lead groups
- strict/robust double boundaries, four P states and structured rejection reasons
- Beat-to-beat pattern clustering, temporal jitter and CI expansion
- Compatible fields are written back after mutual verification with the existing consensus model

Implementation entries, data contracts, validation results, and remaining limitations are provided in
[`p_wave_robust_engine_implementation.md`](p_wave_robust_engine_implementation.en.md).

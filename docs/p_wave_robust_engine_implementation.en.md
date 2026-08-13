<!-- i18n-nav -->
[中文](p_wave_robust_engine_implementation.md) | [English](p_wave_robust_engine_implementation.en.md) | [日本語](p_wave_robust_engine_implementation.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="p-波稳健定界引擎实施与验证审计"></a>
# P-wave robust delimitation engine implementation and verification audit

Generation time: 2026-07-28

<a id="结论"></a>
## Conclusion

The first round of deterministic DSP transformation required in `docs/p wave doc` has been integrated into the `ecgfeat` main process.
The new implementation no longer treats a single consensus boundary as an unconditionally valid result, but simultaneously derives lead-by-lead evidence,
strict/robust boundaries, experience CI, P status, collection chain quality control and structured rejection reasons.

In frozen v3 agent evaluation of 200 LUDB records, full P-boundary coverage remains
93.3%. Compared with the previous version of Ta/TP guard, beat-level onset, offset and P time limit bias,
SD, MAE, and P95 absolute errors are all improved. Record level P timing mean difference improved from -15.03 ms to
-8.52 ms.

This is still not an official IEC/CSE compliance conclusion. LUDB participated in both development and parameter freezing, experience CI
Not yet calibrated in an external database; the record-level average difference after excluding 8 deviations is -10.05 ms, strictly speaking
Still 0.05 ms more than the ±10 ms proxy limit.

<a id="实现范围"></a>
## Implementation scope

<a id="采集链和同步"></a>
### Collection chain and synchronization

Added `feature_extraction/ecgfeat/acquisition_qc.py`:

- Estimating per-lead fixed delays with QRS derivative envelopes, supporting subsampling parabolic refinement.
- Only delay 8–20 ms, inter-beat MAD no more than 2.5 ms, corrected correlation coefficient no less than
  0.75 and the relevant improvement is at least 0.08 before automatically compensating.
- Re-detect QRS after compensation; complete rollback when beat number changes or QRS P95 displacement exceeds 12 ms.
- Check for duplicate channels, abnormal gains within groups, filtering inconsistencies and serious limb lead configuration errors.
- Obvious desynchronization that cannot be safely compensated blocks authoritative fusion with `CHANNEL_DESYNCHRONIZED`.

<a id="基线检测和空间处理"></a>
### Baseline, detection and spatial processing

Added `feature_extraction/ecgfeat/p_wave_engine.py`:

- Build a quiet observation mask from samples with P/QRS/T excluded and low activity.
- Estimating the baseline with a deterministic Kalman state space model of common drift plus lead-by-lead residuals.
- Offline uses forward/backward smoothing, and online only uses forward state to avoid future information leakage.
- Precision branch retains wider bandwidth; detection branch uses 35 Hz low pass.
- Estimating noise covariance from quiet samples and whitening multi-lead derivative activity.
- Choose a 1 or 2 dimensional spatial representation scaled by singular values ​​within the coarse P window.
- Don't fake local SNR when there are no verifiable quiet samples, and explicitly reduce baseline confidence.

<a id="p-on-tt-重建和-ta"></a>
### P-on-T, T reconstruction and Ta

- T reconstruction only learns from other shots with similar RR, and masks all P regions of the source and target shots.
- Template fitting allows only ±8 ms displacement, 0.97/1.00/1.03 stretch, and restricted robust affine transformations.
- Use cosine taper for subtractive edges; residual energy and second-order differential high-frequency ratio must be verified.
- Preserve original signal, verified T-reconstruction residuals and T-space subspace suppression branches in parallel.
- Reduce confidence or reject judgment when T reconstruction fails, is inconsistent across branches, or Ta offset is unrecognizable,
  Do not accept "the residuals are cleaner" as a basis for acceptance.

<a id="逐导联边界和不确定度"></a>
### Lead-by-lead boundaries and uncertainty

Each candidate is run simultaneously:

- 4%, 6%, 8%, 10%, 12% amplitude hysteresis
- Multi-scale derivatives/DoG active boundaries
- Least squares tangent
- Local energy change point
- Conditional area boundary in non-Ta scenarios
- Positive and negative baseline uncertainty perturbations

Output `PWaveLeadBoundary`, including onset/peak/offset, CI at both ends, sigma, independent confidence,
local SNR, quality score, baseline pattern, T reconstruction status, Ta/P-on-T flags, branch differences,
Usage methods and flags. Leads that are isoelectric, high-noise, or fail hard QC do not enter authoritative fusion.

<a id="多导联融合和拒判"></a>
### Multi-lead fusion and rejection

- Candidate clusters use fixed anchors and single chaining is prohibited.
- onset selects the earliest eligible cluster, and offset selects the latest eligible cluster.
- Requires at least 3 valid leads and 2 independent groups: limb, right precordial,
  left recorded.
- The weight of each lead is capped, and the total weight of each relevant lead group is capped again to avoid repeated counting of votes.
- strict boundary retains the earliest onset/latest offset after hard quality control; robust boundary uses group permissions
  Fractional fusion.
- The new robust model and the existing consensus model mutually confirm each other as two independent models. Existing models lack complete boundaries
  , the new candidate is reserved for audit but rejected with `MODEL_DISAGREEMENT`.
- onset gives equal weight to the two models; offset uses late-preserving fusion, and the later model weights
  87.5% for inhibition of proven P-time-limited systemic contraction.

Each beat outputs `PWaveBeatAssessment`, the status is:

- `P_PRESENT`
- `AF_LIKE`
- `ORGANIZED_ATRIAL_ACTIVITY`
- `OVERLAP_UNCERTAIN`

Supported reasons for denial include:

- `BASELINE_UNOBSERVABLE`
- `P_ON_T_UNRESOLVED`
- `P_OFFSET_TA_AMBIGUOUS`
- `INSUFFICIENT_INFORMATIVE_LEADS`
- `CROSS_LEAD_DISAGREEMENT`
- `CHANNEL_DESYNCHRONIZED`
- `LEAD_CONFIGURATION_INVALID`
- `MODEL_DISAGREEMENT`

<a id="拍间一致性与接口"></a>
### Inter-beat consistency and interface

- Establish deterministic beat-to-beat clusters based on onset/offset shape vectors relative to the R point.
- Expand CI with intra-cluster jitter only, without temporal smoothing at boundaries that might mask ectopic beats.
- `ECGFeatures.p_wave_assessments` Save the complete new contract.
- `metadata.acquisition_qc` and `metadata.p_wave_contract` save record-level audit summaries.
- Structured export `provenance` contains acquisition QC, P-contract and beat-by-beat assessment.
- Only robust boundaries that pass mutual verification are written back to the original
  `p_onset_consensus_index`/`p_offset_consensus_index`, maintain downstream compatibility.

<a id="ludb-200-条冻结版结果"></a>
## LUDB 200 frozen version results

The baseline is `ludb_p_boundary_analysis_ta_tp_guard_final/` and the final result is
`ludb_p_boundary_analysis_robust_engine_v3/`.

| Indicators | Old Version | Robust Engine v3 |
|---|---:|---:|
| Complete boundary pairing | 1295 | 1289 |
| Complete border coverage | 93.30% | 93.27% |
| Onset bias | +4.64 ms | +3.00 ms |
| Onset SD | 35.80 ms | 32.15 ms |
| Onset MAE | 22.89 ms | 18.98 ms |
| Onset P95 Absolute Error | 94.60 ms | 67.20 ms |
| Offset bias | -8.69 ms | -3.78 ms |
| Offset SD | 27.58 ms | 25.07 ms |
| Offset MAE | 17.89 ms | 15.04 ms |
| Offset P95 Absolute Error | 70.60 ms | 52.00 ms |
| P-duration bias | -13.33 ms | -6.78 ms |
| P-duration SD | 23.00 ms | 21.45 ms |
| P-duration MAE | 21.78 ms | 17.42 ms |
| P-duration P95 absolute error | 50.00 ms | 44.00 ms |
| Record-level P-duration bias | -15.03 ms | -8.52 ms |
| Recording P-duration SD | 16.68 ms | 14.52 ms |
| Record-level P-duration MAE | 18.66 ms | 13.49 ms |
| Bias after eliminating 8 items | -16.52 ms | -10.05 ms |
| SD after excluding 8 items | 12.56 ms | 11.71 ms |

All 200 records were processed successfully. The results show that the systematic timing contraction is reduced by about 6.55 ms, mainly
P endpoint error and long tail both decrease simultaneously; coverage only decreases by 6 beats and remains unchanged after rounding
93.3%.

<a id="自动化验证"></a>
## Automated verification

- Full test set: `959 passed, 2 skipped, 145 subtests passed`
- Added P engine special tests: independent lead group, strict/robust, baseline unobservable,
  P-on-T, AF overrule, parallel model mutual verification, sub-sampling delay compensation and rollback evidence
- True LUDB Evaluation: 200/200 Success

4 runtime warnings from the empty slice mean in the existing LUDB 104/111 regression; test passed,
No new failures were generated.

<a id="尚未关闭的验证缺口"></a>
## Verification gaps that have not yet been closed

1. Empirical CI only comes from threshold, method, baseline and beat-to-beat perturbations, nominal coverage has not been completed
   Calibrated; output is clearly labeled `empirical_perturbation_uncalibrated`.
2. LUDB has been used for parameter selection, and the second data set with lead-by-lead P annotation must be used after freezing the parameters.
   Do independent verification.
3. AF/atrial flutter, no P, P-on-T, high heart rate without TP, Ta strong interference and low amplitude P still need to be reported separately
   Stratified acceptance rate, false positive rate, and CI coverage.
4. The record-level trimmed bias is -10.05 ms. Although it only exceeds the agent limit by 0.05 ms, it cannot be written as
   Passed ±10 ms.
5. The new engine increases the amount of deterministic multi-branch calculations; it is also necessary to measure online latency and
   Worst run time.

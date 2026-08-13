<!-- i18n-nav -->
[中文](af_afl_f_F_wave_implementation.md) | [English](af_afl_f_F_wave_implementation.en.md) | [日本語](af_afl_f_F_wave_implementation.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="afafl-f-波与-f-波识别实现说明"></a>
# AF/AFL F-wave and F-wave identification implementation instructions

Ruleset: `clinical_rules.v2 / 2026.07.9`

<a id="目标"></a>
## Target

Expand the judgment of AF/AFL from simple RR irregularities and missing P waves to an auditable multi-evidence judgment:

- Atrial fibrillation: Irregular RR, lack of consistent conduction P waves, combined with evidence of broadband, poorly organized `f` waves of atrial fibrillation.
- Atrial flutter: After QRST subtraction, a narrow-band, organized atrial flutter `F` wave with consistent frequency was detected in multiple independent leads.
- Insufficient or conflicting evidence: Output `atrial_fibrillation_flutter_indeterminate`, not forced to choose one of the two.

<a id="信号处理"></a>
## Signal processing

1. Perform continuous rhythm QRST template subtraction on reliable I, II, III, V1, and V2 leads respectively.
   QRS maintains R alignment; ST-T segments are translated by beat-by-beat derivative correlation and fitted with robust affine fitting
   Current beat amplitude/offset. QRS→ST-T Use cosine gradients at both ends of the interface and template window to avoid hard joints.
2. Verify the template correlation coefficient, heart rate, residual energy after subtraction, high-frequency (second-order difference) energy ratio and
   Number of leads available. Verification is rejected when the residual high-frequency energy is amplified by more than `1.25` relative to the original waveform.
3. When the median RR is `<520 ms`, the fixed QRST template window may invade the next P area;
   Direct labeling of continuous residual paths is not available, without subtracting the results to find P-on-T.
4. Perform Welch PSD on the continuous atrial residuals for each lead instead of doing a single FFT after splicing different leads/beats.
5. Search for the F-wave fundamental frequency at 220–400 bpm and calculate:
   - Peak to atrial band background ratio;
   - Fundamental frequency and harmonic power concentration;
   - Spectral entropy;
   - Time domain repeatability and stability.
6. F-wave multi-lead candidates are formed when at least two leads and at least 40% of available leads agree within ±30 bpm.
7. `f` waves use high spectral entropy, low organization, and low temporal repeatability to form a support score; at least 60% of available lead support is required to form multi-lead concordance.

<a id="判定策略"></a>
## Determination strategy

| Output | Necessary evidence |
|---|---|
| `atrial_fibrillation` | RR-CV ≥ 0.15, no consistent P wave or verified strong f wave rejects pseudo P sequence, no strong F wave |
| `atrial_flutter` | Multi-lead F wave consistency, F wave confidence ≥ 0.60; the final clinical output also requires QRST subtraction verification to pass |
| `af_afl_indeterminate` | conflicting f/F evidence, RR-CV at 0.12–0.15 boundary, unverified strong f-wave and P-wave results contradictory, or F-wave confidence at 0.50–0.60 |
| `none` | Does not meet the above conditions |

RR can be irregular in atrial flutter with variable AV conduction, so the proven strong multilead F wave takes precedence over RR irregularities and is no longer unconditionally covered by AF.

<a id="输出证据"></a>
## Output evidence

`rhythm_inputs.af_afl` and `metadata.rhythm_analysis` now output:

- `atrial_rhythm_classification`
- `diagnostic_confidence`
- `probable_af` / `probable_flutter`
- `af_afl_indeterminate` / `indeterminate_reasons`
- `f_wave_confidence` / `f_wave_multilead_consensus`
- `F_wave_confidence` / `F_wave_multilead_consensus`
- `F_wave_rate_bpm`
- `flutter_supporting_leads`
- `fibrillatory_supporting_leads`
- Per-lead spectral and organizational evidence

Text report for `RHYTHM SUMMARY` simultaneously displays classification, RR-CV, P-wave ratio, f/F confidence, F-wave frequency, supporting leads, and QRST verification status.

<a id="验证边界"></a>
## Validate boundaries

These modifications improve algorithm interpretability and conservatism, but clinical diagnostic accuracy cannot be claimed based solely on current weak label data. Before official use, you still need:

- AF, typical atrial flutter, atypical atrial flutter and noise control set reviewed by experts item by item;
- Patient-level independent training/validation partitioning;
- Sensitivity, specificity, positive predictive value and AF/AFL confusion matrix;
- Specialized stress testing for atrial premature, ventricular premature, atrial tachycardia, pacing and electromyographic noise.

In addition, beat-by-beat amplitude/time-shift adaptation can only reduce the template mismatch caused by breathing and RR changes, but cannot solve
The indiscernible problem of "every beat P is buried in T with the same phase". This scenario must output residual
Not available/low confidence, template subtraction cannot be considered as recovered true P wave.

<!-- i18n-nav -->
[中文](ecgfeat_diagnostic_contract_v10.md) | [English](ecgfeat_diagnostic_contract_v10.en.md) | [日本語](ecgfeat_diagnostic_contract_v10.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecgfeat-诊断契约-v20260710"></a>
# ecgfeat Diagnostic Contract v2026.07.10

This version implements "measurement" and "diagnosis" in `docs/流程` separately: even if the diagnostic gating fails,
Obtainable measurements, quality evidence, and failure reasons are still returned, but unreliable measurements are no longer interpreted as negative or
Positive diagnosis. This project is still a research and auxiliary interpretation tool, not a medical device that has undergone regulatory verification.

<a id="项目特定例外"></a>
## Project specific exceptions

This implementation does not require the original sampling rate to be 500 Hz:

- The minimum supported input sample rate remains 100 Hz;
- You can use the original sampling rate or resample via `fs_internal`;
- The report records the original/internal sample rate, temporal quantization resolution and whether it is below 500 Hz;
- Below 500 Hz does not itself trigger a diagnostic stop;
- However, the report will mark that the sensitivity of the pacing nail may be limited, and "no pacing nail detected" cannot be interpreted as determining that there is no pacing.

<a id="已实现的输入和质量契约"></a>
## Implemented input and quality contracts

- Accepts full standard 12 leads, or 8 independent leads with `lead_names`
  `I, II, V1-V6`; the latter derives the remaining limb leads according to the Einthoven/Goldberger relationship.
- Supports `mV`, `uV`, `V` and digital quantities with `gain_uv_per_lsb`, which are internally unified to mV.
- Check NaN/Inf, duration, flat line, saturation, baseline drift, power frequency, EMG,
  pSQI, kSQI, bSQI, limb lead equation residuals, and lead placement abnormalities.
- Check the consistency between the main QRS detection and the independent multi-lead amplitude detection; stop diagnosis when there is serious inconsistency.
- The diagnostic gate status is `pass`, `partial` or `stop` and given
  `stop_reasons`, `partial_reasons` and diagnostic fields that allow continuation.
- Less than 3 beats, less than 8 seconds, critical quality failure, severe lead equation inconsistency, or suspected limb lead
  Diagnosis is stopped when the connection is reversed; complex polymorphic recording only retains quality, rhythm and ectopic beat analysis.

<a id="新增测量与-finding"></a>
## Add measurement and Finding

- HR min/max, RR mean/SD/CV, RMSSD, pNN50, Poincaré SD1/SD2 and RR entropy.
- Bazett, Fridericia, Framingham, Hodges QTc, JT/JTc limits retained at wide QRS.
- QRS-T included angle and R/S conversion area.
- U-wave amplitude, T-wave symmetry, PR segment offset, QRS notching/stuttering, VAT and ST slope.
- Independent three-state Finding: `TRUE`, `FALSE`, `UNKNOWN`. Must be when required input is missing
  `UNKNOWN` and enter `abstentions`, it is not allowed to silently convert to `FALSE`.

<a id="诊断与报告"></a>
## Diagnosis and reporting

- New or enhanced sinus rhythm, 2:1 atrial flutter review at approximately 150 beats/min, pacing rhythm and device abnormalities,
  Wide QRS Tachycardia, severe QTc, LVH/RVH extended criteria.
- Added or enhanced ST depression, de Winter, Wellens, left main type, original/improved
  Sgarbossa, Brugada type 1, hyper/hypokalemia cues, alternans, premature repolarization, pericarditis,
  Dextrocardia, S1Q3T3/right heart load, and digitalis effects.
- Supports incoming `prior_features` for sequence comparison: rhythm, PR, QRS, QT, ST, ECG axis and
  T wave polarity changes.
- Final statement with fixed confidence level `HIGH/MEDIUM/LOW/UNAVAILABLE`, priority `P0-P6`,
  `report_statement`, `human_review_required`, evidence, threshold, suppression reason and available
  SNOMED CT code.
- All reports come with supporting interpretation statements; High Risk, Technical Limitations, Borderline Results and Significant Sequence Change Mandatory
  Manual review.

<a id="api-示例"></a>
## API Example

```python
features = ECGFeatureExtractor(
    fs_internal=None,       # 使用原始采样率；也可指定内部采样率
    mains_freq="auto",      # 也可固定为 50 或 60
).extract(
    signal,
    fs,
    meta=PatientMeta(
        age=55,
        sex="male",
        amplitude_unit="uV",
        device="device-name",
        acquisition_time="2026-07-28T12:00:00Z",
    ),
    lead_names=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    prior_features=previous_features,
)

gate = features.metadata["diagnostic_gate"]
analysis = features.metadata["clinical_interpretation"]
```

Documentation requirements for CSE measurement verification, diagnostic category-by-diagnostic category external validation, and rare high risk still need to be completed before production or clinical validation.
Morphological ad hoc set, cross-device/filter/population robustness, calibration and prospective physician review studies.

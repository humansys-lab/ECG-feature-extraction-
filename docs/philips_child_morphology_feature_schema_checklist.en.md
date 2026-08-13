<!-- i18n-nav -->
[中文](philips_child_morphology_feature_schema_checklist.md) | [English](philips_child_morphology_feature_schema_checklist.en.md) | [日本語](philips_child_morphology_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# Philips Pediatric Morphology Feature Schema Checklist

<a id="结论先行"></a>
## Conclusion first

> 2026-07-12 Status update: The unified authority layer has supported adult/child routing, but it lacks versioning,
> Publicly traceable pediatric PR and ventricular hypertrophy percentile tables, `pediatric_pr_public_reference_table`,
> `pediatric_lvh_percentile_table` and `pediatric_rvh_percentile_table` will be clearly output
>`unavailable`. The existing DXL-inspired child threshold remains in the reference-only payload,
> It will not enter the authoritative final conclusion, nor will it output "normal" accordingly.

`philips_child_morphology.pdf` is a Philips DXL Chapter 4 Pediatric Morphology Analysis, ECG for ages **birth to under 16 years**. The biggest difference from adult morphology is that the normal range of children's ECG is highly dependent on age, and some rules also depend on gender.

The current measurement results and codes can support the first version of **pediatric morphology MVP**:

- patient age/sex
- P/QRS/T/ST axis
- per-lead P/QRS/T/ST amplitude and boundary
- P duration, P amplitude, PTF-V1
- Q/R/S/R'/S' amplitude
- QRS duration, QRS peak-to-peak, VAT, notch/slur/fQRS
- ST J-point/mid/80ms, T amplitude/polarity
- QT/QTc
- The interpretation layer already has `is_pediatric`, age-adjusted QRS axis, age-adjusted QRS width, pediatric QTc, LSH, BVH, pericarditis, early repolarization and other fields.

After this round of modifications, the regenerated full features JSON will clearly distinguish the adult / pediatric morphology context in `morphology_inputs`:

- `record.algorithm_age_group` will output `adult` or `pediatric`; pediatric from birth to under 16 years old.
- When the age is missing or invalid, it defaults to adult according to Philips documentation and is marked by `record.age_defaulted_to_adult_algorithm`.
- `derived_facts.adult` Output adult threshold context.
- `derived_facts.pediatric` output age bucket, pediatric axis / QRS / QTc thresholds, child dextrocardia / RAE / BBB / hypertrophy / ST-T / QT rules required threshold context.

But it cannot fully satisfy the Philips-style pediatric morphology engine:

- Missing Davignon Appendix A's complete table of age-stratified normal values, especially the RVH/LVH voltage 98th percentile thresholds.
- RBBB/IRBBB/LBBB requires R'/terminal 40ms/QRS component duration, currently only partially approximated.
- RVH needs synthesized horizontal-plane vector / terminal angle to assist in distinguishing mild RVH vs IRBBB, which is currently not available.
- P-wave biphasic/notched/initial-terminal components are still incomplete as in adults.
- The output layer already has a unified statement/status/evidence/suppression contract; children's percentile dependency rules are intentionally undecidable because the public table is not versioned, and the complete Philips code/downgrade matrix has not yet been implemented.
- congenital heart defects only gives combination ideas, there are currently no structured congenital heart defects rules.

So the judgment is: **The current feature JSON can support adult/child diversion and partial child measurement, but the authority layer of the public standards will conservatively reject the missing table rule; complete Philips Chapter 4 still requires a legally traceable reference table, terminal morphology and proprietary statement matrix, which cannot be impersonated by existing reference heuristics. **

<a id="文档范围"></a>
## Document scope

This list is compared to the local PDF:

- `/home/chtmedgemma/projects/ecg_gemma/philips_child_morphology.pdf`

Documentation coverage:

- pediatric routing and age handling
- dextrocardia
-RAE/LAE/BAE
- age-adjusted QRS axis
- age-adjusted VCD / RBBB / IRBBB / LBBB / LAFB
-RVH/LSH/LVH/BVH
- low voltage / COPD pattern
- Q wave abnormality/MI
- ST depression / T wave abnormality / repolarization abnormality
- ST elevation / pericarditis / early repolarization
- tall T
- pediatric QTc / electrolyte disturbance
- congenital heart defects

<a id="当前代码与-json-状态"></a>
## Current code and JSON status

There are many pediatric extensions in the current code layer:

- `PEDS_MAX_AGE_YEARS = 16.0`
- pediatric dextrocardia thresholds
- pediatric RAE P amplitude threshold `0.20 mV`
- age-dependent QRS duration normal limits
- age-dependent QRS axis tables
- pediatric RBBB R' amplitude threshold
- pediatric QTc age/sex thresholds
- LSH and BVH helpers
- pericarditis age gate 5-15 years
- early repolarization age gate 13-15 years
- `ECGInterpretation.is_pediatric`
- `ECGInterpretation.lsh_suspected`
- `ECGInterpretation.bvh_suspected`
- `ECGInterpretation.pericarditis_suspected`
- `ECGInterpretation.early_repolarization_suspected`

However, the currently existing sample JSON (such as `JS00029_features.json`, `JS00024_features.json`, `JS00007_features.json`) may still be an old export, and most of the patients are adults, so:

- Older JSON may not contain the latest `morphology_inputs.derived_facts.pediatric`
- Older JSON may not contain the latest `record.algorithm_age_group`
- Adult samples are not suitable for validation of pediatric routing
- Regenerate full features JSON to include the latest schema

<a id="状态定义"></a>
## State definition

- `OK`: Currently the full features JSON / model are provided and can be derived directly or stably.
- `Partial`: There are relevant fields or first version rules, but missing age thresholds, fine measurements, rule evidence or new exports.
- `Missing`: No directly usable measurement fields or structures are currently found.

## Feature Checklist

| Module | Philips child morphology required features | Currently available fields/codes | Status | Gaps/Risks |
| --- | --- | --- | --- | --- |
| Pediatric routing | age From birth to under 16 years old pediatric algorithm | `metadata.patient_meta.age`; `is_pediatric` | OK | If age is missing or invalid, the document requires pressing adult default and printing the hypothesis statement; currently there is no statement evidence. |
| Patient sex | Over 13 years old QTc Thresholds are divided by male/female | `metadata.patient_meta.sex` | OK | unknown fallback is required when sex is missing. |
| Age bins | 0-23h, 1-3d, 4-6d, 7-29d, months/years segmentation | The code is QRS axis/QRS duration lookup tables | Partial | RVH/LVH voltage still lacks the complete Appendix A normal value table. |
| Dextrocardia | P axis 90-180 deg | `global_features.p_axis_deg`; `_peds_dextrocardia()` | OK | Available. |
| Dextrocardia | Lead I or V6 negative P | `p_amp_mv` | OK | Only look at the P amplitude symbol, missing the more refined P morphology. |
| Dextrocardia | I and V6 large S > 0.6 mV | `s_amp_mv`; `PEDS_DEXTRO_S_MV` | OK | Available. |
| Dextrocardia | P amplitude III > II | `p_amp_mv` | OK | Available. |
| Dextrocardia | bypass remainder | `dextrocardia_suspected` | Partial | Rules are present; pediatric reason for `statement_evidence.stop_further_interpretation` is missing. |
| RAE | P duration >= 60 ms | `p_dur_ms` or P bounds aggregation | OK | Available. |
| RAE | P amplitude >= 0.20 mV | `p_amp_mv`; `PEDS_RAE_P_AMP_MV` | OK | The current common P morphology still mainly uses adult 0.24 mV, and it needs to be confirmed whether the pediatric path fully applies 0.20 mV. |
| RAE | V1 biphasic P supports probable RAE | Missing `p_biphasic_flag` | Missing | The same gap as adults. |
| LAE | limb P duration > 110 ms + amplitude > 0.10 mV | `p_dur_ms`, `p_amp_mv` | OK | Available. |
| LAE | notched P adds significance | Missing `p_notch_flag` | Missing | LAE score cannot be restored completely. |
| LAE | V1 negative P terminal duration/amplitude/area | `ptf_v1_mv_ms` | Partial | PTF-V1 Yes; terminal duration/amplitude/area is not split. |
| BAE | RAE + LAE high severity -> biatrial hypertrophy | `p_morphology_class`, `rae_leads`, `lae_*` | Partial | Missing statement severity and high-severity candidate evidence. |
| QRS axis | age-adjusted LAD/RAD | `qrs_axis_deg`; `_peds_classify_qrs_axis()` | OK | The code already has Table 4-1 to 4-4 lookup. |
| QRS axis | borderline LAD/RAD within 15 deg zone | `_peds_classify_qrs_axis()` | OK | `borderline_LAD` / `borderline_RAD` can be output. |
| VCD | age-adjusted QRS normal limit | `qrs_ms`; `_PEDS_QRS_NORMAL_MS` | OK | The code already has Table 4-5. |
| VCD | >110% normal -> borderline IVCD | `_peds_classify_qrs_width()` | OK | Available. |
| VCD | >120% normal -> nonspecific IVCD / BBB range | `_peds_classify_qrs_width()` | OK | Available. |
| RBBB | VCD for age + RSR' or pure R in V1 | `r_prime_amp_mv`, `r_amp_mv`, Q/S amplitude | Partial | RSR'/pure R can be judged roughly; missing R' duration and no-negative-component fact. |
| RBBB | R' duration >= 20 ms and amplitude >= 0.15 mV | `r_prime_amp_mv`; missing duration | Partial | Amplitude is present; R' duration is not measured. |
| IRBBB | RBBB-like morphology but QRS <120% normal | `_peds_classify_qrs_width()` | Partial | Missing synthesized vector differentiation from RVH. |
| LBBB | prolonged QRS for age | `qrs_ms` | OK | Available. |
| LBBB | terminal 40ms QRS axis -90 to 90 clockwise | terminal 40ms axis not found | Missing | The key underlying measurement is missing. |
| LBBB | short/absent S in I/aVL/V5/V6 and small/absent R in V1-V3 | Q/R/S amplitude can be roughly judged | Partial | S duration is missing; the small/absent threshold needs to be configured. |
| LAFB | absent LBBB + mean QRS axis -60 to -90 | `qrs_axis_deg`, `bundle_branch_block` | OK | Code already has pediatric LAFB range. |
| RVH | bypass if RBBB | `bundle_branch_block` | OK | Available. |
| RVH voltage | age-dependent RVH voltage criteria, 6 age groups, 24 conditions | Partial Q/R/S/R' amplitudes are | Partial | Missing complete Appendix A 98th percentile thresholds. |
| RVH | R/R' in V1/V2, S in V6, R/S ratios, QR pattern V1 | `r_amp_mv`, `r_prime_amp_mv`, `s_amp_mv`, `q_amp_mv` | Partial | Amplitude is available; R' duration, QR pattern object, age-specific threshold table are missing. |
| RVH | upright T in V1 for age >48h and <9y, absent inverted T in V5/V6 | `t_amp_mv`, `t_polarity`, age | OK | can be derived. |
| RVH | RAD / borderline RAD support | pediatric `qrs_axis_class` | OK | Available. |
| RVH vs IRBBB | synthesized horizontal plane terminal angle | No direct field found | Missing | Document explicitly for mild RVH vs IRBBB. |
| LSH | prominent R in V1 + Q in V5/V6 | `r_amp_mv`, `q_amp_mv`; `_peds_lsh()` | Partial | Currently uses a fixed proxy threshold, not a complete 98th percentile table. |
| LVH | bypass if RBBB or LBBB | `bundle_branch_block` | OK | The code already has pediatric bypass. |
| LVH voltage | R in I/II/aVL/aVF/V5/V6; S in V1/V2; RV6+SV1; prominent Q | Q/R/S amplitude | Partial | Data available; incomplete age-dependent 98th percentile thresholds. |
| LVH support | LAD and LAE support LVH | `qrs_axis_class`, LAE fields | OK | Available. |
| LVH strain | I/aVL/V4/V5/V6 ST/T repolarization pattern | `st_mid_mv`, `st_on_mv`, `t_amp_mv`, `t_polarity` | Partial | Data available; rules require pediatric-specific pattern evidence. |
| BVH | RVH + LVH high severity | `rvh_class`, `lvh_class`, `bvh_suspected` | Partial | code has summary flag; missing candidate severity evidence. |
| BVH | R V1 >1.0 mV with LVH | `r_amp_mv` | OK | Available. |
| BVH | RVH + Q V6 >10ms and >0.07mV + R V6 >1.0mV | Q duration approx, Q/R amplitude | Partial | Q duration is approximate and lacks independent Q onset/offset. |
| BVH | R+S >6.0mV in >=2 of V2/V3/V4 | Q/R/S amplitude | OK | Derivable. |
| BVH suppression | BVH suppresses individual RVH/LVH | `bvh_suspected` | Partial | The code does some suppression; statement-level suppressed_by is missing. |
| Low voltage | frontal/precordial QRS peak-to-peak | Can be derived from Q/R/S; `low_voltage_class` | OK | Same as adults. |
| COPD pattern | low voltage + rightward P/QRS + RAE | low voltage, axis, RAE | Partial | Can be derived; missing pediatric statement evidence. |
| Q wave abnormality | large Q waves in 2 leads of a group | `q_amp_mv`, `pathological_q_leads` | Partial | group count can be derived; missing pediatric-specific statement. |
| Q wave infarct | Q > 1/5 R amplitude suggests infarct | `q_amp_mv`, `r_amp_mv` | OK | Available. |
| ST depression | anterior/lateral/inferior groups | `st_depression_leads`, territories | OK | Available. |
| ST depression | >0.20 mV in one group -> nonspecific ST depression | `st_on_mv`, etc. | Partial | Data available; current general ST threshold is lower, pediatric-specific threshold output is required. |
| ST depression | tachycardia -> rate-related statement | `heart_rate_bpm`, age | Partial | The document does not provide a pediatric tachycardia table; the current adult-like `190-age` is insufficient. |
| ST suppression depression | hypertrophy or VCD suppresses | RVH/LSH/LVH/BVH/VCD fields | Partial | Missing statement-level suppression output. |
| T abnormality | inverted T in anterior/lateral/anterolateral/inferior groups | `t_amp_mv`, `t_polarity` | OK | Can be derived. |
| T abnormality | inverted T amplitude >1.0mV in >=2 leads in group -> tall T abnormality | `t_amp_mv`, lead groups | OK | Can be derived. |
| T abnormality secondary to RVH/LVH | RVH/LVH flags + inverted T groups | fields available | Partial | Missing final statement evidence. |
| Repolarization abnormality | ST depression + inverted T combination | ST/T facts | Partial | Data available; combined severity/candidate engine missing. |
| ST elevation | all leads tested, >0.15mV suggests probable normal variation | `st_elevation_leads`, `st_on_mv` | Partial | Data available; pediatric threshold and statement required. |
| ST elevation suppression | hypertrophy and VCD suppress | RVH/LSH/LVH/BVH/VCD fields | Partial | Missing statement-level suppression output. |
| Pericarditis | ST elevation in anterior/lateral/inferior groups, age 5-15 | `pericarditis_suspected` | OK | The code already has summary flag. |
| Early repolarization | nonspecific ST elevation, no T inversion, age 13-15 | `early_repolarization_suspected` | OK | The code already has summary flag. |
| Tall T | T >1.20mV or >0.50mV and >0.5 QRS peak-to-peak | `t_amp_mv`, QRS peak-to-peak | OK | Available. |
| QT short | QTc <340ms borderline short | `qtc_bazett_ms`, `qtc_class` | OK | The code already has a pediatric classifier. |
| QT borderline prolonged | >450 <5y; >454 age 5-12; >458 boys >=13; >465 girls >=13 | age/sex/QTc | OK | The code is available. |
| QT prolonged | borderline threshold +20ms | age/sex/QTc | OK | Code already exists. |
| QT suppression | RVH/LSH/LVH/BVH/VCD -> prolonged QT secondary to wide QRS | fields available | Partial | Currently possible normalizes/suppresses class, missing printed secondary statement evidence. |
| Electrolyte | QTc <310 hypercalcemia; >520 hypocalcemia/hypokalemia with ST depression + positive T | QTc, ST, T | OK | The code already has a hint. |
| Congenital defects | combinations of atrial/ventricular hypertrophy, VCD, axis, QRS morphology | summary facts available | Partial | The document does not have full rules; there is currently no CHD candidate engine. |

<a id="和当前-morphologyinputs-的关系"></a>
## Relationship with current `morphology_inputs`

Top-level `morphology_inputs` has been added in the last round, among which `record/global/leads/derived_facts/statement_evidence` is also helpful for children's rules:

- `record.age_years` and `record.sex` can be used for pediatric routing.
- `global.qrs_duration_ms`, `global.qrs_axis_frontal_deg`, `global.qtc_bazett_ms` supports age-adjusted QRS/QTc.
- `leads.*.p/qrs/st/t` supports dextrocardia, RAE/LAE, VCD/BBB, hypertrophy, ST/T/QT.
- `derived_facts` already has adult-style axis/VCD/RVH/LVH/ST/T/QT blocks.

The `derived_facts.pediatric` sub-block has been added in this round to avoid mixing child thresholds with adult facts. The regenerated structure is similar to:

```json
"pediatric": {
  "is_pediatric": true,
  "age_bucket": "5-7_years",
  "route_selected": true,
  "axis_limits": {
    "lad_threshold_deg": -10,
    "borderline_lad_upper_deg": 1,
    "rad_threshold_360_deg": 201,
    "borderline_rad_threshold_360_deg": 160
  },
  "qrs_duration_limits": {
    "normal_limit_ms": 88,
    "borderline_ivcd_ms": 96.8,
    "nonspecific_ivcd_ms": 105.6
  },
  "qtc_limits": {
    "short_ms": 340,
    "borderline_prolonged_ms": 454,
    "prolonged_ms": 474,
    "severe_ms": 520
  },
  "hypertrophy_thresholds": {
    "rvh_lvh_age_percentile_tables": {
      "available": false,
      "reason": "appendix_a_percentile_tables_not_digitized"
    },
    "lsh_r_v1_prominent_mV": 1.0,
    "bvh_rs_sum_v2_v3_v4_mV": 6.0
  },
  "repolarization_thresholds": {
    "st_depression_threshold_mV": 0.20,
    "st_elevation_normal_variation_threshold_mV": 0.15,
    "pericarditis_age_gate": "5-15",
    "early_repolarization_age_gate": "13-15"
  }
}
```

<a id="推荐优先级"></a>
## Recommended priority

- `P0`: Completed. Add `adult` / `pediatric` sub-blocks under `morphology_inputs.derived_facts`, exposing algorithm age route, age bucket, pediatric axis/QRS/QTc thresholds and existing pediatric summary flags.
- `P1`: Complete the pediatric RVH/LVH voltage threshold table, which is the Appendix A normal measurement values ​​cited in the document.
- `P2`: Add R' duration, S duration, terminal 40ms QRS axis, horizontal terminal vector to improve RBBB/IRBBB/LBBB/RVH differentiation ability.
- `P3`: Expand existing unified statement/suppression engine after obtaining publicly traceable and versioned pediatric reference tables; until then remain `unavailable`.
- `P4`: To create a congenital heart defects rule layer, you need to sort out the specific congenital heart disease combination rules first, otherwise you can only output the "Rule Not Implemented" placeholder.

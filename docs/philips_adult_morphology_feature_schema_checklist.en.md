<!-- i18n-nav -->
[中文](philips_adult_morphology_feature_schema_checklist.md) | [English](philips_adult_morphology_feature_schema_checklist.en.md) | [日本語](philips_adult_morphology_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# Philips Adult Morphology Feature Schema Checklist

<a id="结论先行"></a>
## Conclusion first

> 2026-07-12 status update: top-level has been added to the current export
> `clinical_interpretation` (`clinical_rules.v1`/ruleset `2026.07.1`). This is public
> Authoritative rule layer based on AHA/ACCF/HRS, ESC and Fourth Universal Definition of MI;
> `interpretation` and `statement_engine` are reserved and clearly marked as `reference_only`;
> The Glasgow interpretation layer has been removed from the current runtime and export contracts. The project does not equate to, nor does it claim to
> Reproduce Philips/DXL proprietary algorithm.
> Each unified rule outputs status, required/missing inputs, evidence, thresholds, source
> Same as suppression; lack of necessary evidence will return `unavailable` and will not be used as a basis for positivity or "normal".

The current full features JSON (such as `JS00029_features.json`) can already support a version of **Adult morphology MVP**, especially these modules:

- QRS/P/T/ST frontal axis and QRS-T angle
- PR/QRS/QT/QTc/JT and other global intervals
- per-lead P/QRS/T boundary, P/QRS/T amplitude, ST J-point/mid/80ms
- Q/R/S/R'/S' amplitude, QRS area, notch/slur, VAT, fQRS score
- T amplitude, T polarity, TPE, U wave flag
- PTF-V1, P duration, P area
- LVH voltage, low voltage, ST depression/elevation, pathological Q, R progression, tall T and other preliminary explanation fields

After regenerating full features JSON, top-level `morphology_inputs` will be additionally included to organize the current measurements into the input schema for the morphology rule engine. But it cannot fully satisfy the Philips-style adult morphology engine compiled in `要点adult_morphology.pdf`. The main gap is not "no measurement at all", but:

- Structured candidate statements and unified rule auditing are available, but only cover the currently implemented set of public rules; the complete Philips code matrix and downgrade/replacement semantics are not yet covered.
- P-wave has obvious gaps in fine morphology: `is_notched`, `is_biphasic`, initial/terminal duration/amplitude/area are only partially approximate.
- QRS component duration incomplete: Q duration is currently approximate, R'/S'/Q/S duration not measured directly.
- The terminal / horizontal / initial QRS direction required by Dextrocardia, BBB, RVH, MI, etc. is not output as an explicit fact.
- ST shape and slope units do not exactly match Original text: Currently there are `st_slope_mv_per_ms` and `upsloping/horizontal/downsloping`, the original text requires degrees and straight/concave-up/concave-down.
- ST Map, Cabrera coordinates, V4R/V7-V9 extended leads, culprit artery outputs are still missing.
- Insufficient drug/clinical input: `meds` has a location but the current sample is mostly `null`, with no independent `clinical_dx_codes` / `rx_codes` rule context.

So the verdict is: **The unified MVP of the public guidelines is ready to run and output as a final rule; the Philips/DXL style results are for reference only. A complete replica of Philips adult morphology still lacks a proprietary code matrix, some underlying morphological facts, and proven proprietary arbitration logic. **

<a id="文档范围"></a>
## Document scope

This listing compares two local PDFs:

- `/home/chtmedgemma/projects/ecg_gemma/philips_adult_morphology.pdf`
- `/home/chtmedgemma/projects/ecg_gemma/要点adult_morphology.pdf`

`要点adult_morphology.pdf` is an engineered specification compiled from Philips Adult Morphology Analysis, covering dextrocardia, atrial abnormalities, QRS axis, VCD/BBB, RVH, LVH, low voltage/COPD, MI, ST/T/QT, electrolyte/drug effects and suppression engine.

<a id="数据层级判断"></a>
## Data level judgment

| Data level | Can it support adult morphology analysis | Description |
| --- | --- | --- |
| `.txt report` | Insufficient | Suitable for human reading, lacking complete per-lead/per-beat morphology facts, candidate statements and suppression evidence. |
| Current old full features JSON, such as `JS00029_features.json` | Can support MVP, a large number of fields are available | Included `beats`, `beat_features`, `representative_leads`, `global_features`, `metadata`, `interpretation`; but not `morphology_inputs`. |
| Regenerated full features JSON | More suitable for downstream rule engines | The current `to_dict(ECGFeatures)` will add `rhythm_inputs` and `morphology_inputs` and retain the original measurements. |
| `build_structured_payload()` thin payload | Not enough | Does not export complete `beat_features`, `representative_leads`, candidate evidence, and is not suitable for input into the morphology rule engine. |

<a id="js00029-当前-json-快照"></a>
## JS00029 Current JSON snapshot

Currently `JS00029_features.json` is an old export, confirmed to contain:

- top-level: `fs`, `quality`, `beats`, `beat_features`, `representative_leads`, `groups`, `global_features`, `metadata`, `interpretation`
- beats: 14
- beat_features: 168, 12 leads x 14 beats
- beat-level fields: P/QRS/T bounds, `qt_ms`, `pr_ms`, `qrs_ms`, P/Q/R/S/T amplitude, ST J/mid/80ms, `j_index`, `delta_present`, `jt_ms`, confidence, R'/S', QRS peak count, QRS notch count, VAT, P/T duration/area, T polarity, U wave, QRS slur, ST slope, TPE, ST morphology, fQRS, PTF-V1, consensus QT/PR
- representative_leads: 12 leads, each with `params` and `variance`
- global_features: HR, atrial rate, PR/QRS/QT/QTc, P/QRS/T/ST axis, QT dispersion, pacing, PTF-V1
- metadata: input/internal fs, lead_order, n_beats, qrs_detector, representative_group_id, patient_meta
- interpretation: old version morphology flags, such as `pathological_q_leads`, `q_wave_territories`, `r_progression_class`, `st_elevation_leads`, `st_depression_leads`, `lvh_voltage_criteria`, `lvh_class`, `low_voltage_class`, `rvh_suspected`, `tall_t_leads`

The current old JSON does not contain:

- top-level `morphology_inputs`
- top-level `rhythm_inputs`
- structured candidate interpretations
- statement `code/category/severity/reason/suppressed_by`
- `dextrocardia_suspected`, `rvh_class`, `copd_pattern`, `qtc_electrolyte_hint`, `posterior_mi_suspected`, `st_rate_related` and other newer interpretation extension fields

After regenerating full features JSON, it will additionally include top-level `morphology_inputs`, including `record`, `global`, `leads`, `derived_facts`, and `statement_evidence`.

<a id="状态定义"></a>
## State definition

- `OK`: Currently full features JSON are provided, either directly or as stable derivatives.
- `Partial`: There are relevant fields, but they require approximation, aggregation, re-export, or missing evidence chains/fine components.
- `Missing`: No directly usable measurement fields or structures are currently found.

## Feature Checklist

| Module | `要点adult_morphology.pdf` Required Features | Currently Available Fields | Status | Gap/Risk |
| --- | --- | --- | --- | --- |
| Basic input | Sample rate, lead order, record length | `fs`, `metadata.input_fs`, `metadata.internal_fs`, `metadata.lead_order` | OK | Record length is not always explicitly output as `duration_sec`. |
| Patient information | age, sex | `metadata.patient_meta.age`, `metadata.patient_meta.sex` | OK | Age/sex yes; full age/sex correction table still needs to be configured. |
| Clinical input | clinical diagnosis codes | No stable field found | Missing | Mitral Valvular Disease and other suppression conditions cannot be used reliably. |
| Drug input | rx_codes/digitis | `metadata.patient_meta.meds` | Partial | The field position exists, but the sample is mostly `null`; there is no structured `rx_codes`. |
| Quality control | lead reliability, wave-specific reliability | `quality.*.reliable_for_p/qrs/t/qt`, `beat_measurement_reliable` | OK | Can be measured by gate P/QRS/T/ST/QT. |
| Output model | code/category/severity/reason/suppression | `clinical_interpretation.final_statements/domains/suppressed_statements/conflicts`; there is also reference-only `statement_engine` | Partial | The public rules have been fully audited; the complete Philips code/category/downgrade/replace matrix is not covered. |
| Dextrocardia | P axis rightward | `global_features.p_axis_deg` | OK | Can be judged. |
| Dextrocardia | QRS frontal axis rightward | `global_features.qrs_axis_deg` | OK | Can be determined. |
| Dextrocardia | horizontal QRS direction rightward | Can be roughly derived from V5/V6 R/S | Partial | Not explicit `qrs_axis_horizontal_direction`. |
| Dextrocardia | V5/V6 small QRS | Can be derived from Q/R/S peak-to-peak | Partial | The original text does not give a threshold; currently needs to be configured. |
| Dextrocardia | early stop / bypass remaining morphology | The new model has `dextrocardia_suspected`, the old JSON has none | Partial | Lack of structured control-flow evidence. |
| RAE | limb P duration >= 60 ms | `beat_features[*].p_dur_ms`; current code can be aggregated | OK | representative params P duration is not listed directly and needs to be aggregated from beat_features. |
| RAE | limb P amplitude >= 0.24 mV | `p_amp_mv` | OK | Available. |
| RAE | V1 biphasic P | Not found `p_biphasic_flag` | Missing | Only PTF-V1/amplitude approximation, biphasic morphology cannot be completely determined. |
| LAE | limb P duration > 110 ms + amp > 0.10 mV | `p_dur_ms`, `p_amp_mv` | OK | Polymerizable. |
| LAE | notched P wave | Not found `p_notch_flag` | Missing | The original article pointed out that notched P increases the weight and the judgment cannot be stable at present. |
| LAE | V1 terminal negative P duration/amplitude/area | `ptf_v1_mv_ms` | Partial | PTF-V1 Yes, but initial/terminal duration/amplitude/area is not completely split. |
| BAE | RAE + LAE high severity combination | `interpretation.p_morphology_class`, `rae_leads`, `lae_*` | Partial | There are currently summary results, but the candidate statement severity and combination evidence are missing. |
| QRS axis | LAD/RAD basic range | `global_features.qrs_axis_deg`, `interpretation.qrs_axis_class` | OK | Can support basic judgment. |
| QRS axis | age/sex-adjusted normal range | age/sex available | Partial | The original text does not give a complete table and needs to be configured. |
| VCD/BBB | QRS duration 100/110/120 ms layered | `global_features.qrs_ms`, per-lead `qrs_ms` | OK | Available. |
| Fascicular block | LAFB/LPFB axis ranges | `global_features.qrs_axis_deg`, `interpretation.qrs_axis_class` | Partial | Mainly can be judged by axis, the complete counterclockwise/clockwise direction needs to be regularized. |
| RBBB | terminal QRS rightward; I/aVL/V6 terminal negative, V1 positive | R/S/R'/S' amplitude, QRS signed area New export available | Partial | terminal portion direction not directly measured; old JSON missing `qrs_signed_area`. |
| LBBB | terminal QRS leftward; I/aVL/V6 positive, V1 negative | R/S/R'/S' amplitude, QRS signed area New export available | Partial | Missing clear terminal force fact. |
| R'/S' morphology | R' / S' amplitude | `r_prime_amp_mv`, `s_prime_amp_mv` | OK | Amplitudes available. |
| R'/S' duration | R' / S' duration | `r_prime_duration_ms`, `s_duration_ms`, `s_prime_duration_ms`, `r_duration_ms` | OK | Unified BBB/RVH rules will mark the missing duration as `unavailable`. |
| RVH | V1 prominent R/R' | `r_amp_mv`, `r_prime_amp_mv` | Partial | R/R' amplitude available; R' duration missing. |
| RVH | V1 positive component > negative component | Can be approximated from Q/R/S | Partial | No direct `positive_component_mV` / `negative_component_mV`. |
| RVH | I/V6 prominent Q/S/S' duration + amp | `q_amp_mv`, `s_amp_mv`, `s_prime_amp_mv` | Partial | Amplitude yes; Q/S/S' duration missing. |
| RVH | RV strain pattern | `st_on_mv`, `t_amp_mv`, `t_polarity`, `st_depression_leads` | OK | Checkable II/aVF/V1-V3 ST depression + inverted T. |
| RVH | graded consider/probable/definitive | The new model has `rvh_class`; the old JSON is mostly `rvh_suspected` | Partial | The current old JSON lacks grading; lacks complete evidence. |
| LVH voltage | R aVL, R I + S III, R V5/V6, S V1/V2 + R V5/V6 | `r_amp_mv`, `s_amp_mv`, `lvh_voltage_criteria` | OK | Can support Table 3-1 voltage items. |
| LVH Cornell | R aVL + S V3, Cornell product | `r_amp_mv`, `s_amp_mv`, `global_features.qrs_ms` | OK | Available. |
| LVH age/axis suppression | age < 35, right axis | patient age, `qrs_axis_class` | OK | Data available. |
| LVH add-on flags | LAD, LAE, LV strain, QRS/VAT wide | axis, LAE flags, ST/T, VAT | Partial | Determinable; but no structured flags_set. |
| LVH Table 3-2/3-3 outputs | LVHV/LVHCNV/LVHCNP, etc. code + severity + reason | `lvh_voltage_criteria`, `lvh_class` | Partial | Missing code output and reason. |
| Low voltage | frontal/precordial QRS peak-to-peak | Can be derived from Q/R/S amplitude; `low_voltage_class` | OK | It is recommended to output `qrs_peak_to_peak_mV` directly. |
| COPD pattern | low voltage + rightward P/QRS + RAE | low voltage, axis, RAE available; new model has `copd_pattern` | Partial | old JSON missing `copd_pattern`; missing final statement evidence. |
| MI Q wave | Q amplitude, Q/R ratio | `q_amp_mv`, `r_amp_mv`, `pathological_q_leads` | OK | Available. |
| MI Q duration | inferior/lateral/anterior Q duration threshold | `q_onset`, `q_offset`, `q_duration_ms` | OK | If the credible Q time limit cannot be obtained, the unified rule returns `unavailable`, and the Q amplitude will not be used to replace the time limit. |
| MI Q wave area | anterior Q wave area | `q_area_mv_ms`, `initial_qrs_area_mv_ms` | Partial | Direct Q area already exists; territory/threshold applicability still needs to be confirmed according to specific guidelines. |
| MI initial QRS axis | initial QRS axis leftward | direct field not found | Missing | Original text used to enhance inferior MI likelihood. |
| MI R progression | transition / poor progression | `r_progression_class`, `r_s_transition_lead` | OK | Available. |
| MI ST/T age estimation | T inversion depth, ST deviation magnitude | `t_amp_mv`, `st_*`, territories | Partial | Data available; missing infarct age severity/ranking rule output. |
| MI territories | inferior/lateral/anterior/anterolateral/posterior | `q_wave_territories`, ST territories, R progression | Partial | Existing territory rough; posterior/right-sided leads incomplete. |
| Culprit artery | RCA/LCx/LAD/LM-MVD inference | ST II/III/V2/V3/aVR available | Partial | Rules derivable; missing V4R/V7-V9 and structured culprit output. |
| Extended leads | V4R, V7, V8, V9 | `STANDARD_12_LEADS` only | Missing | Insufficient sensitivity for posterior wall/right ventricular infarction. |
| ST Map | Cabrera angle, ST deviation mm, shaded map | ST deviation Yes | Partial | Missing Cabrera mapping, polarity-adjusted value, drawing structure. |
| ST depression | J point, midpoint, J+80 ms | `st_on_mv`, `st_mid_mv`, `st_80ms_mv` | OK | Three key points are available. |
| ST depression | ST end / T beginning | `t.onset` available as T beginning | Partial | No explicit `st_end_mv`, needs to be derived from raw signal or T onset. |
| ST depression | slope degrees | `st_slope_mv_per_ms` | Partial | The unit is not degrees. |
| ST depression | shape straight/concave up/down | `st_morphology` = upsloping/horizontal/downsloping | Partial | The classification system does not match the original shape. |
| Rate-related ST depression | HR > 190 - age | `heart_rate_bpm`, age, `st_depression_leads`; new model has `st_rate_related` | Partial | old JSON is missing `st_rate_related`, missing statement. |
| T wave abnormality | T amplitude | `t_amp_mv` | OK | Available. |
| T wave abnormality | relative T/QRS amplitude | `t_amp_mv` + QRS peak-to-peak can be derived | OK | It is recommended to output ratio directly. |
| T wave abnormality | polarity positive/negative/flat/biphasic | `t_polarity` | Partial | Positive and negative; flat/biphasic is not clearly structured. |
| T axis / QRS-T angle | frontal T axis, QRS-T angle | `global_features.t_axis_deg`, `interpretation.qrs_t_angle_deg` | OK | Available. |
| ST-T repolarization combo | ST depression + T abnormality combined severity | ST/T fields Yes | Partial | Missing candidate category and severity combination. |
| ST elevation | J point and J+80 positive values | `st_on_mv`, `st_80ms_mv` | OK | Available. |
| ST elevation | injury/pericarditis/early repol | ST territories Yes; new model has pericarditis/early repol flags | Partial | Old JSON Missing new flags; Missing complete suppression/confounder list. |
| Tall T | positive T > 1.2 mV, or >0.5 mV and >0.5 QRS PP | `t_amp_mv`, `t_polarity`, QRS peak-to-peak, `tall_t_leads` | OK | Sufficient data; rules must limit positive T. |
| QT short/long | QTc thresholds 310/340/465/485/520 | `qtc_bazett_ms`, `qtc_fridericia_ms`, `qtc_class` | OK | Available. |
| Electrolyte hints | hypercalcemia/hypocalcemia/hypokalemia | QTc + ST depression + positive T | Partial | Data available; old JSON missing `qtc_electrolyte_hint`. |
| Digitalis effect | Rx digitalis + ST/T/QT pattern | `patient_meta.meds` | Partial | Drug input is unstable, missing `rx_codes`. |
| Suppression rules | RVH/LVH/BBB/infarct/drug/electrolyte suppress ST/T/QTc | `clinical_interpretation.suppressed_statements`, per-rule `suppressed_by` | Partial | Covered lead reversal, BBB/IVCD and selected hypertrophy/ischemia rules; not covering the full Philips suppression matrix. |
| Rule audit | all rules reason output | `clinical_interpretation.domains.*` contains status/evidence/thresholds/source/missing_inputs | OK (currently public rule set) | It only means that the implementation rules are auditable, and does not mean that clinical verification or proprietary algorithm equivalence verification has been done. |

<a id="实用判断"></a>
## Practical judgment

The current full features JSON are already suitable for **MVP adult morphology rule engine**:

- dextrocardia rough judgment
- LAD/RAD/QRS axis
- QRS duration layered, basic IVCD/RBBB/LBBB
- low voltage
- LVH voltage / Cornell / Sokolow-Lyon
- RVH coarse screen
- pathological Q wave territory coarse screening
- ST elevation/depression territory coarse screening
- T wave abnormality coarse screen
- tall T
- QTc short/long with basic electrolyte hint

It is not currently suitable for full reproduction Philips Adult Morphology:

- Fine grading of P notched/biphasic/terminal component of RAE/LAE/BAE
- Fine judgment of terminal force and component duration of BBB/RVH
- Complete evidence chain of MI age and culprit artery
- ST Map’s Cabrera spatial display
- V4R/V7-V9 extended lead logic
- Complete LVH Table 3-2/3-3 statement code matrix
- Global suppression/downgrade/replace/ranking rules
- reason, severity, suppressed_by, confidence_rank of each output statement

<a id="已新增的最小目标-schema"></a>
## Added minimum target Schema

full features JSON A new top-level `morphology_inputs` has been added like rhythm, which does not replace the original measurements, but organizes existing fields into rule-engine-friendly facts.

### `morphology_inputs.record`

- `fs`
- `duration_sec`
- `age_years`
- `age_valid`
- `sex`
- `algorithm_age_group`
- `algorithm_age_reason`
- `age_defaulted_to_adult_algorithm`
- `clinical_dx_codes`
- `rx_codes`
- `lead_order`
- `available_extended_leads`
- `record_quality`
- `rejected_functions`

### `morphology_inputs.global`

- `heart_rate_bpm`
- `qrs_duration_ms`
- `qt_ms`
- `qtc_bazett_ms`
- `qtc_fridericia_ms`
- `p_axis_frontal_deg`
- `qrs_axis_frontal_deg`
- `qrs_axis_horizontal_direction`
- `t_axis_frontal_deg`
- `qrs_t_angle_deg`

### `morphology_inputs.leads`

Output per lead:

- `p.duration_ms`
- `p.amplitude_mV`
- `p.area`
- `p.is_notched`
- `p.is_biphasic`
- `p.initial_duration_ms`
- `p.initial_amplitude_mV`
- `p.terminal_duration_ms`
- `p.terminal_amplitude_mV`
- `p.terminal_area_ashman`
- `qrs.q_duration_ms`
- `qrs.q_amplitude_mV`
- `qrs.r_amplitude_mV`
- `qrs.r_prime_duration_ms`
- `qrs.r_prime_amplitude_mV`
- `qrs.s_duration_ms`
- `qrs.s_amplitude_mV`
- `qrs.s_prime_duration_ms`
- `qrs.s_prime_amplitude_mV`
- `qrs.peak_to_peak_mV`
- `qrs.positive_component_mV`
- `qrs.negative_component_mV`
- `qrs.area_sign`
- `qrs.terminal_direction`
- `st.j_point_mV`
- `st.midpoint_mV`
- `st.j80_mV`
- `st.end_mV`
- `st.slope_deg`
- `st.shape`
- `t.amplitude_mV`
- `t.polarity`
- `t.relative_to_qrs`
- `quality`

### `morphology_inputs.derived_facts`

- `adult`
- `pediatric`
- `dextrocardia_criteria`
- `rae_criteria`
- `lae_criteria`
- `bae_criteria`
- `axis_flags`
- `vcd_bbb_criteria`
- `rvh_criteria`
- `lvh_voltage_criteria`
- `lvh_additional_flags`
- `low_voltage_criteria`
- `copd_pattern_criteria`
- `mi_territory_criteria`
- `culprit_artery_criteria`
- `st_depression_criteria`
- `t_wave_criteria`
- `st_elevation_criteria`
- `tall_t_criteria`
- `qt_electrolyte_criteria`

### `morphology_inputs.statement_evidence`

- `candidates[]`
- `final_statements[]`
- `statement.code`
- `statement.category`
- `statement.severity`
- `statement.reason`
- `statement.evidence`
- `statement.flags_set`
- `statement.suppression_tags`
- `statement.suppressed_by`
- `statement.confidence_rank`
- `stop_further_interpretation`

<a id="优先级建议"></a>
## Priority suggestions

- `P0`: Completed. Add `morphology_inputs` schema, supplement adult / pediatric age route and threshold context, output the facts that can be stably derived at present, and do not add complex signal algorithms.
- `P1`: Supplement P notched/biphasic, P terminal component, Q/R'/S'/S duration, QRS terminal direction.
- `P2`: Extend existing statement/suppression engine to exposed rules not yet covered, and keep `unavailable` isolated from the reference algorithm.
- `P3`: Complete enhancement logic for ST Map, V4R/V7-V9, and MI culprit artery.

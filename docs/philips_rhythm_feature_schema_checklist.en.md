<!-- i18n-nav -->
[中文](philips_rhythm_feature_schema_checklist.md) | [English](philips_rhythm_feature_schema_checklist.en.md) | [日本語](philips_rhythm_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# Philips Rhythm Feature Schema Checklist

<a id="结论先行"></a>
## Conclusion first

The current full features JSON (e.g. `JS00024_features.json`) already contains many measurement features required for rhythm analysis:

- R/QRS position and RR interval
- beat-level P/QRS/T border
- PR/QRS/QT/JT/QTc
- P/QRS/T/ST axis
- P wave confidence, P amplitude, P area
- Q/R/S amplitude, QRS area, QRS notch/slur, delta flag
- per-lead quality and wave-specific reliability
- beat grouping, dominant group, group mean RR/PR/QRS/QT

But it cannot fully meet all the input requirements of Philips-style rhythm analysis in `要点.pdf`. The main gaps are concentrated in:

- QRST subtraction and atrial residual signal features of AF/AFL
- pacing's continuous/intermittent/chamber/artifact/capture/sense subdivision features
- Independent P-wave event stream for P waves > QRS complexes, dropped beat, second degree AV block
- Beat-level P morphology, QRS polarity signature, compensatory pause and other evidence for premature ejaculation classification
- delta lead count of preexcitation, short-PR threshold reduction, accessory pathway side
- statement evidence schema for rule engine

<a id="数据层级判断"></a>
## Data level judgment

| Data level | Whether it can support `要点.pdf` rhythm analysis | Description |
| --- | --- | --- |
| `.txt report` | Not enough | Suitable for human reading, only displays HR/RR/PR/QRS/QT/axis/quality/per-lead median, lacking complete beat-level features. |
| full features JSON, such as the regenerated `JS00024_features.json` | can support MVP, a large number of fields are available | included `beats`, `beat_features`, `representative_leads`, `groups`, `global_features`, `metadata`, `interpretation`, and the new `rhythm_inputs`. |
| `build_structured_payload()` is currently exported | Not enough | Currently only `signal / quality / beats / groups / global / provenance` is exported, `beat_features`, `representative_leads`, and `interpretation` are not exported. See `feature_extraction/ecgfeat/export.py:19-42`. |

<a id="状态定义"></a>
## State definition

- `OK`: Currently the full features JSON are provided and can be directly used for rule analysis.
- `Partial`: There are related fields, but they are not stable enough to support the complete rules in `要点.pdf`.
- `Missing`: No directly usable fields or derived features are currently found.

## Feature Checklist

| Module | `要点.pdf` Required Features | Currently Available Fields | Status | Gap/Risk |
| --- | --- | --- | --- | --- |
| Basic input | Sampling rate, lead sequence, record length | `fs`, `metadata.input_fs`, `metadata.internal_fs`, `metadata.lead_order` | OK | full JSON Yes; thin payload is also available. |
| Patient information | age, sex | `metadata.patient_meta.age`, `metadata.patient_meta.sex` | OK | Pediatric thresholds still require an external full age table. |
| Signal quality | per-lead quality, P/QRS/T/QT availability | `quality.*.reliable_for_p/qrs/t/qt`, `flags`, `baseline_wander_score`, `muscle_noise_score`, `powerline_score` | OK | There is no new version in `JS00024_features.json` `record_quality`; The current code is supported in `metadata.record_quality`. |
| R/QRS detection | R peak / QRS position | `beats[*].r_index`, `metadata.qrs_detector.r_locs` | OK | Supports RR, ventricular rate, pause, early beat. |
| RR characteristics | `rr_prev_ms`, `rr_next_ms`, mean RR, RR variability | `beats[*].rr_prev_ms`, `beats[*].rr_next_ms`, `groups[*].mean_rr_ms`, `interpretation.rr_cv` | OK | Background RR currently uses median/mean, and has not clearly output "clean background RR excluding" ectopy/pause/pacing”. |
| Ventricular rate | ventricular rate | `global_features.heart_rate_bpm`, `groups[*].mean_ventr_rate_bpm` | OK | Can support tachy/brady/complete AVB rate conditions. |
| Atrial rate | atrial rate | `global_features.atrial_rate_bpm` | Partial | is a single value estimated from the P test; lacks independent atrial event series and confidence. |
| P-wave detection | P onset/peak/offset per beat/lead | `beat_features[*].p.onset`, `p.peak`, `p.offset` | OK | QRS-centered P detection is available, but not sufficient to detect dropped P or independent P event streams. |
| Existence of P wave | has P wave / P confidence | `beat_features[*].p_confidence`, `representative_leads.*.params.p_confidence_mean` | OK | The current rule can use the threshold to determine "whether there is P". |
| P wave morphology | normal / atypical / retrograde / ectopic P | `interpretation.p_morphology_class`, P amplitude/area/duration | Partial | The current `p_morphology_class` is partial atrial enlargement and is not the beat-level atypical P morphology required by APC/JPC. |
| P axis | frontal P axis | `global_features.p_axis_deg`, `interpretation.p_axis_normal` | OK | Can support sinus origin judgment. |
| PR interval | PR per beat/lead/global | `beat_features[*].pr_ms`, `representative_leads.*.params.pr_ms`, `global_features.pr_ms` | OK | Supports initial assessment of AVB1, short PR, and Wenckebach. |
| PR segment | PR segment duration | No direct field found | Missing | `PR segment < 55 ms` in WPW rules cannot be directly determined; only `baseline_source="pr_segment"` is not a duration. |
| PR trend | PR sequence before pause | Derivable from `beat_features[*].pr_ms` + beat order | Partial | Currently derivable, but no explicit output of per-beat global PR sequence/provenance. |
| QRS duration | QRSd per beat/lead/global | `beat_features[*].qrs_ms`, `representative_leads.*.params.qrs_ms`, `global_features.qrs_ms` | OK | Supports normal/wide QRS and coarse classification of VPC/JPC/APC. |
| QRS morphology | Q/R/S amplitude, R prime, S prime, notch/slur, VAT | `q_amp_mv`, `r_amp_mv`, `s_amp_mv`, `r_prime_amp_mv`, `s_prime_amp_mv`, `qrs_notch_count`, `qrs_slur_flag`, `vat_ms` | OK | Substantial morphological information available, sufficient for further polarity/signature analysis. |
| QRS polarity signature | per-lead polarity vector / morphology vector | Derivable from Q/R/S amplitudes and signed area | Partial | No direct output of `qrs_polarity_signature` or beat-template embedding. |
| QRS signed area | signed QRS area | Internal model has `qrs_signed_area`; current `JS00024_features.json` first beat_features key list does not show this field | Partial | The code model has the field, but the current JSON may have been generated from an old export; need to confirm if the new export includes it. |
| Delta wave | per-beat/lead delta flag | `beat_features[*].delta_present` | OK | Supports initial WPW screening. |
| Delta lead count | number of leads with delta wave | Aggregatable from `beat_features[*].delta_present` | Partial | No direct output of lead count, confidence, or configuration evidence required for short-PR threshold reduction. |
| Initial QRS axis | initial QRS axis for pathway side | No direct field found | Missing | Only global QRS axis is available, which is insufficient to determine left/right accessory pathway. |
| QT/QTc | QT, QTcB, QTcF | `global_features.qt_ms`, `qtc_bazett_ms`, `qtc_fridericia_ms`, per-lead QT/QTc derivable | OK | Rhythm main line is not a core gap. |
| Beat grouping | group id, dominant group, group metrics | `beats[*].group_id`, `groups`, `metadata.representative_group_id` | OK | Supports coarse analysis of dominant rhythm / ectopic group. |
| Paced flag | beat-level paced | `beats[*].paced`, `metadata.paced_beat_ids` | Partial | Reliable only when `enable_pacing=True`; may be disabled in regular calls. |
| Pacing spikes | spike timestamps | `global_features.pacing_spikes`, `metadata.pacing_state` | Partial | `JS00024_features.json` is `null/off`; pacing is usually not enabled in the default pipeline. |
| Pacing type | atrial / ventricular / dual / AV sequential | No direct field found | Missing | Cannot meet the fine-grained control requirements of `PACE-001/002/006/007`. |
| Continuous/intermittent pacing | continuous_pacing, intermittent_pacing | No direct field found | Missing | Currently only beat tagging is available, which is insufficient to stop non-paced rhythm-pattern analysis. |
| Demand behavior | pulse inhibition evidence | No direct field found | Missing | No features for demand pacing behavior. |
| Pacemaker artifact | artifact confidence / noisy spike evidence | No independent field found | Missing | Current `pacemaker_like_artifact` is not independent measurement evidence. |
| Failure to sense/capture | spike-QRS/P decoupling, fixed-rate async spikes | No direct field found | Missing | No input for magnet/failure-to-sense/capture analysis. |
| AF RR features | RR CV, irregularity, RMSSD/Lorenz/entropy | `interpretation.rr_cv`, `rr_irregularity_class`; RR sequence derivable | Partial | CV is available; RMSSD, entropy, and Lorenz-type features are not output. |
| AF P-wave evidence | absent organized P waves | `p_confidence`, `probable_af` | Partial | Coarse judgment possible, but no atrial residual signal support. |
| QRST subtraction | atrial residual signal | No direct field found | Missing | This is the core gap for stable differentiation of AF/AFL. |
| Atrial waveform stability | residual stability metric | No direct field found | Missing | Cannot distinguish AF/AFL according to `要点.pdf`. |
| Atrial repetitiveness | residual repetitive/autocorrelation/frequency feature | Direct field not found | Missing | AFL detects basic missing input. |
| Flutter evidence | flutter wave confidence / atrial cycle length | No direct field found | Missing | There is currently no explicit atrial flutter feature. |
| APC/JPC/VPC early beat evidence | RR shortening vs background RR | `beats[*].rr_prev_ms`; background RR can be derived from groups | OK | Sufficient for 15% early beat screening. |
| APC evidence | early + normal QRS + atypical P morphology | RR + QRS + P confidence available | Partial | missing beat-level atypical P morphology. |
| JPC evidence | early + normal QRS + no P | RR + QRS + P confidence available | OK | Sufficient for initial screening of "no P" type JPC. |
| VPC evidence | early + wide QRS + polarity different + compensatory pause | RR + QRS available; QRS morphology can be derived | Partial | Lack of clear polarity difference, compensatory pause, normal template comparison. |
| VPC pair/NSVT | consecutive V beat run | Can be derived from beat classes; the current interpretation is `non_sustained_vt` | Partial | The current beat class is not exported, only the final flag is exported; the evidence chain is incomplete. |
| Bigeminy/trigeminy | N/V or N/A sequence pattern | The current interpretation has `bigeminy`, `trigeminy` fields; the old version of `JS00024_features.json` lacks these fields | Partial | The new model has fields, but currently JSON only contains 37 interpretation fields and does not include the advanced rhythm field. |
| Pause | RR > 140% background RR | `beats[*].rr_next_ms`, group mean RR; the new interpretation has pause field | Partial | The current `JS00024_features.json` and the old version do not export `pauses_detected/pause_longest_ms`. |
| Escape beat origin | P presence + QRSd after pause | P confidence + QRSd can be used for preliminary screening | Partial | Lack of special post-pause escape beat object and origin label. |
| Second-degree AVB | P waves > QRS complexes | No independent P events found | Missing | QRS-centered P detection is insufficient to reliably count blocked P. |
| Mobitz I | progressive PR lengthening before dropped beat | PR sequence can be derived; new interpretation has `second_degree_avb` | Partial | missing dropped beat/P-only event evidence. |
| AV dissociation | ventricular rate normal + apparent PR variation / AV asynchrony | PR sequence, atrial_rate, heart_rate available | Partial | missing explicit AV asynchrony score and independent atrial/ventricular event streams. |
| Complete AVB | ventricular rate <45 + AV asynchrony | HR available | Partial | Missing AV asynchrony score. |
| Interpolated beats | prev/next RR around 0.5 background RR | `rr_prev_ms`, `rr_next_ms` Available | Partial | Derivable, but no direct fields/flag. |
| Aberrant complexes | slight RR shortening + wide QRS | RR + QRSd available | Partial | Missing evidence of "slight shortening" specific threshold and morphology differentiation. |
| Evidence output | evidence for each rhythm statement | `rhythm_inputs.statement_evidence` and `clinical_interpretation.domains` | Partial | Covered the current set of exposed rules; the full Philips rhythm statement inventory is still not implemented. |
| Rule priority flags | stop/bypass/suppression | `statement_engine` and `clinical_interpretation.suppressed_statements` | Partial | Deterministic stop/suppression is available; the complete rhythm engine priority is still a follow-up work. |

<a id="js00024-当前-json-快照"></a>
## JS00024 Current JSON snapshot

The old version `JS00024_features.json` is currently confirmed to contain:

- top-level: `fs`, `quality`, `beats`, `beat_features`, `representative_leads`, `groups`, `global_features`, `metadata`, `interpretation`
- beats: 19, the fields are `beat_id`, `r_index`, `paced`, `group_id`, `rr_prev_ms`, `rr_next_ms`
- beat_features: 228 items, fields covering P/QRS/T bounds, PR/QRS/QT, P/QRS/T amplitude/area, delta, confidence, notch/slur, VAT, TPE, ST morphology, etc.
- representative_leads: 12 leads, each with `params` and `variance`
- global_features: HR, atrial rate, PR/QRS/QT/QTc, axis, QT dispersion, pacing, PTF-V1
- metadata: input/internal fs, lead order, lead reversal, n_beats, qrs_detector, representative_beat_meta, paced_beat_ids, patient_meta
- interpretation: The old version has 37 fields and lacks the advanced rhythm fields of the new version such as `premature_complexes`, `bigeminy`, `trigeminy`, `pauses_detected`, `complete_av_block`, `second_degree_avb`

After regenerating full features JSON, top-level `rhythm_inputs` will be additionally included for downstream rhythm rule engine.

<a id="已新增的最小目标-schema"></a>
## Added minimum target Schema

full features JSON of `rhythm_inputs` have been output in the following blocks. Fields that can currently be stably derived from existing measurements will be given specific values; fields that require new signal algorithms will be explicitly given `available=false` and `reason`.

### `rhythm_inputs.record`

- `fs`
- `duration_sec`
- `age_years`
- `sex`
- `lead_order`
- `record_quality`
- `rejected_functions`

### `rhythm_inputs.beats[]`

- `beat_id`
- `r_index`
- `r_time_ms`
- `rr_prev_ms`
- `rr_next_ms`
- `group_id`
- `paced`
- `pacer_spike_index`
- `qrs_duration_ms`
- `pr_interval_ms`
- `p_confidence`
- `p_axis_deg`
- `p_morphology`
- `qrs_polarity_signature`
- `delta_leads`
- `beat_quality`

### `rhythm_inputs.p_events[]`

- `p_event_id`
- `time_ms`
- `confidence`
- `associated_qrs_beat_id`
- `association_type`: `conducted`, `blocked`, `retrograde`, `unknown`
- `pr_ms`
- `axis_deg`
- `morphology`

### `rhythm_inputs.background`

- `clean_rr_ms`
- `background_rr_regular`
- `background_ventricular_rate_bpm`
- `background_atrial_rate_bpm`
- `dominant_group_id`
- `excluded_beat_ids`
- `exclusion_reasons`

### `rhythm_inputs.pacing`

- `enabled`
- `spike_times`
- `spike_count`
- `paced_beat_ids`
- `continuous_pacing`
- `intermittent_pacing`
- `ventricular_pacing_present`
- `atrial_pacing_present`
- `dual_chamber_pacing_present`
- `artifact_confidence`
- `capture_failure_suspected`
- `sensing_failure_suspected`

### `rhythm_inputs.af_afl`

- `rr_cv`
- `rr_rmssd`
- `rr_entropy`
- `qrst_subtraction_quality`
- `atrial_residual_signal_summary`
- `atrial_signal_stability`
- `atrial_signal_repetitiveness`
- `dominant_atrial_cycle_ms`
- `flutter_wave_confidence`

### `rhythm_inputs.preexcitation`

- `short_pr_interval`
- `short_pr_segment`
- `delta_lead_count`
- `delta_leads`
- `delta_confidence_by_lead`
- `mean_qrs_duration_ms`
- `initial_qrs_axis_deg`
- `accessory_pathway_side`

### `rhythm_inputs.statement_evidence`

- `primary_statement`
- `additional_statements`
- `statement.code`
- `statement.category`
- `statement.priority`
- `statement.evidence`
- `statement.confidence`
- `statement.suppresses`
- `stop_further_interpretation`
- `bypass_remaining_algorithm`

<a id="实用判断"></a>
## Practical judgment

The current full features JSON is suitable for being an **MVP rhythm rule engine**:

- sinus/tachy/brady
- regular/irregular
- first-degree AV block / PR normal-short-long
- wide/narrow QRS
- simpleWPW
- simple APC/JPC/VPC
-bigeminy/trigeminy
- simple pause

The Glasgow interpretation layer is no longer executed or exported during the current runtime. Auditable output is given by
`statement_engine`, `clinical_interpretation.domains` and each rule
Evidence/threshold/source fields are provided.

It is currently not suitable for these parts of the complete Philips-style rhythm engine:

- robust AF vs AFL
- paced rhythm statement control flow
- second-degree AV block with blocked P waves
-escape-beat origin
- ventricular pacing with AF-only atrial diagnosis
- magnet / capture / sensing failure
-accessory pathway localization

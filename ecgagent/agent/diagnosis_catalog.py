"""Stable diagnostic vocabulary for diagnosis-first ECGAgent output.

The vocabulary is independent of a record's rule evaluations.  That matters:
the language model is allowed to reach a diagnosis even when no ecgfeat rule
matched it.  Record-local statement codes are accepted as an extension for
backward compatibility with the feature artifact, but rule status never gates
whether a diagnosis may be made.
"""
from __future__ import annotations

from typing import Any, Mapping


DIAGNOSIS_CATALOG: dict[str, tuple[str, str]] = {
    # Normal / quality
    "normal_ecg": ("Normal ECG", "other"),
    "non_diagnostic_ecg": ("Non-diagnostic ECG", "quality"),
    "unclassified_ecg_abnormality": ("Unclassified ECG abnormality", "other"),
    "technically_limited": ("Technically limited ECG", "quality"),
    "diagnostic_coverage_limited": ("Diagnostic coverage limited", "quality"),
    "complex_morphology_limited_interpretation": (
        "Complex morphology with limited interpretation",
        "quality",
    ),
    "minor_signal_quality_issue": ("Minor signal quality issue", "quality"),
    "repeat_ecg_required": ("Repeat ECG required", "quality"),
    # A named code, rather than a swap described inside `technically_limited`
    # prose, is what lets the criteria check in `semantic_guard` find the claim.
    "limb_lead_reversal_suspected": ("Suspected limb lead reversal", "quality"),
    "uncalibrated_amplitudes": (
        "Amplitudes not calibrated; voltage criteria not assessable",
        "quality",
    ),
    # Rhythm and rate
    "sinus_rhythm": ("Sinus rhythm", "rhythm"),
    "sinus_mechanism": ("Sinus mechanism", "rhythm"),
    "sinus_bradycardia": ("Sinus bradycardia", "rate"),
    "sinus_tachycardia": ("Sinus tachycardia", "rate"),
    "bradycardia": ("Bradycardia", "rate"),
    "tachycardia": ("Tachycardia", "rate"),
    "rate_abnormality": ("Rate abnormality", "rate"),
    "atrial_fibrillation": ("Atrial fibrillation", "rhythm"),
    "atrial_fibrillation_pattern": ("Atrial fibrillation pattern", "rhythm"),
    "probable_atrial_fibrillation_pattern": (
        "Probable atrial fibrillation pattern",
        "rhythm",
    ),
    "atrial_flutter_pattern": ("Atrial flutter pattern", "rhythm"),
    "possible_atrial_flutter_pattern": (
        "Possible atrial flutter pattern",
        "rhythm",
    ),
    "atrial_fibrillation_flutter_indeterminate": (
        "Atrial fibrillation/flutter indeterminate",
        "rhythm",
    ),
    "exclude_2_to_1_atrial_flutter": (
        "Two-to-one atrial flutter not demonstrated",
        "rhythm",
    ),
    "atrial_tachycardia": ("Atrial tachycardia", "rhythm"),
    "junctional_rhythm_pattern": ("Junctional rhythm pattern", "rhythm"),
    "ectopic_atrial_rhythm_pattern": ("Ectopic atrial rhythm pattern", "rhythm"),
    "p_wave_abnormality": ("P-wave morphology abnormality", "rhythm"),
    "multifocal_atrial_rhythm": ("Multifocal atrial rhythm", "rhythm"),
    "multifocal_atrial_tachycardia": ("Multifocal atrial tachycardia", "rhythm"),
    "ventricular_rhythm": ("Ventricular rhythm", "rhythm"),
    "wide_complex_tachycardia": ("Wide-complex tachycardia", "rhythm"),
    # Ectopy
    "premature_atrial_complexes": ("Premature atrial complexes", "ectopy"),
    "premature_ventricular_complexes": (
        "Premature ventricular complexes",
        "ectopy",
    ),
    "probable_premature_ventricular_complexes": (
        "Probable premature ventricular complexes",
        "ectopy",
    ),
    # AV / intraventricular conduction
    "first_degree_av_block": ("First-degree AV block", "conduction"),
    "first_degree_av_delay": ("First-degree AV delay", "conduction"),
    "possible_first_degree_av_delay": (
        "Possible first-degree AV delay",
        "conduction",
    ),
    "second_degree_av_block": ("Second-degree AV block", "conduction"),
    "second_degree_av_block_pattern": (
        "Second-degree AV block pattern",
        "conduction",
    ),
    "complete_av_block": ("Complete AV block", "conduction"),
    "complete_av_block_pattern": ("Complete AV block pattern", "conduction"),
    "right_bundle_branch_block": ("Right bundle branch block", "conduction"),
    "rbbb_pattern": ("Right bundle branch block pattern", "conduction"),
    "incomplete_rbbb_pattern": ("Incomplete RBBB pattern", "conduction"),
    "left_bundle_branch_block": ("Left bundle branch block", "conduction"),
    "lbbb_pattern": ("Left bundle branch block pattern", "conduction"),
    "nonspecific_ivcd": (
        "Nonspecific intraventricular conduction delay",
        "conduction",
    ),
    "bifascicular_block_pattern": ("Bifascicular block pattern", "conduction"),
    "probable_bifascicular_block_pattern": (
        "Probable bifascicular block pattern",
        "conduction",
    ),
    "trifascicular_block_pattern": ("Trifascicular block pattern", "conduction"),
    "lafb_pattern": ("Left anterior fascicular block pattern", "conduction"),
    "probable_lafb_pattern": (
        "Probable left anterior fascicular block pattern",
        "conduction",
    ),
    "lpfb_pattern": ("Left posterior fascicular block pattern", "conduction"),
    "probable_lbbb_pattern": (
        "Probable left bundle branch block pattern",
        "conduction",
    ),
    "probable_rbbb_pattern": (
        "Probable right bundle branch block pattern",
        "conduction",
    ),
    "probable_nonspecific_ivcd": (
        "Probable nonspecific intraventricular conduction delay",
        "conduction",
    ),
    "wide_qrs_repolarization_review": (
        "Wide-QRS repolarization requires review",
        "conduction",
    ),
    "ventricular_preexcitation_pattern": (
        "Ventricular pre-excitation pattern",
        "conduction",
    ),
    # Intervals / axis
    "short_pr_interval": ("Short PR interval", "interval"),
    "prolonged_qt": ("Prolonged QT", "interval"),
    "markedly_prolonged_qt": ("Markedly prolonged QT", "interval"),
    "short_qt": ("Short QT", "interval"),
    "markedly_short_qt": ("Markedly short QT", "interval"),
    "borderline_short_qt": ("Borderline short QT", "interval"),
    "possible_short_qt_pattern": ("Possible short QT pattern", "interval"),
    "left_axis_deviation": ("Left axis deviation", "axis"),
    "right_axis_deviation": ("Right axis deviation", "axis"),
    "extreme_axis_deviation": ("Extreme axis deviation", "axis"),
    # Voltage / chamber patterns
    "left_ventricular_hypertrophy": (
        "Left ventricular hypertrophy pattern",
        "chamber",
    ),
    "lvh_voltage_criteria": ("LVH voltage criteria", "chamber"),
    "right_ventricular_hypertrophy": (
        "Right ventricular hypertrophy pattern",
        "chamber",
    ),
    "rvh_pattern": ("RVH pattern", "chamber"),
    "left_atrial_abnormality": ("Left atrial abnormality", "chamber"),
    "right_atrial_abnormality": ("Right atrial abnormality", "chamber"),
    "biatrial_abnormality": ("Biatrial abnormality", "chamber"),
    "low_voltage_limb_leads": ("Low voltage in limb leads", "chamber"),
    "low_voltage_precordial_leads": (
        "Low voltage in precordial leads",
        "chamber",
    ),
    "low_qrs_voltage_limb_leads": ("Low QRS voltage in limb leads", "chamber"),
    "low_qrs_voltage_precordial_leads": (
        "Low QRS voltage in precordial leads",
        "chamber",
    ),
    "possible_lvh_voltage": ("Possible LVH voltage pattern", "chamber"),
    "pediatric_left_atrial_abnormality": (
        "Pediatric left atrial abnormality",
        "chamber",
    ),
    "pediatric_right_atrial_abnormality": (
        "Pediatric right atrial abnormality",
        "chamber",
    ),
    "pediatric_lvh_voltage": ("Pediatric LVH voltage pattern", "chamber"),
    "pediatric_rvh_voltage": ("Pediatric RVH voltage pattern", "chamber"),
    # QRS progression and repolarization
    "poor_r_wave_progression": ("Poor R-wave progression", "other"),
    "reversed_r_wave_progression": ("Reversed R-wave progression", "other"),
    "precordial_rotation": ("Precordial rotation", "other"),
    "clockwise_rotation": ("Clockwise precordial rotation", "other"),
    "counterclockwise_rotation": ("Counterclockwise precordial rotation", "other"),
    "dextrocardia_pattern": ("Dextrocardia pattern", "other"),
    "possible_precordial_lead_reversal": (
        "Possible precordial lead reversal",
        "quality",
    ),
    "electrical_alternans": ("Electrical alternans", "other"),
    "st_depression": ("ST depression", "ischemia_repolarization"),
    "st_elevation": ("ST elevation", "ischemia_repolarization"),
    "t_wave_abnormality": ("T-wave abnormality", "ischemia_repolarization"),
    "flat_t_wave_pattern": ("Flat or low-amplitude T-wave pattern", "ischemia_repolarization"),
    "primary_t_wave_abnormality": (
        "Primary T-wave abnormality",
        "ischemia_repolarization",
    ),
    "secondary_t_wave_abnormality": (
        "Secondary T-wave abnormality",
        "ischemia_repolarization",
    ),
    "hyperacute_t_wave_pattern": (
        "Hyperacute T-wave pattern",
        "ischemia_repolarization",
    ),
    "giant_negative_t_wave": (
        "Giant negative T-wave pattern",
        "ischemia_repolarization",
    ),
    "pathological_q_waves": (
        "Pathological Q waves",
        "ischemia_repolarization",
    ),
    "prior_infarct_q_wave_pattern": (
        "Prior infarct Q-wave pattern",
        "ischemia_repolarization",
    ),
    "prominent_u_wave": ("Prominent U wave", "ischemia_repolarization"),
    "inverted_u_wave": ("Inverted U wave", "ischemia_repolarization"),
    "acute_occlusion_pattern": (
        "Acute coronary occlusion pattern",
        "ischemia_repolarization",
    ),
    "left_main_pattern": (
        "Left main or multivessel ischemic pattern",
        "ischemia_repolarization",
    ),
    "posterior_ischemia_screen": (
        "Posterior ischemia screening pattern",
        "ischemia_repolarization",
    ),
    "de_winter_pattern": ("De Winter pattern", "ischemia_repolarization"),
    "wellens_pattern": ("Wellens pattern", "ischemia_repolarization"),
    "sgarbossa_positive": (
        "Positive Sgarbossa pattern",
        "ischemia_repolarization",
    ),
    "acute_pericarditis_pattern": (
        "Acute pericarditis pattern",
        "ischemia_repolarization",
    ),
    "early_repolarization_pattern": (
        "Early repolarization pattern",
        "ischemia_repolarization",
    ),
    "digitalis_effect_pattern": (
        "Digitalis effect pattern",
        "ischemia_repolarization",
    ),
    "hyperkalemia_pattern": ("Hyperkalemia pattern", "ischemia_repolarization"),
    "hypokalemia_pattern": ("Hypokalemia pattern", "ischemia_repolarization"),
    "brugada_type1_screening": (
        "Brugada type 1 screening pattern",
        "ischemia_repolarization",
    ),
    "pulmonary_embolism_pattern": (
        "Pulmonary embolism-associated ECG pattern",
        "ischemia_repolarization",
    ),
    # Pacing
    "paced_rhythm": ("Paced rhythm", "pacing"),
    "ventricular_paced_rhythm": ("Ventricular paced rhythm", "pacing"),
    "intermittent_pacing": ("Intermittent pacing", "pacing"),
    "pacing_failure_to_capture_suspected": (
        "Suspected pacing failure to capture",
        "pacing",
    ),
    "pacing_sensing_failure_suspected": (
        "Suspected pacing sensing failure",
        "pacing",
    ),
}

DIAGNOSTIC_CATEGORIES = frozenset(
    {
        "rhythm",
        "rate",
        "interval",
        "axis",
        "conduction",
        "ectopy",
        "chamber",
        "ischemia_repolarization",
        "pacing",
        "quality",
        "other",
    }
)


def record_statement_codes(document: Mapping[str, Any] | None) -> set[str]:
    """Return vocabulary extensions, not conclusions, from an ecgfeat artifact."""
    if not isinstance(document, Mapping):
        return set()
    clinical = document.get("clinical_interpretation")
    if not isinstance(clinical, Mapping):
        return set()

    rows: list[Mapping[str, Any]] = []
    domains = clinical.get("domains")
    if isinstance(domains, Mapping):
        for domain_rows in domains.values():
            rows.extend(row for row in (domain_rows or []) if isinstance(row, Mapping))
    for key in ("final_statements", "borderline_statements", "suppressed_statements"):
        rows.extend(
            row for row in (clinical.get(key) or []) if isinstance(row, Mapping)
        )

    codes: set[str] = set()
    for row in rows:
        if row.get("statement_code"):
            codes.add(str(row["statement_code"]))
        evidence = row.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("evaluates_code"):
            codes.add(str(evidence["evaluates_code"]))
    return codes


# Codes whose criteria are thresholds in millivolts. They mean nothing on a
# record whose leads were each rescaled, and the rescaling is invisible in the
# measurements themselves - every amplitude still looks like a plausible one.
#
# Deliberately not the whole `chamber` category: atrial abnormality rests on P
# duration and morphology as well as on amplitude, and duration survives
# rescaling intact. Blocking it too would trade one silent error for another.
_AMPLITUDE_DEPENDENT_CODES = frozenset(
    {
        "left_ventricular_hypertrophy",
        "lvh_voltage_criteria",
        "possible_lvh_voltage",
        "right_ventricular_hypertrophy",
        "rvh_pattern",
        "low_voltage_limb_leads",
        "low_voltage_precordial_leads",
        "low_qrs_voltage_limb_leads",
        "low_qrs_voltage_precordial_leads",
        "pediatric_lvh_voltage",
        "pediatric_rvh_voltage",
    }
)


def _amplitudes_are_calibrated(document: Mapping[str, Any] | None) -> bool:
    if not isinstance(document, Mapping):
        return True
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        return True
    contract = metadata.get("input_contract")
    calibration = (
        contract.get("amplitude_calibration")
        if isinstance(contract, Mapping)
        else None
    )
    if not isinstance(calibration, Mapping):
        return True
    return not bool(calibration.get("per_lead_normalized"))


def allowed_diagnosis_codes(document: Mapping[str, Any] | None = None) -> set[str]:
    codes = set(DIAGNOSIS_CATALOG) | record_statement_codes(document)
    if not _amplitudes_are_calibrated(document):
        codes -= _AMPLITUDE_DEPENDENT_CODES
        codes.add("uncalibrated_amplitudes")
    return codes

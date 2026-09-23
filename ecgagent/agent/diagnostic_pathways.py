"""Bounded, candidate-specific reasoning paths for the compact ECG Agent.

The path is a workflow contract, not patient evidence.  Each node names both
the measurement tool and the exact narrow view that must be prefetched.  The
orchestrator owns view execution and final placement; the model only labels
the returned node evidence pass/fail/unknown.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .diagnosis_catalog import DIAGNOSIS_CATALOG
from .deterministic_pathways import program_owns_pathway_step


DIAGNOSTIC_PATHWAY_VERSION = "ecgagent.diagnostic-pathways.v9"


#: Gate semantics for a pathway node.
#:
#: ``required``     every node must ``pass`` before a candidate is confirmed;
#:                  any ``fail`` rejects it and any ``unknown`` leaves it unresolved.
#: ``supporting``   the status is recorded and its evidence is kept, but it never
#:                  gates placement.
#: ``invalidator``  by default only explicit ``fail`` blocks confirmation.
#:                  ``unknown_blocks_confirmation`` makes unproved eligibility
#:                  unresolved; ``failure_means_not_applicable`` distinguishes
#:                  an inapplicable criterion from a refuted disease.
#:                  Necessary definition evidence uses ``required``.
#:                  ``supporting`` cannot express a veto: its status
#:                  never reaches placement, so a veto written as ``supporting``
#:                  is structurally unable to veto.
PATHWAY_GATES: frozenset[str] = frozenset({"required", "supporting", "invalidator"})


def _step(
    step_id: str,
    question: str,
    tool: str,
    *,
    gate: str = "required",
    arguments: Mapping[str, Any] | None = None,
    unknown_blocks_confirmation: bool = False,
    failure_means_not_applicable: bool = False,
) -> dict[str, Any]:
    if gate not in PATHWAY_GATES:
        raise ValueError(
            f"unknown pathway gate {gate!r} for step {step_id!r}; "
            f"expected one of {sorted(PATHWAY_GATES)}"
        )
    row: dict[str, Any] = {
        "id": step_id,
        "gate": gate,
        "question": question,
        "tool": tool,
    }
    if arguments is not None:
        row["arguments"] = copy.deepcopy(dict(arguments))
    if unknown_blocks_confirmation:
        row["unknown_blocks_confirmation"] = True
    if failure_means_not_applicable:
        row["failure_means_not_applicable"] = True
    return row


_SINUS = {"sinus_rhythm", "sinus_mechanism"}
_RATE = {
    "bradycardia",
    "rate_abnormality",
    "sinus_bradycardia",
    "sinus_tachycardia",
    "tachycardia",
}
_AXIS = {"left_axis_deviation", "right_axis_deviation", "extreme_axis_deviation"}
_FIRST_DEGREE = {
    "first_degree_av_block",
    "first_degree_av_delay",
    "possible_first_degree_av_delay",
}
_ADVANCED_AV = {
    "second_degree_av_block",
    "second_degree_av_block_pattern",
    "complete_av_block",
    "complete_av_block_pattern",
}
_IVCD = {"nonspecific_ivcd", "probable_nonspecific_ivcd"}
_BBB = {
    "right_bundle_branch_block",
    "rbbb_pattern",
    "incomplete_rbbb_pattern",
    "probable_rbbb_pattern",
    "left_bundle_branch_block",
    "lbbb_pattern",
    "probable_lbbb_pattern",
    "lafb_pattern",
    "probable_lafb_pattern",
    "lpfb_pattern",
    "bifascicular_block_pattern",
    "probable_bifascicular_block_pattern",
    "trifascicular_block_pattern",
}
_PREEXCITATION = {"ventricular_preexcitation_pattern"}
_LVH = {
    "lvh_voltage_criteria",
    "left_ventricular_hypertrophy",
    "possible_lvh_voltage",
    "pediatric_lvh_voltage",
}
_RVH = {"right_ventricular_hypertrophy", "rvh_pattern", "pediatric_rvh_voltage"}
_ATRIAL_CHAMBER = {
    "left_atrial_abnormality",
    "right_atrial_abnormality",
    "biatrial_abnormality",
    "pediatric_left_atrial_abnormality",
    "pediatric_right_atrial_abnormality",
}
_LOW_VOLTAGE = {
    "low_qrs_voltage_limb_leads",
    "low_qrs_voltage_precordial_leads",
    "low_voltage_limb_leads",
    "low_voltage_precordial_leads",
}
_AF = {
    "atrial_fibrillation",
    "atrial_fibrillation_pattern",
    "probable_atrial_fibrillation_pattern",
    "atrial_fibrillation_flutter_indeterminate",
}
_AFL = {"atrial_flutter_pattern", "possible_atrial_flutter_pattern"}
_OTHER_ATRIAL_RHYTHM = {
    "atrial_tachycardia",
    "ectopic_atrial_rhythm_pattern",
    "multifocal_atrial_rhythm",
    "multifocal_atrial_tachycardia",
    "p_wave_abnormality",
    "junctional_rhythm_pattern",
}
_VENTRICULAR_RHYTHM = {"ventricular_rhythm", "wide_complex_tachycardia"}
_ECTOPY = {
    "premature_atrial_complexes",
    "premature_ventricular_complexes",
    "probable_premature_ventricular_complexes",
}
_QT = {
    "prolonged_qt",
    "markedly_prolonged_qt",
    "short_qt",
    "markedly_short_qt",
    "borderline_short_qt",
    "possible_short_qt_pattern",
}
_R_PROGRESSION = {
    "clockwise_rotation",
    "counterclockwise_rotation",
    "precordial_rotation",
    "poor_r_wave_progression",
    "reversed_r_wave_progression",
    "dextrocardia_pattern",
}
_ST_PATTERNS = {
    "acute_occlusion_pattern",
    "acute_pericarditis_pattern",
    "de_winter_pattern",
    "early_repolarization_pattern",
    "left_main_pattern",
    "posterior_ischemia_screen",
    "sgarbossa_positive",
    "st_depression",
    "st_elevation",
    "wellens_pattern",
}
_T_PATTERNS = {
    "digitalis_effect_pattern",
    "flat_t_wave_pattern",
    "giant_negative_t_wave",
    "hyperacute_t_wave_pattern",
    "primary_t_wave_abnormality",
    "secondary_t_wave_abnormality",
    "t_wave_abnormality",
    "wide_qrs_repolarization_review",
}
_Q_PATTERNS = {"pathological_q_waves", "prior_infarct_q_wave_pattern"}
_TU_METABOLIC = {
    "hyperkalemia_pattern",
    "hypokalemia_pattern",
    "inverted_u_wave",
    "prominent_u_wave",
    "pulmonary_embolism_pattern",
    "brugada_type1_screening",
}


def semantic_candidate_family(code: str) -> str:
    """Return the one-slot semantic family used by compact plan deduplication."""

    token = str(code or "")
    # A documented unresolved AF/AFL distinction is not a confirmed AF mechanism.
    if token == "atrial_fibrillation_flutter_indeterminate":
        return token
    groups: tuple[tuple[str, set[str]], ...] = (
        ("sinus_mechanism", _SINUS),
        ("bradycardia", {"bradycardia", "sinus_bradycardia"}),
        ("tachycardia", {"tachycardia", "sinus_tachycardia"}),
        ("first_degree_av_delay", _FIRST_DEGREE),
        ("second_degree_av_block", {"second_degree_av_block", "second_degree_av_block_pattern"}),
        ("complete_av_block", {"complete_av_block", "complete_av_block_pattern"}),
        ("nonspecific_ivcd", _IVCD),
        ("rbbb", {"right_bundle_branch_block", "rbbb_pattern", "probable_rbbb_pattern"}),
        ("lbbb", {"left_bundle_branch_block", "lbbb_pattern", "probable_lbbb_pattern"}),
        ("lafb", {"lafb_pattern", "probable_lafb_pattern"}),
        ("bifascicular", {"bifascicular_block_pattern", "probable_bifascicular_block_pattern"}),
        ("lvh", _LVH),
        ("rvh", _RVH),
        ("atrial_fibrillation", _AF),
        ("atrial_flutter", _AFL),
        ("pvc", {"premature_ventricular_complexes", "probable_premature_ventricular_complexes"}),
        ("long_qt", {"prolonged_qt", "markedly_prolonged_qt"}),
        ("short_qt", {"short_qt", "markedly_short_qt", "borderline_short_qt", "possible_short_qt_pattern"}),
        ("low_voltage_limb", {"low_qrs_voltage_limb_leads", "low_voltage_limb_leads"}),
        ("low_voltage_precordial", {"low_qrs_voltage_precordial_leads", "low_voltage_precordial_leads"}),
    )
    for family, members in groups:
        if token in members:
            return family
    return token


# Interpretation nodes deliberately use new IDs: do not inherit a deterministic
# resolver merely because an older, less specific question used the same tool.
# Every argument below belongs to an existing tool; absent feature atoms remain
# unavailable rather than being fabricated by the pathway.
_P_FIELDS = ["accepted", "ta_ambiguous", "onset_confidence", "offset_confidence",
             "valid_leads", "reject_reasons", "morphology_cluster_id"]
_ATRIAL_FIELDS = ["p_event_id", "association_type", "confidence", "pr_ms", "source_leads",
                  "associated_qrs_beat_id", "time_ms"]
_QRS_LEADS = ["I", "II", "III", "aVL", "aVF", "V1", "V2", "V5", "V6"]


def _p_identity_step() -> dict[str, Any]:
    return _step(
        "atrial_wave_identity", "Do repeated P-wave boundary, lead-quality and T/A ambiguity assessments "
        "support genuine P waves independently of the PR value and event-association labels? "
        "Missing assessments or unresolved P/T overlap are unknown; another view of the same "
        "detector is not independent waveform remeasurement.",
        "get_interval_waveform_context",
        arguments={"interval": "pr", "include_beat_flags": True},
    )


def _rhythm_step(step_id: str, question: str, *, gate: str = "required") -> dict[str, Any]:
    return _step(step_id, question, "get_rhythm_profile", gate=gate,
                 arguments={"sections": ["background", "atrial_signal", "av_association", "aberrancy", "preexcitation"]})


def _map_step(step_id: str, question: str, profile: str, *, gate: str = "required") -> dict[str, Any]:
    return _step(step_id, question, "get_morphology_map", gate=gate, arguments={"profile": profile})


def _native_step(step_id: str, question: str, profile: str = "qrs_infarct", *, gate: str = "required") -> dict[str, Any]:
    return _step(step_id, question, "get_native_beat_profile", gate=gate,
                 arguments={"profile": profile, "max_beats": 3})


_QUALITY_DEFINITIONS = {
    "non_diagnostic_ecg": "Do explicit quality/coverage measurements show that no diagnostic domain is interpretable? One unavailable interval alone is insufficient.",
    "technically_limited": "Which measured artifact, unreliable lead or unavailable interval materially limits an identified diagnostic domain while other domains remain interpretable?",
    "diagnostic_coverage_limited": "Which required leads or diagnostic domains are explicitly unavailable or not assessed? A normal screening extreme does not establish complete coverage.",
    "complex_morphology_limited_interpretation": "Do measured mixed, paced or ambiguous complexes prevent reliable interpretation of a specified interval or morphology domain? Complexity alone is insufficient.",
    "minor_signal_quality_issue": "Is a documented localized artifact minor, with the required diagnostic measurements still usable? Severe or unquantified degradation cannot be called minor.",
    "repeat_ecg_required": "Does a specific acquisition defect or unresolved measurement limitation prevent answering a named diagnostic question and support obtaining another recording?",
    "uncalibrated_amplitudes": "Does acquisition metadata explicitly document per-lead normalization or unavailable physical amplitude calibration? Do not infer calibration failure from unusual voltages.",
    "unclassified_ecg_abnormality": "Is at least one reliable measurement demonstrably abnormal under an applicable reference, while evidence is insufficient to name a specific pattern? Missing data alone are not an abnormality.",
}


_ST_DEFINITIONS = {
    "st_elevation": "Measure positive J-point displacement from a reliable baseline; give the actual leads and mV, and any contiguous distribution. This is an ST phenotype, not a diagnosis of infarction.",
    "st_depression": "Measure negative ST displacement, its J/60/80-ms timing and slope in the actual leads; distinguish horizontal/downsloping from upsloping change. Report the phenotype separately from ischemia.",
    "acute_occlusion_pattern": "Identify a specified occlusion-compatible pattern: for conventional adult ST elevation require two contiguous leads, J elevation >=0.1 mV except V2-V3 (men <40: >=0.25, men >=40: >=0.20, women: >=0.15 mV). Unknown age/sex or unassessed alternative-pattern components are unknown, not normal.",
    "left_main_pattern": "Is ST depression >=0.1 mV measured in at least six leads with ST elevation in aVR and/or V1? Name each lead; this is a diffuse ischemic pattern, not proof of left-main anatomy.",
    "posterior_ischemia_screen": "Is there reliable ST depression in V1-V3 with compatible terminal positive T waves and anterior R-wave context? This is a posterior ischemia screen; do not invent V7-V9 recordings or posterior infarction confirmation.",
    "de_winter_pattern": "Are upsloping J-point ST depressions in precordial leads accompanied by tall symmetric T waves? Require the measured ST/T combination; isolated upsloping depression or a tall T wave is insufficient.",
    "wellens_pattern": "Are biphasic or deeply symmetric inverted anterior T waves, particularly V2-V3, present with little ST displacement, preserved R progression and no explanatory anterior pathological Q waves? The ECG pattern cannot establish symptom timing or a culprit stenosis.",
    "sgarbossa_positive": "In an established LBBB or ventricular-paced substrate, calculate a named original or modified Sgarbossa rule using same-lead QRS polarity and J-point ST amplitude: concordant elevation >=0.1 mV, concordant depression >=0.1 mV in V1-V3, or the specified discordance rule. Original discordant elevation >=0.5 mV alone scores only 2 (<3); modified proportional discordance requires ST elevation >=0.1 mV and ST/abs(S) >=0.25. Missing signed QRS or zero S denominator is unknown.",
    "acute_pericarditis_pattern": "Is reliable widespread ST elevation in a nonterritorial distribution present, with PR-segment deviation assessed when available? PR interval duration is not PR-segment depression; missing segment data cannot be invented.",
    "early_repolarization_pattern": "Is a terminal QRS notch or slur with J-peak >=0.1 mV present in two contiguous inferior/lateral leads, excluding V1-V3, with QRS <120 ms? A generic notch flag or concave ST elevation alone does not establish its terminal location or this pattern.",
}


_T_DEFINITIONS = {
    "t_wave_abnormality": "Identify a reproducible abnormal T amplitude, polarity or contour in named leads after lead-specific normal variants and signal quality are assessed.",
    "flat_t_wave_pattern": "Identify reliably low/flat T amplitude in named leads against an explicit amplitude reference; distinguish noise, uncalibrated voltage and a missed T peak.",
    "giant_negative_t_wave": "Measure deep negative T amplitudes and distribution against a stated giant-T criterion; an inversion flag alone cannot establish magnitude.",
    "hyperacute_t_wave_pattern": "Assess broad, bulky, relatively large symmetric T waves in contiguous leads relative to their QRS and ST baseline; an absolute T amplitude alone cannot establish acute ischemia.",
    "primary_t_wave_abnormality": "Demonstrate T changes not adequately explained by the measured depolarization pattern; unresolved secondary QRS effects leave primary attribution unknown.",
    "secondary_t_wave_abnormality": "Demonstrate a depolarization abnormality and the corresponding same-lead ST/T relationship; wide QRS alone does not establish secondary T changes.",
    "digitalis_effect_pattern": "Demonstrate a reproducible scooped ST/T contour with compatible repolarization measurements; morphology cannot establish digoxin exposure or toxicity.",
    "wide_qrs_repolarization_review": "Are QRS widening and a specified ST/T or QT interpretation limitation documented? This code denotes a review need, not a positive ischemia finding.",
}


def _definition(code: str) -> tuple[list[dict[str, Any]], list[str]]:
    if code == "normal_ecg":
        return ([
            _step("normal_scalar_coverage", "Are rate, PR, QRS, QT/QTc and axes all reliably available and within appropriate age/sex references? Missing or unassessed domains are unknown, not normal.", "get_diagnostic_overview", arguments={}),
            _map_step("normal_quality_coverage", "Are the required leads and P/QRS/ST/T measurements adequately recorded and calibrated?", "quality"),
            _rhythm_step("normal_rhythm", "Do genuine sinus P waves and their AV relationship support a normal rhythm without significant ectopy or pacing?"),
            _map_step("normal_qrs", "Are QRS morphology, voltage and R progression normal across the required leads?", "qrs"),
            _map_step("normal_st", "Is reliable ST morphology normal throughout the required lead territories?", "st"),
            _map_step("normal_tu", "Are T/U morphology and polarity normal for the lead, age and sex?", "t_u"),
        ], ["A negative rule list, normal screening extremes or absence of raised candidates cannot establish a normal ECG."])
    if code in _QUALITY_DEFINITIONS:
        return ([
            _step("documented_limitation", _QUALITY_DEFINITIONS[code], "get_diagnostic_overview", arguments={}),
            _map_step("limitation_extent", "Identify the affected leads/components and the usable residual evidence; distinguish an unavailable measurement from a measured normal or abnormal result.", "quality"),
        ], ["Report only the measured limitation and its scope; a quality label must not erase usable findings or invent disease."])
    if code in _SINUS:
        return (
            [
                _map_step("sinus_p_morphology", "Do repeated genuine P waves have a sinus-compatible lead distribution (normally positive I/II and negative aVR), after placement and ectopic atrial origin are assessed? Sinus arrhythmia or AV block can coexist with sinus origin.", "p"),
                _step(
                    "organized_atrial_activity",
                    "Characterize background regularity and atrial availability; sinus arrhythmia or AV block need not refute demonstrated sinus P morphology.",
                    "get_rhythm_profile",
                    gate="supporting",
                    arguments={"sections": ["background", "atrial_signal", "av_association"]},
                ),
                _step(
                    "sinus_p_support",
                    "Are multi-beat P-wave assessments reliable and free of material boundary ambiguity?",
                    "get_p_assessment_table",
                    arguments={
                        "fields": ["accepted", "ta_ambiguous", "onset_confidence", "offset_confidence", "valid_leads", "reject_reasons"],
                        "limit": 12,
                    },
                ),
            ],
            ["AV block can coexist with sinus origin; nonconducted P waves alone do not refute a sinus mechanism."],
        )
    if code in _RATE:
        rate_fields = ["heart_rate_bpm", "rr_cv"]
        if code in {"sinus_bradycardia", "sinus_tachycardia"}:
            # Both deterministic nodes consume one shared scalar view.  This
            # avoids spending a second tool slot merely to reread heart rate.
            rate_fields.extend(["atrial_rate_bpm", "p_axis_deg"])
        steps = [
            _step(
                "rate_threshold",
                "Does the global heart rate meet the definition-level criterion for this rate candidate?",
                "get_global_table",
                arguments={"fields": rate_fields},
            )
        ]
        if code in {"sinus_bradycardia", "sinus_tachycardia"}:
            steps.extend(
                [
                    _map_step("sinus_p_morphology", "Does reliable cross-lead P morphology support sinus origin rather than an ectopic atrial or flutter mechanism? Rate alone cannot answer this.", "p"),
                    _step(
                    "sinus_mechanism_support",
                    "Do the directly measured atrial rate, ventricular rate and P-wave axis support sinus origin?",
                    "get_global_table",
                    arguments={"fields": rate_fields},
                    ),
                    _step(
                        "sinus_p_support",
                        "Do repeated clean P waves without T/A boundary ambiguity support sinus origin?",
                        "get_p_assessment_table",
                        arguments={
                            "fields": [
                                "accepted",
                                "ta_ambiguous",
                                "onset_confidence",
                                "offset_confidence",
                                "valid_leads",
                                "reject_reasons",
                            ],
                            "limit": 12,
                        },
                    ),
                    _step(
                        "sinus_candidate_stream_reconciled",
                        "Is the atrial candidate-event stream free of focal over-detection that still requires validation?",
                        "get_rhythm_profile",
                        arguments={"sections": ["av_association"]},
                    ),
                ]
            )
        return (
            steps,
            ["A rate abnormality does not establish the rhythm mechanism; sinus, atrial and ventricular origins require independent assessment."],
        )
    if code in _AXIS:
        return (
            [
                _step(
                    "axis_threshold",
                    "Does the reliable QRS axis fall within the definition range for this axis candidate?",
                    "get_global_table",
                    arguments={"fields": ["qrs_axis_deg"]},
                )
            ],
            ["Axis deviation is a measured phenotype and cannot by itself establish fascicular block or ventricular hypertrophy."],
        )
    if code in _FIRST_DEGREE:
        return (
            [
                _p_identity_step(),
                _step(
                    "pr_criterion",
                    "Does the reliable PR measurement meet this candidate's definition-level criterion?",
                    "get_interval_waveform_context",
                    arguments={"interval": "pr", "include_beat_flags": False},
                ),
                _step(
                    "one_to_one_av",
                    "Is there stable one-to-one P-QRS association?",
                    "get_atrial_event_table",
                    arguments={
                        "fields": _ATRIAL_FIELDS,
                        "limit": 12,
                    },
                ),
            ],
            ["Adult PR must exceed 200 ms with genuine 1:1 P-QRS conduction; pediatric criteria require an age-appropriate reference. Missing PR is unknown, not a normal PR.",
             "PR and event labels can share one detection error; waveform identity and P/T ambiguity must be assessed separately."],
        )
    if code in _ADVANCED_AV:
        complete = code in {"complete_av_block", "complete_av_block_pattern"}
        return ([
            _p_identity_step(),
            _step("repeated_blocked_atrial_events", "Are there repeated reliable atrial events without ventricular conduction, with quality and source leads explicitly available?", "get_atrial_event_table", arguments={"fields": _ATRIAL_FIELDS, "limit": 16}),
            _step("sequential_av_pattern", "Does the full sequential P-QRS timing support this degree of AV block? A truncated sample cannot establish absence of conduction.", "get_rhythm_profile", arguments={"sections": ["av_association"]}),
            _rhythm_step("av_block_mechanism",
                "Do independent atrial and ventricular sequences demonstrate no AV conduction, rather than isorhythmic/interference dissociation, ectopy or pacing? AV dissociation alone is insufficient for complete block."
                if complete else
                "Do on-time genuine P waves intermittently fail to conduct with other P waves conducted? Distinguish Wenckebach, fixed-PR block and untyped 2:1 conduction; exclude premature blocked atrial beats, flutter waves, detector duplicates and concealed ectopy. Do not subtype 2:1 block from ratio alone."),
        ], ["One detector event does not establish advanced AV block; missing atrial quality or insufficient sequence duration is unknown.",
            "AV dissociation does not itself prove complete block; a 2:1 pattern does not distinguish Mobitz I from II."])
    if code in _IVCD:
        return (
            [
                _rhythm_step("conduction_applicability", "Do native beat timing and atrial relation distinguish conduction delay from ventricular rhythm, pacing, pre-excitation or isolated aberrancy?"),
                _step("dominant_qrs_wide", "Is QRS duration reliably prolonged in the dominant morphology group?", "get_morphology_groups", arguments={"include_beats": False}),
                _step("wide_qrs_representative", "Does QRS widening involve most beats rather than one abnormal group?", "get_morphology_groups", arguments={"include_beats": False}),
                _step(
                    "specific_bbb_excluded",
                    "Does cross-lead morphology exclude a specific bundle branch block pattern?",
                    "get_qrs_measurement_bundle",
                    arguments={"leads": ["I", "V1", "V5", "V6"]},
                ),
                _step(
                    "preexcitation_excluded",
                    "Do the rhythm and pre-excitation views exclude a pre-excitation pattern?",
                    "get_rhythm_profile",
                    arguments={"sections": ["preexcitation", "atrial_signal"]},
                ),
            ],
            ["Nonspecific IVCD requires adult QRS >110 ms without complete BBB morphology; unknown exclusion evidence leaves it unresolved. Report measured QRS widening separately.", "Pacing, pre-excitation, ventricular origin and transient aberrancy must be distinguished from native conduction delay."],
        )
    if code in _BBB:
        family = semantic_candidate_family(code)
        criterion = {
            "rbbb": "Require complete adult QRS >=120 ms, compatible terminal R-prime in V1/V2 and broad terminal S in I/V6; an rSr-prime with narrow QRS alone is insufficient.",
            "incomplete_rbbb_pattern": "Require adult QRS 110 to <120 ms with the RBBB terminal pattern in V1/V2 and I/V6; distinguish normal narrow rSr-prime and lead placement.",
            "lbbb": "Require adult QRS >=120 ms, broad/notched or slurred lateral R waves, absent normal lateral q waves (I/V5/V6) and compatible right-precordial morphology; duration alone is insufficient.",
            "lafb": "Require leftward axis approximately -45 to -90 degrees, qR in I/aVL and rS inferiorly; exclude inferior Q-wave change and other axis causes. Coexisting RBBB can widen QRS.",
            "lpfb_pattern": "Require rightward adult axis approximately +90 to +180 degrees, rS in I/aVL and qR inferiorly; assess RVH, vertical heart and infarct mimics before attributing the axis to LPFB.",
            "bifascicular": "Demonstrate both a complete RBBB pattern and a separately supported left anterior or posterior fascicular pattern; axis deviation alone is insufficient.",
            "trifascicular_block_pattern": "Require evidence localizing disease to all three fascicles, such as documented alternating bundle patterns or independently established infranodal disease. Bifascicular block plus long PR does not localize the AV delay and cannot establish this mechanism from morphology alone.",
        }[family]
        return ([
            _step("qrs_duration_support", "Is reliable dominant QRS duration compatible with this conduction pattern and the patient's age?", "get_morphology_groups", arguments={"include_beats": False}),
            _step("conduction_definition", criterion, "get_qrs_measurement_bundle", arguments={"leads": _QRS_LEADS}),
            _step("pattern_representative", "Does this morphology represent principal native beats rather than an isolated premature or paced beat?", "get_morphology_groups", arguments={"include_beats": False}),
            _rhythm_step("conduction_applicability", "Are native conduction and the relevant competing explanations adequately assessed, including pre-excitation, pacing and rate-related aberrancy? Missing evidence needed to distinguish them is unknown."),
            _step("conduction_axis_or_av", "For fascicular patterns, use measured axis with lead morphology; for a three-fascicle claim, PR prolongation alone is insufficient localization. For isolated BBB, axis is contextual only.", "get_global_table", gate="required" if family in {"lafb", "lpfb_pattern", "bifascicular", "trifascicular_block_pattern"} else "supporting", arguments={"fields": ["qrs_axis_deg", "pr_ms"]}),
            _step("preexcitation_excluded", "Does available pre-excitation evidence avoid establishing an alternative activation pattern? Detector-only or unavailable evidence remains unknown.", "get_rhythm_profile", gate="invalidator", arguments={"sections": ["preexcitation", "atrial_signal"]}),
        ], ["BBB requires duration plus specific lead morphology; use age-appropriate criteria for children.",
            "An uncertain specific BBB pattern can leave a measured wide-QRS phenotype; it does not automatically confirm nonspecific IVCD.",
            "Neither a long PR nor axis deviation alone proves fascicular localization."])
    if code in _PREEXCITATION:
        return (
            [
                _step("preexcitation_components", "Are short PR and cross-lead delta candidates both present?", "get_rhythm_profile", arguments={"sections": ["preexcitation"]}),
                _step("pr_component", "Is an independently reportable short PR component present?", "get_interval_waveform_context", arguments={"interval": "pr", "include_beat_flags": False}),
                _step("qrs_onset_support", "Do measured initial QRS slurring across leads and compatible QRS duration corroborate a delta wave, rather than a terminal notch, BBB or pacing? A delta-detector hit alone is insufficient.", "get_qrs_measurement_bundle", arguments={"leads": ["I", "II", "V1", "V4", "V5", "V6"]}),
            ],
            ["A positive delta detector without independent short-PR evidence cannot confirm pre-excitation."],
        )
    if code == "short_pr_interval":
        return (
            [
                _p_identity_step(),
                _step(
                    "short_pr_criterion",
                    "Does the reliable PR measurement meet the short-PR definition?",
                    "get_interval_waveform_context",
                    arguments={"interval": "pr", "include_beat_flags": False},
                )
            ],
            ["Short PR is an interval phenotype and cannot be upgraded to pre-excitation without delta-wave morphology."],
        )
    if code in _LVH:
        return ([
            _step("voltage_criteria_qrs_valid", "Is dominant native QRS duration within the applicability range of this conventional voltage criterion?", "get_morphology_groups", gate="invalidator", arguments={"include_beats": False}, unknown_blocks_confirmation=True, failure_means_not_applicable=True),
            _step("voltage_criteria_pacing_valid", "Is ventricular pacing absent or are the voltage measurements explicitly confined to validated native beats?", "get_pacing_profile", gate="invalidator", arguments={"include_beats": False}, unknown_blocks_confirmation=True, failure_means_not_applicable=True),
            _step("cross_lead_voltage_criterion", "Compute a named criterion from calibrated components: Cornell R(aVL)+abs(S(V3)) with applicable sex reference, or Sokolow-Lyon abs(S(V1))+max(R(V5),R(V6)). Do not substitute a neighboring lead or assume absent components are zero.", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "reliable_for_qrs"], "leads": ["aVL", "V1", "V2", "V3", "V4", "V5", "V6"]}),
            _step("voltage_criterion_applicability", "Are physical mV calibration and the age/sex reference required for the selected criterion explicitly available? Pediatric voltage needs an age-appropriate reference; missing demographics or normalization cannot establish adult thresholds.", "get_diagnostic_overview", gate="invalidator", arguments={}, unknown_blocks_confirmation=True, failure_means_not_applicable=True),
            _step("qrs_morphology_compatible", "Does cross-lead QRS morphology support interpretation, with BBB, pre-excitation and placement confounders characterized? A positive voltage criterion is not anatomic hypertrophy.", "get_qrs_measurement_bundle", arguments={"leads": ["I", "aVL", "V1", "V2", "V5", "V6"]}),
        ], ["Read the actual components of the named voltage criterion; rule hits are not patient evidence.",
            "Conventional voltage criteria under wide QRS or ventricular pacing are not validated here; inapplicability does not exclude anatomic LVH.",
            "ECG voltage is a phenotype; chamber anatomy requires independent assessment."])
    if code in _RVH:
        return (
            [
                _step("rvh_reference_applicability", "Are calibration and an applicable adult or pediatric age reference available? Assess RBBB, posterior Q-wave equivalents, pre-excitation, lead placement and normal juvenile rightward forces before attributing the pattern to RVH.", "get_diagnostic_overview", gate="invalidator", arguments={}, unknown_blocks_confirmation=True, failure_means_not_applicable=True),
                _step("right_precordial_voltage", "Does the voltage relationship between right and left precordial leads support a right-ventricular pattern?", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "qrs_ms", "reliable_for_qrs"], "leads": ["I", "aVF", "V1", "V2", "V5", "V6"]}),
                _step("rvh_qrs_distribution", "Does right-precordial R dominance with left-precordial S persistence have a compatible measured axis and QRS pattern, after distinguishing RBBB, posterior infarct equivalents and pre-excitation?", "get_qrs_measurement_bundle", arguments={"leads": ["I", "aVF", "V1", "V2", "V5", "V6"]}),
                _step("rvh_axis_context", "Does the measured QRS axis support the stated RVH criterion using an appropriate age reference?", "get_global_table", arguments={"fields": ["qrs_axis_deg"]}),
            ],
            ["A tall R wave in V1 alone cannot confirm RVH; assess axis and the full QRS distribution. ECG morphology does not establish chamber anatomy or pulmonary pressure."],
        )
    if code in _ATRIAL_CHAMBER:
        criterion = (
            "Require both independently demonstrated left- and right-atrial P-wave patterns; one enlarged P amplitude is insufficient."
            if code == "biatrial_abnormality" else
            "Assess prolonged/notched P waves and the duration times signed amplitude of terminal negative P in V1 using a stated age reference; amplitude alone does not establish left atrial abnormality."
            if "left" in code else
            "Assess tall positive inferior P waves and a compatible initial positive V1 component using a stated age-specific amplitude criterion; exclude ectopic P origin and placement."
        )
        return (
            [
                _step("p_wave_distribution", criterion, "get_lead_table", arguments={"fields": ["p_dur_ms", "p_amp_mv", "p_notched", "p_biphasic", "p_terminal_duration_ms", "p_terminal_amp_mv"], "leads": ["II", "III", "aVF", "V1", "V2"]}),
                _step("p_measurement_reliability", "Are multi-beat P-wave boundary assessments sufficient to support an atrial abnormality interpretation?", "get_p_assessment_table", arguments={"fields": ["accepted", "ta_ambiguous", "onset_confidence", "offset_confidence", "valid_leads", "reject_reasons"], "limit": 12}),
            ],
            ["Missing P boundaries, calibration or an appropriate pediatric reference leave the relevant criterion unknown; absent P detection is not normal atrial morphology.", "P-wave abnormalities do not establish chamber size; biatrial abnormality requires both component patterns."],
        )
    if code in _LOW_VOLTAGE:
        leads = ["I", "II", "III", "aVR", "aVL", "aVF"] if "limb" in code else ["V1", "V2", "V3", "V4", "V5", "V6"]
        return (
            [_step("territorial_qrs_voltage", "Do calibrated peak-to-trough QRS voltages meet the low-voltage criterion in every required lead of the target group (<0.5 mV limb, <1.0 mV precordial in adults)? Missing leads or invalid signed amplitudes are unknown.", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "q_amp_mv", "reliable_for_qrs"], "leads": leads}),
             _step("low_voltage_calibration", "Are physical voltage calibration, signal quality and all target leads adequate? Low voltage does not establish effusion, obesity or infiltrative disease.", "get_diagnostic_overview", gate="invalidator", arguments={}, unknown_blocks_confirmation=True, failure_means_not_applicable=True)],
            ["Assess limb and precordial lead groups separately; do not extrapolate from a single lead."],
        )
    if code == "atrial_fibrillation_flutter_indeterminate":
        return ([
            _rhythm_step("atrial_arrhythmia_observed", "Is abnormal atrial activity actually demonstrated, beyond irregular RR or failed P detection alone?"),
            _rhythm_step("af_afl_distinction_unresolved", "Do observable atrial organization and cycle measurements leave AF versus flutter unresolved despite a targeted assessment? A generic unavailable atrial signal cannot establish this specific differential."),
            _step("atrial_observability", "Do signal quality and atrial assessments permit identifying abnormal atrial activity while its organization remains unresolved? Do not require sinus P waves for this differential.", "get_p_assessment_table", arguments={"fields": _P_FIELDS, "limit": 12}),
        ], ["This reports a documented unresolved atrial-arrhythmia distinction, not confirmed AF or flutter."])
    if code in _AF:
        return ([
            _step("irregular_ventricular_response", "Does measured RR variability support AF, after ectopic beats and artifact are considered? Regular response may occur with AV block or pacing.", "get_rhythm_profile", gate="supporting", arguments={"sections": ["background", "atrial_signal"]}),
            _step("organized_p_absent", "Do adequately observable atrial assessments demonstrate no consistent organized P waves? Failed or unavailable P extraction is unknown, not absence.", "get_p_assessment_table", arguments={"fields": _P_FIELDS, "limit": 12}),
            _rhythm_step("af_atrial_mechanism", "Does demonstrable disorganized atrial activity with no regular P-wave sequence support AF after flutter, multifocal atrial activity, frequent ectopy and noise are assessed? Interpret RR conditional on AV conduction and pacing; a low P-acceptance fraction or detector confidence alone is insufficient."),
            _step("multilead_atrial_support", "Is the atrial candidate signal corroborated across leads? Multiple outputs of one detector are not independent confirmation.", "get_rhythm_profile", gate="supporting", arguments={"sections": ["background", "atrial_signal"]}),
        ], ["Irregular RR alone does not establish AF; regular RR does not exclude AF with AV block or pacing.",
            "If atrial activity cannot be distinguished from noise or failed P extraction, mechanism remains unknown."])
    if code in _AFL:
        return ([
            _rhythm_step("organized_flutter_activity", "Are repeated organized atrial cycles and consistent flutter-wave morphology demonstrated, rather than T-wave harmonics, sinus P waves or noise? A rate near 150 bpm alone is insufficient."),
            _step("multilead_flutter_support", "Do event times, morphology similarity and source leads corroborate repeated flutter activity across clean leads? Candidate counts alone are insufficient.", "get_atrial_event_table", arguments={"fields": [*_ATRIAL_FIELDS, "template_similarity"], "limit": 16}),
            _rhythm_step("atrial_rhythm_definition", "Is flutter morphology and cycle regularity distinguishable from focal atrial tachycardia or fibrillation? Do not infer a re-entry circuit or isthmus dependence from the surface pattern alone."),
            _step("av_relation_characterized", "Characterize fixed or variable AV conduction if visible; an unresolved conduction ratio does not refute demonstrated flutter.", "get_atrial_event_table", gate="supporting", arguments={"fields": [*_ATRIAL_FIELDS, "template_similarity"], "limit": 16}),
        ], ["A ventricular rate near 150 bpm prompts a flutter search but cannot prove or exclude 2:1 flutter.",
            "Flutter may have fixed or variable ventricular response; typical morphology is not mandatory for every flutter circuit."])
    if code in _OTHER_ATRIAL_RHYTHM:
        criterion = {
            "atrial_tachycardia": "Require organized non-sinus P-wave activity with an atrial rate >100 bpm in adults and a reproducible atrial sequence; distinguish sinus tachycardia and flutter before assigning atrial tachycardia.",
            "ectopic_atrial_rhythm_pattern": "Require repeatable non-sinus P polarity/axis with an organized atrial sequence and compatible AV timing; do not localize an anatomic focus from P morphology alone.",
            "multifocal_atrial_rhythm": "Require at least three distinct genuine P morphologies in the same lead, variable PP/PR timing and an isoelectric interval; in adults distinguish a rate <=100 bpm from multifocal atrial tachycardia. Cluster IDs alone are not three verified morphologies.",
            "multifocal_atrial_tachycardia": "Require at least three distinct genuine P morphologies in the same lead, an isoelectric interval, variable PP/PR/RR timing and adult atrial rate >100 bpm; distinguish AF, ectopy and artifact. Cluster IDs alone are insufficient.",
            "p_wave_abnormality": "Identify a reproducible abnormal P duration, amplitude, polarity or contour in named leads using an applicable reference; report P morphology without requiring or inventing an ectopic rhythm mechanism.",
            "junctional_rhythm_pattern": "Require a compatible junctional sequence with absent or retrograde atrial activity and its timing relative to QRS; failed P extraction or an isolated short PR alone does not establish junctional origin.",
        }[code]
        return ([
            _rhythm_step("atrial_mechanism", "Characterize measured atrial/ventricular rates and AV timing without inferring a mechanism from rate alone.", gate="supporting" if code == "p_wave_abnormality" else "required"),
            _step("p_morphology_support", "Is atrial observability sufficient to assess genuine P morphology, or absent/retrograde P in a junctional candidate, after P/T ambiguity and noise review? Failed P extraction alone is unknown.", "get_p_assessment_table", arguments={"fields": _P_FIELDS, "limit": 12}),
            _map_step("atrial_rhythm_definition", criterion, "p"),
        ], ["Missing atrial morphology is unknown; absence of accepted detector events does not prove absent P waves or a junctional mechanism.", "Adult rate boundaries must not be applied to pediatric rhythms without an appropriate reference."])
    if code in _VENTRICULAR_RHYTHM:
        phenotype = code == "wide_complex_tachycardia"
        return ([
            _step("wide_complex_sequence", "Do ordered beat IDs, durations and RR intervals show a consecutive wide-complex run at the candidate-compatible rate? A morphology group's total count is not a consecutive run.", "get_morphology_groups", arguments={"include_beats": True}),
            _step("ventricular_morphology", "Do capture/fusion morphology or other discriminating QRS features support ventricular origin? Wide QRS alone is insufficient.", "get_morphology_groups", gate="supporting" if phenotype else "required", arguments={"include_beats": True}),
            _rhythm_step("atrial_relation", "Characterize AV dissociation, capture/fusion or atrial-driven aberrancy and pre-excitation; do not equate an uncertain mechanism with absent wide-complex tachycardia.", gate="supporting" if phenotype else "required"),
        ], ["Wide-complex tachycardia is a reportable rate/QRS phenotype even when its mechanism remains unknown.", "Ventricular origin requires additional discrimination from SVT with aberrancy, pre-excitation and pacing."])
    if code in _ECTOPY:
        atrial = code == "premature_atrial_complexes"
        return ([
            _step("premature_timing", "Does event timing demonstrate prematurity relative to the local native cycle, rather than merely a different QRS group?", "get_beat_table", arguments={"fields": ["rr_prev_ms", "rr_next_ms", "group_id", "paced"]}),
            _step("ectopic_morphology", "Does the ectopic event have credible morphology distinct from artifact? QRS widening alone is not ventricular origin.", "get_morphology_groups", gate="supporting", arguments={"include_beats": True}),
            _step("ectopic_event_quality", "Is at least one identified premature event reliably measured and distinguishable from artifact, pacing and escape beats? A genuine single event suffices; it need not belong to the dominant or repeated morphology group.", "get_morphology_groups", arguments={"include_beats": True}),
            _rhythm_step("ectopic_origin",
                "Is a premature non-sinus atrial depolarization and its AV timing demonstrated, allowing normal, aberrant or blocked conduction? A premature QRS without atrial evidence cannot distinguish PAC from PVC."
                if atrial else
                "Does the premature complex have a ventricular activation pattern after assessing conducted premature P waves with aberrancy, pre-excitation and pacing? A compensatory pause is supportive but neither required nor sufficient."),
        ], ["A single well-characterized premature event can establish ectopy; a nondominant group alone cannot.", "Origin is independent of premature timing; unavailable atrial evidence must not automatically label a wide premature beat as PVC."])
    if code in _QT:
        return (
            [
                _step("qt_reference_applicability", "Are the correction formula, RR/rate, age/sex reference and QRS duration appropriate for interpreting this QTc? Wide QRS requires a stated QRS-adjusted or JT approach; do not invent JTc when unavailable. QT phenotype does not establish inherited long/short-QT syndrome.", "get_interval_waveform_context", gate="invalidator", arguments={"interval": "qt", "include_beat_flags": False}, unknown_blocks_confirmation=True, failure_means_not_applicable=True),
                _step("interval_reportable", "Can QT or QTc be reported reliably?", "get_interval_waveform_context", arguments={"interval": "qt", "include_beat_flags": False}),
                _step("qt_threshold", "Does a reportable QTc meet the definition range for this long- or short-QT candidate?", "get_interval_waveform_context", arguments={"interval": "qt", "include_beat_flags": False}),
                _step("component_endpoint_support", "Does T/U-wave endpoint evidence support interval interpretation?", "get_interval_waveform_context", arguments={"interval": "qt", "include_beat_flags": False}),
            ],
            ["When the interval is not reportable, a missing value cannot confirm or exclude a QT abnormality."],
        )
    if code in _R_PROGRESSION or code == "possible_precordial_lead_reversal":
        if code == "dextrocardia_pattern":
            return ([
                _map_step("dextrocardia_limb_pattern", "Are limb P/QRS polarities, including negative lead I and positive aVR, compatible with a dextrocardia pattern? QRS axis alone is insufficient.", "p"),
                _map_step("dextrocardia_precordial_pattern", "Does reversed or persistently poor precordial R progression accompany the limb pattern, rather than preserved precordial progression suggesting limb-electrode reversal?", "qrs"),
            ], ["Surface pattern cannot prove cardiac position; lead placement and imaging/history are separate evidence."])
        if code == "possible_precordial_lead_reversal":
            return ([
                _map_step("precordial_sequence_discontinuity", "Does an abrupt anatomically implausible jump/reversal in neighboring precordial QRS and R/S measurements support a placement problem? Smooth poor progression alone is insufficient.", "qrs"),
                _map_step("placement_corroboration", "Do corresponding P/ST/T lead discontinuities corroborate a lead-placement suspicion rather than a coherent conduction or scar pattern? Do not claim a particular swapped pair without evidence.", "p"),
            ], ["This is suspected lead placement error; repeat acquisition is needed to verify it."])
        return ([
            _step("precordial_progression", "Do signed R/S measurements establish delayed transition (clockwise), early transition (counterclockwise), poor R growth or reversed R progression as specifically named? Missing leads and invalid signed S amplitudes are unknown.", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "q_amp_mv", "qrs_ms", "reliable_for_qrs"], "leads": ["V1", "V2", "V3", "V4", "V5", "V6"]}),
            _native_step("progression_context", "Characterize placement, BBB, pre-excitation and Q-wave confounders without erasing the measured progression phenotype; do not infer physical cardiac rotation or prior infarction.", gate="supporting"),
        ], ["R progression and transition are descriptive phenotypes; they do not establish anatomic rotation or infarction."])
    if code in _ST_PATTERNS:
        phenotype = code in {"st_elevation", "st_depression"}
        substrate = (
            "Is a genuine LBBB or ventricular-paced substrate established in the beats whose QRS and ST are being compared? A spike detector or nonspecific wide QRS alone is insufficient."
            if code == "sgarbossa_positive" else
            "Are calibration, baseline and native QRS context adequate for this specific pattern's interpretation, with LVH, BBB, pre-excitation, pacing and placement effects distinguished? Do not apply ordinary ischemic ST thresholds to unresolved secondary repolarization."
        )
        return ([
            _map_step("st_change_distribution", "Identify reliable ST amplitudes, measurement timing, baseline confidence, contiguous territory and reciprocal changes; a shape flag alone cannot establish disease.", "st"),
            _map_step("st_pattern_definition", _ST_DEFINITIONS[code], "st"),
            _map_step("st_t_components", "Assess the T-wave polarity, amplitude and contour required by this particular ST/T pattern; missing defining T morphology is unknown.", "t_u", gate="supporting" if phenotype or code == "sgarbossa_positive" else "required"),
            _step("st_qrs_components", "Characterize same-lead signed Q/R/S amplitudes, Q waves and QRS onset/terminal morphology relevant to this ST pattern; use the same beat population for ST and QRS ratios.", "get_qrs_measurement_bundle", gate="supporting" if phenotype else "required", arguments={"leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]}),
            _step("st_pattern_applicability", substrate, "get_pacing_profile" if code == "sgarbossa_positive" else "get_native_beat_profile", gate="supporting" if phenotype else "invalidator", arguments={"include_beats": True} if code == "sgarbossa_positive" else {"profile": "qrs_infarct", "max_beats": 3}, unknown_blocks_confirmation=not phenotype, failure_means_not_applicable=not phenotype),
        ], ["Report reliable ST displacement independently of its cause; a pattern label does not prove occlusion, infarction, coronary anatomy or pericardial inflammation.",
            "Newness, symptoms, serial change, troponin, posterior/right-sided leads and clinical diagnoses must not be invented from a single 12-lead artifact.",
            "Unknown defining components remain unknown; a negative pattern does not exclude acute coronary disease."])
    if code in _ST_PATTERNS:
        return (
            [
                _step("st_change_distribution", "Do reliable ST changes and their contiguous-lead distribution support this candidate?", "get_morphology_map", arguments={"profile": "st"}),
                _step("st_territorial_confirmation", "Do ST changes form a consistent distribution in the relevant leads?", "get_lead_table", arguments={"fields": ["st_hybrid_j_mv", "st_hybrid_60ms_mv", "st_hybrid_shape", "st_hybrid_reliable"]}),
                _step("secondary_qrs_context", "Does wide QRS or another secondary cause alter the ST interpretation?", "get_native_beat_profile", gate="supporting", arguments={"profile": "qrs_infarct", "max_beats": 3}),
            ],
            ["A single-lead or low-quality ST candidate cannot confirm an ischemic interpretation."],
        )
    if code in _T_PATTERNS:
        return ([
            _map_step("t_wave_distribution", _T_DEFINITIONS[code], "t_u"),
            _native_step("repolarization_context", "Do repeated reliable ST/T measurements support the distribution and waveform identity, including T/U separation?", "repolarization"),
            _step("t_qrs_relation", "Compare ST/T polarity and magnitude with same-lead QRS activation and voltage; assess BBB, pacing, pre-excitation and LVH before assigning primary or secondary change.", "get_qrs_measurement_bundle", gate="required" if code in {"primary_t_wave_abnormality", "secondary_t_wave_abnormality", "wide_qrs_repolarization_review", "hyperacute_t_wave_pattern"} else "supporting", arguments={"leads": _QRS_LEADS}),
        ], ["T-wave morphology is reportable even when QT cannot be measured; QT failure does not erase T observations.", "Morphology alone does not establish ischemia, cardiomyopathy, electrolyte concentration or drug exposure."])
    if code in _Q_PATTERNS:
        return ([
            _step("q_wave_morphology", "Name the criterion version and route. Fifth UDMI (2026 Table 5): Q duration >=40 ms and/or abs(Q)/R >=0.25 in two contiguous leads. Require signed negative Q and positive nonzero R for a ratio; missing duration leaves the duration route unknown, never zero or a negative test. A separately demonstrated QS pattern may use the legacy Fourth UDMI (2018) lead-specific route, explicitly identified as such; missing R alone is not QS and QS does not provide a finite Q/R ratio.", "get_lead_table", arguments={"fields": ["q_duration_ms", "q_amp_mv", "r_amp_mv", "qrs_ms"]}),
            _native_step("q_wave_distribution", "Verify the qualifying contiguous Q/QS distribution and repeated native morphology. An isolated QS in V1 or small isolated III/aVL Q can be a variant; do not infer QS solely from absent measurements."),
            _step("q_wave_applicability", "Are lead placement and depolarization confounders assessed sufficiently for pathological/prior-infarct interpretation, especially LVH, LBBB, pre-excitation and pacing? Observed Q/QS morphology may remain reportable even when infarct attribution is inapplicable.", "get_qrs_measurement_bundle", gate="invalidator", arguments={"leads": _QRS_LEADS}, unknown_blocks_confirmation=True, failure_means_not_applicable=True),
        ], ["Duration-defined Q waves and separately demonstrated QS morphology are alternative evidence routes; unavailable duration cannot confirm or exclude the former.",
            "A Q-wave pattern cannot date an infarct or establish ischemic scar without considering nonischemic causes and clinical/imaging context."])
    if code == "brugada_type1_screening":
        return ([
            _map_step("brugada_right_precordial_st", "Is coved J/ST elevation >=0.2 mV with a descending ST segment demonstrated in V1 or V2? Generic ST elevation or a saddleback pattern is insufficient for type 1.", "st"),
            _map_step("brugada_terminal_t", "Is the corresponding negative T wave present in the same right-precordial lead, with reliable waveform separation?", "t_u"),
            _step("brugada_qrs_mimics", "Distinguish RBBB, secondary repolarization and placement effects from the coved type-1 morphology; missing defining morphology remains unknown.", "get_qrs_measurement_bundle", arguments={"leads": ["V1", "V2", "I", "V6"]}),
        ], ["This is ECG screening only, not Brugada syndrome; fever, drugs, electrolytes, lead height and clinical/family history are separate unavailable context unless actually supplied."])
    if code == "pulmonary_embolism_pattern":
        return ([
            _map_step("right_strain_qrs", "Describe any measured S in I with Q in III, right-precordial QR or RBBB-like morphology; no single finding establishes pulmonary embolism.", "qrs"),
            _map_step("right_strain_t", "Is a compatible right-precordial V1-V4 and/or inferior T-inversion distribution present, after secondary QRS and normal variants are assessed?", "t_u"),
        ], ["Report a compatible right-heart-strain phenotype only; neither its presence nor absence diagnoses or excludes pulmonary embolism, which requires clinical assessment and appropriate testing."])
    if code in _TU_METABOLIC:
        criterion = {
            "hyperkalemia_pattern": "Assess measured tall narrow/symmetric T morphology with QRS widening and reduced P visibility when present; isolated tall T waves are nonspecific and a normal ECG cannot exclude hyperkalemia.",
            "hypokalemia_pattern": "Assess a coherent pattern of reduced/inverted T waves, ST depression and prominent separate U waves; a long QU interval is not necessarily prolonged QT.",
            "prominent_u_wave": "Require reliable U-wave prominence with demonstrated separation from T and the next P wave in named leads; a fused T/U endpoint or detector flag alone is insufficient.",
            "inverted_u_wave": "Require a reliably negative signed U amplitude/polarity with temporal separation from T and P; unsigned prominence cannot demonstrate inversion.",
        }[code]
        steps = [_map_step("tu_morphology_distribution", criterion, "t_u"),
                  _native_step("tu_wave_identity", "Do repeated beats and reliable T/U endpoints support separate component identity, rather than residual T tail, P overlap or baseline artifact?", "repolarization")]
        if code in {"hyperkalemia_pattern", "hypokalemia_pattern"}:
            steps.append(_map_step("electrolyte_ecg_components", "Assess the companion QRS changes for a hyperkalemic phenotype or ST depression for a hypokalemic phenotype; do not infer potassium concentration or grade biochemical severity from ECG.", "qrs" if code == "hyperkalemia_pattern" else "st", gate="supporting"))
        return steps, ["Electrolyte-pattern labels are morphology hypotheses; serum chemistry is required for the biochemical diagnosis.", "Unavailable U polarity or T/U separation leaves the corresponding criterion unknown."]
    if code == "electrical_alternans":
        return (
            [
                _step(
                    "alternans_rhythm_valid",
                    "Is the underlying rhythm regular and free of atrial fibrillation or flutter, so a fixed-phase beat-to-beat amplitude comparison is interpretable?",
                    "get_rhythm_profile",
                    gate="invalidator",
                    arguments={"sections": ["background", "atrial_signal"]},
                ),
                _step(
                    "alternans_single_morphology",
                    "Is there a single dominant, non-paced QRS morphology group across the run, excluding a mixed-beat or paced confound?",
                    "get_morphology_groups",
                    gate="invalidator",
                    arguments={"include_beats": False},
                ),
                _step(
                    "precordial_amplitude_alternation",
                    "Is sustained beat-to-beat QRS amplitude alternation demonstrated in a consecutive sequence after timing and morphology confounders are checked? qrs_area is only an area proxy; if amplitude observations are unavailable, amplitude alternans remains unknown.",
                    "get_beat_table",
                    arguments={"fields": ["qrs_area", "group_id", "paced"], "lead": "V5"},
                ),
                _step(
                    "limb_amplitude_alternation",
                    "Is the sequence corroborated in a second lead? Shared artifact may affect both leads; neither area alternation nor low voltage establishes tamponade.",
                    "get_beat_table",
                    gate="supporting",
                    arguments={"fields": ["qrs_area", "group_id", "paced"], "lead": "II"},
                ),
            ],
            [
                "Electrical alternans is a beat-to-beat amplitude phenomenon, not a single-beat morphology finding; one pair of unequal beats cannot establish it.",
                "Rate-related or artifact-driven amplitude change must be excluded via rhythm regularity and a single dominant morphology before an alternating pattern can be attributed to alternans.",
            ],
        )
    if code == "exclude_2_to_1_atrial_flutter":
        return (
            [
                _step(
                    "regular_narrow_rate_near_150",
                    "Does the ventricular rate sit in the regular, narrow-QRS zone near 150 bpm associated with 2:1 atrial flutter conduction?",
                    "get_global_table",
                    arguments={"fields": ["heart_rate_bpm", "qrs_ms", "rr_cv"]},
                ),
                _step(
                    "organized_flutter_activity_absent",
                    "Is there no independently demonstrated organized flutter-rate atrial activity? If such activity is found, atrial_flutter_pattern applies instead of this exclusion candidate.",
                    "get_rhythm_profile",
                    gate="invalidator",
                    arguments={"sections": ["background", "atrial_signal"]},
                ),
                _step(
                    "hidden_atrial_activity_search",
                    "Was atrial activity actually observable during a dedicated hidden-wave search, with no demonstrated flutter pattern? An unavailable P detector or an unperformed search is unknown; failure to demonstrate flutter does not exclude it.",
                    "get_p_assessment_table",
                    arguments={"fields": _P_FIELDS, "limit": 12},
                ),
            ],
            [
                "This candidate documents a rate pattern that specifically warrants excluding 2:1 atrial flutter; it is not itself a positive rhythm diagnosis.",
                "If organized flutter-rate atrial activity is independently demonstrated, use atrial_flutter_pattern instead of this exclusion candidate.",
            ],
        )
    if code == "limb_lead_reversal_suspected":
        return (
            [
                _step(
                    "limb_lead_deflection_criterion",
                    "Does lead I show a negative-dominant QRS deflection and aVR a positive-dominant QRS deflection, the defining pattern of an LA/RA limb-lead swap?",
                    "get_lead_table",
                    arguments={"fields": ["r_amp_mv", "s_amp_mv", "reliable_for_qrs"], "leads": ["I", "aVR"]},
                ),
                _step(
                    "p_wave_axis_support",
                    "Does P-wave polarity in the same leads corroborate an electrode swap rather than a genuine rightward physiologic axis?",
                    "get_lead_table",
                    gate="supporting",
                    arguments={"fields": ["p_amp_mv"], "leads": ["I", "II", "aVR"]},
                ),
                _step(
                    "alternative_cause_excluded",
                    "Do preserved precordial R progression and the overall limb pattern distinguish suspected limb-electrode reversal from dextrocardia or an extreme-axis rhythm? Missing precordial information is unknown.",
                    "get_morphology_map",
                    arguments={"profile": "qrs", "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]},
                ),
            ],
            [
                "The defining criterion is the main-deflection direction in lead I and aVR, not any single amplitude read in isolation.",
                "Dextrocardia and a genuine extreme right-axis rhythm can mimic this pattern; do not confirm a lead swap without excluding them.",
            ],
        )

    category = DIAGNOSIS_CATALOG.get(code, ("", "other"))[1]
    if category == "pacing":
        failure = code in {"pacing_failure_to_capture_suspected", "pacing_sensing_failure_suspected"}
        criterion = {
            "paced_rhythm": "Demonstrate credible pacing stimuli with consistent subsequent atrial and/or ventricular capture; establish which chamber is paced when observable. A wide QRS or spike detector alone is insufficient.",
            "ventricular_paced_rhythm": "Demonstrate stimuli temporally preceding compatible ventricular complexes with repeated capture; distinguish atrial-only pacing, fusion and artifact. Modern conduction-system pacing need not have a very wide QRS.",
            "intermittent_pacing": "Demonstrate both captured paced events and genuinely intrinsic unpaced events in the recording. Missing spike detection alone is not proof of intrinsic conduction.",
            "pacing_failure_to_capture_suspected": "Identify a credible pacing stimulus lacking the expected subsequent depolarization in the stimulated chamber at an interpretable time; distinguish artifact, refractory-period stimulation, fusion and undersampled narrow spikes. Absent raw event timing leaves this suspicion unknown.",
            "pacing_sensing_failure_suspected": "Demonstrate inappropriate stimulus timing relative to intrinsic events or inappropriate inhibition, accounting for device mode and programmed intervals when known. Missing spikes alone cannot establish undersensing or oversensing; absent settings/event timing is unknown.",
        }[code]
        return ([
            _step("pacing_stimulus_validity" if failure else "pacing_marker_support", "Are stimulus observations credible after spike burden, noise and timing conflicts are examined? Poor global capture alignment may prompt a failure review but does not itself diagnose one.", "get_pacing_profile", arguments={"include_beats": True}),
            _step("pacing_pattern_definition", criterion, "get_pacing_profile", arguments={"include_beats": True}),
            _step("pacing_beat_corroboration", "Do event-specific QRS groups and paced/native membership corroborate the stated pacing pattern, without treating detector-derived labels as independent device evidence?", "get_morphology_groups", gate="supporting" if failure else "required", arguments={"include_beats": True}),
        ], ["Pacing detection, paced-beat labels and capture matching share one extraction chain; they are not independent confirmations.", "Suspected capture/sensing failure requires event timing and device context; a surface artifact cannot replace device interrogation."])
    return ([], ["Use only measurements directly relevant to this candidate; never confirm it from rule status."])


def attach_waveform_review(
    pathway: Mapping[str, Any],
    profile: str,
    *,
    conflict: bool = True,
) -> dict[str, Any]:
    """Return a copied pathway with one targeted, model-owned raw review.

    Call the two-argument form only after a relevant stored conflict is found.
    For an explicitly requested observation-only review, use ``conflict=False``;
    missing raw evidence then does not gate the candidate. This helper does not
    discover conflicts or attach anything to default catalog paths.

    Repeated attachment is idempotent. Different requested profiles merge into
    ``all`` and a known conflict can never be downgraded to supporting review.
    Preserve every existing node, even on a six-node base path: this targeted
    prerequisite is an additional budgeted node, never a replacement criterion.
    """
    focuses = {
        "p_av": "the identified P-wave candidate's visibility, timing and QRS/T overlap relevant to the P/PR/AV interpretation",
        "qrs": "the identified initial Q/QS identity and signed Q/R/S extrema relevant to the QRS interpretation; extrema alone do not remeasure Q-wave duration or QRS boundaries",
        "st": "the identified same-lead ST polarity/displacement and baseline/noise relevant to the ST interpretation; PR-baselined samples do not by themselves validate the physiological baseline",
        "all": "each specifically implicated P/AV, QRS or ST measurement, in its own lead and event context",
    }
    if profile not in focuses:
        raise ValueError(f"unknown waveform review profile {profile!r}")
    if not isinstance(conflict, bool):
        raise TypeError("conflict must be a boolean, determined by the caller")
    result = copy.deepcopy(dict(pathway))
    steps = result.setdefault("steps", [])
    if not isinstance(steps, list):
        raise TypeError("pathway steps must be a list")
    existing = [i for i, node in enumerate(steps)
                if isinstance(node, Mapping) and node.get("id") == "raw_measurement_consistency"]
    if len(existing) > 1:
        raise ValueError("duplicate raw_measurement_consistency nodes")
    if existing:
        previous = steps[existing[0]]
        previous_profile = (previous.get("arguments") or {}).get("profile")
        if previous_profile != profile:
            profile = "all"
        conflict = conflict or previous.get("gate") == "invalidator"
    question = (
        f"Compare actual raw observations with {focuses[profile]}. "
        "Pass only when specific usable measurements resolve the candidate-relevant "
        "discrepancy or establish consistency; observations_available, reviewed_count, "
        "or a status flag alone cannot pass. Fail only if usable raw measurements "
        "demonstrate an incompatible defining measurement for this candidate. "
        "Missing, noisy, truncated or inconclusive review is unknown; absence of "
        "raw evidence is not agreement or a negative disease finding. "
        "Do not generalize a conflict to unrelated leads, events or diagnoses. "
        "Remeasurement shares the acquisition and exported timing; it is not "
        "independent clinical validation."
    )
    node = _step(
        "raw_measurement_consistency", question, "get_waveform_review",
        gate="invalidator" if conflict else "supporting",
        arguments={"profile": profile},
        unknown_blocks_confirmation=conflict,
        failure_means_not_applicable=conflict,
    )
    node["owner"] = "model"
    if existing:
        steps[existing[0]] = node
    else:
        steps.append(node)
    return result


def build_diagnostic_pathway(
    code: str,
    *,
    fallback_checks: Sequence[Mapping[str, Any]] = (),
    program_owned: bool = True,
) -> dict[str, Any]:
    """Build one compact path, using candidate checks only as a fallback.

    The current deterministic threshold policy is adult-only.  Callers set
    ``program_owned=False`` for pediatric or unresolved age so those steps are
    adjudicated from the visible measurements instead of forcing adult norms.
    """

    steps, cautions = _definition(code)
    if not steps:
        checks = [row for row in fallback_checks if isinstance(row, Mapping)]
        first = checks[0] if checks else {}
        second = checks[1] if len(checks) > 1 else first
        steps = []
        for step_id, check, question in (
            ("direct_measurement_support", first, "State this extension code's operational definition, units, applicable reference and required lead/event distribution, then compare actual reliable measurements. An unspecified definition or missing criterion is unknown; a code name or rule status cannot supply evidence."),
            ("discriminative_countercheck", second, "Identify a concrete competing explanation and test the measurable feature that distinguishes it. Merely finding no contradiction in an incomplete view is unknown, not confirmation."),
        ):
            tool = str(check.get("tool") or "get_native_beat_profile")
            arguments = check.get("arguments")
            if not isinstance(arguments, Mapping):
                arguments = {"profile": "qrs_infarct", "max_beats": 3} if not check.get("tool") else None
            steps.append(_step(step_id, question, tool, arguments=arguments))
    if len(steps) > 6:
        raise ValueError(f"pathway {code!r} exceeds the six-node contract; do not silently drop required evidence")
    for step in steps:
        step["owner"] = (
            "program"
            if program_owned
            and program_owns_pathway_step(code, str(step.get("id") or ""))
            else "model"
        )
    return {
        "version": DIAGNOSTIC_PATHWAY_VERSION,
        "code": code,
        "display_name": DIAGNOSIS_CATALOG.get(code, (code, "other"))[0],
        "steps": steps,
        "cautions": cautions,
        "evidence_policy": "Missing, null, unreliable, unobserved or inapplicable defining evidence is unknown, not a negative finding. Confirm only the stated ECG phenotype; etiologies and mechanisms need their own evidence.",
    }

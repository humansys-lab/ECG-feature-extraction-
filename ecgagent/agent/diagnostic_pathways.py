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


DIAGNOSTIC_PATHWAY_VERSION = "ecgagent.diagnostic-pathways.v8"


#: Gate semantics for a pathway node.
#:
#: ``required``     every node must ``pass`` before a candidate is confirmed;
#:                  any ``fail`` rejects it and any ``unknown`` leaves it unresolved.
#: ``supporting``   the status is recorded and its evidence is kept, but it never
#:                  gates placement.
#: ``invalidator``  an applicability precondition: only an explicit ``fail``
#:                  blocks confirmation.  ``pass``/``unknown`` never stall the
#:                  candidate, so a precondition that cannot be measured costs
#:                  no sensitivity.  Use it where a criterion is only *valid*
#:                  under conditions that are separate from the criterion
#:                  itself -- voltage criteria under pacing or a wide QRS, for
#:                  example.  ``supporting`` cannot express this: its status
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


def _definition(code: str) -> tuple[list[dict[str, Any]], list[str]]:
    if code in _SINUS:
        return (
            [
                _step(
                    "organized_atrial_activity",
                    "Is the rhythm background regular, with atrial measurements available?",
                    "get_rhythm_profile",
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
                        "fields": ["association_type", "confidence", "pr_ms", "source_leads", "associated_qrs_beat_id"],
                        "limit": 12,
                    },
                ),
            ],
            ["When PR does not meet the definition threshold, this step must fail; borderline wording must not upgrade it."],
        )
    if code in _ADVANCED_AV:
        return (
            [
                _step(
                    "repeated_blocked_atrial_events",
                    "Are there repeated, reliable nonconducted atrial events?",
                    "get_atrial_event_table",
                    arguments={
                        "fields": ["association_type", "confidence", "pr_ms", "source_leads", "associated_qrs_beat_id", "time_ms"],
                        "limit": 16,
                    },
                ),
                _step(
                    "sequential_av_pattern",
                    "Does sequential beat timing form a reproducible AV conduction pattern?",
                    "get_rhythm_profile",
                    arguments={"sections": ["av_association"]},
                ),
            ],
            ["A single blocked-detector event is insufficient to confirm second-degree or complete AV block."],
        )
    if code in _IVCD:
        return (
            [
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
            ["When only a wide QRS is established and specific morphologies have not been excluded, retain only an intraventricular conduction-delay candidate."],
        )
    if code in _BBB:
        return (
            [
                _step("qrs_duration_support", "Is dominant-beat QRS duration compatible with this conduction pattern?", "get_morphology_groups", arguments={"include_beats": False}),
                _step(
                    "required_lead_pattern",
                    "Does cross-lead QRS morphology meet the specified bundle or fascicular block pattern?",
                    "get_qrs_measurement_bundle",
                    arguments={"leads": ["I", "II", "III", "aVL", "aVF", "V1", "V2", "V5", "V6"]},
                ),
                _step("pattern_representative", "Does this morphology represent the principal beats rather than an isolated abnormal beat?", "get_morphology_groups", arguments={"include_beats": False}),
                _step(
                    "preexcitation_excluded",
                    "Do the rhythm and pre-excitation views exclude a pre-excitation pattern?",
                    "get_rhythm_profile",
                    gate="invalidator",
                    arguments={"sections": ["preexcitation", "atrial_signal"]},
                ),
            ],
            [
                "QRS widening alone cannot distinguish RBBB, LBBB or nonspecific intraventricular conduction delay.",
                "Pre-excitation also widens the QRS and alters its initial vector; bundle branch block morphology is not interpretable until pre-excitation is excluded.",
            ],
        )
    if code in _PREEXCITATION:
        return (
            [
                _step("preexcitation_components", "Are short PR and cross-lead delta candidates both present?", "get_rhythm_profile", arguments={"sections": ["preexcitation"]}),
                _step("pr_component", "Are the PR interval and constituent P-wave evidence reportable?", "get_interval_waveform_context", gate="supporting", arguments={"interval": "pr", "include_beat_flags": False}),
                _step("qrs_onset_support", "Is QRS-onset morphology compatible with the pre-excitation candidate?", "get_qrs_measurement_bundle", gate="supporting", arguments={"leads": ["I", "II", "V1", "V4", "V5", "V6"]}),
            ],
            ["A positive delta detector without independent short-PR evidence cannot confirm pre-excitation."],
        )
    if code == "short_pr_interval":
        return (
            [
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
        return (
            [
                _step(
                    "voltage_criteria_qrs_valid",
                    "Is dominant QRS duration narrow enough for voltage criteria to remain valid?",
                    "get_morphology_groups",
                    gate="invalidator",
                    arguments={"include_beats": False},
                ),
                _step(
                    "voltage_criteria_pacing_valid",
                    "Is ventricular pacing absent so that voltage criteria remain valid?",
                    "get_pacing_profile",
                    gate="invalidator",
                    arguments={"include_beats": False},
                ),
                _step("cross_lead_voltage_criterion", "Does the cross-lead voltage combination meet the stated hypertrophy criterion?", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv"], "leads": ["aVL", "V2", "V3", "V4"]}),
                _step("qrs_morphology_compatible", "Do QRS morphology and lead distribution support the pattern without invalidating the voltage criterion?", "get_qrs_measurement_bundle", gate="supporting", arguments={"leads": ["I", "aVL", "V1", "V2", "V5", "V6"]}),
            ],
            [
                "A rule hit is not patient evidence; the component voltages must be read directly.",
                "Voltage criteria do not establish anatomic ventricular hypertrophy.",
                "Ventricular pacing, bundle branch block or marked intraventricular conduction delay may increase voltage; voltage criteria are not valid in that context.",
            ],
        )
    if code in _RVH:
        return (
            [
                _step("right_precordial_voltage", "Does the voltage relationship between right and left precordial leads support a right-ventricular pattern?", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "qrs_ms"], "leads": ["I", "aVF", "V1", "V2", "V5", "V6"]}),
                _step("rvh_qrs_distribution", "Is cross-lead QRS morphology compatible with a right-ventricular pattern?", "get_qrs_measurement_bundle", arguments={"leads": ["I", "aVF", "V1", "V2", "V5", "V6"]}),
            ],
            ["A tall R wave in V1 alone cannot confirm right ventricular hypertrophy."],
        )
    if code in _ATRIAL_CHAMBER:
        return (
            [
                _step("p_wave_distribution", "Do P-wave duration, amplitude and terminal components form a consistent cross-lead pattern?", "get_lead_table", arguments={"fields": ["p_dur_ms", "p_amp_mv", "p_notched", "p_biphasic", "p_terminal_duration_ms", "p_terminal_amp_mv"], "leads": ["II", "III", "aVF", "V1", "V2"]}),
                _step("p_measurement_reliability", "Are multi-beat P-wave boundary assessments sufficient to support an atrial abnormality interpretation?", "get_p_assessment_table", arguments={"fields": ["accepted", "ta_ambiguous", "onset_confidence", "offset_confidence", "valid_leads", "reject_reasons"], "limit": 12}),
            ],
            ["When P-wave measurement is limited, a single-lead morphology cannot confirm an atrial abnormality."],
        )
    if code in _LOW_VOLTAGE:
        leads = ["I", "II", "III", "aVR", "aVL", "aVF"] if "limb" in code else ["V1", "V2", "V3", "V4", "V5", "V6"]
        return (
            [_step("territorial_qrs_voltage", "Do QRS voltages consistently meet the low-voltage definition across the target lead group?", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "q_amp_mv"], "leads": leads})],
            ["Assess limb and precordial lead groups separately; do not extrapolate from a single lead."],
        )
    if code in _AF:
        return (
            [
                _step("irregular_ventricular_response", "Does the background rhythm show an irregular ventricular response compatible with atrial fibrillation?", "get_rhythm_profile", arguments={"sections": ["background", "atrial_signal"]}),
                _step("organized_p_absent", "Do reliable P-wave assessments show no consistent organized P waves?", "get_p_assessment_table", arguments={"fields": ["accepted", "ta_ambiguous", "onset_confidence", "offset_confidence", "morphology_cluster_id", "valid_leads", "reject_reasons"], "limit": 12}),
                _step("multilead_atrial_support", "Is atrial activity supported across leads rather than by a single-lead candidate signal?", "get_rhythm_profile", gate="supporting", arguments={"sections": ["background", "atrial_signal"]}),
            ],
            ["A regular rate or a single fibrillatory-wave candidate cannot establish an atrial fibrillation mechanism."],
        )
    if code in _AFL:
        return (
            [
                _step("organized_flutter_activity", "Is there repeated, organized flutter-like atrial activity?", "get_rhythm_profile", arguments={"sections": ["background", "atrial_signal", "av_association"]}),
                _step("multilead_flutter_support", "Does flutter-like activity have consistent cross-lead support?", "get_atrial_event_table", arguments={"fields": ["association_type", "confidence", "template_similarity", "source_leads", "time_ms", "pr_ms", "associated_qrs_beat_id"], "limit": 16}),
                _step("av_relation_characterized", "Is the AV conduction relationship compatible with the atrial flutter candidate?", "get_atrial_event_table", arguments={"fields": ["association_type", "confidence", "template_similarity", "source_leads", "time_ms", "pr_ms", "associated_qrs_beat_id"], "limit": 16}),
            ],
            ["A fixed ventricular rate cannot confirm atrial flutter without cross-lead atrial morphology."],
        )
    if code in _OTHER_ATRIAL_RHYTHM:
        return (
            [
                _step("atrial_mechanism", "Do atrial activity, atrial rate and the AV relationship support this rhythm mechanism?", "get_rhythm_profile", arguments={"sections": ["background", "atrial_signal", "av_association"]}),
                _step("p_morphology_support", "Does reliable P-wave morphology support and distinguish this atrial origin?", "get_p_assessment_table", arguments={"fields": ["accepted", "morphology_cluster_id", "valid_leads", "onset_confidence", "offset_confidence", "reject_reasons"], "limit": 12}),
            ],
            ["Heart rate or an isolated atrial event cannot independently establish atrial origin."],
        )
    if code in _VENTRICULAR_RHYTHM:
        return (
            [
                _step("wide_complex_sequence", "Does sequential beat timing show a representative wide-QRS rhythm?", "get_morphology_groups", arguments={"include_beats": True}),
                _step("ventricular_morphology", "Do the dominant morphology group and its persistence support ventricular origin?", "get_morphology_groups", arguments={"include_beats": True}),
                _step("atrial_relation", "Does the AV relationship support rather than refute a ventricular mechanism?", "get_rhythm_profile", arguments={"sections": ["background", "av_association", "aberrancy"]}),
            ],
            ["Neither wide QRS nor tachycardia alone can confirm a ventricular rhythm."],
        )
    if code in _ECTOPY:
        return (
            [
                _step("premature_timing", "Does beat-to-beat timing show true prematurity?", "get_beat_table", arguments={"fields": ["rr_prev_ms", "rr_next_ms", "group_id", "paced"]}),
                _step("ectopic_morphology", "Does morphology grouping support an ectopic beat rather than artifact?", "get_morphology_groups", arguments={"include_beats": True}),
                _step("representative_event", "Does the candidate event have adequate quality and repeated support?", "get_morphology_groups", arguments={"include_beats": True}),
            ],
            ["A single nondominant morphology group does not automatically represent a premature complex."],
        )
    if code in _QT:
        return (
            [
                _step("interval_reportable", "Can QT or QTc be reported reliably?", "get_interval_waveform_context", arguments={"interval": "qt", "include_beat_flags": False}),
                _step("qt_threshold", "Does a reportable QTc meet the definition range for this long- or short-QT candidate?", "get_interval_waveform_context", arguments={"interval": "qt", "include_beat_flags": False}),
                _step("component_endpoint_support", "Does T/U-wave endpoint evidence support interval interpretation?", "get_interval_waveform_context", arguments={"interval": "qt", "include_beat_flags": False}),
            ],
            ["When the interval is not reportable, a missing value cannot confirm or exclude a QT abnormality."],
        )
    if code in _R_PROGRESSION:
        return (
            [
                _step("precordial_progression", "Do R/S progression and transition from V1 through V6 support this candidate?", "get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "q_amp_mv", "qrs_ms"], "leads": ["V1", "V2", "V3", "V4", "V5", "V6"]}),
                # An open-ended "no alternative explanation exists" cannot be
                # affirmatively proven from one view, so as a `required` node it
                # could only ever stall: measured 0 fail / 4 unknown across the
                # v38 run, blocking three otherwise-complete candidates. As an
                # invalidator it does the job it was written for -- a positively
                # identified infarct/conduction/lead-placement cause vetoes the
                # rotation read, while absence of evidence no longer stalls it.
                _step("alternative_qrs_cause_excluded", "Does the representative QRS view avoid showing an alternative cause such as infarction, conduction abnormality or lead placement?", "get_native_beat_profile", gate="invalidator", arguments={"profile": "qrs_infarct", "max_beats": 3}),
            ],
            ["Abnormal R-wave progression must be distinguished from lead placement, conduction abnormalities and Q-wave changes."],
        )
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
        return (
            [
                _step("t_wave_distribution", "Do reliable T-wave morphology and polarity changes show a candidate-compatible cross-lead distribution?", "get_morphology_map", arguments={"profile": "t_u"}),
                _step("repolarization_context", "Do representative beats support and distinguish primary from secondary repolarization change?", "get_native_beat_profile", arguments={"profile": "repolarization", "max_beats": 3}),
            ],
            ["A T-wave abnormality is a phenotype; do not infer etiology without its distribution and QRS context."],
        )
    if code in _Q_PATTERNS:
        return (
            [
                _step("q_wave_morphology", "Are Q-wave duration, amplitude and the Q/R relationship reliably present in the relevant leads?", "get_lead_table", arguments={"fields": ["q_duration_ms", "q_amp_mv", "r_amp_mv", "qrs_ms"]}),
                _step("q_wave_distribution", "Does representative-beat Q-wave distribution support this candidate and exclude an isolated variant?", "get_native_beat_profile", arguments={"profile": "qrs_infarct", "max_beats": 3}),
            ],
            ["A small Q wave in one lead or an automated detector flag cannot confirm pathological Q waves."],
        )
    if code in _TU_METABOLIC:
        return (
            [
                _step("tu_morphology_distribution", "Do T/U-wave or right-precordial morphologies show a reliable candidate-compatible distribution?", "get_morphology_map", arguments={"profile": "t_u"}),
                _step("tu_lead_confirmation", "Are key T/U measurements supported in multiple relevant leads?", "get_lead_table", arguments={"fields": ["t_amp_mv", "t_polarity", "t_symmetry", "t_sqi_score", "u_prominence_mv"]}),
            ],
            ["An ECG phenotype cannot substitute for serum potassium, imaging or a clinical etiologic diagnosis."],
        )
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
                    "Does beat-to-beat QRS amplitude in a representative precordial lead show a consistent alternating pattern across a run of consecutive beats?",
                    "get_beat_table",
                    arguments={"fields": ["qrs_area", "group_id", "paced"], "lead": "V5"},
                ),
                _step(
                    "limb_amplitude_alternation",
                    "Is the alternating pattern corroborated in a second, independent lead?",
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
                    "Was a dedicated search for notched or hidden atrial activity within the QRS/T complex performed and left inconclusive?",
                    "get_p_assessment_table",
                    gate="supporting",
                    arguments={"fields": ["accepted", "valid_leads", "reject_reasons"], "limit": 8},
                ),
            ],
            [
                "This candidate documents a rate pattern that specifically warrants excluding 2:1 atrial flutter; it is not itself a positive rhythm diagnosis.",
                "If organized flutter-rate atrial activity is independently demonstrated, use atrial_flutter_pattern instead of this exclusion candidate.",
            ],
        )
    if code == "possible_precordial_lead_reversal":
        return (
            [
                _step(
                    "precordial_progression_discontinuity",
                    "Does precordial R/S progression show an abrupt or non-monotonic discontinuity inconsistent with normal electrode placement?",
                    "get_lead_table",
                    arguments={"fields": ["r_amp_mv", "s_amp_mv", "q_amp_mv"], "leads": ["V1", "V2", "V3", "V4", "V5", "V6"]},
                ),
                _step(
                    "cross_lead_morphology_support",
                    "Does cross-lead QRS morphology corroborate a placement-related discontinuity rather than a genuine transition variant?",
                    "get_morphology_map",
                    gate="supporting",
                    arguments={"profile": "qrs", "leads": ["V1", "V2", "V3", "V4", "V5", "V6"]},
                ),
                _step(
                    "alternative_cause_excluded",
                    "Does the representative-beat QRS view avoid showing a genuine pathological cause (infarction, dextrocardia, true rotation) for the same pattern?",
                    "get_native_beat_profile",
                    gate="invalidator",
                    arguments={"profile": "qrs_infarct", "max_beats": 3},
                ),
            ],
            [
                "An abrupt precordial discontinuity can also reflect true anatomic rotation, dextrocardia or infarction; do not confirm lead reversal without excluding those causes.",
                "Upstream acquisition-metadata flags are not patient evidence; the discontinuity must be read directly from the measured amplitudes.",
            ],
        )
    if code == "limb_lead_reversal_suspected":
        return (
            [
                _step(
                    "limb_lead_deflection_criterion",
                    "Does lead I show a negative-dominant QRS deflection and aVR a positive-dominant QRS deflection, the defining pattern of an LA/RA limb-lead swap?",
                    "get_lead_table",
                    arguments={"fields": ["r_amp_mv", "s_amp_mv"], "leads": ["I", "aVR"]},
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
                    "Does the overall limb-lead pattern avoid being explained by dextrocardia or a genuine extreme right-axis rhythm rather than electrode placement?",
                    "get_morphology_map",
                    gate="invalidator",
                    arguments={"profile": "qrs", "leads": ["I", "II", "III", "aVR", "aVL", "aVF"]},
                ),
            ],
            [
                "The defining criterion is the main-deflection direction in lead I and aVR, not any single amplitude read in isolation.",
                "Dextrocardia and a genuine extreme right-axis rhythm can mimic this pattern; do not confirm a lead swap without excluding them.",
            ],
        )

    category = DIAGNOSIS_CATALOG.get(code, ("", "other"))[1]
    if category == "pacing":
        return (
            [
                _step("pacing_marker_support", "Are reliable pacing markers present?", "get_pacing_profile", arguments={"include_beats": True}),
                _step("capture_relation_support", "Is there a stable capture relationship between pacing markers and subsequent beats?", "get_pacing_profile", arguments={"include_beats": True}),
            ],
            ["A candidate pacing spike or wide QRS alone cannot confirm a paced rhythm."],
        )
    return ([], ["Use only measurements directly relevant to this candidate; never confirm it from rule status."])


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
        tool = str(first.get("tool") or "get_native_beat_profile")
        second_tool = str(second.get("tool") or tool)
        steps = [
            _step("direct_measurement_support", "Is there a reliable measurement that directly matches this candidate's definition?", tool),
            _step("discriminative_countercheck", "Does a targeted counterevidence view avoid refuting this candidate?", second_tool),
        ]
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
        "steps": steps[:6],
        "cautions": cautions[:3],
    }

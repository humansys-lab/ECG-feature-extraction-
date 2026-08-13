from __future__ import annotations

from .config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import RuleEvaluation
from .rhythm import disorganized_atrial_activity, has_borderline_af_evidence
from .sources import SOURCE_AHA_RHYTHM


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _result(
    *,
    rule_id: str,
    code: str,
    status: str,
    evidence: dict,
    statement: str | None = None,
    missing: list[str] | None = None,
    suppressed_by: list[str] | None = None,
    confidence: str | None = None,
    coverage: str | None = None,
    priority: str | None = None,
    human_review_required: bool = False,
) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=rule_id,
        domain="advanced_av_block",
        status=status,
        statement_code=code if status in {"matched", "suppressed"} else None,
        statement=statement if status in {"matched", "suppressed"} else None,
        severity="abnormal" if status in {"matched", "suppressed"} else "normal",
        confidence=confidence,
        coverage=coverage,
        required_inputs=[
            "rhythm.atrial_events_per_rr",
            "rhythm.pr_series_ms",
            "rhythm.atrial_and_ventricular_rates",
        ],
        missing_inputs=list(missing or []),
        evidence={**evidence, "evaluates_code": code},
        suppressed_by=list(suppressed_by or []),
        source=dict(SOURCE_AHA_RHYTHM),
        normality_required=False,
        normality_role="optional_screen",
        priority=priority,
        human_review_required=human_review_required,
    )


def evaluate_advanced_av_block(context) -> list[RuleEvaluation]:
    rhythm = _mapping(getattr(context.features, "metadata", {}).get("rhythm_analysis"))
    af_afl = _mapping(rhythm.get("af_afl_summary"))
    pacing = _mapping(rhythm.get("pacing_context"))
    summary = _mapping(rhythm.get("rule_summary"))
    pauses = _mapping(summary.get("pauses"))
    interpretation = getattr(context.features, "interpretation", None)
    premature_complexes = list(
        getattr(interpretation, "premature_complexes", None) or []
    )
    ectopy_confounders = []
    if premature_complexes:
        ectopy_confounders.append("premature_complexes")
    if getattr(interpretation, "bigeminy", None):
        ectopy_confounders.append("bigeminy")
    if bool(getattr(interpretation, "trigeminy", False)):
        ectopy_confounders.append("trigeminy")
    if bool(getattr(interpretation, "non_sustained_vt", False)):
        ectopy_confounders.append("ventricular_run")

    invalid_by = None
    global_value = getattr(context, "global_value", None)
    heart_rate_bpm = (
        global_value("heart_rate_bpm") if callable(global_value) else None
    )
    if af_afl.get("probable_af") or has_borderline_af_evidence(af_afl):
        invalid_by = "atrial_fibrillation_pattern"
    elif af_afl.get("probable_flutter"):
        invalid_by = "atrial_flutter_pattern"
    elif af_afl.get("af_afl_indeterminate"):
        invalid_by = "af_afl_indeterminate"
    elif (
        disorganized_atrial_activity(
            af_afl,
            threshold=(
                DEFAULT_DIAGNOSTIC_CONFIG.rhythm
                .organized_p_ratio_min_for_morphology
            ),
        )
        is not None
    ):
        # Counting dropped P waves presupposes the detected atrial events are
        # P waves. Without organized atrial activity they may be F waves or
        # T-wave residuals, and the excess-event test degenerates.
        invalid_by = "disorganized_atrial_activity"
    elif heart_rate_bpm is not None and float(heart_rate_bpm) > 100.0:
        invalid_by = "heart_rate_gt_100_bpm"
    elif pacing.get("continuous_pacing") or pacing.get("suppress_further_rhythm_interpretation"):
        invalid_by = "continuous_ventricular_pacing"

    if invalid_by:
        evidence = {"not_applicable_by": invalid_by, "manual_confirmation_required": False}
        return [
            _result(
                rule_id="CLIN-RHYTHM-AVB2-01",
                code="second_degree_av_block_pattern",
                status="not_applicable",
                evidence=evidence,
            ),
            _result(
                rule_id="CLIN-RHYTHM-AVB3-01",
                code="complete_av_block_pattern",
                status="not_applicable",
                evidence=evidence,
            ),
        ]

    if not summary or not pauses:
        missing = ["rhythm.rule_summary.pauses_and_av_block"]
        return [
            _result(
                rule_id="CLIN-RHYTHM-AVB2-01",
                code="second_degree_av_block_pattern",
                status="unavailable",
                evidence={},
                missing=missing,
            ),
            _result(
                rule_id="CLIN-RHYTHM-AVB3-01",
                code="complete_av_block_pattern",
                status="unavailable",
                evidence={},
                missing=missing,
            ),
        ]

    av_evidence = _mapping(pauses.get("av_block_evidence"))
    dropped = bool(av_evidence.get("dropped_p_evidence"))
    second_type = pauses.get("second_degree_avb")
    second_matched = bool(second_type and dropped)
    second_indeterminate = bool(dropped and not second_type)
    complete_matched = bool(
        summary.get("complete_av_block")
        and (
            summary.get("av_dissociation")
            or summary.get("atrial_faster_than_ventricular")
        )
    )
    atrial_counts = [
        int(value)
        for value in (av_evidence.get("atrial_events_per_rr") or [])
        if isinstance(value, (int, float))
    ]
    atrial_excess_count = sum(value > 1 for value in atrial_counts)
    atrial_excess_fraction = (
        float(atrial_excess_count) / float(len(atrial_counts))
        if atrial_counts else 0.0
    )
    pr_values = [
        float(value)
        for value in (av_evidence.get("pr_series_ms") or [])
        if isinstance(value, (int, float))
    ]
    persistent_av_dissociation_candidate = bool(
        not complete_matched
        and bool(summary.get("av_dissociation"))
        and not bool(summary.get("atrial_faster_than_ventricular"))
        and heart_rate_bpm is not None
        and float(heart_rate_bpm) <= 70.0
        and len(atrial_counts) >= 6
        and atrial_excess_fraction >= 0.70
        and bool(av_evidence.get("localized_atrial_event_excess"))
        and len(pr_values) >= 5
    )
    common = {
        "pauses": pauses,
        "complete_av_block": bool(summary.get("complete_av_block")),
        "av_dissociation": bool(summary.get("av_dissociation")),
        "atrial_faster_than_ventricular": bool(summary.get("atrial_faster_than_ventricular")),
        "heart_rate_bpm": heart_rate_bpm,
        "atrial_event_excess_count": atrial_excess_count,
        "atrial_event_interval_count": len(atrial_counts),
        "atrial_event_excess_fraction": atrial_excess_fraction,
        "pr_measurement_count": len(pr_values),
        "pr_range_ms": (
            max(pr_values) - min(pr_values) if pr_values else None
        ),
        "persistent_av_dissociation_candidate": (
            persistent_av_dissociation_candidate
        ),
        "ectopy_confounders": ectopy_confounders,
        "manual_confirmation_required": bool(
            second_matched
            or complete_matched
            or persistent_av_dissociation_candidate
        ),
    }

    if complete_matched:
        second_status = "suppressed" if second_matched else "not_matched"
        second_suppressor = ["complete_av_block_pattern"] if second_matched else []
    elif second_matched and ectopy_confounders:
        second_status = "suppressed"
        second_suppressor = ectopy_confounders
    else:
        second_status = (
            "matched" if second_matched
            else "indeterminate" if second_indeterminate
            else "not_matched"
        )
        second_suppressor = []
    type_label = {
        "mobitz_i": "Mobitz I second-degree AV block",
        "mobitz_ii": "Mobitz II second-degree AV block",
    }.get(str(second_type), "Second-degree AV block")
    # Mobitz II is the infranodal, pacemaker-indication pattern and carries a
    # far worse prognosis than Mobitz I, so it must not inherit the generic
    # second-degree priority.
    mobitz_ii = str(second_type) == "mobitz_ii"
    block_level_hint = pauses.get("av_block_level_hint")
    if mobitz_ii:
        statement_text = (
            f"{type_label} ECG pattern"
            f"{' with wide QRS (infranodal)' if block_level_hint == 'infranodal_suspected' else ''}"
            "; urgent manual confirmation required"
        )
    else:
        statement_text = f"{type_label} ECG pattern; manual confirmation required"

    return [
        _result(
            rule_id="CLIN-RHYTHM-AVB2-01",
            code="second_degree_av_block_pattern",
            status=second_status,
            evidence={
                **common,
                "second_degree_type": second_type,
                "second_degree_basis": pauses.get("second_degree_avb_basis") or {},
                "av_block_level_hint": block_level_hint,
                "two_to_one_conduction_suspected": bool(
                    pauses.get("two_to_one_conduction_suspected")
                ),
            },
            statement=statement_text,
            missing=(
                ["stable_p_qrs_association_for_av_block_classification"]
                if second_indeterminate else []
            ),
            suppressed_by=second_suppressor,
            confidence=("moderate" if mobitz_ii else "low") if second_matched else None,
            coverage=(
                "partial"
                if second_indeterminate or second_status == "suppressed"
                else None
            ),
            priority=("P1" if mobitz_ii else "P2") if second_matched else None,
            human_review_required=bool(second_matched),
        ),
        _result(
            rule_id="CLIN-RHYTHM-AVB3-01",
            code="complete_av_block_pattern",
            status=(
                "matched"
                if complete_matched
                else "indeterminate"
                if persistent_av_dissociation_candidate
                else "not_matched"
            ),
            evidence=common,
            statement="Complete AV block ECG pattern; urgent manual confirmation required",
            missing=(
                [
                    "validated_atrial_rate_faster_than_ventricular",
                    "multilead_non_qrst_atrial_event_validation",
                ]
                if persistent_av_dissociation_candidate else []
            ),
            confidence=(
                "moderate"
                if complete_matched
                else "low"
                if persistent_av_dissociation_candidate
                else None
            ),
            coverage="partial" if persistent_av_dissociation_candidate else None,
            priority="P1" if complete_matched else "P2",
            human_review_required=bool(
                complete_matched or persistent_av_dissociation_candidate
            ),
        ),
    ]

from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Iterable, Optional

from .config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import RuleEvaluation
from .sources import SOURCE_AHA_CONDUCTION_2009


_MORPHOLOGY = DEFAULT_DIAGNOSTIC_CONFIG.morphology
_INTERVALS = DEFAULT_DIAGNOSTIC_CONFIG.intervals

_STANDARD_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def _qrs_endpoint_profile(
    context,
    qrs_ms: Optional[float],
) -> tuple[Optional[float], Optional[float], bool]:
    """Return the conservative QRS value used for conduction screening.

    ``qrs_wide_ms`` is the existing P90 terminal-offset consensus produced by
    the delineator specifically for wide-QRS classification.  It is useful
    when the central global aggregation clips a late terminal deflection, but
    disagreement with the global value must lower confidence rather than
    silently replacing the authoritative interval measurement.
    """

    wide_values = [
        float(value)
        for lead in _STANDARD_LEADS
        if (value := context.lead_value(lead, "qrs_wide_ms")) is not None
        and 40.0 <= float(value) <= 220.0
    ]
    wide_qrs_ms = float(median(wide_values)) if wide_values else None
    candidates = [
        float(value)
        for value in (qrs_ms, wide_qrs_ms)
        if value is not None
    ]
    classification_qrs_ms = max(candidates) if candidates else None
    endpoint_discordant = bool(
        qrs_ms is not None
        and wide_qrs_ms is not None
        and wide_qrs_ms > qrs_ms
    )
    return classification_qrs_ms, wide_qrs_ms, endpoint_discordant


def _evaluation(
    *,
    rule_id: str,
    code: str,
    status: str,
    evidence: dict,
    missing_inputs: Optional[Iterable[str]] = None,
    statement: Optional[str] = None,
    statement_code: Optional[str] = None,
    confidence: Optional[str] = None,
) -> RuleEvaluation:
    resolved_code = statement_code if statement_code is not None else code
    return RuleEvaluation(
        rule_id=rule_id,
        domain="conduction",
        status=status,
        statement_code=resolved_code if status == "matched" else None,
        statement=statement if status == "matched" else None,
        severity="abnormal" if status == "matched" else "normal",
        confidence=confidence if status == "matched" else None,
        required_inputs=list(evidence.get("required_inputs", [])),
        missing_inputs=list(missing_inputs or []),
        evidence={**evidence, "evaluates_code": code},
        source=dict(SOURCE_AHA_CONDUCTION_2009),
    )


def _suppressed_by(
    evaluation: RuleEvaluation,
    *suppressors: str,
) -> RuleEvaluation:
    return replace(
        evaluation,
        status="suppressed",
        coverage="partial",
        suppressed_by=sorted({str(value) for value in suppressors if value}),
    )


def _bbb_evidence_rank(evaluation: RuleEvaluation) -> int:
    if evaluation.confidence == "criteria_met":
        return 2
    if evaluation.status == "matched":
        return 1
    return 0


def _rbbb(context, qrs_ms: Optional[float]) -> RuleEvaluation:
    code = "rbbb_pattern"
    classification_qrs_ms, wide_qrs_ms, endpoint_discordant = (
        _qrs_endpoint_profile(context, qrs_ms)
    )
    qrs_evidence = {
        "qrs_ms": qrs_ms,
        "qrs_wide_endpoint_ms": wide_qrs_ms,
        "qrs_for_conduction_screen_ms": classification_qrs_ms,
        "qrs_endpoint_discordant": endpoint_discordant,
    }
    if classification_qrs_ms is None:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-RBBB-01",
            code=code,
            status="unavailable",
            evidence=qrs_evidence,
            missing_inputs=["global.qrs_ms"],
        )
    if classification_qrs_ms < 110.0:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-RBBB-01",
            code=code,
            status="not_matched",
            evidence=qrs_evidence,
        )
    r_prime = context.lead_value("V1", "r_prime_amp_mv")
    r_prime_duration = context.lead_value("V1", "r_prime_duration_ms")
    lateral = []
    for lead in ("I", "V6"):
        lateral.append(
            (
                lead,
                context.lead_value(lead, "s_amp_mv"),
                context.lead_value(lead, "s_duration_ms"),
                context.lead_value(lead, "r_duration_ms"),
            )
        )
    lateral_amp = [
        (lead, amp)
        for lead, amp, _, _ in lateral
        if amp is not None
    ]
    evidence = {
        **qrs_evidence,
        "v1_r_prime_mv": r_prime,
        "v1_r_prime_duration_ms": r_prime_duration,
        "lateral_s": lateral,
    }
    # Amplitude/morphology alone can qualify a lower-confidence "probable"
    # statement when the R'/S duration components could not be measured;
    # the full-criteria `matched` tier still requires duration evidence.
    missing = []
    if r_prime is None:
        missing.append("lead.V1.r_prime_amp_mv")
    if not lateral_amp:
        missing.append("lead.I_or_V6.s_amplitude")
    if missing:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-RBBB-01",
            code=code,
            status="unavailable",
            evidence=evidence,
            missing_inputs=missing,
        )
    amplitude_morphology = bool(
        r_prime > 0.1 and any(amp < -0.1 for _, amp in lateral_amp)
    )
    if not amplitude_morphology:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-RBBB-01",
            code=code,
            status="not_matched",
            evidence=evidence,
        )
    lateral_terminal_delay = any(
        amp is not None
        and amp < -0.1
        and s_duration is not None
        and (
            s_duration >= 40.0
            or (r_duration is not None and s_duration > r_duration)
        )
        for _, amp, s_duration, r_duration in lateral
    )
    v1_peak_delay_with_normal_lateral = bool(
        r_prime_duration is not None
        and r_prime_duration > 50.0
        and any(
            lead == "V6"
            and r_duration is not None
            and r_duration <= 50.0
            for lead, _, _, r_duration in lateral
        )
    )
    duration_supported = bool(
        lateral_terminal_delay or v1_peak_delay_with_normal_lateral
    )
    # When only the P90 terminal endpoint crosses 110 ms while the central
    # global interval remains below it, require the V1-specific terminal
    # delay as well.  A long lateral S alone is not specific enough to turn a
    # discordant narrow global QRS into an RBBB statement.
    if (
        endpoint_discordant
        and qrs_ms is not None
        and qrs_ms < 110.0
        and not v1_peak_delay_with_normal_lateral
    ):
        return _evaluation(
            rule_id="CLIN-CONDUCTION-RBBB-01",
            code=code,
            status="not_matched",
            evidence={
                **evidence,
                "duration_criteria_available": duration_supported,
                "lateral_terminal_delay": lateral_terminal_delay,
                "v1_peak_delay_with_normal_lateral": False,
                "wide_endpoint_rescue_rejected": (
                    "v1_specific_terminal_delay_not_met"
                ),
            },
        )
    if qrs_ms is not None and qrs_ms >= 120.0 and duration_supported:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-RBBB-01",
            code=code,
            status="matched",
            evidence=evidence,
            statement="ECG pattern consistent with right bundle branch block",
            confidence="criteria_met",
        )
    return _evaluation(
        rule_id="CLIN-CONDUCTION-RBBB-01",
        code=code,
        status="matched",
        evidence={
            **evidence,
            "duration_criteria_available": duration_supported,
            "lateral_terminal_delay": lateral_terminal_delay,
            "v1_peak_delay_with_normal_lateral": (
                v1_peak_delay_with_normal_lateral
            ),
            "qrs_complete_criterion": bool(
                qrs_ms is not None and qrs_ms >= 120.0
            ),
        },
        statement=(
            "ECG pattern probably consistent with incomplete or right bundle "
            "branch block (QRS duration/endpoint criteria are not fully "
            "concordant); confirm manually"
        ),
        statement_code="probable_rbbb_pattern",
        confidence=(
            "probable_wide_endpoint_morphology"
            if endpoint_discordant
            else "probable_amplitude_only"
        ),
    )


def _lbbb(context, qrs_ms: Optional[float]) -> RuleEvaluation:
    code = "lbbb_pattern"
    classification_qrs_ms, wide_qrs_ms, endpoint_discordant = (
        _qrs_endpoint_profile(context, qrs_ms)
    )
    qrs_evidence = {
        "qrs_ms": qrs_ms,
        "qrs_wide_endpoint_ms": wide_qrs_ms,
        "qrs_for_conduction_screen_ms": classification_qrs_ms,
        "qrs_endpoint_discordant": endpoint_discordant,
    }
    if classification_qrs_ms is None:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LBBB-01",
            code=code,
            status="unavailable",
            evidence=qrs_evidence,
            missing_inputs=["global.qrs_ms"],
        )
    if classification_qrs_ms < 110.0:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LBBB-01",
            code=code,
            status="not_matched",
            evidence=qrs_evidence,
        )
    v1_r = context.lead_value("V1", "r_amp_mv")
    lateral = []
    for lead in ("I", "aVL", "V5", "V6"):
        lateral.append(
            (
                lead,
                context.lead_value(lead, "q_amp_mv"),
                context.lead_value(lead, "r_amp_mv"),
                context.lead_value(lead, "r_duration_ms"),
            )
        )
    lateral_amp = [
        (lead, q_amp, r_amp)
        for lead, q_amp, r_amp, _ in lateral
        if q_amp is not None and r_amp is not None
    ]
    missing = []
    if v1_r is None:
        missing.append("lead.V1.r_amp_mv")
    if not lateral_amp:
        missing.append("lateral.q_and_r_amplitude")
    evidence = {
        **qrs_evidence,
        "v1_r_mv": v1_r,
        "lateral_q_r": lateral,
    }
    if missing:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LBBB-01",
            code=code,
            status="unavailable",
            evidence=evidence,
            missing_inputs=missing,
        )
    # Amplitude/morphology alone can qualify a lower-confidence "probable"
    # statement when lateral R-wave duration could not be measured; the
    # full-criteria `matched` tier still requires the duration evidence.
    amplitude_morphology = bool(
        v1_r <= 0.1
        and any(abs(q_amp) < 0.04 and r_amp > 0.3 for _, q_amp, r_amp in lateral_amp)
    )
    if not amplitude_morphology:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LBBB-01",
            code=code,
            status="not_matched",
            evidence=evidence,
        )
    complete_lateral = [
        row for row in lateral if row[1] is not None and row[2] is not None and row[3] is not None
    ]
    duration_supported = bool(
        any(
            abs(q_amp) < 0.04 and r_amp > 0.3 and r_duration >= 60.0
            for _, q_amp, r_amp, r_duration in complete_lateral
        )
    )
    if (
        qrs_ms is not None
        and qrs_ms < 120.0
        and not duration_supported
    ):
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LBBB-01",
            code=code,
            status="not_matched",
            evidence={
                **evidence,
                "duration_criteria_available": False,
                "incomplete_lbbb_rejected": (
                    "lateral_r_peak_time_not_supported"
                ),
            },
        )
    if (
        qrs_ms is not None
        and qrs_ms >= 120.0
        and duration_supported
    ):
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LBBB-01",
            code=code,
            status="matched",
            evidence=evidence,
            statement="ECG pattern consistent with left bundle branch block",
            confidence="criteria_met",
        )
    return _evaluation(
        rule_id="CLIN-CONDUCTION-LBBB-01",
        code=code,
        status="matched",
        evidence={
            **evidence,
            "duration_criteria_available": duration_supported,
            "qrs_complete_criterion": bool(
                qrs_ms is not None and qrs_ms >= 120.0
            ),
        },
        statement=(
            "ECG pattern probably consistent with incomplete or left bundle "
            "branch block (QRS duration/endpoint criteria are not fully "
            "concordant); confirm manually"
        ),
        statement_code="probable_lbbb_pattern",
        confidence=(
            "probable_wide_endpoint_morphology"
            if endpoint_discordant
            else "probable_amplitude_only"
        ),
    )


def _lafb(context, axis: Optional[float]) -> RuleEvaluation:
    code = "lafb_pattern"
    if axis is None:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LAFB-01",
            code=code,
            status="unavailable",
            evidence={},
            missing_inputs=["global.qrs_axis_deg"],
        )
    definite_axis = (
        _MORPHOLOGY.lafb_axis_min_deg <= axis <= _MORPHOLOGY.lafb_axis_max_deg
    )
    probable_axis = (
        not definite_axis
        and _MORPHOLOGY.lafb_axis_min_deg
        <= axis
        <= _MORPHOLOGY.lafb_axis_probable_max_deg
    )
    if not (definite_axis or probable_axis):
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LAFB-01",
            code=code,
            status="not_matched",
            evidence={"qrs_axis_deg": axis},
        )
    needed = {
        (lead, name): context.lead_value(lead, name)
        for lead, names in {
            "I": ("r_amp_mv",),
            "aVL": ("q_amp_mv", "r_amp_mv"),
            "II": ("r_amp_mv", "s_amp_mv"),
            "III": ("r_amp_mv", "s_amp_mv"),
            "aVF": ("r_amp_mv", "s_amp_mv"),
        }.items()
        for name in names
    }
    qrs_ms = context.global_value("qrs_ms")
    missing = [f"lead.{lead}.{name}" for (lead, name), value in needed.items() if value is None]
    evidence = {
        "qrs_axis_deg": axis,
        "qrs_ms": qrs_ms,
        "measurements": {f"{lead}.{name}": value for (lead, name), value in needed.items()},
    }
    if missing:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LAFB-01",
            code=code,
            status="unavailable",
            evidence=evidence,
            missing_inputs=missing,
        )
    qr_avl = needed[("aVL", "q_amp_mv")] < 0 < needed[("aVL", "r_amp_mv")]
    rs_inferior = all(
        needed[(lead, "r_amp_mv")] > 0
        and abs(needed[(lead, "s_amp_mv")]) > needed[(lead, "r_amp_mv")]
        for lead in ("II", "III", "aVF")
    )
    # Vector-consistency checks. LAFB's delayed activation points up and to the
    # left at roughly -60 deg, which projects more negatively on III (+120 deg)
    # than on II (+60 deg), and more positively on aVL (-30 deg) than on I
    # (0 deg). Left axis deviation from LVH or a horizontal heart points nearer
    # -15 to -30 deg and fails these, which is what keeps the rule specific
    # once the axis window is not doing that job alone.
    s_iii_deeper_than_s_ii = abs(needed[("III", "s_amp_mv")]) > abs(
        needed[("II", "s_amp_mv")]
    )
    r_avl_taller_than_r_i = needed[("aVL", "r_amp_mv")] > needed[("I", "r_amp_mv")]
    # The textbook "QRS < 0.12 s" applies to *isolated* fascicular block. It is
    # deliberately not a match criterion here: in bifascicular block the width
    # comes from the accompanying RBBB, so gating on it would make
    # RBBB + LAFB unreachable. A wide QRS from LBBB is already handled by the
    # suppression in evaluate_conduction.
    criteria = {
        "qr_in_avl": qr_avl,
        "rs_in_inferior_leads": rs_inferior,
        "s_iii_deeper_than_s_ii": s_iii_deeper_than_s_ii,
        "r_avl_taller_than_r_i": r_avl_taller_than_r_i,
    }
    evidence["qrs_within_isolated_fascicular_limit"] = (
        qrs_ms is None or qrs_ms < _INTERVALS.qrs_wide_ms
    )
    evidence["criteria"] = criteria
    evidence["axis_tier"] = "definite" if definite_axis else "borderline"
    # VAT(aVL) >= 45 ms is the one textbook criterion still unimplemented;
    # vat_ms is not exported. See docs/诊断特征缺口登记.md section 2.3.
    evidence["unimplemented_criteria"] = ["avl_vat_ms_ge_45"]
    if not all(criteria.values()):
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LAFB-01",
            code=code,
            status="not_matched",
            evidence=evidence,
        )
    if definite_axis:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LAFB-01",
            code=code,
            status="matched",
            evidence=evidence,
            statement="ECG pattern consistent with left anterior fascicular block",
            confidence="criteria_met",
        )
    return _evaluation(
        rule_id="CLIN-CONDUCTION-LAFB-01",
        code=code,
        status="matched",
        evidence=evidence,
        statement=(
            "ECG pattern probably consistent with left anterior fascicular "
            "block (axis between -30 and -45 degrees); exclude left "
            "ventricular hypertrophy, horizontal heart and prior inferior "
            "infarction"
        ),
        statement_code="probable_lafb_pattern",
        confidence="probable_borderline_axis",
    )


def _lpfb(context, axis: Optional[float]) -> RuleEvaluation:
    code = "lpfb_pattern"
    if axis is None:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LPFB-01",
            code=code,
            status="unavailable",
            evidence={},
            missing_inputs=["global.qrs_axis_deg"],
        )
    # LPFB is a diagnosis of exclusion; the previous +90 deg floor fired on
    # ordinary vertical hearts. AHA/ACCF/HRS Part III uses +110 deg.
    if not (
        _MORPHOLOGY.lpfb_axis_min_deg <= axis <= _MORPHOLOGY.lpfb_axis_max_deg
    ):
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LPFB-01",
            code=code,
            status="not_matched",
            evidence={
                "qrs_axis_deg": axis,
                "axis_window_deg": [
                    _MORPHOLOGY.lpfb_axis_min_deg,
                    _MORPHOLOGY.lpfb_axis_max_deg,
                ],
            },
        )
    needed = {
        (lead, name): context.lead_value(lead, name)
        for lead, names in {
            "I": ("r_amp_mv", "s_amp_mv"),
            "aVL": ("r_amp_mv", "s_amp_mv"),
            "II": ("r_amp_mv",),
            "III": ("q_amp_mv", "r_amp_mv"),
            "aVF": ("q_amp_mv", "r_amp_mv"),
        }.items()
        for name in names
    }
    qrs_ms = context.global_value("qrs_ms")
    missing = [f"lead.{lead}.{name}" for (lead, name), value in needed.items() if value is None]
    evidence = {
        "qrs_axis_deg": axis,
        "qrs_ms": qrs_ms,
        "measurements": {f"{lead}.{name}": value for (lead, name), value in needed.items()},
        "confounders_require_review": ["rvh", "lateral_infarction", "vertical_heart"],
    }
    if missing:
        return _evaluation(
            rule_id="CLIN-CONDUCTION-LPFB-01",
            code=code,
            status="unavailable",
            evidence=evidence,
            missing_inputs=missing,
        )
    rs_lateral = all(
        needed[(lead, "r_amp_mv")] > 0
        and abs(needed[(lead, "s_amp_mv")]) > needed[(lead, "r_amp_mv")]
        for lead in ("I", "aVL")
    )
    qr_inferior = all(
        needed[(lead, "q_amp_mv")] < 0 < needed[(lead, "r_amp_mv")]
        for lead in ("III", "aVF")
    )
    # Mirror of the LAFB check: LPFB activation points down and to the right,
    # projecting more strongly on III (+120 deg) than on II (+60 deg).
    r_iii_taller_than_r_ii = needed[("III", "r_amp_mv")] > needed[("II", "r_amp_mv")]
    criteria = {
        "rs_in_i_and_avl": rs_lateral,
        "qr_in_inferior_leads": qr_inferior,
        "r_iii_taller_than_r_ii": r_iii_taller_than_r_ii,
    }
    # See _lafb: not a match criterion, because RBBB + LPFB is wide by
    # definition.
    evidence["qrs_within_isolated_fascicular_limit"] = (
        qrs_ms is None or qrs_ms < _INTERVALS.qrs_wide_ms
    )
    evidence["criteria"] = criteria
    return _evaluation(
        rule_id="CLIN-CONDUCTION-LPFB-01",
        code=code,
        status="matched" if all(criteria.values()) else "not_matched",
        evidence=evidence,
        statement="ECG pattern consistent with left posterior fascicular block; exclude competing causes of right axis deviation",
    )


def _fascicular_combinations(
    context,
    rbbb: RuleEvaluation,
    lafb: RuleEvaluation,
    lpfb: RuleEvaluation,
) -> list[RuleEvaluation]:
    """Combine a bundle-branch block with a fascicular block.

    RBBB plus either left fascicular block leaves conduction depending on the
    single remaining fascicle, which is why bifascicular block with syncope is
    a pacemaker evaluation. The individual rules already matched here; nothing
    downstream was joining them, so the combined risk never appeared in a
    report.

    Adding first-degree AV delay on top is the classical "incomplete
    trifascicular" description. It is deliberately reported as a pattern, not
    a diagnosis: the PR prolongation may arise in the AV node rather than in
    the surviving fascicle, and only an electrophysiology study separates the
    two.
    """
    rbbb_matched = rbbb.status == "matched"
    fascicular = [
        (item, name)
        for item, name in ((lafb, "lafb_pattern"), (lpfb, "lpfb_pattern"))
        if item.status == "matched"
    ]
    inputs_pending = [
        item.rule_id
        for item in (rbbb, lafb, lpfb)
        if item.status == "unavailable"
    ]
    pr_ms = context.global_value("pr_ms")

    if not (rbbb_matched and fascicular):
        status = "unavailable" if inputs_pending else "not_matched"
        evidence = {
            "rbbb_status": rbbb.status,
            "lafb_status": lafb.status,
            "lpfb_status": lpfb.status,
        }
        return [
            _evaluation(
                rule_id="CLIN-CONDUCTION-BIFASCICULAR-01",
                code="bifascicular_block_pattern",
                status=status,
                evidence=evidence,
                missing_inputs=inputs_pending,
            ),
            _evaluation(
                rule_id="CLIN-CONDUCTION-TRIFASCICULAR-01",
                code="trifascicular_block_pattern",
                status=status,
                evidence={**evidence, "pr_ms": pr_ms},
                missing_inputs=inputs_pending,
            ),
        ]

    fascicle_name = fascicular[0][1]
    fascicle_label = (
        "left anterior fascicular block"
        if fascicle_name == "lafb_pattern"
        else "left posterior fascicular block"
    )
    # RBBB + LPFB spares only the left anterior fascicle, the thinnest of the
    # three, and carries the worse prognosis of the two combinations.
    higher_risk = fascicle_name == "lpfb_pattern"
    evidence = {
        "components": ["rbbb_pattern", fascicle_name],
        "rbbb_confidence": rbbb.confidence,
        "fascicular_confidence": fascicular[0][0].confidence,
        "pr_ms": pr_ms,
    }
    # A bifascicular statement can be no stronger than its weaker component.
    component_probable = any(
        str(item.statement_code or "").startswith("probable_")
        for item in (rbbb, fascicular[0][0])
    )
    bifascicular = _evaluation(
        rule_id="CLIN-CONDUCTION-BIFASCICULAR-01",
        code="bifascicular_block_pattern",
        status="matched",
        evidence={**evidence, "component_confidence_reduced": component_probable},
        statement=(
            (
                "Probable bifascicular block: right bundle branch block with "
                f"{fascicle_label} (at least one component met only "
                "lower-confidence criteria); confirm manually"
                if component_probable
                else f"Bifascicular block: right bundle branch block with {fascicle_label}"
            )
            + ("; conduction depends on the left anterior fascicle" if higher_risk else "")
        ),
        statement_code=(
            "probable_bifascicular_block_pattern" if component_probable else None
        ),
        confidence="probable_component" if component_probable else "criteria_met",
    )
    bifascicular.severity = "abnormal"
    bifascicular.priority = "P1" if higher_risk else "P2"
    bifascicular.human_review_required = True

    if pr_ms is None:
        trifascicular = _evaluation(
            rule_id="CLIN-CONDUCTION-TRIFASCICULAR-01",
            code="trifascicular_block_pattern",
            status="unavailable",
            evidence=evidence,
            missing_inputs=["global.pr_ms"],
        )
    elif pr_ms > _INTERVALS.pr_prolonged_ms:
        trifascicular = _evaluation(
            rule_id="CLIN-CONDUCTION-TRIFASCICULAR-01",
            code="trifascicular_block_pattern",
            status="matched",
            evidence={
                **evidence,
                "pr_prolonged_gt_ms": _INTERVALS.pr_prolonged_ms,
                "manual_confirmation_required": True,
            },
            statement=(
                "Incomplete trifascicular block pattern: bifascicular block "
                "with first-degree AV delay; the site of delay cannot be "
                "localised from the surface ECG"
            ),
            confidence="pattern_only",
        )
        trifascicular.severity = "abnormal"
        trifascicular.priority = "P1"
        trifascicular.human_review_required = True
    else:
        trifascicular = _evaluation(
            rule_id="CLIN-CONDUCTION-TRIFASCICULAR-01",
            code="trifascicular_block_pattern",
            status="not_matched",
            evidence=evidence,
        )
    return [bifascicular, trifascicular]


def evaluate_conduction(context) -> list[RuleEvaluation]:
    qrs_ms = context.global_value("qrs_ms")
    axis = context.global_value("qrs_axis_deg")
    classification_qrs_ms, wide_qrs_ms, endpoint_discordant = (
        _qrs_endpoint_profile(context, qrs_ms)
    )
    rbbb = _rbbb(context, qrs_ms)
    lbbb = _lbbb(context, qrs_ms)
    if rbbb.status == "matched" and lbbb.status == "matched":
        rbbb_rank = _bbb_evidence_rank(rbbb)
        lbbb_rank = _bbb_evidence_rank(lbbb)
        if rbbb_rank > lbbb_rank:
            lbbb = _suppressed_by(lbbb, rbbb.statement_code or "rbbb_pattern")
        elif lbbb_rank > rbbb_rank:
            rbbb = _suppressed_by(rbbb, lbbb.statement_code or "lbbb_pattern")
        else:
            rbbb = _suppressed_by(rbbb, "conflicting_bundle_branch_morphology")
            lbbb = _suppressed_by(lbbb, "conflicting_bundle_branch_morphology")
    lafb = _lafb(context, axis)
    lpfb = _lpfb(context, axis)
    if lbbb.status == "matched":
        if lafb.status == "matched":
            lafb = _suppressed_by(
                lafb, lbbb.statement_code or "lbbb_pattern"
            )
        if lpfb.status == "matched":
            lpfb = _suppressed_by(
                lpfb, lbbb.statement_code or "lbbb_pattern"
            )
    qrs_evidence = {
        "qrs_ms": qrs_ms,
        "qrs_wide_endpoint_ms": wide_qrs_ms,
        "qrs_for_conduction_screen_ms": classification_qrs_ms,
        "qrs_endpoint_discordant": endpoint_discordant,
    }
    if classification_qrs_ms is None:
        ivcd = _evaluation(
            rule_id="CLIN-CONDUCTION-IVCD-01",
            code="nonspecific_ivcd",
            status="unavailable",
            evidence=qrs_evidence,
            missing_inputs=["global.qrs_ms"],
        )
    elif rbbb.status == "matched" or lbbb.status == "matched":
        ivcd = _evaluation(
            rule_id="CLIN-CONDUCTION-IVCD-01",
            code="nonspecific_ivcd",
            status="not_matched",
            evidence={
                **qrs_evidence,
                "excluded_by": [
                    item.statement_code
                    for item in (rbbb, lbbb)
                    if item.status == "matched"
                ],
            },
        )
    elif (
        qrs_ms is not None
        and qrs_ms >= 120.0
        and (rbbb.status == "unavailable" or lbbb.status == "unavailable")
    ):
        ivcd = _evaluation(
            rule_id="CLIN-CONDUCTION-IVCD-01",
            code="nonspecific_ivcd",
            status="matched",
            statement_code="probable_nonspecific_ivcd",
            confidence="incomplete_bbb_exclusion",
            statement=(
                "Probable nonspecific intraventricular conduction delay "
                "(QRS >=120 ms; complete bundle-branch morphology exclusion "
                "is unavailable); confirm manually"
            ),
            evidence={
                **qrs_evidence,
                "rbbb_status": rbbb.status,
                "lbbb_status": lbbb.status,
                "complete_bbb_morphology_exclusion": False,
                "manual_confirmation_required": True,
            },
        )
    elif (
        qrs_ms is not None
        and qrs_ms <= 110.0
        and wide_qrs_ms is not None
        and 120.0 <= wide_qrs_ms <= 140.0
        and 15.0 <= wide_qrs_ms - qrs_ms <= 30.0
        and qrs_ms >= 100.0
        and rbbb.status == "not_matched"
        and lbbb.status == "not_matched"
    ):
        ivcd = _evaluation(
            rule_id="CLIN-CONDUCTION-IVCD-01",
            code="nonspecific_ivcd",
            status="matched",
            statement_code="probable_nonspecific_ivcd",
            confidence="probable_wide_endpoint_consensus",
            statement=(
                "Probable nonspecific intraventricular conduction delay "
                "(wide-QRS endpoint consensus exceeds the central global "
                "measurement and bundle-branch morphology is not matched); "
                "confirm manually"
            ),
            evidence={
                **qrs_evidence,
                "rbbb_status": rbbb.status,
                "lbbb_status": lbbb.status,
                "manual_confirmation_required": True,
            },
        )
    elif qrs_ms is not None and qrs_ms <= 110.0:
        ivcd = _evaluation(
            rule_id="CLIN-CONDUCTION-IVCD-01",
            code="nonspecific_ivcd",
            status="not_matched",
            evidence=qrs_evidence,
        )
    elif rbbb.status == "unavailable" or lbbb.status == "unavailable":
        ivcd = _evaluation(
            rule_id="CLIN-CONDUCTION-IVCD-01",
            code="nonspecific_ivcd",
            status="unavailable",
            evidence={
                **qrs_evidence,
                "rbbb_status": rbbb.status,
                "lbbb_status": lbbb.status,
            },
            missing_inputs=["complete_bbb_morphology_exclusion"],
        )
    else:
        no_bbb = rbbb.status != "matched" and lbbb.status != "matched"
        ivcd = _evaluation(
            rule_id="CLIN-CONDUCTION-IVCD-01",
            code="nonspecific_ivcd",
            status="matched" if no_bbb else "not_matched",
            evidence={
                **qrs_evidence,
                "manual_confirmation_required": False,
            },
            statement="Nonspecific intraventricular conduction delay",
        )
    return [
        rbbb,
        lbbb,
        lafb,
        lpfb,
        ivcd,
        *_fascicular_combinations(context, rbbb, lafb, lpfb),
    ]

"""Precordial R-wave progression and axial rotation statements.

The measurement side of this already existed in ``interpret.interpret`` as the
``r_progression_class`` / ``r_s_transition_lead`` fields, but nothing in the
statement layer consumed them, so poor R-wave progression never reached a
report even though it is one of the commonest reasons an anterior infarct is
over- or under-called.

Progression findings are descriptive, not causal: the differential includes
prior anterior infarction, LVH, LBBB/LAFB, COPD, pre-excitation, and simply
high electrode placement. The rules below therefore emit an observation with
the competing causes attached, and stay silent when a mechanism that
mechanically rewrites precordial R waves is already established.
"""
from __future__ import annotations

from statistics import median
from typing import Iterable, Optional

from .config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import RuleEvaluation
from .sources import SOURCE_AHA_CONDUCTION_2009


_MORPHOLOGY = DEFAULT_DIAGNOSTIC_CONFIG.morphology

PRECORDIAL_ORDER = ("V1", "V2", "V3", "V4", "V5", "V6")

# Mechanisms that rewrite precordial QRS morphology outright. When one of
# these is established the progression pattern is a consequence of it and
# reporting it separately is misleading.
_SUPPRESSING_CODES = {
    "lbbb_pattern",
    "probable_lbbb_pattern",
    "lafb_pattern",
    "ventricular_preexcitation_pattern",
}

# Causes that must be weighed by a reader but do not invalidate the finding.
_REVIEW_CONFOUNDERS = (
    "prior_anterior_infarction",
    "left_ventricular_hypertrophy",
    "chronic_lung_disease",
    "high_precordial_electrode_placement",
    "body_habitus",
)


def _matched_codes(rows: Iterable[RuleEvaluation]) -> set[str]:
    codes: set[str] = set()
    for row in rows:
        if row.status != "matched":
            continue
        for key in (row.statement_code, row.evidence.get("evaluates_code")):
            if key:
                codes.add(str(key))
    return codes


def _result(
    *,
    rule_id: str,
    code: str,
    status: str,
    evidence: dict,
    statement: Optional[str] = None,
    statement_code: Optional[str] = None,
    missing_inputs: Optional[list[str]] = None,
    suppressed_by: Optional[list[str]] = None,
    severity: str = "observation",
    confidence: Optional[str] = None,
    priority: Optional[str] = None,
) -> RuleEvaluation:
    # `code` is the stable identity of the rule and is what suppression and
    # lookups key on; `statement_code` is the specific finding this run
    # produced and may vary between evaluations of the same rule.
    matched = status == "matched"
    return RuleEvaluation(
        rule_id=rule_id,
        domain="ischemia_infarction",
        status=status,
        statement_code=(
            (statement_code or code) if status in {"matched", "suppressed"} else None
        ),
        statement=statement if status in {"matched", "suppressed"} else None,
        severity=severity if status in {"matched", "suppressed"} else "normal",
        confidence=confidence,
        required_inputs=[f"lead.{lead}.r_amp_mv" for lead in PRECORDIAL_ORDER[:4]],
        missing_inputs=list(missing_inputs or []),
        evidence={**evidence, "evaluates_code": code},
        suppressed_by=list(suppressed_by or []),
        source=dict(SOURCE_AHA_CONDUCTION_2009),
        normality_required=False,
        normality_role="optional_screen",
        priority=priority,
        human_review_required=matched,
    )


def _transition_r_floor(
    r_values: dict[str, Optional[float]],
    s_values: dict[str, Optional[float]],
) -> float:
    """Smallest R amplitude that may count as a real R wave in this record.

    A fixed absolute floor would erase the genuine transition on a low-voltage
    tracing, so it is scaled to the record's own precordial QRS amplitude and
    then clamped at both ends.
    """
    amplitudes = [
        r_amp + abs(s_values.get(lead) or 0.0)
        for lead in PRECORDIAL_ORDER
        for r_amp in (r_values.get(lead),)
        if r_amp is not None
    ]
    if not amplitudes:
        return _MORPHOLOGY.transition_min_r_mv
    scaled = _MORPHOLOGY.transition_min_r_fraction * median(amplitudes)
    return max(
        _MORPHOLOGY.transition_min_r_floor_mv,
        min(_MORPHOLOGY.transition_min_r_mv, scaled),
    )


def _dominant_r(
    lead: str,
    r_values: dict[str, Optional[float]],
    s_values: dict[str, Optional[float]],
    floor_mv: float,
) -> Optional[bool]:
    """Whether R outweighs S in one lead, or None if it cannot be judged."""
    r_amp = r_values.get(lead)
    if r_amp is None:
        return None
    if r_amp < floor_mv:
        # Too small to be an R wave rather than noise on a QS complex.
        return False
    s_amp = s_values.get(lead)
    return r_amp >= (abs(s_amp) if s_amp is not None else 0.0)


def _transition_lead(
    r_values: dict[str, Optional[float]],
    s_values: dict[str, Optional[float]],
    floor_mv: float,
) -> Optional[str]:
    """First precordial lead where R becomes and stays dominant over S.

    The crossover has to hold in the next lead as well. A single lead where R
    edges past S and the following lead reverts is a measurement wobble, not a
    transition zone.
    """
    for index, lead in enumerate(PRECORDIAL_ORDER):
        if not _dominant_r(lead, r_values, s_values, floor_mv):
            continue
        if index + 1 < len(PRECORDIAL_ORDER):
            following = _dominant_r(
                PRECORDIAL_ORDER[index + 1], r_values, s_values, floor_mv
            )
            if following is False:
                continue
        return lead
    return None


def evaluate_r_progression(
    context, conduction: Iterable[RuleEvaluation]
) -> list[RuleEvaluation]:
    r_values = {
        lead: context.lead_value(lead, "r_amp_mv") for lead in PRECORDIAL_ORDER
    }
    s_values = {
        lead: context.lead_value(lead, "s_amp_mv") for lead in PRECORDIAL_ORDER
    }
    anterior = [r_values[lead] for lead in ("V1", "V2", "V3", "V4")]
    evidence_base = {
        "r_amp_mv": dict(r_values),
        "s_amp_mv": dict(s_values),
        "poor_progression_v3_threshold_mv": _MORPHOLOGY.poor_r_progression_v3_mv,
        "normal_transition_leads": list(_MORPHOLOGY.normal_transition_leads),
        "confounders_require_review": list(_REVIEW_CONFOUNDERS),
    }

    if context.precordial_reversal:
        return [
            _result(
                rule_id=rule_id,
                code=code,
                status="suppressed",
                evidence=evidence_base,
                suppressed_by=["precordial_lead_reversal"],
            )
            for rule_id, code in (
                ("CLIN-MORPH-RPROG-01", "poor_r_wave_progression"),
                ("CLIN-MORPH-ROTATION-01", "precordial_rotation"),
            )
        ]

    if any(value is None for value in anterior):
        missing = [
            f"lead.{lead}.r_amp_mv"
            for lead in ("V1", "V2", "V3", "V4")
            if r_values[lead] is None
        ]
        return [
            _result(
                rule_id=rule_id,
                code=code,
                status="unavailable",
                evidence=evidence_base,
                missing_inputs=missing,
            )
            for rule_id, code in (
                ("CLIN-MORPH-RPROG-01", "poor_r_wave_progression"),
                ("CLIN-MORPH-ROTATION-01", "precordial_rotation"),
            )
        ]

    suppressors = sorted(_matched_codes(conduction) & _SUPPRESSING_CODES)
    r_floor_mv = _transition_r_floor(r_values, s_values)
    evidence_base["transition_min_r_mv"] = r_floor_mv
    transition = _transition_lead(r_values, s_values, r_floor_mv)
    r_v3 = r_values["V3"]

    # Reversed progression: R shrinks monotonically away from V1 across the
    # anterior leads. Stronger evidence of anterior scar than plain PRP, and
    # also the classic dextrocardia / precordial-misplacement signature.
    reversed_progression = all(
        anterior[index] < anterior[index - 1] for index in range(1, 4)
    )
    no_transition_by_v4 = (
        transition is None
        or PRECORDIAL_ORDER.index(transition) >= PRECORDIAL_ORDER.index("V5")
    )
    poor_progression = bool(
        r_v3 is not None and r_v3 <= _MORPHOLOGY.poor_r_progression_v3_mv
    ) or no_transition_by_v4

    if reversed_progression:
        prog_statement_code = "reversed_r_wave_progression"
        prog_statement = (
            "Reversed precordial R-wave progression; consider anterior "
            "infarction, dextrocardia or precordial lead misplacement"
        )
        prog_confidence = "moderate"
    elif poor_progression:
        prog_statement_code = "poor_r_wave_progression"
        prog_statement = (
            "Poor precordial R-wave progression; not specific for anterior "
            "infarction, correlate with prior ECG and competing causes"
        )
        prog_confidence = "low"
    else:
        prog_statement_code = None
        prog_statement = None
        prog_confidence = None

    progression = _result(
        rule_id="CLIN-MORPH-RPROG-01",
        code="poor_r_wave_progression",
        statement_code=prog_statement_code,
        status="matched" if prog_statement else "not_matched",
        evidence={
            **evidence_base,
            "r_s_transition_lead": transition,
            "r_v3_mv": r_v3,
            "no_transition_through_v4": no_transition_by_v4,
            "reversed_progression": reversed_progression,
        },
        statement=prog_statement,
        suppressed_by=suppressors if prog_statement else [],
        confidence=prog_confidence,
        priority="P4" if prog_statement else None,
    )

    # Rotation is read off the transition zone: normal at V3-V4, rightward
    # (clockwise) when it moves to V5 or later, leftward (counterclockwise)
    # when V1-V2 already show R >= S.
    if transition is None:
        rotation_statement = None
        rotation_kind = "absent_transition"
    else:
        index = PRECORDIAL_ORDER.index(transition)
        if index >= PRECORDIAL_ORDER.index("V5"):
            rotation_kind = "clockwise"
            rotation_statement = (
                f"Clockwise rotation (R/S transition at {transition}); "
                "consider right ventricular load, chronic lung disease or "
                "vertical heart"
            )
        elif index <= PRECORDIAL_ORDER.index("V2"):
            rotation_kind = "counterclockwise"
            rotation_statement = (
                f"Counterclockwise rotation (R/S transition at {transition}); "
                "usually benign, exclude posterior infarction and right "
                "ventricular hypertrophy"
            )
        else:
            rotation_kind = "normal"
            rotation_statement = None

    rotation = _result(
        rule_id="CLIN-MORPH-ROTATION-01",
        code="precordial_rotation",
        statement_code=(
            f"{rotation_kind}_rotation" if rotation_statement else None
        ),
        status="matched" if rotation_statement else "not_matched",
        evidence={
            **evidence_base,
            "r_s_transition_lead": transition,
            "rotation": rotation_kind,
        },
        statement=rotation_statement,
        suppressed_by=suppressors if rotation_statement else [],
        confidence="low" if rotation_statement else None,
        priority="P5" if rotation_statement else None,
    )
    return [progression, rotation]

"""U-wave diagnostic statements.

Two statements, deliberately asymmetric in how much evidence they demand:

``prominent_u_wave``
    Common and mostly non-specific (bradycardia alone produces it), so it is
    reported as a low-severity observation and needs corroboration across
    leads.

``inverted_u_wave``
    Uncommon, easy to fabricate out of baseline wander, and -- when real --
    highly specific for myocardial ischemia.  It therefore demands a
    polarity-consistent signed measurement, an upright T wave in the same lead
    to invert *against*, and two supporting leads.  A false positive here
    would put ischemia on a report; the thresholds are set to make that hard.

Best observation leads for the U wave are V2-V3, where it is largest, but the
rules accept any lead with a reliable measurement and simply record which leads
carried the finding.

Reference: AHA/ACCF/HRS 2009 Part IV; LITFL ECG Library "U wave".
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from .models import RuleEvaluation
from .sources import SOURCE_AHA_REPOLARIZATION_2009


# Best-observation leads, reported for context. Not a gate: a prominent U wave
# in the inferior leads is still a prominent U wave.
PREFERRED_LEADS = ("V2", "V3")
ALL_LEADS = ("I", "II", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

# 1 mm. LITFL puts the normal ceiling at 1-2 mm; the lower bound is used so the
# screen is sensitive, with severity kept at "observation" to match.
PROMINENT_U_ABS_MV = 0.10
# Classic relative criterion: U exceeding a quarter of the T wave in the lead.
PROMINENT_U_T_RATIO = 0.25
# Above this the U wave is prominent regardless of a small T wave -- the ratio
# alone would fire on any lead with a flat T.
PROMINENT_U_RATIO_FLOOR_MV = 0.05
# The ratio criterion assumes a normally sized T wave to be a quarter of. In a
# lead with a 1 mm T wave every U wave is "disproportionate", which says
# something about the T wave rather than the U wave, so the relative path is
# only open when the T wave is at least 2 mm.
PROMINENT_U_RATIO_MIN_T_MV = 0.20

# Inverted U: amplitude floor is lower (an inverted U is rarely large) but
# every other gate is stricter.
INVERTED_U_MIN_ABS_MV = 0.05
# Amplitude is measured against the beat baseline, so a slow post-T downward
# drift produces a qualifying "amplitude" with almost no actual wave under it.
# Prominence is measured against the deflection's own flanks and is what
# separates a wave from a drift, so the inverted-U call is gated on it.
INVERTED_U_MIN_PROMINENCE_MV = 0.05
# The T wave must be clearly upright for "inverted relative to T" to mean
# anything.
INVERTED_U_MIN_UPRIGHT_T_MV = 0.20

MIN_SUPPORTING_LEADS = 2

# aVR is excluded from both screens: its normally negative T wave makes the
# polarity comparison meaningless.


def _lead_facts(context, lead: str) -> dict:
    amplitude = context.lead_value(lead, "u_amp_signed_mv", "reliable_for_t")
    reliable = context.lead_raw_value(lead, "u_measurement_reliable", "reliable_for_t")
    polarity = context.lead_value(lead, "u_polarity", "reliable_for_t")
    t_amp = context.lead_value(lead, "t_amp_mv", "reliable_for_t")
    ratio: Optional[float] = None
    if amplitude is not None and t_amp not in {None, 0.0}:
        ratio = abs(amplitude) / abs(t_amp)
    return {
        "u_amp_signed_mv": amplitude,
        "u_polarity": None if polarity is None else int(polarity),
        "u_measurement_reliable": bool(reliable),
        "u_duration_ms": context.lead_value(lead, "u_dur_ms", "reliable_for_t"),
        "u_prominence_mv": context.lead_value(
            lead, "u_prominence_mv", "reliable_for_t"
        ),
        "u_polarity_agreement": context.lead_value(
            lead, "u_polarity_agreement", "reliable_for_t"
        ),
        "t_amp_mv": t_amp,
        "u_to_t_ratio": ratio,
    }


def _measured(facts: dict) -> bool:
    return bool(
        facts["u_measurement_reliable"]
        and facts["u_amp_signed_mv"] is not None
    )


def _prominent(facts: dict) -> bool:
    if not _measured(facts):
        return False
    amplitude = abs(float(facts["u_amp_signed_mv"]))
    if float(facts["u_amp_signed_mv"]) < 0.0:
        # An inverted U is reported by the other rule; "prominent" is about an
        # oversized upright U wave.
        return False
    if amplitude >= PROMINENT_U_ABS_MV:
        return True
    ratio = facts["u_to_t_ratio"]
    t_amp = facts["t_amp_mv"]
    return bool(
        ratio is not None
        and ratio > PROMINENT_U_T_RATIO
        and amplitude >= PROMINENT_U_RATIO_FLOOR_MV
        and t_amp is not None
        and abs(t_amp) >= PROMINENT_U_RATIO_MIN_T_MV
    )


def _inverted(facts: dict) -> bool:
    if not _measured(facts):
        return False
    amplitude = float(facts["u_amp_signed_mv"])
    prominence = facts["u_prominence_mv"]
    t_amp = facts["t_amp_mv"]
    return bool(
        amplitude <= -INVERTED_U_MIN_ABS_MV
        and prominence is not None
        and prominence >= INVERTED_U_MIN_PROMINENCE_MV
        and t_amp is not None
        and t_amp >= INVERTED_U_MIN_UPRIGHT_T_MV
    )


def _repolarization_confounders(
    context, conduction: Iterable[RuleEvaluation]
) -> list[str]:
    """Mechanisms that make a U-wave call unsafe.

    Under a wide-QRS activation sequence or ventricular pacing, what looks
    like a discrete U wave is usually the tail of a grossly abnormal T wave.
    Under fibrillation or flutter the T-U segment is filled with atrial
    waves, and a fibrillatory or flutter deflection landing after the T wave
    is indistinguishable from a U wave on morphology alone.
    """
    reasons: list[str] = []
    rhythm_meta = getattr(context.features, "metadata", {}).get(
        "rhythm_analysis", {}
    )
    rhythm_meta = rhythm_meta if isinstance(rhythm_meta, dict) else {}
    af_afl = rhythm_meta.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    if af_afl.get("probable_af"):
        reasons.append("probable_atrial_fibrillation")
    if af_afl.get("probable_flutter"):
        reasons.append("probable_atrial_flutter")
    matched = {
        str(item.evidence.get("evaluates_code") or item.statement_code)
        for item in conduction
        if item.status == "matched"
    }
    for code in ("lbbb_pattern", "rbbb_pattern", "nonspecific_ivcd"):
        if code in matched:
            reasons.append(code)
    qrs_ms = context.global_value("qrs_ms")
    if qrs_ms is not None and float(qrs_ms) >= 120.0:
        reasons.append("qrs_duration_ge_120_ms")
    rhythm = getattr(context.features, "metadata", {}).get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    if pacing.get("continuous_pacing") or pacing.get("ventricular_pacing_present"):
        reasons.append("ventricular_pacing")
    return sorted(set(reasons))


def evaluate_u_wave(
    context,
    *,
    conduction: Iterable[RuleEvaluation] = (),
) -> list[RuleEvaluation]:
    facts = {lead: _lead_facts(context, lead) for lead in ALL_LEADS}
    measured_leads = [lead for lead in ALL_LEADS if _measured(facts[lead])]
    prominent_leads = [lead for lead in ALL_LEADS if _prominent(facts[lead])]
    inverted_leads = [lead for lead in ALL_LEADS if _inverted(facts[lead])]

    heart_rate = context.global_value("heart_rate_bpm")
    # U-wave amplitude is inversely related to rate; at tachycardia the U wave
    # fuses with the following P wave and any "prominent U" is more likely to
    # be that P wave. Refuse the call rather than report it.
    rate_confounded = heart_rate is not None and heart_rate >= 100.0

    # Coverage: without a reliable measurement in at least two leads, absence
    # of a U-wave finding is not evidence that none is present.
    available = len(measured_leads) >= MIN_SUPPORTING_LEADS

    prominent_matched = bool(
        available
        and not rate_confounded
        and len(prominent_leads) >= MIN_SUPPORTING_LEADS
    )
    inverted_matched = bool(
        available and len(inverted_leads) >= MIN_SUPPORTING_LEADS
    )

    confounders = _repolarization_confounders(context, list(conduction))

    shared_evidence = {
        "lead_facts": facts,
        "measured_leads": measured_leads,
        "preferred_observation_leads": list(PREFERRED_LEADS),
        "heart_rate_bpm": heart_rate,
        "repolarization_confounders": confounders,
    }

    prominent = RuleEvaluation(
        rule_id="CLIN-REPOLARIZATION-U-01",
        domain="repolarization",
        status=(
            "matched"
            if prominent_matched
            else "indeterminate"
            if available and rate_confounded and len(prominent_leads) >= MIN_SUPPORTING_LEADS
            else "not_matched"
            if available
            else "unavailable"
        ),
        statement_code="prominent_u_wave" if prominent_matched else None,
        statement=(
            "Prominent U waves; consider bradycardia, hypokalemia, "
            "hypomagnesemia, hypercalcemia, LVH or drug effect "
            "(digoxin, quinidine, amiodarone)"
            if prominent_matched
            else None
        ),
        severity="observation" if prominent_matched else "normal",
        confidence="low" if prominent_matched else None,
        priority="P5",
        required_inputs=[
            "lead.u_amp_signed_mv",
            "lead.u_measurement_reliable",
            "lead.t_amp_mv",
        ],
        missing_inputs=(
            [] if available else ["two_leads_with_reliable_signed_u_measurement"]
        ),
        evidence={
            **shared_evidence,
            "evaluates_code": "prominent_u_wave",
            "qualifying_leads": prominent_leads,
            "rate_confounded": rate_confounded,
        },
        thresholds={
            "prominent_absolute_ge_mv": PROMINENT_U_ABS_MV,
            "prominent_u_to_t_ratio_gt": PROMINENT_U_T_RATIO,
            "prominent_ratio_amplitude_floor_mv": PROMINENT_U_RATIO_FLOOR_MV,
            "prominent_ratio_requires_t_ge_mv": PROMINENT_U_RATIO_MIN_T_MV,
            "minimum_supporting_leads": MIN_SUPPORTING_LEADS,
            "rate_confound_ge_bpm": 100.0,
        },
        source=dict(SOURCE_AHA_REPOLARIZATION_2009),
        normality_required=False,
        normality_role="optional_screen",
        suppressed_by=confounders if prominent_matched else [],
    )

    inverted = RuleEvaluation(
        rule_id="CLIN-REPOLARIZATION-U-02",
        domain="repolarization",
        status=(
            "matched"
            if inverted_matched
            else "not_matched"
            if available
            else "unavailable"
        ),
        statement_code="inverted_u_wave" if inverted_matched else None,
        statement=(
            "Inverted U waves in leads with upright T waves; specific for "
            "myocardial ischemia -- correlate with symptoms and prior ECG"
            if inverted_matched
            else None
        ),
        severity="abnormal" if inverted_matched else "normal",
        confidence="moderate" if inverted_matched else None,
        priority="P2" if inverted_matched else None,
        required_inputs=[
            "lead.u_amp_signed_mv",
            "lead.u_measurement_reliable",
            "lead.t_amp_mv",
        ],
        missing_inputs=(
            [] if available else ["two_leads_with_reliable_signed_u_measurement"]
        ),
        evidence={
            **shared_evidence,
            "evaluates_code": "inverted_u_wave",
            "qualifying_leads": inverted_leads,
            "manual_confirmation_required": inverted_matched,
        },
        thresholds={
            "inverted_u_le_mv": -INVERTED_U_MIN_ABS_MV,
            "minimum_prominence_mv": INVERTED_U_MIN_PROMINENCE_MV,
            "requires_upright_t_ge_mv": INVERTED_U_MIN_UPRIGHT_T_MV,
            "minimum_supporting_leads": MIN_SUPPORTING_LEADS,
        },
        source=dict(SOURCE_AHA_REPOLARIZATION_2009),
        normality_required=False,
        normality_role="optional_screen",
        human_review_required=inverted_matched,
        # RuleEvaluation.__post_init__ turns a matched row with suppressors
        # into "suppressed", which is the intended outcome here.
        suppressed_by=confounders if inverted_matched else [],
    )
    return [prominent, inverted]

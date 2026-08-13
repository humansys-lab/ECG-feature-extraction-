"""T-wave morphology sub-phenotypes.

``evaluate_t_wave_abnormalities`` already answers "is the T wave abnormal, and
is it primary or secondary".  Two specific T-wave shapes carry management
consequences that a generic "primary T-wave abnormality" does not, and each is
distinguished from its look-alike by shape rather than amplitude:

``hyperacute_t_wave_pattern``
    The earliest ECG change of coronary occlusion, appearing *before* ST
    elevation.  Its look-alike is the hyperkalemic T wave, and amplitude does
    not separate them -- both are tall.  Shape does: hyperacute T waves are
    broad-based and out of proportion to the QRS, hyperkalemic T waves are
    narrow and symmetric with a widened QRS and an attenuated P wave.

``giant_negative_t_wave``
    Deep precordial T inversion (conventionally >= 10 mm), the hallmark of
    apical hypertrophic cardiomyopathy.  Its look-alikes are LVH strain and
    Wellens-type ischemic inversion, which are shallower; the >= 1.0 mV depth
    is what makes the call, and confounders are reported alongside rather than
    silently swallowed.

Reference: AHA/ACCF/HRS 2009 Part IV; LITFL ECG Library "T wave", "Hyperacute
T waves"; Yamaguchi et al. on apical HCM giant negative T waves.
"""
from __future__ import annotations

from collections.abc import Iterable

from .models import RuleEvaluation
from .repolarization import CONTIGUOUS_PAIRS, _qrs_peak_to_peak
from .sources import SOURCE_AHA_REPOLARIZATION_2009


HYPERACUTE_LEADS = (
    "I", "II", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")

# Absolute floor. Below this a "tall" T wave is not tall in any lead.
HYPERACUTE_MIN_AMP_MV = 0.60
# Out of proportion to the depolarisation that produced it. This, not the
# absolute height, is what makes a hyperacute T wave recognisable.
HYPERACUTE_T_TO_QRS_RATIO = 0.75
# Broad base. Hyperacute T waves are wide; hyperkalemic ones are narrow.
HYPERACUTE_MIN_DURATION_MS = 160.0
# 160-200 ms is the band where broad-and-hyperacute overlaps
# narrow-and-hyperkalemic on duration alone, so symmetry decides inside it: a
# symmetric T wave in that band is read as the hyperkalemic look-alike and the
# hyperacute call is refused. Above 200 ms the T wave is too broad for
# hyperkalemia and symmetry no longer excludes.
HYPERKALEMIC_SYMMETRY_RANGE = (0.8, 1.2)
HYPERKALEMIC_MAX_DURATION_MS = 200.0

# Conventional giant-negative threshold: 10 mm.
GIANT_NEGATIVE_T_MV = -1.00
# A second, shallower lead must agree, so a single artefactual deflection
# cannot carry the finding.
GIANT_NEGATIVE_SUPPORT_T_MV = -0.50


def _matched_codes(rows: Iterable[RuleEvaluation]) -> set[str]:
    return {
        str(item.evidence.get("evaluates_code") or item.statement_code)
        for item in rows
        if item.status == "matched"
    }


def _secondary_repolarization_confounders(
    context,
    conduction: Iterable[RuleEvaluation],
    hypertrophy: Iterable[RuleEvaluation],
) -> list[str]:
    reasons: list[str] = []
    conduction_codes = _matched_codes(conduction)
    reasons.extend(
        code
        for code in ("lbbb_pattern", "rbbb_pattern", "nonspecific_ivcd")
        if code in conduction_codes
    )
    hypertrophy_codes = _matched_codes(hypertrophy)
    reasons.extend(
        code
        for code in ("lvh_voltage_criteria", "rvh_pattern")
        if code in hypertrophy_codes
    )
    rhythm = getattr(context.features, "metadata", {}).get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    if pacing.get("continuous_pacing") or pacing.get("ventricular_pacing_present"):
        reasons.append("ventricular_pacing")
    return sorted(set(reasons))


def _hyperacute_facts(context, lead: str) -> dict:
    t_amp = context.lead_value(lead, "t_amp_mv", "reliable_for_t")
    qrs_pp = _qrs_peak_to_peak(context, lead)
    r_amp = context.lead_value(lead, "r_amp_mv")
    duration = context.lead_value(lead, "t_dur_ms", "reliable_for_t")
    symmetry = context.lead_value(lead, "t_symmetry", "reliable_for_t")
    return {
        "t_amp_mv": t_amp,
        "t_dur_ms": duration,
        "t_symmetry": symmetry,
        "r_amp_mv": r_amp,
        "qrs_peak_to_peak_mv": qrs_pp,
        "t_to_qrs_ratio": (
            abs(t_amp) / qrs_pp
            if t_amp is not None and qrs_pp not in {None, 0.0}
            else None
        ),
    }


def _hyperacute_lead(facts: dict) -> bool:
    t_amp = facts["t_amp_mv"]
    duration = facts["t_dur_ms"]
    ratio = facts["t_to_qrs_ratio"]
    if t_amp is None or duration is None or ratio is None:
        return False
    if t_amp < HYPERACUTE_MIN_AMP_MV:
        return False
    if ratio < HYPERACUTE_T_TO_QRS_RATIO:
        return False
    if duration < HYPERACUTE_MIN_DURATION_MS:
        return False
    symmetry = facts["t_symmetry"]
    narrow_and_symmetric = bool(
        symmetry is not None
        and HYPERKALEMIC_SYMMETRY_RANGE[0] <= symmetry <= HYPERKALEMIC_SYMMETRY_RANGE[1]
        and duration <= HYPERKALEMIC_MAX_DURATION_MS
    )
    return not narrow_and_symmetric


def evaluate_t_morphology(
    context,
    *,
    conduction: Iterable[RuleEvaluation] = (),
    hypertrophy: Iterable[RuleEvaluation] = (),
) -> list[RuleEvaluation]:
    conduction = list(conduction)
    hypertrophy = list(hypertrophy)
    confounders = _secondary_repolarization_confounders(
        context, conduction, hypertrophy
    )

    # ── Hyperacute T waves ───────────────────────────────────────────────────
    hyper_facts = {
        lead: _hyperacute_facts(context, lead) for lead in HYPERACUTE_LEADS
    }
    hyper_leads = [
        lead for lead in HYPERACUTE_LEADS if _hyperacute_lead(hyper_facts[lead])
    ]
    hyper_lead_set = set(hyper_leads)
    hyper_pairs = [
        [first, second]
        for first, second in CONTIGUOUS_PAIRS
        if first in hyper_lead_set and second in hyper_lead_set
    ]
    hyper_measured = [
        lead
        for lead in HYPERACUTE_LEADS
        if hyper_facts[lead]["t_amp_mv"] is not None
        and hyper_facts[lead]["t_dur_ms"] is not None
        and hyper_facts[lead]["t_to_qrs_ratio"] is not None
    ]
    hyper_available = len(hyper_measured) >= 6
    hyper_matched = bool(hyper_available and hyper_pairs)

    hyperacute = RuleEvaluation(
        rule_id="CLIN-ISCHEMIA-HYPERACUTE-T-01",
        domain="ischemia_infarction",
        status=(
            "matched"
            if hyper_matched
            else "not_matched"
            if hyper_available
            else "unavailable"
        ),
        statement_code="hyperacute_t_wave_pattern" if hyper_matched else None,
        statement=(
            "Hyperacute T-wave pattern; may precede ST elevation in acute "
            "coronary occlusion -- obtain serial ECGs and correlate urgently"
            if hyper_matched
            else None
        ),
        severity="high" if hyper_matched else "normal",
        confidence="low" if hyper_matched else None,
        priority="P1" if hyper_matched else None,
        required_inputs=[
            "lead.t_amp_mv",
            "lead.t_dur_ms",
            "lead.t_symmetry",
            "lead.qrs_peak_to_peak_mv",
        ],
        missing_inputs=(
            []
            if hyper_available
            else ["reliable_t_amplitude_and_duration_in_six_leads"]
        ),
        evidence={
            "evaluates_code": "hyperacute_t_wave_pattern",
            "lead_facts": hyper_facts,
            "qualifying_leads": hyper_leads,
            "qualifying_contiguous_pairs": hyper_pairs,
            "measured_leads": hyper_measured,
            "secondary_repolarization_confounders": confounders,
            "manual_confirmation_required": hyper_matched,
        },
        thresholds={
            "minimum_t_amplitude_mv": HYPERACUTE_MIN_AMP_MV,
            "minimum_t_to_qrs_ratio": HYPERACUTE_T_TO_QRS_RATIO,
            "minimum_t_duration_ms": HYPERACUTE_MIN_DURATION_MS,
            "hyperkalemic_exclusion_symmetry": list(HYPERKALEMIC_SYMMETRY_RANGE),
            "hyperkalemic_exclusion_max_duration_ms": HYPERKALEMIC_MAX_DURATION_MS,
            "minimum_contiguous_leads": 2,
        },
        source=dict(SOURCE_AHA_REPOLARIZATION_2009),
        normality_required=False,
        # Supporting, not core: ST elevation, ST depression and Q waves already
        # carry the ischemia domain's normality claim. A missing T duration
        # must not make the whole domain read as unevaluated.
        normality_role="supporting",
        human_review_required=hyper_matched,
        suppressed_by=confounders if hyper_matched else [],
    )

    # ── Giant negative T waves ───────────────────────────────────────────────
    giant_facts = {
        lead: {
            "t_amp_mv": context.lead_value(lead, "t_amp_mv", "reliable_for_t"),
        }
        for lead in PRECORDIAL_LEADS
    }
    giant_leads = [
        lead
        for lead in PRECORDIAL_LEADS
        if giant_facts[lead]["t_amp_mv"] is not None
        and giant_facts[lead]["t_amp_mv"] <= GIANT_NEGATIVE_T_MV
    ]
    support_leads = [
        lead
        for lead in PRECORDIAL_LEADS
        if giant_facts[lead]["t_amp_mv"] is not None
        and giant_facts[lead]["t_amp_mv"] <= GIANT_NEGATIVE_SUPPORT_T_MV
    ]
    giant_measured = [
        lead for lead in PRECORDIAL_LEADS if giant_facts[lead]["t_amp_mv"] is not None
    ]
    giant_available = len(giant_measured) >= 4
    giant_matched = bool(
        giant_available and giant_leads and len(support_leads) >= 2
    )

    giant = RuleEvaluation(
        rule_id="CLIN-REPOLARIZATION-GIANT-T-01",
        domain="repolarization",
        status=(
            "matched"
            if giant_matched
            else "not_matched"
            if giant_available
            else "unavailable"
        ),
        statement_code="giant_negative_t_wave" if giant_matched else None,
        statement=(
            # Apical HCM is the classic association but not the commonest
            # cause; deep ischemic inversion produces the same depth and is
            # seen far more often, so the statement names both rather than
            # steering the reader to the rarer diagnosis.
            "Giant negative precordial T waves; differential includes deep "
            "ischemic T-wave inversion and apical hypertrophic "
            "cardiomyopathy -- correlate clinically and consider "
            "echocardiography"
            if giant_matched
            else None
        ),
        severity="abnormal" if giant_matched else "normal",
        confidence="moderate" if giant_matched else None,
        priority="P2" if giant_matched else None,
        required_inputs=["lead.t_amp_mv", "lead.reliable_for_t"],
        missing_inputs=(
            [] if giant_available else ["reliable_t_amplitude_in_four_precordial_leads"]
        ),
        evidence={
            "evaluates_code": "giant_negative_t_wave",
            "lead_facts": giant_facts,
            "qualifying_leads": giant_leads,
            "supporting_leads": support_leads,
            "measured_leads": giant_measured,
            # Reported, not suppressive: LVH strain and apical HCM coexist,
            # and a >= 10 mm inversion is beyond what strain alone explains.
            "differential_confounders": confounders,
            "manual_confirmation_required": giant_matched,
        },
        thresholds={
            "giant_negative_t_le_mv": GIANT_NEGATIVE_T_MV,
            "supporting_inversion_le_mv": GIANT_NEGATIVE_SUPPORT_T_MV,
            "minimum_supporting_leads": 2,
        },
        source=dict(SOURCE_AHA_REPOLARIZATION_2009),
        normality_required=False,
        normality_role="optional_screen",
        human_review_required=giant_matched,
    )

    return [hyperacute, giant]

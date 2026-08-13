from __future__ import annotations

from collections.abc import Iterable

from .models import RuleEvaluation
from .sources import SOURCE_AHA_REPOLARIZATION_2009


TERRITORIES = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anterior": ("V2", "V3", "V4"),
}
CONTIGUOUS_PAIRS = (
    ("II", "III"),
    ("III", "aVF"),
    ("II", "aVF"),
    ("I", "aVL"),
    ("aVL", "V5"),
    ("V5", "V6"),
    ("V2", "V3"),
    ("V3", "V4"),
)


def _matched_codes(rows: Iterable[RuleEvaluation]) -> set[str]:
    return {
        str(item.evidence.get("evaluates_code") or item.statement_code)
        for item in rows
        if item.status == "matched"
    }


def _qrs_peak_to_peak(context, lead: str) -> float | None:
    values = [
        context.lead_value(lead, name, "reliable_for_qrs")
        for name in (
            "q_amp_mv",
            "r_amp_mv",
            "s_amp_mv",
            "r_prime_amp_mv",
            "s_prime_amp_mv",
        )
    ]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return max(max(finite), 0.0) - min(min(finite), 0.0)


def evaluate_t_wave_abnormalities(
    context,
    *,
    conduction: Iterable[RuleEvaluation],
    hypertrophy: Iterable[RuleEvaluation],
    preexcitation: RuleEvaluation,
) -> RuleEvaluation:
    code_primary = "primary_t_wave_abnormality"
    code_secondary = "secondary_t_wave_abnormality"
    if context.age_years is not None and context.age_years < 18.0:
        return RuleEvaluation(
            rule_id="CLIN-REPOLARIZATION-T-01",
            domain="repolarization",
            status="unavailable",
            required_inputs=["adult_t_wave_polarity_criteria"],
            missing_inputs=["pediatric_t_wave_reference_table"],
            evidence={"evaluates_code": code_primary, "age_years": context.age_years},
            source=dict(SOURCE_AHA_REPOLARIZATION_2009),
            normality_required=False,
            normality_role="optional_screen",
        )

    all_leads = sorted({lead for leads in TERRITORIES.values() for lead in leads})
    facts = {}
    inverted = set()
    tall = set()
    measured = set()
    for lead in all_leads:
        t_amp = context.lead_value(lead, "t_amp_mv", "reliable_for_t")
        qrs_pp = _qrs_peak_to_peak(context, lead)
        facts[lead] = {
            "t_amp_mv": t_amp,
            "qrs_peak_to_peak_mv": qrs_pp,
            "relative_to_qrs": (
                abs(t_amp) / qrs_pp
                if t_amp is not None and qrs_pp not in {None, 0.0}
                else None
            ),
        }
        if t_amp is None:
            continue
        measured.add(lead)
        if t_amp <= -0.10:
            inverted.add(lead)
        if t_amp >= 1.20 or (
            t_amp >= 0.50 and qrs_pp not in {None, 0.0} and t_amp > 0.5 * qrs_pp
        ):
            tall.add(lead)

    inverted_pairs = [
        [first, second]
        for first, second in CONTIGUOUS_PAIRS
        if first in inverted and second in inverted
    ]
    tall_pairs = [
        [first, second]
        for first, second in CONTIGUOUS_PAIRS
        if first in tall and second in tall
    ]
    abnormal = bool(inverted_pairs or tall_pairs)

    conduction_codes = _matched_codes(conduction)
    hypertrophy_codes = _matched_codes(hypertrophy)
    rhythm = getattr(context.features, "metadata", {}).get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    secondary_by = sorted(
        conduction_codes
        & {"rbbb_pattern", "lbbb_pattern", "nonspecific_ivcd"}
        | hypertrophy_codes
        & {"lvh_voltage_criteria", "rvh_pattern"}
    )
    if preexcitation.status == "matched":
        secondary_by.append("ventricular_preexcitation_pattern")
    if pacing.get("continuous_pacing") or pacing.get("ventricular_pacing_present"):
        secondary_by.append("ventricular_pacing")
    secondary_by = sorted(set(secondary_by))

    coverage_ok = all(
        any(lead in measured for lead in leads)
        for leads in TERRITORIES.values()
    ) and len(measured) >= 6
    if abnormal:
        status = "matched"
        code = code_secondary if secondary_by else code_primary
        statement = (
            "Secondary T-wave abnormality associated with altered ventricular activation or hypertrophy"
            if secondary_by
            else "Primary T-wave abnormality pattern"
        )
        missing: list[str] = []
        coverage = "full" if coverage_ok else "partial"
    elif coverage_ok:
        status = "not_matched"
        code = code_primary
        statement = None
        missing = []
        coverage = "full"
    else:
        status = "unavailable"
        code = code_primary
        statement = None
        missing = ["reliable_t_measurements_across_anterior_inferior_lateral_territories"]
        coverage = "unavailable"

    return RuleEvaluation(
        rule_id="CLIN-REPOLARIZATION-T-01",
        domain="repolarization",
        status=status,
        statement_code=code if status == "matched" else None,
        statement=statement,
        severity="abnormal" if status == "matched" else "normal",
        confidence="moderate" if status == "matched" else None,
        coverage=coverage,
        required_inputs=["lead.t_amp_mv", "lead.qrs_peak_to_peak_mv", "lead.reliable_for_t"],
        missing_inputs=missing,
        evidence={
            "evaluates_code": code,
            "lead_facts": facts,
            "inverted_leads": sorted(inverted),
            "inverted_contiguous_pairs": inverted_pairs,
            "tall_positive_t_leads": sorted(tall),
            "tall_contiguous_pairs": tall_pairs,
            "secondary_by": secondary_by,
            "manual_confirmation_required": status == "matched",
        },
        thresholds={
            "t_inversion_le_mv": -0.10,
            "tall_t_absolute_ge_mv": 1.20,
            "tall_t_relative_ge_mv": 0.50,
            "tall_t_relative_to_qrs_gt": 0.50,
            "minimum_contiguous_leads": 2,
        },
        source=dict(SOURCE_AHA_REPOLARIZATION_2009),
        normality_required=False,
        normality_role="optional_screen",
    )

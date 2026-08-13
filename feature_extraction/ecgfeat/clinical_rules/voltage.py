from __future__ import annotations

from typing import Iterable

from .models import RuleEvaluation
from .sources import SOURCE_AHA_LOW_VOLTAGE


LIMB_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")
PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")


def _peak_to_peak_mv(context, lead: str) -> float | None:
    components = [
        context.lead_value(lead, name, "reliable_for_qrs")
        for name in (
            "q_amp_mv",
            "r_amp_mv",
            "s_amp_mv",
            "r_prime_amp_mv",
            "s_prime_amp_mv",
        )
    ]
    components = [value for value in components if value is not None]
    if not components:
        return None
    return max(max(components), 0.0) - min(min(components), 0.0)


def _evaluate_region(
    context,
    *,
    leads: Iterable[str],
    threshold_mv: float,
    code: str,
    rule_id: str,
    statement: str,
) -> RuleEvaluation:
    lead_list = tuple(leads)
    values = {lead: _peak_to_peak_mv(context, lead) for lead in lead_list}
    measured = {lead: value for lead, value in values.items() if value is not None}
    disproving = {
        lead: value for lead, value in measured.items() if value >= threshold_mv
    }
    qualifying = {
        lead: value for lead, value in measured.items() if value < threshold_mv
    }

    # "Low voltage in all leads of a region" can be disproved by one reliable
    # lead. A positive call, however, requires every lead in that region.
    if disproving:
        status = "not_matched"
        missing: list[str] = []
    elif len(measured) == len(lead_list):
        status = "matched"
        missing = []
    else:
        status = "unavailable"
        missing = [
            f"lead.{lead}.qrs_peak_to_peak_mv"
            for lead in lead_list
            if values[lead] is None
        ]

    return RuleEvaluation(
        rule_id=rule_id,
        domain="voltage",
        status=status,
        statement_code=code if status == "matched" else None,
        statement=statement if status == "matched" else None,
        severity="abnormal" if status == "matched" else "normal",
        required_inputs=[f"lead.{lead}.qrs_peak_to_peak_mv" for lead in lead_list],
        missing_inputs=missing,
        evidence={
            "evaluates_code": code,
            "qrs_peak_to_peak_mv": values,
            "qualifying_leads": sorted(qualifying),
            "disproving_leads": sorted(disproving),
        },
        thresholds={"all_leads_lt_mv": threshold_mv},
        source=dict(SOURCE_AHA_LOW_VOLTAGE),
    )


def evaluate_low_voltage(context) -> list[RuleEvaluation]:
    return [
        _evaluate_region(
            context,
            leads=LIMB_LEADS,
            threshold_mv=0.5,
            code="low_qrs_voltage_limb_leads",
            rule_id="CLIN-VOLTAGE-LOW-LIMB-01",
            statement="Low QRS voltage in the limb leads",
        ),
        _evaluate_region(
            context,
            leads=PRECORDIAL_LEADS,
            threshold_mv=1.0,
            code="low_qrs_voltage_precordial_leads",
            rule_id="CLIN-VOLTAGE-LOW-PRECORDIAL-01",
            statement="Low QRS voltage in the precordial leads",
        ),
    ]

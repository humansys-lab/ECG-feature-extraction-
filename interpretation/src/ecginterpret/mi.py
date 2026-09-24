from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


INFERIOR_LEADS = ("II", "III", "aVF")
ANTERIOR_LEADS = ("V1", "V2", "V3", "V4")
LATERAL_LEADS = ("I", "aVL", "V5", "V6")
POSTERIOR_RECIPROCAL_LEADS = ("V1", "V2", "V3")
TERRITORY_LEADS = {
    "inferior": INFERIOR_LEADS,
    "anterior": ANTERIOR_LEADS,
    "lateral": LATERAL_LEADS,
    "anterolateral": ("I", "aVL", "V3", "V4", "V5", "V6"),
}

Q_INF_R_RATIO = 1.0 / 6.0
Q_LAT_AMP_MV = 0.10
Q_LAT_R_RATIO = 0.20
Q_ANT_AMP_MV = 0.07
Q_ANT_R_RATIO = 0.20
Q_MI_R_RATIO = 1.0 / 5.0


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _lead_param(representative_leads: Dict[str, Any], lead: str, name: str) -> Optional[float]:
    rep = representative_leads.get(lead)
    params = getattr(rep, "params", {}) if rep is not None else {}
    return _finite(params.get(name))


def _q_fact(representative_leads: Dict[str, Any], lead: str) -> Dict[str, Any]:
    q = _lead_param(representative_leads, lead, "q_amp_mv")
    r = _lead_param(representative_leads, lead, "r_amp_mv")
    q_duration = _lead_param(representative_leads, lead, "q_duration_ms")
    q_area = _lead_param(representative_leads, lead, "q_area_mv_ms")
    ratio = abs(q) / abs(r) if q is not None and r not in (None, 0.0) else None
    negative_q = q is not None and q < 0

    significant = False
    if negative_q and ratio is not None:
        if lead in INFERIOR_LEADS:
            significant = ratio > Q_INF_R_RATIO and (q_duration is None or q_duration >= 25.0)
        elif lead in LATERAL_LEADS:
            significant = abs(q) > Q_LAT_AMP_MV and ratio > Q_LAT_R_RATIO and (q_duration is None or q_duration >= 35.0)
        elif lead in ANTERIOR_LEADS:
            significant = (
                abs(q) > Q_ANT_AMP_MV
                and ratio > Q_ANT_R_RATIO
                and (q_duration is None or q_duration >= 30.0)
            )
    infarct_ratio = bool(
        negative_q
        and ratio is not None
        and ratio > Q_MI_R_RATIO
        and (q_duration is None or q_duration >= 35.0)
    )
    pediatric_large_q = bool(negative_q and (abs(q) >= 0.15 or infarct_ratio))

    return {
        "lead": lead,
        "q_amplitude_mV": q,
        "r_amplitude_mV": r,
        "q_duration_ms": q_duration,
        "q_area_mV_ms": q_area,
        "q_r_ratio": ratio,
        "significant_q": significant,
        "q_wave_mi_ratio": infarct_ratio,
        "pediatric_large_q": pediatric_large_q,
    }


def _ordered_present(leads: Iterable[str], present: Iterable[str]) -> List[str]:
    present_set = set(present)
    return [lead for lead in leads if lead in present_set]


def build_mi_evidence(
    *,
    representative_leads: Dict[str, Any],
    st_elevation_leads: Dict[str, float],
    st_depression_leads: Dict[str, float],
    stemi_codes: List[str],
    posterior_mi_suspected: bool,
    r_progression_class: str,
    is_pediatric: bool,
    initial_qrs_axis_deg: Optional[float] = None,
) -> Dict[str, Any]:
    all_leads = sorted(set().union(*TERRITORY_LEADS.values(), POSTERIOR_RECIPROCAL_LEADS))
    q_by_lead = {lead: _q_fact(representative_leads, lead) for lead in all_leads}

    territories: Dict[str, Any] = {}
    for territory, leads in TERRITORY_LEADS.items():
        significant_q_leads = [lead for lead in leads if q_by_lead[lead]["significant_q"]]
        ratio_leads = [lead for lead in leads if q_by_lead[lead]["q_wave_mi_ratio"]]
        large_q_leads = [lead for lead in leads if q_by_lead[lead]["pediatric_large_q"]]
        q_leads = large_q_leads if is_pediatric and len(large_q_leads) >= 2 else significant_q_leads
        st_elev = _ordered_present(leads, st_elevation_leads.keys())
        territories[territory] = {
            "leads": list(leads),
            "q_wave_leads": q_leads,
            "q_wave_count": len(q_leads),
            "q_wave_mi_ratio_leads": ratio_leads,
            "q_wave_mi_pattern": len(significant_q_leads) >= 2 or len(ratio_leads) >= 2,
            "pediatric_large_q_group": bool(is_pediatric and len(large_q_leads) >= 2),
            "st_elevation_leads": st_elev,
            "stemi_codes": list(stemi_codes),
        }

    posterior_depression = _ordered_present(POSTERIOR_RECIPROCAL_LEADS, st_depression_leads.keys())
    territories["posterior"] = {
        "leads": list(POSTERIOR_RECIPROCAL_LEADS),
        "posterior_mi_suspected": bool(posterior_mi_suspected),
        "reciprocal_depression_leads": posterior_depression,
        "r_progression_class": r_progression_class,
    }

    return {
        "available": True,
        "q_by_lead": q_by_lead,
        "territories": territories,
        "initial_qrs_axis_deg": initial_qrs_axis_deg,
        "is_pediatric": bool(is_pediatric),
    }


def build_culprit_artery_evidence(
    *,
    st_elevation_leads: Dict[str, float],
    st_depression_leads: Dict[str, float],
    available_extended_leads: List[str],
) -> Dict[str, object]:
    criteria: List[str] = []
    culprit = None
    if st_elevation_leads.get("III", 0.0) > st_elevation_leads.get("II", 0.0):
        criteria.append("III_greater_than_II")
    if "aVL" in st_depression_leads:
        criteria.append("aVL_reciprocal_depression")
    if {"III_greater_than_II", "aVL_reciprocal_depression"}.issubset(set(criteria)):
        culprit = "RCA"
    return {
        "available": bool(criteria),
        "culprit": culprit,
        "criteria": criteria,
        "available_extended_leads": list(available_extended_leads),
        "limitations": [] if available_extended_leads else ["no_V4R_V7_V8_V9"],
    }


def _suppression_reasons(bundle_branch_block: Optional[str], pacing_context: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if bundle_branch_block == "LBBB":
        reasons.append("lbbb_secondary_repolarization")
    continuous_ventricular_pacing = bool(pacing_context.get("continuous_ventricular_pacing")) or bool(
        pacing_context.get("continuous_pacing")
        and pacing_context.get("ventricular_pacing_present")
    ) or bool(pacing_context.get("suppress_further_rhythm_interpretation"))
    if continuous_ventricular_pacing:
        reasons.append("continuous_ventricular_pacing")
    return reasons


def _candidate(
    *,
    code: str,
    category: str,
    territory: str,
    severity: str,
    probability: str,
    evidence: Dict[str, Any],
    suppressed_by: List[str],
) -> Dict[str, Any]:
    return {
        "code": code,
        "category": category,
        "territory": territory,
        "severity": severity,
        "probability": probability,
        "evidence": evidence,
        "suppressed_by": list(suppressed_by),
        "final": not suppressed_by,
    }


def build_mi_statement_candidates(
    evidence: Dict[str, Any],
    *,
    bundle_branch_block: Optional[str],
    pacing_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    suppressed_by = _suppression_reasons(bundle_branch_block, pacing_context)
    is_pediatric = bool(evidence.get("is_pediatric"))
    territories = evidence.get("territories") or {}
    statements: List[Dict[str, Any]] = []

    for territory, terr in territories.items():
        if territory == "posterior":
            continue
        q_leads = list(terr.get("q_wave_leads") or [])
        ratio_leads = list(terr.get("q_wave_mi_ratio_leads") or [])
        st_elev = list(terr.get("st_elevation_leads") or [])
        if st_elev:
            statements.append(_candidate(
                code="acute_mi_st_elevation",
                category="mi",
                territory=territory,
                severity="acute",
                probability="suspected",
                evidence={"st_elevation_leads": st_elev, "stemi_codes": list(terr.get("stemi_codes") or [])},
                suppressed_by=suppressed_by,
            ))
        if bool(terr.get("q_wave_mi_pattern")):
            statements.append(_candidate(
                code="old_mi_q_wave",
                category="mi",
                territory=territory,
                severity="old_or_age_indeterminate",
                probability="probable",
                evidence={"q_wave_leads": q_leads, "q_wave_mi_ratio_leads": ratio_leads},
                suppressed_by=suppressed_by,
            ))
        if is_pediatric and bool(terr.get("pediatric_large_q_group")):
            statements.append(_candidate(
                code="pediatric_q_wave_abnormality",
                category="pediatric_morphology",
                territory=territory,
                severity="abnormal",
                probability="possible",
                evidence={"q_wave_leads": q_leads},
                suppressed_by=suppressed_by,
            ))

    posterior = territories.get("posterior") or {}
    if bool(posterior.get("posterior_mi_suspected")):
        statements.append(_candidate(
            code="posterior_mi_pattern",
            category="mi",
            territory="posterior",
            severity="possible",
            probability="suspected",
            evidence={
                "reciprocal_depression_leads": list(posterior.get("reciprocal_depression_leads") or []),
                "r_progression_class": posterior.get("r_progression_class"),
            },
            suppressed_by=suppressed_by,
        ))

    return statements

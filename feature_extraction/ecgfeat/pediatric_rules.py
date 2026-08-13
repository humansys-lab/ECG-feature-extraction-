from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class PediatricBin:
    name: str
    min_years: float
    max_years: float


AGE_BINS = (
    PediatricBin("0_23h", 0.0, 1.0 / 365.25),
    PediatricBin("1_3d", 1.0 / 365.25, 4.0 / 365.25),
    PediatricBin("4_6d", 4.0 / 365.25, 7.0 / 365.25),
    PediatricBin("7_29d", 7.0 / 365.25, 30.0 / 365.25),
    PediatricBin("1_11m", 30.0 / 365.25, 1.0),
    PediatricBin("1_2y", 1.0, 3.0),
    PediatricBin("3_4y", 3.0, 5.0),
    PediatricBin("5_7y", 5.0, 8.0),
    PediatricBin("8_11y", 8.0, 12.0),
    PediatricBin("12_15y", 12.0, 16.0),
)


REQUIRED_PEDIATRIC_CRITERIA = (
    "rvh_r_v1_98p_mv",
    "rvh_r_v2_98p_mv",
    "rvh_s_v6_98p_mv",
    "rvh_rs_ratio_v1_98p",
    "lvh_r_i_98p_mv",
    "lvh_r_avl_98p_mv",
    "lvh_r_v5_98p_mv",
    "lvh_r_v6_98p_mv",
    "lvh_s_v1_98p_mv",
    "lvh_s_v2_98p_mv",
    "lvh_sv1_rv6_98p_mv",
)


PEDS_BVH_RS_SUM_MV = 6.0
# Katz–Wachtel biventricular precordial signs (each enforced below).
PEDS_BVH_R_V1_MV = 1.0       # tall R in V1 (right voltage) paired with LVH
PEDS_BVH_R_V6_MV = 1.0       # tall R in V6 (left voltage) paired with a right sign
PEDS_BVH_Q_V6_AMP_MV = 0.07  # deep septal Q in V6 (amplitude), with duration below
PEDS_BVH_Q_V6_DUR_MS = 10.0  # wide septal Q in V6 (duration), with amplitude above


# Values are 98th percentile limits in mV unless the criterion name says ratio.
# Philips Appendix A reports RV1/SV1/RV6/SV6/SV1+RV6 and R/S ratios in mm at
# 10 mm/mV. Required criteria not exposed by the local DXL Appendix A sources
# remain present with None values so later consumers do not use fabricated data.
# Combined task bins use the maximum source threshold across narrower rows.
VOLTAGE_98P_MV: Dict[str, Dict[str, Optional[float]]] = {
    "0_23h": {
        "rvh_r_v1_98p_mv": 2.6,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 0.95,
        "rvh_rs_ratio_v1_98p": None,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 1.1,
        "lvh_s_v1_98p_mv": 2.3,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 2.8,
    },
    "1_3d": {
        "rvh_r_v1_98p_mv": 2.7,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 1.0,
        "rvh_rs_ratio_v1_98p": None,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 1.2,
        "lvh_s_v1_98p_mv": 2.1,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 2.9,
    },
    "4_6d": {
        "rvh_r_v1_98p_mv": 2.4,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 1.0,
        "rvh_rs_ratio_v1_98p": None,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 1.2,
        "lvh_s_v1_98p_mv": 1.7,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 2.45,
    },
    "7_29d": {
        "rvh_r_v1_98p_mv": 2.1,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 1.0,
        "rvh_rs_ratio_v1_98p": None,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 1.65,
        "lvh_s_v1_98p_mv": 1.1,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 2.1,
    },
    "1_11m": {
        "rvh_r_v1_98p_mv": 2.0,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 1.0,
        "rvh_rs_ratio_v1_98p": None,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 2.25,
        "lvh_s_v1_98p_mv": 1.8,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 3.5,
    },
    "1_2y": {
        "rvh_r_v1_98p_mv": 1.7,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 0.65,
        "rvh_rs_ratio_v1_98p": 4.3,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 2.25,
        "lvh_s_v1_98p_mv": 2.1,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 3.9,
    },
    "3_4y": {
        "rvh_r_v1_98p_mv": 1.8,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 0.5,
        "rvh_rs_ratio_v1_98p": 2.8,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 2.45,
        "lvh_s_v1_98p_mv": 2.1,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 4.2,
    },
    "5_7y": {
        "rvh_r_v1_98p_mv": 1.4,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 0.4,
        "rvh_rs_ratio_v1_98p": 2.0,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 2.65,
        "lvh_s_v1_98p_mv": 2.4,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 4.7,
    },
    "8_11y": {
        "rvh_r_v1_98p_mv": 1.2,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 0.4,
        "rvh_rs_ratio_v1_98p": 1.8,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 2.55,
        "lvh_s_v1_98p_mv": 2.5,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 4.55,
    },
    "12_15y": {
        "rvh_r_v1_98p_mv": 1.0,
        "rvh_r_v2_98p_mv": None,
        "rvh_s_v6_98p_mv": 0.4,
        "rvh_rs_ratio_v1_98p": 1.7,
        "lvh_r_i_98p_mv": None,
        "lvh_r_avl_98p_mv": None,
        "lvh_r_v5_98p_mv": None,
        "lvh_r_v6_98p_mv": 2.3,
        "lvh_s_v1_98p_mv": 2.1,
        "lvh_s_v2_98p_mv": None,
        "lvh_sv1_rv6_98p_mv": 4.1,
    },
}


def pediatric_age_bin(age_years: Optional[float]) -> Optional[str]:
    if age_years is None or age_years < 0.0 or age_years >= 16.0:
        return None
    for item in AGE_BINS:
        if item.min_years <= age_years < item.max_years:
            return item.name
    return None


def pediatric_voltage_threshold(
    age_years: float,
    sex: Optional[str],
    criterion: str,
) -> Optional[float]:
    # Accepted for API compatibility; these Appendix A voltage limits are not sex-specific.
    del sex
    if criterion not in REQUIRED_PEDIATRIC_CRITERIA:
        raise KeyError(f"unknown pediatric voltage criterion: {criterion}")
    bin_name = pediatric_age_bin(age_years)
    if bin_name is None:
        return None
    return VOLTAGE_98P_MV.get(bin_name, {}).get(criterion)


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _mapping_value(source: object, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    getter = getattr(source, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(source, key, None)


def _lead_value(
    representative_leads: Dict[str, object],
    lead: str,
    key: str,
) -> Optional[float]:
    rep = representative_leads.get(lead)
    if rep is None:
        return None
    params = _mapping_value(rep, "params")
    source = params if isinstance(params, Mapping) or callable(getattr(params, "get", None)) else rep
    reliable_for_qrs = _mapping_value(source, "reliable_for_qrs")
    if reliable_for_qrs is False:
        return None
    return _finite_float(_mapping_value(source, key))


def _severity_from_count(count: int) -> Optional[str]:
    if count <= 0:
        return None
    if count == 1:
        return "consider"
    if count == 2:
        return "probable"
    return "definitive"


def build_pediatric_hypertrophy_evidence(
    *,
    representative_leads: Dict[str, object],
    age_years: Optional[float],
    sex: Optional[str],
    qrs_axis_class: Optional[str],
    bundle_branch_block: Optional[str],
) -> Dict[str, object]:
    rvh: Dict[str, object] = {"criteria": [], "class": None, "bypassed_by": []}
    lvh: Dict[str, object] = {"criteria": [], "class": None, "bypassed_by": []}
    if age_years is None:
        return {"rvh": rvh, "lvh": lvh, "bvh": {"suspected": False, "criteria": []}}

    if bundle_branch_block == "RBBB":
        rvh["bypassed_by"].append("RBBB")
    if bundle_branch_block in ("RBBB", "LBBB"):
        lvh["bypassed_by"].append(bundle_branch_block)

    if not rvh["bypassed_by"]:
        r_v1 = _lead_value(representative_leads, "V1", "r_amp_mv")
        threshold = pediatric_voltage_threshold(age_years, sex, "rvh_r_v1_98p_mv")
        if r_v1 is not None and threshold is not None and r_v1 >= threshold:
            rvh["criteria"].append("rvh_r_v1_98p_mv")
        if qrs_axis_class in ("RAD", "borderline_RAD"):
            rvh["criteria"].append("right_axis_support")
        rvh["class"] = _severity_from_count(len(rvh["criteria"]))

    if not lvh["bypassed_by"]:
        r_v6 = _lead_value(representative_leads, "V6", "r_amp_mv")
        threshold = pediatric_voltage_threshold(age_years, sex, "lvh_r_v6_98p_mv")
        if r_v6 is not None and threshold is not None and r_v6 >= threshold:
            lvh["criteria"].append("lvh_r_v6_98p_mv")
        lvh["class"] = _severity_from_count(len(lvh["criteria"]))

    bvh_criteria = ["rvh_lvh_combination"] if rvh["class"] and lvh["class"] else []
    rs_sum_count = 0
    for lead in ("V2", "V3", "V4"):
        r_amp = _lead_value(representative_leads, lead, "r_amp_mv")
        s_amp = _lead_value(representative_leads, lead, "s_amp_mv")
        if r_amp is not None and s_amp is not None and r_amp + abs(s_amp) > PEDS_BVH_RS_SUM_MV:
            rs_sum_count += 1
    if rs_sum_count >= 2:
        bvh_criteria.append("bvh_rs_sum_v2_v3_v4_mv")

    # Katz–Wachtel precordial biventricular signs.  A left-sided sign is only
    # counted as biventricular when a right-sided sign co-exists (and vice
    # versa), so an isolated tall precordial R does not by itself flag BVH.
    r_v1 = _lead_value(representative_leads, "V1", "r_amp_mv")
    r_v6 = _lead_value(representative_leads, "V6", "r_amp_mv")
    q_v6_amp = _lead_value(representative_leads, "V6", "q_amp_mv")
    q_v6_dur = _lead_value(representative_leads, "V6", "q_duration_ms")

    tall_r_v1 = r_v1 is not None and r_v1 >= PEDS_BVH_R_V1_MV
    tall_r_v6 = r_v6 is not None and r_v6 >= PEDS_BVH_R_V6_MV
    septal_q_v6 = (
        q_v6_amp is not None and abs(q_v6_amp) >= PEDS_BVH_Q_V6_AMP_MV
        and q_v6_dur is not None and q_v6_dur >= PEDS_BVH_Q_V6_DUR_MS
    )
    right_sign = bool(rvh["class"]) or tall_r_v1

    # Tall R in V1 (right voltage) with confirmed LVH (left) is biventricular.
    if tall_r_v1 and bool(lvh["class"]):
        bvh_criteria.append("bvh_r_v1_with_lvh_mv")
    # Left-precordial signs count when a right-sided sign co-exists.
    if right_sign and tall_r_v6:
        bvh_criteria.append("bvh_r_v6_mv")
    if right_sign and septal_q_v6:
        bvh_criteria.append("bvh_q_v6_septal")

    return {
        "rvh": rvh,
        "lvh": lvh,
        "bvh": {
            "suspected": bool(bvh_criteria),
            "criteria": bvh_criteria,
        },
    }


def build_pediatric_repolarization_candidates(
    *,
    st_depression_leads: Dict[str, float],
    inverted_t_leads: List[str],
    hypertrophy_evidence: Dict[str, object],
    bundle_branch_block: Optional[str],
) -> List[Dict[str, object]]:
    suppressed_by: List[str] = []
    if bundle_branch_block:
        suppressed_by.append(bundle_branch_block)

    lvh = hypertrophy_evidence.get("lvh") if isinstance(hypertrophy_evidence, dict) else None
    rvh = hypertrophy_evidence.get("rvh") if isinstance(hypertrophy_evidence, dict) else None
    bvh = hypertrophy_evidence.get("bvh") if isinstance(hypertrophy_evidence, dict) else None
    if isinstance(lvh, dict) and lvh.get("class"):
        suppressed_by.append("LVH")
    if isinstance(rvh, dict) and rvh.get("class"):
        suppressed_by.append("RVH")
    if isinstance(bvh, dict) and bvh.get("suspected"):
        suppressed_by.append("BVH")

    if not st_depression_leads and not inverted_t_leads:
        return []

    return [
        {
            "code": "secondary_st_t_abnormality" if suppressed_by else "primary_st_t_abnormality",
            "category": "pediatric_repolarization",
            "severity": "abnormal",
            "evidence": {
                "st_depression_leads": dict(st_depression_leads),
                "inverted_t_leads": list(inverted_t_leads),
            },
            "suppressed_by": suppressed_by,
            "final": not suppressed_by,
        }
    ]


def validate_pediatric_tables() -> None:
    for age_bin in AGE_BINS:
        if age_bin.name not in VOLTAGE_98P_MV:
            raise ValueError(f"missing pediatric voltage table for {age_bin.name}")
        missing = [
            key
            for key in REQUIRED_PEDIATRIC_CRITERIA
            if key not in VOLTAGE_98P_MV[age_bin.name]
        ]
        if missing:
            raise ValueError(f"missing pediatric criteria for {age_bin.name}: {missing}")

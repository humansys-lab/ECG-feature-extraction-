from __future__ import annotations

from math import isfinite
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .context import GlasgowContext
from .models import RuleEvaluation, RuleSpec


_SOURCE = "Physician's Guide section 4.1, PDF pages 10-13"
_LIMB_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _lead(context: GlasgowContext, lead: str, key: str, default: Optional[float] = None) -> Optional[float]:
    fact = context.leads.get(lead)
    value = _number(fact.measurements.get(key)) if fact is not None else None
    return default if value is None else value


def _area(context: GlasgowContext, lead: str) -> Optional[float]:
    fact = context.leads.get(lead)
    if fact is None:
        return None
    for key in ("qrs_signed_area_matrix", "qrs_area_matrix"):
        value = _number(fact.measurements.get(key))
        if value is not None:
            return value
    return None


def _axis(context: GlasgowContext, key: str) -> Optional[float]:
    return _number(context.global_value(key))


def _all_present(*values: Optional[float]) -> bool:
    return all(value is not None for value in values)


def criterion(name: str, matched: bool, **operands: Any) -> Dict[str, Any]:
    return {"criterion": name, "matched": bool(matched), "operands": operands}


def _truth(fact: Dict[str, Any]) -> bool:
    return bool(fact["matched"])


def _validity_criteria(context: GlasgowContext) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    chest = ("V1", "V2", "V3", "V4", "V5", "V6")
    for index in range(1, len(chest) - 1):
        previous, current, following = chest[index - 1 : index + 2]
        areas = (_area(context, previous), _area(context, current), _area(context, following))
        a_negative_between_positive = bool(
            _all_present(*areas) and areas[1] < 0.0 and areas[0] > 0.0 and areas[2] > 0.0
        )
        same_sign = bool(
            _all_present(*areas)
            and all(value > 0.0 for value in areas)
            or _all_present(*areas) and all(value < 0.0 for value in areas)
        )
        a_small_same_sign = bool(
            same_sign
            and abs(float(areas[1])) < 0.25 * abs(float(areas[0]))
            and abs(float(areas[1])) < 0.25 * abs(float(areas[2]))
        )
        facts[f"A_{current}"] = criterion(
            f"validity_A_{current}",
            a_negative_between_positive or a_small_same_sign,
            previous=previous,
            current=current,
            following=following,
            qrs_area_matrix=list(areas),
            negative_between_positive=a_negative_between_positive,
            below_25_percent_same_sign=a_small_same_sign,
        )
        facts[f"B_{current}"] = criterion(
            f"validity_B_{current}",
            bool(_all_present(*areas) and all(abs(float(value)) > 500.0 for value in areas)),
            leads=[previous, current, following],
            qrs_area_matrix=list(areas),
            threshold_abs_qrs_area_matrix=500.0,
        )

    r_v1, r_v2, r_v3 = (_lead(context, lead, "r_amp_mv") for lead in ("V1", "V2", "V3"))
    tp_v1, tp_v2, tp_v3 = (
        _lead(context, lead, "t_positive_amp_mv") for lead in ("V1", "V2", "V3")
    )
    tn_v1, tn_v2, tn_v3 = (
        _lead(context, lead, "t_negative_amp_mv") for lead in ("V1", "V2", "V3")
    )
    rp_v1 = _lead(context, "V1", "r_prime_amp_mv", 0.0)
    rp_v2 = _lead(context, "V2", "r_prime_amp_mv", 0.0)
    rp_v3 = _lead(context, "V3", "r_prime_amp_mv", 0.0)
    q_v1 = _lead(context, "V1", "q_amp_mv", 0.0)
    q_v2 = _lead(context, "V2", "q_amp_mv", 0.0)
    s_v1, s_v2, s_v3 = (_lead(context, lead, "s_amp_mv", 0.0) for lead in ("V1", "V2", "V3"))

    facts["C"] = criterion(
        "validity_C",
        bool(
            _all_present(r_v1, r_v2, tp_v1, tp_v2, tp_v3)
            and r_v2 + 0.025 < r_v1
            and tp_v1 > tp_v2 + 0.025
            and tp_v3 > tp_v2 + 0.025
            and tp_v2 > 0.0
        ),
        r_v1_mv=r_v1,
        r_v2_mv=r_v2,
        t_positive_v1_mv=tp_v1,
        t_positive_v2_mv=tp_v2,
        t_positive_v3_mv=tp_v3,
        margin_mv=0.025,
    )
    facts["D"] = criterion(
        "validity_D",
        bool(
            _all_present(r_v1, r_v2, r_v3, tn_v2, tp_v2)
            and r_v1 - r_v2 > 0.2
            and r_v3 - r_v2 > 0.2
            and abs(tn_v2) > tp_v2
        ),
        r_v1_mv=r_v1,
        r_v2_mv=r_v2,
        r_v3_mv=r_v3,
        t_positive_v2_mv=tp_v2,
        t_negative_v2_mv=tn_v2,
        r_margin_mv=0.2,
    )
    facts["E"] = criterion(
        "validity_E",
        bool(
            _all_present(tp_v1, tp_v2, tp_v3, tn_v1, tn_v2, tn_v3)
            and tp_v1 > abs(tn_v1) + 0.025
            and abs(tn_v2) > abs(tp_v2) + 0.025
            and tp_v3 > abs(tn_v3) + 0.025
        ),
        t_positive_mv={"V1": tp_v1, "V2": tp_v2, "V3": tp_v3},
        t_negative_mv={"V1": tn_v1, "V2": tn_v2, "V3": tn_v3},
        margin_mv=0.025,
    )
    facts["F"] = criterion(
        "validity_F",
        bool(
            _all_present(r_v1, r_v2, r_v3, rp_v2)
            and r_v1 > r_v2 + 0.4
            and r_v3 > r_v2 + 0.4
            and abs(rp_v2) <= 1e-9
        ),
        r_mv={"V1": r_v1, "V2": r_v2, "V3": r_v3},
        r_prime_v2_mv=rp_v2,
        margin_mv=0.4,
    )
    areas_v1_v4 = [_area(context, lead) for lead in ("V1", "V2", "V3", "V4")]
    facts["G"] = criterion(
        "validity_G",
        bool(
            _all_present(*areas_v1_v4)
            and areas_v1_v4[0] < 0.0 < areas_v1_v4[1]
            and areas_v1_v4[2] < 0.0 < areas_v1_v4[3]
        ),
        qrs_area_matrix=dict(zip(("V1", "V2", "V3", "V4"), areas_v1_v4)),
    )
    facts["H"] = criterion(
        "validity_H",
        bool(
            (_all_present(r_v2, r_v3) and r_v2 > r_v3 + 0.200)
            or (
                _all_present(s_v1, s_v2, s_v3)
                and abs(s_v1) > 3.0 * abs(s_v2)
                and abs(s_v3) > 3.0 * abs(s_v2)
            )
        ),
        r_v2_mv=r_v2,
        r_v3_mv=r_v3,
        s_mv={"V1": s_v1, "V2": s_v2, "V3": s_v3},
        normalized_r_margin_mv=0.200,
        guide_bare_constant=200,
    )
    max_r = {
        lead: max(abs(_lead(context, lead, "r_amp_mv", 0.0) or 0.0), abs(_lead(context, lead, "r_prime_amp_mv", 0.0) or 0.0))
        for lead in ("V1", "V2", "V3", "V4")
    }
    facts["I"] = criterion(
        "validity_I",
        bool(
            max_r["V2"] > 2.5 * max_r["V1"]
            and max_r["V4"] > 2.5 * max_r["V3"]
            and max_r["V2"] > max_r["V3"] + 0.300
        ),
        max_r_r_prime_mv=max_r,
        normalized_margin_mv=0.300,
        guide_bare_constant=300,
    )
    no_q_v2 = q_v2 is not None and abs(q_v2) <= 1e-9
    rp_pattern = bool(abs(rp_v1 or 0.0) > 0.0 and abs(rp_v2 or 0.0) > 0.0 and abs(rp_v3 or 0.0) <= 1e-9)
    facts["J"] = criterion(
        "validity_J",
        no_q_v2 and not rp_pattern,
        q_v2_mv=q_v2,
        r_prime_mv={"V1": rp_v1, "V2": rp_v2, "V3": rp_v3},
    )
    facts["K"] = criterion(
        "validity_K",
        bool(q_v1 is not None and not (q_v1 < 0.0 and abs(q_v1) > 0.075)),
        q_v1_mv=q_v1,
        threshold_abs_q_mv=0.075,
    )
    return facts


def _net_t(context: GlasgowContext, lead: str) -> Optional[float]:
    positive = _lead(context, lead, "t_positive_amp_mv")
    negative = _lead(context, lead, "t_negative_amp_mv")
    if positive is None or negative is None:
        return None
    return positive - abs(negative)


def _reversal_criteria(context: GlasgowContext) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    p_axis, qrs_axis, t_axis = (
        _axis(context, key) for key in ("p_axis_deg", "qrs_axis_deg", "t_axis_deg")
    )
    area_i = _area(context, "I")
    q_i = _lead(context, "I", "q_amp_mv", 0.0)
    r_i = _lead(context, "I", "r_amp_mv", 0.0)
    rp_i = _lead(context, "I", "r_prime_amp_mv", 0.0)
    s_i = _lead(context, "I", "s_amp_mv", 0.0)
    r_duration_i = _lead(context, "I", "r_duration_ms")
    q_duration_i = _lead(context, "I", "q_duration_ms")

    facts["A"] = criterion("reversal_A", bool(context.rhythm.get("p_wave_flag")), p_wave_flag=context.rhythm.get("p_wave_flag"))
    facts["B"] = criterion(
        "reversal_B",
        bool(p_axis is not None and (85.0 < p_axis <= 180.0 or -180.0 <= p_axis < -85.0)),
        p_axis_deg=p_axis,
    )
    facts["C"] = criterion(
        "reversal_C",
        bool(
            qrs_axis is not None
            and (85.0 < qrs_axis <= 180.0 or -180.0 <= qrs_axis < -85.0)
            and (
                area_i is not None and area_i < 0.0
                or _all_present(r_duration_i, q_duration_i)
                and r_duration_i >= 40.0 and q_duration_i >= 40.0
            )
        ),
        qrs_axis_deg=qrs_axis,
        qrs_area_i_matrix=area_i,
        r_duration_i_ms=r_duration_i,
        q_duration_i_ms=q_duration_i,
    )
    pp_v6 = _lead(context, "V6", "qrs_peak_to_peak_mv")
    area_v6 = _area(context, "V6")
    ppos_v6 = _lead(context, "V6", "p_positive_amp_mv")
    pneg_v6 = _lead(context, "V6", "p_negative_amp_mv")
    facts["D"] = criterion(
        "reversal_D",
        bool(
            _all_present(pp_v6, area_v6, ppos_v6, pneg_v6, r_i, q_i)
            and pp_v6 > 0.5 and area_v6 > 0.0 and ppos_v6 > abs(pneg_v6)
            and (r_i < 0.2 or q_i < 0.0)
        ),
        peak_to_peak_v6_mv=pp_v6,
        qrs_area_v6_matrix=area_v6,
        p_positive_v6_mv=ppos_v6,
        p_negative_v6_mv=pneg_v6,
        r_i_mv=r_i,
        q_i_mv=q_i,
    )
    chest = ("V3", "V4", "V5", "V6")
    r_chest = [_lead(context, lead, "r_amp_mv") for lead in chest]
    area_chest = [_area(context, lead) for lead in chest]
    descending_r = bool(_all_present(*r_chest) and all(0.0 <= r_chest[i + 1] <= r_chest[i] for i in range(3)))
    all_small_r = bool(_all_present(*r_chest) and all(r <= 0.1 for r in r_chest))
    increasing_area = bool(
        _all_present(*area_chest)
        and all(area_chest[i] < area_chest[i + 1] < 100.0 for i in range(3))
    )
    facts["E"] = criterion(
        "reversal_E",
        bool(
            (descending_r or all_small_r)
            and increasing_area
            and _all_present(pp_v6, r_chest[-1], qrs_axis)
            and pp_v6 < 0.8 and r_chest[-1] < 0.1 and qrs_axis > 60.0
        ),
        r_mv=dict(zip(chest, r_chest)),
        qrs_area_matrix=dict(zip(chest, area_chest)),
        descending_r=descending_r,
        all_r_at_most_0_1_mv=all_small_r,
        increasing_area_below_100=increasing_area,
        peak_to_peak_v6_mv=pp_v6,
        qrs_axis_deg=qrs_axis,
    )
    s_v6 = _lead(context, "V6", "s_amp_mv", 0.0)
    r_v6 = _lead(context, "V6", "r_amp_mv", 0.0)
    st_i, st_v6 = (_lead(context, lead, "st_amp_mv") for lead in ("I", "V6"))
    ti, tv6 = (_net_t(context, lead) for lead in ("I", "V6"))
    f_i = bool(
        _all_present(q_i, r_i, rp_i, s_i)
        and (
            abs(q_i) > r_i >= rp_i
            or abs(s_i) > rp_i and abs(q_i) <= 1e-9 and abs(s_i) > r_i + 0.100
        )
    )
    f_v6 = bool(
        _all_present(s_v6, r_v6)
        and abs(s_v6) > 1e-9
        and (abs(s_v6) > 0.25 or abs(r_v6 / s_v6) >= 2.0)
    )
    opposite_st_t = bool(
        _all_present(st_i, st_v6, ti, tv6)
        and st_i * st_v6 < 0.0 and ti * tv6 < 0.0
    )
    facts["F"] = criterion(
        "reversal_F",
        f_i and f_v6 and opposite_st_t,
        lead_i_morphology=f_i,
        lead_v6_morphology=f_v6,
        opposite_st_and_t=opposite_st_t,
        normalized_margin_mv=0.100,
        guide_bare_constant=100,
    )

    def per_lead(letter: str, lead: str) -> Dict[str, Any]:
        r = _lead(context, lead, "r_amp_mv", 0.0)
        rp = _lead(context, lead, "r_prime_amp_mv", 0.0)
        s = _lead(context, lead, "s_amp_mv", 0.0)
        sp = _lead(context, lead, "s_prime_amp_mv", 0.0)
        q = _lead(context, lead, "q_amp_mv", 0.0)
        tp = _lead(context, lead, "t_positive_amp_mv", 0.0)
        tn = _lead(context, lead, "t_negative_amp_mv", 0.0)
        pp = _lead(context, lead, "p_positive_amp_mv", 0.0)
        checks = {
            "G": criterion(f"reversal_G_{lead}", abs(r) < 0.135 and abs(rp) < 0.135, r_mv=r, r_prime_mv=rp),
            "H": criterion(f"reversal_H_{lead}", abs(s) < 0.05 and abs(sp) < 0.05, s_mv=s, s_prime_mv=sp),
            "J": criterion(f"reversal_J_{lead}", abs(q) < 0.06, q_mv=q),
            "L": criterion(f"reversal_L_{lead}", tp + abs(tn) < 0.07, t_positive_mv=tp, t_negative_mv=tn),
            "O": criterion(f"reversal_O_{lead}", pp < 0.075, p_positive_mv=pp),
        }
        return checks[letter]

    for lead in ("I", "II", "III"):
        for letter in ("G", "H", "J", "L", "O"):
            facts[f"{letter}_{lead}"] = per_lead(letter, lead)

    area_ii, area_iii = _area(context, "II"), _area(context, "III")
    facts["K"] = criterion(
        "reversal_K",
        bool(_all_present(area_i, area_ii, area_iii) and abs(area_i + area_iii) < abs(area_ii) + 50.0),
        qrs_area_matrix={"I": area_i, "II": area_ii, "III": area_iii},
        margin=50.0,
    )
    facts["M"] = criterion(
        "reversal_M",
        bool(_all_present(area_i, area_ii, area_iii) and abs(area_ii - area_i) < abs(area_iii) + 50.0),
        qrs_area_matrix={"I": area_i, "II": area_ii, "III": area_iii},
        margin=50.0,
    )
    facts["N"] = criterion(
        "reversal_N",
        bool(
            t_axis is not None and (90.0 < t_axis <= 180.0 or -180.0 <= t_axis <= -90.0)
            and (_lead(context, "I", "p_negative_amp_mv") or 0.0) < -0.1
            and area_i is not None and area_i < -500.0
            and (_lead(context, "I", "t_negative_amp_mv") or 0.0) < -0.05
        ),
        t_axis_deg=t_axis,
        p_negative_i_mv=_lead(context, "I", "p_negative_amp_mv"),
        qrs_area_i_matrix=area_i,
        t_negative_i_mv=_lead(context, "I", "t_negative_amp_mv"),
    )
    facts["P"] = criterion(
        "reversal_P",
        bool(_all_present(area_i, area_ii, area_iii) and abs(area_ii - area_iii) < abs(area_i) + 50.0),
        qrs_area_matrix={"I": area_i, "II": area_ii, "III": area_iii},
        margin=50.0,
    )
    p_negative_sum = sum(abs(_lead(context, lead, "p_negative_amp_mv", 0.0) or 0.0) for lead in ("I", "II", "III"))
    heart_rate = _number(context.global_value("heart_rate_bpm"))
    facts["Q"] = criterion(
        "reversal_Q",
        bool(
            _all_present(p_axis, qrs_axis, t_axis, heart_rate)
            and -180.0 < p_axis <= -90.0
            and -90.0 <= qrs_axis < -30.0
            and -90.0 <= t_axis < 0.0
            and p_negative_sum > 0.200
            and heart_rate < 120.0
        ),
        p_axis_deg=p_axis,
        qrs_axis_deg=qrs_axis,
        t_axis_deg=t_axis,
        sum_abs_p_negative_limb_mv=p_negative_sum,
        normalized_threshold_mv=0.200,
        guide_bare_constant=200,
        heart_rate_bpm=heart_rate,
    )
    return facts


def _result(
    rule_id: str,
    statement: str,
    matched: bool,
    criteria: Sequence[Dict[str, Any]],
    excluded_leads: Sequence[str],
) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=rule_id,
        evaluation_status="matched" if matched else "not_matched",
        resolution_status="advisory" if matched else "no_statement",
        statement=statement,
        evidence={
            "criteria": list(criteria),
            "excluded_leads": list(excluded_leads) if matched else [],
        },
        thresholds={
            "amplitude_unit": "mV",
            "qrs_area_unit": "uV*ms/20",
            "bare_amplitude_constants_normalized_to_mV": True,
        },
        fidelity="glasgow_explicit",
        source={"kind": "glasgow_guide", "reference": _SOURCE},
    )


def evaluate_faulty_vn(context: GlasgowContext) -> RuleEvaluation:
    facts = _validity_criteria(context)
    matches: List[str] = []
    evidence: List[Dict[str, Any]] = []
    chest = ("V1", "V2", "V3", "V4", "V5", "V6")
    for index in range(1, 5):
        lead = chest[index]
        previous, following = chest[index - 1], chest[index + 1]
        current_pp = _lead(context, lead, "qrs_peak_to_peak_mv")
        previous_pp = _lead(context, previous, "qrs_peak_to_peak_mv")
        following_pp = _lead(context, following, "qrs_peak_to_peak_mv")
        t_positive = _lead(context, lead, "t_positive_amp_mv")
        t_negative = _lead(context, lead, "t_negative_amp_mv")
        amplitude = bool(
            _all_present(current_pp, previous_pp, following_pp)
            and (
                current_pp < 0.35 and current_pp < previous_pp / 3.0 and current_pp < following_pp / 3.0
                or current_pp < 0.5 and current_pp < previous_pp / 5.0 and current_pp < following_pp / 5.0
            )
        )
        t_small = bool(_all_present(t_positive, t_negative) and t_positive < 0.10 and t_negative > -0.10)
        item = criterion(
            f"statement_1_{lead}",
            amplitude and t_small,
            lead=lead,
            adjacent_leads=[previous, following],
            peak_to_peak_mv=[previous_pp, current_pp, following_pp],
            t_positive_mv=t_positive,
            t_negative_mv=t_negative,
        )
        evidence.append(item)
        if _truth(item):
            matches.append(lead)
    statement = "Possible faulty Vn – omitted from analysis"
    if matches:
        statement = f"Possible faulty {', '.join(matches)} – omitted from analysis"
    return _result("GAN-04.01-01", statement, bool(matches), evidence, matches)


def evaluate_faulty_v6(context: GlasgowContext) -> RuleEvaluation:
    pp_v5 = _lead(context, "V5", "qrs_peak_to_peak_mv")
    pp_v6 = _lead(context, "V6", "qrs_peak_to_peak_mv")
    p_v6 = _lead(context, "V6", "p_positive_amp_mv", 0.0)
    area_v5, area_v6 = _area(context, "V5"), _area(context, "V6")
    fact = criterion(
        "statement_2_V6",
        bool(
            _all_present(pp_v5, pp_v6)
            and (
                pp_v6 < 0.3 and pp_v6 < pp_v5 / 3.0
                or pp_v6 < 0.5 and pp_v6 < pp_v5 / 6.0
            )
            or _all_present(p_v6, area_v5, area_v6)
            and abs(p_v6) <= 1e-9 and area_v6 < -200.0 and area_v5 > 200.0
        ),
        peak_to_peak_v5_mv=pp_v5,
        peak_to_peak_v6_mv=pp_v6,
        p_positive_v6_mv=p_v6,
        qrs_area_v5_matrix=area_v5,
        qrs_area_v6_matrix=area_v6,
    )
    return _result(
        "GAN-04.01-02",
        "Possible faulty V6 – omitted from analysis",
        _truth(fact),
        [fact],
        ["V6"] if _truth(fact) else [],
    )


def evaluate_sequence_v1_v2(context: GlasgowContext) -> RuleEvaluation:
    facts = _validity_criteria(context)
    matched = any(_truth(facts[name]) for name in ("C", "D", "E", "F")) and _truth(facts["K"])
    evidence = [facts[name] for name in ("C", "D", "E", "F", "K")]
    return _result("GAN-04.01-03", "Possible sequence error: V1, V2 omitted", matched, evidence, ["V1", "V2"] if matched else [])


def evaluate_sequence_v2_v3(context: GlasgowContext) -> RuleEvaluation:
    facts = _validity_criteria(context)
    matched = ((_truth(facts["G"]) and _truth(facts["H"])) or _truth(facts["I"])) and _truth(facts["J"])
    evidence = [facts[name] for name in ("G", "H", "I", "J")]
    return _result("GAN-04.01-04", "Possible sequence error: V2, V3 omitted", matched, evidence, ["V2", "V3"] if matched else [])


def evaluate_sequence_vn(context: GlasgowContext) -> RuleEvaluation:
    facts = _validity_criteria(context)
    matched_pair: Optional[Tuple[str, str]] = None
    evidence: List[Dict[str, Any]] = []
    for lead, next_lead in (("V3", "V4"), ("V4", "V5"), ("V5", "V6")):
        evidence.extend((facts[f"A_{lead}"], facts[f"B_{lead}"]))
        if matched_pair is None and _truth(facts[f"A_{lead}"]) and _truth(facts[f"B_{lead}"]):
            matched_pair = (lead, next_lead)
    leads = list(matched_pair or ())
    statement = "Possible sequence error: Vn, Vn+1 omitted"
    if leads:
        statement = f"Possible sequence error: {leads[0]}, {leads[1]} omitted"
    return _result("GAN-04.01-05", statement, bool(leads), evidence, leads)


def evaluate_v1_v2_too_high(context: GlasgowContext) -> RuleEvaluation:
    operands = {
        lead: {
            key: _lead(context, lead, key, 0.0)
            for key in ("p_negative_amp_mv", "r_amp_mv", "r_prime_amp_mv", "t_negative_amp_mv")
        }
        for lead in ("V1", "V2")
    }
    v1, v2 = operands["V1"], operands["V2"]
    p_negative = all(abs(operands[lead]["p_negative_amp_mv"] or 0.0) > 0.05 for lead in ("V1", "V2"))
    v2_qrs = 0.5 > abs(v2["r_prime_amp_mv"] or 0.0) > abs(v2["r_amp_mv"] or 0.0) > 0.045
    v1_qrs = (
        0.5 > abs(v1["r_prime_amp_mv"] or 0.0) > abs(v1["r_amp_mv"] or 0.0) > 0.045
        or abs(v1["r_prime_amp_mv"] or 0.0) <= 1e-9
    )
    t_negative = all(abs(operands[lead]["t_negative_amp_mv"] or 0.0) > 0.05 for lead in ("V1", "V2"))
    fact = criterion(
        "statement_6_V1_V2_high",
        p_negative and v2_qrs and v1_qrs and t_negative,
        leads=operands,
        p_negative_condition=p_negative,
        v2_qrs_condition=v2_qrs,
        v1_qrs_condition=v1_qrs,
        t_negative_condition=t_negative,
    )
    return _result(
        "GAN-04.01-06",
        "V1/V2 are at least one interspace too high and have been omitted from the analysis",
        _truth(fact),
        [fact],
        ["V1", "V2"] if _truth(fact) else [],
    )


def _arm_reversal_match(context: GlasgowContext, facts: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    age_days = context.patient.age_days
    adult_branch = bool(age_days is None or age_days > 180.0)
    infant_branch = bool(age_days is not None and age_days <= 180.0)
    a, b, c, d, e, f, n = (_truth(facts[name]) for name in ("A", "B", "C", "D", "E", "F", "N"))
    ti, tv6 = _net_t(context, "I"), _net_t(context, "V6")
    branch_a = a and b and c and (d or f or n) and not e and adult_branch
    branch_b = c and f and not a and adult_branch
    branch_c = bool(a and b and infant_branch and _all_present(ti, tv6) and ti * tv6 < 0.0)
    branches = criterion(
        "statement_9_branches",
        branch_a or branch_b or branch_c,
        age_days=age_days,
        age_over_180_days=adult_branch,
        branch_a=branch_a,
        branch_b=branch_b,
        branch_c=branch_c,
        t_net_i_mv=ti,
        t_net_v6_mv=tv6,
    )
    return _truth(branches), [facts[name] for name in ("A", "B", "C", "D", "E", "F", "N")] + [branches]


def evaluate_arm_reversal(context: GlasgowContext) -> RuleEvaluation:
    facts = _reversal_criteria(context)
    matched, evidence = _arm_reversal_match(context, facts)
    excluded = ["I", "II", "III", "aVR", "aVL"] if matched else []
    return _result(
        "GAN-04.01-09",
        "--- Possible arm lead reversal – hence only aVF, V1 – V6 analyzed ---",
        matched,
        evidence,
        excluded,
    )


def evaluate_dextrocardia(context: GlasgowContext) -> RuleEvaluation:
    facts = _reversal_criteria(context)
    arm_reversal, arm_evidence = _arm_reversal_match(context, facts)
    matched = not arm_reversal and (
        _truth(facts["A"]) and _truth(facts["B"]) and _truth(facts["E"])
        or not _truth(facts["A"]) and _truth(facts["C"]) and _truth(facts["E"])
    )
    evidence = arm_evidence + [facts[name] for name in ("A", "B", "C", "E")]
    evidence.append(criterion("statement_10_arm_reversal_false", not arm_reversal, arm_reversal=arm_reversal))
    return _result("GAN-04.01-10", "--- Suggests dextrocardia ---", matched, evidence, [])


def evaluate_limb_reversal(context: GlasgowContext) -> RuleEvaluation:
    facts = _reversal_criteria(context)

    def low_voltage(lead: str) -> bool:
        return all(_truth(facts[f"{letter}_{lead}"]) for letter in ("G", "H", "J", "L", "O"))

    branch_a = low_voltage("II") and _truth(facts["K"])
    branch_b = low_voltage("III") and _truth(facts["M"])
    branch_c = low_voltage("I") and _truth(facts["P"])
    branch = criterion("statement_11_branches", branch_a or branch_b or branch_c, branch_a=branch_a, branch_b=branch_b, branch_c=branch_c)
    evidence = [
        facts[f"{letter}_{lead}"]
        for lead in ("I", "II", "III")
        for letter in ("G", "H", "J", "L", "O")
    ] + [facts[name] for name in ("K", "M", "P")] + [branch]
    matched = _truth(branch)
    return _result(
        "GAN-04.01-11",
        "--- Possible limb lead reversal – hence only V1-V6 analyzed ---",
        matched,
        evidence,
        _LIMB_LEADS if matched else [],
    )


def evaluate_arm_leg_interchange(context: GlasgowContext) -> RuleEvaluation:
    fact = _reversal_criteria(context)["Q"]
    matched = _truth(fact)
    return _result(
        "GAN-04.01-12",
        "--- Possible arm/leg lead interchange – hence only V1-V6 analyzed ---",
        matched,
        [fact],
        _LIMB_LEADS if matched else [],
    )


def lead_exclusions(evaluations: Iterable[RuleEvaluation]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for evaluation in evaluations:
        if evaluation.evaluation_status != "matched":
            continue
        for lead in evaluation.evidence.get("excluded_leads", []):
            rules = result.setdefault(str(lead), [])
            if evaluation.rule_id not in rules:
                rules.append(evaluation.rule_id)
    return result


def preliminary_lead_rules() -> List[RuleSpec]:
    definitions: Sequence[Tuple[str, int, str, Callable[[GlasgowContext], RuleEvaluation]]] = (
        ("GAN-04.01-01", 11, "Possible faulty Vn – omitted from analysis", evaluate_faulty_vn),
        ("GAN-04.01-02", 11, "Possible faulty V6 – omitted from analysis", evaluate_faulty_v6),
        ("GAN-04.01-03", 11, "Possible sequence error: V1, V2 omitted", evaluate_sequence_v1_v2),
        ("GAN-04.01-04", 11, "Possible sequence error: V2, V3 omitted", evaluate_sequence_v2_v3),
        ("GAN-04.01-05", 11, "Possible sequence error: Vn, Vn+1 omitted", evaluate_sequence_vn),
        ("GAN-04.01-06", 11, "V1/V2 are at least one interspace too high and have been omitted from the analysis", evaluate_v1_v2_too_high),
        ("GAN-04.01-09", 13, "--- Possible arm lead reversal – hence only aVF, V1 – V6 analyzed ---", evaluate_arm_reversal),
        ("GAN-04.01-10", 13, "--- Suggests dextrocardia ---", evaluate_dextrocardia),
        ("GAN-04.01-11", 13, "--- Possible limb lead reversal – hence only V1-V6 analyzed ---", evaluate_limb_reversal),
        ("GAN-04.01-12", 13, "--- Possible arm/leg lead interchange – hence only V1-V6 analyzed ---", evaluate_arm_leg_interchange),
    )
    return [
        RuleSpec(
            rule_id=rule_id,
            chapter="4.1",
            pdf_page=page,
            category="preliminary_lead",
            order=int("".join(character for character in rule_id if character.isdigit())),
            statement=statement,
            required_inputs=("representative_leads",),
            fidelity="glasgow_explicit",
            source_kind="glasgow_guide",
            source_reference=_SOURCE,
            evaluator=evaluator,
        )
        for rule_id, page, statement, evaluator in definitions
    ]

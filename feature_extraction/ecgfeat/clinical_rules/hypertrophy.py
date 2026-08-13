from __future__ import annotations

from typing import Iterable, Optional

from .config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import RuleEvaluation
from .rhythm import disorganized_atrial_activity, has_borderline_af_evidence
from .sources import SOURCE_AHA_HYPERTROPHY_2009


_MORPHOLOGY = DEFAULT_DIAGNOSTIC_CONFIG.morphology
_RHYTHM = DEFAULT_DIAGNOSTIC_CONFIG.rhythm
_INTERVALS = DEFAULT_DIAGNOSTIC_CONFIG.intervals


def _has_conduction(conduction: Iterable[RuleEvaluation], code: str) -> bool:
    # `evaluates_code` stays stable across confidence tiers (e.g. a
    # `probable_lbbb_pattern` statement still carries evaluates_code
    # "lbbb_pattern"), so a lower-confidence BBB match still suppresses the
    # voltage criteria it confounds.
    return any(
        item.status == "matched" and item.evidence.get("evaluates_code") == code
        for item in conduction
    )


def _evaluation(
    *,
    rule_id: str,
    code: str,
    status: str,
    evidence: dict,
    statement: Optional[str] = None,
    missing_inputs: Optional[list[str]] = None,
    suppressed_by: Optional[list[str]] = None,
    domain: str = "hypertrophy",
    confidence: Optional[str] = None,
    severity: str = "abnormal",
) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=rule_id,
        domain=domain,
        status=status,
        statement_code=code if status in {"matched", "suppressed"} else None,
        statement=statement if status in {"matched", "suppressed"} else None,
        severity=severity if status in {"matched", "suppressed"} else "normal",
        confidence=confidence,
        missing_inputs=list(missing_inputs or []),
        evidence={**evidence, "evaluates_code": code},
        suppressed_by=list(suppressed_by or []),
        source=dict(SOURCE_AHA_HYPERTROPHY_2009),
    )


def _adult_lvh(context, conduction: Iterable[RuleEvaluation]) -> RuleEvaluation:
    values = {
        "r_avl_mv": context.lead_value("aVL", "r_amp_mv"),
        "s_v1_mv": context.lead_value("V1", "s_amp_mv"),
        "s_v3_mv": context.lead_value("V3", "s_amp_mv"),
        "r_v5_mv": context.lead_value("V5", "r_amp_mv"),
        "r_v6_mv": context.lead_value("V6", "r_amp_mv"),
        "qrs_ms": context.global_value("qrs_ms"),
    }
    for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"):
        values[f"s_{lead}_mv"] = context.lead_value(lead, "s_amp_mv")
    matched: list[str] = []
    assessed: list[str] = []
    cornell_mv = None
    if values["r_avl_mv"] is not None and values["s_v3_mv"] is not None:
        assessed.append("Cornell voltage")
        cornell_mv = values["r_avl_mv"] + abs(values["s_v3_mv"])
        threshold = 2.0 if context.sex in {"female", "f", "woman"} else 2.8
        if cornell_mv > threshold:
            matched.append("Cornell voltage")
        if values["qrs_ms"] is not None:
            assessed.append("Cornell product")
            if cornell_mv * values["qrs_ms"] > 244.0:
                matched.append("Cornell product")
    lateral_r = [
        value for value in (values["r_v5_mv"], values["r_v6_mv"])
        if value is not None
    ]
    if values["s_v1_mv"] is not None and lateral_r:
        assessed.append("Sokolow-Lyon")
        if abs(values["s_v1_mv"]) + max(lateral_r) >= 3.5:
            matched.append("Sokolow-Lyon")
    if values["r_avl_mv"] is not None:
        assessed.append("R-aVL")
        if values["r_avl_mv"] >= 1.1:
            matched.append("R-aVL")
    deepest_s = max(
        (
            abs(value)
            for key, value in values.items()
            if key.startswith("s_") and value is not None
        ),
        default=None,
    )
    s_v4 = values.get("s_V4_mv")
    peguero_mv = (
        deepest_s + abs(s_v4)
        if deepest_s is not None and s_v4 is not None
        else None
    )
    if peguero_mv is not None:
        assessed.append("Peguero-Lo Presti")
        peguero_threshold = 2.3 if context.sex in {"female", "f", "woman"} else 2.8
        if peguero_mv >= peguero_threshold:
            matched.append("Peguero-Lo Presti")
    lateral_single = max(lateral_r) if lateral_r else None
    if lateral_single is not None:
        assessed.append("Single high precordial R")
        if lateral_single >= 2.6:
            matched.append("Single high precordial R")
    # Sokolow-Lyon limb-lead criterion R(I) + S(III). The constant already
    # existed in interpret.py but this layer never evaluated it, so a tracing
    # whose only positive criterion was the limb sum was reported as meeting
    # no voltage criteria at all.
    r_i = context.lead_value("I", "r_amp_mv")
    s_iii = context.lead_value("III", "s_amp_mv")
    sokolow_limb_mv = (
        r_i + abs(s_iii) if r_i is not None and s_iii is not None else None
    )
    if sokolow_limb_mv is not None:
        assessed.append("Sokolow-Lyon limb")
        if sokolow_limb_mv >= _MORPHOLOGY.lvh_sokolow_limb_mv:
            matched.append("Sokolow-Lyon limb")

    evidence = {
        **values,
        "r_i_mv": r_i,
        "s_iii_mv": s_iii,
        "sokolow_lyon_limb_mv": sokolow_limb_mv,
        "cornell_voltage_mv": cornell_mv,
        "assessed_criteria": assessed,
        "matched_criteria": matched,
        "matched_criterion_count": len(matched),
        "deepest_s_mv": deepest_s,
        "peguero_lo_presti_mv": peguero_mv,
    }
    suppressors = [
        code for code in ("lbbb_pattern", "nonspecific_ivcd")
        if _has_conduction(conduction, code)
    ]
    if matched:
        classic_matched = [
            criterion
            for criterion in matched
            if criterion not in {
                "Peguero-Lo Presti",
                "Single high precordial R",
            }
        ]
        possible_only = not classic_matched
        return _evaluation(
            rule_id="CLIN-HYPERTROPHY-LVH-01",
            code=(
                "possible_lvh_voltage"
                if possible_only
                else "lvh_voltage_criteria"
            ),
            status="matched",
            evidence=evidence,
            statement=(
                "Possible LVH by an isolated extended voltage criterion"
                if possible_only
                else "Meets multiple ECG voltage criteria for LVH"
                if len(matched) >= 2
                else "Meets a single ECG voltage criterion for LVH"
            ),
            suppressed_by=suppressors,
            confidence=(
                "low"
                if possible_only
                else "multiple_voltage_criteria"
                if len(matched) >= 2
                else "single_voltage_criterion"
            ),
            severity="borderline" if possible_only else "abnormal",
        )
    # A negative aggregate is valid only when all named voltage families could
    # be assessed. Partial data must not be treated as evidence of absence.
    required = {
        "Cornell voltage",
        "Cornell product",
        "Sokolow-Lyon",
        "Sokolow-Lyon limb",
        "R-aVL",
        "Peguero-Lo Presti",
        "Single high precordial R",
    }
    if set(assessed) != required:
        missing = []
        if values["r_avl_mv"] is None:
            missing.append("lead.aVL.r_amp_mv")
        if values["s_v3_mv"] is None:
            missing.append("lead.V3.s_amp_mv")
        if values["s_v1_mv"] is None:
            missing.append("lead.V1.s_amp_mv")
        if not lateral_r:
            missing.append("lead.V5_or_V6.r_amp_mv")
        if r_i is None:
            missing.append("lead.I.r_amp_mv")
        if s_iii is None:
            missing.append("lead.III.s_amp_mv")
        if values["qrs_ms"] is None:
            missing.append("global.qrs_ms")
        return _evaluation(
            rule_id="CLIN-HYPERTROPHY-LVH-01",
            code="lvh_voltage_criteria",
            status="unavailable",
            evidence=evidence,
            missing_inputs=missing,
        )
    return _evaluation(
        rule_id="CLIN-HYPERTROPHY-LVH-01",
        code="lvh_voltage_criteria",
        status="not_matched",
        evidence=evidence,
    )


def _adult_rvh(context, conduction: Iterable[RuleEvaluation]) -> RuleEvaluation:
    r_v1 = context.lead_value("V1", "r_amp_mv")
    s_v1 = context.lead_value("V1", "s_amp_mv")
    # r_prime/r_prime_duration are retained as evidence context only; the
    # adult RVH morphology criterion below does not use them, so a missing
    # R' duration must not block this rule.
    r_prime = context.lead_value("V1", "r_prime_amp_mv")
    r_prime_duration = context.lead_value("V1", "r_prime_duration_ms")
    axis = context.global_value("qrs_axis_deg")
    s_v5 = context.lead_value("V5", "s_amp_mv")
    s_v6 = context.lead_value("V6", "s_amp_mv")
    evidence = {
        "r_v1_mv": r_v1,
        "s_v1_mv": s_v1,
        "r_prime_v1_mv": r_prime,
        "r_prime_duration_ms": r_prime_duration,
        "qrs_axis_deg": axis,
        "s_v5_mv": s_v5,
        "s_v6_mv": s_v6,
    }
    missing = []
    if r_v1 is None:
        missing.append("lead.V1.r_amp_mv")
    if s_v1 is None:
        missing.append("lead.V1.s_amp_mv")
    if axis is None:
        missing.append("global.qrs_axis_deg")
    if missing:
        return _evaluation(
            rule_id="CLIN-HYPERTROPHY-RVH-01",
            code="rvh_pattern",
            status="unavailable",
            evidence=evidence,
            missing_inputs=missing,
        )
    suppressors = [
        code for code in ("rbbb_pattern", "lpfb_pattern")
        if _has_conduction(conduction, code)
    ]
    # R(V1) + S(V5 or V6): the precordial sum is a distinct RVH criterion from
    # R-dominance in V1 alone, and catches the case where neither the R nor
    # the S is individually large enough but the right-sided shift still is.
    lateral_s = [value for value in (s_v5, s_v6) if value is not None]
    precordial_sum_mv = (
        r_v1 + max(abs(value) for value in lateral_s) if lateral_s else None
    )
    # Right precordial strain: down-sloping repolarisation over V1-V3 is the
    # supporting sign that separates true RVH from an incidental vertical axis.
    strain_leads = []
    for lead in ("V1", "V2", "V3"):
        st_value = context.lead_value(lead, "st_on_mv")
        t_value = context.lead_value(lead, "t_amp_mv", "reliable_for_t")
        if (
            st_value is not None
            and t_value is not None
            and st_value <= _MORPHOLOGY.rvh_strain_st_mv
            and t_value < _MORPHOLOGY.rvh_strain_t_mv
        ):
            strain_leads.append(lead)
    criteria = {
        "dominant_r_v1": r_v1 >= 0.7 and r_v1 > abs(s_v1),
        "right_axis_deviation": axis >= 110.0,
        "deep_s_v5_v6": any(
            value is not None and abs(value) >= 0.7
            for value in (s_v5, s_v6)
        ),
        "precordial_sum": bool(
            precordial_sum_mv is not None
            and precordial_sum_mv >= _MORPHOLOGY.rvh_precordial_sum_mv
        ),
        "right_precordial_strain": len(strain_leads) >= 2,
    }
    morphology = sum(bool(value) for value in criteria.values()) >= 2
    evidence["precordial_sum_mv"] = precordial_sum_mv
    evidence["right_precordial_strain_leads"] = strain_leads
    evidence["thresholds"] = {
        "precordial_sum_mv": _MORPHOLOGY.rvh_precordial_sum_mv,
        "strain_st_mv": _MORPHOLOGY.rvh_strain_st_mv,
    }
    evidence["criteria"] = criteria
    evidence["matched_criterion_count"] = sum(bool(value) for value in criteria.values())
    return _evaluation(
        rule_id="CLIN-HYPERTROPHY-RVH-01",
        code="rvh_pattern",
        status="matched" if morphology else "not_matched",
        evidence=evidence,
        statement="ECG pattern meeting adult RVH criteria",
        suppressed_by=suppressors if morphology else [],
    )


def _adult_atrial(context) -> list[RuleEvaluation]:
    features = getattr(context, "features", None)
    metadata = getattr(features, "metadata", {}) if features is not None else {}
    rhythm_analysis = metadata.get("rhythm_analysis", {}) if isinstance(metadata, dict) else {}
    rhythm_analysis = rhythm_analysis if isinstance(rhythm_analysis, dict) else {}
    af_afl = rhythm_analysis.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    active_atrial_pattern = (
        "atrial_fibrillation_pattern"
        if bool(
            af_afl.get("probable_af")
            or has_borderline_af_evidence(af_afl)
        )
        else "atrial_flutter_pattern"
        if bool(af_afl.get("probable_flutter"))
        else None
    )
    if active_atrial_pattern is not None:
        evidence = {"atrial_morphology_invalid_by": active_atrial_pattern}
        return [
            _evaluation(
                rule_id="CLIN-HYPERTROPHY-RAE-01",
                code="right_atrial_abnormality",
                status="not_applicable",
                evidence=evidence,
                domain="atrial_abnormality",
            ),
            _evaluation(
                rule_id="CLIN-HYPERTROPHY-LAE-01",
                code="left_atrial_abnormality",
                status="not_applicable",
                evidence=evidence,
                domain="atrial_abnormality",
            ),
        ]
    # A flutter/fibrillation call that was never made is not a negative call.
    # On the flutter records in WFDBRecords/01/019 the flutter rule sat at
    # `unavailable`, so the confirmed-pattern guard above never fired and F
    # waves were measured as P waves. Gate on whether organized atrial
    # activity is actually present instead of on whether a mechanism was named.
    disorganized = disorganized_atrial_activity(
        af_afl, threshold=_RHYTHM.organized_p_ratio_min_for_morphology
    )
    if disorganized is not None:
        evidence = {
            "atrial_morphology_unreliable_by": "disorganized_atrial_activity",
            "organized_p_ratio": disorganized,
            "organized_p_ratio_min": (
                _RHYTHM.organized_p_ratio_min_for_morphology
            ),
        }
        return [
            _evaluation(
                rule_id="CLIN-HYPERTROPHY-RAE-01",
                code="right_atrial_abnormality",
                status="unavailable",
                evidence=evidence,
                missing_inputs=["rhythm.organized_atrial_activity"],
                domain="atrial_abnormality",
            ),
            _evaluation(
                rule_id="CLIN-HYPERTROPHY-LAE-01",
                code="left_atrial_abnormality",
                status="unavailable",
                evidence=evidence,
                missing_inputs=["rhythm.organized_atrial_activity"],
                domain="atrial_abnormality",
            ),
        ]
    p_ii = context.lead_value("II", "p_amp_mv", "reliable_for_p")
    p_ii_duration = context.lead_value(
        "II", "p_dur_consensus_ms", "reliable_for_p"
    )
    terminal_amp = context.lead_value(
        "V1", "p_terminal_amp_mv", "reliable_for_p"
    )
    terminal_duration = context.lead_value(
        "V1", "p_terminal_duration_ms", "reliable_for_p"
    )
    rae_missing = [
        key for key, value in (
            ("lead.II.p_amp_mv", p_ii),
            ("lead.II.p_dur_consensus_ms", p_ii_duration),
        ) if value is None
    ]
    lae_missing = [
        key for key, value in (
            ("lead.V1.p_terminal_amp_mv", terminal_amp),
            ("lead.V1.p_terminal_duration_ms", terminal_duration),
        ) if value is None
    ]
    rae = _evaluation(
        rule_id="CLIN-HYPERTROPHY-RAE-01",
        code="right_atrial_abnormality",
        status="unavailable" if rae_missing else (
            "matched" if p_ii > 0.25 and p_ii_duration <= 120.0 else "not_matched"
        ),
        evidence={"p_ii_mv": p_ii, "p_ii_duration_ms": p_ii_duration},
        statement="ECG pattern of right atrial abnormality",
        missing_inputs=rae_missing,
        domain="atrial_abnormality",
    )
    # The terminal negative component is part of the P wave, so it cannot be
    # longer than a P wave is allowed to be. Values at or beyond the whole-P
    # limit mean something other than a P wave was delineated -- on the
    # flutter records here it measured 92-94 ms, which no sinus P has.
    terminal_implausible = bool(
        terminal_duration is not None
        and terminal_duration >= _INTERVALS.p_duration_prolonged_ms
    )
    lae_evidence = {
        "v1_terminal_p_amp_mv": terminal_amp,
        "v1_terminal_p_duration_ms": terminal_duration,
        "terminal_duration_implausible": terminal_implausible,
        "terminal_duration_max_ms": _INTERVALS.p_duration_prolonged_ms,
    }
    if terminal_implausible:
        lae = _evaluation(
            rule_id="CLIN-HYPERTROPHY-LAE-01",
            code="left_atrial_abnormality",
            status="unavailable",
            evidence=lae_evidence,
            missing_inputs=["lead.V1.plausible_p_terminal_duration_ms"],
            domain="atrial_abnormality",
        )
    else:
        lae = _evaluation(
            rule_id="CLIN-HYPERTROPHY-LAE-01",
            code="left_atrial_abnormality",
            status="unavailable" if lae_missing else (
                "matched"
                if terminal_amp <= -0.10 and terminal_duration >= 40.0
                else "not_matched"
            ),
            evidence=lae_evidence,
            statement="ECG pattern of left atrial abnormality",
            missing_inputs=lae_missing,
            domain="atrial_abnormality",
        )
    return [rae, lae]


def _pediatric_unavailable() -> list[RuleEvaluation]:
    return [
        _evaluation(
            rule_id="CLIN-HYPERTROPHY-PEDS-LVH-01",
            code="pediatric_lvh_voltage",
            status="unavailable",
            evidence={"policy": "public_reference_table_required"},
            missing_inputs=["pediatric_lvh_percentile_table"],
        ),
        _evaluation(
            rule_id="CLIN-HYPERTROPHY-PEDS-RVH-01",
            code="pediatric_rvh_voltage",
            status="unavailable",
            evidence={"policy": "public_reference_table_required"},
            missing_inputs=["pediatric_rvh_percentile_table"],
        ),
        _evaluation(
            rule_id="CLIN-HYPERTROPHY-PEDS-RAE-01",
            code="pediatric_right_atrial_abnormality",
            status="unavailable",
            evidence={"policy": "public_reference_table_required"},
            missing_inputs=["pediatric_atrial_percentile_table"],
            domain="atrial_abnormality",
        ),
        _evaluation(
            rule_id="CLIN-HYPERTROPHY-PEDS-LAE-01",
            code="pediatric_left_atrial_abnormality",
            status="unavailable",
            evidence={"policy": "public_reference_table_required"},
            missing_inputs=["pediatric_atrial_percentile_table"],
            domain="atrial_abnormality",
        ),
    ]


def evaluate_hypertrophy(
    context, conduction: Iterable[RuleEvaluation]
) -> list[RuleEvaluation]:
    if context.age_years is not None and context.age_years < 16.0:
        return _pediatric_unavailable()
    conduction = list(conduction)
    return [
        _adult_lvh(context, conduction),
        _adult_rvh(context, conduction),
        *_adult_atrial(context),
    ]

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from .models import RuleEvaluation


def _row(
    rule_id: str,
    code: str,
    statement: str,
    matched: bool,
    available: bool,
    evidence: dict,
    *,
    severity: str = "observation",
    priority: Optional[str] = None,
    confidence: str = "medium",
    human_review_required: Optional[bool] = None,
) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=rule_id,
        domain="high_risk_patterns",
        status="matched" if matched else ("not_matched" if available else "unavailable"),
        statement_code=code if matched else None,
        statement=statement if matched else None,
        severity=severity if matched else "normal",
        confidence=confidence if matched else None,
        priority=priority,
        evidence={**evidence, "evaluates_code": code},
        human_review_required=(
            matched
            if human_review_required is None
            else bool(human_review_required and matched)
        ),
        normality_role="optional_screen",
    )


def _longest_consecutive_run(values: list[int]) -> list[int]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    best: list[int] = []
    current = [ordered[0]]
    for value in ordered[1:]:
        if value == current[-1] + 1:
            current.append(value)
        else:
            if len(current) > len(best):
                best = current
            current = [value]
    if len(current) > len(best):
        best = current
    return best


def _alternans_lead_result(
    sequence: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(sequence, dtype=float)
    if values.size < 8 or not np.all(np.isfinite(values)):
        return {
            "available": False,
            "qualifies": False,
            "beat_count": int(values.size),
        }
    scale = float(np.median(np.abs(values)))
    if scale <= 1e-9:
        return {
            "available": False,
            "qualifies": False,
            "beat_count": int(values.size),
        }
    positions = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(positions, values, 1)
    residual = values - (slope * positions + intercept)
    even_mean = float(np.mean(residual[::2]))
    odd_mean = float(np.mean(residual[1::2]))
    phase_delta = even_mean - odd_mean
    index = abs(phase_delta) / scale
    adjacent = residual[:-1] * residual[1:]
    sign_alternation_fraction = float(np.mean(adjacent < 0.0))

    split = values.size // 2
    if split % 2:
        split -= 1
    segment_indices = []
    segment_phases = []
    for segment in (residual[:split], residual[split:]):
        if segment.size < 4:
            continue
        delta = float(np.mean(segment[::2]) - np.mean(segment[1::2]))
        segment_indices.append(abs(delta) / scale)
        segment_phases.append(int(np.sign(delta)))
    stable_segments = bool(
        len(segment_indices) == 2
        and min(segment_indices) >= 0.10
        and segment_phases[0] != 0
        and segment_phases[0] == segment_phases[1]
    )
    qualifies = bool(
        index >= 0.15
        and sign_alternation_fraction >= 0.70
        and stable_segments
    )
    return {
        "available": True,
        "qualifies": qualifies,
        "beat_count": int(values.size),
        "alternans_index": float(index),
        "phase": int(np.sign(phase_delta)),
        "sign_alternation_fraction": sign_alternation_fraction,
        "split_half_indices": segment_indices,
        "split_half_phase_stable": stable_segments,
        "linear_drift_per_beat": float(slope),
    }


def _electrical_alternans_screen(context) -> tuple[bool, bool, dict]:
    features = getattr(context, "features", None)
    beats = sorted(
        list(getattr(features, "beats", []) or []),
        key=lambda beat: int(getattr(beat, "beat_id", -1)),
    )
    beat_by_id = {
        int(beat.beat_id): beat
        for beat in beats
    }
    group_counts = Counter(
        int(beat.group_id)
        for beat in beats
        if not bool(beat.paced)
    )
    dominant_group = (
        group_counts.most_common(1)[0][0]
        if group_counts
        else None
    )
    excluded_reasons: list[str] = []
    if any(bool(beat.paced) for beat in beats):
        excluded_reasons.append("paced_beats_present")
    if len(group_counts) > 1:
        excluded_reasons.append("multiple_qrs_morphology_groups")

    metadata = getattr(features, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    rhythm = metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    af_afl = rhythm.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    if bool(af_afl.get("probable_af")):
        excluded_reasons.append("probable_atrial_fibrillation")
    if bool(af_afl.get("probable_flutter")):
        excluded_reasons.append("probable_atrial_flutter")

    rr_cv = af_afl.get("rr_cv")
    try:
        rr_cv = float(rr_cv) if rr_cv is not None else None
    except (TypeError, ValueError):
        rr_cv = None
    if rr_cv is None:
        rr_values = [
            float(beat.rr_prev_ms)
            for beat in beats
            if beat.rr_prev_ms is not None and float(beat.rr_prev_ms) > 0.0
        ]
        if len(rr_values) >= 3 and float(np.mean(rr_values)) > 0.0:
            rr_cv = float(np.std(rr_values) / np.mean(rr_values))
    if rr_cv is not None and rr_cv > 0.10:
        excluded_reasons.append("rr_irregularity_or_ectopy")

    per_lead_values: dict[str, dict[int, float]] = {
        lead: {} for lead in ("II", "V1", "V5")
    }
    for feature in list(getattr(features, "beat_features", []) or []):
        if feature.lead not in per_lead_values:
            continue
        beat = beat_by_id.get(int(feature.beat_id))
        if (
            beat is None
            or dominant_group is None
            or int(beat.group_id) != dominant_group
            or bool(beat.paced)
            or not bool(feature.beat_measurement_reliable)
            or feature.qrs_area is None
        ):
            continue
        per_lead_values[feature.lead][int(feature.beat_id)] = float(
            feature.qrs_area
        )

    lead_results: dict[str, dict[str, object]] = {}
    for lead, values in per_lead_values.items():
        run = _longest_consecutive_run(list(values))
        sequence = np.asarray([values[beat_id] for beat_id in run], dtype=float)
        result = _alternans_lead_result(sequence)
        result["beat_ids"] = run
        lead_results[lead] = result

    available_leads = [
        lead for lead, result in lead_results.items()
        if bool(result.get("available"))
    ]
    supporting_leads = [
        lead for lead, result in lead_results.items()
        if bool(result.get("qualifies"))
    ]
    phases = {
        int(lead_results[lead]["phase"])
        for lead in supporting_leads
        if int(lead_results[lead].get("phase") or 0) != 0
    }
    phase_concordant = len(phases) == 1
    rhythm_eligible = not excluded_reasons
    available = len(available_leads) >= 2
    matched = bool(
        available
        and rhythm_eligible
        and len(supporting_leads) >= 2
        and phase_concordant
    )
    return matched, available, {
        "minimum_consecutive_beats": 8,
        "minimum_supporting_leads": 2,
        "minimum_alternans_index": 0.15,
        "minimum_sign_alternation_fraction": 0.70,
        "rr_cv": rr_cv,
        "dominant_group_id": dominant_group,
        "qrs_group_counts": dict(group_counts),
        "excluded_reasons": sorted(set(excluded_reasons)),
        "available_leads": available_leads,
        "supporting_leads": supporting_leads,
        "phase_concordant": phase_concordant,
        "lead_results": lead_results,
    }


def evaluate_other_patterns(context) -> list[RuleEvaluation]:
    potassium_leads = {}
    for lead in ("V2", "V3", "V4", "V5"):
        potassium_leads[lead] = {
            "t_amp": context.lead_value(lead, "t_amp_mv", "reliable_for_t"),
            "t_symmetry": context.lead_value(lead, "t_symmetry", "reliable_for_t"),
            "t_duration_ms": context.lead_value(
                lead, "t_dur_ms", "reliable_for_t"
            ),
            # Signed, not the legacy magnitude: hypokalemia produces a large
            # *upright* U wave, and an inverted U of the same magnitude used to
            # satisfy this criterion.
            "u_amp": context.lead_value(lead, "u_amp_signed_mv", "reliable_for_t"),
            "st_j": context.lead_value(lead, "st_on_mv"),
            "st_mid": context.lead_value(lead, "st_mid_mv"),
            "st_80": context.lead_value(lead, "st_80ms_mv"),
            "st_morphology": context.lead_raw_value(lead, "st_morphology"),
        }
    p_ii = context.lead_value("II", "p_amp_mv", "reliable_for_p")
    qrs = context.global_value("qrs_ms")
    hyper_support = [
        lead for lead, values in potassium_leads.items()
        if values["t_amp"] is not None
        and values["t_amp"] >= 0.60
        and values["t_symmetry"] is not None
        and 0.8 <= values["t_symmetry"] <= 1.2
        and values["t_duration_ms"] is not None
        and values["t_duration_ms"] <= 220.0
    ]
    hyper_available = (
        qrs is not None
        and p_ii is not None
        and sum(
            values["t_symmetry"] is not None
            and values["t_duration_ms"] is not None
            for values in potassium_leads.values()
        ) >= 3
    )
    hyper = (
        hyper_available
        and len(hyper_support) >= 3
        and p_ii < 0.10
        and qrs >= 110.0
    )

    hypo_support = [
        lead for lead, values in potassium_leads.items()
        if values["u_amp"] is not None
        and values["t_amp"] is not None
        and values["u_amp"] >= 0.05
        and values["u_amp"] > 0.75 * max(abs(values["t_amp"]), 0.02)
        and abs(values["t_amp"]) < 0.10
        and values["st_j"] is not None
        and values["st_j"] <= -0.05
        and values["st_mid"] is not None
        and values["st_mid"] <= -0.05
        and values["st_80"] is not None
        and values["st_80"] <= -0.05
        and str(values["st_morphology"] or "").lower()
        in {"horizontal", "downsloping"}
    ]
    hypo_available = sum(
        values["u_amp"] is not None
        and values["t_amp"] is not None
        and values["st_mid"] is not None
        and values["st_80"] is not None
        for values in potassium_leads.values()
    ) >= 3

    alternans, alternans_available, alternans_evidence = (
        _electrical_alternans_screen(context)
    )

    all_leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    st_values = {
        lead: context.lead_value(lead, "st_on_mv")
        for lead in all_leads
    }
    pr_values = {
        lead: context.lead_value(lead, "pr_segment_level_mv", "reliable_for_p")
        for lead in all_leads
    }
    diffuse_ste = [
        lead for lead, value in st_values.items()
        if lead not in {"aVR", "V1"} and value is not None and value >= 0.10
    ]
    reciprocal_std = [
        lead for lead, value in st_values.items()
        if lead not in {"aVR", "V1"} and value is not None and value <= -0.05
    ]
    pr_depressed = [
        lead for lead, value in pr_values.items()
        if lead != "aVR" and value is not None and value <= -0.05
    ]
    avr_pr = pr_values.get("aVR")
    pericarditis_available = (
        len([value for value in st_values.values() if value is not None]) >= 8
        and len([value for value in pr_values.values() if value is not None]) >= 4
    )
    pericarditis = bool(
        len(diffuse_ste) >= 6
        and len(pr_depressed) >= 2
        and avr_pr is not None
        and avr_pr >= 0.05
        and not reciprocal_std
    )

    early_repol_leads = [
        lead for lead in ("V2", "V3", "V4", "V5")
        if st_values.get(lead) is not None
        and st_values[lead] >= 0.10
        and (
            bool(context.lead_raw_value(lead, "qrs_slur_flag"))
            or int(context.lead_raw_value(lead, "qrs_notch_count") or 0) >= 1
        )
    ]
    hr = context.global_value("heart_rate_bpm")
    qrs = context.global_value("qrs_ms")
    early_repol_available = (
        hr is not None and qrs is not None
        and sum(st_values.get(lead) is not None for lead in ("V2", "V3", "V4", "V5")) >= 2
    )
    early_repol = bool(
        early_repol_available
        and hr < 100.0
        and qrs < 120.0
        and len(early_repol_leads) >= 2
        and not reciprocal_std
    )

    qrs_i = context.lead_value("I", "qrs_signed_area")
    qrs_avr = context.lead_value("aVR", "qrs_signed_area")
    r_v1 = context.lead_value("V1", "r_amp_mv")
    r_v6 = context.lead_value("V6", "r_amp_mv")
    dextro_available = all(
        value is not None for value in (qrs_i, qrs_avr, r_v1, r_v6)
    )
    dextrocardia = bool(
        dextro_available
        and qrs_i < 0.0
        and qrs_avr > 0.0
        and r_v1 > r_v6
        and not bool(getattr(context, "limb_reversal", False))
    )

    s_i = context.lead_value("I", "s_amp_mv")
    q_iii = context.lead_value("III", "q_amp_mv")
    t_iii = context.lead_value("III", "t_amp_mv", "reliable_for_t")
    axis = context.global_value("qrs_axis_deg")
    pe_available = all(value is not None for value in (hr, s_i, q_iii, t_iii, axis))
    pe_pattern = bool(
        pe_available
        and hr > 100.0
        and abs(s_i) >= 0.15
        and q_iii <= -0.05
        and t_iii <= -0.05
        and axis > 90.0
    )

    digitalis_leads = [
        lead for lead in ("I", "aVL", "V4", "V5", "V6")
        if st_values.get(lead) is not None
        and st_values[lead] <= -0.05
        and str(context.lead_raw_value(lead, "st_morphology") or "").lower()
        == "downsloping"
    ]
    patient = context.features.metadata.get("patient_meta")
    meds = (
        getattr(patient, "meds", None)
        if patient is not None and not isinstance(patient, dict)
        else patient.get("meds") if isinstance(patient, dict) else None
    ) or []
    digitalis_exposure = any(
        str(med).strip().lower() in {"digoxin", "digitalis"}
        for med in meds
    )
    digitalis_available = len(
        [lead for lead in ("I", "aVL", "V4", "V5", "V6") if st_values.get(lead) is not None]
    ) >= 3
    digitalis_pattern = bool(len(digitalis_leads) >= 3 and digitalis_exposure)

    return [
        _row(
            "CLIN-HIGHRISK-HYPERK-01",
            "hyperkalemia_pattern",
            "Low-confidence ECG screen possibly compatible with hyperkalemia; confirm with serum potassium and serial ECGs",
            hyper,
            hyper_available,
            {
                "lead_measurements": potassium_leads,
                "p_amp_II_mv": p_ii,
                "qrs_ms": qrs,
                "supporting_leads": hyper_support,
            },
            severity="observation",
            priority="P4",
            confidence="low",
        ),
        _row(
            "CLIN-HIGHRISK-HYPOK-01",
            "hypokalemia_pattern",
            "Low-confidence ECG screen possibly compatible with hypokalemia; confirm with serum potassium and serial ECGs",
            len(hypo_support) >= 3,
            hypo_available,
            {
                "lead_measurements": potassium_leads,
                "supporting_leads": hypo_support,
            },
            severity="observation",
            priority="P4",
            confidence="low",
        ),
        _row(
            "CLIN-HIGHRISK-ALTERNANS-01",
            "electrical_alternans",
            "Electrical alternans pattern; evaluate for pericardial effusion and other causes",
            alternans,
            alternans_available,
            alternans_evidence,
            severity="high",
            priority="P1",
        ),
        _row(
            "CLIN-HIGHRISK-PERICARDITIS-01",
            "acute_pericarditis_pattern",
            "Pattern suggestive of acute pericarditis; acute coronary occlusion must be excluded",
            pericarditis,
            pericarditis_available,
            {
                "diffuse_st_elevation_leads": diffuse_ste,
                "pr_depression_leads": pr_depressed,
                "avr_pr_segment_mv": avr_pr,
                "reciprocal_st_depression_leads": reciprocal_std,
            },
            severity="high",
            priority="P1",
        ),
        _row(
            "CLIN-OTHER-EARLYREPOL-01",
            "early_repolarization_pattern",
            "Early-repolarization pattern",
            early_repol,
            early_repol_available,
            {
                "qualifying_leads": early_repol_leads,
                "heart_rate_bpm": hr,
                "qrs_ms": qrs,
                "reciprocal_st_depression_leads": reciprocal_std,
            },
        ),
        _row(
            "CLIN-OTHER-DEXTRO-01",
            "dextrocardia_pattern",
            "ECG pattern raises concern for dextrocardia; verify lead placement and obtain right-sided leads",
            dextrocardia,
            dextro_available,
            {
                "qrs_signed_area_I": qrs_i,
                "qrs_signed_area_aVR": qrs_avr,
                "r_v1_mv": r_v1,
                "r_v6_mv": r_v6,
                "limb_reversal": bool(getattr(context, "limb_reversal", False)),
            },
            severity="abnormal",
            priority="P2",
        ),
        _row(
            "CLIN-OTHER-PE-01",
            "pulmonary_embolism_pattern",
            "S1Q3T3/right-heart-strain pattern; consider pulmonary embolism in the appropriate clinical context",
            pe_pattern,
            pe_available,
            {
                "heart_rate_bpm": hr,
                "s_I_mv": s_i,
                "q_III_mv": q_iii,
                "t_III_mv": t_iii,
                "qrs_axis_deg": axis,
            },
            severity="abnormal",
            priority="P2",
        ),
        _row(
            "CLIN-OTHER-DIGITALIS-01",
            "digitalis_effect_pattern",
            "Digitalis-effect ST pattern",
            digitalis_pattern,
            digitalis_available and bool(meds),
            {
                "downsloping_st_depression_leads": digitalis_leads,
                "medications": list(meds),
                "digitalis_exposure": digitalis_exposure,
            },
        ),
    ]

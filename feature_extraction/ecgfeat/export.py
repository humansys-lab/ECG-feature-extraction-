from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from math import isfinite, log2, sqrt
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .interpret import (
    LAE_P_DUR_MS,
    LAE_V1_NEG_AMP_MV,
    LAE_V1_NEG_DUR_MS,
    PEDS_DEXTRO_P_AXIS_HIGH,
    PEDS_DEXTRO_P_AXIS_LOW,
    PEDS_DEXTRO_S_MV,
    PEDS_EARLY_REPOL_AGE_HIGH,
    PEDS_EARLY_REPOL_AGE_LOW,
    PEDS_LSH_CONSIDER_R_V1_MV,
    PEDS_LSH_R_V1_MV,
    PEDS_PERICARDITIS_AGE_HIGH,
    PEDS_PERICARDITIS_AGE_LOW,
    PEDS_RBBB_R_PRIME_DUR_MS,
    PEDS_RBBB_R_PRIME_MV,
    PTF_V1_DEFINITE_MV_MS,
    PTF_V1_PROBABLE_MV_MS,
    RAE_P_AMP_CONSIDER_MV,
    RAE_P_DUR_MIN_MS,
    ST_DEP_POSTERIOR_MV,
    ST_DEP_SIGNIFICANT_MV,
    ST_ELE_ABNORMAL_MV,
    ST_ELE_BORDERLINE_MV,
    TALL_T_ABS_MV,
    TALL_T_REL_MV,
)
from .mi import build_culprit_artery_evidence
from .models import ECGFeatures, LeadBeatFeatures, STANDARD_12_LEADS
from .pediatric_rules import (
    PEDS_BVH_Q_V6_AMP_MV,
    PEDS_BVH_Q_V6_DUR_MS,
    PEDS_BVH_R_V1_MV,
    PEDS_BVH_R_V6_MV,
    PEDS_BVH_RS_SUM_MV,
    REQUIRED_PEDIATRIC_CRITERIA,
    pediatric_age_bin,
    pediatric_voltage_threshold,
)
from .statement_engine import build_morphology_statement_evidence, resolve_statement_candidates


def _fingerprint_value(value: Any) -> Any:
    """Return a deterministic JSON-safe value for clinical artifact hashing."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {
            str(key): _fingerprint_value(item)
            for key, item in value.items()
            if key not in {"generated_at", "artifact_fingerprint"}
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def clinical_fingerprint(analysis: Any) -> str:
    canonical = json.dumps(
        _fingerprint_value(analysis),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _clinical_export_payload(features: ECGFeatures) -> Dict[str, Any]:
    raw = features.metadata.get("clinical_interpretation", {})
    if not isinstance(raw, dict):
        return {}
    payload = to_dict(raw)
    payload["artifact_fingerprint"] = clinical_fingerprint(payload)
    return payload


_NATIVE_BEAT_PROFILE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "qrs_infarct": (
        "qrs_ms",
        "q_duration_ms",
        "q_amp_mv",
        "q_r_ratio",
        "r_amp_mv",
        "s_amp_mv",
        "initial_qrs_net_mv",
        "qrs_confidence",
        "beat_measurement_reliable",
    ),
    "repolarization": (
        "qt_ms",
        "qt_confidence",
        "st_hybrid_j_mv",
        "st_hybrid_60ms_mv",
        "st_hybrid_80ms_mv",
        "st_hybrid_slope_mv_per_ms",
        "st_hybrid_shape",
        "st_hybrid_reliable",
        "st_hybrid_unreliable_reason",
        "t_amp_mv",
        "t_polarity",
        "t_symmetry",
        "beat_measurement_reliable",
    ),
}


def _build_native_beat_profiles(
    features: ECGFeatures,
    by_beat: Dict[int, List[LeadBeatFeatures]],
) -> Dict[str, Any]:
    """Persist a compact, diagnosis-neutral multibeat morphology contract.

    Full ``beat_features`` are optional in on-disk exports.  These selected
    fields remain available so a diagnostic agent can still inspect repeated
    native-beat Q/R progression and repolarization instead of relying on a
    single representative template.
    """
    beat_annotations = {
        int(beat.beat_id): beat
        for beat in features.beats
    }
    all_fields = tuple(
        dict.fromkeys(
            field_name
            for profile_fields in _NATIVE_BEAT_PROFILE_FIELDS.values()
            for field_name in profile_fields
        )
    )
    rows: List[Dict[str, Any]] = []
    lead_rank = {
        lead: index
        for index, lead in enumerate(STANDARD_12_LEADS)
    }
    for beat_id in sorted(by_beat):
        annotation = beat_annotations.get(int(beat_id))
        group_id = getattr(annotation, "group_id", None)
        paced = bool(getattr(annotation, "paced", False))
        for item in sorted(
            by_beat[beat_id],
            key=lambda row: (lead_rank.get(row.lead, len(lead_rank)), row.lead),
        ):
            row: Dict[str, Any] = {
                "beat_id": int(beat_id),
                "group_id": int(group_id) if group_id is not None else None,
                "paced": paced,
                "lead": item.lead,
            }
            for field_name in all_fields:
                row[field_name] = getattr(item, field_name, None)
            rows.append(row)
    return {
        "available": bool(rows),
        "source": "ecgfeat_per_beat_compact_measurements",
        "profiles": {
            profile: list(fields)
            for profile, fields in _NATIVE_BEAT_PROFILE_FIELDS.items()
        },
        "rows": rows,
    }


def _unavailable(reason: str = "not_implemented") -> Dict[str, Any]:
    return {"available": False, "value": None, "reason": reason}


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if isfinite(f) else None


def _median(values: Iterable[Any]) -> Optional[float]:
    clean = [_finite(v) for v in values]
    vals = [v for v in clean if v is not None]
    return float(median(vals)) if vals else None


def _interp_value(features: ECGFeatures, key: str, default: Any = None) -> Any:
    interp = features.interpretation
    if interp is None:
        return default
    if isinstance(interp, dict):
        return interp.get(key, default)
    return getattr(interp, key, default)


def _patient_value(features: ECGFeatures, key: str) -> Any:
    patient = features.metadata.get("patient_meta", {})
    if isinstance(patient, dict):
        return patient.get(key)
    return getattr(patient, key, None)


def _features_by_beat(features: ECGFeatures) -> Dict[int, List[LeadBeatFeatures]]:
    by_beat: Dict[int, List[LeadBeatFeatures]] = {}
    for bf in features.beat_features:
        by_beat.setdefault(int(bf.beat_id), []).append(bf)
    return by_beat


def _features_by_lead(features: ECGFeatures) -> Dict[str, List[LeadBeatFeatures]]:
    by_lead: Dict[str, List[LeadBeatFeatures]] = {}
    for bf in features.beat_features:
        by_lead.setdefault(bf.lead, []).append(bf)
    return by_lead


def _lead_numeric(
    features: ECGFeatures,
    by_lead: Dict[str, List[LeadBeatFeatures]],
    lead: str,
    key: str,
) -> Optional[float]:
    rep = features.representative_leads.get(lead)
    if rep is not None:
        value = _finite(rep.params.get(key))
        if value is not None:
            return value
    return _median(getattr(bf, key, None) for bf in by_lead.get(lead, []))


_P_FINE_BOOL_KEYS = {"p_notched", "p_biphasic"}


def _has_p_fine_measurement(item: LeadBeatFeatures) -> bool:
    bounds = item.p
    if bounds.onset is not None and bounds.peak is not None and bounds.offset is not None:
        return True
    if bool(item.p_notched) or bool(item.p_biphasic):
        return True
    numeric_attrs = (
        "p_dur_ms",
        "p_area",
        "p_notch_interval_ms",
        "p_initial_duration_ms",
        "p_initial_amp_mv",
        "p_terminal_duration_ms",
        "p_terminal_amp_mv",
        "p_terminal_area_mv_ms",
    )
    return any(_finite(getattr(item, attr, None)) is not None for attr in numeric_attrs)


def _lead_bool(
    features: ECGFeatures,
    by_lead: Dict[str, List[LeadBeatFeatures]],
    lead: str,
    key: str,
) -> Optional[bool]:
    rep = features.representative_leads.get(lead)
    if rep is not None and key in rep.params:
        value = rep.params.get(key)
        if value is not None:
            return bool(value)
        return None
    items = by_lead.get(lead, [])
    if key in _P_FINE_BOOL_KEYS:
        items = [item for item in items if _has_p_fine_measurement(item)]
    if items:
        return any(bool(getattr(bf, key, False)) for bf in items)
    return None


def _lead_numeric_first(
    features: ECGFeatures,
    by_lead: Dict[str, List[LeadBeatFeatures]],
    lead: str,
    *keys: str,
) -> Optional[float]:
    rep = features.representative_leads.get(lead)
    if rep is not None:
        for key in keys:
            value = _finite(rep.params.get(key))
            if value is not None:
                return value
    for key in keys:
        value = _median(getattr(bf, key, None) for bf in by_lead.get(lead, []))
        if value is not None:
            return value
    return None


def _lead_param_value(features: ECGFeatures, lead: str, key: str) -> Any:
    rep = features.representative_leads.get(lead)
    if rep is not None and key in rep.params:
        return rep.params.get(key)
    return None


def _lead_profile_bool(features: ECGFeatures, lead: str, key: str) -> Optional[bool]:
    value = _lead_param_value(features, lead, key)
    if value is None:
        return None
    return bool(value)


def _duration_between_ms(later: Any, earlier: Any) -> Optional[float]:
    later_ms = _finite(later)
    earlier_ms = _finite(earlier)
    if later_ms is None or earlier_ms is None:
        return None
    value = later_ms - earlier_ms
    return float(value) if value > 0.0 else None


def _qtc_bazett_from_profile(qt_ms: Optional[float], hr_bpm: Optional[float]) -> Optional[float]:
    qt = _finite(qt_ms)
    hr = _finite(hr_bpm)
    if qt is None or hr is None or hr <= 0.0:
        return None
    return float(qt * sqrt(hr / 60.0))


def _qtc_fridericia_from_profile(qt_ms: Optional[float], hr_bpm: Optional[float]) -> Optional[float]:
    qt = _finite(qt_ms)
    hr = _finite(hr_bpm)
    if qt is None or hr is None or hr <= 0.0:
        return None
    return float(qt * ((hr / 60.0) ** (1.0 / 3.0)))


def _native_global_measurements(features: ECGFeatures) -> Dict[str, Any]:
    gf = features.global_features
    return {
        "available": True,
        "heart_rate_bpm": _finite(gf.heart_rate_bpm),
        "pr_ms": _finite(gf.pr_ms),
        "qrs_duration_ms": _finite(gf.qrs_ms),
        "qt_ms": _finite(gf.qt_ms),
        "qtc_bazett_ms": _finite(gf.qtc_bazett_ms),
        "qtc_fridericia_ms": _finite(gf.qtc_fridericia_ms),
        "qt_dispersion_ms": _finite(gf.qt_dispersion_ms),
        "p_axis_frontal_deg": _finite(gf.p_axis_deg),
        "qrs_axis_frontal_deg": _finite(gf.qrs_axis_deg),
        "t_axis_frontal_deg": _finite(gf.t_axis_deg),
        "st_axis_frontal_deg": _finite(gf.st_axis_deg),
        "heart_rate_source": "native",
        "interval_source": "native",
        "qtc_source": "native",
        "qt_dispersion_source": "native",
        "axis_source": "native",
    }


def _twelve_sl_global_measurements(profile: Dict[str, Any]) -> Dict[str, Any]:
    fiducials = profile.get("global_fiducials") if isinstance(profile, dict) else None
    fiducials = fiducials if isinstance(fiducials, dict) else {}
    hr = _finite(profile.get("heart_rate_first_last_bpm"))
    p_on = _finite(fiducials.get("p_onset_offset_ms"))
    qrs_on = _finite(fiducials.get("qrs_onset_offset_ms"))
    qrs_off = _finite(fiducials.get("qrs_offset_offset_ms"))
    t_off = _finite(fiducials.get("t_offset_offset_ms"))
    pr_ms = _duration_between_ms(qrs_on, p_on)
    qrs_ms = _duration_between_ms(qrs_off, qrs_on)
    qt_ms = _duration_between_ms(t_off, qrs_on)
    interval_available = any(value is not None for value in (pr_ms, qrs_ms, qt_ms))
    return {
        "available": bool(hr is not None or interval_available),
        "heart_rate_bpm": hr,
        "pr_ms": pr_ms,
        "qrs_duration_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_bazett_ms": _qtc_bazett_from_profile(qt_ms, hr),
        "qtc_fridericia_ms": _qtc_fridericia_from_profile(qt_ms, hr),
        "qt_dispersion_ms": None,
        "p_axis_frontal_deg": None,
        "qrs_axis_frontal_deg": None,
        "t_axis_frontal_deg": None,
        "st_axis_frontal_deg": None,
        "heart_rate_source": "12sl_first_last_qrs" if hr is not None else "unavailable",
        "interval_source": "12sl_global_fiducials" if interval_available else "unavailable",
        "qtc_source": "derived_from_12sl_hr_qt" if qt_ms is not None and hr is not None else "unavailable",
        "qt_dispersion_source": "unavailable",
        "axis_source": "unavailable",
    }


def _prefer_profile_value(primary: Dict[str, Any], fallback: Dict[str, Any], key: str) -> Any:
    value = primary.get(key)
    return value if value is not None else fallback.get(key)


def _hybrid_global_measurements(
    native: Dict[str, Any],
    twelve_sl: Dict[str, Any],
) -> Dict[str, Any]:
    hr = _prefer_profile_value(twelve_sl, native, "heart_rate_bpm")
    pr_ms = _prefer_profile_value(twelve_sl, native, "pr_ms")
    qrs_ms = _prefer_profile_value(twelve_sl, native, "qrs_duration_ms")
    qt_ms = _prefer_profile_value(twelve_sl, native, "qt_ms")
    qtc_b = _qtc_bazett_from_profile(qt_ms, hr)
    qtc_f = _qtc_fridericia_from_profile(qt_ms, hr)
    interval_from_12sl = any(
        twelve_sl.get(key) is not None
        for key in ("pr_ms", "qrs_duration_ms", "qt_ms")
    )
    return {
        "available": True,
        "heart_rate_bpm": hr,
        "pr_ms": pr_ms,
        "qrs_duration_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_bazett_ms": qtc_b if qtc_b is not None else native.get("qtc_bazett_ms"),
        "qtc_fridericia_ms": qtc_f if qtc_f is not None else native.get("qtc_fridericia_ms"),
        "qt_dispersion_ms": native.get("qt_dispersion_ms"),
        "p_axis_frontal_deg": native.get("p_axis_frontal_deg"),
        "qrs_axis_frontal_deg": native.get("qrs_axis_frontal_deg"),
        "t_axis_frontal_deg": native.get("t_axis_frontal_deg"),
        "st_axis_frontal_deg": native.get("st_axis_frontal_deg"),
        "heart_rate_source": (
            twelve_sl.get("heart_rate_source")
            if twelve_sl.get("heart_rate_bpm") is not None
            else native.get("heart_rate_source")
        ),
        "interval_source": "12sl_global_fiducials" if interval_from_12sl else "native",
        "qtc_source": "derived_from_profile_hr_qt" if qtc_b is not None or qtc_f is not None else "native",
        "qt_dispersion_source": native.get("qt_dispersion_source"),
        "axis_source": native.get("axis_source"),
    }


def _lead_text(
    features: ECGFeatures,
    by_lead: Dict[str, List[LeadBeatFeatures]],
    lead: str,
    key: str,
) -> Optional[str]:
    rep = features.representative_leads.get(lead)
    if rep is not None:
        value = rep.params.get(key)
        if value is not None:
            return str(value)
    counts: Dict[str, int] = {}
    for bf in by_lead.get(lead, []):
        value = getattr(bf, key, None)
        if value is not None:
            counts[str(value)] = counts.get(str(value), 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _lead_st_j_point(
    features: ECGFeatures,
    by_lead: Dict[str, List[LeadBeatFeatures]],
    lead: str,
) -> Optional[float]:
    rep = features.representative_leads.get(lead)
    if rep is not None:
        has_rep_status = any(
            key in rep.params
            for key in (
                "st_j_source",
                "j_point_source",
                "st_j_reliable",
                "st_j_unreliable",
                "st_j_unreliable_reason",
                "unreliable_reason",
            )
        )
        if has_rep_status:
            if bool(rep.params.get("st_j_unreliable", False)):
                return None
            if bool(rep.params.get("st_j_reliable", False)):
                return _finite(rep.params.get("st_on_mv"))
            if "st_on_mv" in rep.params:
                return _finite(rep.params.get("st_on_mv"))
    return _median(
        getattr(bf, "st_on_mv", None)
        for bf in by_lead.get(lead, [])
        if "st_j_unreliable" not in bf.flags
    )


def _st_j_status(
    features: ECGFeatures,
    by_lead: Dict[str, List[LeadBeatFeatures]],
    lead: str,
    j_point_mv: Optional[float],
) -> Tuple[str, bool, Optional[str]]:
    rep = features.representative_leads.get(lead)
    source: Optional[str] = None
    reason: Optional[str] = None
    reliable: Optional[bool] = None
    if rep is not None:
        has_rep_status = any(
            key in rep.params
            for key in (
                "st_j_source",
                "j_point_source",
                "st_j_reliable",
                "st_j_unreliable",
                "st_j_unreliable_reason",
                "unreliable_reason",
            )
        )
        source_value = rep.params.get("st_j_source") or rep.params.get("j_point_source")
        if source_value is not None:
            source = str(source_value)
        reason_value = rep.params.get("st_j_unreliable_reason") or rep.params.get("unreliable_reason")
        if reason_value is not None:
            reason = str(reason_value)
        if "st_j_reliable" in rep.params:
            reliable = bool(rep.params.get("st_j_reliable"))
        if bool(rep.params.get("st_j_unreliable", False)):
            reliable = False
            reason = reason or "qrs_tail_guard"
        if has_rep_status:
            if reliable is None:
                reliable = bool(j_point_mv is not None and not rep.params.get("st_j_unreliable", False))
            if source is None:
                source = "qrs_offset" if reliable else "unavailable"
            return source, reliable, reason

    beat_items = by_lead.get(lead, [])
    beat_unreliable = any("st_j_unreliable" in bf.flags for bf in beat_items)
    has_reliable_beat_value = any(
        "st_j_unreliable" not in bf.flags and _finite(getattr(bf, "st_on_mv", None)) is not None
        for bf in beat_items
    )
    if reliable is None and beat_unreliable:
        if j_point_mv is not None and has_reliable_beat_value:
            reliable = True
            source = source or "mixed_reliable_beats"
        else:
            reliable = False
            reason = reason or "qrs_tail_guard"
            source = source or "qrs_tail_guard"
    if reliable is None:
        reliable = j_point_mv is not None and not (j_point_mv is None and beat_unreliable)
    if reason is None and not reliable and beat_unreliable:
        reason = "qrs_tail_guard"
    if source is None:
        source = "qrs_offset" if reliable else (reason or "unavailable")
    return source, reliable, reason


def _lead_measurement_profiles(
    features: ECGFeatures,
    by_lead: Dict[str, List[LeadBeatFeatures]],
    lead: str,
    st_j_point: Optional[float],
) -> Dict[str, Any]:
    native_st_mid = _lead_numeric(features, by_lead, lead, "st_mid_mv")
    native_st_80 = _lead_numeric(features, by_lead, lead, "st_80ms_mv")
    native_qrs_area = _lead_numeric(features, by_lead, lead, "qrs_area")
    native_qrs_signed = _lead_numeric(features, by_lead, lead, "qrs_signed_area")
    native_t_amp = _lead_numeric(features, by_lead, lead, "t_amp_mv")

    sl_stj = _finite(_lead_param_value(features, lead, "twelve_sl_stj_mv"))
    sl_stm = _finite(_lead_param_value(features, lead, "twelve_sl_stm_mv"))
    sl_ste = _finite(_lead_param_value(features, lead, "twelve_sl_ste_mv"))
    sl_st_confidence = _finite(_lead_param_value(features, lead, "twelve_sl_st_confidence"))
    sl_st_confidence_reason = _lead_param_value(features, lead, "twelve_sl_st_confidence_reason")
    sl_min_st = _finite(_lead_param_value(features, lead, "twelve_sl_minimum_st_uv"))
    sl_qrs_area = _finite(_lead_param_value(features, lead, "twelve_sl_qrs_area_uv_ms"))
    sl_qrs_signed = _finite(_lead_param_value(features, lead, "twelve_sl_qrs_signed_area_uv_ms"))
    sl_balance = _finite(_lead_param_value(features, lead, "twelve_sl_qrs_balance_uv"))
    sl_deflection = _finite(_lead_param_value(features, lead, "twelve_sl_qrs_deflection_uv"))
    sl_special_t = _finite(_lead_param_value(features, lead, "twelve_sl_special_t_uv"))
    sl_t_prime = _finite(_lead_param_value(features, lead, "twelve_sl_t_prime_uv"))
    sl_t_prime_area = _finite(_lead_param_value(features, lead, "twelve_sl_t_prime_area_uv_ms"))
    sl_significant = _lead_profile_bool(features, lead, "twelve_sl_qrs_significant")
    sl_available = any(
        value is not None
        for value in (
            sl_stj,
            sl_stm,
            sl_ste,
            sl_st_confidence,
            sl_min_st,
            sl_qrs_area,
            sl_qrs_signed,
            sl_balance,
            sl_deflection,
            sl_special_t,
            sl_t_prime,
            sl_t_prime_area,
            sl_significant,
        )
    )

    native = {
        "st": {
            "j_point_mV": st_j_point,
            "midpoint_mV": native_st_mid,
            "j80_mV": native_st_80,
            "source": "native",
        },
        "qrs": {
            "area_native": native_qrs_area,
            "signed_area_native": native_qrs_signed,
            "source": "native",
        },
        "t": {
            "amplitude_mV": native_t_amp,
            "source": "native",
        },
    }
    twelve_sl = {
        "available": sl_available,
        "st": {
            "stj_mV": sl_stj,
            "stm_mV": sl_stm,
            "ste_mV": sl_ste,
            "confidence": sl_st_confidence,
            "confidence_reason": sl_st_confidence_reason,
            "minimum_st_uV": sl_min_st,
            "source": "12sl_profile" if sl_available else "unavailable",
        },
        "qrs": {
            "area_uV_ms": sl_qrs_area,
            "signed_area_uV_ms": sl_qrs_signed,
            "balance_uV": sl_balance,
            "deflection_uV": sl_deflection,
            "significant": sl_significant,
            "source": "12sl_profile" if sl_available else "unavailable",
        },
        "t": {
            "special_t_uV": sl_special_t,
            "t_prime_uV": sl_t_prime,
            "t_prime_area_uV_ms": sl_t_prime_area,
            "source": (
                "12sl_profile"
                if any(value is not None for value in (sl_special_t, sl_t_prime, sl_t_prime_area))
                else "unavailable"
            ),
        },
    }

    st_uses_12sl = sl_stj is not None or sl_stm is not None or sl_ste is not None
    qrs_uses_12sl = (
        sl_qrs_area is not None
        or sl_qrs_signed is not None
        or sl_balance is not None
        or sl_deflection is not None
        or sl_significant is not None
    )
    t_uses_12sl = any(value is not None for value in (sl_special_t, sl_t_prime, sl_t_prime_area))
    return {
        "native": native,
        "12sl": twelve_sl,
        "hybrid": {
            "st": {
                "j_point_mV": sl_stj if sl_stj is not None else st_j_point,
                "midpoint_mV": sl_stm if sl_stm is not None else native_st_mid,
                "j80_mV": sl_ste if sl_ste is not None else native_st_80,
                "source": "12sl_profile" if st_uses_12sl else "native",
            },
            "qrs": {
                "area_uV_ms": sl_qrs_area,
                "area_native": native_qrs_area,
                "signed_area_uV_ms": sl_qrs_signed,
                "signed_area_native": native_qrs_signed,
                "balance_uV": sl_balance,
                "deflection_uV": sl_deflection,
                "significant": sl_significant,
                "source": "12sl_profile" if qrs_uses_12sl else "native",
            },
            "t": {
                "amplitude_mV": native_t_amp,
                "special_t_uV": sl_special_t,
                "t_prime_uV": sl_t_prime,
                "t_prime_area_uV_ms": sl_t_prime_area,
                "source": "12sl_profile" if t_uses_12sl else "native",
            },
        },
    }


def _bounds_duration_ms(items: List[LeadBeatFeatures], wave: str, fs: int) -> Optional[float]:
    values: List[float] = []
    if fs <= 0:
        return None
    for bf in items:
        bounds = getattr(bf, wave)
        if bounds.onset is None or bounds.offset is None:
            continue
        duration = _sample_to_ms(bounds.offset - bounds.onset, fs)
        if duration is not None and duration > 0:
            values.append(duration)
    return _median(values)


def _rr_values(features: ECGFeatures) -> List[float]:
    vals: List[float] = []
    for beat in features.beats:
        rr = _finite(beat.rr_prev_ms)
        if rr is not None and rr > 0:
            vals.append(rr)
    if vals:
        return vals
    for beat in features.beats:
        rr = _finite(beat.rr_next_ms)
        if rr is not None and rr > 0:
            vals.append(rr)
    return vals


def _rr_rmssd(rr_ms: List[float]) -> Optional[float]:
    if len(rr_ms) < 2:
        return None
    diffs = [rr_ms[i + 1] - rr_ms[i] for i in range(len(rr_ms) - 1)]
    return sqrt(sum(d * d for d in diffs) / len(diffs))


def _rr_entropy(rr_ms: List[float]) -> Optional[float]:
    if len(rr_ms) < 3:
        return None
    buckets: Dict[int, int] = {}
    for rr in rr_ms:
        bucket = int(round(rr / 20.0) * 20)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    total = float(sum(buckets.values()))
    return -sum((count / total) * log2(count / total) for count in buckets.values())


def _qrs_net_value(bf: LeadBeatFeatures) -> Optional[float]:
    signed = _finite(bf.qrs_signed_area)
    if signed is not None:
        return signed
    vals = [_finite(bf.q_amp_mv), _finite(bf.r_amp_mv), _finite(bf.s_amp_mv)]
    present = [v for v in vals if v is not None]
    return float(sum(present)) if present else None


def _polarity(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value > 0.05:
        return "positive"
    if value < -0.05:
        return "negative"
    return "isoelectric"


def _best_p_feature(items: List[LeadBeatFeatures]) -> Optional[LeadBeatFeatures]:
    if not items:
        return None
    return max(items, key=lambda bf: _finite(bf.p_confidence) or 0.0)


def _sample_to_ms(sample: Optional[int], fs: int) -> Optional[float]:
    if sample is None or fs <= 0:
        return None
    return float(sample) * 1000.0 / float(fs)


def _axis_angle_diff(a: Any, b: Any) -> Optional[float]:
    av = _finite(a)
    bv = _finite(b)
    if av is None or bv is None:
        return None
    return abs(((av - bv + 180.0) % 360.0) - 180.0)


def _qrs_peak_to_peak(q: Optional[float], r: Optional[float], s: Optional[float]) -> Optional[float]:
    values = [v for v in [q, r, s] if v is not None]
    if not values:
        return None
    return float(max(values) - min(values))


def _positive_component(q: Optional[float], r: Optional[float], s: Optional[float]) -> Optional[float]:
    positives = [v for v in [q, r, s] if v is not None and v > 0]
    return float(max(positives)) if positives else None


def _negative_component(q: Optional[float], r: Optional[float], s: Optional[float]) -> Optional[float]:
    negatives = [v for v in [q, r, s] if v is not None and v < 0]
    return float(min(negatives)) if negatives else None


def _q_duration_ms(items: List[LeadBeatFeatures], fs: int) -> Optional[float]:
    del fs
    values: List[float] = []
    for bf in items:
        duration = _finite(getattr(bf, "q_duration_ms", None))
        if duration is not None:
            values.append(duration)
    return _median(values)


def _q_duration_source(items: List[LeadBeatFeatures]) -> str:
    if any(_finite(getattr(bf, "q_duration_ms", None)) is not None for bf in items):
        return "measured_q_component"
    return "unavailable"


def _qrs_horizontal_direction(features: ECGFeatures, by_lead: Dict[str, List[LeadBeatFeatures]]) -> str:
    directions: List[str] = []
    for lead in ("V5", "V6"):
        r = _lead_numeric(features, by_lead, lead, "r_amp_mv")
        s = _lead_numeric(features, by_lead, lead, "s_amp_mv")
        if r is None or s is None:
            return "unknown"
        directions.append("rightward" if r < abs(s) else "leftward")
    if directions == ["rightward", "rightward"]:
        return "rightward"
    if directions == ["leftward", "leftward"]:
        return "leftward"
    return "unknown"


def _terminal_direction(lead: str, r: Optional[float], s: Optional[float]) -> str:
    if r is None or s is None:
        return "unknown"
    dominant_positive = r >= abs(s)
    if lead in {"I", "aVL", "V5", "V6"}:
        return "leftward" if dominant_positive else "rightward"
    if lead == "V1":
        return "rightward" if dominant_positive else "leftward"
    return "unknown"


def _t_polarity(t_amp: Optional[float], t_polarity: Optional[float]) -> str:
    if t_polarity is not None:
        if t_polarity > 0:
            return "positive"
        if t_polarity < 0:
            return "negative"
    if t_amp is None:
        return "unknown"
    if t_amp > 0.05:
        return "positive"
    if t_amp < -0.05:
        return "negative"
    return "flat"


def _list_value(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _normalize_atrial_p_event(event: Any) -> Dict[str, Any]:
    event_dict = to_dict(event)
    if not isinstance(event_dict, dict):
        event_dict = {}
    return {
        "p_event_id": event_dict.get("p_event_id"),
        "sample": event_dict.get("sample"),
        "time_ms": event_dict.get("time_ms"),
        "onset_ms": event_dict.get("onset_ms"),
        "offset_ms": event_dict.get("offset_ms"),
        "duration_ms": event_dict.get("duration_ms"),
        "amplitude_mv": event_dict.get("amplitude_mv"),
        "area_mv_ms": event_dict.get("area_mv_ms"),
        "signed_area_mv_ms": event_dict.get("signed_area_mv_ms"),
        "template_similarity": event_dict.get("template_similarity"),
        "pp_ms": event_dict.get("pp_ms"),
        "detection_method": event_dict.get("detection_method"),
        "confidence": event_dict.get("confidence"),
        "associated_qrs_beat_id": event_dict.get("associated_qrs_beat_id"),
        "association_type": event_dict.get("association_type"),
        "pr_ms": event_dict.get("pr_ms"),
        "axis_deg": event_dict.get("axis_deg"),
        "morphology": event_dict.get("morphology"),
        "source": event_dict.get("source", "independent_atrial_event_stream"),
        "source_leads": to_dict(event_dict.get("source_leads", [])),
        "boundary_source": event_dict.get("boundary_source"),
    }


# How far outside the fused P envelope an event may sit and still be the same
# wave. Covers boundary rounding without merging a deflection that merely shares
# a beat: mismatches seen in practice are either a few milliseconds out or a
# couple of hundred, with nothing in between.
_P_BOUNDARY_JOIN_TOLERANCE_MS = 20.0


def _attach_assessment_boundaries(
    p_events: List[Dict[str, Any]],
    assessments: Any,
    fs: int,
) -> None:
    """Give conducted atrial events the fused P boundaries measured for their beat.

    The atrial-event stream locates a P wave but never delineates it, so its
    conducted events ship with no onset or offset while the per-beat P
    assessment holds cross-lead fused boundaries for the same wave.

    Only boundaries are copied. Amplitude, area and shape stay out of this on
    purpose: they are already readable per lead through the morphology map, and
    a second copy under a different field name would let one measurement be
    cited twice as if two detector chains agreed. `boundary_source` records the
    provenance so a consumer can see these are the assessment's numbers rather
    than an independent measurement.
    """
    if not assessments:
        return

    by_beat: Dict[int, Dict[str, Any]] = {}
    for assessment in assessments:
        row = to_dict(assessment)
        if not isinstance(row, dict):
            continue
        beat_id = row.get("beat_id")
        if beat_id is None:
            continue
        by_beat[int(beat_id)] = row

    tolerance = _P_BOUNDARY_JOIN_TOLERANCE_MS * float(fs) / 1000.0
    for event in p_events:
        if event.get("association_type") != "conducted":
            # Assessments are QRS-centred, one per beat. A blocked or retrograde
            # event has no assessment of its own, so borrowing the beat's P
            # boundaries would invent a delineation for a different deflection.
            continue
        if event.get("onset_ms") is not None or event.get("offset_ms") is not None:
            continue
        beat_id = event.get("associated_qrs_beat_id")
        sample = event.get("sample")
        if beat_id is None or sample is None:
            continue
        row = by_beat.get(int(beat_id))
        if row is None or not row.get("accepted"):
            # A rejected assessment carries null boundaries; leaving the event
            # bare is the honest result.
            continue

        envelope_onset = row.get("strict_onset")
        envelope_offset = row.get("strict_offset")
        onset = row.get("robust_onset")
        offset = row.get("robust_offset")
        if onset is None or offset is None or envelope_onset is None or envelope_offset is None:
            continue
        # Sharing a beat is not enough. An event can be associated with a beat
        # and still be a different deflection a couple of hundred milliseconds
        # away, so require it to land inside the widest per-lead envelope.
        if not (
            float(envelope_onset) - tolerance
            <= float(sample)
            <= float(envelope_offset) + tolerance
        ):
            continue

        event["onset_ms"] = _sample_to_ms(int(onset), fs)
        event["offset_ms"] = _sample_to_ms(int(offset), fs)
        event["duration_ms"] = (int(offset) - int(onset)) * 1000.0 / float(fs)
        event["boundary_source"] = "p_wave_assessment_fusion"


PEDS_MAX_AGE_YEARS = 16.0

_PEDS_AGE_BUCKETS: List[Tuple[float, float, str]] = [
    (0.0, 1.0 / 365.0, "0_23_hours"),
    (1.0 / 365.0, 4.0 / 365.0, "1_3_days"),
    (4.0 / 365.0, 7.0 / 365.0, "4_6_days"),
    (7.0 / 365.0, 30.0 / 365.0, "7_29_days"),
    (30.0 / 365.0, 91.0 / 365.0, "1_2_months"),
    (91.0 / 365.0, 182.0 / 365.0, "3_5_months"),
    (182.0 / 365.0, 1.0, "6_11_months"),
    (1.0, 3.0, "1_2_years"),
    (3.0, 5.0, "3_4_years"),
    (5.0, 8.0, "5_7_years"),
    (8.0, 12.0, "8_11_years"),
    (12.0, 16.0, "12_15_years"),
]


def _peds_table(values: List[float]) -> List[Tuple[float, float, float]]:
    return [(low, high, float(value)) for (low, high, _label), value in zip(_PEDS_AGE_BUCKETS, values)]


_PEDS_QRS_NORMAL_MS = _peds_table([70, 70, 70, 70, 84, 84, 84, 78, 88, 88, 88, 100])
_PEDS_LAD_THRESHOLD = _peds_table([54, 54, 54, 54, 20, -6, -6, -6, -10, -10, -10, -15])
_PEDS_BLAD_UPPER = _peds_table([65, 65, 65, 65, 30, 1, 1, 1, 1, 1, 1, 1])
_PEDS_RAD_THRESHOLD_360 = _peds_table([216, 216, 216, 216, 131, 131, 131, 131, 146, 201, 151, 161])
_PEDS_BRAD_THRESHOLD_360 = _peds_table([205, 205, 205, 200, 115, 115, 115, 115, 126, 160, 135, 145])


def _age_route(features: ECGFeatures) -> Tuple[str, str, Optional[float], bool]:
    age = _finite(_patient_value(features, "age"))
    age_valid = age is not None and age >= 0.0
    if not age_valid:
        return "adult", "age_missing_or_invalid_default_to_adult", age, False
    if age < PEDS_MAX_AGE_YEARS:
        return "pediatric", "age_0_to_under_16", age, True
    return "adult", "age_16_or_older", age, True


def _peds_lookup(table: List[Tuple[float, float, float]], age_years: Optional[float]) -> Optional[float]:
    if age_years is None:
        return None
    for low, high, value in table:
        if low <= age_years < high:
            return value
    return None


def _peds_age_bucket(age_years: Optional[float]) -> Optional[str]:
    if age_years is None:
        return None
    for low, high, label in _PEDS_AGE_BUCKETS:
        if low <= age_years < high:
            return label
    return None


def _sex_is_female(sex: Any) -> Optional[bool]:
    if sex is None:
        return None
    text = str(sex).strip().lower()
    if text in {"f", "female", "woman", "girl"}:
        return True
    if text in {"m", "male", "man", "boy"}:
        return False
    return None


def _round_ms(value: Optional[float]) -> Optional[float]:
    return round(value, 1) if value is not None else None


def _peds_qrs_width_class(qrs_ms: Optional[float], normal_ms: Optional[float]) -> str:
    if qrs_ms is None or normal_ms is None:
        return "indeterminate"
    borderline_ms = normal_ms * 1.10
    nonspecific_ms = normal_ms * 1.20
    if qrs_ms > nonspecific_ms:
        return "nonspecific_ivcd_range"
    if qrs_ms > borderline_ms:
        return "borderline_ivcd_range"
    if qrs_ms > normal_ms:
        return "above_normal_below_ivcd"
    return "normal"


def _qtc_class(
    qtc_ms: Optional[float],
    short_ms: float,
    borderline_prolonged_ms: float,
    prolonged_ms: float,
    severe_ms: float,
) -> str:
    if qtc_ms is None:
        return "indeterminate"
    if qtc_ms < short_ms:
        return "borderline_short"
    if qtc_ms > severe_ms:
        return "significantly_prolonged"
    if qtc_ms > prolonged_ms:
        return "prolonged"
    if qtc_ms > borderline_prolonged_ms:
        return "borderline_prolonged"
    return "normal"


def _peds_qtc_limits(age_years: Optional[float], sex: Any) -> Dict[str, Any]:
    female = _sex_is_female(sex)
    if age_years is None:
        borderline = None
        prolonged = None
        age_band = "unknown"
    elif age_years < 5.0:
        borderline = 450.0
        prolonged = 470.0
        age_band = "under_5"
    elif age_years < 13.0:
        borderline = 454.0
        prolonged = 474.0
        age_band = "5_to_12"
    elif female is True:
        borderline = 465.0
        prolonged = 485.0
        age_band = "13_to_15_female"
    else:
        borderline = 458.0
        prolonged = 478.0
        age_band = "13_to_15_male_or_unknown_sex"
    return {
        "age_band": age_band,
        "sex_is_female": female,
        "short_ms": 340.0,
        "borderline_prolonged_ms": borderline,
        "prolonged_ms": prolonged,
        "severe_ms": 520.0,
    }


def _adult_morphology_context(features: ECGFeatures, algorithm_age_group: str, age_years: Optional[float]) -> Dict[str, Any]:
    gf = features.global_features
    qtc_ms = _finite(gf.qtc_bazett_ms)
    return {
        "algorithm": "adult_morphology",
        "route_selected": algorithm_age_group == "adult",
        "is_adult": bool(age_years is not None and age_years >= PEDS_MAX_AGE_YEARS),
        "applies_for_age_years": {"min_inclusive": PEDS_MAX_AGE_YEARS},
        "axis_limits": {
            "qrs_normal_low_deg": -30.0,
            "qrs_normal_high_deg": 90.0,
            "lad_cutoff_deg": -30.0,
            "lafb_cutoff_deg": -40.0,
            "rad_cutoff_deg": 90.0,
            "lpfb_low_deg": 120.0,
            "lpfb_high_deg": 210.0,
            "p_axis_sinus_low_deg": -30.0,
            "p_axis_sinus_high_deg": 120.0,
            "t_axis_normal_low_deg": -10.0,
            "t_axis_normal_high_deg": 100.0,
            "qrs_t_angle_abnormal_deg": 90.0,
            "current_qrs_axis_class": _interp_value(features, "qrs_axis_class"),
        },
        "qrs_duration_limits": {
            "borderline_low_ms": 100.0,
            "borderline_high_ms": 110.0,
            "nonspecific_ivcd_ms": 120.0,
            "bundle_branch_block_ms": 120.0,
            "current_qrs_ms": _finite(gf.qrs_ms),
            "current_qrs_width_class": _interp_value(features, "qrs_width_class"),
        },
        "qtc_limits": {
            "short_ms": 340.0,
            "borderline_prolonged_ms": 465.0,
            "prolonged_ms": 485.0,
            "severe_ms": 520.0,
            "current_qtc_ms": qtc_ms,
            "current_qtc_class": _qtc_class(qtc_ms, 340.0, 465.0, 485.0, 520.0),
        },
        "atrial_enlargement_thresholds": {
            "rae_p_amp_consider_mV": 0.24,
            "rae_p_amp_confirm_mV": 0.25,
            "lae_p_duration_ms": 110.0,
            "lae_v1_negative_amp_mV": 0.09,
            "lae_v1_negative_duration_ms": 30.0,
        },
    }


def _pediatric_morphology_context(
    features: ECGFeatures,
    algorithm_age_group: str,
    age_years: Optional[float],
    sex: Any,
) -> Dict[str, Any]:
    gf = features.global_features
    is_pediatric = bool(age_years is not None and 0.0 <= age_years < PEDS_MAX_AGE_YEARS)
    qrs_ms = _finite(gf.qrs_ms)
    qtc_ms = _finite(gf.qtc_bazett_ms)
    qrs_normal = _peds_lookup(_PEDS_QRS_NORMAL_MS, age_years) if is_pediatric else None
    qtc_limits = _peds_qtc_limits(age_years if is_pediatric else None, sex)
    qtc_current_class = "indeterminate"
    if qtc_limits["borderline_prolonged_ms"] is not None and qtc_limits["prolonged_ms"] is not None:
        qtc_current_class = _qtc_class(
            qtc_ms,
            qtc_limits["short_ms"],
            qtc_limits["borderline_prolonged_ms"],
            qtc_limits["prolonged_ms"],
            qtc_limits["severe_ms"],
        )

    voltage_tables: Dict[str, Any]
    if is_pediatric:
        thresholds: Dict[str, float] = {}
        unavailable_criteria: List[str] = []
        for criterion in REQUIRED_PEDIATRIC_CRITERIA:
            threshold = pediatric_voltage_threshold(float(age_years), sex, criterion)
            if threshold is None:
                unavailable_criteria.append(criterion)
            else:
                thresholds[criterion] = threshold
        voltage_tables = {
            "available": True,
            "source": "dxl_appendix_a_local_digitized_thresholds",
            "age_bin": pediatric_age_bin(float(age_years)),
            "thresholds": thresholds,
            "unavailable_criteria": unavailable_criteria,
        }
    else:
        voltage_tables = _unavailable("age_outside_0_to_under_16_or_missing")

    return {
        "algorithm": "pediatric_morphology",
        "route_selected": algorithm_age_group == "pediatric",
        "is_pediatric": is_pediatric,
        "applies_for_age_years": {"min_inclusive": 0.0, "max_exclusive": PEDS_MAX_AGE_YEARS},
        "age_bucket": _peds_age_bucket(age_years) if is_pediatric else None,
        "not_applicable_reason": None if is_pediatric else "age_outside_0_to_under_16_or_missing",
        "axis_limits": {
            "lad_threshold_deg": _peds_lookup(_PEDS_LAD_THRESHOLD, age_years) if is_pediatric else None,
            "borderline_lad_upper_deg": _peds_lookup(_PEDS_BLAD_UPPER, age_years) if is_pediatric else None,
            "rad_threshold_360_deg": _peds_lookup(_PEDS_RAD_THRESHOLD_360, age_years) if is_pediatric else None,
            "borderline_rad_threshold_360_deg": _peds_lookup(_PEDS_BRAD_THRESHOLD_360, age_years) if is_pediatric else None,
            "rad_zone_max_360_deg": 269.0,
            "current_qrs_axis_class": _interp_value(features, "qrs_axis_class"),
        },
        "qrs_duration_limits": {
            "normal_limit_ms": qrs_normal,
            "borderline_ivcd_ms": _round_ms(qrs_normal * 1.10) if qrs_normal is not None else None,
            "nonspecific_ivcd_ms": _round_ms(qrs_normal * 1.20) if qrs_normal is not None else None,
            "current_qrs_ms": qrs_ms,
            "current_qrs_width_class": _peds_qrs_width_class(qrs_ms, qrs_normal),
            "rbbb_v1_r_prime_amp_min_mV": PEDS_RBBB_R_PRIME_MV,
            "rbbb_v1_r_prime_duration_min_ms": PEDS_RBBB_R_PRIME_DUR_MS,
        },
        "qtc_limits": {
            **qtc_limits,
            "current_qtc_ms": qtc_ms,
            "current_qtc_class": qtc_current_class,
            "prolonged_qtc_suppressed_by": ["RVH", "LSH", "LVH", "BVH", "VCD"],
        },
        "dextrocardia_thresholds": {
            "p_axis_low_deg": PEDS_DEXTRO_P_AXIS_LOW,
            "p_axis_high_deg": PEDS_DEXTRO_P_AXIS_HIGH,
            "large_s_in_i_and_v6_mV": PEDS_DEXTRO_S_MV,
        },
        "atrial_enlargement_thresholds": {
            # RAE/LAE use the shared, non-age-specific P-wave morphology logic
            # (interpret._p_wave_morphology); values mirror the constants it
            # enforces.  A former pediatric-only rae_p_amp_mV=0.20 (dead
            # constant) and an unimplemented "ashman area" key were removed.
            "rae_p_amp_mV": RAE_P_AMP_CONSIDER_MV,
            "rae_p_duration_ms": RAE_P_DUR_MIN_MS,
            "lae_p_duration_ms": LAE_P_DUR_MS,
            "lae_v1_terminal_negative_amp_mV": LAE_V1_NEG_AMP_MV,
            "lae_v1_terminal_negative_duration_ms": LAE_V1_NEG_DUR_MS,
            "lae_ptf_v1_probable_mV_ms": PTF_V1_PROBABLE_MV_MS,
            "lae_ptf_v1_definite_mV_ms": PTF_V1_DEFINITE_MV_MS,
            "requires_notched_or_biphasic_p_context": True,
        },
        "hypertrophy_thresholds": {
            "rvh_lvh_age_percentile_tables": voltage_tables,
            "lsh_r_v1_prominent_mV": PEDS_LSH_R_V1_MV,
            "lsh_r_v1_consider_mV": PEDS_LSH_CONSIDER_R_V1_MV,
            "bvh_r_v1_with_lvh_mV": PEDS_BVH_R_V1_MV,
            "bvh_q_v6_duration_ms": PEDS_BVH_Q_V6_DUR_MS,
            "bvh_q_v6_amplitude_mV": PEDS_BVH_Q_V6_AMP_MV,
            "bvh_r_v6_mV": PEDS_BVH_R_V6_MV,
            "bvh_rs_sum_v2_v3_v4_mV": PEDS_BVH_RS_SUM_MV,
        },
        "repolarization_thresholds": {
            # ST elevation/depression lead sets are produced by the shared
            # J-point analysis (interpret._st_analysis); the pediatric and adult
            # routes use the same cutoffs — there is no pediatric-specific ST
            # threshold in this implementation.  Values mirror the constants that
            # actually drive the classification so this block cannot drift.
            "classification_source": "shared_j_point_st_analysis",
            "st_elevation_borderline_mV": ST_ELE_BORDERLINE_MV,
            "st_elevation_abnormal_mV": ST_ELE_ABNORMAL_MV,
            "st_depression_significant_mV": ST_DEP_SIGNIFICANT_MV,
            "st_depression_posterior_reciprocal_mV": abs(ST_DEP_POSTERIOR_MV),
            "pericarditis_age_low_years": PEDS_PERICARDITIS_AGE_LOW,
            "pericarditis_age_high_years": PEDS_PERICARDITIS_AGE_HIGH,
            "early_repolarization_age_low_years": PEDS_EARLY_REPOL_AGE_LOW,
            "early_repolarization_age_high_years": PEDS_EARLY_REPOL_AGE_HIGH,
            "tall_t_amplitude_mV": TALL_T_ABS_MV,
            "tall_t_relative_mV": TALL_T_REL_MV,
            "secondary_repolarization_suppressed_by": ["hypertrophy", "VCD"],
        },
        "summary_flags": {
            "qrs_axis_class": _interp_value(features, "qrs_axis_class"),
            "qrs_width_class": _interp_value(features, "qrs_width_class"),
            "bundle_branch_block": _interp_value(features, "bundle_branch_block"),
            "rvh_class": _interp_value(features, "rvh_class"),
            "lvh_class": _interp_value(features, "lvh_class"),
            "lsh_class": _interp_value(features, "lsh_class"),
            "bvh_suspected": bool(_interp_value(features, "bvh_suspected", False)),
            "qtc_class": _interp_value(features, "qtc_class", qtc_current_class),
        },
    }


def build_morphology_inputs(features: ECGFeatures) -> Dict[str, Any]:
    """Build the feature schema consumed by the morphology rule layer."""
    by_lead = _features_by_lead(features)
    gf = features.global_features
    record_quality = features.metadata.get("record_quality", {})
    twelve_sl_profile = features.metadata.get("twelve_sl_measurement_profile")
    twelve_sl_available = (
        isinstance(twelve_sl_profile, dict)
        and bool(twelve_sl_profile.get("profile_version"))
    )
    native_global_profile = _native_global_measurements(features)
    twelve_sl_global_profile = (
        _twelve_sl_global_measurements(twelve_sl_profile)
        if twelve_sl_available
        else {
            "available": False,
            "reason": "12sl_profile_not_available",
            "heart_rate_bpm": None,
            "pr_ms": None,
            "qrs_duration_ms": None,
            "qt_ms": None,
            "qtc_bazett_ms": None,
            "qtc_fridericia_ms": None,
            "qt_dispersion_ms": None,
            "p_axis_frontal_deg": None,
            "qrs_axis_frontal_deg": None,
            "t_axis_frontal_deg": None,
            "st_axis_frontal_deg": None,
            "heart_rate_source": "unavailable",
            "interval_source": "unavailable",
            "qtc_source": "unavailable",
            "qt_dispersion_source": "unavailable",
            "axis_source": "unavailable",
        }
    )
    hybrid_global_profile = _hybrid_global_measurements(
        native_global_profile,
        twelve_sl_global_profile,
    )
    lead_order = list(features.metadata.get("lead_order") or features.representative_leads.keys())
    available_extended_leads = [lead for lead in lead_order if lead not in STANDARD_12_LEADS]
    qrs_t_angle = _interp_value(features, "qrs_t_angle_deg")
    if qrs_t_angle is None:
        qrs_t_angle = _axis_angle_diff(gf.qrs_axis_deg, gf.t_axis_deg)

    clinical_dx_codes = _list_value(features.metadata.get("clinical_dx_codes"))
    rx_codes = _list_value(features.metadata.get("rx_codes"))
    if not rx_codes:
        rx_codes = _list_value(_patient_value(features, "meds"))
    algorithm_age_group, algorithm_age_reason, age_years, age_valid = _age_route(features)
    sex = _patient_value(features, "sex")
    mi_evidence = _interp_value(features, "mi_evidence", {"available": False, "reason": "not_computed"})
    mi_territories = mi_evidence.get("territories", {}) if isinstance(mi_evidence, dict) else {}
    st_elevation_leads = _interp_value(features, "st_elevation_leads", {}) or {}
    st_depression_leads = _interp_value(features, "st_depression_leads", {}) or {}
    culprit_artery_evidence = build_culprit_artery_evidence(
        st_elevation_leads=st_elevation_leads,
        st_depression_leads=st_depression_leads,
        available_extended_leads=available_extended_leads,
    )

    leads: Dict[str, Any] = {}
    for lead in lead_order:
        items = by_lead.get(lead, [])
        p_amp = _lead_numeric(features, by_lead, lead, "p_amp_mv")
        p_dur = _lead_numeric(features, by_lead, lead, "p_dur_ms")
        if p_dur is None:
            p_dur = _bounds_duration_ms(items, "p", features.fs)
        p_notched = _lead_bool(features, by_lead, lead, "p_notched")
        p_biphasic = _lead_bool(features, by_lead, lead, "p_biphasic")
        p_initial_duration = _lead_numeric(features, by_lead, lead, "p_initial_duration_ms")
        p_initial_amp = _lead_numeric_first(
            features,
            by_lead,
            lead,
            "p_initial_amp_mv",
            "p_initial_amplitude_mV",
        )
        p_terminal_duration = _lead_numeric(features, by_lead, lead, "p_terminal_duration_ms")
        p_terminal_amp = _lead_numeric_first(
            features,
            by_lead,
            lead,
            "p_terminal_amp_mv",
            "p_terminal_amplitude_mV",
        )
        p_terminal_area = _lead_numeric(features, by_lead, lead, "p_terminal_area_mv_ms")
        q = _lead_numeric(features, by_lead, lead, "q_amp_mv")
        r = _lead_numeric(features, by_lead, lead, "r_amp_mv")
        s = _lead_numeric(features, by_lead, lead, "s_amp_mv")
        qrs_signed = _lead_numeric(features, by_lead, lead, "qrs_signed_area")
        qrs_pp = _qrs_peak_to_peak(q, r, s)
        r_duration = _lead_numeric(features, by_lead, lead, "r_duration_ms")
        r_prime_duration = _lead_numeric(features, by_lead, lead, "r_prime_duration_ms")
        s_duration = _lead_numeric(features, by_lead, lead, "s_duration_ms")
        s_prime_duration = _lead_numeric(features, by_lead, lead, "s_prime_duration_ms")
        t_amp = _lead_numeric(features, by_lead, lead, "t_amp_mv")
        t_pol_value = _lead_numeric(features, by_lead, lead, "t_polarity")
        relative_t = None
        if t_amp is not None and qrs_pp is not None and qrs_pp > 0:
            relative_t = t_amp / qrs_pp
        st_j_point = _lead_st_j_point(features, by_lead, lead)
        st_j_source, st_j_reliable, st_j_unreliable_reason = _st_j_status(
            features,
            by_lead,
            lead,
            st_j_point,
        )

        leads[lead] = {
            "p": {
                "duration_ms": p_dur,
                "amplitude_mV": p_amp,
                "area": _lead_numeric(features, by_lead, lead, "p_area"),
                "is_notched": p_notched if p_notched is not None else _unavailable("p_notch_not_measured"),
                "is_biphasic": p_biphasic if p_biphasic is not None else _unavailable("p_biphasic_not_measured"),
                "initial_duration_ms": (
                    p_initial_duration
                    if p_initial_duration is not None
                    else _unavailable("p_initial_component_not_measured")
                ),
                "initial_amplitude_mV": (
                    p_initial_amp
                    if p_initial_amp is not None
                    else _unavailable("p_initial_component_not_measured")
                ),
                "terminal_duration_ms": (
                    p_terminal_duration
                    if p_terminal_duration is not None
                    else _unavailable("p_terminal_component_not_measured")
                ),
                "terminal_amplitude_mV": (
                    p_terminal_amp
                    if p_terminal_amp is not None
                    else _unavailable("p_terminal_component_not_measured")
                ),
                "terminal_area_ashman": (
                    p_terminal_area
                    if p_terminal_area is not None
                    else _unavailable("p_terminal_area_not_measured")
                ),
                "ptf_v1_mv_ms": _lead_numeric(features, by_lead, lead, "ptf_v1_mv_ms"),
                "confidence": _lead_numeric(features, by_lead, lead, "p_confidence_mean")
                or _lead_numeric(features, by_lead, lead, "p_confidence"),
            },
            "qrs": {
                "duration_ms": _lead_numeric(features, by_lead, lead, "qrs_ms"),
                "q_duration_ms": _q_duration_ms(items, features.fs),
                "q_amplitude_mV": q,
                "q_area_mV_ms": _lead_numeric(features, by_lead, lead, "q_area_mv_ms"),
                "q_r_ratio": _lead_numeric(features, by_lead, lead, "q_r_ratio"),
                "q_duration_source": _q_duration_source(items),
                "r_duration_ms": (
                    r_duration
                    if r_duration is not None
                    else _unavailable("r_duration_not_measured")
                ),
                "r_amplitude_mV": r,
                "r_prime_duration_ms": (
                    r_prime_duration
                    if r_prime_duration is not None
                    else _unavailable("r_prime_duration_not_measured")
                ),
                "r_prime_amplitude_mV": _lead_numeric(features, by_lead, lead, "r_prime_amp_mv"),
                "s_duration_ms": (
                    s_duration
                    if s_duration is not None
                    else _unavailable("s_duration_not_measured")
                ),
                "s_amplitude_mV": s,
                "s_prime_duration_ms": (
                    s_prime_duration
                    if s_prime_duration is not None
                    else _unavailable("s_prime_duration_not_measured")
                ),
                "s_prime_amplitude_mV": _lead_numeric(features, by_lead, lead, "s_prime_amp_mv"),
                "peak_to_peak_mV": qrs_pp,
                "positive_component_mV": _positive_component(q, r, s),
                "negative_component_mV": _negative_component(q, r, s),
                "area": _lead_numeric(features, by_lead, lead, "qrs_area"),
                "area_sign": _polarity(qrs_signed if qrs_signed is not None else _qrs_net_value(items[0]) if items else None),
                "terminal_direction": _terminal_direction(lead, r, s),
                "notch_count": _lead_numeric(features, by_lead, lead, "qrs_notch_count"),
                "slur_present": bool(_lead_numeric(features, by_lead, lead, "qrs_slur_flag") or False),
                "vat_ms": _lead_numeric(features, by_lead, lead, "vat_ms"),
                "initial_qrs_area_mV_ms": _lead_numeric(features, by_lead, lead, "initial_qrs_area_mv_ms"),
                "initial_qrs_net_mV": _lead_numeric(features, by_lead, lead, "initial_qrs_net_mv"),
                "fqrs_score": _lead_numeric(features, by_lead, lead, "fqrs_score"),
            },
            "st": {
                "j_point_mV": st_j_point,
                "j_point_source": st_j_source,
                "j_point_reliable": st_j_reliable,
                "unreliable_reason": st_j_unreliable_reason,
                "midpoint_mV": _lead_numeric(features, by_lead, lead, "st_mid_mv"),
                "j80_mV": _lead_numeric(features, by_lead, lead, "st_80ms_mv"),
                "end_mV": _unavailable("st_end_not_measured"),
                "slope_mv_per_ms": _lead_numeric(features, by_lead, lead, "st_slope_mv_per_ms"),
                "slope_deg": _unavailable("st_slope_degrees_not_measured"),
                "shape": _lead_text(features, by_lead, lead, "st_morphology") or "unknown",
                "hybrid_robust": {
                    "profile": "enhanced_robust_v2",
                    "j_point_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_j_mv"
                    ),
                    "j20_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_20ms_mv"
                    ),
                    "j40_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_40ms_mv"
                    ),
                    "j60_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_60ms_mv"
                    ),
                    "j80_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_80ms_mv"
                    ),
                    "adaptive_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_adaptive_mv"
                    ),
                    "mean_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_mean_mv"
                    ),
                    "area_mV_ms": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_area_mv_ms"
                    ),
                    "slope_mV_per_ms": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_slope_mv_per_ms"
                    ),
                    "curvature_mV_per_ms2": _lead_numeric(
                        features,
                        by_lead,
                        lead,
                        "st_hybrid_curvature_mv_per_ms2",
                    ),
                    "trend": _lead_text(
                        features, by_lead, lead, "st_hybrid_trend"
                    ) or "unknown",
                    "shape": _lead_text(
                        features, by_lead, lead, "st_hybrid_shape"
                    ) or "unknown",
                    # Named clinical class: j_point_elevation | slow_upsloping
                    # | horizontal | downsloping | unknown
                    #
                    # Shape only -- it says nothing about how far the segment
                    # is displaced, so read it together with the ST amplitudes
                    # beside it and never on its own. Measured 2026-08-09 over
                    # PTB-XL (135 records, 1620 lead-beats): `horizontal` is
                    # assigned to 49% of leads and its median ST80 is
                    # +0.001 mV -- it means "flat on the baseline" far more
                    # often than the ischemia-suspicious "horizontal
                    # depression" the name suggests. As an ischemia indicator
                    # the class is anti-correlated (AUC 0.38-0.47 against
                    # normal records) while the depressed-lead fraction alone
                    # scores 0.86, and combining the two is worse than
                    # amplitude by itself. `downsloping` (median -0.066 mV)
                    # and `j_point_elevation` (+0.143 mV) do point the right
                    # way but fire on ~3% and ~0.4% of leads.
                    #
                    # The rule engine is unaffected: `clinical_rules/ischemia`
                    # gates on `st_morphology` (the slope trend), not on this,
                    # and that gate was checked -- it excludes 9% of
                    # amplitude-qualifying depressed leads, all genuinely
                    # upsloping, which is the correct clinical call.
                    "pattern_class": _lead_text(
                        features, by_lead, lead, "st_pattern_class"
                    ) or "unknown",
                    "baseline_mV": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_baseline_mv"
                    ),
                    "baseline_source": _lead_text(
                        features, by_lead, lead, "st_hybrid_baseline_source"
                    ),
                    "baseline_confidence": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_baseline_confidence"
                    ),
                    "j_method": _lead_text(
                        features, by_lead, lead, "st_hybrid_j_method"
                    ),
                    "confidence": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_j_confidence"
                    ),
                    "consensus_support": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_consensus_support"
                    ),
                    "beat_support": _lead_numeric(
                        features, by_lead, lead, "st_hybrid_beat_support"
                    ),
                    "source": _lead_text(
                        features, by_lead, lead, "st_hybrid_source"
                    ),
                    "reliable": _lead_bool(
                        features, by_lead, lead, "st_hybrid_reliable"
                    ),
                    "unreliable_reason": _lead_text(
                        features, by_lead, lead, "st_hybrid_unreliable_reason"
                    ),
                },
            },
            "t": {
                "duration_ms": _lead_numeric(features, by_lead, lead, "t_dur_ms")
                or _bounds_duration_ms(items, "t", features.fs),
                "amplitude_mV": t_amp,
                "area": _lead_numeric(features, by_lead, lead, "t_area"),
                "polarity": _t_polarity(t_amp, t_pol_value),
                "relative_to_qrs": relative_t,
                "tpe_ms": _lead_numeric(features, by_lead, lead, "tpe_ms"),
                "u_wave_present": bool(_lead_numeric(features, by_lead, lead, "u_wave_flag") or False),
                "u_wave": {
                    # Signed relative to baseline: a negative value is an
                    # inverted U wave, which the legacy magnitude below cannot
                    # express.
                    "amplitude_mV": _lead_numeric(
                        features, by_lead, lead, "u_amp_signed_mv"
                    ),
                    "amplitude_magnitude_mV": _lead_numeric(
                        features, by_lead, lead, "u_amp_mv"
                    ),
                    "polarity": _lead_numeric(features, by_lead, lead, "u_polarity"),
                    "duration_ms": _lead_numeric(features, by_lead, lead, "u_dur_ms"),
                    "isoelectric_gap_ms": _lead_numeric(
                        features, by_lead, lead, "u_isoelectric_gap_ms"
                    ),
                    "prominence_mV": _lead_numeric(
                        features, by_lead, lead, "u_prominence_mv"
                    ),
                    "measurement_reliable": bool(
                        _lead_bool(features, by_lead, lead, "u_measurement_reliable")
                    ),
                    "beats_measured": _lead_numeric(
                        features, by_lead, lead, "u_beats_measured"
                    ),
                    "polarity_agreement": _lead_numeric(
                        features, by_lead, lead, "u_polarity_agreement"
                    ),
                },
                "refinement": {
                    "wavelet_onset_index": _lead_numeric(
                        features, by_lead, lead, "t_wavelet_onset_index"
                    ),
                    "wavelet_offset_index": _lead_numeric(
                        features, by_lead, lead, "t_wavelet_offset_index"
                    ),
                    "trapezium_offset_index": _lead_numeric(
                        features, by_lead, lead, "t_trapezium_offset_index"
                    ),
                    "candidate_spread_ms": _lead_numeric(
                        features, by_lead, lead, "t_candidate_spread_ms"
                    ),
                    "local_snr_db": _lead_numeric(
                        features, by_lead, lead, "t_local_snr_db"
                    ),
                    "boundary_stability_ms": _lead_numeric(
                        features, by_lead, lead, "t_boundary_stability_ms"
                    ),
                    "sqi_score": _lead_numeric(
                        features, by_lead, lead, "t_sqi_score"
                    ),
                    "sqi_pass": _lead_bool(
                        features, by_lead, lead, "t_sqi_pass"
                    ),
                    "fusion_weight": _lead_numeric(
                        features, by_lead, lead, "t_fusion_weight"
                    ),
                    "fusion_support": _lead_numeric(
                        features, by_lead, lead, "t_offset_fusion_support"
                    ),
                    "fusion_mad_ms": _lead_numeric(
                        features, by_lead, lead, "t_offset_fusion_mad_ms"
                    ),
                    "fusion_ci_half_width_ms": _lead_numeric(
                        features,
                        by_lead,
                        lead,
                        "t_offset_fusion_ci_half_width_ms",
                    ),
                    "fusion_reliable": _lead_bool(
                        features, by_lead, lead, "t_offset_fusion_reliable"
                    ),
                    "statistical_fusion_reliable": _lead_bool(
                        features,
                        by_lead,
                        lead,
                        "t_offset_statistical_fusion_reliable",
                    ),
                    "fusion_reliability_reason": _lead_text(
                        features,
                        by_lead,
                        lead,
                        "t_offset_fusion_reliability_reason",
                    ),
                    "fusion_used_leads": _lead_text(
                        features, by_lead, lead, "t_offset_fusion_used_leads"
                    ),
                    "cluster_count": _lead_numeric(
                        features, by_lead, lead, "t_offset_cluster_count"
                    ),
                    "selected_cluster_score": _lead_numeric(
                        features,
                        by_lead,
                        lead,
                        "t_offset_selected_cluster_score",
                    ),
                    "selected_cluster_support": _lead_numeric(
                        features,
                        by_lead,
                        lead,
                        "t_offset_selected_cluster_support",
                    ),
                    "selected_cluster_lead_groups": _lead_text(
                        features,
                        by_lead,
                        lead,
                        "t_offset_selected_cluster_lead_groups",
                    ),
                    "selected_cluster_methods": _lead_text(
                        features,
                        by_lead,
                        lead,
                        "t_offset_selected_cluster_methods",
                    ),
                    "global_tpte_ms": _lead_numeric(
                        features, by_lead, lead, "t_global_tpte_ms"
                    ),
                    "derived_disagreement_ms": _lead_numeric(
                        features,
                        by_lead,
                        lead,
                        "t_offset_derived_disagreement_ms",
                    ),
                    "tail_incomplete_leads": _lead_text(
                        features,
                        by_lead,
                        lead,
                        "t_offset_tail_incomplete_leads",
                    ),
                    "systematic_early_risk": _lead_bool(
                        features,
                        by_lead,
                        lead,
                        "t_offset_systematic_early_risk",
                    ),
                    "morphology_guard_pass": _lead_bool(
                        features,
                        by_lead,
                        lead,
                        "t_offset_morphology_guard_pass",
                    ),
                    "rms_offset_index": _lead_numeric(
                        features, by_lead, lead, "t_rms_offset_index"
                    ),
                    "pc1_offset_index": _lead_numeric(
                        features, by_lead, lead, "t_pc1_offset_index"
                    ),
                    "derived_spread_ms": _lead_numeric(
                        features, by_lead, lead, "t_derived_spread_ms"
                    ),
                },
            },
            "measurement_profiles": _lead_measurement_profiles(
                features,
                by_lead,
                lead,
                st_j_point,
            ),
            "quality": to_dict(features.quality.get(lead)),
        }

    return {
        "record": {
            "fs": int(features.fs),
            "duration_sec": features.metadata.get("duration_sec"),
            "age_years": _patient_value(features, "age"),
            "age_valid": age_valid,
            "sex": sex,
            "algorithm_age_group": algorithm_age_group,
            "algorithm_age_reason": algorithm_age_reason,
            "age_defaulted_to_adult_algorithm": algorithm_age_reason == "age_missing_or_invalid_default_to_adult",
            "clinical_dx_codes": to_dict(clinical_dx_codes),
            "rx_codes": to_dict(rx_codes),
            "lead_order": to_dict(lead_order),
            "available_extended_leads": available_extended_leads,
            "record_quality": to_dict(record_quality),
            "rejected_functions": to_dict(record_quality.get("rejected_functions", [])) if isinstance(record_quality, dict) else [],
        },
        "global": {
            "heart_rate_bpm": _finite(gf.heart_rate_bpm),
            "qrs_duration_ms": _finite(gf.qrs_ms),
            "qt_ms": _finite(gf.qt_ms),
            "qtc_bazett_ms": _finite(gf.qtc_bazett_ms),
            "qtc_fridericia_ms": _finite(gf.qtc_fridericia_ms),
            "qt_source": getattr(gf, "qt_source", None),
            "qt_used_leads": list(getattr(gf, "qt_used_leads", []) or []),
            "qt_reliability": getattr(gf, "qt_reliability", "unavailable"),
            "qt_reportable": bool(getattr(gf, "qt_reportable", False)),
            "qt_unreliable_reasons": list(
                getattr(gf, "qt_unreliable_reasons", []) or []
            ),
            "qt_rejected": bool(getattr(gf, "qt_rejected", False)),
            "qt_reject_reason": getattr(gf, "qt_reject_reason", None),
            "qrs_wide_ms": _finite(getattr(gf, "qrs_wide_ms", None)),
            "qt_robust_center_ms": _finite(
                getattr(gf, "qt_robust_center_ms", None)
            ),
            "qt_latest_p85_ms": _finite(
                getattr(gf, "qt_latest_p85_ms", None)
            ),
            "t_fusion_support": int(
                getattr(gf, "t_fusion_support", 0) or 0
            ),
            "t_fusion_mad_ms": _finite(
                getattr(gf, "t_fusion_mad_ms", None)
            ),
            "t_fusion_ci_half_width_ms": _finite(
                getattr(gf, "t_fusion_ci_half_width_ms", None)
            ),
            "t_fusion_reliable": bool(
                getattr(gf, "t_fusion_reliable", False)
            ),
            "t_fusion_cluster_count": int(
                getattr(gf, "t_fusion_cluster_count", 0) or 0
            ),
            "t_fusion_selected_cluster_score": _finite(
                getattr(gf, "t_fusion_selected_cluster_score", None)
            ),
            "t_fusion_selected_cluster_support": int(
                getattr(gf, "t_fusion_selected_cluster_support", 0) or 0
            ),
            "t_fusion_lead_groups": list(
                getattr(gf, "t_fusion_lead_groups", []) or []
            ),
            "t_fusion_methods": list(
                getattr(gf, "t_fusion_methods", []) or []
            ),
            "t_fusion_reliability_reasons": list(
                getattr(gf, "t_fusion_reliability_reasons", []) or []
            ),
            "t_global_tpte_ms": _finite(
                getattr(gf, "t_global_tpte_ms", None)
            ),
            "t_derived_disagreement_ms": _finite(
                getattr(gf, "t_derived_disagreement_ms", None)
            ),
            "t_tail_incomplete_leads": list(
                getattr(gf, "t_tail_incomplete_leads", []) or []
            ),
            "t_tail_incomplete_fraction": _finite(
                getattr(gf, "t_tail_incomplete_fraction", 0.0)
            ),
            "t_systematic_early_risk": bool(
                getattr(gf, "t_systematic_early_risk", False)
            ),
            "t_systematic_early_fraction": _finite(
                getattr(gf, "t_systematic_early_fraction", 0.0)
            ),
            "t_morphology_guard_pass": bool(
                getattr(gf, "t_morphology_guard_pass", False)
            ),
            "p_axis_frontal_deg": _finite(gf.p_axis_deg),
            "qrs_axis_frontal_deg": _finite(gf.qrs_axis_deg),
            "qrs_axis_horizontal_direction": _qrs_horizontal_direction(features, by_lead),
            "t_axis_frontal_deg": _finite(gf.t_axis_deg),
            "st_axis_frontal_deg": _finite(gf.st_axis_deg),
            "qrs_t_angle_deg": _finite(qrs_t_angle),
        },
        "measurement_profiles": {
            "active": "hybrid",
            "available": {
                "native": {
                    "available": True,
                    "global_measurements": native_global_profile,
                },
                "12sl": (
                    {
                        **to_dict(twelve_sl_profile),
                        "global_measurements": twelve_sl_global_profile,
                    }
                    if twelve_sl_available
                    else {
                        **_unavailable("12sl_profile_not_available"),
                        "global_measurements": twelve_sl_global_profile,
                    }
                ),
                "hybrid": {
                    "available": True,
                    "st_policy": "prefer_12sl_rr_scaled_st_when_available",
                    "qrs_policy": "prefer_12sl_area_balance_deflection_when_available",
                    "t_policy": "prefer_12sl_special_t_when_available",
                    "global_policy": "prefer_12sl_hr_and_fiducial_intervals_when_available_keep_native_axes_and_qt_dispersion",
                    "global_measurements": hybrid_global_profile,
                },
            },
        },
        "leads": leads,
        "derived_facts": {
            "adult": _adult_morphology_context(features, algorithm_age_group, age_years),
            "pediatric": _pediatric_morphology_context(features, algorithm_age_group, age_years, sex),
            "dextrocardia_criteria": {
                "suspected": bool(_interp_value(features, "dextrocardia_suspected", False)),
                "horizontal_qrs_direction": _qrs_horizontal_direction(features, by_lead),
                "small_v5_v6_threshold_mV": _unavailable("threshold_configuration_required"),
            },
            "rae_criteria": {
                "p_morphology_class": _interp_value(features, "p_morphology_class"),
                "rae_leads": to_dict(_interp_value(features, "rae_leads", [])),
            },
            "lae_criteria": {
                "suspected": bool(_interp_value(features, "lae_suspected", False)),
                "definite": bool(_interp_value(features, "lae_definite", False)),
                "ptf_v1_class": _interp_value(features, "ptf_v1_class"),
            },
            "bae_criteria": {
                "p_morphology_class": _interp_value(features, "p_morphology_class"),
            },
            "axis_flags": {
                "qrs_axis_class": _interp_value(features, "qrs_axis_class"),
                "p_axis_normal": _interp_value(features, "p_axis_normal"),
                "t_axis_class": _interp_value(features, "t_axis_class"),
            },
            "vcd_bbb_criteria": {
                "qrs_width_class": _interp_value(features, "qrs_width_class"),
                "bundle_branch_block": _interp_value(features, "bundle_branch_block"),
            },
            "rvh_criteria": {
                "suspected": bool(_interp_value(features, "rvh_suspected", False)),
                "class": _interp_value(features, "rvh_class"),
            },
            "lvh_voltage_criteria": to_dict(_interp_value(features, "lvh_voltage_criteria", [])),
            "lvh_additional_flags": {
                "class": _interp_value(features, "lvh_class"),
            },
            "pediatric_hypertrophy_evidence": to_dict(
                _interp_value(features, "pediatric_hypertrophy_evidence", {})
            ),
            "low_voltage_criteria": {
                "class": _interp_value(features, "low_voltage_class"),
            },
            "copd_pattern_criteria": {
                "suspected": bool(_interp_value(features, "copd_pattern", False)),
            },
            "mi_territory_criteria": {
                "pathological_q_leads": to_dict(_interp_value(features, "pathological_q_leads", {})),
                "q_wave_territories": to_dict(_interp_value(features, "q_wave_territories", [])),
                "posterior_mi_suspected": bool(_interp_value(features, "posterior_mi_suspected", False)),
                "q_wave_evidence": to_dict(mi_evidence),
                "territory_evidence": to_dict(mi_territories),
                "statement_candidates": to_dict(_interp_value(features, "mi_statement_candidates", [])),
            },
            "culprit_artery_criteria": to_dict(culprit_artery_evidence),
            "st_depression_criteria": {
                "leads": to_dict(st_depression_leads),
                "territories": to_dict(_interp_value(features, "st_territories_depressed", [])),
                "rate_related": bool(_interp_value(features, "st_rate_related", False)),
            },
            "t_wave_criteria": {
                "t_axis_class": _interp_value(features, "t_axis_class"),
                "tall_t_leads": to_dict(_interp_value(features, "tall_t_leads", [])),
            },
            "st_elevation_criteria": {
                "leads": to_dict(st_elevation_leads),
                "territories": to_dict(_interp_value(features, "st_territories_elevated", [])),
                "stemi_suspected_codes": to_dict(_interp_value(features, "stemi_suspected_codes", [])),
                "reciprocal_change_detected": bool(_interp_value(features, "reciprocal_change_detected", False)),
                "reciprocal_pairs": to_dict(_interp_value(features, "reciprocal_pairs", [])),
            },
            "tall_t_criteria": {
                "leads": to_dict(_interp_value(features, "tall_t_leads", [])),
            },
            "qt_electrolyte_criteria": {
                "qtc_class": _interp_value(features, "qtc_class"),
                "electrolyte_hint": _interp_value(features, "qtc_electrolyte_hint"),
            },
        },
        "statement_evidence": build_morphology_statement_evidence(features),
    }


def build_rhythm_inputs(features: ECGFeatures) -> Dict[str, Any]:
    """Build the feature schema consumed by the Philips-style rhythm rule layer."""
    by_beat = _features_by_beat(features)
    rr_ms = _rr_values(features)
    clean_rr = _median(rr_ms)
    record_quality = features.metadata.get("record_quality", {})
    gf = features.global_features
    n_beats = len(features.beats)
    paced_ids = [int(v) for v in features.metadata.get("paced_beat_ids", []) or []]
    spike_times = gf.pacing_spikes or []
    rhythm_analysis = features.metadata.get("rhythm_analysis") or {}
    atrial_events = rhythm_analysis.get("atrial_events") if isinstance(rhythm_analysis, dict) else None
    atrial_residual = (
        rhythm_analysis.get("atrial_residual")
        if isinstance(rhythm_analysis, dict)
        else None
    ) or {"available": False, "reason": "not_computed"}
    af_afl_summary = (
        rhythm_analysis.get("af_afl_summary")
        if isinstance(rhythm_analysis, dict)
        else None
    ) or {}
    pacing_context = (
        rhythm_analysis.get("pacing_context")
        if isinstance(rhythm_analysis, dict)
        else None
    ) or {}
    pacing_failures = (
        rhythm_analysis.get("pacing_failures")
        if isinstance(rhythm_analysis, dict)
        else None
    ) or {}
    rule_summary = (
        rhythm_analysis.get("rule_summary")
        if isinstance(rhythm_analysis, dict)
        else None
    ) or {}
    availability = (
        rhythm_analysis.get("availability")
        if isinstance(rhythm_analysis, dict)
        else None
    )
    if not isinstance(availability, dict):
        availability = {
            "atrial_rhythm_available": True,
            "pr_available": True,
            "p_axis_available": True,
            "reasons": [],
        }
    preexcitation_summary = (
        rule_summary.get("preexcitation")
        if isinstance(rule_summary, dict)
        else None
    ) or {}
    statement_evidence = (
        rule_summary.get("statement_evidence")
        if isinstance(rule_summary, dict)
        else None
    ) or {
        "available": False,
        "reason": "rule_engine_not_available",
        "primary_statement": None,
        "additional_statements": [],
        "statements": [],
        "stop_further_interpretation": False,
        "bypass_remaining_algorithm": False,
    }
    pauses_summary = (
        rule_summary.get("pauses")
        if isinstance(rule_summary, dict)
        else None
    ) or {}
    post_pause_or_interpolated_beats = (
        rule_summary.get("post_pause_or_interpolated_beats")
        if isinstance(rule_summary, dict)
        else None
    ) or []
    qrst_subtraction_validated = bool(atrial_residual.get("validated_qrst_subtraction", False))
    qrst_subtraction_reason = None
    if not qrst_subtraction_validated:
        qrst_subtraction_reason = (
            atrial_residual.get("reason") or "qrst_template_subtraction_not_implemented"
            if bool(atrial_residual.get("available", False))
            else atrial_residual.get("reason", "qrst_template_subtraction_not_implemented")
        )

    rhythm_beats: List[Dict[str, Any]] = []
    p_events: List[Dict[str, Any]] = (
        [_normalize_atrial_p_event(event) for event in atrial_events]
        if atrial_events is not None
        else []
    )
    delta_leads_seen = set()
    delta_beat_ids = set()

    for beat in features.beats:
        items = by_beat.get(int(beat.beat_id), [])
        delta_leads = sorted({bf.lead for bf in items if bool(bf.delta_present)})
        if delta_leads:
            delta_leads_seen.update(delta_leads)
            delta_beat_ids.add(int(beat.beat_id))

        p_conf = max((_finite(bf.p_confidence) or 0.0 for bf in items), default=0.0)
        pr_ms = _median(bf.pr_ms for bf in items)
        qrs_ms = _median(bf.qrs_ms for bf in items)
        signature = {}
        for bf in items:
            net = _qrs_net_value(bf)
            signature[bf.lead] = {
                "net": net,
                "polarity": _polarity(net),
            }

        unreliable = [bf.lead for bf in items if not bool(bf.beat_measurement_reliable)]
        rhythm_beats.append({
            "beat_id": int(beat.beat_id),
            "r_index": int(beat.r_index),
            "r_time_ms": _sample_to_ms(int(beat.r_index), features.fs),
            "rr_prev_ms": _finite(beat.rr_prev_ms),
            "rr_next_ms": _finite(beat.rr_next_ms),
            "group_id": int(beat.group_id),
            "paced": bool(beat.paced),
            "pacer_spike_index": None,
            "qrs_duration_ms": qrs_ms,
            "pr_interval_ms": pr_ms,
            "p_confidence": p_conf,
            "p_axis_deg": _finite(gf.p_axis_deg),
            "p_morphology": _interp_value(features, "p_morphology_class"),
            "qrs_polarity_signature": signature,
            "delta_leads": delta_leads,
            "beat_quality": {
                "reliable": not unreliable,
                "unreliable_leads": unreliable,
            },
        })

        best_p = _best_p_feature(items)
        if atrial_events is None and best_p is not None and (_finite(best_p.p_confidence) or 0.0) >= 0.30:
            p_events.append({
                "p_event_id": len(p_events),
                "time_ms": _sample_to_ms(best_p.p.peak, features.fs),
                "onset_ms": _sample_to_ms(best_p.p.onset, features.fs),
                "offset_ms": _sample_to_ms(best_p.p.offset, features.fs),
                "confidence": _finite(best_p.p_confidence),
                "associated_qrs_beat_id": int(beat.beat_id),
                "association_type": "conducted" if pr_ms is not None else "unknown",
                "pr_ms": pr_ms,
                "axis_deg": _finite(gf.p_axis_deg),
                "morphology": _interp_value(features, "p_morphology_class"),
                "source": "qrs_centered_p_detector",
            })

    _attach_assessment_boundaries(
        p_events, features.p_wave_assessments, int(features.fs)
    )

    excluded = [{"beat_id": beat_id, "reason": "paced"} for beat_id in paced_ids]
    dominant_group_id = features.metadata.get("representative_group_id")

    pacing_state = features.metadata.get("pacing_state") or "off"
    measurement_pacing_state = features.metadata.get("measurement_pacing_state") or pacing_state
    pacing_detection_state = features.metadata.get("pacing_detection_state") or pacing_state
    pacing_enabled = pacing_state != "off" or bool(spike_times) or bool(paced_ids)
    continuous_pacing = bool(n_beats and len(paced_ids) == n_beats)
    intermittent_pacing = bool(0 < len(paced_ids) < n_beats)

    return {
        "record": {
            "fs": int(features.fs),
            "duration_sec": features.metadata.get("duration_sec"),
            "age_years": _patient_value(features, "age"),
            "sex": _patient_value(features, "sex"),
            "lead_order": to_dict(features.metadata.get("lead_order")),
            "record_quality": to_dict(record_quality),
            "rejected_functions": to_dict(record_quality.get("rejected_functions", [])) if isinstance(record_quality, dict) else [],
            "availability": to_dict(availability),
        },
        "beats": rhythm_beats,
        "native_beat_profiles": _build_native_beat_profiles(features, by_beat),
        "p_events": p_events,
        "background": {
            "clean_rr_ms": clean_rr,
            "background_rr_regular": _interp_value(features, "rr_irregularity_class") == "regular",
            "background_ventricular_rate_bpm": _finite(gf.heart_rate_bpm),
            "background_atrial_rate_bpm": _finite(gf.atrial_rate_bpm),
            "dominant_group_id": dominant_group_id,
            "excluded_beat_ids": paced_ids,
            "exclusion_reasons": excluded,
        },
        "pacing": {
            "enabled": bool(pacing_context.get("enabled", pacing_enabled)),
            "state": pacing_state,
            "measurement_state": measurement_pacing_state,
            "detection_state": pacing_detection_state,
            "spike_times": to_dict(spike_times),
            "spike_count": pacing_context.get("spike_count", len(spike_times)),
            "paced_beat_ids": paced_ids,
            "continuous_pacing": bool(pacing_context.get("continuous_pacing", continuous_pacing)),
            "intermittent_pacing": bool(pacing_context.get("intermittent_pacing", intermittent_pacing)),
            "ventricular_pacing_present": pacing_context.get("ventricular_pacing_present", _unavailable()),
            "atrial_pacing_present": pacing_context.get("atrial_pacing_present", _unavailable()),
            "dual_chamber_pacing_present": pacing_context.get("dual_chamber_pacing_present", _unavailable()),
            "confidence_state": pacing_context.get("confidence_state", "indeterminate"),
            "evidence_conflicted": bool(
                pacing_context.get("evidence_conflicted", False)
            ),
            "conflict_reasons": to_dict(
                pacing_context.get("conflict_reasons", [])
            ),
            "supports_measurement_routing": bool(
                pacing_context.get("supports_measurement_routing", False)
            ),
            "qrs_count": pacing_context.get("qrs_count", n_beats),
            "qrs_associated_spike_count": pacing_context.get(
                "qrs_associated_spike_count"
            ),
            "spikes_per_qrs": pacing_context.get("spikes_per_qrs"),
            "qrs_associated_spike_fraction": pacing_context.get(
                "qrs_associated_spike_fraction"
            ),
            "associated_beat_fraction": pacing_context.get(
                "associated_beat_fraction"
            ),
            "capture_alignment_fraction": pacing_context.get(
                "capture_alignment_fraction",
                pacing_failures.get("capture_alignment_fraction"),
            ),
            "median_abs_spike_qrs_offset_ms": pacing_context.get(
                "median_abs_spike_qrs_offset_ms"
            ),
            "spike_qrs_offset_mad_ms": pacing_context.get(
                "spike_qrs_offset_mad_ms"
            ),
            # Backward-compatible field.  Historically this was named
            # ``artifact_confidence`` but actually held capture alignment.
            "artifact_confidence": pacing_failures.get("artifact_confidence", _unavailable()),
            "capture_failure_suspected": pacing_failures.get("capture_failure_suspected", _unavailable()),
            "sensing_failure_suspected": pacing_failures.get("sensing_failure_suspected", _unavailable()),
        },
        "af_afl": {
            "rr_cv": af_afl_summary.get("rr_cv", _interp_value(features, "rr_cv")),
            "rr_rmssd": _rr_rmssd(rr_ms),
            "rr_entropy": _rr_entropy(rr_ms),
            "probable_af": af_afl_summary.get(
                "probable_af",
                _interp_value(features, "probable_af", False),
            ),
            "probable_flutter": bool(af_afl_summary.get("probable_flutter", False)),
            "af_afl_indeterminate": bool(
                af_afl_summary.get("af_afl_indeterminate", False)
            ),
            "atrial_rhythm_classification": af_afl_summary.get(
                "atrial_rhythm_classification", "none"
            ),
            "diagnostic_confidence": af_afl_summary.get("diagnostic_confidence"),
            "indeterminate_reasons": to_dict(
                af_afl_summary.get("indeterminate_reasons", [])
            ),
            "f_wave_confidence": af_afl_summary.get("f_wave_confidence"),
            "f_wave_multilead_consensus": af_afl_summary.get(
                "f_wave_multilead_consensus"
            ),
            "F_wave_confidence": af_afl_summary.get("F_wave_confidence"),
            "F_wave_multilead_consensus": af_afl_summary.get(
                "F_wave_multilead_consensus"
            ),
            "qrst_subtraction": {
                "available": qrst_subtraction_validated,
                "method": atrial_residual.get("method"),
                "scaffold_available": bool(atrial_residual.get("available", False)),
                "reason": qrst_subtraction_reason,
                "validation_metrics": to_dict(atrial_residual.get("validation_metrics", _unavailable("not_computed"))),
            },
            "qrst_subtraction_quality": atrial_residual,
            "atrial_residual_signal_summary": atrial_residual,
            "atrial_signal_stability": atrial_residual.get("stability"),
            "atrial_signal_repetitiveness": atrial_residual.get("repetitiveness"),
            "dominant_atrial_cycle_ms": atrial_residual.get("dominant_cycle_ms"),
            "flutter_wave_confidence": af_afl_summary.get("flutter_wave_confidence"),
        },
        "preexcitation": {
            "short_pr_interval": preexcitation_summary.get(
                "short_pr_interval",
                bool(gf.pr_ms is not None and gf.pr_ms < 120.0),
            ),
            "short_pr_segment": preexcitation_summary.get(
                "short_pr_segment",
                _unavailable("pr_segment_duration_not_measured"),
            ),
            "delta_lead_count": preexcitation_summary.get("delta_lead_count", len(delta_leads_seen)),
            "delta_leads": to_dict(preexcitation_summary.get("delta_leads", sorted(delta_leads_seen))),
            "delta_beat_ids": to_dict(preexcitation_summary.get("delta_beat_ids", sorted(delta_beat_ids))),
            "delta_confidence_by_lead": to_dict(preexcitation_summary.get(
                "delta_confidence_by_lead",
                _unavailable("delta_confidence_not_measured"),
            )),
            "mean_qrs_duration_ms": preexcitation_summary.get("mean_qrs_duration_ms", _finite(gf.qrs_ms)),
            "initial_qrs_axis_deg": preexcitation_summary.get(
                "initial_qrs_axis_deg",
                _unavailable("initial_qrs_axis_not_measured"),
            ),
            "accessory_pathway_side": preexcitation_summary.get(
                "accessory_pathway_side",
                _unavailable("pathway_localization_not_implemented"),
            ),
            "wpw_pattern": preexcitation_summary.get(
                "wpw_pattern",
                _interp_value(features, "wpw_pattern", False),
            ),
        },
        "av_block": {
            "second_degree_avb": pauses_summary.get("second_degree_avb"),
            "evidence": to_dict(pauses_summary.get("av_block_evidence", _unavailable("not_computed"))),
            "atrial_events_per_rr_max": pauses_summary.get("atrial_events_per_rr_max"),
            "escape_origin": pauses_summary.get("escape_origin"),
        },
        "aberrancy": {
            "post_pause_or_interpolated_beats": to_dict(post_pause_or_interpolated_beats),
            "escape_candidates": to_dict([
                event for event in post_pause_or_interpolated_beats
                if event.get("type") == "escape_candidate"
            ]),
            "interpolated_candidates": to_dict([
                event for event in post_pause_or_interpolated_beats
                if event.get("type") == "interpolated_candidate"
            ]),
        },
        "statement_evidence": to_dict(statement_evidence),
    }


def build_statement_engine_payload(
    features: ECGFeatures,
    *,
    rhythm_inputs: Optional[Dict[str, Any]] = None,
    morphology_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata_payload = features.metadata.get("statement_engine")
    if isinstance(metadata_payload, dict):
        return to_dict(metadata_payload)

    rhythm_analysis = features.metadata.get("rhythm_analysis") or {}
    rule_summary = rhythm_analysis.get("rule_summary", {}) if isinstance(rhythm_analysis, dict) else {}
    rhythm_statement_evidence = (
        rule_summary.get("statement_evidence", {})
        if isinstance(rule_summary, dict)
        else {}
    )
    if rhythm_inputs is None:
        rhythm_inputs = build_rhythm_inputs(features)
    if morphology_inputs is None:
        morphology_inputs = build_morphology_inputs(features)

    candidates: List[Dict[str, Any]] = []
    if isinstance(rhythm_statement_evidence, dict):
        candidates.extend(rhythm_statement_evidence.get("candidates") or rhythm_statement_evidence.get("statements") or [])
    morphology_statement_evidence = morphology_inputs.get("statement_evidence", {})
    if isinstance(morphology_statement_evidence, dict):
        candidates.extend(
            morphology_statement_evidence.get("candidates")
            or morphology_statement_evidence.get("final_statements")
            or []
        )

    interpretation = features.interpretation
    pacing_context = rhythm_analysis.get("pacing_context", {}) if isinstance(rhythm_analysis, dict) else {}
    context = {
        "bundle_branch_block": _interp_value(features, "bundle_branch_block"),
        "dextrocardia_suspected": bool(_interp_value(features, "dextrocardia_suspected", False)),
        "lead_reversal": features.metadata.get("lead_reversal", {}),
        "pacing_context": pacing_context,
        "preexcitation": rule_summary.get("preexcitation", {}) if isinstance(rule_summary, dict) else {},
        "af_afl": rhythm_analysis.get("af_afl_summary", {}) if isinstance(rhythm_analysis, dict) else {},
        "availability": rhythm_analysis.get("availability", {}) if isinstance(rhythm_analysis, dict) else {},
    }
    if interpretation is not None and not context["bundle_branch_block"]:
        context["bundle_branch_block"] = getattr(interpretation, "bundle_branch_block", None)

    resolution = resolve_statement_candidates(candidates, context=context).to_dict()
    resolution["sources"] = {
        "rhythm_statement_evidence_available": bool(
            isinstance(rhythm_statement_evidence, dict)
            and rhythm_statement_evidence.get("available")
        ),
        "morphology_statement_evidence_available": bool(
            isinstance(morphology_statement_evidence, dict)
            and morphology_statement_evidence.get("available")
        ),
    }
    return resolution


def _reference_metadata_payload(clinical: Any) -> Dict[str, Any]:
    clinical_mapping = clinical if isinstance(clinical, dict) else {}
    conflicts = [
        dict(item)
        for item in clinical_mapping.get("conflicts", [])
        if isinstance(item, dict)
    ]

    def reference_entry(authority: str, reference: str) -> Dict[str, Any]:
        relevant = [
            item for item in conflicts if str(item.get("reference")) == reference
        ]
        return {
            "reference_only": True,
            "authority": authority,
            "diagnostic_disposition": (
                "superseded_by_authoritative_unified_rules"
                if relevant
                else "reference_only"
            ),
            "diagnostic_conflicts": relevant,
        }

    return {
        "interpretation": reference_entry("legacy_dxl_inspired", "dxl"),
        "statement_engine": {
            "reference_only": True,
            "authority": "legacy_candidate_resolver",
            "diagnostic_disposition": "reference_only",
            "diagnostic_conflicts": [],
        },
    }


def to_dict(obj: Any) -> Any:
    if isinstance(obj, ECGFeatures):
        payload = asdict(obj)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("glasgow_analysis", None)
        payload["rhythm_inputs"] = build_rhythm_inputs(obj)
        payload["morphology_inputs"] = build_morphology_inputs(obj)
        payload["statement_engine"] = build_statement_engine_payload(
            obj,
            rhythm_inputs=payload["rhythm_inputs"],
            morphology_inputs=payload["morphology_inputs"],
        )
        payload["clinical_interpretation"] = _clinical_export_payload(obj)
        if payload["clinical_interpretation"]:
            payload.setdefault("metadata", {})["clinical_interpretation"] = dict(
                payload["clinical_interpretation"]
            )
        payload["reference_metadata"] = _reference_metadata_payload(
            payload["clinical_interpretation"]
        )
        return payload
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def _round_floats(obj: Any, ndigits: Optional[int]) -> Any:
    if isinstance(obj, float):
        if not isfinite(obj):
            return None
        return obj if ndigits is None else round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def prepare_json_export(
    payload: Dict[str, Any],
    *,
    include_beat_features: bool = False,
    round_ndigits: Optional[int] = 6,
) -> Dict[str, Any]:
    """Shrink a `to_dict()` payload before writing it to disk.

    `beat_features` is the full per-beat/per-lead measurement detail (every
    detected beat, every lead, ~178 fields each); most consumers (reports,
    MedGemma, the clinical rules layer) only ever read `representative_leads`
    or `clinical_interpretation`, so it is dropped by default. Pass
    `include_beat_features=True` to keep it for pipeline debugging/auditing.
    `round_ndigits` truncates float precision (default 6 decimals, far more
    than clinically meaningful for mV/ms quantities); pass `None` to disable.

    This only reshapes the already-built `to_dict()` payload for the on-disk
    artifact — it does not change `to_dict()`'s own return value, so no
    existing in-memory consumer of `to_dict()` is affected.
    """
    trimmed = dict(payload)
    trimmed.pop("glasgow", None)
    metadata = trimmed.get("metadata")
    if isinstance(metadata, dict) and "glasgow_analysis" in metadata:
        metadata = dict(metadata)
        metadata.pop("glasgow_analysis", None)
        trimmed["metadata"] = metadata
    if not include_beat_features:
        trimmed.pop("beat_features", None)
    # This pass is also the JSON-safety boundary: real-world records can
    # produce NaN/Inf in optional measurements. JSON has no representation for
    # those values, so export them as null even when rounding is disabled.
    return _round_floats(trimmed, round_ndigits)


def build_structured_payload(features: ECGFeatures) -> Dict[str, Any]:
    record_quality = features.metadata.get("record_quality", {})
    diagnostic_gate = features.metadata.get("diagnostic_gate", {})
    rhythm_inputs = build_rhythm_inputs(features)
    morphology_inputs = build_morphology_inputs(features)
    statement_engine = build_statement_engine_payload(
        features,
        rhythm_inputs=rhythm_inputs,
        morphology_inputs=morphology_inputs,
    )
    clinical = _clinical_export_payload(features)
    return {
        "record": {
            "fs": int(features.fs),
            "duration_sec": features.metadata.get("duration_sec"),
            "age_years": _patient_value(features, "age"),
            "sex": _patient_value(features, "sex"),
            "lead_order": to_dict(features.metadata.get("lead_order")),
            "record_quality": to_dict(record_quality),
            "diagnostic_gate": to_dict(diagnostic_gate),
        },
        "signal": {
            "fs": features.fs,
            "lead_order": to_dict(features.metadata.get("lead_order")),
            "n_beats": features.metadata.get("n_beats"),
        },
        "quality": {
            "record_grade": record_quality.get("record_grade"),
            "reason_codes": to_dict(record_quality.get("reason_codes", [])),
            "leads": to_dict(features.quality),
            "diagnostic_gate": to_dict(diagnostic_gate),
        },
        "beats": to_dict(features.beats),
        "groups": to_dict(features.groups),
        "global": to_dict(features.global_features),
        "pacing": to_dict(rhythm_inputs.get("pacing", {})),
        "rhythm_inputs": to_dict(rhythm_inputs),
        "morphology_inputs": to_dict(morphology_inputs),
        "statement_engine": to_dict(statement_engine),
        "clinical_interpretation": clinical,
        "reference_metadata": _reference_metadata_payload(clinical),
        "provenance": {
            "input_fs": features.metadata.get("input_fs"),
            "internal_fs": features.metadata.get("internal_fs"),
            "input_contract": to_dict(features.metadata.get("input_contract")),
            "lead_reversal": to_dict(features.metadata.get("lead_reversal")),
            "acquisition_qc": to_dict(
                features.metadata.get("acquisition_qc", {})
            ),
            "p_wave_contract": to_dict(
                features.metadata.get("p_wave_contract", {})
            ),
            "p_wave_assessments": to_dict(features.p_wave_assessments),
            "pacing_state": features.metadata.get("pacing_state"),
            "measurement_pacing_state": features.metadata.get(
                "measurement_pacing_state",
                features.metadata.get("pacing_state"),
            ),
            "pacing_detection_state": features.metadata.get(
                "pacing_detection_state",
                features.metadata.get("pacing_state"),
            ),
            "representative_group_id": features.metadata.get("representative_group_id"),
            "global_p_duration": {
                "source": getattr(features.global_features, "p_duration_source", None),
                "used_leads": list(
                    getattr(features.global_features, "p_duration_used_leads", []) or []
                ),
                "support": int(
                    getattr(features.global_features, "p_duration_support", 0) or 0
                ),
                "spread_ms": getattr(
                    features.global_features,
                    "p_duration_spread_ms",
                    None,
                ),
                "reliability": getattr(
                    features.global_features,
                    "p_duration_reliability",
                    "unavailable",
                ),
            },
            "global_qt": {
                "source": getattr(features.global_features, "qt_source", None),
                "used_leads": list(getattr(features.global_features, "qt_used_leads", []) or []),
                "reliability": getattr(features.global_features, "qt_reliability", "unavailable"),
                "reportable": bool(getattr(features.global_features, "qt_reportable", False)),
                "unreliable_reasons": list(
                    getattr(features.global_features, "qt_unreliable_reasons", []) or []
                ),
                "path": getattr(features.global_features, "qt_path", None),
                "confidence_reason": getattr(features.global_features, "qt_confidence_reason", None),
                "excluded_leads": to_dict(getattr(features.global_features, "qt_excluded_leads", {}) or {}),
                "lead_weights": to_dict(getattr(features.global_features, "qt_lead_weights", {}) or {}),
                "consensus_vs_independent_per_lead": to_dict(
                    getattr(features.global_features, "consensus_vs_independent_per_lead", {}) or {}
                ),
            },
        },
        "schema_version": "ecgfeat_structured_payload.v3",
    }

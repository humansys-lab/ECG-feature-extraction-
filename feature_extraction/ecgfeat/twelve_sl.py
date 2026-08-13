from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .models import LeadBeatFeatures, RepresentativeLeadFeatures


TWELVE_SL_PROFILE_VERSION = "12sl_measurement_profile_v1"

TWELVE_SL_CONSTANTS: Dict[str, float] = {
    "pace_spike_high_uv": 1000.0,
    "pace_spike_low_uv": 250.0,
    "qrs_refractory_ms": 200.0,
    "wave_significance_area_uv_ms": 160.0,
    "stm_rr_fraction": 1.0 / 16.0,
    "ste_rr_fraction": 1.0 / 8.0,
    "special_t_small_uv": 70.0,
    "special_t_multiplier": 4.0,
}


_PROFILE_NUMERIC_KEYS = (
    "twelve_sl_stj_mv",
    "twelve_sl_stm_mv",
    "twelve_sl_ste_mv",
    "twelve_sl_stm_offset_ms",
    "twelve_sl_ste_offset_ms",
    "twelve_sl_qrs_area_uv_ms",
    "twelve_sl_qrs_signed_area_uv_ms",
    "twelve_sl_qrs_balance_uv",
    "twelve_sl_qrs_deflection_uv",
    "twelve_sl_minimum_st_uv",
    "twelve_sl_special_t_uv",
    "twelve_sl_t_prime_uv",
    "twelve_sl_t_prime_area_uv_ms",
    "twelve_sl_st_confidence",
)


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if isfinite(f) else None


def _median(values: Iterable[Any]) -> Optional[float]:
    clean = [_finite_float(value) for value in values]
    vals = [value for value in clean if value is not None]
    return float(median(vals)) if vals else None


def _sample(sig: np.ndarray, idx: Optional[int], reference_mv: float) -> Optional[float]:
    if idx is None or idx < 0 or idx >= len(sig):
        return None
    return float(sig[idx] - reference_mv)


def _st_sample_index(qrs_off: int, offset_ms: float, fs: int) -> int:
    return int(qrs_off + round(offset_ms * float(fs) / 1000.0))


def _trapezoid(y: np.ndarray, dx: float) -> float:
    values = np.asarray(y, dtype=float)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return 0.0
    return float(dx * (0.5 * values[0] + float(np.sum(values[1:-1])) + 0.5 * values[-1]))


def _wave_sign(value: float, eps_mv: float = 0.005) -> int:
    if value > eps_mv:
        return 1
    if value < -eps_mv:
        return -1
    return 0


def _t_components(
    sig: np.ndarray,
    *,
    lo: int,
    hi: int,
    reference_mv: float,
    fs: int,
) -> List[Dict[str, float]]:
    lo_i = max(0, int(lo))
    hi_i = min(len(sig) - 1, int(hi))
    if fs <= 0 or hi_i <= lo_i:
        return []
    rel = sig[lo_i : hi_i + 1].astype(float) - float(reference_mv)
    signs = [_wave_sign(float(value)) for value in rel]
    components: List[Dict[str, float]] = []
    start: Optional[int] = None
    sign = 0

    def close_component(end_local: int) -> None:
        nonlocal start, sign
        if start is None or sign == 0:
            start = None
            sign = 0
            return
        end_global = lo_i + int(end_local)
        start_global = lo_i + int(start)
        if end_global < start_global:
            start = None
            sign = 0
            return
        segment = sig[start_global : end_global + 1].astype(float) - float(reference_mv)
        if segment.size == 0:
            start = None
            sign = 0
            return
        if sign > 0:
            peak_local = int(np.argmax(segment))
        else:
            peak_local = int(np.argmin(segment))
        peak = start_global + peak_local
        amp_mv = float(sig[peak] - reference_mv)
        area_uv_ms = float(_trapezoid(np.abs(segment), dx=1000.0 / float(fs)) * 1000.0)
        components.append(
            {
                "start": float(start_global),
                "end": float(end_global),
                "sign": float(sign),
                "peak": float(peak),
                "amp_mv": amp_mv,
                "area_uv_ms": area_uv_ms,
            }
        )
        start = None
        sign = 0

    for local, current_sign in enumerate(signs):
        if current_sign == 0:
            close_component(local - 1)
            continue
        if start is None:
            start = int(local)
            sign = int(current_sign)
            continue
        if current_sign != sign:
            close_component(local - 1)
            start = int(local)
            sign = int(current_sign)
    close_component(len(signs) - 1)
    return components


def _t_prime_component(
    sig: np.ndarray,
    *,
    t_peak: Optional[int],
    t_off: Optional[int],
    reference_mv: float,
    fs: int,
    explicit_peak: Optional[int],
) -> Optional[Dict[str, float]]:
    if t_peak is None or t_off is None or fs <= 0:
        return None
    peak_i = int(t_peak)
    off_i = int(t_off)
    if peak_i < 0 or off_i <= peak_i or peak_i >= len(sig):
        return None
    components = _t_components(sig, lo=peak_i, hi=off_i, reference_mv=reference_mv, fs=fs)
    if explicit_peak is not None:
        explicit_i = int(explicit_peak)
        for component in components:
            if int(component["start"]) <= explicit_i <= int(component["end"]):
                return component
        return None

    significance = TWELVE_SL_CONSTANTS["wave_significance_area_uv_ms"]
    main_sign = _wave_sign(float(sig[peak_i] - reference_mv))
    later = [
        component
        for component in components
        if int(component["peak"]) > peak_i
        and int(component["start"]) > peak_i
        and component["area_uv_ms"] >= significance
        and (main_sign == 0 or int(component["sign"]) != main_sign)
    ]
    if not later:
        return None
    return max(later, key=lambda item: (item["area_uv_ms"], int(item["peak"])))


def _st_morphology_confidence(
    *,
    stj_mv: Optional[float],
    stm_mv: Optional[float],
    ste_mv: Optional[float],
    qrs_offset_confidence: Optional[float],
    qrs_offset_repaired: bool,
) -> tuple[float, str]:
    confidence = 1.0
    reasons: List[str] = []

    qrs_conf = _finite_float(qrs_offset_confidence)
    if qrs_conf is not None and qrs_conf < 0.45:
        confidence = min(confidence, max(0.0, qrs_conf))
        reasons.append("low_qrs_offset_confidence")
    if qrs_offset_repaired:
        confidence = min(confidence, 0.55)
        reasons.append("qrs_offset_repaired")

    st_values = [
        value
        for value in (_finite_float(stj_mv), _finite_float(stm_mv), _finite_float(ste_mv))
        if value is not None
    ]
    if len(st_values) < 3:
        confidence = min(confidence, 0.0)
        reasons.append("missing_st_samples")
    else:
        spread_mv = float(max(st_values) - min(st_values))
        if spread_mv >= 0.20:
            confidence = min(confidence, 0.45)
            reasons.append("unstable_st_samples")

    return float(confidence), ";".join(reasons) if reasons else "stable_st_samples"


def special_t_amplitude_mv(
    *,
    t_amp_mv: Optional[float],
    t_prime_amp_mv: Optional[float],
    ste_mv: Optional[float],
    t_offset_amp_mv: Optional[float],
) -> Optional[float]:
    """12SL special-T amplitude, expressed in mV."""
    t_amp = _finite_float(t_amp_mv)
    if t_amp is None:
        return None

    ste = _finite_float(ste_mv) or 0.0
    t_prime = _finite_float(t_prime_amp_mv) or 0.0
    t_offset = _finite_float(t_offset_amp_mv)
    small_mv = TWELVE_SL_CONSTANTS["special_t_small_uv"] / 1000.0
    multiplier = TWELVE_SL_CONSTANTS["special_t_multiplier"]

    special = min(t_amp, t_amp - ste)
    if t_prime < 0.0:
        small_negative_ignored = abs(t_prime) < small_mv and t_amp >= multiplier * abs(t_prime)
        if not small_negative_ignored:
            special = t_prime
    elif (
        t_prime > 0.0
        and t_amp > -small_mv
        and t_prime >= multiplier * max(abs(t_amp), 1e-9)
    ):
        special = t_prime

    if t_prime == 0.0 and t_amp < 0.0 and t_offset is not None:
        special = min(special, t_amp - t_offset)
    return float(special)


def twelve_sl_wave_measurements_from_signal(
    *,
    sig: np.ndarray,
    qrs_on: Optional[int],
    qrs_off: Optional[int],
    t_peak: Optional[int],
    t_off: Optional[int],
    avg_rr_ms: Optional[float],
    fs: int,
    t_prime_peak: Optional[int] = None,
    qrs_offset_confidence: Optional[float] = None,
    qrs_offset_repaired: bool = False,
) -> Dict[str, Any]:
    """Measure 12SL-style wave quantities from one local beat signal."""
    rr_ms = _finite_float(avg_rr_ms)
    if (
        qrs_on is None
        or qrs_off is None
        or rr_ms is None
        or fs <= 0
        or qrs_on < 0
        or qrs_off <= qrs_on
        or qrs_off >= len(sig)
    ):
        return {}

    qrs_reference_mv = float(sig[qrs_on])
    qrs_seg = sig[qrs_on : qrs_off + 1].astype(float) - qrs_reference_mv
    dx_ms = 1000.0 / float(fs)
    qrs_area_uv_ms = float(_trapezoid(np.abs(qrs_seg), dx=dx_ms) * 1000.0)
    qrs_signed_area_uv_ms = float(_trapezoid(qrs_seg, dx=dx_ms) * 1000.0)
    max_r_mv = max(0.0, float(np.max(qrs_seg)))
    max_s_mv = max(0.0, -float(np.min(qrs_seg)))

    stm_offset_ms = rr_ms * TWELVE_SL_CONSTANTS["stm_rr_fraction"]
    ste_offset_ms = rr_ms * TWELVE_SL_CONSTANTS["ste_rr_fraction"]
    stj_mv = _sample(sig, qrs_off, qrs_reference_mv)
    stm_mv = _sample(sig, _st_sample_index(qrs_off, stm_offset_ms, fs), qrs_reference_mv)
    ste_mv = _sample(sig, _st_sample_index(qrs_off, ste_offset_ms, fs), qrs_reference_mv)
    st_confidence, st_confidence_reason = _st_morphology_confidence(
        stj_mv=stj_mv,
        stm_mv=stm_mv,
        ste_mv=ste_mv,
        qrs_offset_confidence=qrs_offset_confidence,
        qrs_offset_repaired=qrs_offset_repaired,
    )
    t_amp_mv = _sample(sig, t_peak, qrs_reference_mv)
    t_offset_amp_mv = _sample(sig, t_off, qrs_reference_mv)
    t_prime_component = _t_prime_component(
        sig,
        t_peak=t_peak,
        t_off=t_off,
        reference_mv=qrs_reference_mv,
        fs=fs,
        explicit_peak=t_prime_peak,
    )
    inferred_t_prime_peak = (
        int(t_prime_component["peak"]) if t_prime_component is not None else t_prime_peak
    )
    t_prime_amp_mv = _sample(sig, inferred_t_prime_peak, qrs_reference_mv)
    ste_for_special = ste_mv if st_confidence >= 0.50 else None
    special_t_mv = special_t_amplitude_mv(
        t_amp_mv=t_amp_mv,
        t_prime_amp_mv=t_prime_amp_mv,
        ste_mv=ste_for_special,
        t_offset_amp_mv=t_offset_amp_mv,
    )

    st_values_uv = [
        value * 1000.0
        for value in (stj_mv, stm_mv)
        if value is not None
    ]
    minimum_st_uv = min(st_values_uv) if st_values_uv else None
    return {
        "stj_mv": stj_mv,
        "stm_mv": stm_mv,
        "ste_mv": ste_mv,
        "stm_offset_ms": float(stm_offset_ms),
        "ste_offset_ms": float(ste_offset_ms),
        "st_confidence": st_confidence,
        "st_confidence_reason": st_confidence_reason,
        "qrs_area_uv_ms": qrs_area_uv_ms,
        "qrs_signed_area_uv_ms": qrs_signed_area_uv_ms,
        "qrs_balance_uv": float((max_r_mv - max_s_mv) * 1000.0),
        "qrs_deflection_uv": float((max_r_mv + max_s_mv) * 1000.0),
        "minimum_st_uv": minimum_st_uv,
        "t_prime_peak": inferred_t_prime_peak,
        "t_prime_amp_uv": None if t_prime_amp_mv is None else float(t_prime_amp_mv * 1000.0),
        "t_prime_area_uv_ms": (
            None if t_prime_component is None else float(t_prime_component["area_uv_ms"])
        ),
        "special_t_uv": None if special_t_mv is None else float(special_t_mv * 1000.0),
        "qrs_significant": qrs_area_uv_ms >= TWELVE_SL_CONSTANTS["wave_significance_area_uv_ms"],
    }


def _heart_rate_first_last_bpm(r_locs: np.ndarray, fs: int) -> Optional[float]:
    if len(r_locs) < 2 or fs <= 0:
        return None
    span_samples = int(r_locs[-1]) - int(r_locs[0])
    if span_samples <= 0:
        return None
    span_ms = span_samples * 1000.0 / float(fs)
    return float((len(r_locs) - 1) * 60000.0 / span_ms)


def _sample_offset_ms(
    sample: Optional[int],
    beat_id: int,
    r_locs: np.ndarray,
    fs: int,
) -> Optional[float]:
    if sample is None or beat_id < 0 or beat_id >= len(r_locs) or fs <= 0:
        return None
    return float((int(sample) - int(r_locs[beat_id])) * 1000.0 / float(fs))


def _global_fiducials(
    beat_features: List[LeadBeatFeatures],
    r_locs: np.ndarray,
    fs: int,
) -> Dict[str, Any]:
    by_beat: Dict[int, List[LeadBeatFeatures]] = defaultdict(list)
    for feature in beat_features:
        by_beat[int(feature.beat_id)].append(feature)

    qrs_onsets: List[float] = []
    qrs_offsets: List[float] = []
    p_onsets: List[float] = []
    p_offsets: List[float] = []
    t_offsets: List[float] = []
    for beat_id, items in by_beat.items():
        qrs_on = [
            offset
            for offset in (
                _sample_offset_ms(item.qrs.onset, beat_id, r_locs, fs)
                for item in items
            )
            if offset is not None
        ]
        qrs_off = [
            offset
            for offset in (
                _sample_offset_ms(item.qrs.offset, beat_id, r_locs, fs)
                for item in items
            )
            if offset is not None
        ]
        p_on = [
            offset
            for offset in (
                _sample_offset_ms(item.p.onset, beat_id, r_locs, fs)
                for item in items
            )
            if offset is not None
        ]
        p_off = [
            offset
            for offset in (
                _sample_offset_ms(item.p.offset, beat_id, r_locs, fs)
                for item in items
            )
            if offset is not None
        ]
        t_off = [
            offset
            for offset in (
                _sample_offset_ms(item.t.offset, beat_id, r_locs, fs)
                for item in items
            )
            if offset is not None
        ]
        if qrs_on:
            qrs_onsets.append(min(qrs_on))
        if qrs_off:
            qrs_offsets.append(max(qrs_off))
        if p_on:
            p_onsets.append(min(p_on))
        if p_off:
            p_offsets.append(max(p_off))
        if t_off:
            t_offsets.append(max(t_off))

    return {
        "method": "beat_feature_multilead_summary",
        "support_beats": len(by_beat),
        "qrs_onset_offset_ms": _median(qrs_onsets),
        "qrs_offset_offset_ms": _median(qrs_offsets),
        "p_onset_offset_ms": _median(p_onsets),
        "p_offset_offset_ms": _median(p_offsets),
        "t_offset_offset_ms": _median(t_offsets),
    }


def apply_twelve_sl_measurement_profile(
    *,
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    beat_features: List[LeadBeatFeatures],
    r_locs: np.ndarray,
    fs: int,
) -> Dict[str, Any]:
    """Append 12SL-style representative parameters and return profile metadata."""
    by_lead: Dict[str, List[LeadBeatFeatures]] = defaultdict(list)
    for feature in beat_features:
        if not bool(getattr(feature, "beat_measurement_reliable", True)):
            continue
        by_lead[feature.lead].append(feature)

    for lead, rep in representative_leads.items():
        items = by_lead.get(lead, [])
        for key in _PROFILE_NUMERIC_KEYS:
            value = _median(getattr(item, key, None) for item in items)
            if value is not None:
                rep.params[key] = value
        reason_counts: Dict[str, int] = defaultdict(int)
        for item in items:
            reason = getattr(item, "twelve_sl_st_confidence_reason", None)
            if reason:
                reason_counts[str(reason)] += 1
        if reason_counts:
            rep.params["twelve_sl_st_confidence_reason"] = max(
                reason_counts.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
        area = _finite_float(rep.params.get("twelve_sl_qrs_area_uv_ms"))
        if area is not None:
            rep.params["twelve_sl_qrs_significant"] = (
                area >= TWELVE_SL_CONSTANTS["wave_significance_area_uv_ms"]
            )
        elif items:
            rep.params["twelve_sl_qrs_significant"] = any(
                bool(getattr(item, "twelve_sl_qrs_significant", False))
                for item in items
            )

    return {
        "profile_version": TWELVE_SL_PROFILE_VERSION,
        "source": "parallel_profile_no_diagnostic_override",
        "constants": dict(TWELVE_SL_CONSTANTS),
        "heart_rate_first_last_bpm": _heart_rate_first_last_bpm(r_locs, fs),
        "global_fiducials": _global_fiducials(beat_features, r_locs, fs),
    }

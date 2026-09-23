from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..foundation.numeric import trapezoid

_EARLY_ST_TROUGH_MIN_AREA_RATIO = 0.45
_EARLY_ST_TROUGH_MIN_AMP_RATIO = 1.25
_EARLY_ST_TROUGH_MIN_WIDTH_SEC = 0.014
_EARLY_ST_TROUGH_MAX_TIME_MS = 150.0
_LATE_SAME_POLARITY_T_MIN_TIME_MS = 180.0
_LATE_SAME_POLARITY_T_MIN_GAP_MS = 60.0
_LATE_SAME_POLARITY_T_MIN_AREA_RATIO = 0.45
_LATE_SAME_POLARITY_T_MIN_AMP_RATIO = 0.45
_LATE_SAME_POLARITY_T_MIN_WIDTH_SEC = 0.030
_ST_T_CONFUSION_MIN_SELECTED_AMP_MV = 0.040
_ST_T_CONFUSION_MIN_OPPOSITE_AMP_MV = 0.050
_T_SIGNIFICANT_AREA_MV_MS = 0.160
_T_COMPONENT_EPS_MV = 0.005
# Boundary between the ST segment and the T wave, measured from the QRS offset.
# Shared by every place that decides whether a candidate is "late" enough to be
# a T wave rather than an ST-segment deflection, so those places cannot drift
# apart.  Used only to *exclude* candidates -- see
# docs/qtdb_cross_dataset_validation.md 5c.5 for the two measured attempts at
# using it to *relocate* a peak, both of which cost LUDB T peak precision.
_T_ST_SEGMENT_GUARD_MS = 120.0
_LATE_POLARITY_OVERRIDE_MIN_SELECTED_TIME_MS = 360.0
_LATE_POLARITY_OVERRIDE_MIN_EARLIER_TIME_MS = 150.0
_LATE_POLARITY_OVERRIDE_MAX_EARLIER_TIME_MS = 340.0
_LATE_POLARITY_OVERRIDE_MIN_AREA_RATIO = 0.70
_LATE_POLARITY_OVERRIDE_MIN_AMP_RATIO = 0.50
_T_PRIME_MAX_PEAK_GAP_MS = 180.0
_T_PRIME_MAX_START_GAP_MS = 140.0


@dataclass(frozen=True)
class TWaveCandidate:
    index: int
    amp_mv: float
    signed_area: float
    abs_area: float
    width_samples: int
    time_from_qrs_off_ms: float
    lead: str
    flags: Tuple[str, ...] = ()


@dataclass
class TWaveMeasurement:
    peak: Optional[int]
    offset: Optional[int]
    confidence: float
    method: str
    polarity_expected: Optional[int]
    polarity_observed: Optional[int]
    st_t_confusion: bool = False
    confidence_reason: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    t_prime_peak: Optional[int] = None


@dataclass(frozen=True)
class TWaveComponent:
    start: int
    end: int
    peak: int
    amp_mv: float
    signed_area_mv_ms: float
    abs_area_mv_ms: float
    width_samples: int
    time_from_qrs_off_ms: float
    sign: int


def _sign(value: float, *, min_abs: float = 0.015) -> int:
    if value > min_abs:
        return 1
    if value < -min_abs:
        return -1
    return 0


def _trapz(y: np.ndarray) -> float:
    return float(trapezoid(y))


def _trapz_dx(y: np.ndarray, dx: float) -> float:
    values = np.asarray(y, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(dx * (0.5 * values[0] + float(np.sum(values[1:-1])) + 0.5 * values[-1]))


def _component_sign(value: float) -> int:
    if value > _T_COMPONENT_EPS_MV:
        return 1
    if value < -_T_COMPONENT_EPS_MV:
        return -1
    return 0


def _t_components(
    sig: np.ndarray,
    *,
    lo: int,
    hi: int,
    baseline: float,
    fs: int,
    qrs_off: Optional[int],
) -> List[TWaveComponent]:
    lo_i = max(1, int(lo))
    hi_i = min(len(sig) - 1, int(hi))
    if fs <= 0 or hi_i <= lo_i:
        return []
    rel = sig[lo_i : hi_i + 1].astype(float) - float(baseline)
    signs = [_component_sign(float(value)) for value in rel]
    components: List[TWaveComponent] = []
    start: Optional[int] = None
    sign = 0

    def close_component(end_local: int) -> None:
        nonlocal start, sign
        if start is None or sign == 0:
            start = None
            sign = 0
            return
        start_global = lo_i + int(start)
        end_global = lo_i + int(end_local)
        if end_global < start_global:
            start = None
            sign = 0
            return
        seg = sig[start_global : end_global + 1].astype(float) - float(baseline)
        if seg.size <= 1:
            start = None
            sign = 0
            return
        peak_local = int(np.argmax(seg) if sign > 0 else np.argmin(seg))
        peak = start_global + peak_local
        signed_area = _trapz_dx(seg, dx=1000.0 / float(fs))
        abs_area = _trapz_dx(np.abs(seg), dx=1000.0 / float(fs))
        dt_ms = (peak - int(qrs_off)) * 1000.0 / fs if qrs_off is not None else 0.0
        components.append(
            TWaveComponent(
                start=start_global,
                end=end_global,
                peak=peak,
                amp_mv=float(sig[peak] - baseline),
                signed_area_mv_ms=float(signed_area),
                abs_area_mv_ms=float(abs_area),
                width_samples=end_global - start_global + 1,
                time_from_qrs_off_ms=dt_ms,
                sign=sign,
            )
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


def _component_to_candidate(component: TWaveComponent, lead: str) -> TWaveCandidate:
    flags: List[str] = []
    if component.time_from_qrs_off_ms < 150.0:
        flags.append("early_st_trough_candidate")
    return TWaveCandidate(
        index=int(component.peak),
        amp_mv=float(component.amp_mv),
        signed_area=float(component.signed_area_mv_ms),
        abs_area=float(component.abs_area_mv_ms),
        width_samples=int(component.width_samples),
        time_from_qrs_off_ms=float(component.time_from_qrs_off_ms),
        lead=lead,
        flags=tuple(flags),
    )


def _choose_component_candidate(
    components: List[TWaveComponent],
    *,
    polarity: Optional[int],
    search_hi: int,
    lead: str,
    fs: int,
    prefer_late_component: bool = True,
) -> Tuple[Optional[TWaveCandidate], Optional[TWaveComponent]]:
    significant = [
        item
        for item in components
        if item.peak < int(search_hi)
        and item.abs_area_mv_ms >= _T_SIGNIFICANT_AREA_MV_MS
        and item.width_samples >= max(3, int(0.012 * fs))
    ]
    if not significant:
        return None, None
    if polarity not in (None, 0):
        signed = [item for item in significant if item.sign == int(polarity)]
        pool = signed or significant
    else:
        pool = significant
    late = [
        item for item in pool
        if item.time_from_qrs_off_ms >= _T_ST_SEGMENT_GUARD_MS
    ]
    if prefer_late_component and late:
        pool = late
    selected = max(
        pool,
        key=lambda item: (
            item.abs_area_mv_ms * max(item.width_samples, 1),
            item.abs_area_mv_ms,
            item.peak,
        ),
    )
    return _component_to_candidate(selected, lead), selected


def _later_t_prime_component(
    components: List[TWaveComponent],
    selected: Optional[TWaveComponent],
    *,
    fs: int,
) -> Optional[TWaveComponent]:
    if selected is None:
        return None
    later = [
        item
        for item in components
        if item.start > selected.end
        and item.sign != selected.sign
        and item.abs_area_mv_ms >= _T_SIGNIFICANT_AREA_MV_MS
        and (item.peak - selected.peak) * 1000.0 / fs <= _T_PRIME_MAX_PEAK_GAP_MS
        and (item.start - selected.end) * 1000.0 / fs <= _T_PRIME_MAX_START_GAP_MS
    ]
    if not later:
        return None
    return max(later, key=lambda item: (item.abs_area_mv_ms, item.peak))


def infer_cluster_polarity(
    candidates_by_lead: Dict[str, List[TWaveCandidate]],
    expected: Optional[int] = None,
) -> Optional[int]:
    score = 0.0
    for candidates in candidates_by_lead.values():
        if not candidates:
            continue
        plausible_t_candidates = [
            item for item in candidates
            if item.time_from_qrs_off_ms >= _T_ST_SEGMENT_GUARD_MS
        ]
        best = max(plausible_t_candidates or candidates, key=lambda item: item.abs_area)
        score += float(best.signed_area)
    observed = _sign(score, min_abs=0.001)
    if observed:
        return observed
    return expected


def candidates_from_triplets(
    candidates_by_lead: Dict[str, List[Tuple[int, float, float]]],
    *,
    qrs_off: Optional[int],
    fs: int,
) -> Dict[str, List[TWaveCandidate]]:
    out: Dict[str, List[TWaveCandidate]] = {}
    for lead, candidates in candidates_by_lead.items():
        lead_items: List[TWaveCandidate] = []
        for peak, amp, area in candidates:
            signed_area = float(area) if float(amp) >= 0.0 else -float(area)
            dt_ms = (int(peak) - int(qrs_off)) * 1000.0 / fs if qrs_off is not None else 0.0
            flags: List[str] = []
            if qrs_off is not None and dt_ms < 150.0:
                flags.append("early_st_trough_candidate")
            lead_items.append(
                TWaveCandidate(
                    index=int(peak),
                    amp_mv=float(amp),
                    signed_area=signed_area,
                    abs_area=abs(float(area)),
                    width_samples=1,
                    time_from_qrs_off_ms=dt_ms,
                    lead=lead,
                    flags=tuple(flags),
                )
            )
        out[lead] = lead_items
    return out


def _candidate_indices(
    sig_sm: np.ndarray,
    lo: int,
    hi: int,
    baseline: float,
    fs: int,
    lead: str,
    qrs_off: Optional[int],
) -> List[TWaveCandidate]:
    lo = max(1, int(lo))
    hi = min(len(sig_sm) - 1, int(hi))
    if hi - lo < 4:
        return []
    x = sig_sm[lo:hi].astype(float) - float(baseline)
    abs_x = np.abs(x)
    if abs_x.size == 0:
        return []
    max_amp = float(np.max(abs_x))
    if max_amp <= 0.0:
        return []
    min_amp = max(0.010, 0.25 * max_amp)
    d1 = np.diff(abs_x)
    turns = np.where(np.sign(d1[:-1]) != np.sign(d1[1:]))[0] + 1
    idxs = set(int(idx) for idx in turns if abs_x[int(idx)] >= min_amp)
    idxs.add(int(np.argmax(abs_x)))
    area_half = max(4, int(0.045 * fs))
    width_half = max(4, int(0.070 * fs))
    out: List[TWaveCandidate] = []
    for local_idx in idxs:
        peak = lo + int(local_idx)
        amp = float(sig_sm[peak] - baseline)
        area_lo = max(lo, peak - area_half)
        area_hi = min(hi, peak + area_half + 1)
        seg = sig_sm[area_lo:area_hi] - baseline
        signed_area = _trapz(seg)
        abs_area = _trapz(np.abs(seg))
        width_lo = max(lo, peak - width_half)
        width_hi = min(hi, peak + width_half + 1)
        width = int(np.sum(np.abs(sig_sm[width_lo:width_hi] - baseline) >= max(0.006, 0.40 * abs(amp))))
        dt_ms = (peak - int(qrs_off)) * 1000.0 / fs if qrs_off is not None else 0.0
        flags: List[str] = []
        if qrs_off is not None and dt_ms < 150.0:
            flags.append("early_st_trough_candidate")
        out.append(TWaveCandidate(peak, amp, signed_area, abs_area, width, dt_ms, lead, tuple(flags)))
    out.sort(key=lambda item: (item.abs_area * max(item.width_samples, 1), item.abs_area), reverse=True)
    return out


def _choose_signed_candidate(
    candidates: List[TWaveCandidate],
    polarity: Optional[int],
) -> Optional[TWaveCandidate]:
    if not candidates:
        return None
    if polarity is None or polarity == 0:
        return candidates[0]
    signed = [item for item in candidates if _sign(item.amp_mv) == polarity]
    if signed:
        return max(signed, key=lambda item: (item.abs_area * max(item.width_samples, 1), item.index))
    return candidates[0]


def _earlier_opposite_component_over_late_polarity_candidate(
    candidates: List[TWaveCandidate],
    selected: Optional[TWaveCandidate],
    *,
    fs: int,
) -> Optional[TWaveCandidate]:
    if selected is None:
        return None
    selected_sign = _sign(selected.amp_mv)
    if selected_sign == 0:
        return None
    if selected.time_from_qrs_off_ms < _LATE_POLARITY_OVERRIDE_MIN_SELECTED_TIME_MS:
        return None
    selected_ref = next(
        (item for item in candidates if int(item.index) == int(selected.index)),
        selected,
    )

    min_width = max(6, int(round(0.030 * fs)))
    min_gap = max(4, int(round(0.060 * fs)))
    earlier = [
        item
        for item in candidates
        if item.index < selected.index - min_gap
        and _sign(item.amp_mv) == -selected_sign
        and _LATE_POLARITY_OVERRIDE_MIN_EARLIER_TIME_MS
        <= item.time_from_qrs_off_ms
        <= _LATE_POLARITY_OVERRIDE_MAX_EARLIER_TIME_MS
        and item.width_samples >= min_width
        and item.abs_area >= max(
            _T_SIGNIFICANT_AREA_MV_MS,
            _LATE_POLARITY_OVERRIDE_MIN_AREA_RATIO * selected_ref.abs_area,
        )
        and abs(item.amp_mv) >= max(
            _ST_T_CONFUSION_MIN_SELECTED_AMP_MV,
            _LATE_POLARITY_OVERRIDE_MIN_AMP_RATIO * abs(selected_ref.amp_mv),
        )
    ]
    if not earlier:
        return None
    return max(earlier, key=lambda item: (item.abs_area * max(item.width_samples, 1), item.abs_area))


def _later_same_polarity_t_after_st_trough(
    candidates: List[TWaveCandidate],
    selected: TWaveCandidate,
    *,
    fs: int,
) -> Optional[TWaveCandidate]:
    selected_sign = _sign(selected.amp_mv)
    if selected_sign == 0:
        return None
    if selected.time_from_qrs_off_ms >= _EARLY_ST_TROUGH_MAX_TIME_MS:
        return None
    if "early_st_trough_candidate" not in selected.flags:
        return None

    min_gap = int(round(_LATE_SAME_POLARITY_T_MIN_GAP_MS * fs / 1000.0))
    min_width = max(6, int(round(_LATE_SAME_POLARITY_T_MIN_WIDTH_SEC * fs)))
    later = [
        item
        for item in candidates
        if item.index > selected.index + min_gap
        and item.time_from_qrs_off_ms >= _LATE_SAME_POLARITY_T_MIN_TIME_MS
        and _sign(item.amp_mv) == selected_sign
        and item.width_samples >= min_width
        and item.abs_area >= _T_SIGNIFICANT_AREA_MV_MS
        and (
            item.abs_area >= _LATE_SAME_POLARITY_T_MIN_AREA_RATIO * selected.abs_area
            or abs(item.amp_mv) >= _LATE_SAME_POLARITY_T_MIN_AMP_RATIO * abs(selected.amp_mv)
        )
    ]
    if not later:
        return None
    return max(later, key=lambda item: (item.abs_area * max(item.width_samples, 1), item.abs_area, item.index))


def _simple_t_end(
    sig: np.ndarray,
    peak: Optional[int],
    local_t_cap: int,
    baseline: float,
    polarity: Optional[int],
    fs: int,
) -> Optional[int]:
    if peak is None:
        return None
    cap = min(len(sig) - 1, int(local_t_cap))
    if cap <= peak:
        return None
    amp = abs(float(sig[peak] - baseline))
    threshold = max(0.006, 0.08 * amp)
    for idx in range(peak + max(1, int(0.040 * fs)), cap + 1):
        if abs(float(sig[idx] - baseline)) <= threshold:
            return int(idx)
    return int(cap)


def detect_t_wave(
    *,
    sig: np.ndarray,
    sig_sm: np.ndarray,
    baseline: float,
    search_lo: int,
    search_hi: int,
    qrs_off: Optional[int],
    local_t_cap: int,
    fs: int,
    lead: str,
    expected_polarity: Optional[int] = None,
    cluster_polarity: Optional[int] = None,
    allow_late_same_polarity_st_rescue: bool = True,
) -> TWaveMeasurement:
    polarity = cluster_polarity if cluster_polarity is not None else expected_polarity
    candidates = _candidate_indices(sig_sm, search_lo, search_hi, baseline, fs, lead, qrs_off)
    selected = _choose_signed_candidate(candidates, polarity)
    components = _t_components(
        sig_sm,
        lo=search_lo,
        hi=local_t_cap,
        baseline=baseline,
        fs=fs,
        qrs_off=qrs_off,
    )
    component_selected, selected_component = _choose_component_candidate(
        components,
        polarity=polarity,
        search_hi=search_hi,
        lead=lead,
        fs=fs,
        prefer_late_component=allow_late_same_polarity_st_rescue,
    )
    if component_selected is not None:
        if selected is None:
            selected = component_selected
        elif (
            (
                allow_late_same_polarity_st_rescue
                and selected.time_from_qrs_off_ms < _T_ST_SEGMENT_GUARD_MS
            )
            or component_selected.abs_area * max(component_selected.width_samples, 1)
            > selected.abs_area * max(selected.width_samples, 1)
        ):
            selected = component_selected
        else:
            selected_component = next(
                (item for item in components if item.start <= selected.index <= item.end),
                selected_component,
            )
    else:
        selected_component = next(
            (item for item in components if selected is not None and item.start <= selected.index <= item.end),
            None,
        )

    st_t_confusion = False
    reason = None
    polarity_override = False
    override = _earlier_opposite_component_over_late_polarity_candidate(
        candidates,
        selected,
        fs=fs,
    )
    if override is not None:
        selected = override
        selected_component = next(
            (item for item in components if item.start <= selected.index <= item.end),
            selected_component,
        )
        polarity_override = True
    if (selected is not None and qrs_off is not None
            and selected.time_from_qrs_off_ms >= _T_ST_SEGMENT_GUARD_MS):
        selected_sign = _sign(selected.amp_mv)
        early_st: List[TWaveCandidate] = []
        if abs(selected.amp_mv) >= _ST_T_CONFUSION_MIN_SELECTED_AMP_MV:
            early_st = [
                item
                for item in candidates
                if item.index < selected.index - int(0.040 * fs)
                and item.time_from_qrs_off_ms < 150.0
                and selected_sign != 0
                and _sign(item.amp_mv) == -selected_sign
                and abs(item.amp_mv) >= _ST_T_CONFUSION_MIN_OPPOSITE_AMP_MV
                # Brief but deep post-ST troughs can have modest area while still
                # competing with the later signed T wave.
                and (
                    item.abs_area >= _EARLY_ST_TROUGH_MIN_AREA_RATIO * selected.abs_area
                    or abs(item.amp_mv) >= _EARLY_ST_TROUGH_MIN_AMP_RATIO * abs(selected.amp_mv)
                )
                and item.width_samples >= max(4, int(_EARLY_ST_TROUGH_MIN_WIDTH_SEC * fs))
            ]
        if early_st:
            st_t_confusion = True
            reason = "later_signed_t_after_st_trough"
    if (
        selected is not None
        and qrs_off is not None
        and allow_late_same_polarity_st_rescue
        and selected.time_from_qrs_off_ms < 150.0
    ):
        selected_sign = _sign(selected.amp_mv)
        later_signed = [
            item
            for item in candidates
            if item.index > selected.index + int(0.040 * fs)
            and (
                (_sign(item.amp_mv) == polarity)
                if polarity not in (None, 0)
                else selected_sign != 0 and _sign(item.amp_mv) == -selected_sign
            )
            and item.time_from_qrs_off_ms >= _T_ST_SEGMENT_GUARD_MS
            and abs(selected.amp_mv) >= _ST_T_CONFUSION_MIN_OPPOSITE_AMP_MV
            and abs(item.amp_mv) >= _ST_T_CONFUSION_MIN_SELECTED_AMP_MV
            and item.width_samples >= max(6, int(0.030 * fs))
        ]
        if later_signed:
            later = max(later_signed, key=lambda item: (item.abs_area * max(item.width_samples, 1), item.index))
            if selected.abs_area >= 0.75 * later.abs_area:
                selected = later
                st_t_confusion = True
                reason = "later_signed_t_after_st_trough"
    if selected is not None and qrs_off is not None and allow_late_same_polarity_st_rescue:
        later_same = _later_same_polarity_t_after_st_trough(candidates, selected, fs=fs)
        if later_same is not None:
            selected = later_same
            selected_component = next(
                (item for item in components if item.start <= selected.index <= item.end),
                selected_component,
            )
            st_t_confusion = True
            reason = "later_signed_t_after_st_trough"
    observed = _sign(selected.amp_mv) if selected is not None else 0
    offset = _simple_t_end(sig, None if selected is None else selected.index, local_t_cap, baseline, observed, fs)
    t_prime_component = _later_t_prime_component(components, selected_component, fs=fs)
    t_prime_peak = None if t_prime_component is None else int(t_prime_component.peak)
    if t_prime_component is not None:
        if offset is None or int(t_prime_component.end) > int(offset):
            offset = int(t_prime_component.end)
    confidence = 0.0 if selected is None else min(1.0, max(0.15, selected.abs_area / (selected.abs_area + 0.05)))
    return TWaveMeasurement(
        peak=None if selected is None else int(selected.index),
        offset=offset,
        confidence=confidence,
        method="polarity_cluster",
        polarity_expected=expected_polarity,
        polarity_observed=observed or None,
        st_t_confusion=st_t_confusion,
        confidence_reason=reason or (
            "earlier_opposite_t_over_late_polarity_candidate"
            if polarity_override else None
        ),
        flags=(
            (["st_t_confusion"] if st_t_confusion else [])
            + (["late_polarity_override"] if polarity_override else [])
        ),
        t_prime_peak=t_prime_peak,
    )

"""Bounded measurement review of the calibrated *original* ECG.

This is a second observation of the same acquisition, not an independent
acquisition, clinical validation, diagnosis, or replacement delineator. Exported
boundaries are hypotheses: local P morphology and QRS/T overlap can contradict
them, and signed extrema can contradict exported R/S measurements. Quiet PR
anchors support a raw ST observation only between two nearby anchors. Agreement
does not validate the upstream timing or establish the identity of a P wave.

The fixed tolerances below are engineering guards, not diagnostic thresholds.
Sampling quantization, imperfect boundaries, residual noise, and shared timing
errors remain limitations. No experimental RefinementConfig or candidate
detector is enabled. The small existing P chord/width helper is reused alone.
Input arrays/documents are never changed; no source text or patient labels are
copied. Work and output are capped, with omitted observations reported.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

SCHEMA_VERSION = "ecgagent.waveform-review.v1"
STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PROFILES = ("p_av", "qrs", "st")
MAX_FEATURE_ROWS = 4096
MAX_P_EVENTS = 16
MAX_BEATS_PER_LEAD = 3
MAX_OBSERVATIONS = 36


def _map(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, (bool, str)) or value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _profile(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "conflict_reasons": [],
            "missing_requirements": [reason], "input_count": 0,
            "reviewed_count": 0, "truncated": False, "observations": []}


def unavailable_waveform_review(reason: str = "original_waveform_missing") -> dict[str, Any]:
    """An explicit unavailable artifact; never reconstruct raw ECG from features."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unavailable",
        "provenance": {
            "source": "original_calibrated_ecg", "same_acquisition": True,
            "independent_acquisition": False, "clinical_validation": False,
            "experimental_candidates_enabled": False,
            "bounds_source": "exported_features", "amplitude_unit": "mV",
            "timebase": "feature_samples_to_original_samples",
        },
        "profiles": {name: _profile(reason) for name in PROFILES},
    }


def calibrate_to_mv(ecg: Any, units: Sequence[str]) -> np.ndarray:
    """Convert explicitly declared physical WFDB units; never assume ADC units."""
    values = np.asarray(ecg, dtype=float)
    scales = {"mv": 1.0, "uv": .001, "µv": .001, "μv": .001, "v": 1000.0}
    if values.ndim != 2 or len(units) != values.shape[0]:
        raise ValueError("waveform lead units are missing or do not match channels")
    try:
        gains = [scales[str(unit).strip().lower()] for unit in units]
    except KeyError as exc:
        raise ValueError("waveform requires explicit physical V, mV, or uV units") from exc
    return values * np.asarray(gains)[:, None]


def _finish(profile: dict[str, Any], rows: list[dict[str, Any]], count: int) -> None:
    profile.update(observations=rows, input_count=count, reviewed_count=len(rows),
                   truncated=count > len(rows))
    profile["conflict_reasons"] = sorted({reason for row in rows for reason in row["conflict_reasons"]})
    profile["missing_requirements"] = sorted({reason for row in rows for reason in row["missing_requirements"]})
    if not rows:
        profile["missing_requirements"] = ["exported_candidates_missing"]
    if profile["truncated"]:
        profile["missing_requirements"].append("review_limit_reached")
    profile["status"] = (
        "conflict" if profile["conflict_reasons"] else
        "observations_available" if any(row["status"] == "observations_available" for row in rows)
        else "unavailable"
    )


def _row(**values: Any) -> dict[str, Any]:
    return {"status": "unavailable", "conflict_reasons": [], "missing_requirements": [], **values}


def _missing(row: dict, reason: str) -> dict:
    row["missing_requirements"].append(reason)
    return row


def _conflict(row: dict, reason: str) -> dict:
    row["status"] = "conflict"
    row["conflict_reasons"].append(reason)
    return row


def build_waveform_review(
    ecg: Any,
    fs: float | None,
    features: Mapping[str, Any],
    *,
    lead_names: Sequence[str] | None = None,
    amplitude_unit: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Review exported candidates against original channels in [lead, sample] order.

    ``fs`` is the original rate; ``features['fs']`` is the boundary-index rate.
    Both refer to sample zero of the same record. Callers must pair the record
    explicitly; duration/rate checks cannot authenticate record identity.
    Units and lead names must be explicit arguments or acquisition metadata.
    Missing/contradictory metadata never produces affirmative review evidence.
    """
    artifact = unavailable_waveform_review()
    if ecg is None:
        return artifact
    metadata = _map(features.get("metadata"))
    contract = _map(metadata.get("input_contract"))
    original_fs, feature_fs = _number(fs), _number(features.get("fs"))
    if original_fs is None or feature_fs is None or not (50 <= original_fs <= 4000 and 50 <= feature_fs <= 4000):
        return unavailable_waveform_review("sampling_rate_missing_or_invalid")
    try:
        values = np.asarray(ecg, dtype=float)
    except (TypeError, ValueError):
        return unavailable_waveform_review("raw_shape_invalid")
    if values.ndim != 2 or not (1 <= values.shape[0] <= 12) or values.shape[1] < 10:
        return unavailable_waveform_review("raw_shape_invalid")
    if values.shape[1] / original_fs > 120:
        return unavailable_waveform_review("review_limit_reached")
    if lead_names is None:
        lead_names = metadata.get("lead_order")
    if (not isinstance(lead_names, (list, tuple)) or len(lead_names) != values.shape[0]
            or any(name not in STANDARD_LEADS for name in lead_names)
            or len(set(lead_names)) != len(lead_names)):
        return unavailable_waveform_review("lead_mapping_missing_or_invalid")
    if amplitude_unit is None:
        amplitude_unit = contract.get("amplitude_input_unit")
    units = [amplitude_unit] * len(lead_names) if isinstance(amplitude_unit, str) else amplitude_unit
    if not isinstance(units, (list, tuple)):
        return unavailable_waveform_review("amplitude_units_missing_or_invalid")
    try:
        values = calibrate_to_mv(values, units)
    except ValueError:
        return unavailable_waveform_review("amplitude_units_missing_or_invalid")
    for owner, key, expected in ((metadata, "input_fs", original_fs), (contract, "input_fs", original_fs),
                                 (metadata, "internal_fs", feature_fs), (contract, "internal_fs", feature_fs)):
        if key in owner and (_number(owner[key]) is None or not math.isclose(float(owner[key]), expected, abs_tol=1e-6)):
            return unavailable_waveform_review("timebase_conflict")
    for owner in (metadata, contract):
        if "duration_sec" in owner:
            duration = _number(owner["duration_sec"])
            if duration is None or abs(duration - values.shape[1] / original_fs) > 2 / min(original_fs, feature_fs):
                return unavailable_waveform_review("record_duration_conflict")
    if contract.get("valid") is False:
        return unavailable_waveform_review("input_contract_invalid")
    artifact["provenance"].update(
        original_fs_hz=original_fs, feature_fs_hz=feature_fs,
        sample_count=values.shape[1], duration_ms=1000 * values.shape[1] / original_fs,
        lead_names=list(lead_names), sample_rounding_error_ms=500 / original_fs,
        raw_sha256=hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest(),
        max_p_events=MAX_P_EVENTS, max_beats_per_lead=MAX_BEATS_PER_LEAD,
    )
    quality = _map(features.get("quality"))
    channels = dict(zip(lead_names, values))
    feature_rows = features.get("beat_features")
    feature_rows = feature_rows if isinstance(feature_rows, list) else []
    bounded_features = [(i, row) for i, row in enumerate(feature_rows[:MAX_FEATURE_ROWS]) if isinstance(row, Mapping)]
    by_lead = {lead: [(i, row) for i, row in bounded_features if row.get("lead") == lead] for lead in lead_names}

    @lru_cache(maxsize=36)
    def quality_ok(lead: str, domain: str) -> bool:
        q = _map(quality.get(lead))
        signal = channels[lead]
        for field, ceiling in (("clipping_score", .05), ("saturation_fraction", .05), ("flatline_fraction", .5)):
            if q.get(field) is not None:
                number = _number(q[field])
                if number is None or not 0 <= number <= ceiling:
                    return False
        return (q.get("reliable") is True and q.get("missing") is False
                and q.get(f"reliable_for_{domain}", True) is True
                and np.isfinite(signal).all() and np.ptp(signal) > .005)

    def sample(value: Any) -> int | None:
        number = _number(value)
        if number is None or number < 0 or not number.is_integer():
            return None
        index = int(round(number * original_fs / feature_fs))
        return index if 0 <= index < values.shape[1] else None

    def bounds(row: Mapping, wave: str) -> tuple[int, int, int] | None:
        wave_values = _map(row.get(wave))
        a, peak, b = (sample(wave_values.get(key)) for key in ("onset", "peak", "offset"))
        if a is None or peak is None or b is None or not a <= peak < b:
            return None
        return a, peak, b

    # Filtering is only a local morphology aid; R/S and ST use raw samples.
    try:
        from ecgfeat.atrial_validation import p_waveform_evidence
        from ecgfeat.preprocess import lowpass_filter
    except ImportError:
        return unavailable_waveform_review("measurement_helper_unavailable")
    filtered = {lead: lowpass_filter(signal, original_fs, 15.)
                for lead, signal in channels.items() if quality_ok(lead, "p")}

    def p_evidence(lead: str, peak: int) -> dict | None:
        if lead not in filtered:
            return None
        return p_waveform_evidence(channels[lead], filtered[lead], peak, original_fs)

    ventricular_intervals = {
        lead: [(bounds(feature, wave), reason)
               for _, feature in rows
               for wave, reason in (("qrs", "qrs_overlap"), ("t", "t_overlap"))]
        for lead, rows in by_lead.items()
    }
    timing_complete = {
        lead: bool(intervals) and all(interval is not None for interval, _ in intervals)
        for lead, intervals in ventricular_intervals.items()
    }

    def overlaps(lead: str, start: int, end: int) -> set[str]:
        conflicts = set()
        for interval, reason in ventricular_intervals[lead]:
            if interval is not None:
                guard = round(.012 * original_fs) if reason == "qrs_overlap" else 0
                if start <= interval[2] + guard and end >= interval[0] - guard:
                    conflicts.add(reason)
        return conflicts

    p_rows: list[dict] = []
    events = _map(features.get("rhythm_inputs")).get("p_events")
    events = events if isinstance(events, list) else []
    for event_index, event in enumerate(events[:MAX_P_EVENTS]):
        row = _row(event_index=event_index)
        p_rows.append(row)
        event = _map(event)
        time_ms = _number(event.get("time_ms"))
        peak = int(round(time_ms * original_fs / 1000)) if time_ms is not None else sample(event.get("sample"))
        if peak is None or not 0 <= peak < values.shape[1]:
            _missing(row, "candidate_timing_missing_or_invalid")
            continue
        if event.get("sample") is not None and sample(event["sample"]) != peak:
            _conflict(row, "event_timebase_conflict")
            continue
        row.update(raw_sample=peak, time_ms=1000 * peak / original_fs)
        if peak - round(.080 * original_fs) < 0 or peak + round(.080 * original_fs) >= values.shape[1]:
            _missing(row, "local_window_incomplete")
            continue
        sources = event.get("source_leads")
        if not sources:
            sources = list(lead_names)
        if (not isinstance(sources, (list, tuple)) or len(sources) > 12
                or any(not isinstance(lead, str) or lead not in channels for lead in sources)):
            _missing(row, "lead_mapping_missing_or_invalid")
            continue
        candidates = [lead for lead in sources if quality_ok(lead, "p")]
        if not candidates:
            _missing(row, "lead_quality_missing_or_invalid")
            continue
        onset, offset = (_number(event.get(key)) for key in ("onset_ms", "offset_ms"))
        start, end = peak, peak
        if onset is not None and offset is not None:
            start, end = round(onset * original_fs / 1000), round(offset * original_fs / 1000)
            if not 0 <= start <= peak <= end < values.shape[1]:
                _conflict(row, "event_timebase_conflict")
                continue
        else:
            _missing(row, "candidate_boundaries_missing_or_invalid")
        conflicts = set().union(*(overlaps(lead, start, end) for lead in candidates))
        for reason in sorted(conflicts):
            _conflict(row, reason)
        evidence = [(lead, p_evidence(lead, peak)) for lead in candidates]
        supported = [(lead, ev) for lead, ev in evidence if ev is not None]
        row["reviewed_leads"] = candidates
        row["supporting_leads"] = [lead for lead, _ in supported]
        if supported:
            lead, ev = max(supported, key=lambda pair: pair[1]["snr"])
            row.update(lead=lead, raw_prominence_mv=ev["amplitude_mv"],
                       local_snr=ev["snr"], half_height_width_ms=ev["half_height_width_ms"])
        else:
            _conflict(row, "local_p_morphology_not_supported")
        if not all(timing_complete[lead] for lead in candidates):
            _missing(row, "ventricular_timing_missing")
        if row["status"] != "conflict" and not row["missing_requirements"]:
            row["status"] = "observations_available"
    _finish(artifact["profiles"]["p_av"], p_rows, len(events))

    qrs_rows: list[dict] = []
    st_rows: list[dict] = []
    for lead in lead_names:
        signal = channels[lead]
        rows = by_lead[lead]
        anchors: list[tuple[float, float, float]] = []
        # PR baseline requires explicit trustworthy P/QRS bounds, local P shape,
        # and quiet raw samples. Never substitute a QRS baseline or zero.
        if quality_ok(lead, "p") and quality_ok(lead, "qrs") and timing_complete[lead]:
            for _, feature in rows:
                p, qrs = bounds(feature, "p"), bounds(feature, "qrs")
                flags = feature.get("flags") or []
                if (p is None or qrs is None or feature.get("beat_measurement_reliable") is not True
                        or not isinstance(flags, (list, tuple))
                        or "p_unreliable" in flags or p_evidence(lead, p[1]) is None
                        or overlaps(lead, p[0], p[2])):
                    continue
                hi = qrs[0] - round(.008 * original_fs)
                lo = max(p[2] + round(.008 * original_fs), hi - round(.060 * original_fs))
                if hi - lo < max(3, round(.016 * original_fs)):
                    continue
                segment = signal[lo:hi]
                baseline = float(np.median(segment))
                noise = 1.4826 * float(np.median(abs(segment - baseline)))
                half = max(1, len(segment) // 2)
                drift = abs(float(np.median(segment[:half]) - np.median(segment[-half:])))
                if noise <= .020 and drift <= .025:
                    anchors.append(((lo + hi - 1) / 2, baseline, max(noise, .001)))
        anchors = sorted({anchor[0]: anchor for anchor in anchors}.values())
        chosen = np.linspace(0, len(rows) - 1, min(MAX_BEATS_PER_LEAD, len(rows)), dtype=int) if rows else []
        for selected in chosen:
            feature_index, feature = rows[selected]
            base = {"lead": lead, "feature_index": feature_index}
            beat_id = _number(feature.get("beat_id"))
            if beat_id is not None:
                base["beat_id"] = beat_id
            qr, st = _row(**base), _row(**base)
            qrs_rows.append(qr)
            st_rows.append(st)
            if not quality_ok(lead, "qrs") or feature.get("beat_measurement_reliable") is not True:
                _missing(qr, "lead_quality_missing_or_invalid")
                _missing(st, "lead_quality_missing_or_invalid")
                continue
            qrs = bounds(feature, "qrs")
            if qrs is None or not .020 <= (qrs[2] - qrs[0]) / original_fs <= .300:
                _missing(qr, "qrs_timing_missing_or_invalid")
                _missing(st, "qrs_timing_missing_or_invalid")
                continue
            # A quiet pre-QRS window supports polarity when a P/PR anchor is
            # unavailable. This fallback is never used for ST measurement.
            preceding = [a for a in anchors if 0 < qrs[0] - a[0] <= .200 * original_fs]
            qr["baseline_method"] = "quiet_pr_anchor"
            if not preceding:
                lo, hi = qrs[0] - round(.050 * original_fs), qrs[0] - round(.016 * original_fs)
                p = bounds(feature, "p")
                if (lo >= 0 and hi - lo >= 3 and not overlaps(lead, lo, hi)
                        and not (p is not None and lo <= p[2] and hi >= p[0])):
                    segment = signal[lo:hi]
                    center = float(np.median(segment))
                    noise = 1.4826 * float(np.median(abs(segment - center)))
                    if noise <= .020 and float(np.ptp(segment)) <= .050:
                        preceding = [((lo + hi - 1) / 2, center, max(noise, .001))]
                        qr["baseline_method"] = "quiet_pre_qrs"
            if preceding:
                anchor = preceding[-1]
                segment = signal[qrs[0]:qrs[2] + 1] - anchor[1]
                maximum, minimum = float(np.max(segment)), float(np.min(segment))
                r, s = _number(feature.get("r_amp_mv")), _number(feature.get("s_amp_mv"))
                tol = max(.025, 3 * anchor[2])
                qr.update(raw_max_mv=maximum, raw_min_mv=minimum, baseline_mv=anchor[1],
                          raw_qs_candidate=maximum <= tol and minimum < -tol,
                          raw_qrs_onset_sample=qrs[0], raw_qrs_offset_sample=qrs[2],
                          amplitude_tolerance_mv=tol)
                if r is None or s is None:
                    _missing(qr, "exported_signed_amplitudes_missing")
                else:
                    qr.update(exported_r_mv=r, exported_s_mv=s)
                    if r < -tol or s > tol or (r > tol and maximum <= tol) or (s < -tol and minimum >= -tol):
                        _conflict(qr, "signed_rs_conflict")
                    if qr["raw_qs_candidate"] and r > tol:
                        _conflict(qr, "qs_positive_r_conflict")
                    if qr["status"] != "conflict":
                        qr["status"] = "observations_available"
            else:
                _missing(qr, "quiet_qrs_baseline_missing")
            if not quality_ok(lead, "t") or not quality_ok(lead, "p"):
                _missing(st, "lead_quality_missing_or_invalid")
                continue
            st["anchor_count"] = len(anchors)
            j = qrs[2]
            st["st_anchor_source"] = "exported_qrs_offset"
            if feature.get("st_j_remeasured_index") is not None:
                j = sample(feature["st_j_remeasured_index"])
                if j is None or j < qrs[1] or abs(j - qrs[2]) > .100 * original_fs:
                    _missing(st, "st_anchor_timing_invalid")
                    continue
                st["st_anchor_source"] = "exported_remeasured_j"
            st["raw_j_sample"] = j
            target = j + round(.080 * original_fs)
            radius = max(1, round(.008 * original_fs))
            t = bounds(feature, "t")
            if t is None or target - radius <= qrs[2] or target + radius >= t[0] or target + radius >= len(signal):
                _missing(st, "st_window_overlaps_t_or_record_edge")
                continue
            before = [a for a in anchors if a[0] <= target]
            after = [a for a in anchors if a[0] > target]
            if not before or not after:
                _missing(st, "bracketing_pr_anchors_missing")
                continue
            left, right = before[-1], after[0]
            if (right[0] - left[0]) / original_fs > 2.5:
                _missing(st, "pr_anchor_gap_too_large")
                continue
            baseline = float(np.interp(target, [left[0], right[0]], [left[1], right[1]]))
            local = signal[target - radius:target + radius + 1]
            level = float(np.median(local))
            local_noise = max(
                1.4826 * float(np.median(abs(local - level))),
                float(np.median(abs(np.diff(local)))) / math.sqrt(2),
            )
            if local_noise > .025:
                _missing(st, "st_local_noise_excessive")
                continue
            amplitude = level - baseline
            uncertainty = max(left[2], right[2]) + .010 * min(target - left[0], right[0] - target) / original_fs
            st.update(raw_st_80ms_mv=amplitude, baseline_mv=baseline,
                      baseline_uncertainty_mv=uncertainty, raw_sample=target,
                      left_anchor_sample=left[0], right_anchor_sample=right[0],
                      amplitude_tolerance_mv=max(.050, 3 * uncertainty))
            exported = _number(feature.get("st_80ms_mv"))
            if exported is None:
                _missing(st, "exported_st_amplitude_missing")
            else:
                st["exported_st_80ms_mv"] = exported
                st["difference_mv"] = amplitude - exported
                st["status"] = "observations_available"
                if abs(amplitude - exported) > st["amplitude_tolerance_mv"]:
                    _conflict(st, "st_amplitude_disagreement")
    count = sum(len(rows) for rows in by_lead.values())
    _finish(artifact["profiles"]["qrs"], qrs_rows, count)
    _finish(artifact["profiles"]["st"], st_rows, count)
    if len(feature_rows) > MAX_FEATURE_ROWS:
        for profile in artifact["profiles"].values():
            profile["truncated"] = True
            if "review_limit_reached" not in profile["missing_requirements"]:
                profile["missing_requirements"].append("review_limit_reached")
    statuses = [profile["status"] for profile in artifact["profiles"].values()]
    artifact["status"] = ("conflict" if "conflict" in statuses else
                          "observations_available" if "observations_available" in statuses else "unavailable")
    return artifact


__all__ = ["SCHEMA_VERSION", "build_waveform_review", "unavailable_waveform_review", "calibrate_to_mv"]

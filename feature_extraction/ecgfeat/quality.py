from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks, welch

from .clinical_rules.config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import LeadQuality, STANDARD_12_LEADS
from .numeric import trapezoid
from .preprocess import bandpass_filter, highpass_filter, lowpass_filter


# 12SL pacer-spike amplitude gates (Marquette 12SL Measurement chapter):
#   - large spike  : > 1000 microvolt  -> accepted outright
#   - low  spike   : > 250  microvolt  -> accepted only under further scrutiny
# The default detector enforces the low gate as its absolute prominence floor.
# Attenuated-spike rescue, when needed, lives in api.py and must pass QRS/capture
# validation before it can affect downstream measurements.
PACE_SPIKE_HIGH_UV = 1000.0
PACE_SPIKE_LOW_UV = 250.0


def _band_power(x: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), max(256, fs * 2)))
    return _spectrum_band_power(freqs, psd, fmin, fmax)


def _spectrum_band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Integrate an existing spectrum, preserving the historical masks/order."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def _clipping_score(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    diffs = np.diff(x)
    return float(np.mean(np.isclose(diffs, 0.0, atol=1e-8)))


def _flatline_score(x: np.ndarray) -> float:
    return float(np.std(x))


def _flatline_fraction(x: np.ndarray, fs: int, amplitude_mv: float) -> float:
    if x.size == 0 or fs <= 0:
        return 1.0
    window = max(1, int(round(float(fs))))
    local_range = (
        maximum_filter1d(x, size=window, mode="nearest")
        - minimum_filter1d(x, size=window, mode="nearest")
    )
    return float(np.mean(local_range < float(amplitude_mv)))


def _saturation_fraction(
    x: np.ndarray, adc_full_scale_mv: Optional[float]
) -> Optional[float]:
    if adc_full_scale_mv is None:
        return None
    try:
        full_scale = abs(float(adc_full_scale_mv))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(full_scale) or full_scale <= 0.0:
        return None
    return float(np.mean(np.abs(x) >= 0.95 * full_scale))


def _kurtosis_score(x: np.ndarray) -> float:
    centered = np.asarray(x, dtype=float) - float(np.mean(x))
    variance = float(np.mean(centered * centered))
    if variance <= 1e-12:
        return 0.0
    return float(np.mean(centered**4) / (variance * variance))


def _simple_qrs_detector(
    sig: np.ndarray,
    fs: int,
    *,
    method: str,
) -> np.ndarray:
    if fs <= 0 or sig.size < max(8, int(0.5 * fs)):
        return np.asarray([], dtype=int)
    high = min(25.0 if method == "energy" else 18.0, fs / 2.0 - 1.0)
    low = 5.0 if method == "energy" else 8.0
    if high <= low:
        return np.asarray([], dtype=int)
    filtered = bandpass_filter(np.asarray(sig, dtype=float), fs, low, high, order=2)
    if method == "energy":
        detection = np.gradient(filtered) ** 2
        window = max(1, int(round(0.100 * fs)))
        detection = np.convolve(detection, np.ones(window) / window, mode="same")
    else:
        detection = np.abs(filtered - np.median(filtered))
        window = max(1, int(round(0.060 * fs)))
        detection = np.convolve(detection, np.ones(window) / window, mode="same")
    median = float(np.median(detection))
    mad = 1.4826 * float(np.median(np.abs(detection - median)))
    threshold = max(
        median + (3.5 if method == "energy" else 3.0) * mad,
        float(np.percentile(detection, 90)) * 0.45,
        1e-10,
    )
    peaks, _ = find_peaks(
        detection,
        height=threshold,
        distance=max(1, int(round(0.20 * fs))),
    )
    return np.asarray(peaks, dtype=int)


def _bsqi(sig: np.ndarray, fs: int) -> float:
    first = _simple_qrs_detector(sig, fs, method="energy")
    second = _simple_qrs_detector(sig, fs, method="amplitude")
    if first.size == 0 and second.size == 0:
        return 0.0
    tolerance = max(1, int(round(0.100 * fs)))
    used: set[int] = set()
    matches = 0
    for peak in first:
        candidates = [
            (abs(int(other) - int(peak)), idx)
            for idx, other in enumerate(second)
            if idx not in used and abs(int(other) - int(peak)) <= tolerance
        ]
        if not candidates:
            continue
        _, winner = min(candidates)
        used.add(winner)
        matches += 1
    return float(matches / max(len(first), len(second), 1))


def compute_qrs_detector_agreement(
    ecg: np.ndarray,
    fs: int,
    reference_r_locs: np.ndarray,
    qualities: Dict[str, LeadQuality],
) -> Dict[str, object]:
    """Compare the production QRS detector with an independent amplitude path."""
    candidates: List[tuple[int, str]] = []
    for lead in ("II", "V1", "V2", "V5"):
        quality = qualities.get(lead)
        if quality is None or not bool(quality.reliable_for_qrs):
            continue
        row = STANDARD_12_LEADS.index(lead)
        for peak in _simple_qrs_detector(ecg[row], fs, method="amplitude"):
            candidates.append((int(peak), lead))
    tolerance = max(1, int(round(0.10 * fs)))
    clusters: List[List[tuple[int, str]]] = []
    for peak, lead in sorted(candidates):
        if not clusters or peak - int(np.median([item[0] for item in clusters[-1]])) > tolerance:
            clusters.append([(peak, lead)])
        else:
            clusters[-1].append((peak, lead))
    alternative = np.asarray(
        [
            int(round(np.median([peak for peak, _ in cluster])))
            for cluster in clusters
            if len({lead for _, lead in cluster}) >= 2
        ],
        dtype=int,
    )
    reference = np.asarray(reference_r_locs, dtype=int)
    used: set[int] = set()
    matches = 0
    for peak in reference:
        eligible = [
            (abs(int(other) - int(peak)), index)
            for index, other in enumerate(alternative)
            if index not in used and abs(int(other) - int(peak)) <= tolerance
        ]
        if eligible:
            _, index = min(eligible)
            used.add(index)
            matches += 1
    precision = matches / float(max(len(reference), 1))
    recall = matches / float(max(len(alternative), 1))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "production_count": int(len(reference)),
        "independent_count": int(len(alternative)),
        "matched_count": int(matches),
        "tolerance_ms": 100.0,
        "precision_vs_independent": float(precision),
        "recall_vs_independent": float(recall),
        "f1_agreement": float(f1),
        "independent_r_locs": alternative.tolist(),
        "usable": bool(len(reference) >= 2 and len(alternative) >= 2),
    }


def summarize_record_quality(qualities: Dict[str, LeadQuality], *, available_leads=None) -> Dict[str, object]:
    if available_leads is not None:
        active = [qualities[name] for name in available_leads if name in qualities]
        qrs = sum(bool(q.reliable_for_qrs) for q in active)
        p = sum(bool(q.reliable_for_p) for q in active)
        qt = sum(bool(q.reliable_for_qt) for q in active)
        required = min(2, len(available_leads))
        rejected = ([] if qrs else ["record"]) + ([] if p >= required else ["p_measurement"]) + ([] if qt >= required else ["qt_measurement"])
        reasons = sorted({code for q in active for code in (q.reason_codes or q.flags)})
        if len(active) != len(available_leads):
            rejected = sorted(set(rejected) | {"record"})
            reasons.append("partial_quality_map")
        return {"record_grade": "Q3" if "record" in rejected else "Q2" if rejected else "Q1" if reasons else "Q0",
                "rejected_functions": rejected, "reason_codes": reasons,
                "n_reliable_qrs_leads": qrs, "n_reliable_p_leads": p, "n_reliable_qt_leads": qt,
                "diagnostic_gate": "stop" if "record" in rejected else "partial",
                "quality_scope": "explicitly_supplied_channels", "available_leads": list(available_leads)}
    lead_qualities = [
        quality
        for lead in STANDARD_12_LEADS
        if (quality := qualities.get(lead)) is not None
    ]
    reliable_qrs = sum(int(bool(q.reliable_for_qrs)) for q in lead_qualities)
    reliable_p = sum(int(bool(q.reliable_for_p)) for q in lead_qualities)
    reliable_qt = sum(int(bool(q.reliable_for_qt)) for q in lead_qualities)
    reason_counts: Dict[str, int] = {}
    for quality in lead_qualities:
        for code in set(quality.reason_codes or quality.flags):
            reason_counts[code] = reason_counts.get(code, 0) + 1

    # pSQI, bSQI and kurtosis are useful lead-level detector diagnostics, but
    # isolated excursions are common on otherwise interpretable 12-lead
    # records. Preserve their counts for audit and promote them to a record
    # quality reason only when the issue is multi-lead.
    advisory_thresholds = {
        "psqi_out_of_range": max(4, len(lead_qualities) // 2),
        "low_bsqi": max(4, len(lead_qualities) // 3),
        "low_kurtosis": max(4, len(lead_qualities) // 2),
    }
    advisory_reason_counts = {
        code: count
        for code, count in reason_counts.items()
        if code in advisory_thresholds
    }
    unpromoted_advisories = sorted(
        code
        for code, count in advisory_reason_counts.items()
        if count < advisory_thresholds[code]
    )
    reason_codes = sorted(
        code
        for code, count in reason_counts.items()
        if code not in advisory_thresholds
        or count >= advisory_thresholds[code]
    )
    partial_map = len(lead_qualities) != len(STANDARD_12_LEADS)
    if partial_map:
        reason_codes = sorted(set(reason_codes) | {"partial_quality_map"})

    rejected_functions: List[str] = []
    if reliable_p < 2:
        rejected_functions.append("p_measurement")
    if reliable_qt < 2:
        rejected_functions.append("qt_measurement")
    preferred_anchor_available = any(
        bool(getattr(qualities.get(lead), "reliable_for_qrs", False))
        for lead in ("II", "V1")
    )
    fallback_anchor_leads = [
        lead
        for lead in ("I", "III", "aVF", "V2", "V5")
        if bool(getattr(qualities.get(lead), "reliable_for_qrs", False))
    ]
    required_anchor_available = bool(
        preferred_anchor_available
        or (
            reliable_qrs >= DEFAULT_DIAGNOSTIC_CONFIG.quality.minimum_usable_leads
            and len(fallback_anchor_leads) >= 2
        )
    )
    b_sqi_values = [
        float(q.b_sqi)
        for q in lead_qualities
        if q.b_sqi is not None and np.isfinite(float(q.b_sqi))
    ]
    all_bsqi_below_stop = bool(
        b_sqi_values
        and all(
            value < DEFAULT_DIAGNOSTIC_CONFIG.quality.bsqi_record_stop
            for value in b_sqi_values
        )
    )
    if (
        reliable_qrs < DEFAULT_DIAGNOSTIC_CONFIG.quality.minimum_usable_leads
        or not required_anchor_available
        or all_bsqi_below_stop
        or partial_map
    ):
        rejected_functions.append("record")

    if "record" in rejected_functions:
        record_grade = "Q3"
    elif rejected_functions:
        record_grade = "Q2"
    elif reason_codes:
        record_grade = "Q1"
    else:
        record_grade = "Q0"

    return {
        "record_grade": record_grade,
        "rejected_functions": rejected_functions,
        "reason_codes": reason_codes,
        "reason_counts": dict(sorted(reason_counts.items())),
        "advisory_reason_counts": dict(sorted(advisory_reason_counts.items())),
        "unpromoted_advisory_reason_codes": unpromoted_advisories,
        "advisory_promotion_thresholds": advisory_thresholds,
        "n_reliable_qrs_leads": reliable_qrs,
        "n_reliable_p_leads": reliable_p,
        "n_reliable_qt_leads": reliable_qt,
        "required_anchor_available": required_anchor_available,
        "preferred_anchor_available": preferred_anchor_available,
        "fallback_anchor_leads": fallback_anchor_leads,
        "all_bsqi_below_stop": all_bsqi_below_stop,
        "diagnostic_gate": "stop" if "record" in rejected_functions else (
            "partial"
            if reliable_qrs < DEFAULT_DIAGNOSTIC_CONFIG.quality.full_coverage_leads
            else "pass"
        ),
    }


# Gate reasons that name a testable claim about the patient, with the criteria
# that claim rests on. A reason is published with its criteria so downstream
# reasoning can *check* it rather than inherit it: a technical flag is a
# hypothesis about the recording, and hypotheses that cannot be refuted by the
# measurements are how a detector artefact becomes a diagnosis. Reasons absent
# from this table are reported without criteria and marked non-refutable, which
# says only that this module makes no checkable claim about them.
_REFUTABLE_GATE_FLAGS: Dict[str, Dict[str, object]] = {
    "probable_limb_lead_reversal": {
        "claim": "limb electrodes are probably swapped (most often LA/RA)",
        "refuted_when_any_fails": True,
        "criteria": [
            {
                "id": "lead_I_dominantly_negative",
                "description": "lead I QRS dominant deflection is negative",
                "pointers": [
                    "/representative_leads/I/params/r_amp_mv",
                    "/representative_leads/I/params/s_amp_mv",
                ],
                "test": "abs(s_amp_mv) > r_amp_mv",
            },
            {
                "id": "aVR_dominantly_positive",
                "description": "aVR QRS dominant deflection is positive",
                "pointers": [
                    "/representative_leads/aVR/params/r_amp_mv",
                    "/representative_leads/aVR/params/s_amp_mv",
                ],
                "test": "r_amp_mv > abs(s_amp_mv)",
            },
            {
                "id": "precordial_progression_normal",
                "description": (
                    "precordial leads are unaffected, which separates a limb "
                    "swap from dextrocardia"
                ),
                "pointers": ["/global_features/poor_r_progression"],
                "test": "poor_r_progression is not True",
            },
        ],
    },
    "precordial_lead_placement_suspected": {
        "claim": "precordial electrodes are probably misplaced or out of order",
        "refuted_when_any_fails": False,
        "criteria": [
            {
                "id": "adjacent_precordial_shape_discontinuity",
                "description": (
                    "adjacent precordial leads lose the smooth QRS shape "
                    "continuity that neighbouring electrodes produce"
                ),
                "pointers": ["/metadata/precordial_reversal/state"],
                "test": "state in {possible, confirmed}",
            }
        ],
    },
    "amplitudes_not_calibrated_per_lead_normalized": {
        "claim": (
            "every lead was rescaled to the same peak-to-peak span, so "
            "absolute amplitudes are not the patient's"
        ),
        # A property of the file, not of the heart. No patient measurement can
        # refute it, and none should be asked to.
        "refutable": False,
        "criteria": [],
    },
}


def build_refutable_gate_flags(
    *,
    stop_reasons: Sequence[str],
    partial_reasons: Sequence[str],
) -> List[Dict[str, object]]:
    """Publish each raised gate reason together with the criteria behind it."""

    flags: List[Dict[str, object]] = []
    for severity, reasons in (("stop", stop_reasons), ("partial", partial_reasons)):
        for reason in sorted(set(reasons)):
            spec = _REFUTABLE_GATE_FLAGS.get(reason)
            if spec is None:
                flags.append(
                    {
                        "reason": reason,
                        "severity": severity,
                        "refutable": False,
                        "criteria": [],
                    }
                )
                continue
            flags.append(
                {
                    "reason": reason,
                    "severity": severity,
                    "claim": spec.get("claim"),
                    "refutable": bool(spec.get("refutable", True)),
                    "refuted_when_any_fails": bool(
                        spec.get("refuted_when_any_fails", False)
                    ),
                    "criteria": copy.deepcopy(spec.get("criteria") or []),
                }
            )
    return flags


def build_diagnostic_gate(
    *,
    record_quality: Dict[str, object],
    duration_sec: float,
    n_beats: int,
    beat_groups: Dict[int, List[int]],
    limb_reversal: Optional[Dict[str, object]] = None,
    precordial_reversal: Optional[Dict[str, object]] = None,
    limb_lead_consistency: Optional[Dict[str, object]] = None,
    qrs_detector_agreement: Optional[Dict[str, object]] = None,
    amplitude_calibration: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Combine technical prerequisites into a single auditable gate.

    The gate controls diagnostic statements, not feature extraction.  This is
    why short/noisy records still return measurements and detailed reasons.
    """
    cfg = DEFAULT_DIAGNOSTIC_CONFIG
    stop_reasons: List[str] = []
    partial_reasons: List[str] = []
    if str(record_quality.get("diagnostic_gate")) == "stop":
        stop_reasons.append("record_signal_quality_stop")
    elif str(record_quality.get("diagnostic_gate")) == "partial":
        partial_reasons.append("partial_lead_quality")
    if float(duration_sec) < cfg.rhythm.minimum_diagnostic_duration_seconds:
        partial_reasons.append("duration_below_10_seconds")
    if float(duration_sec) < cfg.rhythm.minimum_rhythm_duration_seconds:
        stop_reasons.append("duration_below_rhythm_minimum")
    if int(n_beats) < cfg.rhythm.minimum_detected_beats:
        stop_reasons.append("fewer_than_3_detected_beats")
    if qrs_detector_agreement and bool(qrs_detector_agreement.get("usable")):
        detector_f1 = float(qrs_detector_agreement.get("f1_agreement") or 0.0)
        if detector_f1 < 0.50:
            stop_reasons.append("qrs_detectors_strongly_disagree")
        elif detector_f1 < 0.70:
            partial_reasons.append("qrs_detectors_partially_disagree")
    if bool((limb_reversal or {}).get("probable_extremity_reversal")):
        stop_reasons.append("probable_limb_lead_reversal")
    if limb_lead_consistency:
        if bool(limb_lead_consistency.get("severely_inconsistent", False)):
            stop_reasons.append("limb_lead_equation_severely_inconsistent")
        elif not bool(limb_lead_consistency.get("consistent", True)):
            partial_reasons.append("limb_lead_equation_residual_high")
    precordial_state = str((precordial_reversal or {}).get("state") or "")
    if precordial_state in {"possible", "confirmed"}:
        partial_reasons.append("precordial_lead_placement_suspected")
    # Rescaled amplitudes invalidate voltage-derived reasoning and nothing
    # else, so they scope the gate rather than stopping the record.
    suppressed_domains: List[str] = []
    if bool((amplitude_calibration or {}).get("per_lead_normalized")):
        partial_reasons.append("amplitudes_not_calibrated_per_lead_normalized")
        suppressed_domains.extend(("voltage_chamber_r_progression", "axis_quantitative"))

    group_sizes = sorted(
        (len(members) for members in beat_groups.values()),
        reverse=True,
    )
    dominant_fraction = (
        float(group_sizes[0]) / float(max(int(n_beats), 1))
        if group_sizes
        else 0.0
    )
    morphology_complex = bool(
        n_beats > 0
        and (
            dominant_fraction < cfg.morphology.dominant_group_fraction_min
            or len(group_sizes) > cfg.morphology.maximum_template_classes
        )
    )
    if morphology_complex:
        partial_reasons.append("morphology_too_complex_for_representative_diagnosis")

    state = "stop" if stop_reasons else ("partial" if partial_reasons else "pass")
    allowed_domains = (
        []
        if state == "stop"
        else (
            ["quality", "rhythm", "ectopy"]
            if morphology_complex
            else ["all"]
        )
    )
    return {
        "state": state,
        "stop_reasons": sorted(set(stop_reasons)),
        "partial_reasons": sorted(set(partial_reasons)),
        "allowed_domains": allowed_domains,
        "suppressed_domains": sorted(set(suppressed_domains)),
        "refutable_flags": build_refutable_gate_flags(
            stop_reasons=stop_reasons,
            partial_reasons=partial_reasons,
        ),
        "duration_sec": float(duration_sec),
        "n_beats": int(n_beats),
        "group_count": len(group_sizes),
        "dominant_group_fraction": dominant_fraction,
        "morphology_complex": morphology_complex,
        "requires_original_500_hz": False,
    }


def compute_quality(
    ecg: np.ndarray,
    fs: int,
    mains_hz: int = 50,
    *,
    adc_full_scale_mv: Optional[float] = None,
) -> Dict[str, LeadQuality]:
    qualities: Dict[str, LeadQuality] = {}
    thresholds = DEFAULT_DIAGNOSTIC_CONFIG.quality
    total_band = (0.05, min(150.0, fs / 2.0 - 1.0))
    for i, lead in enumerate(STANDARD_12_LEADS):
        sig = np.asarray(ecg[i], dtype=float)
        freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), max(256, fs * 2)))
        def power(fmin: float, fmax: float) -> float:
            return _spectrum_band_power(freqs, psd, fmin, fmax)
        total_power = power(*total_band) + 1e-12
        baseline_power = power(0.05, min(0.5, fs / 2.0 - 1.0))
        diagnostic_power = power(0.05, min(40.0, fs / 2.0 - 1.0)) + 1e-12
        baseline_wander = float(baseline_power / diagnostic_power)
        muscle_power = power(40.0, min(100.0, fs / 2.0 - 1.0)) / total_power
        powerline_power = power(mains_hz - 1.0, mains_hz + 1.0) / total_power
        clipping = _clipping_score(sig)
        flat_std = _flatline_score(sig)
        flat_fraction = _flatline_fraction(
            sig, fs, thresholds.flatline_peak_to_peak_mv
        )
        saturation_fraction = _saturation_fraction(sig, adc_full_scale_mv)
        psqi_denominator = power(5.0, min(40.0, fs / 2.0 - 1.0)) + 1e-12
        p_sqi = power(5.0, min(15.0, fs / 2.0 - 1.0)) / psqi_denominator
        k_sqi = _kurtosis_score(sig)
        b_sqi = _bsqi(sig, fs)
        # A slow rhythm can contain a physiologic one-second isoelectric
        # interval. Require both a high flat-window fraction and globally tiny
        # amplitude before declaring the lead absent.
        missing = (
            flat_fraction > thresholds.flatline_fraction_max
            and flat_std < 0.02
        )
        flags: List[str] = []
        if baseline_wander > thresholds.baseline_drift_ratio_max:
            flags.append("baseline_wander")
        if muscle_power > thresholds.emg_ratio_max:
            flags.append("muscle_noise")
        if powerline_power > thresholds.powerline_ratio_max:
            flags.append("powerline_noise")
        if saturation_fraction is not None and saturation_fraction > thresholds.saturation_fraction_max:
            flags.append("saturation")
        elif clipping > 0.05:
            flags.append("clipping")
        if missing:
            flags.append("missing_or_flat")
        if p_sqi < thresholds.psqi_min or p_sqi > thresholds.psqi_max:
            flags.append("psqi_out_of_range")
        if k_sqi < thresholds.ksqi_min:
            flags.append("low_kurtosis")
        if b_sqi < thresholds.bsqi_min:
            flags.append("low_bsqi")
        saturation_ok = (
            saturation_fraction is None
            or saturation_fraction <= thresholds.saturation_fraction_max
        )
        reliable = (
            not missing
            and saturation_ok
            and baseline_wander <= thresholds.baseline_drift_ratio_max
            and muscle_power <= thresholds.emg_ratio_max
            and b_sqi >= thresholds.bsqi_record_stop
        )

        # Wave-specific reliability (T003) — stricter thresholds per wave type
        reliable_for_p = (
            not missing
            and saturation_ok
            and baseline_wander <= thresholds.baseline_drift_ratio_max
            and muscle_power < 0.30
            and powerline_power <= thresholds.powerline_ratio_max
            and b_sqi >= thresholds.bsqi_record_stop
        )
        reliable_for_qrs = (
            not missing
            and saturation_ok
            and muscle_power <= thresholds.emg_ratio_max
            and powerline_power <= 0.20
            and b_sqi >= thresholds.bsqi_record_stop
        )
        reliable_for_t = (
            not missing
            and saturation_ok
            and baseline_wander < 0.25
            and muscle_power < 0.30
            and powerline_power < 0.12
            and b_sqi >= thresholds.bsqi_record_stop
            and flat_std        > 0.02  # must have visible T amplitude
        )
        reliable_for_qt = reliable_for_t and reliable_for_qrs
        reason_codes = list(flags)
        grade = "Q0"
        if missing:
            grade = "Q3"
        elif not reliable_for_qrs:
            grade = "Q3"
        elif not reliable_for_p or not reliable_for_qt:
            grade = "Q2"
        elif reason_codes or not reliable:
            grade = "Q1"

        qualities[lead] = LeadQuality(
            lead=lead,
            baseline_wander_score=baseline_wander,
            muscle_noise_score=float(muscle_power),
            powerline_score=float(powerline_power),
            clipping_score=float(clipping),
            flatline_score=float(flat_std),
            missing=missing,
            reliable=reliable,
            flags=flags,
            grade=grade,
            reason_codes=reason_codes,
            reliable_for_p=reliable_for_p,
            reliable_for_qrs=reliable_for_qrs,
            reliable_for_t=reliable_for_t,
            reliable_for_qt=reliable_for_qt,
            flatline_fraction=flat_fraction,
            saturation_fraction=saturation_fraction,
            p_sqi=float(p_sqi),
            k_sqi=float(k_sqi),
            b_sqi=float(b_sqi),
        )
    return qualities


def _rolling_mad_sigma(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    win = max(3, int(window))
    if win % 2 == 0:
        win += 1
    half = win // 2
    padded = np.pad(arr, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    medians = np.median(windows, axis=-1)
    sigma = 1.4826 * np.median(np.abs(windows - medians[:, None]), axis=-1)
    positive = sigma[sigma > 1e-9]
    fallback = float(np.median(positive)) if positive.size else 1e-6
    return np.maximum(sigma, fallback)


def _pacing_enhancer(sig: np.ndarray) -> np.ndarray:
    arr = np.asarray(sig, dtype=float)
    if arr.size == 0:
        return arr.copy()
    return np.diff(arr, prepend=arr[0])


def _spike_prominence_uv(sig: np.ndarray, peak: int, fs: int, width_ms: float) -> float:
    """Absolute deflection height of a narrow spike above its local flanks (microvolt).

    A genuine pacing spike is a sharp, narrow deflection that stands out from the
    signal immediately flanking it. Measuring prominence against short flanks just
    outside the spike width keeps a real spike (even one riding on a QRS) large
    while collapsing broad QRS energy and low-amplitude high-frequency wiggle.
    """
    n = len(sig)
    if fs <= 0 or n == 0 or peak < 0 or peak >= n:
        return 0.0
    half = max(1, int(round((max(width_ms, 1.0) * 0.5 + 3.0) * float(fs) / 1000.0)))
    flank = max(2, int(round(0.004 * float(fs))))
    left = sig[max(0, peak - half - flank):max(0, peak - half)]
    right = sig[min(n, peak + half + 1):min(n, peak + half + 1 + flank)]
    bases = [float(np.median(seg)) for seg in (left, right) if seg.size > 0]
    baseline = float(np.median(bases)) if bases else float(np.median(sig))
    return abs(float(sig[peak]) - baseline) * 1000.0


def _lead_pacing_candidates(
    sig: np.ndarray,
    fs: int,
    min_prominence_uv: float = PACE_SPIKE_LOW_UV,
    *,
    prepared: Optional[List[tuple]] = None,
) -> List[tuple]:
    details = prepared if prepared is not None else _prepare_lead_pacing_candidates(sig, fs)
    return [
        (int(peak), float(score), float(width_ms))
        for peak, score, width_ms, prominence_uv in details
        if float(prominence_uv) >= float(min_prominence_uv)
    ]


def _prepare_lead_pacing_candidates(sig: np.ndarray, fs: int) -> List[tuple]:
    """Compute threshold-independent pacing candidates once per lead."""
    if fs <= 0 or len(sig) == 0:
        return []
    enhanced = _pacing_enhancer(sig)
    noise_window = max(9, int(round(0.050 * fs)))
    sigma = _rolling_mad_sigma(enhanced, noise_window)
    threshold = np.maximum(4.0 * sigma, 0.001)
    mask = np.abs(enhanced) >= threshold
    candidates: List[tuple] = []
    idx = 0
    n_samp = len(sig)
    max_width_ms = 6.0
    while idx < n_samp:
        if not bool(mask[idx]):
            idx += 1
            continue
        start = idx
        while idx + 1 < n_samp and bool(mask[idx + 1]):
            idx += 1
        end = idx
        seg = np.abs(enhanced[start:end + 1])
        enhanced_peak = start + int(np.argmax(seg))
        peak_radius = max(1, int(round(0.002 * fs)))
        raw_lo = max(0, enhanced_peak - peak_radius)
        raw_hi = min(n_samp, enhanced_peak + peak_radius + 1)
        peak = raw_lo + int(np.argmax(np.abs(sig[raw_lo:raw_hi])))
        over_threshold_width_ms = float(end - start + 1) * 1000.0 / float(fs)
        raw_width_ms = _cluster_half_height_width_ms(sig, peak, fs)
        width_ms = float(max(over_threshold_width_ms, raw_width_ms))
        if over_threshold_width_ms <= max_width_ms and raw_width_ms <= max_width_ms:
            prominence_uv = _spike_prominence_uv(sig, peak, fs, width_ms)
            score = float(
                np.max(seg / np.maximum(sigma[start:end + 1], 1e-9))
            )
            candidates.append((int(peak), score, width_ms, prominence_uv))
        idx += 1
    return candidates


def _accept_pacing_cluster(lead_count: int, scores: List[float], min_lead_votes: int) -> bool:
    if lead_count >= min_lead_votes:
        return True
    weak_vote_floor = max(2, min_lead_votes - 1)
    weighted_score = float(np.sum(np.clip(scores, 0.0, 12.0))) if scores else 0.0
    return lead_count >= weak_vote_floor and weighted_score >= 12.0


def _legacy_highpass_pacing_detection(
    ecg: np.ndarray,
    fs: int,
    min_lead_votes: int,
    min_prominence_uv: float = PACE_SPIKE_LOW_UV,
    *,
    prepared: Optional[List[tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, object]:
    n_leads, _ = ecg.shape
    win_samples = max(1, int(0.050 * fs))
    amplitude_floor_mv = float(min_prominence_uv) / 1000.0
    spike_events: List[tuple] = []
    for lead_idx in range(min(n_leads, 12)):
        if prepared is not None and lead_idx < len(prepared):
            hp, local_rms = prepared[lead_idx]
        else:
            sig = ecg[lead_idx].astype(float)
            hp = highpass_filter(
                sig[None, :],
                fs,
                cutoff_hz=_pacing_highpass_cutoff_hz(fs),
            )[0]
            hp_sq = hp ** 2
            kernel = np.ones(win_samples) / win_samples
            _conv = np.convolve(hp_sq, kernel, mode="full")
            rms_inc = np.sqrt(np.clip(_conv[: len(hp)], 0.0, None))
            local_rms = np.concatenate([[rms_inc[0]], rms_inc[:-1]])
        # 12SL absolute-amplitude gate on the high-pass envelope (replaces the old
        # 10 microvolt floor, which admitted sub-threshold QRS-edge artefacts).
        spike_mask = np.abs(hp) >= np.maximum(5.0 * local_rms, amplitude_floor_mv)
        for s in np.where(spike_mask)[0].astype(int):
            spike_events.append((s, lead_idx))

    if not spike_events:
        return {"spike_times": [], "paced": False, "state": "off", "lead_vote_count": 0}

    spike_events.sort()
    merge_win = max(1, int(0.005 * fs))
    merged: List[int] = []
    lead_vote_count = 0
    cluster_start = spike_events[0][0]
    cluster_last = spike_events[0][0]
    cluster_leads: set = {spike_events[0][1]}
    for s, lead_idx in spike_events[1:]:
        if s - cluster_last <= merge_win:
            cluster_last = s
            cluster_leads.add(lead_idx)
        else:
            if len(cluster_leads) >= min_lead_votes:
                merged.append((cluster_start + cluster_last) // 2)
                lead_vote_count = max(lead_vote_count, len(cluster_leads))
            cluster_start = s
            cluster_last = s
            cluster_leads = {lead_idx}
    if len(cluster_leads) >= min_lead_votes:
        merged.append((cluster_start + cluster_last) // 2)
        lead_vote_count = max(lead_vote_count, len(cluster_leads))

    if len(merged) >= 3:
        ivs = np.diff(merged).astype(float)
        median_iv = float(np.median(ivs))
        if median_iv > 0:
            short_thr = median_iv / 2.0
            to_remove = {
                i + 1
                for i, iv in enumerate(ivs)
                if float(iv) < short_thr
            }
            merged = [m for i, m in enumerate(merged) if i not in to_remove]

    paced = False
    if len(merged) >= 3:
        intervals = np.diff(merged).astype(float)
        median_iv = float(np.median(intervals))
        if median_iv > 0:
            mad = float(np.median(np.abs(intervals - median_iv)))
            paced = bool((1.4826 * mad / median_iv) < 0.15)
    return {
        "spike_times": merged,
        "paced": paced,
        "state": "on" if paced else "unknown",
        "lead_vote_count": int(lead_vote_count),
        "candidate_summary": {
            "detector": "legacy_highpass_rms",
        },
    }


def _strong_legacy_pacing_result(result: Dict[str, object]) -> bool:
    return (
        bool(result.get("paced", False))
        and int(result.get("lead_vote_count", 0) or 0) >= 10
        and len(result.get("spike_times", []) or []) >= 3
    )


def _pacing_highpass_cutoff_hz(fs: int, requested_hz: float = 100.0) -> float:
    """Keep the pacing high-pass safely below Nyquist at low sample rates."""
    if fs <= 0:
        raise ValueError("fs must be greater than zero")
    return float(min(requested_hz, 0.90 * (float(fs) / 2.0)))


def prepare_pacing_detection_cache(ecg: np.ndarray, fs: int) -> Dict[str, Any]:
    """Cache threshold-independent filters used by normal and rescue passes."""
    values = np.asarray(ecg, dtype=float)
    n_leads = min(values.shape[0], 12)
    win_samples = max(1, int(0.050 * fs))
    legacy: List[tuple[np.ndarray, np.ndarray]] = []
    candidates: List[List[tuple]] = []
    for lead_idx in range(n_leads):
        signal = values[lead_idx]
        legacy_hp = highpass_filter(
            signal[None, :],
            fs,
            cutoff_hz=_pacing_highpass_cutoff_hz(fs),
        )[0]
        kernel = np.ones(win_samples) / win_samples
        convolution = np.convolve(legacy_hp * legacy_hp, kernel, mode="full")
        rms_increment = np.sqrt(
            np.clip(convolution[: legacy_hp.size], 0.0, None)
        )
        local_rms = np.concatenate(
            ([rms_increment[0]], rms_increment[:-1])
        )
        legacy.append((legacy_hp, local_rms))
        pacing_signal = highpass_filter(
            signal[None, :], fs, cutoff_hz=0.5
        )[0]
        candidates.append(_prepare_lead_pacing_candidates(pacing_signal, fs))
    return {
        "shape": tuple(int(value) for value in values.shape),
        "fs": int(fs),
        "legacy": legacy,
        "candidates": candidates,
    }


def detect_pacing_spikes(
    ecg: np.ndarray,
    fs: int,
    min_lead_votes: int = 4,
    min_prominence_uv: float = PACE_SPIKE_LOW_UV,
    *,
    detection_cache: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    T024 — Detect pacing spikes in 12-lead ECG.
    Returns dict with keys: spike_times (list of sample indices), paced (bool),
    and state ("off" | "unknown" | "on").

    A spike cluster is only accepted if at least min_lead_votes distinct leads
    independently detected it (multilead consensus per DXL spec), and each
    per-lead candidate must clear the 12SL absolute-amplitude gate
    (min_prominence_uv, default 250 microvolt).
    """
    n_leads, n_samp = ecg.shape
    # Collect (sample_index, lead_idx) so we can count per-lead votes
    spike_events: List[tuple] = []
    candidate_events: List[Dict[str, object]] = []
    cache = detection_cache
    if (
        not isinstance(cache, dict)
        or cache.get("shape") != tuple(int(value) for value in ecg.shape)
        or cache.get("fs") != int(fs)
    ):
        cache = prepare_pacing_detection_cache(ecg, fs)
    legacy_result = _legacy_highpass_pacing_detection(
        ecg,
        fs,
        min_lead_votes,
        min_prominence_uv=min_prominence_uv,
        prepared=cache.get("legacy"),
    )

    for lead_idx in range(min(n_leads, 12)):
        prepared_candidates = cache.get("candidates", [])
        lead_candidates = (
            prepared_candidates[lead_idx]
            if lead_idx < len(prepared_candidates)
            else None
        )
        for s, score, width_ms in _lead_pacing_candidates(
            ecg[lead_idx],
            fs,
            min_prominence_uv,
            prepared=lead_candidates,
        ):
            spike_events.append((s, lead_idx))
            candidate_events.append({
                "sample": int(s),
                "lead_index": int(lead_idx),
                "score": float(score),
                "width_ms": float(width_ms),
            })

    if not spike_events:
        if _strong_legacy_pacing_result(legacy_result):
            return dict(legacy_result)
        return {"spike_times": [], "paced": False, "state": "off", "lead_vote_count": 0}

    spike_events.sort()
    merge_win = max(1, int(0.005 * fs))

    # Merge within ~4-5ms; track which leads voted for each cluster.
    merged: List[int] = []
    merged_events: List[Dict[str, object]] = []
    lead_vote_count = 0
    cluster_start  = spike_events[0][0]
    cluster_last   = spike_events[0][0]
    cluster_leads: set = {spike_events[0][1]}
    cluster_scores: List[float] = []
    cluster_widths: List[float] = []

    event_by_key = {
        (int(ev["sample"]), int(ev["lead_index"])): ev
        for ev in candidate_events
    }
    first_ev = event_by_key.get((int(spike_events[0][0]), int(spike_events[0][1])), {})
    if first_ev:
        cluster_scores.append(float(first_ev.get("score", 0.0) or 0.0))
        cluster_widths.append(float(first_ev.get("width_ms", 0.0) or 0.0))

    for s, lead_idx in spike_events[1:]:
        if s - cluster_last <= merge_win:
            cluster_last = s
            cluster_leads.add(lead_idx)
            ev = event_by_key.get((int(s), int(lead_idx)), {})
            if ev:
                cluster_scores.append(float(ev.get("score", 0.0) or 0.0))
                cluster_widths.append(float(ev.get("width_ms", 0.0) or 0.0))
        else:
            if _accept_pacing_cluster(len(cluster_leads), cluster_scores, min_lead_votes):
                sample = (cluster_start + cluster_last) // 2
                merged.append(sample)
                lead_vote_count = max(lead_vote_count, len(cluster_leads))
                merged_events.append({
                    "sample": int(sample),
                    "lead_votes": int(len(cluster_leads)),
                    "weighted_score": float(np.sum(np.clip(cluster_scores, 0.0, 12.0))),
                    "median_width_ms": float(np.median(cluster_widths)) if cluster_widths else None,
                })
            cluster_start = s
            cluster_last  = s
            cluster_leads = {lead_idx}
            ev = event_by_key.get((int(s), int(lead_idx)), {})
            cluster_scores = [float(ev.get("score", 0.0) or 0.0)] if ev else []
            cluster_widths = [float(ev.get("width_ms", 0.0) or 0.0)] if ev else []
    if _accept_pacing_cluster(len(cluster_leads), cluster_scores, min_lead_votes):
        sample = (cluster_start + cluster_last) // 2
        merged.append(sample)
        lead_vote_count = max(lead_vote_count, len(cluster_leads))
        merged_events.append({
            "sample": int(sample),
            "lead_votes": int(len(cluster_leads)),
            "weighted_score": float(np.sum(np.clip(cluster_scores, 0.0, 12.0))),
            "median_width_ms": float(np.median(cluster_widths)) if cluster_widths else None,
        })

    if not merged:
        if _strong_legacy_pacing_result(legacy_result):
            return dict(legacy_result)
        return {"spike_times": [], "paced": False, "state": "off", "lead_vote_count": 0}

    # Remove outlier spikes: any interval shorter than half the median marks a
    # false positive (QRS / T-wave / boundary artefact).  The pacing stimulus
    # arrives first, so we always remove the LATER spike of a close pair.
    if len(merged) >= 3:
        ivs = np.diff(merged).astype(float)
        median_iv = float(np.median(ivs))
        if median_iv > 0:
            short_thr = median_iv / 2.0
            to_remove: set = set()
            for i, iv in enumerate(ivs):
                if iv < short_thr:
                    to_remove.add(i + 1)   # drop the later of the two close spikes
            merged = [m for i, m in enumerate(merged) if i not in to_remove]

    if not merged:
        return {"spike_times": [], "paced": False, "state": "off", "lead_vote_count": 0}

    # paced = True if >= 3 spikes with regular intervals.
    # Use robust (MAD-based) CV so that 1–2 boundary artefacts do not inflate the
    # variance and mask a genuinely paced rhythm.
    paced = False
    if len(merged) >= 3:
        intervals = np.diff(merged).astype(float)
        median_iv = float(np.median(intervals))
        if median_iv > 0:
            mad = float(np.median(np.abs(intervals - median_iv)))
            robust_cv = 1.4826 * mad / median_iv
            paced = robust_cv < 0.15

    mad_result = {
        "spike_times": merged,
        "paced": paced,
        "state": "on" if paced else "unknown",
        "lead_vote_count": int(lead_vote_count),
        "events": merged_events,
        "candidate_summary": {
            "detector": "mad_diff",
            "candidate_count": int(len(candidate_events)),
        },
    }
    if _strong_legacy_pacing_result(legacy_result):
        return dict(legacy_result)
    return mad_result


def _cluster_half_height_width_ms(hp: np.ndarray, spike: int, fs: int) -> float:
    n_samp = len(hp)
    radius = max(1, int(round(0.006 * fs)))
    lo = max(0, int(spike) - radius)
    hi = min(n_samp, int(spike) + radius + 1)
    seg = np.abs(hp[lo:hi])
    if seg.size == 0:
        return float("inf")
    peak_idx = int(np.argmax(seg))
    peak = float(seg[peak_idx])
    if peak <= 0.0:
        return float("inf")
    threshold = 0.50 * peak
    left = peak_idx
    while left > 0 and float(seg[left - 1]) >= threshold:
        left -= 1
    right = peak_idx
    while right + 1 < seg.size and float(seg[right + 1]) >= threshold:
        right += 1
    return float(right - left + 1) * 1000.0 / float(fs)


def validate_pacing_spikes_against_qrs(
    ecg: np.ndarray,
    fs: int,
    spike_times: List[int],
    r_locs: np.ndarray,
    pacing_result: Dict[str, Any],
    *,
    detection_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reject multilead high-frequency QRS edges misclassified as pacing spikes."""
    result: Dict[str, Any] = dict(pacing_result or {})
    spikes = [int(s) for s in spike_times]
    r_arr = np.asarray(r_locs, dtype=int)
    result["raw_spike_times"] = list(spikes)
    if not spikes or len(r_arr) == 0 or fs <= 0:
        result.setdefault("spike_times", spikes)
        return result

    n_leads, n_samp = ecg.shape
    cached_legacy = (
        detection_cache.get("legacy")
        if isinstance(detection_cache, dict)
        and detection_cache.get("shape") == tuple(int(value) for value in ecg.shape)
        and detection_cache.get("fs") == int(fs)
        else None
    )
    if cached_legacy is not None and len(cached_legacy) >= min(n_leads, 12):
        hp = np.stack(
            [cached_legacy[index][0] for index in range(min(n_leads, 12))],
            axis=0,
        )
    else:
        hp = highpass_filter(
            np.asarray(ecg[: min(n_leads, 12)], dtype=float),
            fs,
            cutoff_hz=_pacing_highpass_cutoff_hz(fs),
        )
    spike_radius = max(1, int(round(0.002 * fs)))
    qrs_lo = max(1, int(round(0.008 * fs)))
    qrs_hi = max(qrs_lo + 1, int(round(0.030 * fs)))
    exclusion = max(1, int(round(0.004 * fs)))

    kept: List[int] = []
    rejected: List[int] = []
    spike_to_qrs_offsets: List[int] = []
    all_qrs_offsets: List[int] = []

    for spike in spikes:
        nearest_r = int(r_arr[int(np.argmin(np.abs(r_arr - spike)))])
        offset = int(spike) - nearest_r
        all_qrs_offsets.append(offset)
        near_qrs = int(round(-0.080 * fs)) <= offset <= int(round(0.020 * fs))
        if near_qrs:
            spike_to_qrs_offsets.append(offset)

        spike_peaks: List[float] = []
        qrs_peaks: List[float] = []
        widths: List[float] = []
        for lead_idx in range(hp.shape[0]):
            sig = hp[lead_idx]
            slo = max(0, spike - spike_radius)
            shi = min(n_samp, spike + spike_radius + 1)
            spike_peak = float(np.max(np.abs(sig[slo:shi]))) if shi > slo else 0.0
            qlo = max(0, nearest_r - qrs_lo)
            qhi = min(n_samp, nearest_r + qrs_hi)
            qrs_seg = np.abs(sig[qlo:qhi]).copy()
            ex_lo = max(qlo, spike - exclusion) - qlo
            ex_hi = min(qhi, spike + exclusion + 1) - qlo
            if 0 <= ex_lo < ex_hi <= qrs_seg.size:
                qrs_seg[ex_lo:ex_hi] = 0.0
            qrs_peak = float(np.max(qrs_seg)) if qrs_seg.size else 0.0
            if spike_peak > 0.0:
                spike_peaks.append(spike_peak)
                qrs_peaks.append(qrs_peak)
                widths.append(_cluster_half_height_width_ms(sig, spike, fs))

        spike_peak_med = float(np.median(spike_peaks)) if spike_peaks else 0.0
        qrs_peak_med = float(np.median(qrs_peaks)) if qrs_peaks else 0.0
        width_med = float(np.median(widths)) if widths else float("inf")
        if spike_peak_med <= 0.0:
            kept.append(spike)
            continue
        isolated_narrow = width_med <= 4.0
        dominates_qrs_hf = spike_peak_med >= max(0.03, 2.5 * qrs_peak_med)
        if (not near_qrs) or (isolated_narrow and dominates_qrs_hf):
            kept.append(spike)
        else:
            rejected.append(spike)

    beat_sync_fraction = (
        float(len(spike_to_qrs_offsets)) / float(max(1, len(r_arr)))
    )
    fixed_offset = False
    if len(spike_to_qrs_offsets) >= 3:
        offsets = np.asarray(spike_to_qrs_offsets, dtype=float)
        fixed_offset = float(np.std(offsets)) * 1000.0 / float(fs) <= 8.0

    pre_qrs_offsets_ms = [
        float(offset) * 1000.0 / float(fs)
        for offset in all_qrs_offsets
        if -260.0 <= float(offset) * 1000.0 / float(fs) <= -100.0
    ]
    dual_spike_context = False
    if len(pre_qrs_offsets_ms) >= 2:
        sorted_pre_offsets = sorted(pre_qrs_offsets_ms)
        dual_spike_context = any(
            hi - lo <= 25.0
            for lo, hi in zip(sorted_pre_offsets, sorted_pre_offsets[1:])
        )

    if rejected and dual_spike_context:
        result["spike_times"] = spikes
        result["rejected_spike_times"] = rejected
        result["rejection_reason"] = "mixed_pacing_and_qrs_edge_candidates"
        result["pacing_event_class"] = "mixed_pacing_and_qrs_edge"
        mixed_offsets_ms: List[float] = []
        for spike in spikes:
            nearest_r = int(r_arr[int(np.argmin(np.abs(r_arr - int(spike))))])
            mixed_offsets_ms.append(float(int(spike) - nearest_r) * 1000.0 / float(fs))
        near_capture_count = sum(1 for offset in mixed_offsets_ms if abs(offset) <= 50.0)
        strong_legacy_context = (
            result.get("candidate_summary", {}).get("detector") == "legacy_highpass_rms"
            and int(result.get("lead_vote_count", 0) or 0) >= 10
        )
        if near_capture_count >= 2 or strong_legacy_context:
            result["paced"] = True
            result["state"] = "on"
        else:
            result["paced"] = False
            result["state"] = "unknown"
        return result

    if not kept or (
        rejected
        and beat_sync_fraction >= 0.80
        and fixed_offset
    ):
        result["spike_times"] = []
        result["paced"] = False
        result["state"] = "off"
        result["lead_vote_count"] = 0
        result["rejection_reason"] = "qrs_edge_artifact"
        result["rejected_spike_times"] = rejected or spikes
        result["pacing_event_class"] = "qrs_edge_artifact"
        return result

    result["spike_times"] = kept
    result["rejected_spike_times"] = rejected
    result["rejection_reason"] = None
    kept_offsets_ms: List[float] = []
    for spike in kept:
        nearest_r = int(r_arr[int(np.argmin(np.abs(r_arr - int(spike))))])
        kept_offsets_ms.append(float(int(spike) - nearest_r) * 1000.0 / float(fs))
    atrial_or_nonventricular = (
        len(kept_offsets_ms) >= 2
        and int(result.get("lead_vote_count", 0) or 0) < 10
        and all(-260.0 <= offset <= -100.0 for offset in kept_offsets_ms)
    )
    if atrial_or_nonventricular:
        result["paced"] = False
        result["state"] = "unknown"
        result["rejection_reason"] = "atrial_or_nonventricular_pacing"
        result["pacing_event_class"] = "atrial_or_nonventricular"
        return result
    if len(kept) >= 3:
        intervals = np.diff(kept).astype(float)
        median_iv = float(np.median(intervals))
        if median_iv > 0:
            mad = float(np.median(np.abs(intervals - median_iv)))
            result["paced"] = bool((1.4826 * mad / median_iv) < 0.15)
    else:
        result["paced"] = False
    result["state"] = "on" if bool(result.get("paced")) else "unknown"
    stable_capture_offsets = [
        offset
        for offset in kept_offsets_ms
        if abs(offset) <= 50.0
    ]
    if (
        len(stable_capture_offsets) >= 2
        and len(kept_offsets_ms) >= 3
    ):
        result["paced"] = True
        result["state"] = "on"
    if bool(result.get("paced")) and stable_capture_offsets:
        result["pacing_event_class"] = "ventricular_capture"
    elif bool(result.get("paced")):
        result["pacing_event_class"] = "uncertain"
    else:
        result["pacing_event_class"] = "uncertain"
    return result


def remove_pacing_spikes(
    ecg: np.ndarray,
    spike_times: List[int],
    fs: int,
    half_width_ms: float = 4.0,
) -> np.ndarray:
    """Linearly interpolate narrow pacer-spike windows before waveform analysis."""
    cleaned = np.asarray(ecg, dtype=float).copy()
    if cleaned.ndim != 2 or not spike_times:
        return cleaned

    n_samp = cleaned.shape[1]
    half_width = max(1, int(round(half_width_ms * fs / 1000.0)))
    for spike in sorted({int(s) for s in spike_times}):
        lo = max(0, spike - half_width)
        hi = min(n_samp - 1, spike + half_width)
        left = lo - 1
        right = hi + 1
        if left >= 0 and right < n_samp:
            interp = np.linspace(cleaned[:, left], cleaned[:, right], hi - lo + 1, axis=1)
            cleaned[:, lo : hi + 1] = interp
        elif left >= 0:
            cleaned[:, lo : hi + 1] = cleaned[:, left][:, None]
        elif right < n_samp:
            cleaned[:, lo : hi + 1] = cleaned[:, right][:, None]
        else:
            cleaned[:, lo : hi + 1] = 0.0
    return cleaned


def _finite_correlation(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    """Pearson correlation that abstains for constant or malformed signals."""
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return None
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator <= 1e-12:
        return None
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def detect_limb_lead_reversal(ecg: np.ndarray) -> Dict[str, object]:
    """Detect electrode swaps from lead-specific polarity/correlation patterns.

    A high Einthoven residual is deliberately *not* evidence of a swap. It says
    the limb leads are not mutually derivable, which any per-lead gain change,
    re-derived lead or filter mismatch produces just as readily as a swapped
    cable, and a swap has its own signature in the correlation tests below.
    Folding the residual in made every calibration defect present as an
    electrode swap: LUDB rescales each lead to 1.000 mV peak-to-peak, so its
    records violate II = I + III in physical units while satisfying it exactly
    in raw counts, and the whole database read as swapped limb leads.

    The residual is still reported here for audit, and it is still acted on -
    by `limb_lead_consistency` from input validation, which owns it and has a
    proportionate severity ladder.
    """
    # Heuristic, not a medical-device-grade detector.
    I, II, III, aVR, aVL, aVF = ecg[:6]
    swaps = {
        "probable_ra_la": False,
        "probable_ra_ll": False,
        "probable_la_ll": False,
        "probable_ra_rl": False,
        "probable_la_rl": False,
        "probable_ll_rl": False,
    }
    out: Dict[str, object] = dict(swaps)
    einthoven_err = np.sqrt(np.mean((II - I - III) ** 2))
    ref_scale = np.std(II) + 1e-6
    corr_i_ii = _finite_correlation(I, -II)
    corr_avr_avl = _finite_correlation(aVR, aVL)
    corr_ii_i = _finite_correlation(II, -I)
    corr_avr_avf = _finite_correlation(aVR, aVF)
    corr_iii_i = _finite_correlation(III, -I)
    corr_avl_avf = _finite_correlation(aVL, aVF)
    if corr_i_ii is not None and corr_avr_avl is not None and corr_i_ii > 0.75 and corr_avr_avl > 0.55:
        out["probable_ra_la"] = True
    if corr_ii_i is not None and corr_avr_avf is not None and corr_ii_i > 0.75 and corr_avr_avf > 0.55:
        out["probable_ra_ll"] = True
    if corr_iii_i is not None and corr_avl_avf is not None and corr_iii_i > 0.65 and corr_avl_avf > 0.55:
        out["probable_la_ll"] = True
    limb_ranges = {
        "I": float(np.ptp(I)),
        "II": float(np.ptp(II)),
        "III": float(np.ptp(III)),
    }
    visible = max(limb_ranges.values()) > 0.15
    if visible and limb_ranges["II"] < 0.08 and limb_ranges["III"] < 0.08:
        out["probable_ra_rl"] = True
    if visible and limb_ranges["I"] < 0.08 and limb_ranges["III"] < 0.08:
        out["probable_la_rl"] = True
    if visible and limb_ranges["I"] < 0.08 and limb_ranges["II"] < 0.08:
        out["probable_ll_rl"] = True
    out["probable_extremity_reversal"] = any(
        bool(out[name]) for name in swaps
    )
    out["einthoven_residual_ratio"] = float(einthoven_err / ref_scale)
    return out


_POOR_PROGRESSION_QRS_MS_LIMIT = 120.0
_PATHOLOGICAL_Q_AMP_MV = -0.20
_PATHOLOGICAL_Q_LEAD_COUNT = 2
_SOKOLOW_LYON_MV = 3.5
_ADJACENT_CORRELATION_LOW = 0.50
_ADJACENT_CORRELATION_MARGIN = 0.20


def _resample_and_normalize(segment: np.ndarray, target_len: int = 100) -> Optional[np.ndarray]:
    if segment.size < 3:
        return None
    baseline = float(np.median(segment[: max(1, segment.size // 8)]))
    centered = segment.astype(float) - baseline
    resampled = np.interp(
        np.linspace(0, centered.size - 1, target_len),
        np.arange(centered.size),
        centered,
    )
    scale = float(np.std(resampled))
    if scale <= 1e-9:
        return None
    return resampled / scale


def compute_adjacent_precordial_correlations(
    representative_beat_ecg: np.ndarray,
    lead_qrs_bounds: Dict[str, "tuple[Optional[int], Optional[int]]"],
    lead_to_row: Dict[str, int],
) -> Dict[str, float]:
    """Pearson correlation of QRS-window waveform shape between each pair of
    adjacent precordial leads, on the single representative-beat waveform.

    Physically adjacent electrodes are close together and produce smoothly
    varying, highly correlated QRS morphology (commonly >0.85). A swapped
    pair breaks that local continuity. This is independent of R-wave
    amplitude, so it can corroborate (or fail to corroborate) an
    amplitude-progression-based reversal candidate rather than replace it.
    """
    leads = [f"V{i}" for i in range(1, 7)]
    snippets: Dict[str, np.ndarray] = {}
    for lead in leads:
        row = lead_to_row.get(lead)
        bounds = lead_qrs_bounds.get(lead)
        if row is None or bounds is None:
            continue
        onset, offset = bounds
        if (
            onset is None
            or offset is None
            or offset <= onset
            or row >= representative_beat_ecg.shape[0]
            or offset >= representative_beat_ecg.shape[1]
            or onset < 0
        ):
            continue
        snippet = _resample_and_normalize(representative_beat_ecg[row, onset : offset + 1])
        if snippet is not None:
            snippets[lead] = snippet

    correlations: Dict[str, float] = {}
    for lead_a, lead_b in zip(leads, leads[1:]):
        if lead_a in snippets and lead_b in snippets:
            corr = float(np.corrcoef(snippets[lead_a], snippets[lead_b])[0, 1])
            if np.isfinite(corr):
                correlations[f"{lead_a}_{lead_b}"] = corr
    return correlations
# A decrease only counts as a progression "violation" once it is large
# relative to the R-wave peak (or an absolute floor for low-voltage traces);
# this keeps normal beat-to-beat/measurement noise (a few hundredths of a
# mV) from being treated the same as a genuine lead-swap-sized discontinuity.
_VIOLATION_REL_FRACTION_OF_PEAK = 0.15
_VIOLATION_ABS_FLOOR_MV = 0.05


def _ascending_violations(arr: np.ndarray) -> tuple[int, int]:
    """Return (violation_count, peak_index).

    Only the segment from V1 up to the R-wave peak is expected to be
    non-decreasing. A normal heart's R wave peaks at V4/V5 and is commonly
    *lower* at V6 than at V5; penalizing that decline (as a strict V1->V6
    monotonic check would) systematically flags normal R-wave progression as
    reversed.
    """
    peak_idx = int(np.argmax(arr))
    if peak_idx == 0:
        return 0, peak_idx
    peak_amp = float(arr[peak_idx])
    violation_floor = max(_VIOLATION_ABS_FLOOR_MV, _VIOLATION_REL_FRACTION_OF_PEAK * peak_amp)
    ascending = arr[: peak_idx + 1]
    violations = int(np.sum(np.diff(ascending) < -violation_floor))
    return violations, peak_idx


def detect_precordial_reversal(
    lead_metrics: Dict[str, Dict[str, Optional[float]]],
    adjacent_correlations: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Screen for possible precordial lead swaps from R-wave progression.

    This remains a screening heuristic (still no cross-validation against
    limb-lead reversal, and no modeling of non-adjacent/systematic placement
    errors), so it can only ever produce a "possible" screening flag, never
    a confirmed wiring-error diagnosis. `lead_metrics` maps "V1".."V6" to a
    dict with (at least) "r_amp_mv", "s_amp_mv", "q_amp_mv", "qrs_ms".

    `adjacent_correlations`, if given, maps "V{i}_V{i+1}" to the QRS-window
    waveform-shape correlation between that adjacent pair (see
    `compute_adjacent_precordial_correlations`). Amplitude progression alone
    is noisy (see module docs); when an amplitude-based candidate swap is
    adjacent and its waveform-continuity correlation is anomalously low
    relative to the other adjacent pairs, that corroborates the candidate and
    raises confidence from "low" to "moderate". If continuity evidence is
    available but does *not* support the candidate, confidence stays "low"
    and the disagreement is recorded so a consumer can flag it for manual
    review rather than trusting amplitude alone.
    """
    leads = [f"V{i}" for i in range(1, 7)]
    r_arr = np.asarray(
        [lead_metrics.get(lead, {}).get("r_amp_mv", np.nan) for lead in leads],
        dtype=float,
    )
    r_arr = np.where(np.isfinite(r_arr), r_arr, np.nan)
    if np.any(np.isnan(r_arr)):
        return {
            "suspected": False,
            "state": "not_suspected",
            "confidence": "unavailable",
            "implicated_leads": [],
            "progression_score": None,
            "peak_lead": None,
            "best_adjacent_swap": None,
            "best_swap_score": None,
        }
    if float(np.nanmax(np.abs(r_arr))) <= 0.50:
        return {
            "suspected": False,
            "state": "not_suspected",
            "confidence": "low",
            "implicated_leads": [],
            "progression_score": None,
            "peak_lead": leads[int(np.argmax(r_arr))],
            "best_adjacent_swap": None,
            "best_swap_score": None,
            "reason": "precordial_low_voltage",
        }

    # Known, common causes of poor R-wave progression that are not lead
    # reversal. Flagging these as "possible reversal" would be a frequent
    # false positive: old anterior MI, LVH, wide-QRS conduction disease and
    # similar all disrupt V1-V6 R-wave progression on their own.
    qrs_values = [lead_metrics.get(lead, {}).get("qrs_ms") for lead in leads]
    wide_qrs = any(v is not None and v >= _POOR_PROGRESSION_QRS_MS_LIMIT for v in qrs_values)
    q_values = [lead_metrics.get(lead, {}).get("q_amp_mv") for lead in ("V1", "V2", "V3", "V4")]
    pathological_q_count = sum(
        1 for v in q_values if v is not None and v <= _PATHOLOGICAL_Q_AMP_MV
    )
    s_v1 = lead_metrics.get("V1", {}).get("s_amp_mv")
    lateral_r = [
        v for v in (lead_metrics.get("V5", {}).get("r_amp_mv"), lead_metrics.get("V6", {}).get("r_amp_mv"))
        if v is not None
    ]
    sokolow_lyon_positive = bool(
        s_v1 is not None and lateral_r and abs(s_v1) + max(lateral_r) >= _SOKOLOW_LYON_MV
    )
    excluded_causes = []
    if wide_qrs:
        excluded_causes.append("wide_qrs")
    if pathological_q_count >= _PATHOLOGICAL_Q_LEAD_COUNT:
        excluded_causes.append("pathological_q_pattern")
    if sokolow_lyon_positive:
        excluded_causes.append("lvh_voltage_pattern")
    if excluded_causes:
        return {
            "suspected": False,
            "state": "not_suspected",
            "confidence": "unavailable",
            "implicated_leads": [],
            "progression_score": None,
            "peak_lead": leads[int(np.argmax(r_arr))],
            "best_adjacent_swap": None,
            "best_swap_score": None,
            "reason": "excluded_known_poor_progression_cause",
            "excluded_causes": excluded_causes,
        }

    violations, peak_idx = _ascending_violations(r_arr)
    progression_score = 1.0 - (violations / peak_idx) if peak_idx > 0 else 1.0
    if violations < 1:
        return {
            "suspected": False,
            "state": "not_suspected",
            "confidence": "moderate",
            "implicated_leads": [],
            "progression_score": progression_score,
            "peak_lead": leads[peak_idx],
            "best_adjacent_swap": None,
            "best_swap_score": None,
        }

    # Search every lead pair (not just adjacent ones) for a swap that fully
    # resolves the ascending-segment violations, preferring adjacent/shorter
    # spans since those are the far more common real-world wiring error.
    best_candidate: Optional[tuple[int, int, str, str]] = None
    for i in range(len(leads)):
        for j in range(i + 1, len(leads)):
            trial = r_arr.copy()
            trial[i], trial[j] = trial[j], trial[i]
            trial_violations, _ = _ascending_violations(trial)
            if trial_violations == 0:
                candidate = (0 if j == i + 1 else 1, j - i, leads[i], leads[j])
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

    if best_candidate is None:
        return {
            "suspected": False,
            "state": "not_suspected",
            "confidence": "low",
            "implicated_leads": [],
            "progression_score": progression_score,
            "peak_lead": leads[peak_idx],
            "best_adjacent_swap": None,
            "best_swap_score": None,
            "reason": "no_single_swap_resolves_progression",
        }

    adjacency_bonus, _, lead_a, lead_b = best_candidate
    is_adjacent = adjacency_bonus == 0
    confidence = "low"
    continuity_corroboration = None
    if is_adjacent and adjacent_correlations:
        pair_key = f"{lead_a}_{lead_b}"
        pair_corr = adjacent_correlations.get(pair_key)
        other_corrs = [v for k, v in adjacent_correlations.items() if k != pair_key]
        if pair_corr is not None and other_corrs:
            reference = min(np.median(other_corrs), 1.0)
            if pair_corr < _ADJACENT_CORRELATION_LOW or pair_corr < reference - _ADJACENT_CORRELATION_MARGIN:
                confidence = "moderate"
                continuity_corroboration = True
            else:
                continuity_corroboration = False
    return {
        "suspected": True,
        # A peak-aware R-progression heuristic is advisory screening
        # evidence only. It cannot confirm a wiring error without
        # cross-validation against limb-lead reversal. Adjacent-lead
        # waveform continuity (when available) can corroborate or
        # contradict the amplitude-based candidate; see
        # `continuity_corroboration`.
        "state": "possible",
        "confidence": confidence,
        "implicated_leads": [lead_a, lead_b],
        "progression_score": progression_score,
        "peak_lead": leads[peak_idx],
        "best_adjacent_swap": (lead_a, lead_b) if is_adjacent else None,
        "best_swap_score": 1.0,
        "swap_is_adjacent": is_adjacent,
        "continuity_corroboration": continuity_corroboration,
        "adjacent_correlations": dict(adjacent_correlations) if adjacent_correlations else {},
    }

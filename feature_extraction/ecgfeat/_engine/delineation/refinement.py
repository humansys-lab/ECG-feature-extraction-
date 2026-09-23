"""Conservative optional boundary candidates with auditable replacements."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ...models import STANDARD_12_LEADS, WaveBounds
from ..foundation.numeric import trapezoid
from ..atrial.p_morphology import measure_p_components


def sustained_qrs_offset(signal, onset, offset, peak, fs):
    """A multi-scale late-tail candidate; preserve an uncorroborated endpoint."""
    if onset is None or offset is None or (offset - onset) * 1000 / fs < 95:
        return offset
    x = np.asarray(signal, dtype=float)
    lo, hi = max(0, onset - int(.06 * fs)), max(0, onset - int(.012 * fs))
    if hi - lo < 5:
        return offset
    # Restrict this candidate to the terminal QRS region, before broad ST/T.
    limit = min(len(x) - 1, offset + int(.06 * fs), onset + int(.22 * fs))
    candidates = []
    for scale in (.003, .007):
        derivative = np.abs(np.gradient(gaussian_filter1d(x, max(.5, scale * fs))))
        quiet = derivative[lo:hi]
        noise = np.median(quiet) + 3 * 1.4826 * np.median(np.abs(quiet - np.median(quiet)))
        threshold = max(float(noise), .025 * float(np.max(derivative[onset:offset + 1])))
        run = max(3, int(.012 * fs))
        start = max(peak, offset - int(.016 * fs))
        for index in range(start, limit - run + 1):
            if np.all(derivative[index:index + run] <= threshold):
                candidates.append(index)
                break
    if len(candidates) != 2 or abs(candidates[0] - candidates[1]) > .012 * fs:
        return offset
    candidate = int(round(np.median(candidates)))
    return candidate if offset + .008 * fs <= candidate <= offset + .05 * fs else offset


def t_onset_change_point(signal, lo, peak, fs):
    """Fit a quiet ST segment followed by a ramp; return evidence, not a prior."""
    if peak - lo < max(12, int(.06 * fs)):
        return None
    segment = np.asarray(signal[lo:peak + 1], dtype=float)
    t = np.arange(len(segment), dtype=float)
    best = None
    margin = max(4, int(.016 * fs))
    # A constant ST + linear ramp allows an elevated ST level without forcing
    # the signal to zero. Compare against a single-line null model.
    design = np.column_stack((np.ones_like(t), t))
    null_residual = segment - design @ np.linalg.lstsq(design, segment, rcond=None)[0]
    null_error = float(np.dot(null_residual, null_residual))
    for pivot in range(margin, len(segment) - margin, max(1, int(.004 * fs))):
        ramp = np.maximum(t - pivot, 0)
        model = np.column_stack((np.ones_like(t), ramp))
        residual = segment - model @ np.linalg.lstsq(model, segment, rcond=None)[0]
        error = float(np.dot(residual, residual))
        if best is None or error < best[0]:
            best = (error, pivot)
    if best is None or null_error < 1e-10 or best[0] > .60 * null_error:
        return None
    return int(lo + best[1])


def correct_p_boundaries(features, assessments, ecg, fs):
    """Replace weak existing P bounds only after the final atrial state gate."""
    from .core import _p_isoelectric_reference, _ptf_v1
    audit = []
    by_id = {a.beat_id: a for a in assessments}
    for feature in features:
        assessment = by_id.get(feature.beat_id)
        if assessment is None or not assessment.accepted or assessment.p_state != "P_PRESENT":
            continue
        row = assessment.per_lead.get(feature.lead)
        if (row is None or feature.lead not in assessment.valid_leads or not row.informative
                or row.onset is None or row.offset is None or row.peak is None
                or row.on_t_overlap or row.ta_ambiguous or not row.hard_quality_pass):
            continue
        old = feature.p
        # Existing good detections are authoritative. All-missing P continues
        # to use the existing dedicated backfill path.
        if old.peak is None:
            continue
        strength = min(row.onset_confidence, row.offset_confidence)
        old_strength = min(feature.p_onset_confidence or 0, feature.p_offset_confidence or 0)
        if strength < .65 or old_strength >= .45 or strength < old_strength + .2:
            continue
        onset, peak, offset = int(row.onset), int(row.peak), int(row.offset)
        if not (0 <= onset < peak < offset < ecg.shape[1]):
            continue
        if feature.qrs.onset is None or offset >= feature.qrs.onset - .008 * fs:
            continue
        if not (35 <= (offset - onset) * 1000 / fs <= 180):
            continue
        shift = max(abs(onset - (old.onset if old.onset is not None else onset)),
                    abs(offset - (old.offset if old.offset is not None else offset))) * 1000 / fs
        if not 8 <= shift <= 80:
            continue
        signal = ecg[STANDARD_12_LEADS.index(feature.lead)]
        baseline, _ = _p_isoelectric_reference(signal, p_on=onset, p_off=offset,
                      qrs_on=feature.qrs.onset, fs=fs, fallback=float(signal[onset]))
        segment = signal[onset:offset + 1] - baseline
        components = measure_p_components(signal, p_on=onset, p_peak=peak, p_off=offset, baseline=baseline, fs=fs)
        audit.append({"beat_id": feature.beat_id, "lead": feature.lead,
                      "before": [old.onset, old.peak, old.offset], "after": [onset, peak, offset],
                      "reason": "accepted_robust_p_replaces_weak_local", "old_strength": old_strength,
                      "new_strength": strength})
        feature.p = WaveBounds(onset=onset, peak=peak, offset=offset)
        feature.pr_ms = (feature.qrs.onset - onset) * 1000 / fs
        feature.p_dur_ms = (offset - onset) * 1000 / fs
        feature.p_amp_mv = float(signal[peak] - baseline)
        feature.p_area, feature.p_signed_area = float(trapezoid(np.abs(segment))), float(trapezoid(segment))
        feature.p_confidence = strength
        feature.p_onset_confidence, feature.p_offset_confidence = row.onset_confidence, row.offset_confidence
        feature.p_onset_corrected_index, feature.p_offset_corrected_index = onset, offset
        feature.p_boundary_source = "robust_evidence_correction"
        feature.p_notched, feature.p_biphasic = bool(components["is_notched"]), bool(components["is_biphasic"])
        for target, source in (("p_notch_interval_ms", "notch_interval_ms"),
                ("p_initial_duration_ms", "initial_duration_ms"), ("p_initial_amp_mv", "initial_amplitude_mV"),
                ("p_terminal_duration_ms", "terminal_duration_ms"), ("p_terminal_amp_mv", "terminal_amplitude_mV"),
                ("p_terminal_area_mv_ms", "terminal_area_mv_ms")):
            setattr(feature, target, components[source])
        feature.ptf_v1_mv_ms = _ptf_v1(signal, onset, offset, baseline, fs) if feature.lead == "V1" else None
        feature.flags = [flag for flag in feature.flags if flag != "p_unreliable"]
        feature.flags.append("p_robust_evidence_corrected")
    return audit

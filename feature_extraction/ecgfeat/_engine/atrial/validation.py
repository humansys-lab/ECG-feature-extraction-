"""Waveform verification for classical P candidates and independent atrial events.

Checks use only measured channels and detected ventricular timing. They do not
infer absent P from RR irregularity, so ectopy/AV block is not equated to AF.
"""
from __future__ import annotations
import numpy as np

from ...models import STANDARD_12_LEADS
from ..preprocess import lowpass_filter


def p_waveform_evidence(signal, filtered, peak, fs):
    """Local chord prominence and width; returns evidence, not a probability."""
    half = max(3, round(.08*fs))
    if peak-half < 0 or peak+half >= len(signal):
        return None
    lo, hi = peak-half, peak+half+1
    segment = filtered[lo:hi]
    edge = max(1, round(.01*fs))
    baseline = np.linspace(np.median(segment[:edge]), np.median(segment[-edge:]), len(segment))
    wave = segment-baseline
    center = half
    radius = max(1, round(.02*fs))
    local = abs(wave[center-radius:center+radius+1])
    local_peak = center-radius+int(np.argmax(local))
    amplitude = abs(float(wave[local_peak]))
    residual = signal[lo:hi]-filtered[lo:hi]
    noise = max(.004, 1.4826*float(np.median(abs(residual-np.median(residual)))))
    if amplitude < max(.018, 2.5*noise):
        return None
    polarity = 1 if wave[local_peak] >= 0 else -1
    mask = polarity*wave > .5*amplitude
    left = right = local_peak
    while left > 0 and mask[left-1]:
        left -= 1
    while right+1 < len(wave) and mask[right+1]:
        right += 1
    width = (right-left+1)/fs
    if not .020 <= width <= .120:
        return None
    return {"peak": int(lo+local_peak), "amplitude_mv": amplitude,
            "snr": float(amplitude/noise), "half_height_width_ms": width*1000,
            "snippet": wave / max(float(np.linalg.norm(wave)), 1e-12)}


def ventricular_windows(features, fs):
    by_beat = {}
    for f in features:
        by_beat.setdefault(f.beat_id, []).append(f)
    result = []
    for items in by_beat.values():
        on = [f.qrs.onset for f in items if f.qrs.onset is not None]
        off = [f.qrs.offset for f in items if f.qrs.offset is not None]
        if on and off:
            result.append((int(np.median(on))-round(.015*fs), int(np.median(off))+round(.015*fs)))
    return result


def validate_atrial_events(events, ecg, fs, features, quality, *, audit=None):
    """Reject ventricular/T conflicts; lack of repetition never rejects a P.

    Tight 25ms agreement with QRS envelopes in two measured channels catches
    missed ventricular activations without blanking the short-PR region.
    Remaining events are still candidates, not accepted diagnostic P waves.
    """
    from ..detection.adaptive_qrs import channel_candidates
    active = [i for i in range(min(len(ecg), len(STANDARD_12_LEADS)))
              if np.isfinite(ecg[i]).all() and np.std(ecg[i]) > 1e-8]
    independent = [channel_candidates(ecg[i], fs)[0] for i in active]
    filtered = lowpass_filter(ecg, fs, 15.)
    qrs_windows = ventricular_windows(features, fs)
    by_beat = {}
    active_names = {STANDARD_12_LEADS[i] for i in active}
    for feature in features:
        if (feature.lead in active_names and getattr(feature, "t", None) is not None
                and feature.t.peak is not None):
            by_beat.setdefault(feature.beat_id, []).append(feature.t)
    t_windows = []
    for waves in by_beat.values():
        on = [t.onset for t in waves if t.onset is not None]
        off = [t.offset for t in waves if t.offset is not None]
        peaks = [t.peak for t in waves]
        if on and off:
            center = float(np.median(peaks))
            t_windows.append((max(float(np.median(on)), center-.060*fs),
                              min(float(np.median(off)), center+.060*fs)))
    kept = []
    rejected = {"ventricular_overlap": 0, "independent_qrs_conflict": 0,
                "measured_t_peak_conflict": 0}
    tolerance = round(.025*fs)
    for event in events:
        sample = int(event["sample"])
        if any(a <= sample <= b for a,b in qrs_windows):
            rejected["ventricular_overlap"] += 1
            continue
        support = sum(np.searchsorted(peaks, sample+tolerance, side="right") >
                      np.searchsorted(peaks, sample-tolerance) for peaks in independent)
        if support >= 2:
            rejected["independent_qrs_conflict"] += 1
            continue
        if any(a <= sample <= b for a,b in t_windows):
            rejected["measured_t_peak_conflict"] += 1
            continue
        evidence = [STANDARD_12_LEADS[i] for i in active
                    if p_waveform_evidence(ecg[i], filtered[i], sample, fs) is not None]
        result = dict(event)
        result["validation"] = {"method": "independent_qrs_and_measured_t_conflict_check",
                                "support_leads": evidence,
                                "status": "candidate_without_ventricular_conflict"}
        kept.append(result)
    if audit is not None:
        audit.update(input_count=len(events), accepted_count=len(kept), rejected=rejected)
    return kept

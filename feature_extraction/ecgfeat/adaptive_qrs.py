"""Traditional per-channel QRS envelopes with local evidence selection.

The two moving-average construction follows Elgendi et al. (2013),
doi:10.1371/journal.pone.0073557. Channel selection, corroboration and
refractory arbitration below are project-specific, not a paper reproduction.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d

from .preprocess import bandpass_filter
from .models import QRSDetectorResult, QRSCandidateWindow, STANDARD_12_LEADS


def _nearest_distances(values, reference):
    """Nearest sorted-reference distance without a quadratic distance matrix."""
    insertion = np.searchsorted(reference, values)
    left = reference[np.clip(insertion-1, 0, len(reference)-1)]
    right = reference[np.clip(insertion, 0, len(reference)-1)]
    return np.minimum(abs(values-left), abs(values-right))


def guard_weak_additions(candidates, baseline, ecg, fs, leads):
    """Reject a secondary faint activation population beside stable QRS.

    Nonconducted P can trigger a narrow-band QRS envelope. The guard operates
    only when both detectors retain >=90% of at least six original activations
    and at least one channel has stable QRS amplitude. Additions must be faint
    on every measured channel to be removed. It never removes an
    original activation, assumes no RR regularity and uses gain-free ratios.
    """
    original = np.asarray(baseline.r_locs, dtype=int)
    proposed = np.asarray([c.refined_r_index for c in candidates], dtype=int)
    if len(original) < 6 or len(leads) < 2 or len(proposed) == 0:
        return candidates
    shared = original[_nearest_distances(original, proposed) <= .050*fs]
    extras = _nearest_distances(proposed, original) > .075*fs
    if len(shared) < .90*len(original) or np.count_nonzero(extras) < 3:
        return candidates
    filtered = bandpass_filter(ecg[list(leads)], fs, 8., min(20., .4*fs))
    half = max(1, round(.080*fs))

    def amplitudes(peaks):
        return np.array([np.max(abs(filtered[:, max(0,p-half):min(ecg.shape[1],p+half+1)]), axis=1)
                         for p in peaks])

    reference = amplitudes(shared)
    center = np.median(reference, axis=0)
    variation = 1.4826*np.median(abs(reference-center), axis=0)/np.maximum(center, 1e-12)
    stable = variation <= .25
    if np.any(center < 1e-8) or not np.any(stable):
        return candidates
    ratio = amplitudes(proposed)/center
    weak = extras & (np.max(ratio,axis=1) < .35) & (np.min(ratio[:,stable],axis=1) < .15)
    return [candidate for candidate, reject in zip(candidates, weak) if not reject]


def channel_candidates(signal, fs):
    filtered = bandpass_filter(signal, fs, 8., min(20., .4*fs))
    energy = filtered**2
    short = uniform_filter1d(energy, max(1, round(.111*fs)), mode="nearest")
    threshold = uniform_filter1d(energy, max(1, round(.667*fs)), mode="nearest") + .02*float(np.mean(energy))
    edges = np.diff(np.r_[False, short > threshold, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    peaks, confidence = [], []
    for start, stop in zip(starts, stops):
        if stop-start < .111*fs:
            continue
        peak = int(start + np.argmax(abs(filtered[start:stop])))
        # Narrow high-frequency activation relative to its neighborhood is
        # useful evidence against slow electrode-motion excursions. It is not
        # a calibrated probability and never enforces a regular RR pattern.
        lo, hi = max(0, peak-round(.15*fs)), min(len(signal), peak+round(.15*fs)+1)
        a, b = max(lo, peak-round(.04*fs)), min(hi, peak+round(.04*fs)+1)
        concentration = float(np.sum(energy[a:b])/(np.sum(energy[lo:hi])+1e-12))
        peaks.append(peak); confidence.append(concentration)
    return np.asarray(peaks, dtype=int), np.asarray(confidence, dtype=float), filtered


def detect_adaptive_qrs(ecg, fs, leads, *, channel_quality=None):
    active = [i for i in dict.fromkeys(leads) if 0 <= i < ecg.shape[0]
              and np.isfinite(ecg[i]).all() and np.std(ecg[i]) > 1e-8]
    channels = {i: channel_candidates(ecg[i], fs) for i in active}
    candidates = []
    tolerance = max(1, round(.05*fs))
    for i, (peaks, strengths, _) in channels.items():
        for peak, strength in zip(peaks, strengths):
            local_scores = {}
            for j, (other, values, _) in channels.items():
                nearby = values[abs(other-peak) <= 4*fs]
                local_scores[j] = (.5*float(np.median(nearby)) + .5*float((channel_quality or {}).get(j, 0.))) if nearby.size else 0.
            best = max(active, key=lambda j: (local_scores[j], j == 1))
            supporters = [j for j, (other, _, _) in channels.items() if np.any(abs(other-peak) <= tolerance)]
            if i != best:
                # An independent strong corroborated peak may fill a missed
                # activation without assuming sinus rhythm or excluding PVCs.
                if len(supporters) < 2 or strength < .65:
                    continue
            elif len(active) > 1 and len(supporters) < 2 and strength < .55:
                continue
            confidence = float(np.clip(.7*strength + .3*len(supporters)/max(len(active), 1), 0, 1))
            candidates.append((int(peak), confidence, i, float(strength)))
    # Competing leads may point to different lobes of one QRS. Retain the
    # strongest evidence in a 200ms refractory period, independent of order.
    selected = []
    for row in sorted(candidates, key=lambda r: (-r[1], -r[3], r[0])):
        if all(abs(row[0]-r[0]) >= round(.20*fs) for r in selected):
            selected.append(row)
    selected.sort()
    margin = round(.16*fs)
    selected = [r for r in selected if margin <= r[0] <= ecg.shape[1]-margin]
    windows = [QRSCandidateWindow(detector_peak_index=p, search_start_index=max(0, p-round(.05*fs)),
                                 search_end_index=min(ecg.shape[1]-1, p+round(.05*fs)),
                                 refined_r_index=p, detector_height=h, confidence=c) for p,c,_,h in selected]
    return QRSDetectorResult(r_locs=[r[0] for r in selected], qrs_candidate_windows=windows,
                             qrs_detector_confidence=[r[1] for r in selected], energy_threshold=0.,
                             detection_leads=[STANDARD_12_LEADS[i] for i in active],
                             fallback_used=False, fallback_reason="adaptive_per_channel_envelopes")

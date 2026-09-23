"""Optional deterministic candidates; no fitted model or deep-learning runtime.

The phase candidate follows the phasor-transform idea (Saclova et al., 2022),
not the paper's full pathology classifier. Projection and sequence selection
are experimental adaptations, not reproductions of published benchmark scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def phasor_p_candidates(signal, *, lo, hi, baseline, fs, min_amp=.008):
    """Return up to three prominence-ranked, polarity-independent P peaks.

    The caller supplies a QRS/T-excluded search region and performs its usual
    morphology, rhythm and cross-lead checks. Peaks alone are not P boundaries.
    """
    x = np.asarray(signal, dtype=float)
    lo, hi = max(0, int(lo)), min(len(x), int(hi))
    if fs <= 0 or hi - lo < max(8, int(.04 * fs)):
        return []
    segment = x[lo:hi] - baseline
    if not np.all(np.isfinite(segment)):
        return []
    smooth = gaussian_filter1d(segment, max(.5, .004 * fs))
    residual = segment - smooth
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    phase = np.abs(np.arctan2(smooth, max(.005, 2 * sigma)))
    peaks, properties = find_peaks(
        phase, prominence=.25, distance=max(1, int(.025 * fs)),
        width=max(1, int(.008 * fs)),
    )
    candidates = []
    for index in np.argsort(-properties["prominences"], kind="stable"):
        peak = int(peaks[index])
        if abs(smooth[peak]) < max(min_amp, 3 * sigma):
            continue
        radius = max(1, int(.008 * fs))
        left, right = max(0, peak - radius), min(len(segment), peak + radius + 1)
        peak = left + int(np.argmax(np.abs(segment[left:right])))
        if all(abs(lo + peak - item) > .016 * fs for item in candidates):
            candidates.append(lo + peak)
        if len(candidates) == 3:
            break
    return candidates


def boundary_projection(signals, noise, *, center, fs):
    """Fit a noise-weighted direction to derivatives near ONE boundary.

    Return a projected signal and coherence, or None when fewer than two
    finite, nonflat independent physical leads support the projection.
    Sign and lead gain are removed by whitening; unrelated noise is rejected.
    """
    x, residual = np.asarray(signals, float), np.asarray(noise, float)
    if x.ndim != 2 or residual.shape != x.shape or fs <= 0 or x.shape[1] < 8:
        return None
    finite = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(residual), axis=1)
    usable = finite & (np.ptp(x, axis=1) >= .015)
    x, residual = x[usable], residual[usable]
    if len(x) < 2:
        return None
    scale = np.maximum(.002, 1.4826 * np.median(
        np.abs(residual - np.median(residual, axis=1, keepdims=True)), axis=1))
    # Limit a nearly noiseless channel's leverage without counting derived
    # limb leads as extra independent evidence.
    scale = np.maximum(scale, .25 * np.median(scale))
    centered = x - np.median(x, axis=1, keepdims=True)
    whitened = centered / scale[:, None]
    derivative = np.gradient(gaussian_filter1d(whitened, max(.5, .008 * fs), axis=1), axis=1)
    radius = max(3, int(.06 * fs))
    lo, hi = max(0, int(center) - radius), min(x.shape[1], int(center) + radius + 1)
    if hi - lo < 6:
        return None
    local = derivative[:, lo:hi]
    try:
        u, singular, _ = np.linalg.svd(local, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    energy = float(np.sum(singular ** 2))
    if energy < 1e-10:
        return None
    coherence = float(singular[0] ** 2 / energy)
    direction = u[:, 0]
    # A single dominant lead is not multilead corroboration.
    support = np.count_nonzero(np.abs(direction) >= .20 * np.max(np.abs(direction)))
    if coherence < .75 or support < 2:
        return None
    projected = direction @ whitened
    projected *= float(np.median(scale))
    return projected, coherence


@dataclass(frozen=True)
class BoundaryState:
    onset: int
    offset: int
    cost: float


def select_boundary_sequence(states, anchors, rr_samples, qrs_widths, fs):
    """Exact DP over supplied T-boundary pairs with soft duration transitions.

    Empty candidate sets break the chain. Ectopy/rate changes reset temporal
    coupling. No missing wave is created and no rigid P-QRS-T grammar imposed.
    All states already satisfy local peak/search-window constraints.
    """
    n = len(states)
    if fs <= 0 or not (n == len(anchors) == len(rr_samples) == len(qrs_widths)):
        raise ValueError("sequence geometry must have matching lengths and positive fs")
    chosen = [None] * n
    costs, back = [], []
    for i, row in enumerate(states):
        if not row:
            costs.append(np.empty(0))
            back.append([])
            continue
        local = np.asarray([s.cost for s in row], float)
        parents = [-1] * len(row)
        if i and states[i - 1]:
            reset = (abs(qrs_widths[i] - qrs_widths[i - 1]) > .03 * fs
                     or min(rr_samples[i], rr_samples[i - 1]) <= 0
                     or max(rr_samples[i], rr_samples[i - 1]) > 1.2 * min(rr_samples[i], rr_samples[i - 1])
                     or anchors[i] - anchors[i - 1] > 1.5 * max(rr_samples[i], rr_samples[i - 1]))
            rate_shift = .12 * (rr_samples[i] - rr_samples[i - 1])
            for j, current in enumerate(row):
                transitions = []
                for previous in states[i - 1]:
                    # Bounded penalties cannot erase a strong local observation.
                    delta_on = (current.onset - anchors[i]) - (previous.onset - anchors[i - 1]) - rate_shift
                    delta_off = (current.offset - anchors[i]) - (previous.offset - anchors[i - 1]) - rate_shift
                    transitions.append(0.0 if reset else .35 * (
                        min(abs(delta_on) / (.04 * fs), 2.) + min(abs(delta_off) / (.04 * fs), 2.)))
                totals = costs[i - 1] + transitions
                parents[j] = int(np.argmin(totals))
                local[j] += totals[parents[j]]
        costs.append(local)
        back.append(parents)
    i = n - 1
    while i >= 0:
        if not states[i]:
            i -= 1
            continue
        index = int(np.argmin(costs[i]))
        while i >= 0 and index >= 0:
            chosen[i] = index
            index = back[i][index]
            i -= 1
    return chosen

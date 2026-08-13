"""Scratch exploratory checks preserved from earlier multilead P/T work."""
from __future__ import annotations

import numpy as np

from feature_extraction.ecgfeat.delineate import (
    _candidate_triplets,
    _fuse_peak_anchor,
    _median_prior_peak,
    _t_end_geometric,
    delineate_beats,
)


def test_candidate_triplets_finds_peak():
    sig = np.zeros(100)
    sig[40] = 0.3
    candidates = _candidate_triplets(sig, lo=10, hi=90, baseline=0.0, min_amp=0.01)
    assert len(candidates) >= 1
    assert candidates[0][0] == 40


def test_candidate_triplets_filters_small_amplitude():
    sig = np.zeros(100)
    sig[40] = 0.005
    candidates = _candidate_triplets(sig, lo=10, hi=90, baseline=0.0, min_amp=0.008)
    assert len(candidates) == 0


def test_candidate_triplets_max_candidates():
    sig = np.zeros(200)
    for i in [30, 70, 110, 150]:
        sig[i] = 0.1
    candidates = _candidate_triplets(sig, lo=10, hi=190, baseline=0.0, max_candidates=2)
    assert len(candidates) <= 2


class MockLeadQuality:
    def __init__(self, reliable_for_p=True, reliable_for_qt=True):
        self.reliable_for_p = reliable_for_p
        self.reliable_for_qt = reliable_for_qt


def test_fuse_peak_anchor_consensus():
    candidates = {
        "I": [(100, 0.2, 0.5)],
        "II": [(102, 0.3, 0.6)],
        "III": [(101, 0.25, 0.55)],
    }
    quality = {lead: MockLeadQuality(reliable_for_p=True) for lead in ("I", "II", "III")}
    anchor = _fuse_peak_anchor(
        candidates,
        quality,
        reliable_attr="reliable_for_p",
        prior_peak=None,
        cluster_radius=5,
    )
    assert anchor is not None
    assert 99 <= anchor <= 103


def test_fuse_peak_anchor_ignores_unreliable_leads():
    candidates = {
        "I": [(100, 0.2, 0.5)],
        "II": [(500, 0.5, 1.0)],
    }
    quality = {
        "I": MockLeadQuality(reliable_for_p=True),
        "II": MockLeadQuality(reliable_for_p=False),
    }
    anchor = _fuse_peak_anchor(
        candidates,
        quality,
        reliable_attr="reliable_for_p",
        prior_peak=None,
        cluster_radius=5,
    )
    assert anchor == 100


def test_fuse_peak_anchor_returns_none_when_no_reliable():
    candidates = {"I": [(100, 0.2, 0.5)]}
    quality = {"I": MockLeadQuality(reliable_for_p=False)}
    anchor = _fuse_peak_anchor(
        candidates,
        quality,
        reliable_attr="reliable_for_p",
        prior_peak=None,
        cluster_radius=5,
    )
    assert anchor is None


def test_median_prior_peak_basic():
    prior = {"I": {"t_peak": 100}, "II": {"t_peak": 104}, "III": {"t_peak": 102}}
    result = _median_prior_peak(prior, "t_peak")
    assert result == 102


def test_median_prior_peak_missing_field():
    prior = {"I": {"qrs_on": 50}}
    result = _median_prior_peak(prior, "t_peak")
    assert result is None


def _make_t_wave(fs=500, peak_ms=200, end_ms=360, amp=0.4, n=600):
    """Simple upright T-wave: Gaussian rise to peak, linear descent to end."""
    sig = np.zeros(n)
    t_peak = int(peak_ms * fs / 1000)
    t_end = int(end_ms * fs / 1000)
    rise = np.linspace(0, amp, t_peak + 1)
    sig[: t_peak + 1] = rise
    fall = np.linspace(amp, 0, t_end - t_peak + 1)
    sig[t_peak : t_end + 1] = fall
    return sig, t_peak, t_end


def test_t_end_geometric_upright(fs=500):
    sig, t_peak, true_end = _make_t_wave(fs=fs)
    search_end = int(0.50 * fs)
    t_off, conf, method = _t_end_geometric(sig, t_peak, search_end, 0.0, fs)
    assert method == "dxl_chord"
    assert t_off is not None
    assert abs(t_off - true_end) <= int(0.030 * fs), f"got {t_off}, expected {true_end}"


def test_t_end_geometric_inverted(fs=500):
    sig, t_peak, true_end = _make_t_wave(fs=fs, amp=-0.3)
    search_end = int(0.50 * fs)
    t_off, conf, method = _t_end_geometric(sig, t_peak, search_end, 0.0, fs)
    assert method == "dxl_chord"
    assert t_off is not None
    assert abs(t_off - true_end) <= int(0.030 * fs)


def test_t_end_geometric_fallback_flat():
    sig = np.zeros(200)
    fs = 500
    t_off, conf, method = _t_end_geometric(sig, 50, 150, 0.0, fs)
    assert method in ("threshold_fallback", "dxl_chord", "missing_peak", "chord_fallback")


def test_t_end_geometric_confidence_range():
    fs = 500
    sig, t_peak, _ = _make_t_wave(fs=fs)
    _, conf, _ = _t_end_geometric(sig, t_peak, int(0.50 * fs), 0.0, fs)
    assert 0.0 <= conf <= 1.0


def _synthetic_12lead(fs=500, n_beats=3, rr_ms=700):
    """Minimal synthetic 12-lead ECG with obvious P and T waves."""
    n = int(fs * (n_beats * rr_ms / 1000 + 0.5))
    ecg = np.zeros((12, n))
    r_locs = [int((i + 0.5) * rr_ms * fs / 1000) for i in range(n_beats)]
    for r in r_locs:
        for li in range(12):
            amp = 1.0 if li < 6 else 0.5
            p = r - int(0.16 * fs)
            if p - 10 >= 0 and p + 10 < n:
                ecg[li, p - 10 : p + 11] += amp * 0.12 * np.hanning(21)
            if r - 5 >= 0 and r + 5 < n:
                ecg[li, r - 5 : r + 6] += amp * np.hanning(11)
            t = r + int(0.24 * fs)
            if t - 20 >= 0 and t + 21 < n:
                ecg[li, t - 20 : t + 21] += amp * 0.35 * np.hanning(41)
    return ecg, np.asarray(r_locs)


def test_delineate_beats_runs_with_fusion():
    fs = 500
    ecg, r_locs = _synthetic_12lead(fs=fs)
    results = delineate_beats(ecg, fs, r_locs)
    assert len(results) > 0
    for feat in results:
        assert feat.qt_ms is None or feat.qt_ms > 0

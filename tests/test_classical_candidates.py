from itertools import product

import numpy as np
import pytest

from feature_extraction.ecgfeat.classical_candidates import (
    BoundaryState, boundary_projection, phasor_p_candidates, select_boundary_sequence,
)
from feature_extraction.ecgfeat import RefinementConfig
from feature_extraction.ecgfeat.t_wave_refinement import refine_t_wave_boundaries
from tests.test_t_wave_refinement import _feature, _synthetic_t_signal


@pytest.mark.parametrize("fs", [250, 500, 1000])
@pytest.mark.parametrize("polarity", [-1, 1])
def test_phasor_detects_small_p_beside_large_separate_wave(fs, polarity):
    t = np.arange(fs) / fs
    x = .4 * np.exp(-.5 * ((t - .25) / .04) ** 2)
    x += polarity * .025 * np.exp(-.5 * ((t - .65) / .02) ** 2)
    candidates = phasor_p_candidates(x + .2, lo=0, hi=fs, baseline=.2, fs=fs)
    assert any(abs(p / fs - .65) < .012 for p in candidates)
    assert phasor_p_candidates(np.ones(fs), lo=0, hi=fs, baseline=1, fs=fs) == []
    assert phasor_p_candidates(x, lo=20, hi=10, baseline=0, fs=fs) == []


def test_phasor_does_not_cross_requested_qrs_exclusion():
    x = np.zeros(500)
    x[400] = 2
    assert phasor_p_candidates(x, lo=100, hi=390, baseline=0, fs=500) == []
    x[200] = np.nan
    assert phasor_p_candidates(x, lo=100, hi=390, baseline=0, fs=500) == []


def test_projection_handles_two_polarities_and_ignores_absent_channels():
    t = np.arange(200)
    wave = .15 * np.exp(-.5 * ((t - 100) / 25) ** 2)
    x = np.array([wave, -2 * wave, np.zeros(200)])
    noise = np.zeros_like(x)
    result = boundary_projection(x, noise, center=145, fs=500)
    assert result is not None
    projected, coherence = result
    assert coherence > .99
    assert abs(np.corrcoef(projected, wave)[0, 1]) > .99
    assert boundary_projection(x[[0, 2]], noise[[0, 2]], center=145, fs=500) is None
    assert boundary_projection(np.zeros_like(x), noise, center=145, fs=500) is None


def test_dynamic_programming_agrees_with_exhaustive_joint_selection():
    rows = [[BoundaryState(100, 190, 0), BoundaryState(110, 200, .15)],
            [BoundaryState(640, 730, 0), BoundaryState(610, 700, .15)],
            [BoundaryState(1100, 1190, 0), BoundaryState(1110, 1200, .15)]]
    anchors = [0, 500, 1000]
    def cost(path):
        value = sum(rows[i][j].cost for i, j in enumerate(path))
        for i in range(1, 3):
            a, b = rows[i-1][path[i-1]], rows[i][path[i]]
            value += .35 * (min(abs(b.onset-a.onset-500)/20, 2) + min(abs(b.offset-a.offset-500)/20, 2))
        return value
    selected = select_boundary_sequence(rows, anchors, [500]*3, [50]*3, 500)
    assert cost(selected) == pytest.approx(min(cost(p) for p in product(range(2), repeat=3)))
    assert selected[1] == 1
    assert select_boundary_sequence(rows, anchors, [500]*3, [50, 90, 50], 500) == [0, 0, 0]
    assert select_boundary_sequence([rows[0], [], rows[2]], anchors, [500]*3, [50]*3, 500) == [0, None, 0]
    assert select_boundary_sequence([], [], [], [], 500) == []


@pytest.mark.parametrize("option", ["t_boundary_projection", "t_sequence_selection"])
def test_t_candidates_preserve_missing_p_and_refresh_derived_intervals(option):
    base = _synthetic_t_signal()
    ecg = np.zeros((12, len(base)))
    ecg[1], ecg[7] = base, -.8 * base
    features = [_feature("II"), _feature("V2")]
    for f in features:
        f.p.onset = f.p.peak = f.p.offset = None
    refine_t_wave_boundaries(features, measurement_ecg=ecg, r_locs=np.array([250]),
                            fs=500, refinement=RefinementConfig(**{option: True}))
    for f in features:
        assert f.p.peak is None
        assert f.qrs.offset < f.t.onset < f.t.peak < f.t.offset
        assert f.qt_ms == (f.t.offset - f.qrs.onset) * 2
        assert f.t_dur_ms == (f.t.offset - f.t.onset) * 2


def test_offset_projection_updates_intervals_without_moving_onset(monkeypatch):
    from types import SimpleNamespace
    import feature_extraction.ecgfeat.t_wave_refinement as mod
    ecg = np.tile(_synthetic_t_signal(.1), (12, 1))
    features = [_feature("II", offset=480), _feature("V2", offset=480)]
    candidates = {}
    for f in features:
        candidates[(0, f.lead)] = mod._LeadTCandidates(
            f, ecg[1], ecg[1], 0., .1, 12., {"chord": 330, "mallat": 335},
            {"chord": 480, "mallat": 460, "trapezium": 462}, 330, 480, "chord", True)
    # Search window begins at QRS offset+10ms = 285.
    monkeypatch.setattr(mod, "boundary_projection", lambda *a, **k: (ecg[1, 285:501], .99))
    monkeypatch.setattr(mod, "mallat_t_boundaries", lambda *a, **k: (55, 175))
    audit = []
    mod._classical_t_refinement(candidates, {0: features}, ecg, ecg, np.array([250]), 500,
                               RefinementConfig(t_projection_offset_only=True), audit, None)
    for f in features:
        assert (f.t.onset, f.t.offset) == (330, 460)
        assert f.qt_ms == 470 and f.jt_ms == 360 and f.tpe_ms == 140
        assert f.t_dur_ms == 260 and f.t_area > 0
    assert len(audit) == 2
    # Removing one physical support lead must disable the projection entirely.
    features = [_feature("II", offset=480), _feature("V2", offset=480)]
    for f in features:
        candidates[(0, f.lead)].feature = f
    quality = {"II": SimpleNamespace(reliable_for_t=True), "V2": SimpleNamespace(reliable_for_t=False)}
    audit.clear()
    mod._classical_t_refinement(candidates, {0: features}, ecg, ecg, np.array([250]), 500,
                               RefinementConfig(t_projection_offset_only=True), audit, quality)
    assert not audit and all(f.t.offset == 480 for f in features)

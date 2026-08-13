from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from feature_extraction.ecgfeat.models import STANDARD_12_LEADS
from feature_extraction.ecgfeat.st_localization import (
    apply_hybrid_st_measurement,
    localize_st_robust,
)


def _st_signal(
    *,
    fs: int = 500,
    j_index: int = 450,
    st_j_mv: float = 0.18,
    st80_mv: float = 0.18,
) -> np.ndarray:
    signal = np.zeros(1000, dtype=float)
    signal[380:400] = np.linspace(0.0, 1.0, 20, endpoint=False)
    signal[400:425] = np.linspace(1.0, -0.40, 25, endpoint=False)
    signal[425 : j_index + 1] = np.linspace(-0.40, st_j_mv, j_index - 424)
    st80_index = j_index + int(round(0.080 * fs))
    signal[j_index : st80_index + 1] = np.linspace(
        st_j_mv, st80_mv, st80_index - j_index + 1
    )
    signal[st80_index:540] = st80_mv
    signal[540:650] = st80_mv + 0.30 * np.sin(np.linspace(0.0, np.pi, 110))
    return signal


def test_robust_st_measures_horizontal_elevation_with_pr_baseline() -> None:
    signal = _st_signal(st_j_mv=0.18, st80_mv=0.18)
    signal[330:340] = 0.12  # P wave, excluded by the post-P PR baseline.

    result = localize_st_robust(
        signal,
        fs=500,
        r_index=400,
        qrs_onset=380,
        qrs_offset=450,
        p_offset=345,
        t_onset=540,
        next_r_index=850,
        consensus_j_index=450,
        consensus_support=10,
    )

    assert result.reliable
    assert result.baseline_source == "post_p_pr_segment"
    assert result.j_index == 450
    assert result.j_mv == 0.18
    assert result.st20_mv == 0.18
    assert result.st40_mv == 0.18
    assert result.st60_mv == 0.18
    assert result.st80_mv == 0.18
    assert result.adaptive_index == 490
    assert result.adaptive_mv == 0.18
    assert result.curvature_mv_per_ms2 is not None
    assert abs(result.curvature_mv_per_ms2) < 1e-8
    assert result.trend == "horizontal"
    assert result.shape == "straight"


def test_robust_st_detects_downsloping_profile() -> None:
    result = localize_st_robust(
        _st_signal(st_j_mv=0.20, st80_mv=0.08),
        fs=500,
        r_index=400,
        qrs_onset=380,
        qrs_offset=450,
        p_offset=345,
        t_onset=540,
        next_r_index=850,
        consensus_j_index=450,
        consensus_support=8,
    )

    assert result.reliable
    assert result.slope_mv_per_ms is not None
    assert result.slope_mv_per_ms < -0.0005
    assert result.curvature_mv_per_ms2 is not None
    assert result.trend == "downsloping"


def test_robust_st_marks_early_t_overlap_unreliable() -> None:
    result = localize_st_robust(
        _st_signal(),
        fs=500,
        r_index=400,
        qrs_onset=380,
        qrs_offset=450,
        p_offset=345,
        t_onset=480,
        next_r_index=850,
        consensus_j_index=450,
        consensus_support=8,
    )

    assert not result.reliable
    assert result.unreliable_reason == "early_t_overlap"
    assert result.j_mv is not None


def test_apply_hybrid_st_uses_consensus_without_replacing_native_fields() -> None:
    ecg = np.vstack([_st_signal() for _ in STANDARD_12_LEADS])
    features = []
    for index, lead in enumerate(STANDARD_12_LEADS):
        native_offset = 470 if lead == "II" else 450
        features.append(
            SimpleNamespace(
                lead=lead,
                beat_id=0,
                p=SimpleNamespace(onset=320, peak=335, offset=345),
                qrs=SimpleNamespace(onset=380, peak=400, offset=native_offset),
                t=SimpleNamespace(onset=540, peak=590, offset=650),
                j_index=native_offset,
                st_j_remeasured_index=None,
                flags=[],
            )
        )

    apply_hybrid_st_measurement(
        features,
        measurement_ecg=ecg,
        r_locs=np.asarray([400]),
        fs=500,
    )

    lead_ii = features[1]
    assert lead_ii.qrs.offset == 470
    assert lead_ii.j_index == 470
    assert lead_ii.st_hybrid_consensus_index == 450
    assert lead_ii.st_hybrid_consensus_support == 12
    assert abs(lead_ii.st_hybrid_j_index - lead_ii.qrs.offset) <= 15
    assert lead_ii.st_hybrid_j_index < lead_ii.qrs.offset
    assert lead_ii.st_hybrid_20ms_mv is not None
    assert lead_ii.st_hybrid_60ms_mv is not None
    assert lead_ii.st_hybrid_adaptive_mv is not None
    assert lead_ii.st_hybrid_curvature_mv_per_ms2 is not None
    assert lead_ii.st_hybrid_reliable

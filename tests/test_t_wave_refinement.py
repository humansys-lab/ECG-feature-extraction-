from __future__ import annotations

import numpy as np

from feature_extraction.ecgfeat.features import _apply_t_fusion_qt_reliability
from feature_extraction.ecgfeat.models import (
    LeadBeatFeatures,
    STANDARD_12_LEADS,
    WaveBounds,
)
from feature_extraction.ecgfeat.t_wave_refinement import (
    mallat_t_boundaries,
    refine_t_wave_boundaries,
    robust_t_offset_fusion,
    trapezium_t_offset,
)


def _feature(lead: str, *, peak: int = 390, offset: int = 475) -> LeadBeatFeatures:
    return LeadBeatFeatures(
        lead=lead,
        beat_id=0,
        p=WaveBounds(onset=120, peak=135, offset=150),
        qrs=WaveBounds(onset=225, peak=250, offset=280),
        t=WaveBounds(onset=330, peak=peak, offset=offset),
        qt_ms=float((offset - 225) * 2),
        pr_ms=150.0,
        qrs_ms=110.0,
        p_amp_mv=0.10,
        qrs_area=10.0,
        q_amp_mv=-0.10,
        r_amp_mv=1.0,
        s_amp_mv=-0.20,
        st_on_mv=0.0,
        st_mid_mv=0.0,
        st_80ms_mv=0.0,
        t_amp_mv=0.30,
        j_index=280,
        qt_confidence=0.80,
        t_end_method="dxl_chord",
    )


def _synthetic_t_signal(amplitude: float = 0.30) -> np.ndarray:
    samples = np.arange(800, dtype=float)
    # A mildly asymmetric T wave with a longer terminal tail.
    left = amplitude * np.exp(-0.5 * ((samples - 390.0) / 34.0) ** 2)
    tail = 0.05 * np.exp(-0.5 * ((samples - 445.0) / 48.0) ** 2)
    return left + tail


def test_robust_t_offset_fusion_rejects_single_late_outlier() -> None:
    result = robust_t_offset_fusion(
        [400, 402, 399, 401, 500],
        [1.0, 0.9, 0.8, 1.0, 1.0],
        ["II", "V2", "V3", "V5", "V1"],
        fs=500,
    )

    assert result.reliable
    assert result.support == 4
    assert 399 <= result.center_index <= 402
    assert result.latest_p85_index is not None
    assert "V1" in result.excluded_sources


def test_robust_t_offset_fusion_selects_compact_diverse_temporal_cluster() -> None:
    result = robust_t_offset_fusion(
        [473, 478, 448, 493, 423, 472, 470],
        [
            0.9,
            1.0,
            0.8,
            0.4788028098128683,
            0.95,
            0.8974485958027787,
            0.8941161894894329,
        ],
        ["II", "V2", "I", "III", "V5", "RMS", "PC1"],
        fs=500,
    )

    assert result.support == 4
    assert result.reliable
    assert result.cluster_count >= 3
    assert result.mad_ms is not None and result.mad_ms < 12.0
    assert set(result.used_sources) == {"II", "V2", "RMS", "PC1"}
    assert {"limb", "precordial", "derived"} <= set(
        result.selected_cluster_lead_groups
    )


def test_robust_t_offset_fusion_requires_both_lead_systems() -> None:
    result = robust_t_offset_fusion(
        [400, 402, 399, 401],
        [1.0, 0.9, 0.8, 1.0],
        ["V2", "V3", "V4", "V5"],
        fs=500,
    )

    assert result.support == 4
    assert not result.reliable
    assert "missing_limb_or_precordial_support" in result.reliability_reasons


def test_systematic_early_risk_downgrades_global_qt_even_with_low_mad() -> None:
    reliability, reportable, reasons = _apply_t_fusion_qt_reliability(
        qt_ms=337.0,
        qt_source="reliable_lead_median",
        qt_reliability="reliable",
        t_fusion={
            "evidence_present": True,
            "reliable": False,
            "systematic_early_risk": True,
            "reasons": ["multilead_residual_t_tail"],
        },
    )

    assert reliability == "unreliable"
    assert not reportable
    assert reasons == ["multilead_residual_t_tail"]


def test_independently_tail_confirmed_rescue_remains_reportable() -> None:
    reliability, reportable, reasons = _apply_t_fusion_qt_reliability(
        qt_ms=394.0,
        qt_source="tail_confirmed_late_raw_qt_rescue",
        qt_reliability="rescued",
        t_fusion={
            "evidence_present": True,
            "reliable": False,
            "systematic_early_risk": True,
            "reasons": ["dispersion_above_20ms"],
        },
    )

    assert reliability == "rescued"
    assert reportable
    assert reasons == []


def test_mallat_and_trapezium_return_plausible_independent_candidates() -> None:
    signal = _synthetic_t_signal()
    onset, offset = mallat_t_boundaries(
        signal,
        fs=500,
        peak=390,
        search_start=285,
        search_end=520,
    )
    trapezium = trapezium_t_offset(
        signal,
        fs=500,
        peak=390,
        search_end=520,
    )

    assert onset is not None and 285 <= onset < 390
    assert offset is not None and 390 < offset <= 520
    assert trapezium is not None and 390 < trapezium <= 520


def test_refinement_populates_sqi_fusion_and_rms_pc1_audit_fields() -> None:
    base = _synthetic_t_signal()
    ecg = np.vstack(
        [
            (1.0 + 0.03 * index) * base
            for index, _lead in enumerate(STANDARD_12_LEADS)
        ]
    )
    features = [_feature(lead) for lead in STANDARD_12_LEADS]

    refine_t_wave_boundaries(
        features,
        measurement_ecg=ecg,
        r_locs=np.asarray([250]),
        fs=500,
    )

    lead_ii = features[1]
    assert lead_ii.t_wavelet_onset_index is not None
    assert lead_ii.t_wavelet_offset_index is not None
    assert lead_ii.t_trapezium_offset_index is not None
    assert lead_ii.t_sqi_score is not None
    assert lead_ii.t_sqi_pass
    assert lead_ii.t_offset_fusion_support is not None
    assert lead_ii.t_offset_fusion_support >= 4
    assert lead_ii.t_offset_robust_center_index is not None
    assert lead_ii.t_offset_latest_p85_index is not None
    assert lead_ii.t_rms_offset_index is not None
    assert lead_ii.t_pc1_offset_index is not None
    assert lead_ii.t_derived_spread_ms is not None
    assert lead_ii.t_offset_cluster_count is not None
    assert lead_ii.t_offset_selected_cluster_support is not None
    assert lead_ii.t_global_tpte_ms is not None
    assert lead_ii.t_offset_fusion_reliability_reason is not None

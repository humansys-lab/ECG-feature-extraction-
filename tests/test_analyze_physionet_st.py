from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from analyze_physionet_st import (
    _assess_sparse_hybrid,
    _nearest_stf_values,
    _protocol_value,
    _select_evenly,
)


def test_protocol_uses_j80_at_normal_rate_and_interpolated_j60_at_tachycardia() -> None:
    assert _protocol_value(-0.08, -0.12, 90.0) == -0.12
    assert _protocol_value(-0.08, -0.12, 130.0) == -0.10


def _hybrid_feature(**overrides):
    values = {
        "st_hybrid_unreliable_reason": "insufficient_consensus_support",
        "flags": [],
        "st_hybrid_consensus_support": 2,
        "st_hybrid_j_method": "local_settling_consensus",
        "st_hybrid_j_index": 100,
        "st_hybrid_j_mv": -0.08,
        "st_hybrid_40ms_mv": -0.10,
        "st_hybrid_80ms_mv": -0.12,
        "st_hybrid_slope_mv_per_ms": -0.0005,
        "st_hybrid_baseline_confidence": 0.90,
        "st_hybrid_j_confidence": 0.55,
        "t": SimpleNamespace(onset=160),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sparse_adapter_recomputes_two_lead_consensus_quality() -> None:
    assessment = _assess_sparse_hybrid(
        _hybrid_feature(),
        physical_lead_count=2,
        fs=250,
    )

    assert assessment.usable
    assert assessment.reason == "usable"
    assert assessment.adjusted_confidence is not None
    assert assessment.adjusted_confidence > 0.55


def test_sparse_adapter_records_early_t_as_fixed_protocol_warning() -> None:
    assessment = _assess_sparse_hybrid(
        _hybrid_feature(t=SimpleNamespace(onset=120)),
        physical_lead_count=2,
        fs=250,
    )

    assert assessment.usable
    assert assessment.reason == "usable"
    assert assessment.warnings == ("st80_t_overlap",)


def test_sparse_adapter_does_not_let_consensus_reason_mask_incomplete_window() -> None:
    assessment = _assess_sparse_hybrid(
        _hybrid_feature(st_hybrid_80ms_mv=None),
        physical_lead_count=2,
        fs=250,
    )

    assert not assessment.usable
    assert assessment.reason == "incomplete_fixed_protocol_st"


def test_sparse_adapter_keeps_fixed_amplitude_when_only_slope_is_missing() -> None:
    assessment = _assess_sparse_hybrid(
        _hybrid_feature(st_hybrid_slope_mv_per_ms=None),
        physical_lead_count=2,
        fs=250,
    )

    assert assessment.usable
    assert assessment.reason == "usable"
    assert assessment.warnings == ("missing_st_slope",)


def test_sparse_adapter_requires_two_real_lead_anchors() -> None:
    assessment = _assess_sparse_hybrid(
        _hybrid_feature(st_hybrid_consensus_support=1),
        physical_lead_count=3,
        fs=250,
    )

    assert not assessment.usable
    assert assessment.reason == "insufficient_real_lead_support"


def test_sparse_adapter_keeps_hard_qrs_rejection() -> None:
    assessment = _assess_sparse_hybrid(
        _hybrid_feature(st_hybrid_unreliable_reason="lead_qrs_quality"),
        physical_lead_count=2,
        fs=250,
    )

    assert not assessment.usable
    assert assessment.reason == "lead_qrs_quality"


def test_ltstdb_stf_units_are_converted_to_mv() -> None:
    # sample, then level/reference/deviation for two physical leads.
    stf = np.asarray(
        [
            [1000, 20.0, 4.0, 16.0, -30.0, -10.0, -20.0],
            [1500, 22.0, 4.0, 18.0, -32.0, -10.0, -22.0],
        ]
    )

    values = _nearest_stf_values(stf, sample=1400, lead_index=1)

    assert values["stf_sample"] == 1500
    assert values["stf_level_mv"] == -0.16
    assert values["stf_reference_mv"] == -0.05
    assert values["stf_deviation_mv"] == -0.11


def test_even_selection_keeps_endpoints() -> None:
    assert _select_evenly(list(range(10)), 3) == [0, 4, 9]

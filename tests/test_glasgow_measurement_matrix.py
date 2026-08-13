from __future__ import annotations

from feature_extraction.ecgfeat.glasgow_rules.context import build_context
from feature_extraction.ecgfeat.glasgow_rules.measurement_matrix import (
    EXPECTED_MATRIX_ROWS,
    build_measurement_matrix,
)
from feature_extraction.ecgfeat.models import PatientMeta, STANDARD_12_LEADS
from tests.glasgow_test_helpers import make_features


def _complete_profile() -> dict[str, float | int]:
    return {
        "p_onset_ms": 100.0,
        "p_duration_ms": 90.0,
        "qrs_onset_ms": 200.0,
        "qrs_duration_ms": 90.0,
        "q_duration_ms": 20.0,
        "r_duration_ms": 30.0,
        "s_duration_ms": 25.0,
        "r_prime_duration_ms": 15.0,
        "s_prime_duration_ms": 10.0,
        "t_onset_ms": 320.0,
        "p_positive_duration_ms": 50.0,
        "vat_ms": 35.0,
        "p_positive_amp_mv": 0.12,
        "p_negative_amp_mv": -0.04,
        "qrs_peak_to_peak_mv": 1.1,
        "q_amp_mv": -0.05,
        "r_amp_mv": 0.8,
        "s_amp_mv": -0.3,
        "r_prime_amp_mv": 0.2,
        "s_prime_amp_mv": -0.1,
        "st_amp_mv": 0.02,
        "st_2_8_amp_mv": 0.03,
        "st_3_8_amp_mv": 0.04,
        "t_positive_amp_mv": 0.25,
        "t_negative_amp_mv": -0.08,
        "qrs_area_matrix": 250.0,
        "t_morphology": 2,
        "r_wave_notch_count": 1,
        "delta_confidence_pct": 80.0,
        "st_slope_deg": 12.0,
        "qt_ms": 420.0,
        "qrs_notch_slur_amp_mv": 0.15,
        "pr_amp_mv": -0.02,
        "st_adjusted_amp_mv": 0.04,
    }


def _context():
    features = make_features(PatientMeta(age=59, sex="Female"))
    for lead in STANDARD_12_LEADS:
        features.representative_leads[lead].params["glasgow_measurements"] = _complete_profile()
    return build_context(features)


def test_measurement_matrix_always_contains_all_34_guide_rows() -> None:
    matrix = build_measurement_matrix(_context())

    assert [row["row"] for row in matrix["rows"]] == EXPECTED_MATRIX_ROWS
    assert len(matrix["rows"]) == 34
    assert all(set(row) >= {"row", "content", "unit", "cells"} for row in matrix["rows"])
    assert all(list(row["cells"]) == STANDARD_12_LEADS for row in matrix["rows"])


def test_amplitude_rows_convert_mv_to_uv() -> None:
    matrix = build_measurement_matrix(_context())
    p_positive = next(row for row in matrix["rows"] if row["row"] == 19)

    assert p_positive["unit"] == "uV"
    assert p_positive["cells"]["II"]["value"] == 120.0


def test_unavailable_cell_is_not_silently_dropped() -> None:
    context = _context()
    context.leads["V1"].measurements["t_morphology"] = None
    matrix = build_measurement_matrix(context)
    t_row = next(row for row in matrix["rows"] if row["row"] == 39)

    assert t_row["cells"]["V1"]["status"] == "unavailable"
    assert t_row["cells"]["V1"]["reason"] == "t_morphology_unavailable"

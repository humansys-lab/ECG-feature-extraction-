from __future__ import annotations

from types import SimpleNamespace

from evaluate_ludb_cse import (
    record_p_duration_ms,
    summarize_rows,
    trim_largest_deviations,
)


def test_trim_largest_deviations_uses_distance_from_initial_mean() -> None:
    values = [("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 100.0)]

    kept, removed = trim_largest_deviations(values, outlier_count=1)

    assert removed == [("d", 100.0)]
    assert kept == [("a", 0.0), ("b", 1.0), ("c", 2.0)]


def test_record_p_duration_prefers_representative_consensus() -> None:
    result = SimpleNamespace(
        representative_leads={
            "I": SimpleNamespace(params={"p_dur_consensus_ms": 82.0}),
            "II": SimpleNamespace(params={"p_dur_consensus_ms": 86.0}),
        },
        beat_features=[
            SimpleNamespace(p_dur_ms=120.0, flags=[]),
        ],
    )

    value, source = record_p_duration_ms(result)

    assert value == 84.0
    assert source == "representative_multilead_consensus"


def test_record_p_duration_prefers_formal_global_measurement() -> None:
    result = SimpleNamespace(
        global_features=SimpleNamespace(
            p_duration_ms=91.0,
            p_duration_source="reliable_lead_median_fallback",
        ),
        representative_leads={
            "II": SimpleNamespace(params={"p_dur_consensus_ms": 84.0}),
        },
        beat_features=[],
    )

    value, source = record_p_duration_ms(result)

    assert value == 91.0
    assert source == "reliable_lead_median_fallback"


def test_summary_checks_mean_sd_and_coverage_separately() -> None:
    rows = []
    for index, difference in enumerate([0.0, 2.0, -2.0, 4.0]):
        row = {"record_id": str(index + 1), "error": None}
        for metric in (
            "p_duration",
            "pr_interval",
            "qrs_duration",
            "qt_interval",
        ):
            row[f"algorithm_{metric}_ms"] = 100.0 + difference
            row[f"reference_{metric}_ms"] = 100.0
            row[f"diff_{metric}_ms"] = difference
        rows.append(row)

    rows[0]["algorithm_pr_interval_ms"] = None
    rows[0]["diff_pr_interval_ms"] = None
    summary = summarize_rows(rows, outlier_count=0)

    assert summary["metrics"]["p_duration"]["numeric_pass"] is True
    assert summary["metrics"]["p_duration"]["complete_on_reference_available"] is True
    assert summary["metrics"]["pr_interval"]["numeric_pass"] is True
    assert summary["metrics"]["pr_interval"]["complete_on_reference_available"] is False
    assert summary["all_metrics_complete"] is False
    assert summary["proxy_verdict"] == "meets_numeric_limits_with_incomplete_coverage"


def test_failed_requested_record_makes_operational_coverage_incomplete() -> None:
    complete = {"record_id": "ok", "error": None}
    for metric in (
        "p_duration",
        "pr_interval",
        "qrs_duration",
        "qt_interval",
    ):
        complete[f"algorithm_{metric}_ms"] = 100.0
        complete[f"reference_{metric}_ms"] = 100.0
        complete[f"diff_{metric}_ms"] = 0.0

    summary = summarize_rows(
        [complete, {"record_id": "failed", "error": "extractor error"}],
        outlier_count=0,
    )

    assert summary["all_metrics_numeric_pass"] is True
    assert summary["measurement_coverage_complete"] is True
    assert summary["operational_coverage_complete"] is False
    assert summary["all_metrics_complete"] is False
    assert summary["proxy_verdict"] == "meets_numeric_limits_with_incomplete_coverage"

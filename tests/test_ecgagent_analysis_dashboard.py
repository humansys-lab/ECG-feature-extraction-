from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from ecgagent import analysis_dashboard as dashboard


def _metrics(rows):
    """Expand shorthand ``(key, scope, source, tp, fp, fn)`` tuples."""
    expanded = []
    for key, scope, source, tp, fp, fn in rows:
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        expanded.append(
            {
                "source": source,
                "category": key,
                "category_zh": f"{key}-中文",
                "semantic_scope": scope,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": 10,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return expanded


SAMPLE_METRICS = _metrics(
    [
        ("atrial_fibrillation", "direct", "baseline", 3, 3, 3),
        ("atrial_fibrillation", "direct", "agent", 2, 0, 4),
        ("t_wave_abnormality", "broad", "baseline", 0, 6, 7),
        ("t_wave_abnormality", "broad", "agent", 0, 3, 7),
        ("complete_av_block", "direct", "baseline", 0, 0, 0),
        ("complete_av_block", "direct", "agent", 0, 0, 0),
    ]
)

SAMPLE_SUMMARY = {
    "analysis_status": "complete",
    "record_count": 10,
    "execution": {"verified": 9, "error": 1},
    "label_evaluated_records": 9,
    "total_tool_attempts": 42,
    "total_revisions": 3,
    "verified_revision_histogram": {"0": 7, "1": 2},
    "error_category_counts": {"deepseek_insufficient_balance": 1},
    "label_effect_record_counts": {"improved": 2, "unchanged": 6, "worsened": 1},
    "runtime_seconds_by_outcome": {
        "verified": {"median": 200.0, "p90": 300.0, "max": 400.0}
    },
    "direct_label_agreement": {
        "baseline": {"tp": 3, "fp": 3, "fn": 3, "precision": 0.5,
                     "recall": 0.5, "f1": 0.5},
        "agent": {"tp": 2, "fp": 0, "fn": 4, "precision": 1.0,
                  "recall": 0.3333333333333333, "f1": 0.5},
        "delta_f1": 0.0,
    },
}

SAMPLE_RECORDS = [
    {
        "record": f"0000{index}_hr",
        "outcome": "verified",
        "runtime_seconds": 120.0 + index * 30,
        "label_effect": effect,
        "reference_categories": "atrial_fibrillation",
        "baseline_categories": "",
        "agent_categories": "atrial_fibrillation",
    }
    for index, effect in enumerate(
        ["improved", "improved", "unchanged", "worsened"]
    )
]


def _svgs(html):
    return re.findall(r"<svg\b.*?</svg>", html, flags=re.S)


def test_dashboard_is_self_contained_and_well_formed():
    html = dashboard._build_html(SAMPLE_SUMMARY, SAMPLE_METRICS,
                                 record_rows=SAMPLE_RECORDS)

    assert html.startswith("<!doctype html>")
    assert "<script" in html and "src=" not in html
    fetchable = html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "http://" not in fetchable and "https://" not in fetchable

    svgs = _svgs(html)
    assert svgs, "expected inline SVG charts"
    for source in svgs:
        ET.fromstring(source)  # raises if a chart emitted malformed XML


def test_dashboard_renders_both_sources_for_every_active_family():
    html = dashboard._build_html(SAMPLE_SUMMARY, SAMPLE_METRICS,
                                 record_rows=SAMPLE_RECORDS)

    for label, source, tp, fp, fn in (
        ("Atrial Fibrillation", "Rule baseline", 3, 3, 3),
        ("Atrial Fibrillation", "ECGAgent", 2, 0, 4),
        ("T Wave Abnormality", "ECGAgent", 0, 3, 7),
    ):
        assert (
            f"{label} · {source} · True positive TP {tp} | False positive FP {fp} | False negative FN {fn}"
            in html
        )


def test_families_without_evidence_are_named_rather_than_drawn():
    html = dashboard._build_html(SAMPLE_SUMMARY, SAMPLE_METRICS,
                                 record_rows=SAMPLE_RECORDS)

    assert "Not plotted: 1 families" in html
    assert "Complete Av Block" in html
    # A row group spans both facets, so two active families give two rows in
    # the detection chart and two more in the F1 dumbbell.
    assert html.count('<g class="row" data-scope=') == 4


def test_zero_hit_family_plots_as_f1_zero_not_as_missing():
    """``_metric_rows`` reports F1 None both for 'no data' and 'nothing right'."""
    scored = {"tp": 0, "fp": 3, "fn": 7, "f1": None}
    empty = {"tp": 0, "fp": 0, "fn": 0, "f1": None}

    assert dashboard._plot_f1(scored) == 0.0
    assert dashboard._plot_f1(empty) is None


def test_scope_filter_offers_only_the_scopes_present():
    html = dashboard._build_html(SAMPLE_SUMMARY, SAMPLE_METRICS,
                                 record_rows=SAMPLE_RECORDS)

    scopes = set(re.findall(r'<button data-scope="(\w+)"', html))
    assert scopes == {"all", "direct", "broad"}
    assert 'data-scope="screening"' not in html


def test_every_focusable_mark_carries_an_accessible_name():
    html = dashboard._build_html(SAMPLE_SUMMARY, SAMPLE_METRICS,
                                 record_rows=SAMPLE_RECORDS)

    for tag in re.findall(r"<(?:g|path|rect|circle)\b[^>]*tabindex[^>]*>", html):
        assert "aria-label=" in tag, tag


def test_empty_analysis_degrades_without_raising():
    summary = {
        "record_count": 0,
        "execution": {},
        "label_evaluated_records": 0,
        "direct_label_agreement": {"baseline": {}, "agent": {}},
    }
    html = dashboard._build_html(summary, _metrics([]), record_rows=[])

    assert "No comparable diagnostic family" in html
    assert "Not plotted" not in html


def test_write_dashboard_round_trips_through_an_output_dir(tmp_path):
    import csv
    import json

    (tmp_path / "analysis.json").write_text(
        json.dumps(SAMPLE_SUMMARY, ensure_ascii=False), encoding="utf-8"
    )
    with (tmp_path / "category_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_METRICS[0]))
        writer.writeheader()
        writer.writerows(SAMPLE_METRICS)

    path = dashboard.build_from_output_dir(tmp_path)

    assert path == tmp_path / "RESULTS.html"
    html = path.read_text(encoding="utf-8")
    assert "Atrial Fibrillation" in html
    for source in _svgs(html):
        ET.fromstring(source)


@pytest.mark.parametrize(
    "value, reference, expected_direction",
    [(0.6, 0.5, "up"), (0.4, 0.5, "down"), (0.5, 0.5, "flat"),
     (None, 0.5, "flat")],
)
def test_delta_direction(value, reference, expected_direction):
    _, direction, _ = dashboard._delta(value, reference)
    assert direction == expected_direction

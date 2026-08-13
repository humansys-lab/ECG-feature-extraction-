from __future__ import annotations

from pathlib import Path

import numpy as np

import demo_feature_extraction as demo
from feature_extraction.ecgfeat.glasgow_rules.engine import analyze_glasgow
from feature_extraction.ecgfeat.glasgow_rules.render import build_display_model
from feature_extraction.ecgfeat.models import PatientMeta
from tests.glasgow_test_helpers import make_features


def _analysis():
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.global_features.heart_rate_bpm = 110.0
    features.global_features.qt_ms = 350.0
    return features, analyze_glasgow(features).to_dict()


def test_display_model_uses_only_resolved_glasgow_primary_statements() -> None:
    _features, analysis = _analysis()

    model = build_display_model(analysis)

    assert model["summary_code"]["label"] == "Normal ECG except for rate"
    assert [row["rule_id"] for row in model["final_statements"]] == ["GAN-05.01-01"]
    assert len(model["audit_rows"]) == 67
    assert len(model["measurement_matrix"]["rows"]) == 34


def test_text_report_ignores_stale_glasgow_analysis(tmp_path: Path) -> None:
    features, analysis = _analysis()
    features.metadata["glasgow_analysis"] = analysis
    out = tmp_path / "report.txt"
    hdr = {
        "fs": 500,
        "n_samples": 1000,
        "age": 59,
        "sex": "Female",
        "dx": ["426783006"],
    }

    demo.generate_report("rec001", hdr, np.zeros((12, 1000)), features, out)
    text = out.read_text(encoding="utf-8")

    assert "Dx Codes" in text
    assert "Dx Detail" in text
    assert "GLASGOW MEASUREMENT MATRIX" not in text
    assert "GLASGOW INTERPRETIVE STATEMENTS" not in text
    assert "SUMMARY CODE" not in text
    assert "RULE AUDIT APPENDIX" not in text
    assert "GAN-05.01-01" not in text
    assert "not_matched" not in text
    assert "CLINICAL INTERPRETATION  (DXL thresholds" not in text


def test_image_summary_ignores_stale_glasgow_analysis() -> None:
    features, analysis = _analysis()
    features.metadata["glasgow_analysis"] = analysis
    features.interpretation = None
    summary = demo.build_report_summary(
        "rec001",
        {"fs": 500, "n_samples": 1000, "age": 59, "sex": "Female", "dx": []},
        np.zeros((12, 1000)),
        features,
    )

    assert "glasgow_final_rule_ids" not in summary
    assert "glasgow_display" not in summary
    assert not any("GAN-" in line for line in summary["interpretation_lines"])

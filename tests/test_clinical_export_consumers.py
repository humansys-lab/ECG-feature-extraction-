from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.engine import analyze_clinical
from feature_extraction.ecgfeat.clinical_rules.sources import RULESET_VERSION
from feature_extraction.ecgfeat.export import clinical_fingerprint, to_dict
from feature_extraction.ecgfeat.models import (
    ECGFeatures,
    GlobalFeatures,
    LeadQuality,
    PatientMeta,
    RepresentativeLeadFeatures,
)


def _features(*, qt_ms=500.0, heart_rate=60.0) -> ECGFeatures:
    lead_params = {
        lead: {
            "q_amp_mv": 0.0,
            "q_duration_ms": 0.0,
            "r_amp_mv": 0.5,
            "s_amp_mv": -0.5,
            "st_on_mv": 0.0,
            "t_amp_mv": 0.2,
        }
        for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    }
    lead_params["II"].update({"p_amp_mv": 0.1, "p_dur_consensus_ms": 100.0})
    lead_params["V1"].update(
        {"p_terminal_amp_mv": -0.02, "p_terminal_duration_ms": 20.0}
    )
    quality = {
        lead: LeadQuality(
            lead=lead,
            baseline_wander_score=0.0,
            muscle_noise_score=0.0,
            powerline_score=0.0,
            clipping_score=0.0,
            flatline_score=0.0,
            missing=False,
            reliable=True,
        )
        for lead in lead_params
    }
    features = ECGFeatures(
        fs=500,
        quality=quality,
        beats=[],
        beat_features=[],
        representative_leads={
            lead: RepresentativeLeadFeatures(lead=lead, params=params, variance={})
            for lead, params in lead_params.items()
        },
        groups={},
        global_features=GlobalFeatures(
            heart_rate_bpm=heart_rate,
            atrial_rate_bpm=heart_rate,
            pr_ms=160.0,
            qrs_ms=90.0,
            qt_ms=qt_ms,
            qtc_bazett_ms=None,
            qtc_fridericia_ms=None,
            p_axis_deg=40.0,
            qrs_axis_deg=20.0,
            t_axis_deg=30.0,
            st_axis_deg=None,
            qt_dispersion_ms=None,
            qt_reliability="reliable",
        ),
        metadata={
            "patient_meta": PatientMeta(age=50.0, sex="female"),
            "record_quality": {"record_grade": "Q0"},
            "rhythm_analysis": {
                "atrial_residual": {"validated_qrst_subtraction": True},
                "af_afl_summary": {
                    "probable_af": False,
                    "probable_flutter": False,
                    "rr_cv": 0.02,
                    "organized_p_ratio": 0.95,
                },
            },
        },
    )
    features.interpretation = SimpleNamespace(
        rr_cv=0.02,
        precordial_reversal_suspected=False,
        limb_reversal_suspected=None,
        qtc_class="prolonged",
    )
    return features


def test_engine_populates_authoritative_analysis_without_glasgow_reference() -> None:
    features = _features()
    features.metadata["glasgow_analysis"] = {
        "statement_resolution": {
            "summary_code": {"code": 1, "label": "Normal ECG"}
        }
    }

    result = analyze_clinical(features)

    assert result.schema_version == "clinical_rules.v2"
    assert result.ruleset_version == RULESET_VERSION
    assert result.summary["resolver_policy"] == "coverage-confidence-v1"
    assert "glasgow" not in result.reference_interpretations
    assert result.reference_interpretations["dxl"]["reference_only"] is True
    assert result.overall_status == "abnormal_with_limited_coverage"
    # QTc 500 ms clears the >=480 ms markedly-prolonged tier, not just the
    # 450/460 ms entry threshold.
    assert any(
        item["statement_code"] == "markedly_prolonged_qt"
        for item in result.final_statements
    )
    assert not any(item["reference"] == "glasgow" for item in result.conflicts)


def test_engine_does_not_consume_stale_glasgow_metadata() -> None:
    features = _features(qt_ms=400.0)
    legacy = features.interpretation
    glasgow = {"statement_resolution": {"summary_code": {"code": 5, "label": "Abnormal ECG"}}}
    features.metadata["glasgow_analysis"] = glasgow

    result = analyze_clinical(features)
    features.metadata["clinical_interpretation"] = result.to_dict()

    assert features.interpretation is legacy
    assert "glasgow" not in result.reference_interpretations
    assert features.metadata["clinical_interpretation"]["summary"]["authoritative"] is True


def test_validated_flutter_marks_legacy_probable_af_as_superseded() -> None:
    features = _features(qt_ms=400.0)
    features.metadata["rhythm_analysis"] = {
        "atrial_residual": {
            "validated_qrst_subtraction": True,
            "F_wave_morphology_validated": True,
        },
        "af_afl_summary": {
            "probable_af": False,
            "probable_flutter": True,
            "af_afl_indeterminate": False,
            "F_wave_multilead_consensus": True,
            "F_wave_confidence": 0.90,
            "rr_cv": 0.20,
        },
    }
    features.interpretation.probable_af = True

    result = analyze_clinical(features)
    features.metadata["clinical_interpretation"] = result.to_dict()
    payload = to_dict(features)

    conflict = next(
        item
        for item in result.conflicts
        if item.get("reference") == "dxl"
        and item.get("reference_statement") == "probable_af"
    )
    assert conflict["authoritative_statement"] == "atrial_flutter_pattern"
    assert conflict["resolution"] == "use_authoritative_unified_statement"
    dxl_meta = payload["reference_metadata"]["interpretation"]
    assert dxl_meta["diagnostic_disposition"] == "superseded_by_authoritative_unified_rules"
    assert dxl_meta["diagnostic_conflicts"]


def test_full_export_preserves_legacy_and_adds_clinical_without_glasgow() -> None:
    features = _features(qt_ms=400.0)
    features.metadata["glasgow_analysis"] = {}
    features.metadata["clinical_interpretation"] = analyze_clinical(features).to_dict()

    payload = to_dict(features)

    assert "interpretation" in payload
    assert "glasgow" not in payload
    assert "glasgow_analysis" not in payload["metadata"]
    assert payload["clinical_interpretation"]["schema_version"] == "clinical_rules.v2"
    assert payload["clinical_interpretation"]["ruleset_version"] == RULESET_VERSION
    assert payload["clinical_interpretation"]["artifact_fingerprint"].startswith("sha256:")
    assert payload["reference_metadata"]["interpretation"]["reference_only"] is True
    assert "glasgow" not in payload["reference_metadata"]


def test_fingerprint_ignores_generation_timestamp_and_existing_fingerprint() -> None:
    analysis = analyze_clinical(_features()).to_dict()
    first = clinical_fingerprint(
        {**analysis, "generated_at": "2026-01-01T00:00:00Z", "artifact_fingerprint": "old"}
    )
    second = clinical_fingerprint(
        {**analysis, "generated_at": "2026-02-01T00:00:00Z", "artifact_fingerprint": "new"}
    )

    assert first == second


def test_medgemma_summary_ignores_stale_glasgow_payload() -> None:
    from medgemma_ecg_core import summarize_current_features

    features = _features()
    clinical = analyze_clinical(features).to_dict()
    payload = to_dict(features)
    payload["interpretation"] = {}
    payload["clinical_interpretation"] = clinical
    payload["glasgow"] = {
        "schema_version": "glasgow_rules.v2",
        "statement_resolution": {
            "summary_code": {"code": 1, "label": "Normal ECG"}
        },
    }

    text = summarize_current_features(payload)

    assert "Unified clinical summary" in text
    assert "Glasgow" not in text


def test_report_summary_shows_unified_result_without_glasgow_reference() -> None:
    import numpy as np
    import demo_feature_extraction as demo

    features = _features(qt_ms=400.0)
    clinical = analyze_clinical(features).to_dict()
    clinical["artifact_fingerprint"] = clinical_fingerprint(clinical)
    features.metadata["clinical_interpretation"] = clinical
    features.metadata["glasgow_analysis"] = {
        "schema_version": "glasgow_rules.v2",
        "statement_resolution": {
            "summary_code": {"code": 1, "label": "Normal ECG"},
            "preliminary": [],
            "final": [],
            "suppressed": [],
            "unavailable": [],
            "restricted_analysis": {"active": False, "stopped_by": []},
        },
        "measurement_matrix": {"rows": []},
        "rule_evaluations": [],
        "coverage": {},
    }

    summary = demo.build_report_summary(
        "rec", {"fs": 500, "n_samples": 1000, "dx": []}, np.zeros((12, 1000)), features
    )

    assert summary["interpretation_lines"][0].startswith("Unified clinical summary")
    assert not any("Glasgow" in line for line in summary["interpretation_lines"])
    assert any("DXL-inspired reference" in line for line in summary["interpretation_lines"])


def test_stale_clinical_fingerprint_is_visible_in_report_summary() -> None:
    import numpy as np
    import demo_feature_extraction as demo

    features = _features(qt_ms=400.0)
    clinical = analyze_clinical(features).to_dict()
    clinical["artifact_fingerprint"] = "sha256:stale"
    features.metadata["clinical_interpretation"] = clinical

    summary = demo.build_report_summary(
        "rec", {"fs": 500, "n_samples": 1000, "dx": []}, np.zeros((12, 1000)), features
    )

    assert summary["artifact_warning"].startswith("*** STALE ARTIFACT")
    assert summary["interpretation_lines"][0].startswith("*** STALE ARTIFACT")

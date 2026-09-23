from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ecgagent.evidence.diagnostic_contract import build_diagnostic_document
from ecgagent.evidence.store import EvidenceStore
from ecgagent.evidence.waveform_review import (
    MAX_P_EVENTS, STANDARD_LEADS, build_waveform_review, unavailable_waveform_review,
)
from ecgagent.tools.waveform_review import SPECS, get_waveform_review


def _synthetic(*, raw_fs=500, feature_fs=500, qs=False, st_mv=.15):
    time = np.arange(4 * raw_fs) / raw_fs
    signal = np.zeros(len(time))
    beat_features, events = [], []
    for index, center in enumerate((.8, 1.8, 2.8)):
        p_center = center - .200
        signal += .12 * np.exp(-.5 * ((time - p_center) / .020) ** 2)
        if qs:
            signal -= .8 * np.exp(-.5 * ((time - center) / .018) ** 2)
        else:
            signal += 1.0 * np.exp(-.5 * ((time - center) / .008) ** 2)
            signal -= .30 * np.exp(-.5 * ((time - center - .025) / .008) ** 2)
        signal[(time >= center + .065) & (time <= center + .180)] += st_mv
        signal += .22 * np.exp(-.5 * ((time - center - .280) / .035) ** 2)

        def wave(a, p, b):
            return dict(onset=round(a * feature_fs), peak=round(p * feature_fs), offset=round(b * feature_fs))

        beat_features.append({
            "lead": "II", "beat_id": index, "beat_measurement_reliable": True, "flags": [],
            "p": wave(p_center - .050, p_center, p_center + .050),
            "qrs": wave(center - .050, center, center + .060),
            "t": wave(center + .200, center + .280, center + .380),
            "r_amp_mv": 0.0 if qs else 1.0, "s_amp_mv": -.8 if qs else -.30,
            "st_80ms_mv": st_mv,
        })
        events.append({"sample": round(p_center * feature_fs), "time_ms": p_center * 1000,
                       "onset_ms": (p_center - .050) * 1000, "offset_ms": (p_center + .050) * 1000,
                       "source_leads": ["II"]})
    features = {
        "fs": feature_fs,
        "metadata": {"input_fs": raw_fs, "internal_fs": feature_fs,
                     "duration_sec": 4., "lead_order": ["II"],
                     "input_contract": {"amplitude_input_unit": "mV", "valid": True}},
        "quality": {"II": {"reliable": True, "missing": False,
                           "reliable_for_p": True, "reliable_for_qrs": True, "reliable_for_t": True}},
        "beat_features": beat_features, "rhythm_inputs": {"p_events": events},
    }
    return signal[None, :], features


def _build(raw, features, **kwargs):
    return build_waveform_review(raw, features["metadata"]["input_fs"], features, **kwargs)


def test_missing_raw_is_explicit_unavailable_and_feature_only_contract():
    _, features = _synthetic()
    artifact = build_waveform_review(None, 500, features)
    assert artifact["status"] == "unavailable"
    assert all(p["missing_requirements"] == ["original_waveform_missing"] for p in artifact["profiles"].values())
    store = EvidenceStore.from_dict(features).diagnostic_view()
    result = get_waveform_review(store)
    assert not result.ok
    assert result.note == "measurement_unavailable"
    assert "original_waveform_missing" in result.text
    assert all(store.try_resolve(pointer) is not None for pointer in result.citations)


@pytest.mark.parametrize("raw_fs,feature_fs", [(500, 500), (250, 500), (500, 250)])
def test_original_and_internal_timebases_map_to_same_measurement(raw_fs, feature_fs):
    raw, features = _synthetic(raw_fs=raw_fs, feature_fs=feature_fs)
    artifact = _build(raw, features)
    assert artifact["profiles"]["p_av"]["status"] == "observations_available"
    first = artifact["profiles"]["st"]["observations"][0]
    assert first["status"] == "observations_available"
    assert first["raw_st_80ms_mv"] == pytest.approx(.15, abs=.006)
    assert artifact["provenance"]["original_fs_hz"] == raw_fs
    assert artifact["provenance"]["feature_fs_hz"] == feature_fs
    assert first["raw_sample"] == round(.94 * raw_fs)


@pytest.mark.parametrize("mutation,reason", [
    (lambda f: f.pop("fs"), "sampling_rate_missing_or_invalid"),
    (lambda f: f["metadata"].update(internal_fs=1000), "timebase_conflict"),
    (lambda f: f["metadata"].update(duration_sec=8), "record_duration_conflict"),
    (lambda f: f["metadata"].pop("lead_order"), "lead_mapping_missing_or_invalid"),
    (lambda f: f["metadata"]["input_contract"].pop("amplitude_input_unit"), "amplitude_units_missing_or_invalid"),
    (lambda f: f["metadata"]["input_contract"].update(amplitude_input_unit="ADC"), "amplitude_units_missing_or_invalid"),
])
def test_missing_or_conflicting_acquisition_metadata_never_supports_review(mutation, reason):
    raw, features = _synthetic()
    mutation(features)
    artifact = _build(raw, features)
    assert artifact["status"] == "unavailable"
    assert artifact["profiles"]["qrs"]["missing_requirements"] == [reason]


@pytest.mark.parametrize("bad_quality", [{}, {"missing": False}, {"reliable": True}, {"reliable": False, "missing": False}])
def test_missing_or_invalid_quality_never_supports_review(bad_quality):
    raw, features = _synthetic()
    features["quality"]["II"] = bad_quality
    assert _build(raw, features)["status"] == "unavailable"


@pytest.mark.parametrize("event_time,reason", [(800., "qrs_overlap"), (1080., "t_overlap")])
def test_p_candidates_overlapping_ventricular_morphology_conflict(event_time, reason):
    raw, features = _synthetic()
    features["rhythm_inputs"]["p_events"] = [{"time_ms": event_time, "source_leads": ["II"]}]
    profile = _build(raw, features)["profiles"]["p_av"]
    assert profile["status"] == "conflict"
    assert reason in profile["conflict_reasons"]


def test_p_candidates_need_local_raw_morphology_not_just_detector_confidence():
    raw, features = _synthetic()
    raw[:, 250:350] = 0.
    profile = _build(raw, features)["profiles"]["p_av"]
    assert profile["observations"][0]["status"] == "conflict"
    assert "local_p_morphology_not_supported" in profile["conflict_reasons"]


def test_event_sample_and_time_ms_disagreement_conflicts():
    raw, features = _synthetic()
    features["rhythm_inputs"]["p_events"][0]["sample"] += 30
    row = _build(raw, features)["profiles"]["p_av"]["observations"][0]
    assert row["conflict_reasons"] == ["event_timebase_conflict"]


def test_qs_sign_conflict_challenges_exported_positive_r():
    raw, features = _synthetic(qs=True)
    for row in features["beat_features"]:
        row["r_amp_mv"] = .8
        row["s_amp_mv"] = .2
    profile = _build(raw, features)["profiles"]["qrs"]
    assert profile["status"] == "conflict"
    assert {"signed_rs_conflict", "qs_positive_r_conflict"} <= set(profile["conflict_reasons"])
    assert profile["observations"][0]["raw_min_mv"] < -.75
    assert profile["observations"][0]["raw_qs_candidate"] is True


def test_raw_st_known_amplitude_challenges_upstream_baseline_and_no_extrapolation():
    raw, features = _synthetic(st_mv=.20)
    for row in features["beat_features"]:
        row["st_80ms_mv"] = -.05
    artifact = _build(raw, features)
    rows = artifact["profiles"]["st"]["observations"]
    assert rows[0]["raw_st_80ms_mv"] == pytest.approx(.20, abs=.004)
    assert rows[0]["status"] == "conflict"
    assert rows[0]["conflict_reasons"] == ["st_amplitude_disagreement"]
    assert rows[-1]["status"] == "unavailable"
    assert rows[-1]["missing_requirements"] == ["bracketing_pr_anchors_missing"]


def test_st_without_valid_pr_anchors_and_nan_channel_is_unavailable():
    raw, features = _synthetic()
    for row in features["beat_features"]:
        row["p"]["offset"] = None
    assert _build(raw, features)["profiles"]["st"]["status"] == "unavailable"
    raw[0, 200] = np.nan
    assert _build(raw, features)["status"] == "unavailable"


def test_physical_unit_scaling_purity_bounded_output_and_no_experimental_detector(monkeypatch):
    raw, features = _synthetic()
    features["rhythm_inputs"]["p_events"] *= 20
    before = copy.deepcopy(features)
    original = raw.copy()
    import ecgfeat.adaptive_qrs

    def forbidden(*args, **kwargs):
        pytest.fail("candidate detector must not be enabled by waveform review")

    monkeypatch.setattr(ecgfeat.adaptive_qrs, "channel_candidates", forbidden)
    artifact = _build(raw * 1000, features, amplitude_unit="uV")
    assert artifact["profiles"]["p_av"]["reviewed_count"] == MAX_P_EVENTS
    assert artifact["profiles"]["p_av"]["truncated"] is True
    assert artifact["profiles"]["st"]["observations"][0]["raw_st_80ms_mv"] == pytest.approx(.15, abs=.004)
    assert features == before
    np.testing.assert_array_equal(raw, original)
    assert artifact["provenance"]["independent_acquisition"] is False
    assert artifact["provenance"]["clinical_validation"] is False
    json.dumps(artifact, allow_nan=False)


def test_review_artifact_strict_allowlist_removes_diagnoses_and_patient_text():
    raw, features = _synthetic()
    artifact = _build(raw, features)
    artifact["patient_name"] = "Secret Patient"
    artifact["diagnosis"] = "Secret Diagnosis"
    artifact["provenance"]["source_diagnostic_text"] = "Secret Diagnosis"
    row = artifact["profiles"]["p_av"]["observations"][0]
    row.update(patient_id="Secret Patient", support="Secret Diagnosis", raw_prominence_mv="Secret Diagnosis")
    row["missing_requirements"] = ["Secret Diagnosis"]
    features["waveform_review"] = artifact
    document, audit = build_diagnostic_document(features)
    serialized = json.dumps(document["waveform_review"])
    assert "Secret" not in serialized
    assert any("/support" in path for path in audit.dropped_sensitive_fields)
    assert any("/raw_prominence_mv" in path for path in audit.dropped_sensitive_fields)
    assert "Secret" not in get_waveform_review(EvidenceStore.from_dict(features)).render()


def test_unavailable_tool_in_full_store_has_no_unresolvable_virtual_citations():
    result = get_waveform_review(EvidenceStore.from_dict({}))
    assert not result.ok
    assert result.note == "measurement_unavailable"
    assert result.citations == ()


def test_invalid_saved_provenance_cannot_be_replaced_with_successful_defaults():
    raw, features = _synthetic()
    artifact = _build(raw, features)
    artifact["provenance"]["source"] = "Secret Diagnostic Text"
    features["waveform_review"] = artifact
    document, _ = build_diagnostic_document(features)
    assert document["waveform_review"]["status"] == "unavailable"
    assert "Secret" not in json.dumps(document["waveform_review"])
    result = get_waveform_review(EvidenceStore.from_dict(features))
    assert result.note == "measurement_unavailable"
    assert all(EvidenceStore.from_dict(features).try_resolve(p) is not None for p in result.citations)


def test_qrs_can_challenge_qs_without_p_baseline_but_st_cannot():
    raw, features = _synthetic(qs=True)
    for row in features["beat_features"]:
        row["p"] = {}
        row["r_amp_mv"] = .8
    artifact = _build(raw, features)
    assert artifact["profiles"]["qrs"]["status"] == "conflict"
    assert artifact["profiles"]["qrs"]["observations"][0]["baseline_method"] == "quiet_pre_qrs"
    assert artifact["profiles"]["st"]["status"] == "unavailable"


def test_missing_candidate_bounds_or_ventricular_timing_stays_unavailable():
    raw, features = _synthetic()
    features["rhythm_inputs"]["p_events"][0].pop("onset_ms")
    row = _build(raw, features)["profiles"]["p_av"]["observations"][0]
    assert row["status"] == "unavailable"
    assert "candidate_boundaries_missing_or_invalid" in row["missing_requirements"]
    features["beat_features"][0]["t"] = {}
    row = _build(raw, features)["profiles"]["p_av"]["observations"][1]
    assert row["status"] == "unavailable"
    assert "ventricular_timing_missing" in row["missing_requirements"]


def test_record_edge_candidate_is_unavailable_not_a_morphology_rejection():
    raw, features = _synthetic()
    features["rhythm_inputs"]["p_events"] = [{"time_ms": 20.}]
    row = _build(raw, features)["profiles"]["p_av"]["observations"][0]
    assert row["status"] == "unavailable"
    assert row["conflict_reasons"] == []
    assert row["missing_requirements"] == ["local_window_incomplete"]


def test_st_repaired_j_uses_exported_sample_and_rejects_invalid_timing():
    raw, features = _synthetic()
    features["beat_features"][0]["st_j_remeasured_index"] = 440
    row = _build(raw, features)["profiles"]["st"]["observations"][0]
    assert row["status"] == "observations_available"
    assert row["raw_j_sample"] == 440
    assert row["raw_sample"] == 480
    features["beat_features"][0]["st_j_remeasured_index"] = 999999
    row = _build(raw, features)["profiles"]["st"]["observations"][0]
    assert row["status"] == "unavailable"
    assert row["missing_requirements"] == ["st_anchor_timing_invalid"]


def test_no_diagnostic_labels_from_source_are_copied_into_built_artifact():
    raw, features = _synthetic()
    features["clinical_interpretation"] = {"diagnosis": "Secret Diagnosis"}
    features["metadata"]["patient_name"] = "Secret Patient"
    for event in features["rhythm_inputs"]["p_events"]:
        event.update(morphology="Secret Diagnosis", source="Secret Source")
    artifact = _build(raw, features)
    assert "Secret" not in json.dumps(artifact)


def test_bad_local_st_quality_never_yields_an_amplitude():
    raw, features = _synthetic()
    raw[0, 466:475] += np.tile([-.2, .2], 5)[:9]
    row = _build(raw, features)["profiles"]["st"]["observations"][0]
    assert row["status"] == "unavailable"
    assert "raw_st_80ms_mv" not in row
    assert row["missing_requirements"] == ["st_local_noise_excessive"]


def test_tool_specs_and_atom_citations_remain_resolvable():
    raw, features = _synthetic()
    features["waveform_review"] = _build(raw, features)
    store = EvidenceStore.from_dict(features).diagnostic_view()
    result = get_waveform_review(store, "st")
    assert result.ok
    assert SPECS[0].name == "get_waveform_review"
    assert result.citations
    for pointer in result.citations:
        assert pointer.startswith("/waveform_review/")
        assert store.try_resolve(pointer) is not None
        assert f"ev:{pointer}" in result.text
    assert not get_waveform_review(store, "diagnosis").ok
    all_result = get_waveform_review(store)
    assert all_result.truncated
    assert "showing 2 of 3 observations" in all_result.text
    assert "p_av/observations/2/" not in all_result.text


def test_cli_feature_only_retains_unavailable_and_explicit_record_computes(tmp_path, monkeypatch, capsys):
    from ecgagent import batch, cli

    raw, features = _synthetic()
    path = tmp_path / "features.json"
    path.write_text(json.dumps(features))
    captured = []

    def registry(store, **kwargs):
        captured.append(store)
        return SimpleNamespace(anthropic_tools=lambda: [])

    monkeypatch.setattr(cli, "build_default_registry", registry)
    assert cli.main(["--features", str(path), "--schema", "anthropic"]) == 0
    assert captured[-1].document["waveform_review"]["status"] == "unavailable"
    monkeypatch.setattr(batch, "_load_wfdb_record", lambda path: (np.repeat(raw, 12, axis=0), 500.))
    # Explicit record loader returns standard lead order; keep II metadata for
    # its source features while all original channels are available for review.
    assert cli.main(["--features", str(path), "--schema", "anthropic", "--waveform-record", "record.hea"]) == 0
    assert captured[-1].document["waveform_review"]["status"] == "observations_available"
    assert "waveform_review" not in json.loads(path.read_text())


def test_batch_loader_checks_and_normalizes_physical_units(monkeypatch):
    from ecgagent import batch

    record = SimpleNamespace(sig_name=list(STANDARD_LEADS), units=["uV"] * 12,
                             p_signal=np.ones((100, 12)) * 1000, fs=500)
    monkeypatch.setitem(sys.modules, "wfdb", SimpleNamespace(rdrecord=lambda path: record))
    raw, fs = batch._load_wfdb_record("unused")
    np.testing.assert_array_equal(raw, np.ones((12, 100)))
    assert fs == 500
    record.units = None
    with pytest.raises(ValueError, match="units"):
        batch._load_wfdb_record("unused")


def test_batch_extraction_attaches_review_to_export_without_changing_extractor_config(tmp_path, monkeypatch):
    from ecgagent import batch
    import ecgfeat.api
    import ecgfeat.export

    raw, features = _synthetic()
    options = {}

    class Extractor:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def extract(self, *args, **kwargs):
            return features

    monkeypatch.setattr(ecgfeat.api, "ECGFeatureExtractor", Extractor)
    monkeypatch.setattr(ecgfeat.export, "to_dict", lambda f: copy.deepcopy(f))
    monkeypatch.setattr(batch, "_load_wfdb_record", lambda path: (np.repeat(raw, 12, axis=0), 500.))
    monkeypatch.setattr(batch, "_extraction_provenance", lambda *args: {"schema_version": "test"})
    result = batch._extract_one(({"record": "test", "record_path": "unused"}, str(tmp_path), 500))
    assert result["status"] == "ok"
    payload = json.loads((tmp_path / "features" / "test_features.json").read_text())
    assert payload["waveform_review"]["status"] == "observations_available"
    assert options == {"fs_internal": 500, "mains_freq": 50}

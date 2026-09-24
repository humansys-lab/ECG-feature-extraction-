"""Regressions for errors found by the library-design implementation audit."""
from dataclasses import asdict
import hashlib
import io
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.io import savemat

from feature_extraction.ecgfeat import ECGConfig, RefinementConfig, ecg_prepare, ecg_emit, ecg_measure
from feature_extraction.ecgfeat.config import config_provenance
from feature_extraction.ecgfeat.errors import ConfigurationError, SignalShapeError
from feature_extraction.ecgfeat.io import read_wfdb, read_wfdb_header
from feature_extraction.ecgfeat.pipeline.extractor import ECGMeasurements
from feature_extraction.ecgfeat.record import query_measurement
from feature_extraction.ecgfeat.record import dumps_record, loads_record, record_from_document, validate_record
from feature_extraction.ecgfeat.errors import RecordValidationError, AddressSyntaxError, AddressNotFoundError, MeasurementSelectorError


def measured_fixture(*, fs=250, internal_fs=500, unit="mV", source="analysis", name="sensor"):
    prepared = ecg_prepare(np.ones((1, fs * 2)), sampling_rate=fs, lead_names=[name], amplitude_unit=unit, input_mode="limited")
    config = ECGConfig(input_mode="limited", fs_internal=internal_fs, st_amplitude_source=source)
    bf = NS(lead="II", beat_id=0, p=NS(onset=260, offset=300), qrs=NS(onset=380, peak=400, offset=430),
            t=NS(offset=600), r_peak_index=402, j_index=430, p_dur_ms=80, pr_ms=240, qrs_ms=100,
            qt_ms=440, tpe_ms=70, p_amp_mv=.1, r_amp_mv=1.2, s_amp_mv=-.2, st_80ms_mv=None,
            t_amp_mv=.3, qrs_signed_area=5.0, jt_ms=340, u_amp_signed_mv=None)
    features = NS(fs=internal_fs, beats=[NS(beat_id=0, r_index=400)], beat_features=[bf],
                  global_features=NS(heart_rate_bpm=60, qrs_axis_deg=40, qtc_bazett_ms=440, qrs_wide_ms=110),
                  quality={"II": NS(reliable=True)}, metadata={"record_quality": {"record_grade": "Q0"},
                  "input_contract": {"lead_slot_map": {name: "II"}}, "mains_frequency_hz": 50})
    return ECGMeasurements(prepared, features, config, config_provenance(config))


def test_record_import_is_independent_of_numerical_and_plot_dependencies():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "import sys; from feature_extraction.ecgfeat.record import loads_record; assert not {'numpy','scipy','matplotlib'} & sys.modules.keys()"], check=True)


@pytest.mark.parametrize("profile", ["summary", "all", "debug"])
def test_profiles_roundtrip_losslessly_and_preserve_unknown_members(profile):
    record = ecg_emit(measured_fixture(), profile=profile)
    document = record.as_dict()
    document["future_extension"] = {"list": [1, 2, None]}
    document["measurements"]["intervals"]["future_interval_ms"] = document["measurements"]["intervals"]["qt_interval_ms"].copy()
    document["metadata"] = {}
    document["extensions"] = {}
    record = record_from_document(document)
    restored = loads_record(dumps_record(record))
    assert restored.as_dict() == document
    assert restored.as_dict()["intended_use"] == "research_and_engineering_only"
    document["future_extension"]["list"].append(3)
    assert restored.as_dict()["future_extension"]["list"] == [1, 2, None]


def test_profile_promotion_cannot_fabricate_missing_measurements():
    record = ecg_emit(measured_fixture(), profile="summary")
    for profile in ("all", "debug"):
        with pytest.raises(RecordValidationError):
            dumps_record(record, profile=profile)


@pytest.mark.parametrize("failure", ["shape", "unit", "integer", "nullable", "sample", "absence", "coordinate", "duplicate_absence", "provenance", "evidence", "lead_axis", "duplicate_beat", "version", "object", "record_id"])
def test_strict_loading_rejects_contract_violations(failure):
    document = ecg_emit(measured_fixture()).as_dict()
    field = document["delineation"]["fiducials"]["r_peak"]
    if failure == "shape": field["values"] = [[201, 202]]
    elif failure == "unit": field["unit"] = "ms"
    elif failure == "integer": field["values"] = [[True]]
    elif failure == "nullable": field.update(values=[[None]], nullable=False)
    elif failure == "sample": field["values"] = [[500]]
    elif failure == "absence": field["absence"] = {"kind": "unmeasurable", "reason": "bad"}
    elif failure == "coordinate": field["absence"] = {"unmeasurable": {"b9999|sensor": "bad"}}
    elif failure == "duplicate_absence": field.update(values=[[None]], absence={"unmeasurable": {"b0001|sensor": "a"}, "not_applicable": {"b0001|sensor": "b"}})
    elif failure == "provenance": field["provenance_ref"] = "/provenance/missing"
    elif failure == "evidence": field["validation"] = {"status": "benchmark_validated", "evidence": []}
    elif failure == "lead_axis": document["acquisition"]["leads"] = ["other"]
    elif failure == "duplicate_beat": document["axes"]["beats"] *= 2
    elif failure == "version": document["schema_version"] = "2.0.0"
    elif failure == "object": document["quality"] = []
    elif failure == "record_id": document["record_id"] = 12
    with pytest.raises(RecordValidationError):
        loads_record(json.dumps(document))


def test_duplicate_json_members_are_not_silently_overwritten():
    with pytest.raises(RecordValidationError, match="duplicate"):
        loads_record('{"schema_version":"1.0.0","schema_version":"2.0.0"}')


def test_pointer_and_selector_validation():
    from feature_extraction.ecgfeat.record import resolve_pointer, make_record_address
    record = ecg_emit(measured_fixture())
    with pytest.raises(AddressNotFoundError):
        resolve_pointer(record, "/axes/leads/00")
    with pytest.raises(AddressSyntaxError):
        make_record_address(record, "/bad~2escape")
    with pytest.raises(MeasurementSelectorError):
        query_measurement(record, "r_peak", lead="sensor", beat=True)


def test_packaged_schema_matches_source_and_emitted_records():
    from pathlib import Path
    import jsonschema
    source = Path("schemas/ecg-record/1.0/schema.json").read_bytes()
    packaged = Path("feature_extraction/ecgfeat/schemas/ecg-record/1.0/schema.json").read_bytes()
    assert source == packaged
    schema = json.loads(source)
    jsonschema.Draft202012Validator.check_schema(schema)
    for profile in ("summary", "all", "debug"):
        jsonschema.validate(ecg_emit(measured_fixture(), profile=profile).as_dict(), schema)


def test_summary_reference_shape_size_budget():
    from copy import deepcopy
    from feature_extraction.ecgfeat.config import STANDARD_12_LEADS
    measurements = measured_fixture(fs=500)
    prepared = ecg_prepare(np.zeros((12, 5000)), sampling_rate=500, lead_names=STANDARD_12_LEADS)
    features = measurements.legacy_features
    prototype = features.beat_features[0]
    features.beats = [NS(beat_id=i, r_index=200 + i * 450) for i in range(10)]
    features.beat_features = []
    for i in range(10):
        for lead in STANDARD_12_LEADS:
            row = deepcopy(prototype)
            row.beat_id, row.lead = i, lead
            row.st_80ms_mv = .015
            for wave in (row.p, row.qrs, row.t):
                for key, value in vars(wave).items():
                    setattr(wave, key, value + i * 400)
            row.r_peak_index += i * 400
            row.j_index += i * 400
            features.beat_features.append(row)
    features.metadata["input_contract"] = {}
    config = ECGConfig()
    record = ecg_emit(ECGMeasurements(prepared, features, config, config_provenance(config)))
    assert len(record.axes["beats"]) == 10 and len(record.axes["leads"]) == 12
    assert len(dumps_record(record)) <= 24_000


def test_cli_validation_stdin_selection_and_error_codes(tmp_path, capsys, monkeypatch):
    from feature_extraction.ecgfeat.cli import main
    record = ecg_emit(measured_fixture())
    path = tmp_path / "record.json"
    path.write_bytes(dumps_record(record))
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([{"name": "r_peak", "lead": "sensor", "beat": 0}]))
    assert main(["select", str(path), "--queries", str(queries)]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["value"] == 201
    monkeypatch.setattr("sys.stdin", io.StringIO(dumps_record(record).decode()))
    assert main(["validate", "-"]) == 0
    capsys.readouterr()
    assert main(["query", str(path), "missing"]) == 3
    assert capsys.readouterr().out == ""
    assert main(["validate", str(tmp_path / "missing.json")]) == 5
    assert main(["batch", str(path), "--output-dir", str(tmp_path), "--jobs", "0"]) == 2


def test_cli_config_rejects_unknown_and_unwired_controls(tmp_path):
    from feature_extraction.ecgfeat.cli import _config_from_file
    path = tmp_path / "config.json"
    for data in ({"typo": True}, {"p_wave": {"enabled": False}}):
        path.write_text(json.dumps(data))
        with pytest.raises(ConfigurationError):
            _config_from_file(path)
    path.write_text(json.dumps({"refinement": {"t_wave_boundaries": True}}))
    # Actual flag names are enforced instead of silently ignoring options.
    with pytest.raises(ConfigurationError):
        _config_from_file(path)


def test_batch_continues_after_bad_json_and_preserves_order(tmp_path, capsys):
    from feature_extraction.ecgfeat.cli import cli_batch
    np.save(tmp_path / "signal.npy", np.zeros((1, 500)))
    manifest = tmp_path / "manifest.jsonl"
    good = {"id": "good", "signal": "signal.npy", "sampling_rate": 500, "lead_names": ["sensor"], "input_mode": "limited"}
    manifest.write_text("{bad json\n" + json.dumps(good) + "\n" + json.dumps({**good, "id": "escape", "output": "../escape.json"}) + "\n")
    output = tmp_path / "out"
    assert cli_batch(manifest, output_dir=output, jobs=2) == 4
    statuses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(statuses) == 3 and statuses[1]["id"] == "good"
    assert "error" in statuses[0] and "error" in statuses[2]
    validate_record(loads_record((output / "good.json").read_bytes()))
    assert not (tmp_path / "escape.json").exists()
    assert not list(output.glob(".ecg-record-*"))


def test_atomic_write_preserves_prior_file_on_failure(tmp_path, monkeypatch):
    from feature_extraction.ecgfeat.cli import _write_text
    path = tmp_path / "existing.json"
    path.write_text("prior")
    def fail(*args): raise OSError("simulated replacement failure")
    monkeypatch.setattr("os.replace", fail)
    with pytest.raises(OSError):
        _write_text("new", path)
    assert path.read_text() == "prior"
    assert not list(tmp_path.glob(".ecg-record-*"))


def test_full_documentation_example_is_machine_valid():
    from pathlib import Path
    import re
    import jsonschema
    text = Path("docs/site/guide/record-format.md").read_text()
    examples = []
    for block in re.findall(r"```json\n(.*?)\n```", text, re.S):
        obj = json.loads(block)
        if isinstance(obj, dict) and "acquisition" in obj:
            examples.append(obj)
    assert len(examples) == 1
    schema = json.loads(Path("schemas/ecg-record/1.0/schema.json").read_text())
    jsonschema.validate(examples[0], schema)
    validate_record(examples[0])


def test_non_objects_and_invalid_profile_raise_record_errors():
    for data in ("[]", "null", "1", '"record"'):
        with pytest.raises(RecordValidationError):
            loads_record(data)
    document = ecg_emit(measured_fixture()).as_dict()
    document["profile"] = []
    with pytest.raises(RecordValidationError):
        loads_record(json.dumps(document))


def test_limited_applicability_and_global_units_are_checked_on_read():
    document = ecg_emit(measured_fixture(), profile="all").as_dict()
    field = document["measurements"]["global"]["qtc_bazett_ms"]
    field.update(values=400)
    field.pop("absence")
    with pytest.raises(RecordValidationError): validate_record(document)
    document = ecg_emit(measured_fixture()).as_dict()
    document["measurements"]["global"]["heart_rate_bpm"]["unit"] = "ms"
    with pytest.raises(RecordValidationError): validate_record(document)


def test_st_provenance_does_not_claim_hybrid_candidates_are_published():
    record = ecg_emit(measured_fixture(source="adaptive_pr_tp"))
    result = query_measurement(record, "st_80ms_uv", lead="sensor", beat=0)
    assert result.provenance.field_method == "legacy_native_st_80ms_mv"
    assert record.provenance["algorithm_paths"]["st_hybrid_candidate_source"] == "adaptive_pr_tp"


def test_unmigrated_stages_and_policies_cannot_return_fake_success():
    from feature_extraction.ecgfeat.pipeline.stages import quality
    from feature_extraction.ecgfeat.pipeline.policies.qt import QTPolicy, QTEvidence
    from feature_extraction.ecgfeat.compat.models_v0 import features_from_record
    with pytest.raises(NotImplementedError): quality.run(None)
    with pytest.raises(NotImplementedError): QTPolicy().decide(QTEvidence(candidates_ms={"arbitrary": 400}), context=None)
    with pytest.raises(NotImplementedError): features_from_record(ecg_emit(measured_fixture()))


def test_legacy_export_does_not_silently_change_wire_format():
    from feature_extraction.ecgfeat.compat.export_v0 import to_dict
    with pytest.raises(TypeError): to_dict(ecg_emit(measured_fixture()))


def test_malformed_engine_state_is_not_disguised_as_missing_measurement():
    from feature_extraction.ecgfeat.errors import ComputationInvariantError
    measurements = measured_fixture()
    del measurements.legacy_features.beat_features[0].r_peak_index
    with pytest.raises(ComputationInvariantError): ecg_emit(measurements)


def test_npz_encoding_forbids_pickle_objects_and_nonfinite_values():
    from feature_extraction.ecgfeat.record import encode_npz_sidecar
    for array in (np.array([object()], dtype=object), np.array([np.nan])):
        with pytest.raises(RecordValidationError): encode_npz_sidecar({"bad": array})
    encoded = encode_npz_sidecar({"x": np.array([[1, 2]], dtype=np.int32)})
    with np.load(io.BytesIO(encoded), allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["x"], [[1, 2]])


def test_internal_coordinates_are_mapped_to_original_signal_and_unknown_leads():
    record = ecg_emit(measured_fixture())
    assert record.axes["beats"][0]["r_sample"] == 200
    assert query_measurement(record, "r_peak", lead="sensor", beat=0).value == 201
    assert query_measurement(record, "qrs_signed_area_uv_ms", lead="sensor", beat=0).value == 10_000
    assert record.quality["leads"]["sensor"]["status"] == "usable"
    assert record.quality["record"]["status"] == "usable"


def test_zero_r_sample_is_not_replaced_by_detection_fiducial():
    measurements = measured_fixture()
    cell = measurements.legacy_features.beat_features[0]
    cell.r_peak_index = 0
    cell.qrs.onset = 0  # keep the published onset <= R <= offset invariant
    assert query_measurement(ecg_emit(measurements), "r_peak", lead="sensor", beat=0).value == 0


def test_r_peak_outside_its_qrs_window_is_withheld_not_published():
    measurements = measured_fixture()
    measurements.legacy_features.beat_features[0].r_peak_index = 0  # before QRS onset 380
    record = ecg_emit(measurements)
    for name in ("r_peak", "qrs_onset", "qrs_offset", "qrs_duration_ms"):
        result = query_measurement(record, name, lead="sensor", beat=0)
        assert result.value is None and result.absence.reason == "fiducial_order_violation", name


def test_inverted_qrs_and_negative_interval_become_unmeasurable():
    measurements = measured_fixture()
    cell = measurements.legacy_features.beat_features[0]
    cell.qrs.onset, cell.qrs.offset, cell.qrs_ms = 432, 428, -8.0
    cell.pr_ms = -4.0
    record = ecg_emit(measurements)
    assert query_measurement(record, "qrs_duration_ms", lead="sensor", beat=0).absence.reason == "fiducial_order_violation"
    assert query_measurement(record, "r_peak", lead="sensor", beat=0).value is not None
    assert query_measurement(record, "pr_interval_ms", lead="sensor", beat=0).absence.reason == "negative_interval"


def test_missing_true_r_peak_is_not_replaced_by_detection_fiducial():
    measurements = measured_fixture()
    measurements.legacy_features.beat_features[0].r_peak_index = None
    value = query_measurement(ecg_emit(measurements), "r_peak", lead="sensor", beat=0)
    assert value.value is None
    assert value.absence.kind == "unmeasurable"


def test_limited_qtc_is_not_reported_in_all_profile():
    record = ecg_emit(measured_fixture(), profile="all")
    assert query_measurement(record, "qtc_bazett_ms").absence.kind == "not_applicable"


def test_run_identity_covers_calibration_and_configuration_but_not_profile():
    base = measured_fixture()
    record = ecg_emit(base)
    assert ecg_emit(base, profile="all").record_id == record.record_id
    assert ecg_emit(measured_fixture(unit="uV")).record_id != record.record_id
    assert ecg_emit(measured_fixture(source="calibrated_pr")).record_id != record.record_id
    digest = hashlib.sha256(base.prepared.signal.astype("<f8").tobytes()).hexdigest()
    assert record.artifacts["raw_signal"]["sha256"] == "sha256:" + digest


def test_prepare_does_not_freeze_or_alias_caller_memory():
    signal = np.zeros((1, 500))
    prepared = ecg_prepare(signal, sampling_rate=500, lead_names=["II"], input_mode="limited")
    assert signal.flags.writeable
    signal[:] = 5
    assert not prepared.signal.any()
    with pytest.raises(ValueError):
        prepared.signal.setflags(write=True)


def test_prepare_rejects_transposed_signal():
    with pytest.raises(SignalShapeError):
        ecg_prepare(np.zeros((500, 1)), sampling_rate=500, lead_names=["II"], input_mode="limited")


def test_config_mismatch_is_rejected_before_engine(monkeypatch):
    prepared = measured_fixture().prepared
    with pytest.raises(ConfigurationError, match="input_mode"):
        ecg_measure(prepared, config=ECGConfig())


def test_default_sampling_and_refinements_match_legacy_engine():
    assert ECGConfig().fs_internal is None
    config = ECGConfig(refinement=RefinementConfig(t_bidirectional=True))
    from feature_extraction.ecgfeat.api import ECGFeatureExtractor
    from feature_extraction.ecgfeat.pipeline.context import ExtractorSettings
    from feature_extraction.ecgfeat.pipeline.extractor import _settings_from_config
    assert _settings_from_config(config).refinement.t_bidirectional
    # The record path's settings equal the legacy extractor's for the same options.
    for options in ({}, {"input_mode": "limited", "fs_internal": 500}, {"st_amplitude_source": "calibrated_pr"},
                    {"mains_frequency_hz": 60, "refinement": RefinementConfig(t_bidirectional=True)}):
        cfg = ECGConfig(**options)
        legacy = ECGFeatureExtractor(
            fs_internal=cfg.fs_internal, mains_freq=cfg.mains_frequency_hz, st_amplitude_source=cfg.st_amplitude_source,
            refinement=cfg.refinement, input_mode="limited" if cfg.input_mode == "limited" else "standard")
        assert _settings_from_config(cfg) == ExtractorSettings.from_extractor(legacy)
    assert json.loads(json.dumps(asdict(config.refinement)))["t_bidirectional"] is True
    with pytest.raises(ValueError, match="unknown"):
        RefinementConfig.experimental("st_localization")


def header_text():
    return ("r 2 250 2\n"
            "r.mat 16+24 200(100)/mV 16 0 0 0 0 V2\n"
            "r.mat 16+24 2/uV 16 -10 0 0 0 sensor\n")


def test_wfdb_uses_header_names_sampling_rate_gain_baseline_and_units(tmp_path):
    mat = tmp_path / "r.mat"
    savemat(mat, {"val": np.array([[300, -100], [1990, -2010]])})
    mat.with_suffix(".hea").write_text(header_text())
    result = read_wfdb(mat)
    assert result.lead_names == ("V2", "sensor")
    assert result.sampling_rate == 250
    np.testing.assert_allclose(result.values, [[1, -1], [1, -1]])
    header = read_wfdb_header(io.StringIO(header_text()))
    assert header.baseline == (100, -10)


def test_wfdb_missing_header_does_not_guess_sampling_or_leads(tmp_path):
    mat = tmp_path / "r.mat"
    savemat(mat, {"val": np.zeros((2, 2))})
    with pytest.raises(FileNotFoundError):
        read_wfdb(mat)


@pytest.mark.parametrize("text", ["r 2 250 2\n", header_text().replace("200(100)/mV", "0/mV"),
                                  header_text().replace("16+24", "16:3+24"),
                                  header_text().replace(" sensor", " V2")])
def test_wfdb_rejects_incomplete_or_unsupported_headers(text):
    with pytest.raises(ValueError):
        read_wfdb_header(io.StringIO(text))

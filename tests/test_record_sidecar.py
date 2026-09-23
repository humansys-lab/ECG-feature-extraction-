"""NPZ dense-matrix sidecar contract (document 02 D.5/F, document 06 assertion 7)."""

import hashlib
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from feature_extraction.ecgfeat import ecg_emit
from feature_extraction.ecgfeat.errors import RecordValidationError
from feature_extraction.ecgfeat.record import (
    SidecarError, dumps_record, load_record, materialize_record, query_measurement, serialize_record,
)
from feature_extraction.ecgfeat.record import sidecar as sidecar_module
from tests.test_library_design_review import measured_fixture

URI = "rec.dense.npz"


def _write(tmp_path, record, uri=URI):
    encoded = serialize_record(record, sidecar_uri=uri)
    (tmp_path / "rec.json").write_bytes(encoded.json_bytes)
    (tmp_path / uri).write_bytes(encoded.sidecars[uri])
    return encoded


@pytest.mark.parametrize("profile", ["summary", "all", "debug"])
def test_sidecar_round_trip_restores_the_inline_record_exactly(tmp_path, profile):
    record = ecg_emit(measured_fixture(), profile=profile)
    _write(tmp_path, record)
    loaded = load_record(tmp_path / "rec.json")
    assert materialize_record(loaded).as_dict() == record.as_dict()
    for name in ("qrs_onset", "qrs_duration_ms", "st_80ms_uv"):
        assert query_measurement(loaded, name, lead="sensor", beat=0) == query_measurement(record, name, lead="sensor", beat=0)


def test_json_moves_every_dense_matrix_and_keeps_absence_reasons():
    document = json.loads(serialize_record(ecg_emit(measured_fixture(), profile="all"), sidecar_uri=URI).json_bytes)
    descriptor = document["artifacts"]["dense_measurements"]
    assert descriptor["uri"] == URI and descriptor["media_type"] == "application/x-npz"
    assert descriptor["record_id"] == document["record_id"] and descriptor["shape"] == [1, 1]
    st = document["measurements"]["amplitudes"]["st_80ms_uv"]
    assert st["values"] is None and st["sidecar"]["array_key"] == "measurements.amplitudes.st_80ms_uv"
    assert st["absence"] == {"kind": "unmeasurable", "reason": "measurement_unavailable"}
    assert document["measurements"]["global"]["heart_rate_bpm"]["values"] == 60  # scalars stay inline


def test_sidecar_bytes_are_deterministic_and_digest_bound():
    record = ecg_emit(measured_fixture(), profile="all")
    first, second = serialize_record(record, sidecar_uri=URI), serialize_record(record, sidecar_uri=URI)
    assert first.sidecars == second.sidecars and first.json_bytes == second.json_bytes
    digest = json.loads(first.json_bytes)["artifacts"]["dense_measurements"]["sha256"]
    assert digest == "sha256:" + hashlib.sha256(first.sidecars[URI]).hexdigest()
    with zipfile.ZipFile(io.BytesIO(first.sidecars[URI])) as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_tampered_missing_or_foreign_sidecars_are_rejected(tmp_path):
    record = ecg_emit(measured_fixture(), profile="all")
    encoded = _write(tmp_path, record)
    (tmp_path / URI).write_bytes(encoded.sidecars[URI] + b"\0")
    with pytest.raises(SidecarError, match="digest"):
        load_record(tmp_path / "rec.json")
    other = serialize_record(ecg_emit(measured_fixture(unit="uV"), profile="all"), sidecar_uri=URI)
    with pytest.raises(SidecarError):
        load_record(tmp_path / "rec.json", sidecar=other.sidecars[URI])
    (tmp_path / URI).unlink()
    with pytest.raises(SidecarError):
        load_record(tmp_path / "rec.json")
    lazy = load_record(tmp_path / "rec.json", validate="schema")
    with pytest.raises(SidecarError):
        query_measurement(lazy, "qrs_duration_ms", lead="sensor", beat=0)
    assert query_measurement(lazy, "heart_rate_bpm").value == 60  # inline scalars need no sidecar


def test_state_mask_must_agree_with_json_absence():
    record = ecg_emit(measured_fixture(), profile="all")
    document, data = sidecar_module.split_dense(json.loads(dumps_record(record)), URI)
    del document["measurements"]["amplitudes"]["st_80ms_uv"]["absence"]  # JSON now says plain null
    with pytest.raises(SidecarError, match="state mask"):
        sidecar_module.read_sidecar(document, data)


@pytest.mark.parametrize("uri", ["../escape.npz", "/abs/x.npz", "dir/x.npz", "x.json", "https://h/x.npz"])
def test_sidecar_locator_is_a_plain_relative_file_name(uri):
    with pytest.raises(SidecarError):
        serialize_record(ecg_emit(measured_fixture(), profile="all"), sidecar_uri=uri)


def test_descriptor_must_list_exactly_the_sidecar_backed_fields():
    record = ecg_emit(measured_fixture(), profile="all")
    document = json.loads(serialize_record(record, sidecar_uri=URI).json_bytes)
    document["artifacts"]["dense_measurements"]["fields"].pop()
    with pytest.raises(RecordValidationError):
        load_record(document, validate="schema")


def test_cli_measure_writes_verified_sidecar_next_to_record(tmp_path):
    signal = np.zeros((12, 5000))
    t = np.arange(5000) / 500
    for index in range(12):
        signal[index] = np.sin(2 * np.pi * 1.2 * t) * 0.1 * (index + 1)
    np.save(tmp_path / "signal.npy", signal)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    command = [sys.executable, "-m", "ecgfeat", "measure", str(tmp_path / "signal.npy"), "--sampling-rate", "500",
               "--lead-names", *leads, "--profile", "all", "--sidecar", str(tmp_path / "out.npz"), "-o", str(tmp_path / "out.json")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "out.json").read_text())["artifacts"]["dense_measurements"]["uri"] == "out.npz"
    check = subprocess.run([sys.executable, "-m", "ecgfeat", "validate", str(tmp_path / "out.json")], capture_output=True, text=True)
    assert check.returncode == 0 and json.loads(check.stdout)["valid"] is True
    elsewhere = subprocess.run(command[:-4] + ["--sidecar", str(tmp_path / "sub" / "x.npz"), "-o", str(tmp_path / "o2.json")],
                               capture_output=True, text=True)
    assert elsewhere.returncode == 2

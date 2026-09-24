"""ecgagent's ECG Record boundary: schema-checked addresses, preserved absence states."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ecgagent.evidence.record_boundary import (
    RecordAddressError, load_record, parse_record_address, record_path_for, resolve_record_evidence,
)

REFERENCE = "tests/fixtures/golden/reference_10s_12lead/record.all.json"


@pytest.fixture(scope="module")
def record():
    return load_record(REFERENCE)


def _address(record, pointer):
    return f"ecg-record:{record.record_id}@{record.schema_version}#{pointer}"


def test_measured_cell_carries_unit_lead_beat_and_provenance(record):
    evidence = resolve_record_evidence(record, _address(record, "/measurements/intervals/qrs_duration_ms/values/3/1"))
    assert evidence.state == "measured" and isinstance(evidence.value, int)
    assert (evidence.unit, evidence.lead, evidence.beat) == ("ms", "II", 3)
    assert evidence.validation == "unvalidated" and evidence.config_hash


def test_absent_cell_keeps_its_state_and_reason(record):
    document = record.as_dict()
    groups = {"delineation/fiducials": document["delineation"]["fiducials"],
              **{f"measurements/{k}": v for k, v in document["measurements"].items()}}
    found = [(root, name, i, j) for root, fields in groups.items() for name, field in fields.items()
             if field["axes"] == ["beat", "lead"]
             for i, row in enumerate(field["values"]) for j, v in enumerate(row) if v is None]
    assert found, "the reference record has absent P-wave cells"
    root, name, i, j = found[0]
    evidence = resolve_record_evidence(record, _address(record, f"/{root}/{name}/values/{i}/{j}"))
    assert evidence.value is None and evidence.state in {"unmeasurable", "not_applicable"} and evidence.reason


def test_structure_nodes_resolve_without_cell_semantics(record):
    evidence = resolve_record_evidence(record, _address(record, "/acquisition/sample_rate_hz"))
    assert evidence.state == "structure" and evidence.value == 500


@pytest.mark.parametrize("mutate", ["schema", "record_id", "syntax", "missing"])
def test_bad_addresses_are_rejected_before_use(record, mutate):
    address = {
        "schema": f"ecg-record:{record.record_id}@2.0.0#/acquisition",
        "record_id": f"ecg-record:rec-other@{record.schema_version}#/acquisition",
        "syntax": "not-an-address",
        "missing": _address(record, "/measurements/intervals/nope/values/0/0"),
    }[mutate]
    with pytest.raises(RecordAddressError):
        resolve_record_evidence(record, address)


def test_unsupported_schema_is_rejected_at_parse_time():
    with pytest.raises(RecordAddressError, match="unsupported"):
        parse_record_address("ecg-record:rec-1@3.1.0#/acquisition")


def test_batch_writes_a_valid_record_next_to_the_features(tmp_path):
    from ecgagent import batch
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    signal = np.load("tests/fixtures/golden/reference_10s_12lead/signal.npz")["signal"]
    features = ECGFeatureExtractor(fs_internal=500, mains_freq=50).extract(signal, fs=500)
    path, error = batch._write_ecg_record(tmp_path, {"record": "ref", "age": 60, "sex": "M"}, signal, 500.0, 500, features)
    assert error is None and path == str(record_path_for(tmp_path, "ref"))
    loaded = load_record(path)
    assert loaded.profile == "all" and loaded.metadata["patient"] == {"age": 60, "sex": "M"}

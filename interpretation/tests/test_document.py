"""Interpretation document contract (ecginterpret 0.1)."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import ecginterpret
from ecginterpret import InterpretationInputError, interpret_features, interpret_record

REFERENCE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "golden" / "reference_10s_12lead"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SCHEMA = json.loads((Path(ecginterpret.__file__).parent / "schemas" / "interpretation" / "1.0" / "schema.json").read_text())


@pytest.fixture(scope="module")
def signal():
    return np.load(REFERENCE / "signal.npz")["signal"]


@pytest.fixture(scope="module")
def legacy_features(signal):
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ECGFeatureExtractor().extract(signal, 500, lead_names=LEADS, amplitude_unit="mV")


def _normalized(document):
    data = json.loads(json.dumps(document.as_dict()))
    data["clinical"].pop("generated_at", None)
    return data


def test_document_is_versioned_json_and_schema_valid(legacy_features):
    jsonschema = pytest.importorskip("jsonschema")
    document = interpret_features(legacy_features)
    data = json.loads(json.dumps(document.as_dict(), allow_nan=False))
    assert data["schema"] == "ecg-interpretation" and data["schema_version"] == ecginterpret.INTERPRETATION_SCHEMA_VERSION
    assert data["interpreter_version"] == ecginterpret.__version__
    assert data["input"] == {"kind": "legacy_measurement_object"}
    assert list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(data)) == []


def test_document_members_equal_what_the_legacy_extractor_embedded(legacy_features):
    document = interpret_features(legacy_features)
    assert document.interpretation == json.loads(json.dumps(
        ecginterpret.document._json_safe(legacy_features.interpretation)))
    embedded = ecginterpret.document._json_safe(legacy_features.metadata["clinical_interpretation"])
    ours = dict(document.clinical)
    for side in (ours, embedded):
        side.pop("generated_at", None)
    assert ours == embedded


def test_interpret_record_rederives_the_same_interpretation(signal, legacy_features):
    jsonschema = pytest.importorskip("jsonschema")
    from ecgfeat import ecg_record

    record = ecg_record(signal, sampling_rate=500, lead_names=LEADS)
    from_record = interpret_record(record, signal=signal)
    from_features = interpret_features(legacy_features)
    a, b = _normalized(from_record), _normalized(from_features)
    assert a["input"]["kind"] == "ecg_record" and a["input"]["record_id"] == record.record_id
    assert a["interpretation"] == b["interpretation"] and a["clinical"] == b["clinical"]
    assert list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(from_record.as_dict())) == []


def test_interpret_record_requires_the_matching_signal(signal):
    from ecgfeat import ecg_record

    record = ecg_record(signal, sampling_rate=500, lead_names=LEADS)
    with pytest.raises(InterpretationInputError, match="needs the raw signal"):
        interpret_record(record)
    with pytest.raises(InterpretationInputError, match="does not match"):
        interpret_record(record, signal=signal * 1.0001)
    document = record.as_dict()
    document["schema_version"] = "2.0.0"
    with pytest.raises(InterpretationInputError, match="unsupported"):
        interpret_record(document, signal=signal)


def test_interpret_features_rejects_other_inputs():
    with pytest.raises(InterpretationInputError):
        interpret_features({"not": "features"})

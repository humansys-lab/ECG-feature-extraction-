from __future__ import annotations

import io

import numpy as np
import pytest

from feature_extraction.ecgfeat import (
    AddressNotFoundError,
    AmplitudeUnitError,
    ECGConfig,
    LeadNameError,
    MeasurementQuery,
    NotApplicable,
    NotApplicableAbsence,
    SignalShapeError,
    dumps_record,
    ecg_record,
    loads_record,
    query_measurement,
    select_measurements,
)
from feature_extraction.ecgfeat.record import build_record as _build_record, dump_measurements, measured, not_applicable, unavailable
from feature_extraction.ecgfeat.record.query import make_record_address, resolve_address


LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def build_record(**kwargs):
    return _build_record(**kwargs, acquisition={"sample_rate_hz": 500, "sample_count": 1000,
        "input_mode": "limited", "amplitude_unit": "mV", "leads": ["II"]}, axes={"leads": ["II"], "beats": []})


def test_availability_states_survive_record_round_trip() -> None:
    record = build_record(
        record_id="contract-1",
        schema_version="1.0.0",
        measurements={"measured": measured(3), "unavailable": unavailable("low_snr"), "limited": not_applicable("input_mode_limited")},
        provenance={},
        validation={},
    )
    loaded = loads_record(dumps_record(record))
    assert query_measurement(loaded, "measured").value == 3
    assert query_measurement(loaded, "unavailable").absence.kind == "unmeasurable"
    assert isinstance(query_measurement(loaded, "limited").absence, NotApplicableAbsence)


def test_rfc6901_address_resolution_checks_identity_and_missing_paths() -> None:
    record = build_record(record_id="contract", schema_version="1.0.0", measurements={"a/b~c": measured(7)}, provenance={}, validation={})
    address = make_record_address(record, "/measurements/global/a~1b~0c/values")
    assert resolve_address(record, str(address)).value == 7
    with pytest.raises(AddressNotFoundError):
        resolve_address(record, "ecg-record:contract@1.0.0#/measurements/nope")


def test_summary_is_compact_and_limited_mode_marks_axis_not_applicable() -> None:
    signal = np.zeros((2, 1000), dtype=float)
    record = ecg_record(signal, sampling_rate=500, lead_names=("I", "II"), input_mode="limited")
    assert len(dumps_record(record)) < 24_000
    result = query_measurement(record, "frontal_qrs_axis_deg")
    assert isinstance(result.absence, NotApplicableAbsence)
    assert result.absence.reason == "input_mode_limited"


def test_new_input_contract_rejects_standard_lead_padding() -> None:
    with pytest.raises(LeadNameError):
        ecg_record(np.zeros((2, 1000)), sampling_rate=500, lead_names=("I", "II"))
    with pytest.raises(AmplitudeUnitError):
        ecg_record(np.zeros((12, 1000)), sampling_rate=500, lead_names=LEADS, amplitude_unit="digital")


def test_measurement_selection_preserves_query_order() -> None:
    record = build_record(record_id="select", schema_version="1.0.0", measurements={"a": measured(1), "b": measured(2)}, provenance={}, validation={})
    selection = select_measurements(record, (MeasurementQuery("b"), MeasurementQuery("a")))
    stream = io.StringIO()
    dump_measurements(selection, stream, format="jsonl")
    lines = stream.getvalue().splitlines()
    assert '"name":"b"' in lines[0]
    assert '"name":"a"' in lines[1]

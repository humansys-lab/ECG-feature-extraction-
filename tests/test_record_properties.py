"""Property-based invariants of the ECG Record boundary (document 06).

Two generators are used: random *engine states* (the legacy per-beat carriers
the record builder consumes, including impossible orderings, negative and
non-finite values) and a small number of real extractions of a synthetic ECG.
"""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

from feature_extraction.ecgfeat import ECGConfig, ecg_emit, ecg_prepare, ecg_record  # noqa: E402
from feature_extraction.ecgfeat.config import config_provenance  # noqa: E402
from feature_extraction.ecgfeat.errors import ECGInputError, RecordValidationError  # noqa: E402
from feature_extraction.ecgfeat.pipeline.extractor import ECGMeasurements  # noqa: E402
from feature_extraction.ecgfeat.record import (  # noqa: E402
    dumps_record, loads_record, make_record_address, query_measurement, resolve_address,
)
from tests.fixtures.golden.reference_10s_12lead.make_signal import LEADS, make_signal  # noqa: E402

FAST = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
SLOW = settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
PROFILES = ("summary", "all", "debug")
UNITS = {"fiducials": "sample", "intervals": "ms", "amplitudes": "uV", "areas": "uV_ms"}

maybe_number = st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True, width=32),
                         st.integers(min_value=-2000, max_value=6000))
maybe_sample = st.one_of(st.none(), st.integers(min_value=-50, max_value=1100))


@st.composite
def engine_states(draw):
    """Random legacy measurement state for a 2 s limited-mode record."""
    n_leads = draw(st.integers(min_value=1, max_value=3))
    n_beats = draw(st.integers(min_value=0, max_value=4))
    names = [f"ch{i}" for i in range(n_leads)]
    slots = ["II", "V2", "V5"][:n_leads]
    fs, internal_fs = draw(st.sampled_from([(250, 500), (500, 500), (360, 500)]))
    cells = []
    for beat in range(n_beats):
        for slot in slots:
            cells.append(NS(
                lead=slot, beat_id=beat,
                p=NS(onset=draw(maybe_sample), offset=draw(maybe_sample)),
                qrs=NS(onset=draw(maybe_sample), peak=None, offset=draw(maybe_sample)),
                t=NS(offset=draw(maybe_sample)), r_peak_index=draw(maybe_sample), j_index=draw(maybe_sample),
                **{name: draw(maybe_number) for name in (
                    "p_dur_ms", "pr_ms", "qrs_ms", "qt_ms", "tpe_ms", "p_amp_mv", "r_amp_mv", "s_amp_mv",
                    "st_80ms_mv", "t_amp_mv", "qrs_signed_area", "jt_ms", "u_amp_signed_mv")}))
    features = NS(
        fs=internal_fs, beats=[NS(beat_id=b, r_index=draw(maybe_sample)) for b in range(n_beats)], beat_features=cells,
        global_features=NS(heart_rate_bpm=draw(maybe_number), qrs_axis_deg=draw(maybe_number),
                           qtc_bazett_ms=draw(maybe_number), qrs_wide_ms=draw(maybe_number)),
        quality={slot: NS(reliable=draw(st.booleans())) for slot in slots},
        metadata={"record_quality": {"record_grade": draw(st.sampled_from(["Q0", "Q1", "Q2", None]))},
                  "input_contract": {"lead_slot_map": dict(zip(names, slots))}, "mains_frequency_hz": 50})
    prepared = ecg_prepare(np.zeros((n_leads, fs * 2)), sampling_rate=fs, lead_names=names, input_mode="limited")
    config = ECGConfig(input_mode="limited", fs_internal=internal_fs)
    return ECGMeasurements(prepared, features, config, config_provenance(config))


def _dense(document):
    groups = {"fiducials": document["delineation"]["fiducials"], **document["measurements"]}
    for group, fields in groups.items():
        for name, field in fields.items():
            yield group, name, field


def _cell_states(document, field):
    """(value, kind, reason) per cell, resolving every absence encoding."""
    absence = field.get("absence") or {}
    beats = [b["id"] for b in document["axes"]["beats"]]
    leads = document["axes"]["leads"]
    default = absence.get("default") or {}
    out = []
    for i, row in enumerate(field["values"]):
        for j, value in enumerate(row):
            coordinate = f"{beats[i]}|{leads[j]}"
            if value is not None:
                out.append((value, None, None))
            elif "kind" in absence:
                out.append((None, absence["kind"], absence["reason"]))
            elif coordinate in absence.get("unmeasurable", {}):
                out.append((None, "unmeasurable", absence["unmeasurable"][coordinate]))
            elif coordinate in absence.get("not_applicable", {}):
                out.append((None, "not_applicable", absence["not_applicable"][coordinate]))
            elif default:
                out.append((None, default["kind"], default["reason"]))
            else:
                out.append((None, "null", None))
    return out


@FAST
@given(engine_states(), st.sampled_from(PROFILES))
def test_any_engine_state_yields_a_valid_record(state, profile):
    # Well-formed engine carriers must always yield a valid record: a
    # ComputationInvariantError here would mean the builder emitted a record
    # its own contract rejects.
    record = ecg_emit(state, profile=profile)
    json.loads(dumps_record(record))  # allow_nan=False: no NaN/inf reaches JSON
    assert loads_record(dumps_record(record)).as_dict() == record.as_dict()


@FAST
@given(engine_states())
def test_published_fiducials_are_ordered_and_intervals_non_negative(state):
    document = ecg_emit(state, profile="all").as_dict()
    fid = document["delineation"]["fiducials"]
    for i, row in enumerate(fid["qrs_onset"]["values"]):
        for j in range(len(row)):
            chain = [fid[n]["values"][i][j] for n in ("qrs_onset", "r_peak", "qrs_offset")]
            present = [v for v in chain if v is not None]
            assert present == sorted(present)
            p = [fid[n]["values"][i][j] for n in ("p_onset", "p_offset")]
            assert None in p or p[0] <= p[1]
    for name, field in document["measurements"]["intervals"].items():
        assert all(v is None or v >= 0 for row in field["values"] for v in row), name


@FAST
@given(engine_states(), st.sampled_from(PROFILES))
def test_every_absence_has_exactly_one_state_and_a_reason(state, profile):
    document = ecg_emit(state, profile=profile).as_dict()
    for group, name, field in _dense(document):
        if field["axes"] != ["beat", "lead"]:
            if field["values"] is None:
                assert field["absence"]["kind"] in {"unmeasurable", "not_applicable"} and field["absence"]["reason"]
            continue
        for value, kind, reason in _cell_states(document, field):
            if value is None:
                assert kind in {"unmeasurable", "not_applicable"} and reason, (group, name)
            else:
                assert kind is None and isinstance(value, int) and not isinstance(value, bool)


@FAST
@given(engine_states())
def test_profiles_are_monotone_and_share_identical_values(state):
    from benchmarks.golden.runner import iter_leaves

    leaves = {p: {k: v for k, v in iter_leaves(ecg_emit(state, profile=p).as_dict()) if k != "/profile"}
              for p in PROFILES}
    assert set(leaves["summary"]) <= set(leaves["all"]) <= set(leaves["debug"])
    for lower, higher in (("summary", "all"), ("all", "debug")):
        assert {k: leaves[higher][k] for k in leaves[lower]} == leaves[lower]


@FAST
@given(engine_states(), st.sampled_from(PROFILES))
def test_canonical_encoding_is_idempotent_and_round_trips(state, profile):
    data = dumps_record(ecg_emit(state, profile=profile))
    again = dumps_record(loads_record(data))
    assert again == data
    assert loads_record(again).as_dict() == loads_record(data).as_dict()


@FAST
@given(engine_states())
def test_addressing_is_total_over_published_cells(state):
    record = ecg_emit(state, profile="all")
    document = record.as_dict()
    for group, name, field in _dense(document):
        root = "/delineation/fiducials" if group == "fiducials" else f"/measurements/{group}"
        if field["axes"] == ["beat", "lead"]:
            for i, row in enumerate(field["values"]):
                for j, value in enumerate(row):
                    address = make_record_address(record, f"{root}/{name}/values/{i}/{j}")
                    assert resolve_address(record, str(address)).value == value
                    result = query_measurement(record, name, lead=document["axes"]["leads"][j], beat=i)
                    assert result.value == value and (value is None) == (result.absence is not None)
        else:
            assert query_measurement(record, name).value == field["values"]


@FAST
@given(engine_states(), st.sampled_from(["benchmark_validated", "indirectly_validated", "validated"]))
def test_serialization_never_upgrades_validation(state, status):
    document = json.loads(dumps_record(ecg_emit(state, profile="all")))
    assert {f["validation"]["status"] for _, _, f in _dense(document)} == {"unvalidated"}
    document["measurements"]["intervals"]["qt_interval_ms"]["validation"] = {"status": status, "evidence": ["e"]}
    with pytest.raises(RecordValidationError):
        loads_record(json.dumps(document))


# --------------------------------------------------------------------------- #
# real extractions of the synthetic reference ECG
# --------------------------------------------------------------------------- #

def _resample(signal, fs):
    t_old = np.arange(signal.shape[1]) / 500
    t_new = np.arange(int(round(signal.shape[1] * fs / 500))) / fs
    return np.stack([np.interp(t_new, t_old, row) for row in signal])


@pytest.fixture(scope="module")
def reference_signal():
    return make_signal()


@SLOW
@given(fs=st.sampled_from([250, 360, 500, 1000]))
def test_units_do_not_depend_on_sampling_rate(reference_signal, fs):
    record = ecg_record(_resample(reference_signal, fs), sampling_rate=fs, lead_names=list(LEADS), profile="all")
    document = record.as_dict()
    for group, name, field in _dense(document):
        if group in UNITS:
            assert field["unit"] == UNITS[group] and field["type"] == "integer", name
    assert document["measurements"]["global"]["heart_rate_bpm"]["unit"] == "1/min"
    assert document["acquisition"]["sample_rate_hz"] == fs
    fid = document["delineation"]["fiducials"]["r_peak"]["values"]
    assert all(v is None or 0 <= v < document["acquisition"]["sample_count"] for row in fid for v in row)


@SLOW
@given(order=st.permutations(list(range(12))))
def test_lead_permutation_cannot_relabel_measurements(reference_signal, order):
    base = ecg_record(reference_signal, sampling_rate=500, lead_names=list(LEADS), profile="all")
    permuted = ecg_record(reference_signal[order], sampling_rate=500, lead_names=[LEADS[i] for i in order], profile="all")
    for name in ("r_amplitude_uv", "t_amplitude_uv", "qrs_duration_ms"):
        for beat in range(len(base.as_dict()["axes"]["beats"])):
            for lead in LEADS:
                assert query_measurement(base, name, lead=lead, beat=beat).value == \
                    query_measurement(permuted, name, lead=lead, beat=beat).value


@SLOW
@given(seed=st.integers(min_value=0, max_value=2**32 - 1), scale=st.floats(min_value=1e-4, max_value=5.0))
def test_finite_noise_never_crashes_or_emits_nonfinite_values(seed, scale):
    signal = np.random.default_rng(seed).normal(0.0, scale, (12, 1500))
    try:
        record = ecg_record(signal, sampling_rate=500, lead_names=list(LEADS), profile="all")
    except ECGInputError:
        return
    text = dumps_record(record).decode()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda token: pytest.fail(f"non-finite JSON constant {token}"))

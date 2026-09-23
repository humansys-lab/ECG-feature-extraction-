"""Golden reference record: exact regeneration and the 24,000-byte summary contract.

Update procedure: ``python tests/fixtures/golden/update_fixture.py reference_10s_12lead``
(one named case at a time) and review the printed pointer diff (document 06).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.golden.update_fixture import CASES, produce

CASE = Path("tests/fixtures/golden/reference_10s_12lead")
MANIFEST = json.loads((CASE / "manifest.json").read_text())
SUMMARY_LIMIT = 24_000


@pytest.fixture(scope="module")
def regenerated():
    return produce("reference_10s_12lead")


def test_signal_identity_matches_manifest():
    assert hashlib.sha256((CASE / "signal.npz").read_bytes()).hexdigest() == MANIFEST["signal_file_sha256"]
    signal = np.load(CASE / "signal.npz")["signal"]
    assert hashlib.sha256(np.ascontiguousarray(signal, dtype="<f8").tobytes()).hexdigest() == MANIFEST["signal_sha256"]
    assert list(signal.shape) == MANIFEST["signal_shape"] == [12, 5000]


@pytest.mark.parametrize("profile", ["summary", "all", "debug"])
def test_extraction_reproduces_the_frozen_record_bytes(regenerated, profile):
    from benchmarks.golden.compare import pointer_diff

    expected = (CASE / f"record.{profile}.json").read_bytes()
    assert hashlib.sha256(expected).hexdigest() == MANIFEST["records"][profile]["sha256"]
    if regenerated[profile] != expected:
        diff = pointer_diff(json.loads(expected), json.loads(regenerated[profile]))
        pytest.fail(f"golden drift in {profile}: {json.dumps(diff)[:2000]}")


def test_summary_fits_the_24000_byte_contract(regenerated, record_property):
    size = len(regenerated["summary"])
    record_property("summary_bytes", size)
    print(f"reference summary: {size} bytes, headroom {SUMMARY_LIMIT - size}")
    assert size <= SUMMARY_LIMIT
    document = json.loads(regenerated["summary"])
    assert len(document["axes"]["beats"]) == 10 and len(document["axes"]["leads"]) == 12
    assert document["acquisition"]["sample_rate_hz"] == 500 and document["acquisition"]["sample_count"] == 5000


def test_summary_is_the_compact_canonical_encoding(regenerated):
    data = regenerated["summary"]
    assert data == json.dumps(json.loads(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("profile", ["summary", "all", "debug"])
def test_frozen_records_validate_against_the_json_schema(profile):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(Path("schemas/ecg-record/1.0/schema.json").read_text())
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(json.loads((CASE / f"record.{profile}.json").read_text())))
    assert errors == []


def _published_leaves(document):
    from benchmarks.golden.runner import iter_leaves

    return {pointer: value for pointer, value in iter_leaves(document) if not pointer.startswith("/profile")}


def test_profiles_nest_strictly_with_identical_shared_values():
    documents = {p: json.loads((CASE / f"record.{p}.json").read_text()) for p in ("summary", "all", "debug")}
    leaves = {p: _published_leaves(d) for p, d in documents.items()}
    assert set(leaves["summary"]) < set(leaves["all"]) < set(leaves["debug"])
    for lower, higher in (("summary", "all"), ("all", "debug")):
        assert {k: leaves[higher][k] for k in leaves[lower]} == leaves[lower]


@pytest.mark.parametrize("profile", ["summary", "all", "debug"])
def test_decode_encode_is_byte_identical(profile):
    from feature_extraction.ecgfeat.record import dumps_record, loads_record

    data = (CASE / f"record.{profile}.json").read_bytes()
    assert dumps_record(loads_record(data)) == data


def test_every_published_leaf_is_addressable():
    from feature_extraction.ecgfeat.record import loads_record, make_record_address, resolve_address
    from benchmarks.golden.runner import iter_leaves

    record = loads_record((CASE / "record.all.json").read_bytes())
    count = 0
    for pointer, value in iter_leaves(record.as_dict()):
        if pointer.startswith(("/delineation", "/measurements")) and "/values" in pointer:
            assert resolve_address(record, str(make_record_address(record, pointer))).value == value
            count += 1
    assert count > 2000


def test_fixture_case_catalogue_matches_directory():
    assert set(CASES) == {path.name for path in Path("tests/fixtures/golden").iterdir()
                          if path.is_dir() and not path.name.startswith("__")}

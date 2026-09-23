"""Unit tests for the golden parity harness (no datasets required)."""

import json

import numpy as np
import pytest

from benchmarks.golden import compare, manifest, runner
from benchmarks.golden.datasets import signal_sha256


def _case(outputs, *, signal="abc"):
    identity = {"adapter": "a", "window": [0, 10], "fs": 500.0, "units": "mV", "lead_names": ["II"],
                "shape": [1, 10], "signal_sha256": signal, "signal_meta_sha256": "m", "source_files": []}
    return {"case_id": "cfg/ds/1", "input": identity, "outputs": outputs}


def _legacy(sha):
    return {"legacy": {p: {"sha256": sha, "bytes": 1, "leaf_digest": sha} for p in runner.LEGACY_PROFILES}}


def test_leaf_digest_ignores_key_order_but_not_values():
    assert runner.leaf_digest({"a": 1, "b": [1, {"c": 2}]}) == runner.leaf_digest({"b": [1, {"c": 2}], "a": 1})
    assert runner.leaf_digest({"a": 1}) != runner.leaf_digest({"a": 1.0000001})
    assert runner.leaf_digest({"a": {}}) != runner.leaf_digest({"a": []})


def test_legacy_bytes_keep_exporter_key_order():
    assert runner.legacy_bytes({"b": 1, "a": 2}) != runner.legacy_bytes({"a": 2, "b": 1})
    with pytest.raises(TypeError):
        runner.legacy_bytes({"x": np.float64(1.0)} | {"y": object()})
    with pytest.raises(ValueError):
        runner.legacy_bytes({"x": float("nan")})


def test_normalization_touches_only_pinned_timestamps():
    payload = {"clinical_interpretation": {"generated_at": "t1", "other": "t1"},
               "metadata": {"clinical_interpretation": {"generated_at": "t2"}, "generated_at": "t3"}}
    runner.normalize_legacy(payload)
    assert payload["clinical_interpretation"] == {"generated_at": runner.NORMALIZED_VALUE, "other": "t1"}
    assert payload["metadata"]["clinical_interpretation"]["generated_at"] == runner.NORMALIZED_VALUE
    assert payload["metadata"]["generated_at"] == "t3"


def test_signal_hash_is_dtype_and_layout_canonical():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert signal_sha256(values) == signal_sha256(np.asfortranarray(values.astype(np.float64)))
    assert signal_sha256(values) != signal_sha256(values.T)


def test_compare_detects_byte_diff_and_integrity_error():
    want = _case(_legacy("x"))
    assert compare.compare_case("legacy-bytes", want, _case(_legacy("x")), baseline_payloads=None,
                                candidate_payloads=None)["status"] == "pass"
    diff = compare.compare_case("legacy-bytes", want, _case(_legacy("y")), baseline_payloads=None,
                                candidate_payloads=None)
    assert diff["status"] == "diff" and len(diff["differences"]) == len(runner.LEGACY_PROFILES)
    broken = compare.compare_case("legacy-bytes", want, _case(_legacy("x"), signal="changed"),
                                  baseline_payloads=None, candidate_payloads=None)
    assert broken["status"] == "integrity_error" and broken["integrity"] == ["signal_sha256"]


def test_frozen_failures_must_reproduce_exactly():
    want = _case({"record": {"error": "ComputationInvariantError: x"}})
    same = compare.compare_case("record-bytes", want, _case({"record": {"error": "ComputationInvariantError: x"}}),
                                baseline_payloads=None, candidate_payloads=None)
    fixed = compare.compare_case("record-bytes", want, _case({"record": {"summary": {"sha256": "s"}}}),
                                 baseline_payloads=None, candidate_payloads=None)
    assert same["status"] == "pass" and fixed["status"] == "diff"


def test_selection_rule_is_output_independent_and_stable():
    assert manifest.rank_key("ludb", "1") == manifest.rank_key("ludb", "1")
    assert manifest.rank_key("ludb", "1") != manifest.rank_key("qtdb", "1")
    assert sum(manifest.SENTINEL_COUNTS.values()) == 96


def test_no_all_flags_refinement_configuration():
    for name, spec in manifest.CONFIGS.items():
        assert sum(spec["refinement"].values()) <= 1, name
    defaults = manifest.CONFIGS["standard-default-v0"]
    assert defaults == {"fs_internal": None, "mains_freq": 50, "input_mode": "standard",
                        "st_amplitude_source": "analysis", "refinement": {}}


def test_pointer_diff_reports_changed_added_removed():
    report = compare.pointer_diff({"a": 1, "b": {"c": 2}}, {"a": 2, "b": {"d": 2}})
    assert (report["changed"], report["removed"], report["added"]) == (1, 1, 1)
    assert json.dumps(report)

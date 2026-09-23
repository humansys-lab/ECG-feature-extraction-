"""Synthetic tests for the benchmark-harness compare logic (no datasets needed).

Run: ``.venv/bin/python -m pytest benchmarks/harness/test_harness.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.harness.compare import compare_dirs, compare_tool
from benchmarks.harness.core import (
    FileHashCache,
    HarnessError,
    compare_metric,
    dataset_manifest,
    finalize_metrics,
    info,
    metric,
    tolerance_for,
)


def _base(m):
    return finalize_metrics([m])[0]


# ── tolerance application per family ────────────────────────────────────────


def test_bounded_accuracy_aggregate_allows_half_point_only():
    base = _base(metric("se", 0.900, family="bounded_accuracy", unit="fraction"))
    assert base["tolerance"]["allowed_regression"] == pytest.approx(0.005)
    assert compare_metric(base, dict(base, value=0.8951))["status"] == "pass"
    assert compare_metric(base, dict(base, value=0.8950))["status"] == "pass"  # exactly at the limit
    assert compare_metric(base, dict(base, value=0.8949))["status"] == "regression"
    assert compare_metric(base, dict(base, value=0.95))["status"] == "improvement"


def test_bounded_accuracy_subgroup_allows_one_point():
    base = _base(metric("se.lead_V1", 0.80, family="bounded_accuracy", unit="fraction", level="subgroup"))
    assert base["tolerance"]["allowed_regression"] == pytest.approx(0.010)
    assert compare_metric(base, dict(base, value=0.791))["status"] == "pass"
    assert compare_metric(base, dict(base, value=0.789))["status"] == "regression"


def test_error_uses_max_of_two_percent_and_quantum():
    big = _base(metric("mae", 20.0, family="error", unit="ms", quantum=0.1))
    assert big["tolerance"]["allowed_regression"] == pytest.approx(0.4)  # 2 % of 20 ms > 0.1 ms
    assert compare_metric(big, dict(big, value=20.39))["status"] == "pass"
    assert compare_metric(big, dict(big, value=20.41))["status"] == "regression"
    small = _base(metric("mae_small", 2.0, family="error", unit="ms", quantum=0.1))
    assert small["tolerance"]["allowed_regression"] == pytest.approx(0.1)  # quantum dominates
    assert compare_metric(small, dict(small, value=2.09))["status"] == "pass"
    assert compare_metric(small, dict(small, value=2.11))["status"] == "regression"
    assert compare_metric(small, dict(small, value=1.5))["status"] == "improvement"


def test_signed_bias_compared_on_absolute_value():
    base = _base(metric("bias", -5.0, family="error", unit="ms", quantum=0.1, compare_on="abs"))
    assert base["tolerance"]["allowed_regression"] == pytest.approx(0.1)
    assert compare_metric(base, dict(base, value=5.05))["status"] == "pass"  # sign flip, same size
    assert compare_metric(base, dict(base, value=-5.2))["status"] == "regression"
    assert compare_metric(base, dict(base, value=1.0))["status"] == "improvement"


def test_coverage_fraction_and_count():
    frac = _base(metric("cov", 0.95, family="coverage", unit="fraction"))
    assert compare_metric(frac, dict(frac, value=0.9451))["status"] == "pass"
    assert compare_metric(frac, dict(frac, value=0.9449))["status"] == "regression"
    count = _base(metric("n", 1000, family="coverage", unit="count"))
    assert count["tolerance"]["allowed_regression"] == pytest.approx(5.0)
    assert compare_metric(count, dict(count, value=995))["status"] == "pass"
    assert compare_metric(count, dict(count, value=994))["status"] == "regression"


def test_failure_count_allows_zero_new_failures():
    base = _base(metric("failed", 2, family="failure_count", unit="count"))
    assert compare_metric(base, dict(base, value=2))["status"] == "pass"
    assert compare_metric(base, dict(base, value=3))["status"] == "new_failure"
    assert compare_metric(base, dict(base, value=0))["status"] == "improvement"


def test_exact_results_allow_no_difference():
    base = _base(metric("digest", "abc", family="exact", unit="sha256"))
    assert compare_metric(base, dict(base, value="abc"))["status"] == "pass"
    assert compare_metric(base, dict(base, value="abd"))["status"] == "exact_diff"


def test_ungated_values_never_fail():
    base = _base(info("runtime", 12.0, unit="s"))
    assert base["tolerance"] is None
    row = compare_metric(base, dict(base, value=99.0))
    assert row["status"] == "informational" and row["changed"] is True


def test_missing_and_unavailable_metrics_fail():
    base = _base(metric("se", 0.9, family="bounded_accuracy", unit="fraction"))
    assert compare_metric(base, None)["status"] == "missing"
    assert compare_metric(base, dict(base, value=None))["status"] == "became_unavailable"
    none_base = _base(metric("se2", None, family="bounded_accuracy", unit="fraction"))
    assert compare_metric(none_base, dict(none_base, value=0.1))["status"] == "pass"


def test_definition_change_is_a_failure():
    base = _base(metric("mae", 5.0, family="error", unit="ms", quantum=0.1))
    changed = dict(base, unit="s")
    assert compare_metric(base, changed)["status"] == "definition_changed"


def test_tolerance_uses_baseline_not_candidate():
    base = _base(metric("mae", 10.0, family="error", unit="ms", quantum=0.1))
    cand = dict(base, value=10.5, tolerance={"kind": "absolute", "allowed_regression": 100.0})
    assert compare_metric(base, cand)["status"] == "regression"


def test_error_metric_requires_quantum_and_known_family():
    with pytest.raises(HarnessError):
        metric("mae", 1.0, family="error", unit="ms")
    with pytest.raises(HarnessError):
        metric("x", 1.0, family="made_up", unit="ms")
    with pytest.raises(HarnessError):
        finalize_metrics([metric("a", 1, family="exact", unit=""), metric("a", 2, family="exact", unit="")])


def test_tolerance_for_percent_unit():
    assert tolerance_for(metric("p", 90.0, family="bounded_accuracy", unit="percent"))["allowed_regression"] == pytest.approx(0.5)


# ── tool-level compare: manifest, config, missing metrics ───────────────────


def _tool_result(tool_id="t", manifest="m" * 64, metrics=None, status="pinned", config="c"):
    metrics = metrics if metrics is not None else [
        metric("se", 0.9, family="bounded_accuracy", unit="fraction"),
        metric("mae", 10.0, family="error", unit="ms", quantum=0.1),
        metric("failed", 0, family="failure_count", unit="count"),
    ]
    return {
        "schema_version": "ecgfeat-benchmark-result.v1",
        "status": status,
        "tool": {"id": tool_id, "name": f"{tool_id}.py", "sha256": "s", "config_digest": config},
        "datasets": [{"name": "D", "manifest": {"sha256": manifest}, "records_sha256": "r"}],
        "dataset_manifest_sha256": manifest,
        "metrics": finalize_metrics(metrics),
        "digests": {},
    }


def test_identical_candidate_passes():
    base = _tool_result()
    assert compare_tool(base, json.loads(json.dumps(base)))["passed"] is True


def test_manifest_mismatch_fails_before_scoring():
    base = _tool_result()
    cand = _tool_result(manifest="x" * 64)
    # Even a candidate with better metrics must fail when the corpus changed.
    for m in cand["metrics"]:
        if m["name"] == "se":
            m["value"] = 0.99
    report = compare_tool(base, cand)
    assert report["passed"] is False
    assert report["status"] == "manifest_mismatch"
    assert "failures" not in report  # metrics were not scored


def test_config_mismatch_fails():
    report = compare_tool(_tool_result(), _tool_result(config="other"))
    assert report["passed"] is False and report["status"] == "config_mismatch"


def test_missing_metric_fails_tool():
    base = _tool_result()
    cand = _tool_result()
    cand["metrics"] = [m for m in cand["metrics"] if m["name"] != "mae"]
    report = compare_tool(base, cand)
    assert report["passed"] is False
    assert [row["metric"] for row in report["failures"]] == ["mae"]
    assert report["failures"][0]["status"] == "missing"


def test_new_failure_fails_tool():
    cand = _tool_result()
    for m in cand["metrics"]:
        if m["name"] == "failed":
            m["value"] = 1
    report = compare_tool(_tool_result(), cand)
    assert report["passed"] is False and report["counts"]["new_failure"] == 1


def test_blocked_baseline_is_ungated_not_passing():
    base = {"status": "blocked", "tool": {"id": "t"}, "reason": "dataset unavailable"}
    report = compare_tool(base, None)
    assert report["passed"] is None and report["status"] == "ungated"


def test_blocked_candidate_for_pinned_tool_fails():
    cand = {"status": "blocked", "tool": {"id": "t"}, "reason": "dataset unavailable"}
    report = compare_tool(_tool_result(), cand)
    assert report["passed"] is False and report["status"] == "candidate_blocked"


def test_missing_candidate_for_pinned_tool_fails():
    assert compare_tool(_tool_result(), None)["passed"] is False


def test_compare_dirs_exit_semantics(tmp_path: Path):
    bdir, cdir = tmp_path / "b", tmp_path / "c"
    bdir.mkdir()
    cdir.mkdir()
    (bdir / "t.json").write_text(json.dumps(_tool_result()))
    (bdir / "u.json").write_text(json.dumps(_tool_result("u")))
    (cdir / "t.json").write_text(json.dumps(_tool_result()))
    worse = _tool_result("u")
    for m in worse["metrics"]:
        if m["name"] == "mae":
            m["value"] = 10.5
    (cdir / "u.json").write_text(json.dumps(worse))
    report = compare_dirs(bdir, cdir)
    assert report["passed"] is False
    assert report["summary"]["tools_failed"] == ["u"]
    assert report["summary"]["regressions"] == 1
    assert compare_dirs(bdir, cdir, tools=["t"])["passed"] is True


# ── dataset manifest hashing ────────────────────────────────────────────────


def test_dataset_manifest_changes_with_content_and_fails_on_missing(tmp_path: Path):
    (tmp_path / "a.dat").write_bytes(b"1234")
    (tmp_path / "b.hea").write_text("hdr")
    cache = FileHashCache(None)
    first = dataset_manifest(tmp_path, ["b.hea", "a.dat"], cache)
    assert first["file_count"] == 2
    assert dataset_manifest(tmp_path, ["a.dat", "b.hea"], cache)["sha256"] == first["sha256"]  # order-free
    (tmp_path / "a.dat").write_bytes(b"1235")
    import os
    os.utime(tmp_path / "a.dat", ns=(1, 1))  # defeat the (size, mtime) cache key explicitly
    assert dataset_manifest(tmp_path, ["a.dat", "b.hea"], cache)["sha256"] != first["sha256"]
    with pytest.raises(HarnessError):
        dataset_manifest(tmp_path, ["a.dat", "missing.dat"], cache)

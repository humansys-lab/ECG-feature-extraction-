"""Performance budget gate (benchmarks/performance/budget.json)."""

import json

from tools import check_performance


def test_reference_case_meets_the_absolute_budget():
    measured, failures = check_performance.reference(repeats=2)
    assert failures == [], (measured, failures)
    assert measured["summary_bytes"] <= 24000


def test_golden_timing_requires_a_both_surface_report(tmp_path):
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"tier": "sentinel", "surfaces": ["record"], "case_seconds": {"a": 1.0}}))
    measured, failures = check_performance.golden(report)
    assert failures and "legacy-bytes --with-record" in failures[0]


def test_golden_ratio_limits(tmp_path, monkeypatch):
    baseline = json.loads(open("benchmarks/golden/baselines/phase0-52a339c/expected.sentinel.json").read())
    cases = list(baseline["case_seconds"])[:20]
    report = tmp_path / "r.json"
    for factor, ok in ((1.0, True), (2.0, False)):
        report.write_text(json.dumps({"tier": "sentinel", "surfaces": ["legacy", "record"],
                                      "case_seconds": {c: baseline["case_seconds"][c] * factor for c in cases}}))
        measured, failures = check_performance.golden(report)
        assert (failures == []) is ok, (factor, measured, failures)

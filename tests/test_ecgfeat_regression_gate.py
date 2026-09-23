from __future__ import annotations

import csv
import json

from scripts.validate_ecgfeat_regression import compare_runs, numerically_equal


def _run(path, *, errors=(2.0, 4.0), qt=400.0, extra=False, failed=False):
    path.mkdir()
    features = {"qt_ms": qt, "qt_reportable": True}
    if extra:
        features["qt_dispersion_independent_ms"] = 40.0
    record = {"record": "1", "beats": 2, "global_features": features,
              "failures": ["failed"] if failed else []}
    (path / "records.jsonl").write_text(json.dumps(record) + "\n")
    with (path / "ludb_errors.csv").open("w") as handle:
        fields = ["record", "lead", "r_sample", "qrs_on_err", "qrs_off_err",
                  "p_on_err", "p_off_err", "t_on_err", "t_off_err", "qt_err"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, error in enumerate(errors):
            writer.writerow({"record": "1", "lead": "II", "r_sample": 100+index*500,
                             **{field: error for field in fields[3:]}})
    with (path / "coverage.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record", "lead", "gt_qrs", "matched_qrs"])
        writer.writeheader()
        writer.writerow({"record":"1","lead":"II","gt_qrs":2,"matched_qrs":len(errors)})


def test_gate_accepts_additive_fields_without_changing_established_values(tmp_path):
    _run(tmp_path/"before")
    _run(tmp_path/"after", extra=True)
    assert compare_runs(tmp_path/"before", tmp_path/"after")["passed"]


def test_gate_rejects_worse_patient_errors_even_when_mean_error_is_unchanged(tmp_path):
    _run(tmp_path/"before")
    _run(tmp_path/"after", errors=(4.0, 2.0))
    result = compare_runs(tmp_path/"before", tmp_path/"after")
    assert not result["passed"]
    assert result["changed_boundary_errors"] == 14


def test_gate_rejects_missing_hard_examples_even_when_mae_improves(tmp_path):
    _run(tmp_path/"before")
    _run(tmp_path/"after", errors=(2.0,))
    result = compare_runs(tmp_path/"before", tmp_path/"after")
    assert not result["passed"]
    assert any(f["kind"] == "coverage_count_changed" for f in result["failures"])


def test_gate_rejects_changed_global_values_and_operational_failures(tmp_path):
    _run(tmp_path/"before")
    _run(tmp_path/"after", qt=420.0, failed=True)
    result = compare_runs(tmp_path/"before", tmp_path/"after")
    assert not result["passed"]
    assert {f["kind"] for f in result["failures"]} >= {"failed_record", "global_value_changed"}


def test_boolean_gate_is_not_equivalent_to_a_numeric_value():
    assert not numerically_equal({"reportable": True}, {"reportable": 1})

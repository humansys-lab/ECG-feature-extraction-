#!/usr/bin/env python3
"""Run LUDB and fail if established measurements regress against a saved run.

Accepts the review artifact format (records.jsonl, ludb_errors.csv, coverage.csv).
Default measurement fields must remain equal within numerical tolerance; the
gate intentionally does not approve trading one patient's accuracy for another.
New additive fields are allowed and require their own validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ERROR_FIELDS = ("qrs_on_err", "qrs_off_err", "p_on_err", "p_off_err",
                "t_on_err", "t_off_err", "qt_err")


def numerically_equal(before, after, *, atol=1e-9):
    if isinstance(before, dict):
        return isinstance(after, dict) and all(
            key in after and numerically_equal(value, after[key], atol=atol)
            for key, value in before.items())
    if isinstance(before, list):
        return isinstance(after, list) and len(before) == len(after) and all(
            numerically_equal(a, b, atol=atol) for a, b in zip(before, after))
    if isinstance(before, bool):
        return isinstance(after, bool) and before == after
    if before is None:
        return after is None
    if isinstance(before, (int, float)):
        return (isinstance(after, (int, float)) and not isinstance(after, bool)
                and math.isfinite(before) and math.isfinite(after)
                and math.isclose(before, after, rel_tol=0.0, abs_tol=atol))
    return before == after


def _csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _json_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare_runs(baseline: Path, candidate: Path):
    """A stricter-than-average regression gate, with per-field failure details."""
    failures = []
    before_records = _json_records(baseline / "records.jsonl")
    after_records = _json_records(candidate / "records.jsonl")
    before = {r["record"]: r for r in before_records}
    after = {r["record"]: r for r in after_records}
    if not before or len(before) != len(before_records) or len(after) != len(after_records):
        failures.append({"kind": "empty_or_duplicate_record_ids"})
    if before.keys() != after.keys():
        failures.append({"kind": "record_coverage", "missing": sorted(before.keys()-after.keys()),
                         "extra": sorted(after.keys()-before.keys())})
    for rid in before.keys() & after.keys():
        new, old = after[rid], before[rid]
        if new.get("failures") or old.get("failures"):
            failures.append({"kind": "failed_record", "record": rid})
        if not numerically_equal(old.get("beats"), new.get("beats")):
            failures.append({"kind": "beat_count", "record": rid})
        old_features, new_features = old.get("global_features", {}), new.get("global_features", {})
        if not old_features or not new_features:
            failures.append({"kind": "missing_global_features", "record": rid})
        for key, value in old_features.items():
            if key not in new_features or not numerically_equal(value, new_features[key]):
                failures.append({"kind": "global_value_changed", "record": rid,
                                 "field": key, "before": value, "after": new_features.get(key)})

    def event_key(row):
        return row["record"], row["lead"], float(row["r_sample"])

    before_rows = _csv_rows(baseline / "ludb_errors.csv")
    after_rows = _csv_rows(candidate / "ludb_errors.csv")
    before_events = {event_key(row): row for row in before_rows}
    after_events = {event_key(row): row for row in after_rows}
    if len(before_events) != len(before_rows) or len(after_events) != len(after_rows):
        failures.append({"kind": "duplicate_matched_events"})
    lost = before_events.keys() - after_events.keys()
    added = after_events.keys() - before_events.keys()
    if lost or added:
        failures.append({"kind": "matched_event_set_changed", "lost": len(lost), "added": len(added)})
    changed_errors = 0
    for key in before_events.keys() & after_events.keys():
        for field in ERROR_FIELDS:
            a, b = before_events[key][field], after_events[key][field]
            old = float(a) if a else None
            new = float(b) if b else None
            if not numerically_equal(old, new):
                changed_errors += 1
                if len(failures) < 100:
                    failures.append({"kind": "boundary_error_changed", "event": key,
                                     "field": field, "before": old, "after": new})
    before_coverage = {(r["record"], r["lead"]): r for r in _csv_rows(baseline / "coverage.csv")}
    after_coverage = {(r["record"], r["lead"]): r for r in _csv_rows(candidate / "coverage.csv")}
    if before_coverage.keys() != after_coverage.keys():
        failures.append({"kind": "annotation_coverage_set_changed"})
    for key in before_coverage.keys() & after_coverage.keys():
        for field, value in before_coverage[key].items():
            # det_total includes unannotated edge beats; total record beats
            # are already checked above. Other count fields must agree.
            if field in {"record", "lead", "det_total"}:
                continue
            if field not in after_coverage[key] or int(value) != int(after_coverage[key][field]):
                failures.append({"kind": "coverage_count_changed", "lead_record": key, "field": field})
    return {"passed": not failures, "baseline_records": len(before), "candidate_records": len(after),
            "baseline_matched_rows": len(before_rows), "candidate_matched_rows": len(after_rows),
            "changed_boundary_errors": changed_errors,
            "policy": "existing global values, matched errors and coverage equal within 1e-9; additive fields allowed",
            "failures": failures}


def _evaluate(rid):
    import evaluate_ludb as ev

    class Capture:
        result = None

        def extract(self, ecg, fs):
            self.result = ev.ECGFeatureExtractor(fs_internal=500, mains_freq=50).extract(ecg, fs)
            return self.result

    capture = Capture()
    failures, coverage = [], []
    started = time.perf_counter()
    rows = ev.evaluate_record(rid, capture, failure_details=failures, coverage_rows=coverage)
    record = {"record": rid, "seconds": time.perf_counter()-started,
              "failures": failures, "matched_rows": len(rows)}
    if capture.result is not None:
        record.update(global_features=asdict(capture.result.global_features),
                      beats=len(capture.result.beats),
                      record_quality=capture.result.metadata.get("record_quality"))
    return record, rows, coverage


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _source_hashes():
    paths = [*(ROOT/"feature_extraction/ecgfeat").rglob("*.py"),
             ROOT/"evaluate_ludb.py", Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() == args.baseline.resolve() or args.workers < 1:
        parser.error("out must differ from baseline and workers must be positive")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.compare_only:
        import numpy as np
        import scipy
        import wfdb
        import evaluate_ludb as ev
        records = [r["record"] for r in _json_records(args.baseline / "records.jsonl")]
        all_records, all_rows, all_coverage = [], [], []
        source_hashes = _source_hashes()
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            with (args.out / "records.jsonl").open("w") as handle:
                for record, rows, coverage in pool.map(_evaluate, records):
                    all_records.append(record); all_rows.extend(rows); all_coverage.extend(coverage)
                    handle.write(json.dumps(record) + "\n"); handle.flush()
                    print(f"{len(all_records)}/{len(records)} record={record['record']} pairs={len(rows)}", flush=True)
        _write_csv(args.out / "ludb_errors.csv", all_rows)
        _write_csv(args.out / "coverage.csv", all_coverage)
        summary = ev.compute_summary(all_rows, coverage_rows=all_coverage)
        summary.update(records=len(records), wall_seconds=time.perf_counter()-started,
                       workers=args.workers, settings={"fs_internal":500,"mains_freq":50,
                                                       "st_amplitude_source":"analysis","patient_meta":None},
                       versions={"python":platform.python_version(),"numpy":np.__version__,
                                 "scipy":scipy.__version__,"wfdb":wfdb.__version__},
                       source_sha256=source_hashes)
        if source_hashes != _source_hashes():
            raise RuntimeError("Extraction/evaluation sources changed during validation; rerun on stable sources")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    result = compare_runs(args.baseline, args.out)
    (args.out / "regression_gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "failures"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

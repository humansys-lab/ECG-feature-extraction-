#!/usr/bin/env python3
"""Fit on a record-disjoint development partition; report remainder performance."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feature_extraction"))
from ecgfeat.calibration import QTErrorCalibration, confidence_bin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Development cohort manifest with record IDs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = set(json.loads(args.manifest.read_text())["records"])
    rows = list(csv.DictReader(args.errors.open()))
    train = [r for r in rows if r["record"] in records]
    test = [r for r in rows if r["record"] not in records]
    if not train or not test:
        parser.error("both training and record-disjoint validation rows are required")
    model = QTErrorCalibration.fit(train, provenance={
        "errors_sha256": hashlib.sha256(args.errors.read_bytes()).hexdigest(),
        "errors_path": str(args.errors), "split_manifest": str(args.manifest),
        "scope": "LUDB internal validation; default pipeline; missing outcomes excluded"})
    def finite(row):
        try:
            return math.isfinite(float(row["qt_err"]))
        except (TypeError, ValueError):
            return False
    valid = [r for r in test if finite(r)]
    if not valid:
        parser.error("validation partition has no finite QT error")
    counts = Counter(r["record"] for r in valid)
    bins = []
    brier = 0.
    constant_brier = 0.
    for key in ["missing", "0", "1", "2", "3", "4"]:
        subset = [r for r in valid if confidence_bin(r["qt_conf"]) == key]
        weight = sum(1 / counts[r["record"]] for r in subset)
        observed = sum((abs(float(r["qt_err"])) > model.threshold_ms) / counts[r["record"]] for r in subset)
        probability = model.risk_by_bin[key]
        for r in subset:
            label = float(abs(float(r["qt_err"])) > model.threshold_ms)
            brier += (probability-label)**2 / counts[r["record"]]
            constant_brier += (model.global_risk-label)**2 / counts[r["record"]]
        bins.append({"bin": key, "rows": len(subset), "records": len({r["record"] for r in subset}),
                     "predicted_risk": probability, "observed_risk": observed / weight if weight else None})
    report = {"training_records": len(model.training_records), "validation_records": len(counts),
              "validation_matched_rows": len(test), "validation_finite_qt_rows": len(valid),
              "record_weighted_brier": brier / len(counts),
              "constant_training_risk_brier": constant_brier / len(counts), "bins": bins,
              "caveat": "LUDB was used for historical algorithm development; external calibration is still required"}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "model.json").write_text(json.dumps(model.to_dict(), indent=2))
    (args.out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

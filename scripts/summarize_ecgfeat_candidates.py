#!/usr/bin/env python3
"""Paired errors, coverage and record-cluster uncertainty for saved experiments."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ecgfeat_improvements_20260922"
BASE = ROOT / "ecgfeat_validation_fixes_20260908/final"
ERRORS = ["p_on_err", "p_off_err", "qrs_on_err", "qrs_off_err", "t_on_err", "t_off_err", "qt_err"]


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def paired(before, after, key, field):
    old, new = ({tuple(r[k] for k in key): r for r in table} for table in (before, after))
    per_record = defaultdict(list)
    a, b = [], []
    for identity in old.keys() & new.keys():
        x, y = number(old[identity].get(field)), number(new[identity].get(field))
        if x is None or y is None:
            continue
        a.append(abs(x))
        b.append(abs(y))
        per_record[old[identity]["record"]].append(abs(y) - abs(x))
    if not a:
        return {"n": 0}
    record_deltas = np.asarray([np.mean(per_record[r]) for r in sorted(per_record)])
    rng = np.random.default_rng(20260922)
    ci = np.percentile(np.mean(rng.choice(record_deltas, size=(5000, len(record_deltas)), replace=True), axis=1), [2.5, 97.5])
    return {"n": len(a), "paired_mae_before": float(np.mean(a)), "paired_mae_after": float(np.mean(b)),
            "record_macro_mae_delta": float(np.mean(record_deltas)), "record_bootstrap_delta_ci95": ci.tolist(),
            "improved_records": int(np.sum(record_deltas < -1e-9)),
            "worsened_records": int(np.sum(record_deltas > 1e-9)),
            "unchanged_records": int(np.sum(np.abs(record_deltas) <= 1e-9)),
            "before_finite": sum(number(r.get(field)) is not None for r in before),
            "after_finite": sum(number(r.get(field)) is not None for r in after)}


def coverage(folder):
    table = rows(folder / "coverage.csv")
    return {wave: {"gt": sum(int(r['gt_' + wave]) for r in table),
                   "scored": sum(int(r['scored_' + wave]) for r in table)}
            for wave in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}


def main():
    result = {"method": "paired common events; bootstrap record macro error delta, 5000 resamples; not adjusted for multiple candidates"}
    for cohort, baseline in (("development", OUT / "development/default"), ("ludb_full", BASE)):
        before = rows(baseline / "ludb_errors.csv")
        result[cohort] = {}
        for folder in sorted((OUT / cohort).iterdir()):
            if not folder.is_dir():
                continue
            if cohort == "ludb_full" and folder.name == "all" and (OUT / "ludb_all_final/all/summary.json").exists():
                folder = OUT / "ludb_all_final/all"
            summary = json.loads((folder / "summary.json").read_text())
            after = rows(folder / "ludb_errors.csv")
            result[cohort][folder.name] = {
                "summary": {key: summary[key] for key in ERRORS}, "coverage": coverage(folder),
                "paired": {key: paired(before, after, ("record", "lead", "r_sample"), key) for key in ERRORS}}
    before = rows(OUT / "edb/analysis/events.csv")
    result["edb"] = {}
    for mode in ("analysis", "calibrated_pr", "adaptive_pr_tp"):
        path = OUT / "edb" / mode
        summary = json.loads((path / "summary.json").read_text())
        after = rows(path / "events.csv")
        hybrid = next(r for r in summary["metrics"] if r["method"] == "hybrid" and r["comparison"] == "peak_ST_deviation")
        classification = next(r for r in summary["metrics"] if r["method"] == "hybrid" and "threshold" in r["comparison"])
        result["edb"][mode] = {"hybrid": hybrid, "classification_conditional_on_output": classification,
            "event_output_fraction": hybrid["n"] / summary["event_rows"],
            "positive_event_detection_fraction_all_selected": classification["tp"] / summary["event_rows"],
            "paired": paired(before, after, ("record", "lead_index", "peak_sample"), "hybrid_deviation_error_mv")}
    (OUT / "comparisons.json").write_text(json.dumps(result, indent=2))
    # Stratify ventricular boundaries with annotation widths, not a width
    # predicted by the algorithm under comparison. P labels use the following
    # QRS in LUDB, so do not attach this QRS stratum to P errors.
    sys.path.insert(0, str(ROOT))
    import evaluate_ludb as ev
    labels = {}
    denominators = {name: {"qrs": 0, "t_off": 0} for name in ("qrs_lt_120ms", "qrs_ge_120ms")}
    for record in range(1, 201):
        for lead, extension in ev.LEAD_TO_ANN_EXT.items():
            for annotation in ev.parse_ludb_annotations(str(ev.LUDB_DIR / str(record)), extension):
                on, off = annotation.get("qrs_on"), annotation.get("qrs_off")
                if on is None or off is None:
                    continue
                label = "qrs_ge_120ms" if (off - on) * 2 >= 120 else "qrs_lt_120ms"
                labels[(str(record), lead, str(float(annotation["r_sample"])))] = label
                denominators[label]["qrs"] += 1
                denominators[label]["t_off"] += annotation.get("t_off") is not None
    stratified = {"scope": "LUDB 500Hz ground-truth lead-local QRS duration; annotated spans; matched-event errors",
                  "annotation_denominators": denominators, "variants": {}}
    for name, folder in [("default", BASE), *[(v, OUT / "ludb_full" / v) for v in ("t_bidirectional", "t_correlated_fusion")],
                         ("all", OUT / "ludb_all_final/all" if (OUT / "ludb_all_final/all/summary.json").exists() else OUT / "ludb_full/all")]:
        table = rows(folder / "ludb_errors.csv")
        strata = {}
        for label in denominators:
            subset = [r for r in table if labels.get((r["record"], r["lead"], r["r_sample"])) == label]
            metrics = {}
            for field in ("qrs_on_err", "qrs_off_err", "t_on_err", "t_off_err", "qt_err"):
                values = [abs(v) for r in subset if (v := number(r.get(field))) is not None]
                metrics[field] = {"n": len(values), "mae": float(np.mean(values)) if values else None,
                                  "p95": float(np.percentile(values, 95)) if values else None}
            strata[label] = metrics
        stratified["variants"][name] = strata
    (OUT / "stratified.json").write_text(json.dumps(stratified, indent=2))
    print(OUT / "comparisons.json")


if __name__ == "__main__":
    main()

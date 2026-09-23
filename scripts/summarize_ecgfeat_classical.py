#!/usr/bin/env python3
"""Summarize frozen classical ablations with paired errors and coverage."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_ecgfeat_candidates import ERRORS, coverage, paired, rows


def summarize(root):
    output = {"uncertainty": "record-cluster bootstrap, 5000 draws; no multiple-comparison correction",
              "scope": "LUDB used in historical development; QTDB is a previously inspected cross-dataset test"}
    for cohort in ("development", "ludb_full", "qtdb"):
        directory = root / cohort
        if not (directory / "manifest.json").exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        is_qt = cohort == "qtdb"
        filename = "matched_events.csv" if is_qt else "ludb_errors.csv"
        before = rows(directory / "default" / filename)
        output[cohort] = {"records": len(manifest["records"]), "variants": {}}
        for variant in manifest["variants"]:
            folder = directory / variant
            summary = json.loads((folder / "summary.json").read_text())
            after = rows(folder / filename)
            result = {"summary": summary}
            if is_qt:
                result["paired"] = {}
                for wave in ("P", "QRS", "T"):
                    a = [r for r in before if r["wave"] == wave]
                    b = [r for r in after if r["wave"] == wave]
                    result["paired"][wave] = {
                        field: paired(a, b, ("record", "lead", "wave", "gt_peak_sample"), field)
                        for field in ("onset_error_ms", "offset_error_ms")}
            else:
                result["coverage"] = coverage(folder)
                result["paired"] = {field: paired(before, after, ("record", "lead", "r_sample"), field)
                                    for field in ERRORS}
            output[cohort]["variants"][variant] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    path = args.root / "comparisons.json"
    path.write_text(json.dumps(result, indent=2))
    print(path)


if __name__ == "__main__":
    main()

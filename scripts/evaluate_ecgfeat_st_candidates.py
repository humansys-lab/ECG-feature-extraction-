#!/usr/bin/env python3
"""Compare ST amplitude paths on the same EDB records and event protocol."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "feature_extraction"))
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


def task(payload):
    record, mode = payload
    import analyze_physionet_st as ev
    from unittest.mock import patch
    args = ev.build_parser().parse_args(["--dataset", "edb", "--no-plots"])
    args.plot_examples = 0
    start = time.perf_counter()
    original = ev.ECGFeatureExtractor
    try:
        with patch.object(ev, "ECGFeatureExtractor", lambda **kwargs: original(**kwargs, st_amplitude_source=mode)):
            events, controls = ev._analyse_edb_record(ROOT / "data/edb" / record,
                                                      args=args, plot_state={"edb": 0, "ltstdb": 0})
        return mode, record, events, controls, None, time.perf_counter() - start
    except Exception as exc:
        return mode, record, [], [], f"{type(exc).__name__}: {exc}", time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--records", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=("analysis", "calibrated_pr", "adaptive_pr_tp"),
                        default=["analysis", "calibrated_pr", "adaptive_pr_tp"])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.out.exists():
        parser.error("use a new output directory")
    import analyze_physionet_st as ev
    records = args.records or [p.stem for p in sorted((ROOT / "data/edb").glob("*.hea"))]
    args.out.mkdir(parents=True)
    source_paths = [ROOT / "analyze_physionet_st.py", *sorted((ROOT / "feature_extraction/ecgfeat").rglob("*.py"))]
    manifest = {"records": records, "modes": args.modes,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "protocol": {"max_events_per_record": 3, "controls_per_record": 1, "window_sec": 12,
                             "reference_sec": 30, "scope": "sparse physical leads; not 12-lead clinical validation"}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = {m: {"events": [], "controls": [], "records": []} for m in args.modes}
    tasks = [(r, m) for r in records for m in args.modes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (mode, record, events, controls, failure, elapsed) in enumerate(pool.map(task, tasks), 1):
            target = results[mode]
            target["events"].extend(events)
            target["controls"].extend(controls)
            target["records"].append({"record": record, "failure": failure, "seconds": elapsed,
                                      "event_rows": len(events), "control_rows": len(controls)})
            # Persist incremental data so failed runs remain auditable.
            with (args.out / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps({"mode": mode, "record": record, "failure": failure,
                                         "events": events, "controls": controls}) + "\n")
            print(f"{i}/{len(tasks)} {mode} {record} failure={failure}", flush=True)
    for mode, result in results.items():
        folder = args.out / mode
        folder.mkdir()
        ev._write_csv(folder / "events.csv", result["events"])
        ev._write_csv(folder / "controls.csv", result["controls"])
        metrics = ev.build_metrics(edb_events=result["events"], edb_controls=result["controls"],
                                   ltst_events=[], ltst_controls=[])
        summary = {"metrics": [r for r in metrics if r["dataset"] == "edb"],
                   "records": result["records"], "event_rows": len(result["events"]),
                   "control_rows": len(result["controls"])}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    return int(any(r["failure"] for value in results.values() for r in value["records"]))


if __name__ == "__main__":
    raise SystemExit(main())

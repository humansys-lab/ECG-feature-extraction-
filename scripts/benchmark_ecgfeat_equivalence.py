#!/usr/bin/env python3
"""Sequential isolated A/B timings plus exact, unrounded export comparison."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONHASHSEED"] = "0"


def child(args):
    sys.path.insert(0, str(args.source_root / "feature_extraction"))
    import numpy as np
    import scipy
    import wfdb
    from ecgfeat import ECGFeatureExtractor, RefinementConfig
    from ecgfeat.export import to_dict, prepare_json_export
    import inspect
    signal, header = wfdb.rdsamp(str(ROOT / "data/lobachevsky-university-electrocardiography-database-1.0.1/data" / args.record))
    options = {} if args.variant == "default" else {"refinement": RefinementConfig(**{args.variant: True})}
    extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50, **options)
    extractor.extract(signal.T, header["fs"])
    walls, cpus, exports, direct_exports = [], [], [], []
    direct = "profile" in inspect.signature(to_dict).parameters
    for _ in range(args.repeats):
        wall, cpu = time.perf_counter(), time.process_time()
        result = extractor.extract(signal.T, header["fs"])
        walls.append(time.perf_counter() - wall)
        cpus.append(time.process_time() - cpu)
        wall = time.perf_counter()
        full = to_dict(result)
        summary = prepare_json_export(full, profile="summary", round_ndigits=None)
        exports.append(time.perf_counter() - wall)
        if direct:
            wall = time.perf_counter()
            fast = prepare_json_export(to_dict(result, profile="summary"), profile="summary", round_ndigits=None)
            direct_exports.append(time.perf_counter() - wall)
            assert fast == summary, "direct summary changed content"
    def digest(payload):
        payload = deepcopy(payload)
        # Only the two wall-clock creation stamps are nondeterministic.
        # Keep all measurements, quality fields, provenance and audit data.
        payload.get("clinical_interpretation", {}).pop("generated_at", None)
        payload.get("metadata", {}).get("clinical_interpretation", {}).pop("generated_at", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    report = {"record": args.record, "source": str(args.source_root), "wall_seconds": walls, "cpu_seconds": cpus,
              "median_seconds": statistics.median(walls), "export_seconds": exports,
              "direct_export_seconds": direct_exports, "summary_sha256": digest(summary),
              "debug_sha256": digest(prepare_json_export(full, profile="debug", round_ndigits=None)),
              "beats": len(result.beats), "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                                       "scipy": scipy.__version__, "wfdb": wfdb.__version__}}
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--records", nargs="+", default=["1", "74", "125", "127", "57"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--record")
    parser.add_argument("--variant", default="default", help="single refinement flag, or default")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.record:
        return child(args)
    if args.baseline_source is None or args.out is None:
        parser.error("--baseline-source and --out required")
    report = {"records": [], "settings": {"threads": 1, "warmup": 1, "repeats": args.repeats, "variant": args.variant,
                                          "execution": "sequential isolated processes; order alternates by record",
                                          "excluded_fields": ["clinical_interpretation.generated_at", "metadata.clinical_interpretation.generated_at"]}}
    for i, record in enumerate(args.records):
        row = {"record": record}
        pair = [("baseline", args.baseline_source), ("current", args.source_root)]
        for name, source in (pair if i % 2 == 0 else pair[::-1]):
            output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "--record", record,
                "--source-root", str(source.resolve()), "--repeats", str(args.repeats), "--variant", args.variant], text=True)
            row[name] = json.loads(output)
        row["speedup"] = row["baseline"]["median_seconds"] / row["current"]["median_seconds"]
        row["debug_equal"] = row["baseline"]["debug_sha256"] == row["current"]["debug_sha256"]
        row["summary_equal"] = row["baseline"]["summary_sha256"] == row["current"]["summary_sha256"]
        report["records"].append(row)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"{record}: {row['speedup']:.3f}x; debug_equal={row['debug_equal']}; summary_equal={row['summary_equal']}", flush=True)
    return int(not all(r["debug_equal"] and r["summary_equal"] for r in report["records"]))


if __name__ == "__main__":
    raise SystemExit(main())

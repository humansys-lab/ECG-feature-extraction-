#!/usr/bin/env python3
"""Performance budget gate (benchmarks/performance/budget.json).

``reference``: time ``import`` and repeated extraction of the pinned reference
record in a fresh interpreter and check the absolute budget (time, peak RSS,
summary size).  ``golden``: compare per-case wall time of a golden check report
with the frozen Phase 0 baseline on the same runner.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = json.loads((ROOT / "benchmarks" / "performance" / "budget.json").read_text())

_PROBE = r"""
import json, resource, sys, time, warnings
warnings.simplefilter("ignore")
t0 = time.perf_counter()
from ecgfeat import dumps_record, ecg_record
import_s = time.perf_counter() - t0
import numpy as np
signal = np.load(sys.argv[1])["signal"]
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
times = []
for _ in range(int(sys.argv[2])):
    start = time.perf_counter()
    record = ecg_record(signal, sampling_rate=500, lead_names=leads)
    times.append(time.perf_counter() - start)
def peak_rss_mb():
    # ru_maxrss survives fork+exec on Linux (it would report the parent's
    # high-water mark); VmHWM belongs to this process image only.
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
print(json.dumps({"import_s": import_s, "times": times, "summary_bytes": len(dumps_record(record)),
                  "peak_rss_mb": peak_rss_mb()}))
"""


def reference(repeats: int) -> tuple[dict, list[str]]:
    signal = ROOT / "tests" / "fixtures" / "golden" / "reference_10s_12lead" / "signal.npz"
    out = subprocess.run([sys.executable, "-c", _PROBE, str(signal), str(repeats)], capture_output=True, text=True, check=True)
    probe = json.loads(out.stdout.strip().splitlines()[-1])
    limits = BUDGET["absolute"]
    measured = {
        "import_record_api_s": probe["import_s"],
        "first_extraction_s": probe["times"][0],
        "warm_extraction_median_s": statistics.median(probe["times"][1:] or probe["times"]),
        "peak_rss_mb": probe["peak_rss_mb"],
        "summary_bytes": probe["summary_bytes"],
    }
    failures = [f"{key}: {measured[key]:.2f} > {limits[key]}" for key in measured if measured[key] > limits[key]]
    return measured, failures


def golden(report_path: Path) -> tuple[dict, list[str]]:
    rules = BUDGET["relative_to_phase0_golden"]
    report = json.loads(report_path.read_text())
    if report.get("surfaces") != ["legacy", "record"]:
        return {}, ["timings compare only with a 'check --mode legacy-bytes --with-record' report "
                    "(the Phase 0 baseline timed both surfaces)"]
    tier = report["tier"]
    baseline = json.loads((ROOT / "benchmarks" / "golden" / "baselines" / rules["baseline_id"] / f"expected.{tier}.json").read_text())
    ratios = sorted(report["case_seconds"][case] / baseline["case_seconds"][case]
                    for case in report.get("case_seconds", {}) if baseline["case_seconds"].get(case))
    if not ratios:
        return {}, ["report has no per-case timings to compare"]
    measured = {"cases": len(ratios), "median_ratio": statistics.median(ratios),
                "p95_ratio": ratios[min(len(ratios) - 1, int(0.95 * len(ratios)))]}
    failures = []
    if measured["median_ratio"] > rules["median_ratio_max"]:
        failures.append(f"median per-case time ratio {measured['median_ratio']:.3f} > {rules['median_ratio_max']}")
    if measured["p95_ratio"] > rules["p95_ratio_max"]:
        failures.append(f"p95 per-case time ratio {measured['p95_ratio']:.3f} > {rules['p95_ratio_max']}")
    return measured, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--repeats", type=int, default=5)
    gold = sub.add_parser("golden")
    gold.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    measured, failures = reference(args.repeats) if args.mode == "reference" else golden(args.report)
    print(json.dumps({"mode": args.mode, "measured": measured, "failures": failures}, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot PTB-XL records the pipeline fails to call as flutter or AF.

Shows leads II and V1 -- where flutter sawtooth and fibrillatory waves are
most visible -- with the R peaks the detector found and, in the title, the
measurements each verdict actually hinges on, so a wrong call can be read
against what the trace shows.

Matplotlib here has no CJK font, so in-figure text is English by design.

Usage:
    python plot_af_afl_failures.py --records 10752 5252 --out-dir af_failure_plots
    python plot_af_afl_failures.py --auto-flutter 6 --auto-af 6
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "feature_extraction"))

PTBXL = Path("/workspace/ecg_gemma/data/ptb-xl")
META = "/workspace/ecg_gemma/data/ptb-xl-metadata-1.0.1/ptbxl_database.csv"
LEADS = ("II", "V1")
LEAD_IDX = {"I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
            "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}


def rpath(ecg_id: int) -> Path:
    return PTBXL / f"{(ecg_id // 1000) * 1000:05d}" / f"{ecg_id:05d}_hr"


def expert_labels(ecg_id: int) -> str:
    db = pd.read_csv(META, index_col="ecg_id")
    codes = ast.literal_eval(db.loc[ecg_id, "scp_codes"])
    return ",".join(sorted(codes))


def analyse(ecg_id: int):
    import wfdb
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    rec = wfdb.rdrecord(str(rpath(ecg_id)))
    fs = int(round(float(rec.fs)))
    sig12 = np.asarray(rec.p_signal, dtype=float).T
    feats = ECGFeatureExtractor(fs_internal=fs, mains_freq=50.0).extract(
        sig12, fs=float(fs)
    )
    r_locs = sorted({int(b.r_index) for b in feats.beats})
    ra = feats.metadata["rhythm_analysis"]
    return sig12, fs, r_locs, ra["af_afl_summary"], ra["atrial_residual"]


def plot_record(ecg_id: int, label: str, out_dir: Path, seconds: float = 6.0) -> Path:
    sig12, fs, r_locs, summary, residual = analyse(ecg_id)

    rr = np.diff(r_locs) * 1000.0 / fs if len(r_locs) > 1 else np.asarray([])
    hr = 60000.0 / float(np.median(rr)) if rr.size else float("nan")

    n = min(int(seconds * fs), sig12.shape[1])
    start = max(0, (sig12.shape[1] - n) // 2)
    stop = start + n
    t = np.arange(start, stop) / fs

    fig, axes = plt.subplots(len(LEADS), 1, figsize=(16, 3.4 * len(LEADS)), sharex=True)
    if len(LEADS) == 1:
        axes = [axes]

    for ax, lead in zip(axes, LEADS):
        ax.plot(t, sig12[LEAD_IDX[lead], start:stop], color="#111111", lw=1.2, zorder=3)
        for r in r_locs:
            if start <= r < stop:
                ax.axvline(r / fs, color="#cc4444", lw=0.9, alpha=0.55, zorder=1)
        ax.set_ylabel(f"{lead} (mV)", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)

    axes[-1].set_xlabel("time (s)  —  red lines = detected R peaks", fontsize=10)

    verdict = summary.get("atrial_rhythm_classification")
    narrow = residual.get("flutter_peak_narrowness")
    reasons = summary.get("indeterminate_reasons") or []
    line2 = (
        f"pipeline said: {verdict!r}"
        f"     RR cv {summary.get('rr_cv'):.3f} (AF needs >= 0.12)"
        f"     organized P {summary.get('organized_p_ratio'):.2f} "
        f"(AF needs <= 0.60)"
    )
    line3 = (
        f"flutter narrowness "
        f"{'n/a' if narrow is None else format(narrow, '.3f')} (needs >= 0.60)"
        f"     F-wave conf {summary.get('flutter_wave_confidence') or 0:.3f}"
        f"     median HR {hr:.0f} bpm"
        + (f"     abstained on: {', '.join(reasons)}" if reasons else "")
    )
    fig.suptitle(
        f"PTB-XL {ecg_id}  —  expert label: {label}   MISSED\n{line2}\n{line3}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ptbxl_{ecg_id}_{label.split(',')[0]}_missed.png"
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", nargs="*", type=int, default=None)
    ap.add_argument("--auto-flutter", type=int, default=0,
                    help="plot N missed AFLT records, most regular first")
    ap.add_argument("--auto-af", type=int, default=0,
                    help="plot N missed AFIB records, highest RR cv first")
    ap.add_argument("--results-csv", default=None,
                    help="af_safety_check output used to pick the misses")
    ap.add_argument("--out-dir", default="af_failure_plots")
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    targets: list[tuple[int, str]] = []

    if args.records:
        db = pd.read_csv(META, index_col="ecg_id")
        for rid in args.records:
            codes = ast.literal_eval(db.loc[rid, "scp_codes"])
            tag = "AFLT" if "AFLT" in codes else "AFIB" if "AFIB" in codes else "OTHER"
            targets.append((rid, tag))

    if (args.auto_flutter or args.auto_af) and args.results_csv:
        d = pd.read_csv(args.results_csv)
        d = d[d.status == "ok"]
        if args.auto_flutter:
            miss = d[(d.label == "AFLT") & (d.probable_flutter != True)]  # noqa: E712
            for rid in miss.sort_values("rr_cv").ecg_id.head(args.auto_flutter):
                targets.append((int(rid), "AFLT"))
        if args.auto_af:
            miss = d[(d.label == "AFIB") & (d.probable_af != True)]  # noqa: E712
            for rid in miss.sort_values("rr_cv", ascending=False).ecg_id.head(args.auto_af):
                targets.append((int(rid), "AFIB"))

    if not targets:
        print("nothing to plot: pass --records, or --auto-* together with --results-csv")
        return 1

    for rid, tag in targets:
        try:
            print(f"{rid} ({tag}): {plot_record(rid, tag, out_dir, args.seconds)}")
        except Exception as exc:  # noqa: BLE001
            print(f"{rid} ({tag}): FAILED {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

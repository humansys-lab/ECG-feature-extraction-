#!/usr/bin/env python3
"""Plot LUDB beats where ecgfeat's P-wave detection fails.

Renders, per record and lead, the expert P annotation against what ecgfeat
detected, plus the preceding T offset -- the quantity that drives the
dominant failure mode (a short TP interval hiding P in the T tail).

Matplotlib has no CJK font in this environment, so all in-figure text is
English by design; see docs/ for the Chinese write-up.

Usage:
    python plot_p_wave_failures.py --records 92 117 60 --out-dir p_wave_failure_plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "feature_extraction"))

import compare_ludb_detectors as cld

DATA = Path(
    "/workspace/ecg_gemma/data/"
    "lobachevsky-university-electrocardiography-database-1.0.1/data"
)
WAVE_TOL_MS = 150.0


def load_record(record_id: str):
    import wfdb

    path = DATA / record_id
    rec = wfdb.rdrecord(str(path))
    fs = int(round(float(rec.fs)))
    signal = np.asarray(rec.p_signal, dtype=float)
    idx = cld._lead_indices(rec.sig_name)
    ecg = signal[:, [idx[l] for l in cld.STANDARD_12_LEADS]].T
    return rec, fs, signal, idx, ecg


def analyse(record_id: str):
    """Return per-lead ground truth, detections and the TP/FN split."""
    rec, fs, signal, idx, ecg = load_record(record_id)
    detected, _ = cld.detect_ecgfeat(
        ecg, fs, cld.STANDARD_12_LEADS, mains_freq=50.0,
        qrs_peak_source="global", wave_peak_source="native",
    )
    tol = int(round(WAVE_TOL_MS / 1000 * fs))
    out = {}
    for lead in cld.STANDARD_12_LEADS:
        gt = cld.parse_ludb_waves(DATA / record_id, cld.LEAD_TO_ANN_EXT[lead])
        gt_p = [e for e in gt["P"] if e.peak is not None]
        det_p = [e for e in detected[lead]["P"] if e.peak is not None]
        pairs, _fn, _fp = cld.match_events(gt_p, det_p, tol)
        matched_gt = {int(g.peak) for g, _ in pairs}
        matched_det = {int(d.peak) for _, d in pairs}
        out[lead] = {
            "signal": signal[:, idx[lead]],
            "gt_p": gt_p,
            "det_p": det_p,
            "matched_gt": matched_gt,
            "matched_det": matched_det,
            "gt_t": [e for e in gt["T"] if e.offset is not None],
            "gt_qrs": [e for e in gt["QRS"] if e.peak is not None],
        }
    return fs, out


def pick_worst_leads(per_lead, n=3):
    scored = []
    for lead, d in per_lead.items():
        gt = d["gt_p"]
        if not gt:
            continue
        fn = sum(1 for e in gt if int(e.peak) not in d["matched_gt"])
        scored.append((fn / len(gt), fn, lead))
    scored.sort(reverse=True)
    return [lead for _rate, _fn, lead in scored[:n]]


def plot_record(record_id: str, out_dir: Path, n_leads: int = 3) -> Path:
    fs, per_lead = analyse(record_id)
    leads = pick_worst_leads(per_lead, n_leads)
    if not leads:
        raise SystemExit(f"record {record_id}: no annotated P waves")

    # Median RR for the header, from the expert QRS marks.
    qrs = sorted(int(e.peak) for e in per_lead[leads[0]]["gt_qrs"])
    rr_ms = float(np.median(np.diff(qrs))) * 1000.0 / fs if len(qrs) > 1 else float("nan")
    hr = 60000.0 / rr_ms if rr_ms and np.isfinite(rr_ms) else float("nan")

    total_gt = sum(len(per_lead[l]["gt_p"]) for l in cld.STANDARD_12_LEADS)
    total_fn = sum(
        1
        for l in cld.STANDARD_12_LEADS
        for e in per_lead[l]["gt_p"]
        if int(e.peak) not in per_lead[l]["matched_gt"]
    )

    fig, axes = plt.subplots(
        len(leads), 1, figsize=(16, 3.1 * len(leads)), sharex=True
    )
    if len(leads) == 1:
        axes = [axes]

    # Show the annotated span only -- LUDB annotates a middle window.
    all_p = [int(e.peak) for l in leads for e in per_lead[l]["gt_p"]]
    lo = max(0, min(all_p) - int(0.8 * fs))
    hi = min(len(per_lead[leads[0]]["signal"]), max(all_p) + int(0.8 * fs))
    t = np.arange(lo, hi) / fs

    for ax, lead in zip(axes, leads):
        d = per_lead[lead]
        sig = d["signal"][lo:hi]
        ax.plot(t, sig, color="#222222", lw=0.9, zorder=3)

        # Preceding T offsets: the tail that hides P at short TP intervals.
        for e in d["gt_t"]:
            if lo <= e.offset < hi:
                ax.axvline(
                    e.offset / fs, color="#1f77b4", lw=1.0, ls=":", alpha=0.75, zorder=2
                )

        for e in d["gt_qrs"]:
            if lo <= e.peak < hi:
                ax.axvline(e.peak / fs, color="#999999", lw=0.7, alpha=0.5, zorder=1)

        # Expert P: green span = onset..offset, marker at the peak.
        hit = miss = 0
        for e in d["gt_p"]:
            if not (lo <= e.peak < hi):
                continue
            found = int(e.peak) in d["matched_gt"]
            hit += found
            miss += not found
            if e.onset is not None and e.offset is not None:
                ax.axvspan(
                    e.onset / fs, e.offset / fs,
                    color="#2ca02c" if found else "#d62728",
                    alpha=0.16 if found else 0.28, zorder=1,
                )
            ax.plot(
                e.peak / fs, d["signal"][int(e.peak)],
                marker="v", ms=8,
                color="#2ca02c" if found else "#d62728",
                mec="black", mew=0.5, ls="none", zorder=5,
            )

        # ecgfeat detections: x = matched, open circle = false positive.
        for e in d["det_p"]:
            if not (lo <= e.peak < hi):
                continue
            ok = int(e.peak) in d["matched_det"]
            ax.plot(
                e.peak / fs, d["signal"][int(e.peak)],
                marker="x" if ok else "o", ms=7,
                color="#1a1aff" if ok else "#ff7f0e",
                mfc="none" if not ok else None, mew=1.6, ls="none", zorder=6,
            )

        ax.set_ylabel(f"{lead}\n(mV)", fontsize=10)
        ax.text(
            0.995, 0.94,
            f"expert P: {hit + miss}   detected {hit}   MISSED {miss}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.85, pad=2),
        )
        ax.grid(alpha=0.25, lw=0.5)

    axes[-1].set_xlabel("time (s)", fontsize=10)
    fig.suptitle(
        f"LUDB record {record_id} — P-wave detection failures     "
        f"median RR {rr_ms:.0f} ms ({hr:.0f} bpm)     "
        f"record-wide P miss rate {100.0 * total_fn / max(total_gt, 1):.0f}% "
        f"({total_fn}/{total_gt} over 12 leads)",
        fontsize=12, y=0.995,
    )
    handles = [
        plt.Line2D([], [], marker="v", ls="none", color="#2ca02c", mec="black",
                   label="expert P (detected)"),
        plt.Line2D([], [], marker="v", ls="none", color="#d62728", mec="black",
                   label="expert P (MISSED)"),
        plt.Line2D([], [], marker="x", ls="none", color="#1a1aff",
                   label="ecgfeat P (true positive)"),
        plt.Line2D([], [], marker="o", ls="none", color="#ff7f0e", mfc="none",
                   label="ecgfeat P (false positive)"),
        plt.Line2D([], [], color="#1f77b4", ls=":", label="expert T offset"),
        plt.Line2D([], [], color="#999999", lw=0.7, label="expert QRS peak"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=6, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ludb_{record_id}_p_failures.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_zoom(record_id: str, out_dir: Path, n_beats: int = 4) -> Path:
    """Few-beat close-up: the failure mechanism is invisible at 10 s scale."""
    fs, per_lead = analyse(record_id)
    lead = pick_worst_leads(per_lead, 1)[0]
    d = per_lead[lead]
    missed = [e for e in d["gt_p"] if int(e.peak) not in d["matched_gt"]]
    if not missed:
        raise SystemExit(f"record {record_id}: nothing missed in {lead}")

    centre = int(missed[len(missed) // 2].peak)
    qrs = sorted(int(e.peak) for e in d["gt_qrs"])
    rr = float(np.median(np.diff(qrs))) if len(qrs) > 1 else 0.8 * fs
    lo = max(0, int(centre - 1.1 * rr))
    hi = min(len(d["signal"]), int(centre + (n_beats - 0.6) * rr))
    t = np.arange(lo, hi) / fs

    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.plot(t, d["signal"][lo:hi], color="#111111", lw=1.5, zorder=3)

    for e in d["gt_t"]:
        if lo <= e.offset < hi:
            ax.axvline(e.offset / fs, color="#1f77b4", lw=1.4, ls=":", zorder=2)
            ax.annotate(
                "T off", (e.offset / fs, ax.get_ylim()[1]),
                xytext=(2, -10), textcoords="offset points",
                fontsize=8, color="#1f77b4",
            )
    for e in d["gt_qrs"]:
        if lo <= e.peak < hi:
            ax.axvline(e.peak / fs, color="#999999", lw=0.9, alpha=0.6, zorder=1)

    for e in d["gt_p"]:
        if not (lo <= e.peak < hi):
            continue
        found = int(e.peak) in d["matched_gt"]
        col = "#2ca02c" if found else "#d62728"
        if e.onset is not None and e.offset is not None:
            ax.axvspan(e.onset / fs, e.offset / fs, color=col,
                       alpha=0.18 if found else 0.30, zorder=1)
            # TP interval: previous T offset -> this P onset.
            prior = [x.offset for x in d["gt_t"] if x.offset is not None and x.offset < e.onset]
            if prior:
                gap = (e.onset - max(prior)) * 1000.0 / fs
                ax.annotate(
                    f"TP {gap:.0f} ms", (e.onset / fs, ax.get_ylim()[0]),
                    xytext=(-14, 12), textcoords="offset points",
                    fontsize=8, color="#555555",
                )
        ax.plot(e.peak / fs, d["signal"][int(e.peak)], marker="v", ms=11,
                color=col, mec="black", mew=0.6, ls="none", zorder=5)

    for e in d["det_p"]:
        if not (lo <= e.peak < hi):
            continue
        ok = int(e.peak) in d["matched_det"]
        ax.plot(e.peak / fs, d["signal"][int(e.peak)],
                marker="x" if ok else "o", ms=10,
                color="#1a1aff" if ok else "#ff7f0e",
                mfc="none" if not ok else None, mew=2.0, ls="none", zorder=6)

    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{lead} (mV)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(
        f"LUDB record {record_id}, lead {lead} — close-up of missed P waves "
        f"(median RR {float(np.median(np.diff(qrs))) * 1000 / fs:.0f} ms)",
        fontsize=12,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ludb_{record_id}_p_failures_zoom.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", nargs="+", default=["92", "117", "121", "60", "125", "56"])
    ap.add_argument("--out-dir", default="p_wave_failure_plots")
    ap.add_argument("--leads", type=int, default=3, help="worst N leads per record")
    ap.add_argument("--zoom", action="store_true", help="also render a few-beat close-up")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    for rid in args.records:
        try:
            path = plot_record(rid, out_dir, args.leads)
            print(f"record {rid}: {path}")
            if args.zoom:
                print(f"record {rid}: {plot_zoom(rid, out_dir)}")
        except Exception as exc:  # noqa: BLE001
            print(f"record {rid}: FAILED {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

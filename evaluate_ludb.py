#!/usr/bin/env python3
"""
T030: Evaluation of P/QRS/T boundary detection against LUDB expert annotations.

Database: Lobachevsky University ECG Database (200 records, 12 leads, 500 Hz)
Annotation format (WFDB):
    ( = wave onset    N = R peak    ) = wave offset
    t = T peak        p = P peak

Usage:
    python evaluate_ludb.py                  # all 200 records
    python evaluate_ludb.py --n 20           # first 20 records (quick test)
    python evaluate_ludb.py --out results/   # save CSVs + summary to folder
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "feature_extraction"))

import wfdb
from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.models import STANDARD_12_LEADS

# ── Constants ────────────────────────────────────────────────────────────────

_LUDB_ROOT_CANDIDATES = (
    PROJECT_ROOT / "data" / "lobachevsky-university-electrocardiography-database-1.0.1",
    PROJECT_ROOT / "lobachevsky-university-electrocardiography-database-1.0.1",
)
LUDB_ROOT = next(
    (path for path in _LUDB_ROOT_CANDIDATES if path.exists()),
    _LUDB_ROOT_CANDIDATES[0],
)
LUDB_DIR = LUDB_ROOT / "data"

# WFDB annotation extension → our standard lead name
ANN_EXT_TO_LEAD = {
    "i":   "I",   "ii":  "II",  "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1":  "V1",  "v2":  "V2",  "v3":  "V3",
    "v4":  "V4",  "v5":  "V5",  "v6":  "V6",
}
LEAD_TO_ANN_EXT = {v: k for k, v in ANN_EXT_TO_LEAD.items()}

# Max distance (ms) to match a detected R-peak to a GT R-peak
MATCH_TOLERANCE_MS = 75

# ── Annotation parsing ───────────────────────────────────────────────────────

def parse_ludb_annotations(record_path: str, ext: str) -> List[Dict]:
    """
    Parse WFDB annotation file for one lead.

    Returns list of dicts, one per beat:
        {r_sample, qrs_on, qrs_off, t_on, t_peak, t_off, p_on, p_peak, p_off}
    All values are sample indices (None if not annotated).
    """
    try:
        ann = wfdb.rdann(record_path, ext)
    except Exception:
        return []

    symbols = ann.symbol
    samples = ann.sample
    beats: List[Dict] = []

    i = 0
    n = len(symbols)
    while i < n:
        # Scan for QRS complex: ( N )
        if symbols[i] == "(" and i + 2 < n and symbols[i + 1] == "N":
            qrs_on   = int(samples[i])
            r_sample = int(samples[i + 1])
            qrs_off  = int(samples[i + 2]) if symbols[i + 2] == ")" else None
            beat = {
                "r_sample": r_sample,
                "qrs_on":   qrs_on,
                "qrs_off":  qrs_off,
                "t_on":     None, "t_peak": None, "t_off":  None,
                "p_on":     None, "p_peak": None, "p_off":  None,
            }
            j = i + (3 if qrs_off is not None else 2)

            # T wave: ( t )
            if j < n and symbols[j] == "(" and j + 2 < n and symbols[j + 1] == "t":
                beat["t_on"]   = int(samples[j])
                beat["t_peak"] = int(samples[j + 1])
                beat["t_off"]  = int(samples[j + 2]) if symbols[j + 2] == ")" else None
                j += 3 if beat["t_off"] is not None else 2

            # P wave: ( p )
            if j < n and symbols[j] == "(" and j + 2 < n and symbols[j + 1] == "p":
                beat["p_on"]   = int(samples[j])
                beat["p_peak"] = int(samples[j + 1])
                beat["p_off"]  = int(samples[j + 2]) if symbols[j + 2] == ")" else None
                j += 3 if beat["p_off"] is not None else 2

            beats.append(beat)
            i = j
        else:
            i += 1

    return beats


# ── Beat matching ────────────────────────────────────────────────────────────

def match_beats(
    gt_beats: List[Dict],
    det_beats,           # List[LeadBeatFeatures] filtered to one lead
    fs: int,
) -> List[Tuple[Dict, object]]:
    """
    Match GT beats to detected beats by R-peak proximity.
    Returns list of (gt_beat, det_beat) pairs within MATCH_TOLERANCE_MS.
    """
    tol = int(MATCH_TOLERANCE_MS * fs / 1000)
    gt_rs  = np.array([b["r_sample"] for b in gt_beats])
    det_rs = np.array([d.qrs.peak for d in det_beats if d.qrs.peak is not None])
    if len(det_rs) == 0 or len(gt_rs) == 0:
        return []

    pairs = []
    used_det = set()
    for gi, gt in enumerate(gt_beats):
        dists = np.abs(det_rs - gt["r_sample"])
        best = int(np.argmin(dists))
        if dists[best] <= tol and best not in used_det:
            # Find the actual LeadBeatFeatures object
            det_obj = [d for d in det_beats if d.qrs.peak is not None][best]
            pairs.append((gt, det_obj))
            used_det.add(best)
    return pairs


# ── Error computation ────────────────────────────────────────────────────────

def boundary_error_ms(gt_sample: Optional[int], det_sample: Optional[int], fs: int) -> Optional[float]:
    """Detected − GT in milliseconds (positive = detected later)."""
    if gt_sample is None or det_sample is None:
        return None
    return (det_sample - gt_sample) * 1000.0 / fs


def _det_p_by_gt_p_timing(
    gt_beats: List[Dict],
    det_beats: List[object],
    fs: int,
) -> Dict[int, object]:
    """Map LUDB post-T P annotations to the detected beat they precede."""
    det_with_qrs = [
        det for det in det_beats
        if getattr(getattr(det, "qrs", None), "peak", None) is not None
    ]
    det_with_qrs.sort(key=lambda det: int(det.qrs.peak))

    det_p: Dict[int, object] = {}
    for gt in gt_beats:
        if gt.get("p_on") is None or gt.get("p_peak") is None or gt.get("p_off") is None:
            continue
        gt_r = int(gt["r_sample"])
        p_on = int(gt["p_on"])
        p_peak = int(gt["p_peak"])
        p_off = int(gt["p_off"])

        candidates: List[Tuple[int, object]] = []
        for det in det_with_qrs:
            qrs = det.qrs
            qrs_ref = qrs.onset if qrs.onset is not None else qrs.peak
            if qrs_ref is None:
                continue
            qrs_ref = int(qrs_ref)
            if qrs_ref <= gt_r:
                continue

            p_on_to_qrs_ms = (qrs_ref - p_on) * 1000.0 / fs
            p_peak_to_qrs_ms = (qrs_ref - p_peak) * 1000.0 / fs
            p_off_to_qrs_ms = (qrs_ref - p_off) * 1000.0 / fs
            if not (60.0 <= p_on_to_qrs_ms <= 420.0):
                continue
            if not (40.0 <= p_peak_to_qrs_ms <= 360.0):
                continue
            if not (-30.0 <= p_off_to_qrs_ms <= 280.0):
                continue
            candidates.append((qrs_ref, det))

        if candidates:
            det_p[gt_r] = min(candidates, key=lambda item: item[0])[1]
    return det_p


# ── Per-record evaluation ────────────────────────────────────────────────────

def evaluate_record(
    record_id: str,
    extractor: ECGFeatureExtractor,
) -> List[Dict]:
    """
    Run extraction on one LUDB record and return a list of error dicts
    (one per matched beat × lead).
    """
    record_path = str(LUDB_DIR / record_id)
    try:
        rec = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"  [skip] {record_id}: cannot read signal — {e}")
        return []

    ecg = rec.p_signal.T   # [12, N]
    fs  = rec.fs

    try:
        feat = extractor.extract(ecg, float(fs))
    except Exception as e:
        print(f"  [skip] {record_id}: extractor failed — {e}")
        return []

    # Build per-lead dict of detected beat features
    det_by_lead: Dict[str, list] = defaultdict(list)
    for bf in feat.beat_features:
        det_by_lead[bf.lead].append(bf)

    rows = []
    for lead in STANDARD_12_LEADS:
        ext = LEAD_TO_ANN_EXT.get(lead)
        if ext is None:
            continue
        gt_beats = parse_ludb_annotations(record_path, ext)
        if not gt_beats:
            continue
        det_beats = det_by_lead.get(lead, [])
        pairs = match_beats(gt_beats, det_beats, feat.fs)

        # P-wave convention: LUDB annotates the P-wave of beat N AFTER beat N's
        # T-wave (i.e., it is the P-wave preceding the following QRS).  Some
        # leads omit that following QRS even when the P annotation is present,
        # so map by the GT P-to-detected-QRS timing rather than by GT beat index.
        det_p_by_gt_r = _det_p_by_gt_p_timing(gt_beats, det_beats, feat.fs)

        for gt, det in pairs:
            det_p = det_p_by_gt_r.get(gt["r_sample"])  # det beat whose P to compare
            row = {
                "record":   record_id,
                "lead":     lead,
                "r_sample": gt["r_sample"],
                # QRS
                "qrs_on_err":  boundary_error_ms(gt["qrs_on"],  det.qrs.onset,  feat.fs),
                "qrs_off_err": boundary_error_ms(gt["qrs_off"], det.qrs.offset, feat.fs),
                # T wave
                "t_on_err":    boundary_error_ms(gt["t_on"],    det.t.onset,    feat.fs),
                "t_off_err":   boundary_error_ms(gt["t_off"],   det.t.offset,   feat.fs),
                # P wave (corrected convention: GT P of beat i → det P of beat i+1)
                "p_on_err":    boundary_error_ms(
                    gt["p_on"],
                    det_p.p.onset if det_p is not None else None,
                    feat.fs,
                ),
                "p_off_err":   boundary_error_ms(
                    gt["p_off"],
                    det_p.p.offset if det_p is not None else None,
                    feat.fs,
                ),
                # QT interval errors
                "qt_gt_ms":    (
                    (gt["t_off"] - gt["qrs_on"]) * 1000.0 / feat.fs
                    if gt["t_off"] is not None and gt["qrs_on"] is not None else None
                ),
                "qt_det_ms":   det.qt_ms,
                "qt_conf":     det.qt_confidence,
                "beat_reliable": det.beat_measurement_reliable,
            }
            row["qt_err"] = (
                (row["qt_det_ms"] - row["qt_gt_ms"])
                if row["qt_det_ms"] is not None and row["qt_gt_ms"] is not None
                else None
            )
            rows.append(row)

    return rows


# ── Statistics ───────────────────────────────────────────────────────────────

def _stats(vals: List[float]) -> Dict:
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "mae": None,
                "p50": None, "p95": None}
    a = np.array(vals)
    return {
        "n":    len(a),
        "mean": float(np.mean(a)),
        "sd":   float(np.std(a)),
        "mae":  float(np.mean(np.abs(a))),
        "p50":  float(np.median(np.abs(a))),
        "p95":  float(np.percentile(np.abs(a), 95)),
    }


def compute_summary(rows: List[Dict]) -> Dict:
    """Aggregate errors across all records and leads."""
    fields = [
        "qrs_on_err", "qrs_off_err",
        "t_on_err",   "t_off_err",
        "p_on_err",   "p_off_err",
        "qt_err",
    ]
    summary = {}
    for f in fields:
        vals = [r[f] for r in rows if r[f] is not None]
        summary[f] = _stats(vals)

    # Also stratify QT error by lead
    qt_by_lead: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["qt_err"] is not None:
            qt_by_lead[r["lead"]].append(r["qt_err"])
    summary["qt_err_by_lead"] = {
        lead: _stats(vals) for lead, vals in qt_by_lead.items()
    }

    # Breakdown: reliable vs unreliable beats
    rel_qt  = [r["qt_err"] for r in rows if r["qt_err"] is not None and r["beat_reliable"]]
    unrel_qt = [r["qt_err"] for r in rows if r["qt_err"] is not None and not r["beat_reliable"]]
    summary["qt_err_reliable"]   = _stats(rel_qt)
    summary["qt_err_unreliable"] = _stats(unrel_qt)

    return summary


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_summary(summary: Dict, n_records: int, n_beats: int) -> None:
    W = 70
    SEP = "─" * W

    def row(label, s):
        if s["n"] == 0:
            print(f"  {label:<22s}   no data")
            return
        print(
            f"  {label:<22s}  n={s['n']:>5d}  "
            f"mean={s['mean']:+6.1f}ms  sd={s['sd']:5.1f}  "
            f"MAE={s['mae']:5.1f}  p50={s['p50']:5.1f}  p95={s['p95']:5.1f}"
        )

    print()
    print("=" * W)
    print(f"  LUDB EVALUATION  —  {n_records} records  /  {n_beats} matched beat×lead pairs")
    print("=" * W)
    print()
    print(f"  {'Boundary':<22s}  {'N':>7s}  {'mean':>10s}  {'SD':>6s}  "
          f"{'MAE':>6s}  {'p50':>6s}  {'p95':>6s}")
    print(f"  {SEP}")
    row("QRS onset",   summary["qrs_on_err"])
    row("QRS offset",  summary["qrs_off_err"])
    print(f"  {SEP}")
    row("T onset",     summary["t_on_err"])
    row("T offset",    summary["t_off_err"])
    print(f"  {SEP}")
    row("P onset",     summary["p_on_err"])
    row("P offset",    summary["p_off_err"])
    print(f"  {SEP}")
    row("QT interval", summary["qt_err"])
    print()
    print(f"  QT error by reliability:")
    row("  reliable beats",   summary["qt_err_reliable"])
    row("  unreliable beats", summary["qt_err_unreliable"])
    print()

    print(f"  QT error by lead (MAE):")
    lead_rows = sorted(
        summary["qt_err_by_lead"].items(),
        key=lambda x: (x[1]["mae"] or 999),
    )
    for lead, s in lead_rows:
        if s["n"] > 0:
            print(f"    {lead:<5s}  n={s['n']:>4d}  "
                  f"mean={s['mean']:+6.1f}  MAE={s['mae']:5.1f}  p95={s['p95']:5.1f}")
    print()
    print("  All errors in ms.  Positive = detected later than GT.")
    print("=" * W)


# ── Visualisation ────────────────────────────────────────────────────────────

def plot_results(rows: List[Dict], summary: Dict, out_dir: Path) -> None:
    """Generate and save evaluation visualisation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    out_dir.mkdir(parents=True, exist_ok=True)
    CLIP = 300   # clip errors to ±300 ms for histogram display

    # ── 1. Error distribution panel (6 boundaries + QT) ──────────────────────
    FIELDS = [
        ("qrs_on_err",  "QRS Onset",   "#2166AC"),
        ("qrs_off_err", "QRS Offset",  "#4DAC26"),
        ("t_on_err",    "T Onset",     "#F4A582"),
        ("t_off_err",   "T Offset",    "#D6604D"),
        ("p_on_err",    "P Onset",     "#92C5DE"),
        ("p_off_err",   "P Offset",    "#0571B0"),
        ("qt_err",      "QT Interval", "#762A83"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes_flat = axes.flatten()
    for ax in axes_flat[len(FIELDS):]:
        ax.set_visible(False)

    for idx, (field, label, color) in enumerate(FIELDS):
        vals = np.array([r[field] for r in rows if r[field] is not None])
        clipped = np.clip(vals, -CLIP, CLIP)
        ax = axes_flat[idx]
        ax.hist(clipped, bins=80, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        s = summary.get(field, {})
        if s.get("n", 0) > 0:
            ax.axvline(s["mean"], color="red", lw=1.2, label=f"mean={s['mean']:+.1f}")
            ax.axvline(s["mean"] - s["sd"], color="red", lw=0.7, ls=":")
            ax.axvline(s["mean"] + s["sd"], color="red", lw=0.7, ls=":")
            title = (f"{label}\n"
                     f"n={s['n']}  mean={s['mean']:+.1f}  MAE={s['mae']:.1f}  "
                     f"p50={s['p50']:.1f}  p95={s['p95']:.1f} ms")
        else:
            title = f"{label}\n(no data)"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Error (ms)", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-CLIP, CLIP)
        ax.legend(fontsize=7)

    fig.suptitle("Boundary Detection Error Distributions  (clipped to ±300 ms)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = out_dir / "1_error_distributions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")

    # ── 2. Per-lead QT error box plot ─────────────────────────────────────────
    qt_by_lead = summary.get("qt_err_by_lead", {})
    lead_order = [l for l in STANDARD_12_LEADS if l in qt_by_lead]
    data_by_lead = [
        np.clip([r["qt_err"] for r in rows
                 if r["lead"] == lead and r["qt_err"] is not None], -CLIP, CLIP)
        for lead in lead_order
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(data_by_lead, tick_labels=lead_order, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=0.8),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    colors = plt.cm.tab20(np.linspace(0, 1, len(lead_order)))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    for i, lead in enumerate(lead_order, 1):
        s = qt_by_lead.get(lead, {})
        if s.get("n", 0):
            ax.text(i, CLIP * 0.92, f"MAE\n{s['mae']:.0f}",
                    ha="center", va="top", fontsize=6.5, color="#333333")
    ax.set_ylabel("QT Error (ms)  [det − GT]", fontsize=9)
    ax.set_title("QT Interval Error by Lead  (clipped to ±300 ms)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_ylim(-CLIP, CLIP)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / "2_qt_error_by_lead.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")

    # ── 3. QT detected vs GT scatter ─────────────────────────────────────────
    qt_gt  = np.array([r["qt_gt_ms"]  for r in rows if r["qt_gt_ms"]  is not None
                       and r["qt_det_ms"] is not None])
    qt_det = np.array([r["qt_det_ms"] for r in rows if r["qt_gt_ms"]  is not None
                       and r["qt_det_ms"] is not None])
    conf   = np.array([r["qt_conf"]   for r in rows if r["qt_gt_ms"]  is not None
                       and r["qt_det_ms"] is not None])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter coloured by confidence
    sc = axes[0].scatter(qt_gt, qt_det, c=conf, cmap="RdYlGn",
                         s=4, alpha=0.4, vmin=0, vmax=1)
    lo, hi = 200, 700
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect")
    axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("GT QT (ms)", fontsize=9)
    axes[0].set_ylabel("Detected QT (ms)", fontsize=9)
    axes[0].set_title("Detected vs GT QT  (coloured by T-end confidence)", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].tick_params(labelsize=8)
    plt.colorbar(sc, ax=axes[0], label="T-end confidence")

    # Bland-Altman
    mean_qt = (qt_gt + qt_det) / 2
    diff_qt = qt_det - qt_gt
    md  = float(np.mean(diff_qt))
    loa = 1.96 * float(np.std(diff_qt))
    axes[1].scatter(mean_qt, diff_qt, s=4, alpha=0.3, color="#2166AC")
    axes[1].axhline(md,       color="red",   lw=1.5, label=f"Mean bias {md:+.1f} ms")
    axes[1].axhline(md + loa, color="orange", lw=1,  ls="--",
                    label=f"+1.96 SD  {md+loa:+.1f} ms")
    axes[1].axhline(md - loa, color="orange", lw=1,  ls="--",
                    label=f"−1.96 SD  {md-loa:+.1f} ms")
    axes[1].axhline(0, color="black", lw=0.7, ls=":")
    axes[1].set_xlabel("Mean of GT and Detected QT (ms)", fontsize=9)
    axes[1].set_ylabel("Detected − GT  (ms)", fontsize=9)
    axes[1].set_title("Bland-Altman: QT Agreement", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(labelsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "3_qt_scatter_bland_altman.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")

    # ── 4. Cumulative accuracy curves ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    BOUNDARY_GROUPS = [
        ("QRS boundaries",
         [("qrs_on_err", "QRS onset", "#2166AC"),
          ("qrs_off_err", "QRS offset", "#4DAC26")]),
        ("T / P boundaries",
         [("t_off_err",   "T offset",  "#D6604D"),
          ("p_on_err",    "P onset",   "#92C5DE"),
          ("qt_err",      "QT",        "#762A83")]),
    ]
    thresholds = np.arange(0, 201, 2)
    for ax, (title, pairs) in zip(axes, BOUNDARY_GROUPS):
        for field, label, color in pairs:
            vals = np.abs([r[field] for r in rows if r[field] is not None])
            pct  = [100 * np.mean(np.array(vals) <= t) for t in thresholds]
            ax.plot(thresholds, pct, color=color, lw=1.8, label=label)
        ax.axvline(10, color="gray", lw=0.6, ls=":")
        ax.axvline(20, color="gray", lw=0.6, ls=":")
        ax.axvline(30, color="gray", lw=0.6, ls=":")
        ax.axhline(90, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("Tolerance (ms)", fontsize=9)
        ax.set_ylabel("% of beats within tolerance", fontsize=9)
        ax.set_title(f"Cumulative Accuracy — {title}", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, 200); ax.set_ylim(0, 101)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = out_dir / "4_cumulative_accuracy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")

    # ── 5. MAE summary bar chart ──────────────────────────────────────────────
    labels = ["QRS\nonset", "QRS\noffset", "T\nonset", "T\noffset",
              "P\nonset", "P\noffset", "QT"]
    fields_bar = ["qrs_on_err", "qrs_off_err", "t_on_err", "t_off_err",
                  "p_on_err",   "p_off_err",   "qt_err"]
    maes  = [summary[f]["mae"]  if summary[f]["n"] else 0 for f in fields_bar]
    means = [summary[f]["mean"] if summary[f]["n"] else 0 for f in fields_bar]
    p50s  = [summary[f]["p50"]  if summary[f]["n"] else 0 for f in fields_bar]
    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, maes,  w, label="MAE",  color="#4393C3", alpha=0.85)
    ax.bar(x,     p50s,  w, label="p50 |err|", color="#2CA02C", alpha=0.85)
    ax.bar(x + w, [abs(m) for m in means], w,
           label="|mean|", color="#FF7F0E", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Error (ms)", fontsize=9)
    ax.set_title("Boundary Detection Accuracy Summary  (200 LUDB records)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=8)
    for xi, mae in zip(x - w, maes):
        ax.text(xi, mae + 1, f"{mae:.0f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    p = out_dir / "5_mae_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── CSV export ───────────────────────────────────────────────────────────────

def save_csv(rows: List[Dict], out_dir: Path) -> None:
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ludb_errors.csv"
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Per-beat errors saved to: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate boundary detection on LUDB")
    parser.add_argument("--n",   type=int, default=None,
                        help="Limit to first N records (default: all 200)")
    parser.add_argument("--out", type=str, default=None,
                        help="Directory to save CSV results (optional)")
    args = parser.parse_args()

    records_file = LUDB_ROOT / "RECORDS"
    all_records = [
        Path(line.strip()).name
        for line in records_file.read_text().splitlines()
        if line.strip()
    ]
    if args.n:
        all_records = all_records[: args.n]

    extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)

    print(f"Evaluating {len(all_records)} records against LUDB expert annotations...")
    print()

    all_rows: List[Dict] = []
    for i, rid in enumerate(all_records, 1):
        print(f"  [{i:>3d}/{len(all_records)}]  {rid}", end="  ", flush=True)
        rows = evaluate_record(rid, extractor)
        n_pairs = len(rows)
        print(f"→ {n_pairs} beat×lead pairs matched")
        all_rows.extend(rows)

    if not all_rows:
        print("No results — check that LUDB data directory is accessible.")
        return

    summary = compute_summary(all_rows)
    print_summary(summary, len(all_records), len(all_rows))

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "results"
    save_csv(all_rows, out_dir)
    print()
    print(f"  Generating visualisations...")
    plot_results(all_rows, summary, out_dir)
    print(f"  All outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

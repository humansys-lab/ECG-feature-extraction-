#!/usr/bin/env python3
"""
ECG feature extraction result visualization.

Usage:
    python visualize_ecg.py [RECORD_ID] [--window SECONDS]

Examples:
    python visualize_ecg.py               # random record, 5s window
    python visualize_ecg.py JS00010       # specific record
    python visualize_ecg.py JS00010 --window 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np

PROJECT_ROOT = Path(__file__).parent
FEATURE_EXTRACTION_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTION_ROOT))

from ecgfeat.io import load_wfdb_mat as load_ecg
from ecgfeat.io import parse_wfdb_header as parse_hea

DATASET_DIR = PROJECT_ROOT / "dataset"

# Standard 12-lead clinical layout (row × col)
CLINICAL_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II"
ALL_LEADS   = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# Beat group colors (up to 5 groups)
GROUP_COLORS = {
    1: "#1565C0",  # deep blue  (dominant)
    2: "#E65100",  # deep orange
    3: "#6A1B9A",  # deep purple
    4: "#B71C1C",  # deep red
    5: "#00695C",  # deep teal
}
DEFAULT_GROUP_COLOR = "#37474F"

# Wave annotation colors
P_COLOR   = "#0288D1"  # blue
QRS_COLOR = "#2E7D32"  # green
T_COLOR   = "#E65100"  # orange


def load_features(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# Annotation helpers
# ──────────────────────────────────────────────────────────────

def _build_beat_annot(features: dict) -> dict[str, list]:
    id2group = {b["beat_id"]: b["group_id"] for b in features["beats"]}
    id2ridx  = {b["beat_id"]: b["r_index"]  for b in features["beats"]}
    by_lead: dict[str, list] = defaultdict(list)
    for bf in features["beat_features"]:
        lead    = bf["lead"]
        beat_id = bf["beat_id"]
        by_lead[lead].append({
            "beat_id":  beat_id,
            "group_id": id2group.get(beat_id, 0),
            "r_idx":    id2ridx.get(beat_id),
            "p":   bf["p"],
            "qrs": bf["qrs"],
            "t":   bf["t"],
            "flags": bf.get("flags", []),
        })
    return dict(by_lead)


def _valid(idx) -> bool:
    """Return True if wave index is a real detection (not sentinel zero)."""
    return idx is not None and idx > 0


# ──────────────────────────────────────────────────────────────
# Single-lead plot
# ──────────────────────────────────────────────────────────────

def plot_lead(ax, signal: np.ndarray, lead_name: str,
              annots: list, fs: int,
              t_start_ms: float = 0.0,
              t_end_ms: float | None = None,
              show_xaxis: bool = False,
              is_rhythm: bool = False):
    """
    Plot one ECG lead with P/QRS/T annotations.

    t_start_ms / t_end_ms: visible time window in milliseconds.
    Annotations outside the window are still processed but invisible.
    """
    n = len(signal)
    t_ms = np.arange(n) / fs * 1000.0

    if t_end_ms is None:
        t_end_ms = n / fs * 1000.0

    # Clip signal to display window
    i0 = max(0, int(t_start_ms / 1000 * fs))
    i1 = min(n, int(t_end_ms   / 1000 * fs) + 1)
    t_win  = t_ms[i0:i1]
    sig_win = signal[i0:i1]

    # ── ECG paper background ──────────────────────────────────
    ax.set_facecolor("#FFFAF8")
    ax.set_xlim(t_start_ms, t_end_ms)

    # Auto y-range: signal extent + 25% padding + at least ±0.3 mV
    sig_range = sig_win.max() - sig_win.min() if len(sig_win) > 0 else 1.0
    pad = max(sig_range * 0.25, 0.3)
    ylo = sig_win.min() - pad if len(sig_win) > 0 else -1.0
    yhi = sig_win.max() + pad if len(sig_win) > 0 else  1.0
    ax.set_ylim(ylo, yhi)

    # Grid: major 200 ms / 0.5 mV, minor 40 ms / 0.1 mV
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(40))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which="major", color="#F5C6C6", linewidth=0.6, zorder=1)
    ax.grid(which="minor", color="#FAE0E0", linewidth=0.25, zorder=1)

    # ── Wave annotations ─────────────────────────────────────
    for beat in annots:
        gid  = beat["group_id"]
        gcol = GROUP_COLORS.get(gid, DEFAULT_GROUP_COLOR)
        p, qrs, t_w = beat["p"], beat["qrs"], beat["t"]

        def _t(idx):
            return idx / fs * 1000.0

        # --- P wave ---
        if _valid(p["onset"]) and _valid(p["offset"]):
            ax.axvspan(_t(p["onset"]), _t(p["offset"]),
                       color=P_COLOR, alpha=0.10, linewidth=0, zorder=2)
            for x in (_t(p["onset"]), _t(p["offset"])):
                ax.axvline(x, color=P_COLOR, linewidth=0.8,
                           alpha=0.7, zorder=2, linestyle="--")
        if _valid(p["peak"]):
            pk = p["peak"]
            if 0 <= pk < n:
                ax.plot(_t(pk), signal[pk], "o",
                        color=P_COLOR, markersize=3.5,
                        zorder=6, markeredgewidth=0)

        # --- QRS ---
        if _valid(qrs["onset"]) and _valid(qrs["offset"]):
            ax.axvspan(_t(qrs["onset"]), _t(qrs["offset"]),
                       color=QRS_COLOR, alpha=0.15, linewidth=0, zorder=2)
            for x in (_t(qrs["onset"]), _t(qrs["offset"])):
                ax.axvline(x, color=QRS_COLOR, linewidth=1.0,
                           alpha=0.8, zorder=2)

        # --- T wave ---
        if _valid(t_w["onset"]) and _valid(t_w["offset"]):
            ax.axvspan(_t(t_w["onset"]), _t(t_w["offset"]),
                       color=T_COLOR, alpha=0.10, linewidth=0, zorder=2)
            for x in (_t(t_w["onset"]), _t(t_w["offset"])):
                ax.axvline(x, color=T_COLOR, linewidth=0.8,
                           alpha=0.7, zorder=2, linestyle="--")
        if _valid(t_w["peak"]):
            pk = t_w["peak"]
            if 0 <= pk < n:
                ax.plot(_t(pk), signal[pk], "o",
                        color=T_COLOR, markersize=3.5,
                        zorder=6, markeredgewidth=0)

        # --- R peak marker ---
        # Use per-lead qrs.peak (local max deflection) so the marker sits on
        # each lead's actual waveform peak, not the shared global fiducial.
        r_idx = qrs.get("peak") or beat["r_idx"]
        if r_idx is not None and 0 <= r_idx < n:
            ax.plot(_t(r_idx), signal[r_idx],
                    "v", color=gcol, markersize=6, zorder=7,
                    markeredgewidth=0.8, markeredgecolor="white")

    # ── ECG signal (drawn on top of annotations) ─────────────
    ax.plot(t_win, sig_win, color="#111111", linewidth=0.9, zorder=5)

    # ── 1 mV calibration bar (left edge, rhythm strip only) ──
    if is_rhythm:
        cal_x = t_start_ms + (t_end_ms - t_start_ms) * 0.005
        cal_mid = (ylo + yhi) / 2
        ax.plot([cal_x, cal_x], [cal_mid - 0.5, cal_mid + 0.5],
                color="#555555", linewidth=1.5, zorder=8, solid_capstyle="butt")
        ax.text(cal_x + 8, cal_mid + 0.52, "1mV",
                fontsize=6, va="bottom", color="#555555")

    # ── Axes formatting ──────────────────────────────────────
    # Lead name: inside plot, left-aligned
    ax.text(t_start_ms + (t_end_ms - t_start_ms) * 0.01, yhi - (yhi - ylo) * 0.04,
            lead_name, fontsize=8.5, fontweight="bold",
            va="top", ha="left", color="#222222", zorder=8)

    ax.tick_params(left=False, labelleft=False)
    if show_xaxis:
        ax.tick_params(bottom=True, labelbottom=True, labelsize=6.5)
        ax.set_xlabel("Time (ms)", fontsize=7.5)
    else:
        ax.tick_params(bottom=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_visible(False)


# ──────────────────────────────────────────────────────────────
# Main visualization function
# ──────────────────────────────────────────────────────────────

def visualize(record_id: str, window_sec: float | None = None) -> Path:
    hea_path  = DATASET_DIR / f"{record_id}.hea"
    mat_path  = DATASET_DIR / f"{record_id}.mat"
    json_path = PROJECT_ROOT / f"{record_id}_features.json"

    if not mat_path.exists():
        sys.exit(f"[Error] Not found: {mat_path}")
    if not json_path.exists():
        sys.exit(f"[Error] Features file not found: {json_path}\n"
                 f"        Run first: python demo_feature_extraction.py {record_id}")

    hdr      = parse_hea(hea_path)
    ecg      = load_ecg(mat_path)          # [12, N]
    features = load_features(json_path)
    fs       = hdr["fs"]
    lead2idx = {l: i for i, l in enumerate(ALL_LEADS)}

    annot_by_lead = _build_beat_annot(features)
    gf = features.get("global_features", {})

    total_ms   = hdr["n_samples"] / fs * 1000.0
    window_ms  = total_ms if window_sec is None else min(window_sec * 1000.0, total_ms)

    # ── Figure layout ─────────────────────────────────────────
    # 3 rows × 4 cols for 12-lead grid + 1 full-width rhythm strip
    n_rows = len(CLINICAL_LAYOUT)  # 3
    n_cols = len(CLINICAL_LAYOUT[0])  # 4

    # Scale width to window duration:
    #   5 s → ~22 in wide (same density as before)
    #   10 s → ~42 in wide
    w_per_col = max(1.05 * window_ms / 1000.0, 5.5)  # ~1 in per second per column
    fig_w = w_per_col * n_cols + 2.0                  # +2 for margins/labels
    fig_h = max(14.0, n_rows * 3.8 + 3.2)            # fixed height per row

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs  = GridSpec(
        n_rows + 1, n_cols,
        figure=fig,
        height_ratios=[1, 1, 1, 0.9],
        hspace=0.18,
        wspace=0.06,
        left=0.03, right=0.98,
        top=0.88, bottom=0.07,
    )

    # ── Draw 12 leads ────────────────────────────────────────
    for r, row in enumerate(CLINICAL_LAYOUT):
        for c, lead_name in enumerate(row):
            ax  = fig.add_subplot(gs[r, c])
            sig = ecg[lead2idx[lead_name]]
            annots = annot_by_lead.get(lead_name, [])
            plot_lead(ax, sig, lead_name, annots, fs,
                      t_start_ms=0.0, t_end_ms=window_ms,
                      show_xaxis=(r == n_rows - 1))

    # ── Rhythm strip (full-width, full duration) ──────────────
    ax_r = fig.add_subplot(gs[n_rows, :])
    plot_lead(ax_r, ecg[lead2idx[RHYTHM_LEAD]],
              f"{RHYTHM_LEAD}  —  Rhythm strip (full {total_ms/1000:.0f}s)",
              annot_by_lead.get(RHYTHM_LEAD, []), fs,
              t_start_ms=0.0, t_end_ms=total_ms,
              show_xaxis=True, is_rhythm=True)

    # ── Title ────────────────────────────────────────────────
    def _f(v, unit="", p=0):
        return "N/A" if v is None else f"{v:.{p}f}{unit}"

    age = hdr.get("age", "?")
    sex = hdr.get("sex", "?")
    dx  = ", ".join(hdr.get("dx", []))

    line1 = f"12-Lead ECG  |  {record_id}  |  Age {age}  {sex}  |  Dx: {dx}"
    line2 = (
        f"HR={_f(gf.get('heart_rate_bpm'), ' bpm', 1)}   "
        f"PR={_f(gf.get('pr_ms'), ' ms', 0)}   "
        f"QRS={_f(gf.get('qrs_ms'), ' ms', 0)}   "
        f"QT={_f(gf.get('qt_ms'), ' ms', 0)}   "
        f"QTc(Bazett)={_f(gf.get('qtc_bazett_ms'), ' ms', 0)}   "
        f"QTc(Frid)={_f(gf.get('qtc_fridericia_ms'), ' ms', 0)}   "
        f"QRS-Axis={_f(gf.get('qrs_axis_deg'), 'deg', 0)}   "
        f"T-Axis={_f(gf.get('t_axis_deg'), 'deg', 0)}   "
        f"(12-lead window: {window_ms/1000:.1f}s)"
    )
    fig.text(0.03, 0.96, line1, fontsize=10, fontweight="bold",
             va="top", ha="left")
    fig.text(0.03, 0.925, line2, fontsize=8.5, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F4FD",
                       edgecolor="#90CAF9", linewidth=0.8, alpha=0.95))

    # ── Legend ───────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=P_COLOR,   alpha=0.55, label="P wave",
                       linewidth=0),
        mpatches.Patch(color=QRS_COLOR, alpha=0.55, label="QRS",
                       linewidth=0),
        mpatches.Patch(color=T_COLOR,   alpha=0.55, label="T wave",
                       linewidth=0),
        plt.Line2D([0],[0], color=QRS_COLOR, linewidth=1.2,
                   label="QRS onset/offset"),
        plt.Line2D([0],[0], color=P_COLOR,   linewidth=1.0,
                   linestyle="--", label="P/T boundary"),
    ]
    groups = features.get("groups", {})
    for gid_str, g in groups.items():
        gid = int(gid_str)
        col = GROUP_COLORS.get(gid, DEFAULT_GROUP_COLOR)
        dom = " *" if g.get("flags", {}).get("dominant_group") else ""
        legend_items.append(
            plt.Line2D([0],[0], marker="v", color="w",
                       markerfacecolor=col, markeredgecolor="white",
                       markeredgewidth=0.5, markersize=8,
                       label=f"Group {gid}{dom}: {g['member_count']} beats "
                             f"({g['member_pct']:.0f}%)")
        )

    fig.legend(handles=legend_items,
               loc="upper right",
               fontsize=7.5,
               framealpha=0.95,
               edgecolor="#CCCCCC",
               ncol=len(legend_items),
               bbox_to_anchor=(0.98, 0.975))

    # ── Save ─────────────────────────────────────────────────
    out_path = PROJECT_ROOT / f"{record_id}_ecg_annotated.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize annotated ECG")
    parser.add_argument("record_id", nargs="?", default=None,
                        help="Record ID (e.g. JS00010). Omit for random.")
    parser.add_argument("--window", type=float, default=None,
                        help="Seconds to display in the 12-lead grid (default: full recording)")
    args = parser.parse_args()

    if args.record_id:
        record_id = args.record_id
    else:
        available = sorted(p.stem for p in DATASET_DIR.glob("*.hea"))
        record_id = random.choice(available)
        print(f"[Random] {len(available)} records available, selected: {record_id}")

    json_path = PROJECT_ROOT / f"{record_id}_features.json"
    if not json_path.exists():
        print(f"[Info] Features not found, running extraction first...")
        from demo_feature_extraction import main as run_extraction
        run_extraction(record_id)

    visualize(record_id, window_sec=args.window)

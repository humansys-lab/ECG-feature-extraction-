from __future__ import annotations

"""
T027: Boundary overlay visualisation tools.

Quick-use examples
------------------
# Plot a single beat on a single lead
plot_beat(ecg, feat, lead="II", beat_id=3)

# Plot the representative beat for a group
plot_rep_beat(rep_beats, feat, group_id=1, lead="II")

# Plot all 12 leads for one beat (small multi-panel)
plot_beat_all_leads(ecg, feat, beat_id=3)
"""

from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .models import ECGFeatures, LeadBeatFeatures, STANDARD_12_LEADS


# ── colour palette ────────────────────────────────────────────────────────────
_COLOURS = {
    "p_on":    "#4C72B0",
    "p_peak":  "#4C72B0",
    "p_off":   "#4C72B0",
    "qrs_on":  "#DD8452",
    "qrs_off": "#DD8452",
    "t_on":    "#55A868",
    "t_peak":  "#55A868",
    "t_off":   "#C44E52",
}
_LABELS = {
    "p_on": "P-on", "p_off": "P-off",
    "qrs_on": "QRS-on", "qrs_off": "QRS-off",
    "t_on": "T-on", "t_off": "T-end",
}


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualisation: pip install matplotlib")


def _get_beat_features(
    feat: ECGFeatures,
    lead: str,
    beat_id: int,
) -> Optional[LeadBeatFeatures]:
    for bf in feat.beat_features:
        if bf.lead == lead and bf.beat_id == beat_id:
            return bf
    return None


def _draw_boundaries(
    ax,
    bf: LeadBeatFeatures,
    beat_start: int,
    fs: int,
    show_labels: bool = True,
) -> None:
    """Draw vertical boundary lines on an existing axes object."""
    boundaries = {
        "p_on":    bf.p.onset,
        "p_off":   bf.p.offset,
        "qrs_on":  bf.qrs.onset,
        "qrs_off": bf.qrs.offset,
        "t_on":    bf.t.onset,
        "t_off":   bf.t.offset,
    }
    peaks = {
        "p_peak": bf.p.peak,
        "t_peak": bf.t.peak,
    }
    plotted: Dict[str, bool] = {}
    for name, idx in boundaries.items():
        if idx is None:
            continue
        t_s = (idx - beat_start) / fs
        colour = _COLOURS[name]
        ls = "--" if "p_" in name or "t_" in name else "-"
        ax.axvline(t_s, color=colour, lw=1.2, ls=ls, alpha=0.85)
        if show_labels and name in _LABELS and name not in plotted:
            ax.text(t_s, ax.get_ylim()[1] * 0.92, _LABELS[name],
                    color=colour, fontsize=6, ha="center", va="top", rotation=90)
            plotted[name] = True
    for name, idx in peaks.items():
        if idx is None:
            continue
        t_s = (idx - beat_start) / fs
        colour = _COLOURS[name]
        ax.axvline(t_s, color=colour, lw=0.8, ls=":", alpha=0.6)


def plot_beat(
    ecg: np.ndarray,
    feat: ECGFeatures,
    lead: str = "II",
    beat_id: int = 0,
    context_ms: int = 200,
    ax=None,
    show: bool = True,
) -> "plt.Axes":
    """
    Plot one beat on one lead with P/QRS/T boundary overlays.

    Parameters
    ----------
    ecg        : [12, N] raw/filtered signal array
    feat       : ECGFeatures from ECGFeatureExtractor
    lead       : lead name (default "II")
    beat_id    : beat index
    context_ms : extra samples to show left/right of beat window (ms)
    ax         : existing matplotlib Axes (created if None)
    show       : call plt.show() when True
    """
    _require_mpl()
    bf = _get_beat_features(feat, lead, beat_id)
    if bf is None:
        raise ValueError(f"No features found for lead={lead}, beat_id={beat_id}")

    li = STANDARD_12_LEADS.index(lead)
    fs = feat.fs
    r = bf.qrs.peak
    ctx = int(context_ms * fs / 1000)
    beat_start = max(0, r - int(0.35 * fs) - ctx)
    beat_end   = min(ecg.shape[1], r + int(0.55 * fs) + ctx)
    sig = ecg[li, beat_start:beat_end]
    t_ax = np.arange(len(sig)) / fs

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    ax.plot(t_ax, sig, color="k", lw=1.0, zorder=3)
    _draw_boundaries(ax, bf, beat_start, fs)

    # Annotate quality
    noise_str = f"noise={bf.beat_noise_score:.2f}"
    corr_str  = f"corr={bf.beat_template_corr:.2f}"
    rel_str   = "OK" if bf.beat_measurement_reliable else "UNRELIABLE"
    title = f"{lead}  beat {beat_id}  |  {noise_str}  {corr_str}  [{rel_str}]"
    if bf.qt_ms is not None:
        title += f"  QT={bf.qt_ms:.0f}ms"
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("mV", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_rep_beat(
    rep_beats: Dict[int, np.ndarray],
    feat: ECGFeatures,
    group_id: int = 1,
    lead: str = "II",
    left_ms: int = 300,
    ax=None,
    show: bool = True,
) -> "plt.Axes":
    """
    Plot the representative beat for a group/lead.

    Parameters
    ----------
    rep_beats : {group_id: [12, N]} from build_representative_beats
    feat      : ECGFeatures (used for fs)
    group_id  : group to plot (default 1 = dominant)
    lead      : lead name
    left_ms   : ms before R in representative beat window
    """
    _require_mpl()
    if group_id not in rep_beats:
        raise ValueError(f"group_id={group_id} not in rep_beats")
    li = STANDARD_12_LEADS.index(lead)
    fs = feat.fs
    rep = rep_beats[group_id]
    if li >= rep.shape[0]:
        raise ValueError(f"Lead index {li} out of range for rep beat with {rep.shape[0]} leads")
    sig = rep[li]
    local_r = int(left_ms * fs / 1000)
    t_ax = (np.arange(len(sig)) - local_r) / fs * 1000  # ms relative to R

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    ax.plot(t_ax, sig, color="navy", lw=1.2, zorder=3)
    ax.axvline(0, color="gray", lw=0.8, ls="--", label="R")
    ax.set_title(f"Representative beat  group={group_id}  lead={lead}", fontsize=9)
    ax.set_xlabel("Time relative to R (ms)", fontsize=7)
    ax.set_ylabel("mV", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_beat_all_leads(
    ecg: np.ndarray,
    feat: ECGFeatures,
    beat_id: int = 0,
    context_ms: int = 100,
    show: bool = True,
) -> "plt.Figure":
    """
    Plot one beat on all 12 leads with boundary overlays (4×3 grid).

    Parameters
    ----------
    ecg       : [12, N] signal
    feat      : ECGFeatures
    beat_id   : beat index
    context_ms: extra context on each side (ms)
    """
    _require_mpl()
    fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=False)
    axes_flat = axes.flatten()
    for idx, lead in enumerate(STANDARD_12_LEADS):
        plot_beat(ecg, feat, lead=lead, beat_id=beat_id,
                  context_ms=context_ms, ax=axes_flat[idx], show=False)
        axes_flat[idx].set_title(lead, fontsize=8)
        axes_flat[idx].set_xlabel("")
    fig.suptitle(f"Beat {beat_id} — all leads", fontsize=10)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_quality_summary(
    feat: ECGFeatures,
    show: bool = True,
) -> "plt.Figure":
    """
    Bar chart of per-lead mean beat quality scores (T004/T020).
    Useful for a quick overview of which leads are most reliable.
    """
    _require_mpl()
    leads, noise_scores, tcorr_scores, qt_confs = [], [], [], []
    for lead in STANDARD_12_LEADS:
        items = [bf for bf in feat.beat_features if bf.lead == lead]
        if not items:
            continue
        leads.append(lead)
        noise_scores.append(float(np.mean([b.beat_noise_score for b in items])))
        tcorr_scores.append(float(np.mean([b.beat_template_corr for b in items])))
        qt_confs.append(float(np.mean([b.qt_confidence for b in items])))

    x = np.arange(len(leads))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w, noise_scores, w, label="noise_score", color="#e07070")
    ax.bar(x,     tcorr_scores, w, label="template_corr", color="#70a0e0")
    ax.bar(x + w, qt_confs,    w, label="qt_confidence",  color="#70c070")
    ax.set_xticks(x)
    ax.set_xticklabels(leads, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, color="red", lw=0.8, ls="--", alpha=0.6, label="threshold=0.7")
    ax.legend(fontsize=8)
    ax.set_title("Per-lead beat quality summary", fontsize=10)
    ax.set_ylabel("Score", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return fig

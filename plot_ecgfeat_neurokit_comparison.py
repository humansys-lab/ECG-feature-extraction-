#!/usr/bin/env python3
"""Plot ecgfeat and NeuroKit2 annotations on the same LUDB ECG.

The first panel overlays P/QRS/T peaks from both methods on one raw ECG trace.
The following panels repeat that exact trace and show each method's waveform
boundaries separately. LUDB expert annotations can be included as a reference.

Examples
--------
Full record 1, lead II:

    python plot_ecgfeat_neurokit_comparison.py 1 --lead II

Zoom around the third LUDB-annotated QRS:

    python plot_ecgfeat_neurokit_comparison.py 8 --lead V2 --beat 3

Show four consecutive annotated beats:

    python plot_ecgfeat_neurokit_comparison.py 8 --lead V2 --beat-range 2 5

Explicit time window without the LUDB reference panel:

    python plot_ecgfeat_neurokit_comparison.py 1 --lead II \
        --start-sec 1.5 --end-sec 4.5 --no-ludb
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from compare_ludb_detectors import (
    DEFAULT_LUDB_ROOT,
    LEAD_TO_ANN_EXT,
    STANDARD_12_LEADS,
    WaveEvent,
    _lead_indices,
    detect_ecgfeat,
    detect_neurokit2,
    match_events,
    parse_ludb_waves,
    resolve_dataset_data_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ludb_ecgfeat_neurokit_plots"

WAVES = ("P", "QRS", "T")
WAVE_COLORS = {
    "P": "#0072B2",
    "QRS": "#D55E00",
    "T": "#009E73",
}
WAVE_MARKERS = {
    "P": "o",
    "QRS": "v",
    "T": "^",
}
METHOD_COLORS = {
    "ecgfeat": "#2166AC",
    "neurokit2": "#D95F02",
    "ludb": "#333333",
}
METHOD_LABELS = {
    "ecgfeat": "Project ecgfeat",
    "neurokit2": "NeuroKit2",
    "ludb": "LUDB expert",
}


def _events_in_window(
    events: Sequence[WaveEvent],
    first_sample: int,
    last_sample: int,
) -> list[WaveEvent]:
    """Select events whose peak/fiducial lies inside the visible window."""

    return [
        event
        for event in events
        if event.peak is not None
        and first_sample <= int(event.peak) <= last_sample
    ]


def resolve_time_window(
    *,
    n_samples: int,
    fs: int,
    start_sec: float | None,
    end_sec: float | None,
    beat: int | None,
    beat_range: tuple[int, int] | None,
    reference_qrs: Sequence[WaveEvent],
    before_sec: float,
    after_sec: float,
) -> tuple[int, int]:
    """Resolve a full-record, explicit-time, or beat-centred plot window."""

    if n_samples <= 1:
        raise ValueError("ECG must contain at least two samples")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive")
    selected_modes = sum(
        (
            beat is not None,
            beat_range is not None,
            start_sec is not None or end_sec is not None,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "--beat, --beat-range, and --start-sec/--end-sec "
            "are mutually exclusive"
        )
    if before_sec <= 0 or after_sec <= 0:
        raise ValueError("--before-sec and --after-sec must be positive")

    qrs = [event for event in reference_qrs if event.peak is not None]
    if beat is not None:
        if beat <= 0:
            raise ValueError("--beat is 1-based and must be positive")
        if beat > len(qrs):
            raise ValueError(
                f"--beat {beat} is out of range; this lead has "
                f"{len(qrs)} LUDB-annotated QRS events"
            )
        center = int(qrs[beat - 1].peak)
        first = center - int(round(before_sec * fs))
        last = center + int(round(after_sec * fs))
    elif beat_range is not None:
        first_beat, last_beat = beat_range
        if first_beat <= 0 or last_beat <= 0:
            raise ValueError("--beat-range is 1-based and must be positive")
        if first_beat > last_beat:
            raise ValueError("--beat-range FIRST must not exceed LAST")
        if last_beat > len(qrs):
            raise ValueError(
                f"--beat-range {first_beat} {last_beat} is out of range; "
                f"this lead has {len(qrs)} LUDB-annotated QRS events"
            )
        first = int(qrs[first_beat - 1].peak) - int(round(before_sec * fs))
        last = int(qrs[last_beat - 1].peak) + int(round(after_sec * fs))
    else:
        first = 0 if start_sec is None else int(math.floor(start_sec * fs))
        last = (
            n_samples - 1
            if end_sec is None
            else int(math.ceil(end_sec * fs))
        )

    first = max(0, first)
    last = min(n_samples - 1, last)
    if first >= last:
        raise ValueError(
            "The selected time window is empty; check --start-sec/--end-sec"
        )
    return first, last


def _signal_limits(signal: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(signal, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = max(high - low, 0.2)
    padding = max(0.18 * span, 0.12)
    return low - padding, high + padding


def _sample_value(signal: np.ndarray, sample: int | None) -> float | None:
    if sample is None or sample < 0 or sample >= signal.size:
        return None
    value = float(signal[sample])
    return value if math.isfinite(value) else None


def _setup_ecg_axis(
    ax: Any,
    *,
    signal: np.ndarray,
    fs: int,
    first_sample: int,
    last_sample: int,
    y_limits: tuple[float, float],
) -> None:
    import matplotlib.ticker as ticker

    samples = np.arange(first_sample, last_sample + 1)
    ax.plot(
        samples / float(fs),
        signal[first_sample : last_sample + 1],
        color="#161616",
        linewidth=0.9,
        zorder=5,
    )
    ax.set_xlim(first_sample / fs, last_sample / fs)
    ax.set_ylim(*y_limits)
    ax.set_facecolor("#FFFCFB")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which="major", color="#E8B9B9", linewidth=0.55, alpha=0.75)
    ax.grid(which="minor", color="#F4DADA", linewidth=0.3, alpha=0.75)
    ax.set_ylabel("mV")
    for spine in ax.spines.values():
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.6)


def _plot_peak_overlay(
    ax: Any,
    *,
    signal: np.ndarray,
    fs: int,
    first_sample: int,
    last_sample: int,
    methods: Sequence[tuple[str, Mapping[str, Sequence[WaveEvent]]]],
) -> None:
    """Overlay method peaks on the same signal using fill style by method."""

    for method_index, (method, waves) in enumerate(methods):
        color = METHOD_COLORS[method]
        for wave in WAVES:
            visible = _events_in_window(
                waves.get(wave, []),
                first_sample,
                last_sample,
            )
            for event in visible:
                peak = int(event.peak)
                value = _sample_value(signal, peak)
                if value is None:
                    continue
                if method == "ecgfeat":
                    ax.plot(
                        peak / fs,
                        value,
                        marker=WAVE_MARKERS[wave],
                        markersize=6.5,
                        linestyle="none",
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        alpha=0.90,
                        zorder=10 + method_index,
                    )
                elif method == "neurokit2":
                    ax.plot(
                        peak / fs,
                        value,
                        marker=WAVE_MARKERS[wave],
                        markersize=8.5,
                        linestyle="none",
                        markerfacecolor="none",
                        markeredgecolor=color,
                        markeredgewidth=1.5,
                        alpha=0.95,
                        zorder=12 + method_index,
                    )
                else:
                    ax.plot(
                        peak / fs,
                        value,
                        marker="x",
                        markersize=6.0,
                        linestyle="none",
                        color=color,
                        markeredgewidth=1.1,
                        alpha=0.78,
                        zorder=8,
                    )

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_COLORS["ecgfeat"],
            markeredgecolor="white",
            markersize=7,
            label=METHOD_LABELS["ecgfeat"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=METHOD_COLORS["neurokit2"],
            markeredgewidth=1.5,
            markersize=8,
            label=METHOD_LABELS["neurokit2"],
        ),
    ]
    if any(method == "ludb" for method, _waves in methods):
        method_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="none",
                color=METHOD_COLORS["ludb"],
                markersize=7,
                label=METHOD_LABELS["ludb"],
            )
        )
    wave_handles = [
        Line2D(
            [0],
            [0],
            marker=WAVE_MARKERS[wave],
            linestyle="none",
            color="#666666",
            markerfacecolor="#666666",
            markersize=6,
            label=f"{wave} peak" if wave != "QRS" else "R/QRS peak",
        )
        for wave in WAVES
    ]
    ax.legend(
        handles=method_handles + wave_handles,
        loc="upper right",
        ncol=2,
        fontsize=8,
        framealpha=0.94,
    )
    ax.set_title(
        "Peak overlay on the same raw ECG "
        "(filled=ecgfeat, open=NeuroKit2, x=LUDB)",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )


def _plot_wave_boundaries(
    ax: Any,
    *,
    signal: np.ndarray,
    fs: int,
    first_sample: int,
    last_sample: int,
    waves: Mapping[str, Sequence[WaveEvent]],
    method: str,
) -> None:
    """Draw P/QRS/T intervals and peaks for one method."""

    visible_duration = (last_sample - first_sample) / fs
    for wave in WAVES:
        color = WAVE_COLORS[wave]
        visible = _events_in_window(
            waves.get(wave, []),
            first_sample,
            last_sample,
        )
        for event in visible:
            onset = event.onset
            offset = event.offset
            if onset is not None and offset is not None and onset <= offset:
                clipped_onset = max(first_sample, int(onset))
                clipped_offset = min(last_sample, int(offset))
                if clipped_onset <= clipped_offset:
                    ax.axvspan(
                        clipped_onset / fs,
                        clipped_offset / fs,
                        color=color,
                        alpha=0.105,
                        linewidth=0,
                        zorder=2,
                    )
            for boundary, linestyle in ((onset, ":"), (offset, "--")):
                if (
                    boundary is not None
                    and first_sample <= int(boundary) <= last_sample
                ):
                    ax.axvline(
                        int(boundary) / fs,
                        color=color,
                        linestyle=linestyle,
                        linewidth=0.9,
                        alpha=0.64,
                        zorder=4,
                    )
            peak = int(event.peak) if event.peak is not None else None
            value = _sample_value(signal, peak)
            if peak is not None and value is not None:
                ax.plot(
                    peak / fs,
                    value,
                    marker=WAVE_MARKERS[wave],
                    markersize=6.5,
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    zorder=9,
                )
                if visible_duration <= 3.0:
                    ax.annotate(
                        "R" if wave == "QRS" else wave,
                        (peak / fs, value),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=color,
                        fontweight="bold",
                    )

    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=WAVE_COLORS[wave],
            edgecolor=WAVE_COLORS[wave],
            alpha=0.25,
            label=f"{wave} interval",
        )
        for wave in WAVES
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=3,
        fontsize=8,
        framealpha=0.92,
    )
    ax.set_title(
        f"{METHOD_LABELS[method]} boundaries "
        "(dotted=onset, dashed=offset, marker=peak)",
        loc="left",
        fontsize=10,
        color=METHOD_COLORS[method],
        fontweight="bold",
    )


def calculate_window_metrics(
    *,
    ground_truth: Mapping[str, Sequence[WaveEvent]],
    detected: Mapping[str, Sequence[WaveEvent]],
    first_sample: int,
    last_sample: int,
    fs: int,
    r_tolerance_ms: float,
    wave_tolerance_ms: float,
) -> list[dict[str, float | int | str | None]]:
    """Calculate compact visual-window metrics against LUDB."""

    rows: list[dict[str, float | int | str | None]] = []
    for wave in WAVES:
        gt = _events_in_window(ground_truth.get(wave, []), first_sample, last_sample)
        det = _events_in_window(detected.get(wave, []), first_sample, last_sample)
        tolerance_ms = r_tolerance_ms if wave == "QRS" else wave_tolerance_ms
        tolerance_samples = max(1, int(round(tolerance_ms * fs / 1000.0)))
        pairs, fn, fp = match_events(gt, det, tolerance_samples)
        tp = len(pairs)
        denominator = 2 * tp + fp + fn
        errors_ms = [
            (int(found.peak) - int(reference.peak)) * 1000.0 / fs
            for reference, found in pairs
            if reference.peak is not None and found.peak is not None
        ]
        rows.append(
            {
                "wave": wave,
                "gt": len(gt),
                "det": len(det),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": 2 * tp / denominator if denominator else None,
                "peak_mae_ms": (
                    float(np.mean(np.abs(errors_ms))) if errors_ms else None
                ),
            }
        )
    return rows


def _fmt(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def print_metrics(
    method_rows: Sequence[
        tuple[str, Sequence[Mapping[str, float | int | str | None]]]
    ],
) -> None:
    print()
    print(
        f"{'Method':<12} {'Wave':<5} {'GT':>5} {'Det':>5} "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'F1':>7} {'Peak MAE':>10}"
    )
    for method, rows in method_rows:
        for row in rows:
            print(
                f"{method:<12} {str(row['wave']):<5} "
                f"{int(row['gt']):>5} {int(row['det']):>5} "
                f"{int(row['tp']):>5} {int(row['fp']):>5} "
                f"{int(row['fn']):>5} {_fmt(row['f1']):>7} "
                f"{_fmt(row['peak_mae_ms'], 1):>10}"
            )


def render_comparison(
    *,
    signal: np.ndarray,
    fs: int,
    record_id: str,
    lead: str,
    first_sample: int,
    last_sample: int,
    ecgfeat_waves: Mapping[str, Sequence[WaveEvent]],
    neurokit_waves: Mapping[str, Sequence[WaveEvent]],
    ludb_waves: Mapping[str, Sequence[WaveEvent]] | None,
    output_path: Path,
    dpi: int,
    custom_title: str | None,
    ecgfeat_r_source: str,
    ecgfeat_wave_source: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods: list[tuple[str, Mapping[str, Sequence[WaveEvent]]]] = [
        ("ecgfeat", ecgfeat_waves),
        ("neurokit2", neurokit_waves),
    ]
    panels: list[tuple[str, Mapping[str, Sequence[WaveEvent]]]] = list(methods)
    if ludb_waves is not None:
        methods.append(("ludb", ludb_waves))
        panels.append(("ludb", ludb_waves))

    panel_count = 1 + len(panels)
    duration = (last_sample - first_sample) / fs
    figure_width = min(26.0, max(14.0, duration * 1.75))
    figure_height = 2.25 * panel_count + 1.0
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(figure_width, figure_height),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.19},
    )
    axes = np.atleast_1d(axes)
    visible_signal = signal[first_sample : last_sample + 1]
    y_limits = _signal_limits(visible_signal)
    for ax in axes:
        _setup_ecg_axis(
            ax,
            signal=signal,
            fs=fs,
            first_sample=first_sample,
            last_sample=last_sample,
            y_limits=y_limits,
        )

    _plot_peak_overlay(
        axes[0],
        signal=signal,
        fs=fs,
        first_sample=first_sample,
        last_sample=last_sample,
        methods=methods,
    )
    for ax, (method, waves) in zip(axes[1:], panels):
        _plot_wave_boundaries(
            ax,
            signal=signal,
            fs=fs,
            first_sample=first_sample,
            last_sample=last_sample,
            waves=waves,
            method=method,
        )

    axes[-1].set_xlabel("Time (s)")
    title = custom_title or (
        f"LUDB record {record_id} · lead {lead} · "
        "ecgfeat vs NeuroKit2 on the same ECG"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.005,
        (
            "All panels show the same raw ECG samples. "
            "Each detector uses its own internal preprocessing. "
            f"ecgfeat QRS marker source: {ecgfeat_r_source}; "
            f"P/T source: {ecgfeat_wave_source}."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(top=0.955, bottom=0.055, left=0.055, right=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot project ecgfeat and NeuroKit2 P/QRS/T annotations on "
            "the same LUDB ECG."
        )
    )
    parser.add_argument(
        "record",
        nargs="?",
        default="1",
        help="LUDB record ID (default: 1).",
    )
    parser.add_argument(
        "--lead",
        choices=STANDARD_12_LEADS,
        default="II",
        help="Lead to plot (default: II).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_LUDB_ROOT,
        help="LUDB root or its data/ directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: ludb_ecgfeat_neurokit_plots/...",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Visible window start in seconds.",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Visible window end in seconds.",
    )
    parser.add_argument(
        "--beat",
        type=int,
        default=None,
        help="Zoom around the Nth LUDB QRS event (1-based).",
    )
    parser.add_argument(
        "--beat-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        default=None,
        help=(
            "Show an inclusive range of consecutive LUDB QRS events, "
            "for example --beat-range 2 5."
        ),
    )
    parser.add_argument(
        "--before-sec",
        type=float,
        default=0.45,
        help="Seconds before the selected beat (default: 0.45).",
    )
    parser.add_argument(
        "--after-sec",
        type=float,
        default=0.70,
        help="Seconds after the selected beat (default: 0.70).",
    )
    parser.add_argument(
        "--no-ludb",
        action="store_true",
        help="Hide the LUDB expert-reference markers and panel.",
    )
    parser.add_argument(
        "--neurokit-peak-method",
        choices=(
            "neurokit",
            "pantompkins1985",
            "hamilton2002",
            "elgendi2010",
            "engzeemod2012",
        ),
        default="neurokit",
        help="NeuroKit2 cleaning/R-peak method.",
    )
    parser.add_argument(
        "--neurokit-delineate-method",
        choices=("peak", "prominence", "cwt", "dwt"),
        default="dwt",
        help="NeuroKit2 delineation method (default: dwt).",
    )
    parser.add_argument(
        "--r-tolerance-ms",
        type=float,
        default=75.0,
        help="R/QRS metric matching tolerance (default: 75 ms).",
    )
    parser.add_argument(
        "--wave-tolerance-ms",
        type=float,
        default=150.0,
        help="P/T metric matching tolerance (default: 150 ms).",
    )
    parser.add_argument(
        "--mains-freq",
        type=float,
        default=50.0,
        help="Mains frequency passed to ecgfeat (default: 50 Hz).",
    )
    parser.add_argument(
        "--ecgfeat-r-source",
        choices=(
            "hybrid-prominence",
            "lead-fiducial",
            "positive-r",
            "global",
        ),
        default="hybrid-prominence",
        help=(
            "ecgfeat QRS marker to plot: the parallel per-lead prominence "
            "location (default), each lead's local QRS fiducial, its maximum "
            "positive R deflection, or the global multilead fiducial."
        ),
    )
    parser.add_argument(
        "--ecgfeat-wave-source",
        choices=("native", "hybrid-peaks"),
        default="hybrid-peaks",
        help=(
            "ecgfeat P/T peak source. The default refines T with the bipolar "
            "prominence side channel while retaining native P timing."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image resolution (default: 180 dpi).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom figure title.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    import wfdb

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.r_tolerance_ms <= 0 or args.wave_tolerance_ms <= 0:
        raise ValueError("Matching tolerances must be positive")

    data_dir = resolve_dataset_data_dir(args.dataset_dir)
    record_path = data_dir / str(args.record)
    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(f"LUDB record not found: {record_path}.hea")

    record = wfdb.rdrecord(str(record_path))
    fs = int(round(float(record.fs)))
    signal_matrix = np.asarray(record.p_signal, dtype=float)
    indices = _lead_indices(record.sig_name)
    lead_signal = signal_matrix[:, indices[args.lead]]
    ecg_12lead = signal_matrix[
        :,
        [indices[lead] for lead in STANDARD_12_LEADS],
    ].T

    print(
        f"Running ecgfeat and NeuroKit2: record={args.record} "
        f"lead={args.lead} fs={fs}Hz samples={lead_signal.size}",
        flush=True,
    )
    ecgfeat_all, ecgfeat_seconds = detect_ecgfeat(
        ecg_12lead,
        fs,
        (args.lead,),
        args.mains_freq,
        qrs_peak_source=args.ecgfeat_r_source,
        wave_peak_source=args.ecgfeat_wave_source,
    )
    neurokit_waves = detect_neurokit2(
        lead_signal,
        fs,
        args.neurokit_peak_method,
        args.neurokit_delineate_method,
    )
    ludb_waves = parse_ludb_waves(
        record_path,
        LEAD_TO_ANN_EXT[args.lead],
    )

    first_sample, last_sample = resolve_time_window(
        n_samples=lead_signal.size,
        fs=fs,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        beat=args.beat,
        beat_range=(
            tuple(args.beat_range)
            if args.beat_range is not None
            else None
        ),
        reference_qrs=ludb_waves["QRS"],
        before_sec=args.before_sec,
        after_sec=args.after_sec,
    )
    ecgfeat_waves = ecgfeat_all[args.lead]
    output_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (
            DEFAULT_OUTPUT_DIR
            / f"record_{args.record}_{args.lead}_ecgfeat_vs_neurokit.png"
        ).resolve()
    )

    render_comparison(
        signal=lead_signal,
        fs=fs,
        record_id=str(args.record),
        lead=args.lead,
        first_sample=first_sample,
        last_sample=last_sample,
        ecgfeat_waves=ecgfeat_waves,
        neurokit_waves=neurokit_waves,
        ludb_waves=None if args.no_ludb else ludb_waves,
        output_path=output_path,
        dpi=args.dpi,
        custom_title=args.title,
        ecgfeat_r_source=args.ecgfeat_r_source,
        ecgfeat_wave_source=args.ecgfeat_wave_source,
    )

    method_rows = []
    for method, detected in (
        ("ecgfeat", ecgfeat_waves),
        ("neurokit2", neurokit_waves),
    ):
        method_rows.append(
            (
                method,
                calculate_window_metrics(
                    ground_truth=ludb_waves,
                    detected=detected,
                    first_sample=first_sample,
                    last_sample=last_sample,
                    fs=fs,
                    r_tolerance_ms=args.r_tolerance_ms,
                    wave_tolerance_ms=args.wave_tolerance_ms,
                ),
            )
        )
    print_metrics(method_rows)
    print()
    print(
        f"Window: {first_sample / fs:.3f}s–{last_sample / fs:.3f}s | "
        f"ecgfeat runtime: {ecgfeat_seconds:.3f}s"
    )
    print(f"Saved: {output_path}")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

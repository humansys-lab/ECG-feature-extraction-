#!/usr/bin/env python3
"""Plot complete 10-second, 12-lead ecgfeat vs NeuroKit2 comparisons.

For each LUDB record this script runs ecgfeat once on all 12 leads, runs
NeuroKit2 independently on each lead, and writes four full-duration pages:

1. ecgfeat / NeuroKit2 / LUDB peak overlay;
2. ecgfeat P/QRS/T boundaries;
3. NeuroKit2 P/QRS/T boundaries;
4. LUDB expert P/QRS/T boundaries.

The pages are saved as individual PNGs and as one multi-page PDF.

Examples
--------
Record 1:

    python plot_ludb_12lead_10s_comparison.py 1

Several records:

    python plot_ludb_12lead_10s_comparison.py 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import math
import time
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
    parse_ludb_waves,
    resolve_dataset_data_dir,
)
from plot_ecgfeat_neurokit_comparison import (
    METHOD_COLORS,
    WAVE_COLORS,
    WAVE_MARKERS,
    WAVES,
    _sample_value,
    _signal_limits,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ludb_12lead_10s_comparison"


def _visible_events(
    events: Sequence[WaveEvent],
    n_samples: int,
) -> list[WaveEvent]:
    return [
        event
        for event in events
        if event.peak is not None and 0 <= int(event.peak) < n_samples
    ]


def _setup_axis(
    ax: Any,
    *,
    signal: np.ndarray,
    fs: int,
    lead: str,
    show_x_labels: bool,
) -> None:
    import matplotlib.ticker as ticker

    n_samples = signal.size
    time_axis = np.arange(n_samples) / float(fs)
    ax.plot(time_axis, signal, color="#161616", linewidth=0.72, zorder=5)
    ax.set_xlim(0.0, n_samples / float(fs))
    ax.set_ylim(*_signal_limits(signal))
    ax.set_facecolor("#FFFCFB")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which="major", color="#E8B9B9", linewidth=0.45, alpha=0.68)
    ax.grid(which="minor", color="#F4DADA", linewidth=0.22, alpha=0.68)
    ax.text(
        0.006,
        0.88,
        lead,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.86,
            "pad": 2.0,
        },
        zorder=20,
    )
    ax.set_ylabel("mV", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(
        axis="x",
        labelsize=7,
        labelbottom=show_x_labels,
    )
    for spine in ax.spines.values():
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.5)


def _plot_method_peaks(
    ax: Any,
    *,
    signal: np.ndarray,
    fs: int,
    waves: Mapping[str, Sequence[WaveEvent]],
    method: str,
) -> None:
    color = METHOD_COLORS[method]
    for wave in WAVES:
        for event in _visible_events(waves.get(wave, []), signal.size):
            peak = int(event.peak)
            value = _sample_value(signal, peak)
            if value is None:
                continue
            if method == "ecgfeat":
                ax.plot(
                    peak / fs,
                    value,
                    marker=WAVE_MARKERS[wave],
                    markersize=4.3,
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    alpha=0.92,
                    zorder=12,
                )
            elif method == "neurokit2":
                ax.plot(
                    peak / fs,
                    value,
                    marker=WAVE_MARKERS[wave],
                    markersize=5.7,
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    alpha=0.94,
                    zorder=13,
                )
            else:
                ax.plot(
                    peak / fs,
                    value,
                    marker="x",
                    markersize=4.0,
                    linestyle="none",
                    color=color,
                    markeredgewidth=0.8,
                    alpha=0.78,
                    zorder=10,
                )


def _plot_boundaries(
    ax: Any,
    *,
    signal: np.ndarray,
    fs: int,
    waves: Mapping[str, Sequence[WaveEvent]],
) -> None:
    for wave in WAVES:
        color = WAVE_COLORS[wave]
        for event in _visible_events(waves.get(wave, []), signal.size):
            if (
                event.onset is not None
                and event.offset is not None
                and int(event.onset) <= int(event.offset)
            ):
                onset = max(0, int(event.onset))
                offset = min(signal.size - 1, int(event.offset))
                if onset <= offset:
                    ax.axvspan(
                        onset / fs,
                        offset / fs,
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                        zorder=2,
                    )
            for boundary, linestyle in (
                (event.onset, ":"),
                (event.offset, "--"),
            ):
                if boundary is not None and 0 <= int(boundary) < signal.size:
                    ax.axvline(
                        int(boundary) / fs,
                        color=color,
                        linestyle=linestyle,
                        linewidth=0.60,
                        alpha=0.58,
                        zorder=4,
                    )
            peak = int(event.peak)
            value = _sample_value(signal, peak)
            if value is not None:
                ax.plot(
                    peak / fs,
                    value,
                    marker=WAVE_MARKERS[wave],
                    markersize=4.2,
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    zorder=9,
                )


def _peak_legend_handles(include_ludb: bool) -> list[Any]:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_COLORS["ecgfeat"],
            markeredgecolor="white",
            markersize=7,
            label="Project ecgfeat (filled)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=METHOD_COLORS["neurokit2"],
            markersize=7,
            label="NeuroKit2 (open)",
        ),
    ]
    if include_ludb:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="none",
                color=METHOD_COLORS["ludb"],
                markersize=7,
                label="LUDB expert",
            )
        )
    handles.extend(
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
    )
    return handles


def _boundary_legend_handles() -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(
            facecolor=WAVE_COLORS[wave],
            edgecolor=WAVE_COLORS[wave],
            alpha=0.24,
            label=f"{wave} interval",
        )
        for wave in WAVES
    ]


def _create_page(
    *,
    signal_matrix: np.ndarray,
    indices: Mapping[str, int],
    fs: int,
    record_id: str,
    page_kind: str,
    ecgfeat_by_lead: Mapping[str, Mapping[str, Sequence[WaveEvent]]],
    neurokit_by_lead: Mapping[str, Mapping[str, Sequence[WaveEvent]]],
    ludb_by_lead: Mapping[str, Mapping[str, Sequence[WaveEvent]]],
    include_ludb: bool,
    ecgfeat_r_source: str,
    ecgfeat_wave_source: str,
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(STANDARD_12_LEADS),
        1,
        figsize=(23.0, 25.5),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    duration = signal_matrix.shape[0] / float(fs)
    page_titles = {
        "peaks": (
            "Peak overlay: project ecgfeat vs NeuroKit2"
            + (" vs LUDB expert" if include_ludb else "")
        ),
        "ecgfeat": "Project ecgfeat P/QRS/T boundaries",
        "neurokit2": "NeuroKit2 P/QRS/T boundaries",
        "ludb": "LUDB expert P/QRS/T boundaries",
    }
    for lead_index, (ax, lead) in enumerate(zip(axes, STANDARD_12_LEADS)):
        signal = signal_matrix[:, indices[lead]]
        _setup_axis(
            ax,
            signal=signal,
            fs=fs,
            lead=lead,
            show_x_labels=lead_index == len(STANDARD_12_LEADS) - 1,
        )
        if page_kind == "peaks":
            _plot_method_peaks(
                ax,
                signal=signal,
                fs=fs,
                waves=ecgfeat_by_lead[lead],
                method="ecgfeat",
            )
            _plot_method_peaks(
                ax,
                signal=signal,
                fs=fs,
                waves=neurokit_by_lead[lead],
                method="neurokit2",
            )
            if include_ludb:
                _plot_method_peaks(
                    ax,
                    signal=signal,
                    fs=fs,
                    waves=ludb_by_lead[lead],
                    method="ludb",
                )
        else:
            source = {
                "ecgfeat": ecgfeat_by_lead,
                "neurokit2": neurokit_by_lead,
                "ludb": ludb_by_lead,
            }[page_kind]
            _plot_boundaries(
                ax,
                signal=signal,
                fs=fs,
                waves=source[lead],
            )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"LUDB record {record_id} · complete 12-lead ECG · "
        f"{duration:.1f} s\n{page_titles[page_kind]}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    handles = (
        _peak_legend_handles(include_ludb)
        if page_kind == "peaks"
        else _boundary_legend_handles()
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.969),
        ncol=len(handles),
        fontsize=9,
        framealpha=0.95,
    )
    footer = (
        "All methods are displayed on the same raw ECG. "
        f"ecgfeat QRS marker source: {ecgfeat_r_source}; "
        f"P/T source: {ecgfeat_wave_source}; "
        "beat presence still comes from its global multilead detection; "
        "NeuroKit2 is run independently per lead."
        if page_kind == "peaks"
        else "Dotted lines = onset · dashed lines = offset · marker = peak"
    )
    fig.text(
        0.5,
        0.006,
        footer,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(
        top=0.948,
        bottom=0.035,
        left=0.047,
        right=0.992,
    )
    return fig


def render_record(
    *,
    record_id: str,
    signal_matrix: np.ndarray,
    indices: Mapping[str, int],
    fs: int,
    ecgfeat_by_lead: Mapping[str, Mapping[str, Sequence[WaveEvent]]],
    neurokit_by_lead: Mapping[str, Mapping[str, Sequence[WaveEvent]]],
    ludb_by_lead: Mapping[str, Mapping[str, Sequence[WaveEvent]]],
    output_dir: Path,
    dpi: int,
    include_ludb: bool,
    make_pdf: bool,
    ecgfeat_r_source: str,
    ecgfeat_wave_source: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    record_dir = output_dir / f"record_{record_id}"
    record_dir.mkdir(parents=True, exist_ok=True)
    page_kinds = ["peaks", "ecgfeat", "neurokit2"]
    if include_ludb:
        page_kinds.append("ludb")

    output_paths: list[Path] = []
    pdf_path = record_dir / f"record_{record_id}_12lead_10s_comparison.pdf"
    pdf_context = PdfPages(pdf_path) if make_pdf else None
    try:
        for page_kind in page_kinds:
            figure = _create_page(
                signal_matrix=signal_matrix,
                indices=indices,
                fs=fs,
                record_id=record_id,
                page_kind=page_kind,
                ecgfeat_by_lead=ecgfeat_by_lead,
                neurokit_by_lead=neurokit_by_lead,
                ludb_by_lead=ludb_by_lead,
                include_ludb=include_ludb,
                ecgfeat_r_source=ecgfeat_r_source,
                ecgfeat_wave_source=ecgfeat_wave_source,
            )
            png_path = (
                record_dir
                / f"record_{record_id}_12lead_10s_{page_kind}.png"
            )
            figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
            output_paths.append(png_path)
            if pdf_context is not None:
                pdf_context.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    finally:
        if pdf_context is not None:
            pdf_context.close()
    if make_pdf:
        output_paths.append(pdf_path)
    return output_paths


def process_record(
    *,
    record_id: str,
    data_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    import wfdb

    record_path = data_dir / record_id
    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(f"LUDB record not found: {record_path}.hea")

    record = wfdb.rdrecord(str(record_path))
    fs = int(round(float(record.fs)))
    signal_matrix = np.asarray(record.p_signal, dtype=float)
    indices = _lead_indices(record.sig_name)
    ecg_12lead = signal_matrix[
        :,
        [indices[lead] for lead in STANDARD_12_LEADS],
    ].T
    duration = signal_matrix.shape[0] / float(fs)
    print(
        f"Record {record_id}: {len(STANDARD_12_LEADS)} leads, "
        f"{duration:.3f}s, {fs}Hz",
        flush=True,
    )

    started = time.perf_counter()
    ecgfeat_by_lead, _ecgfeat_seconds = detect_ecgfeat(
        ecg_12lead,
        fs,
        STANDARD_12_LEADS,
        args.mains_freq,
        qrs_peak_source=args.ecgfeat_r_source,
        wave_peak_source=args.ecgfeat_wave_source,
    )
    neurokit_by_lead = {
        lead: detect_neurokit2(
            signal_matrix[:, indices[lead]],
            fs,
            args.neurokit_peak_method,
            args.neurokit_delineate_method,
        )
        for lead in STANDARD_12_LEADS
    }
    ludb_by_lead = {
        lead: parse_ludb_waves(record_path, LEAD_TO_ANN_EXT[lead])
        for lead in STANDARD_12_LEADS
    }
    detection_seconds = time.perf_counter() - started

    paths = render_record(
        record_id=record_id,
        signal_matrix=signal_matrix,
        indices=indices,
        fs=fs,
        ecgfeat_by_lead=ecgfeat_by_lead,
        neurokit_by_lead=neurokit_by_lead,
        ludb_by_lead=ludb_by_lead,
        output_dir=output_dir,
        dpi=args.dpi,
        include_ludb=not args.no_ludb,
        make_pdf=not args.no_pdf,
        ecgfeat_r_source=args.ecgfeat_r_source,
        ecgfeat_wave_source=args.ecgfeat_wave_source,
    )
    print(
        f"Record {record_id}: detection={detection_seconds:.2f}s, "
        f"artifacts={len(paths)}, output={paths[0].parent}",
        flush=True,
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw complete 10-second, 12-lead ecgfeat/NeuroKit2/LUDB "
            "comparison pages."
        )
    )
    parser.add_argument(
        "records",
        nargs="+",
        help="One or more LUDB record IDs.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_LUDB_ROOT,
        help="LUDB root or its data/ directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root directory.",
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
    )
    parser.add_argument(
        "--neurokit-delineate-method",
        choices=("peak", "prominence", "cwt", "dwt"),
        default="dwt",
    )
    parser.add_argument("--mains-freq", type=float, default=50.0)
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
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--no-ludb",
        action="store_true",
        help="Omit LUDB markers/page.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write PNG pages only.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.mains_freq <= 0:
        raise ValueError("--mains-freq must be positive")
    data_dir = resolve_dataset_data_dir(args.dataset_dir)
    output_dir = args.out_dir.expanduser().resolve()
    seen: set[str] = set()
    records = []
    for value in args.records:
        record_id = str(value)
        if record_id not in seen:
            seen.add(record_id)
            records.append(record_id)
    for index, record_id in enumerate(records, start=1):
        print(f"[{index}/{len(records)}]", end=" ", flush=True)
        process_record(
            record_id=record_id,
            data_dir=data_dir,
            output_dir=output_dir,
            args=args,
        )
    print(f"Completed {len(records)} record(s): {output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render every LUDB waveform detection against the expert annotation.

For each record this runs ecgfeat once on all 12 leads and draws, per lead,
the full 10 s trace with:

* the LUDB expert annotation as a shaded band per wave (onset..offset) with a
  tick at the annotated peak;
* ecgfeat's detection as a stem at the detected peak with onset/offset caps;
* every scoring outcome marked -- a hollow circle on a missed reference event
  (FN) and a cross on an unmatched detection (FP);
* the annotated span, since LUDB only labels a middle window of each record
  and detections outside it are not scoreable either way.

Matching uses ``compare_ludb_detectors.match_events`` with the benchmark's own
tolerances, so the FP/FN marks here are exactly the events counted in
``compare_ludb_detectors.py`` / ``evaluate_ludb.py`` runs.

Matplotlib has no CJK font in this environment, so all in-figure text is
English by design.

Examples
--------
Every record, one page per record (the default; 200 PNGs, ~10 min on 12 workers)::

    python plot_ludb_delineation.py --workers 12

A few records, and one page per lead instead::

    python plot_ludb_delineation.py --records 92 117 121 --layout lead

Only the leads that actually contain an error, P waves only::

    python plot_ludb_delineation.py --waves P --only-errors

Zoom into 2-5 s of lead II::

    python plot_ludb_delineation.py --records 117 --leads II --time-window 2 5
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "feature_extraction"))

from compare_ludb_detectors import (  # noqa: E402
    DEFAULT_LUDB_ROOT,
    LEAD_TO_ANN_EXT,
    STANDARD_12_LEADS,
    WaveEvent,
    _lead_indices,
    detect_ecgfeat,
    discover_records,
    match_events,
    parse_ludb_waves,
    resolve_dataset_data_dir,
    restrict_to_annotated_span,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ludb_delineation_plots"
WAVES = ("P", "QRS", "T")

# Tolerances the LUDB benchmark scores with; keep these in sync with
# compare_ludb_detectors' defaults or the marks stop meaning the same thing.
TOLERANCE_MS = {"QRS": 75.0, "P": 150.0, "T": 150.0}

# Categorical hues, fixed order, validated for CVD separation as a triple.
WAVE_COLOR = {"P": "#2a78d6", "QRS": "#eb6834", "T": "#1baf7a"}
ERROR_COLOR = "#c1272d"
TRACE_COLOR = "#22272b"
SPAN_COLOR = "#9aa5a8"


# ── scoring ──────────────────────────────────────────────────────────────────


def score_lead(
    gt_events: Sequence[WaveEvent],
    detected_events: Sequence[WaveEvent],
    fs: int,
    wave: str,
    restrict_span: bool,
) -> dict[str, Any]:
    """Match one lead's events and return the drawable outcome, not just counts.

    ``match_events`` reports false positives and negatives as integers, which
    is all the benchmark needs.  A plot needs to know *which* events they were,
    so the matched pairs are used to partition the two event lists by identity.
    """

    tolerance_samples = max(1, int(round(TOLERANCE_MS[wave] * fs / 1000.0)))
    outside_span = 0
    scoreable = list(detected_events)
    if restrict_span:
        scoreable, outside_span = restrict_to_annotated_span(
            gt_events, scoreable, tolerance_samples
        )
    pairs, _, _ = match_events(gt_events, scoreable, tolerance_samples)

    matched_gt = {id(gt) for gt, _ in pairs}
    matched_det = {id(det) for _, det in pairs}
    scoreable_ids = {id(event) for event in scoreable}
    dropped = [event for event in detected_events if id(event) not in scoreable_ids]
    return {
        "wave": wave,
        "pairs": pairs,
        "false_negatives": [
            event
            for event in gt_events
            if event.peak is not None and id(event) not in matched_gt
        ],
        "false_positives": [
            event
            for event in scoreable
            if event.peak is not None and id(event) not in matched_det
        ],
        "outside_span": dropped,
        "outside_span_count": outside_span,
        "gt_count": sum(event.peak is not None for event in gt_events),
        "tp": len(pairs),
    }


def annotated_span(gt_by_wave: Mapping[str, Sequence[WaveEvent]]) -> tuple[int, int] | None:
    """Sample range the expert actually labelled, across all three waves."""

    peaks = [
        int(event.peak)
        for events in gt_by_wave.values()
        for event in events
        if event.peak is not None
    ]
    return (min(peaks), max(peaks)) if peaks else None


# ── drawing ──────────────────────────────────────────────────────────────────


def _draw_lead(
    axis: plt.Axes,
    signal: np.ndarray,
    fs: int,
    lead: str,
    gt_by_wave: Mapping[str, Sequence[WaveEvent]],
    det_by_wave: Mapping[str, Sequence[WaveEvent]],
    scores: Mapping[str, dict[str, Any]],
    waves: Sequence[str],
    window: tuple[float, float] | None,
    show_outside_span: bool,
) -> None:
    times = np.arange(signal.size) / float(fs)
    axis.plot(times, signal, color=TRACE_COLOR, linewidth=0.75, zorder=3)

    finite = signal[np.isfinite(signal)]
    if finite.size:
        low, high = float(np.min(finite)), float(np.max(finite))
    else:
        low, high = -1.0, 1.0
    pad = max((high - low) * 0.18, 0.05)
    axis.set_ylim(low - pad, high + pad * 1.9)

    span = annotated_span(gt_by_wave)
    if span is not None:
        for edge in span:
            axis.axvline(edge / fs, color=SPAN_COLOR, linewidth=0.8,
                         linestyle=(0, (4, 3)), zorder=1)
        axis.axvspan(times[0], span[0] / fs, color=SPAN_COLOR, alpha=0.10, zorder=0)
        axis.axvspan(span[1] / fs, times[-1], color=SPAN_COLOR, alpha=0.10, zorder=0)

    stem_top = high + pad * 0.55
    for wave in waves:
        color = WAVE_COLOR[wave]

        # Reference: a band over the annotated extent plus a peak tick.
        for event in gt_by_wave.get(wave, ()):
            if event.onset is not None and event.offset is not None:
                axis.axvspan(event.onset / fs, event.offset / fs,
                             color=color, alpha=0.16, linewidth=0, zorder=2)
            if event.peak is not None:
                axis.plot([event.peak / fs], [low - pad * 0.55], marker="^",
                          markersize=3.6, color=color, zorder=4)

        # Detection: a stem at the peak with onset/offset caps.
        for event in det_by_wave.get(wave, ()):
            if event.peak is None:
                continue
            axis.vlines(event.peak / fs, low - pad * 0.2, stem_top,
                        color=color, linewidth=0.9, alpha=0.85, zorder=4)
            for boundary in (event.onset, event.offset):
                if boundary is not None:
                    axis.plot([boundary / fs], [stem_top], marker="|",
                              markersize=5, color=color, zorder=4)

        outcome = scores.get(wave)
        if not outcome:
            continue
        for event in outcome["false_negatives"]:
            axis.plot([event.peak / fs], [stem_top], marker="o", markersize=7,
                      markerfacecolor="none", markeredgecolor=ERROR_COLOR,
                      markeredgewidth=1.5, zorder=6)
        for event in outcome["false_positives"]:
            axis.plot([event.peak / fs], [stem_top], marker="x", markersize=7,
                      color=ERROR_COLOR, markeredgewidth=1.5, zorder=6)
        if show_outside_span:
            for event in outcome["outside_span"]:
                if event.peak is not None:
                    axis.plot([event.peak / fs], [stem_top], marker=".",
                              markersize=4, color=SPAN_COLOR, zorder=5)

    summary = "  ".join(
        f"{wave} {scores[wave]['tp']}/{scores[wave]['gt_count']}"
        + (f" +{len(scores[wave]['false_positives'])}fp" if scores[wave]["false_positives"] else "")
        for wave in waves
        if wave in scores
    )
    axis.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=10,
                    fontweight="bold", labelpad=12)
    axis.text(0.998, 0.94, summary, transform=axis.transAxes, ha="right", va="top",
              fontsize=7.5, family="monospace", color="#4a5457")
    axis.set_xlim(*(window if window else (times[0], times[-1])))
    axis.grid(True, axis="x", color="#d8dcde", linewidth=0.5, zorder=0)
    axis.tick_params(labelsize=8)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#c3c9cb")


def _legend_handles(waves: Sequence[str], show_outside_span: bool) -> list[Any]:
    handles: list[Any] = []
    for wave in waves:
        handles.append(Patch(facecolor=WAVE_COLOR[wave], alpha=0.3,
                             label=f"{wave} reference (onset-offset)"))
        handles.append(Line2D([], [], color=WAVE_COLOR[wave], linewidth=1.2,
                              label=f"{wave} detected peak"))
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor="none", markeredgecolor=ERROR_COLOR,
                          markeredgewidth=1.5, label="missed reference (FN)"))
    handles.append(Line2D([], [], marker="x", linestyle="none", markersize=7,
                          color=ERROR_COLOR, markeredgewidth=1.5,
                          label="unmatched detection (FP)"))
    handles.append(Line2D([], [], color=SPAN_COLOR, linewidth=0.9,
                          linestyle=(0, (4, 3)), label="annotated span edge"))
    if show_outside_span:
        handles.append(Line2D([], [], marker=".", linestyle="none", markersize=6,
                              color=SPAN_COLOR, label="detection outside span (not scored)"))
    return handles


def _title(record_id: str, scores_by_lead: Mapping[str, Mapping[str, dict[str, Any]]],
           waves: Sequence[str], header: str) -> str:
    parts = []
    for wave in waves:
        tp = sum(s[wave]["tp"] for s in scores_by_lead.values() if wave in s)
        gt = sum(s[wave]["gt_count"] for s in scores_by_lead.values() if wave in s)
        fp = sum(len(s[wave]["false_positives"]) for s in scores_by_lead.values() if wave in s)
        se = f"{tp / gt * 100:.1f}%" if gt else "n/a"
        parts.append(f"{wave} Se {se} ({tp}/{gt}), {fp} FP")
    subtitle = "   |   ".join(parts)
    return f"LUDB record {record_id} — ecgfeat vs expert annotation\n{subtitle}\n{header}"


def _record_header(data_dir: Path, record_id: str) -> str:
    """One-line clinical context from the record header, for the figure title."""

    header_path = data_dir / f"{record_id}.hea"
    if not header_path.exists():
        return ""
    fields = [
        line[1:].strip()
        for line in header_path.read_text().splitlines()
        if line.startswith("#") and line[1:].strip()
    ]
    text = " ".join(fields).replace("<age>:", "age").replace("<sex>:", "sex")
    text = text.replace("<diagnoses>:", "|")
    return text[:170] + ("..." if len(text) > 170 else "")


# ── per-record driver ────────────────────────────────────────────────────────


def render_record(record_id: str, args: argparse.Namespace, data_dir: Path) -> list[str]:
    import wfdb

    record_path = data_dir / record_id
    record = wfdb.rdrecord(str(record_path))
    fs = int(round(float(record.fs)))
    signal_matrix = np.asarray(record.p_signal, dtype=float)
    indices = _lead_indices(record.sig_name)
    ecg_12lead = signal_matrix[:, [indices[lead] for lead in STANDARD_12_LEADS]].T

    per_lead, _ = detect_ecgfeat(
        ecg_12lead,
        fs,
        tuple(args.leads),
        args.mains_freq,
        qrs_peak_source=args.ecgfeat_r_source,
        wave_peak_source=args.ecgfeat_wave_source,
    )

    gt_by_lead: dict[str, dict[str, list[WaveEvent]]] = {}
    scores_by_lead: dict[str, dict[str, dict[str, Any]]] = {}
    for lead in args.leads:
        gt = parse_ludb_waves(record_path, LEAD_TO_ANN_EXT[lead])
        gt_by_lead[lead] = gt
        scores_by_lead[lead] = {
            wave: score_lead(gt.get(wave, []), per_lead[lead].get(wave, []),
                             fs, wave, args.restrict_to_annotated_span)
            for wave in args.waves
        }

    drawn = [
        lead
        for lead in args.leads
        if not args.only_errors
        or any(
            scores_by_lead[lead][wave]["false_negatives"]
            or scores_by_lead[lead][wave]["false_positives"]
            for wave in args.waves
        )
    ]
    if not drawn:
        return []

    window = tuple(args.time_window) if args.time_window else None
    header = _record_header(data_dir, record_id)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    groups = [drawn] if args.layout == "record" else [[lead] for lead in drawn]
    for group in groups:
        # Title and legend need a fixed number of *inches*, not a fixed fraction
        # of the figure, or they collide with the axes on short single-lead pages
        # and swallow them on tall ones.  Lay the page out in inches and convert
        # to fractions once, instead of letting tight_layout guess.
        lead_height = 1.15 if args.layout == "record" else 2.6
        title_in, legend_in = 1.0, 0.85
        height = title_in + legend_in + lead_height * len(group)
        figure, axes = plt.subplots(
            len(group), 1, figsize=(args.width, height), sharex=True, squeeze=False
        )
        for axis, lead in zip(axes[:, 0], group):
            _draw_lead(
                axis,
                signal_matrix[:, indices[lead]],
                fs,
                lead,
                gt_by_lead[lead],
                per_lead[lead],
                scores_by_lead[lead],
                args.waves,
                window,
                args.show_outside_span,
            )
        axes[-1, 0].set_xlabel("time (s)", fontsize=9)
        figure.suptitle(
            _title(record_id, {lead: scores_by_lead[lead] for lead in group},
                   args.waves, header),
            fontsize=11, ha="left", x=0.012, y=0.995,
        )
        figure.legend(
            handles=_legend_handles(args.waves, args.show_outside_span),
            loc="lower center", ncol=min(5, 2 * len(args.waves) + 1),
            fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.004),
        )
        figure.subplots_adjust(
            left=0.05, right=0.995,
            top=1.0 - title_in / height, bottom=legend_in / height,
            hspace=0.28,
        )

        stem = record_id if args.layout == "record" else f"{record_id}_{group[0]}"
        path = output_dir / f"{stem}.png"
        figure.savefig(path, dpi=args.dpi)
        plt.close(figure)
        written.append(str(path))

    return written


def _task(payload: tuple[str, argparse.Namespace, Path]) -> tuple[str, list[str], str]:
    record_id, args, data_dir = payload
    try:
        return record_id, render_record(record_id, args, data_dir), ""
    except Exception as exc:  # noqa: BLE001 - one bad record must not stop the run
        return record_id, [], f"{type(exc).__name__}: {exc}"


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_LUDB_ROOT,
                        help="LUDB root or its data/ directory.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the PNGs (created if missing).")
    parser.add_argument("--records", nargs="+", default=None,
                        help="Record IDs to render. Default: every record found.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Render only the first N selected records.")
    parser.add_argument("--leads", nargs="+", default=list(STANDARD_12_LEADS),
                        choices=list(STANDARD_12_LEADS), metavar="LEAD",
                        help="Leads to draw. Default: all 12.")
    parser.add_argument("--waves", nargs="+", default=list(WAVES),
                        choices=list(WAVES), metavar="WAVE",
                        help="Waves to draw. Default: P QRS T.")
    parser.add_argument("--layout", choices=("record", "lead"), default="record",
                        help="One page per record (default) or one page per lead.")
    parser.add_argument("--only-errors", action="store_true",
                        help="Skip leads with no FP and no FN (with --layout record, "
                             "skips the record if every lead is clean).")
    parser.add_argument("--time-window", nargs=2, type=float, metavar=("START", "END"),
                        default=None, help="Zoom to START..END seconds.")
    parser.add_argument("--no-restrict-to-annotated-span", dest="restrict_to_annotated_span",
                        action="store_false",
                        help="Count detections outside the annotated span as false "
                             "positives. Off by default because LUDB labels only a "
                             "middle window, so those are annotation gaps, not errors.")
    parser.add_argument("--show-outside-span", action="store_true",
                        help="Draw unscored out-of-span detections as faint dots.")
    parser.add_argument("--ecgfeat-r-source", default="global",
                        choices=("global", "lead-fiducial", "positive-r", "hybrid-prominence"),
                        help="QRS marker to draw. 'global' matches the benchmark; "
                             "'positive-r' is the clinical R peak.")
    parser.add_argument("--ecgfeat-wave-source", default="native",
                        choices=("native", "hybrid-peaks"))
    parser.add_argument("--mains-freq", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 2)))
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--width", type=float, default=19.0,
                        help="Figure width in inches.")
    parser.set_defaults(restrict_to_annotated_span=True)
    return parser


def run(args: argparse.Namespace) -> int:
    data_dir = resolve_dataset_data_dir(args.dataset_dir)
    records = args.records or discover_records(data_dir)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("No LUDB records selected.", file=sys.stderr)
        return 1

    args.leads = list(args.leads)
    args.waves = [wave for wave in WAVES if wave in set(args.waves)]
    args.out_dir = Path(args.out_dir)

    print(
        f"LUDB delineation plots: {len(records)} records, {len(args.leads)} leads, "
        f"waves={'/'.join(args.waves)}, layout={args.layout}, workers={args.workers}",
        flush=True,
    )

    payloads = [(record_id, args, data_dir) for record_id in records]
    written, failures, done = 0, [], 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_task, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                record_id, paths, error = future.result()
                done += 1
                written += len(paths)
                if error:
                    failures.append((record_id, error))
                print(f"  [{done}/{len(records)}] record {record_id}: "
                      f"{len(paths)} page(s){' FAILED ' + error if error else ''}",
                      flush=True)
    else:
        for payload in payloads:
            record_id, paths, error = _task(payload)
            done += 1
            written += len(paths)
            if error:
                failures.append((record_id, error))
            print(f"  [{done}/{len(records)}] record {record_id}: "
                  f"{len(paths)} page(s){' FAILED ' + error if error else ''}", flush=True)

    print(f"\nWrote {written} PNG(s) to {args.out_dir}")
    if failures:
        print(f"{len(failures)} record(s) failed:", file=sys.stderr)
        for record_id, error in failures:
            print(f"  {record_id}: {error}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

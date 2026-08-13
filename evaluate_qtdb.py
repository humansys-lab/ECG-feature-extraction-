#!/usr/bin/env python3
"""Evaluate ecgfeat P/QRS/T delineation on the PhysioNet QT Database (QTDB).

This is the first delineation benchmark in this project that runs on a
database other than LUDB.  QTDB is the field-standard reference set for
waveform boundary algorithms, and it is independent of LUDB in every way that
matters for generalisation:

* different populations (MIT-BIH arrhythmia/ST, ESC ST-T, Holter recordings),
* 250 Hz instead of 500 Hz,
* two Holter-style channels instead of a standard 12-lead acquisition,
* boundaries placed by a different group of cardiologists under a different
  annotation protocol.

Ground truth
------------
``.q1c`` holds the manually determined boundaries of cardiologist 1 for ~30
beats per record; ``.q2c`` holds the second cardiologist's boundaries for the
11 records that were annotated twice.  Both are written on channel 0 only, so
only channel 0 is scored.  The annotation grammar is ``( peak )`` where the
onset and the offset are each optional -- QTDB frequently omits the T onset,
which is why this parser is peak-anchored rather than ``(``-anchored.

Sparse-lead adapter
-------------------
``ECGFeatureExtractor`` accepts only a 12-lead matrix.  QTDB supplies two
channels, so this script reuses the adapter convention already established in
``analyze_physionet_st.py`` for EDB/LTSTDB: real channels are placed in the
ecgfeat QRS detector slots II and V2 and the remaining slots stay zero.  The
slots are adapter positions, not anatomical claims, and no 12-lead axis or
diagnostic interpretation is derived from them.  Because QTDB annotates only
channel 0, results are read back from slot II.

Scoring
-------
Matching, error statistics and aggregation are imported from
``compare_ludb_detectors`` rather than reimplemented, so QTDB numbers are
directly comparable with the existing LUDB numbers.

Only ~30 beats inside a ~30 s window of each 15-minute record are annotated,
so restricting detections to the annotated span is mandatory here, not
optional: without it every unannotated real beat in the record would be
counted as a false positive and precision would measure annotation coverage
instead of detector precision.  ``--restrict-to-annotated-span`` is therefore
on by default and can be turned off only for diagnosis.

Usage
-----
    python evaluate_qtdb.py --n 10 --out qtdb_smoke        # quick check
    python evaluate_qtdb.py --out qtdb_evaluation          # all 105 records
    python evaluate_qtdb.py --annotator q2c --records-with-q2c   # inter-rater
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_EXTRACTION_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTION_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the LUDB harness primitives verbatim so both benchmarks score
# identically.  Only the ground-truth reader and the lead adapter are new.
from compare_ludb_detectors import (  # noqa: E402
    WaveEvent,
    _deduplicate_events,
    _json_safe,
    _package_version,
    aggregate_metrics,
    compare_wave,
    detect_ecgfeat,
    detect_neurokit2,
    write_csv,
)

DEFAULT_QTDB_ROOT = PROJECT_ROOT / "data" / "qtdb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "qtdb_evaluation"

WAVES = ("QRS", "P", "T")
METHODS = ("ecgfeat", "neurokit2")

# Adapter slots, mirroring analyze_physionet_st.py.  QTDB channel 0 carries the
# annotations, so slot II is the scored lead.
SPARSE_ADAPTER_SLOTS = ("II", "V2", "V3")
ANNOTATED_CHANNEL = 0
SCORED_SLOT = SPARSE_ADAPTER_SLOTS[ANNOTATED_CHANNEL]

# QTDB peak symbols.  Anything that is not a P, T or U peak and not a
# parenthesis is a WFDB beat label and therefore anchors a QRS complex.
_P_SYMBOL = "p"
_T_SYMBOL = "t"
_U_SYMBOL = "u"
_NON_QRS_PEAK_SYMBOLS = frozenset({_P_SYMBOL, _T_SYMBOL, _U_SYMBOL})
_BOUNDARY_SYMBOLS = frozenset({"(", ")"})

# CSE Working Party 2-sigma tolerances for boundary location, the limits QTDB
# delineation papers are conventionally judged against.  These bound the
# standard deviation of the error, not its mean.
CSE_2SD_TOLERANCE_MS: Mapping[tuple[str, str], float] = {
    ("P", "onset"): 10.2,
    ("P", "offset"): 12.7,
    ("QRS", "onset"): 6.5,
    ("QRS", "offset"): 11.6,
    ("T", "offset"): 30.6,
}


@dataclass(frozen=True)
class QTDBConfig:
    qtdb_root: str
    annotator: str
    methods: tuple[str, ...]
    r_tolerance_ms: float
    wave_tolerance_ms: float
    window_pad_sec: float
    ecgfeat_r_source: str
    ecgfeat_wave_source: str
    neurokit_peak_method: str
    neurokit_delineate_method: str
    mains_freq: float
    restrict_to_annotated_span: bool
    annotation_cluster_gap_sec: float


@dataclass
class RecordResult:
    detection_rows: list[dict[str, Any]]
    match_rows: list[dict[str, Any]]
    runtime_rows: list[dict[str, Any]]
    failure_rows: list[dict[str, Any]]


# ── Ground truth ─────────────────────────────────────────────────────────────

def parse_qtdb_waves(
    record_path: Path,
    annotator: str,
    *,
    channel: int = ANNOTATED_CHANNEL,
) -> dict[str, list[WaveEvent]]:
    """Read one QTDB manual annotation file into independent P/QRS/T events.

    The parser is anchored on the peak symbol and treats both the onset ``(``
    and the offset ``)`` as optional, because QTDB records routinely annotate a
    T wave as ``t)`` with no onset.  Anchoring on ``(`` instead -- as the LUDB
    reader does, where every wave is fully bracketed -- would silently discard
    those T waves and inflate sensitivity by shrinking the reference set.
    """

    import wfdb

    annotation = wfdb.rdann(str(record_path), annotator)
    symbols = list(annotation.symbol)
    samples = [int(value) for value in annotation.sample]
    channels = [int(value) for value in getattr(annotation, "chan", [])]

    waves: dict[str, list[WaveEvent]] = {wave: [] for wave in WAVES}
    for index, symbol in enumerate(symbols):
        if symbol in _BOUNDARY_SYMBOLS:
            continue
        if channels and channels[index] != channel:
            continue
        if symbol == _U_SYMBOL:
            # ecgfeat exposes no per-beat U-wave bounds, so U is out of scope.
            continue
        if symbol == _P_SYMBOL:
            wave = "P"
        elif symbol == _T_SYMBOL:
            wave = "T"
        else:
            wave = "QRS"

        onset = (
            samples[index - 1]
            if index > 0 and symbols[index - 1] == "("
            else None
        )
        offset = (
            samples[index + 1]
            if index + 1 < len(symbols) and symbols[index + 1] == ")"
            else None
        )
        waves[wave].append(
            WaveEvent(onset=onset, peak=samples[index], offset=offset)
        )

    return {wave: _deduplicate_events(events) for wave, events in waves.items()}


def annotated_clusters(
    gt_events: Sequence[WaveEvent],
    max_gap_samples: int,
) -> list[tuple[int, int]]:
    """Group reference peaks into contiguously annotated regions.

    LUDB annotates one unbroken window per record, so a single ``[min, max]``
    span describes its annotated region exactly.  QTDB does not: ``sel102``
    carries 85 annotated beats spread over 28 clusters separated by gaps of up
    to 62 s, and roughly a quarter of the database is shaped like that.  Inside
    those gaps the record contains ordinary unannotated beats, so a single
    ``[min, max]`` span sweeps them all into the scored region and counts every
    one as a false positive.  That is what drives QTDB precision down to ~20%
    for *every* method, ecgfeat and NeuroKit2 alike -- it measures where the
    cardiologist stopped annotating, not where a detector was wrong.

    Clustering restores the intended meaning: a detection is scored only where
    consecutive reference beats show the annotator was working beat by beat.
    With one cluster this reduces exactly to the single-span behaviour, so LUDB
    scoring is unchanged.
    """

    peaks = sorted(
        int(event.peak) for event in gt_events if event.peak is not None
    )
    if not peaks:
        return []
    clusters: list[tuple[int, int]] = []
    start = previous = peaks[0]
    for peak in peaks[1:]:
        if peak - previous > max_gap_samples:
            clusters.append((start, previous))
            start = peak
        previous = peak
    clusters.append((start, previous))
    return clusters


def restrict_to_annotated_clusters(
    gt_events: Sequence[WaveEvent],
    detected_events: Sequence[WaveEvent],
    tolerance_samples: int,
    max_gap_samples: int,
) -> tuple[list[WaveEvent], int, int]:
    """Drop detections that fall outside every annotated cluster.

    The margin is the match tolerance itself, which keeps the filter exact
    rather than a judgement call: a detection farther than the tolerance from
    every annotated cluster could not have matched any reference event even in
    principle.  Returns ``(kept, dropped, cluster_count)``.
    """

    clusters = annotated_clusters(gt_events, max_gap_samples)
    if not clusters:
        # Nothing annotated for this wave: no detection here is scoreable.
        return [], sum(event.peak is not None for event in detected_events), 0

    windows = [
        (low - tolerance_samples, high + tolerance_samples)
        for low, high in clusters
    ]
    kept = [
        event
        for event in detected_events
        if event.peak is None
        or any(low <= int(event.peak) <= high for low, high in windows)
    ]
    dropped = sum(event.peak is not None for event in detected_events) - sum(
        event.peak is not None for event in kept
    )
    return kept, dropped, len(clusters)


def _shift_events(
    events: Iterable[WaveEvent],
    offset: int,
) -> list[WaveEvent]:
    """Move absolute record samples into window-relative samples."""

    shifted = []
    for event in events:
        shifted.append(
            WaveEvent(
                onset=None if event.onset is None else event.onset - offset,
                peak=None if event.peak is None else event.peak - offset,
                offset=None if event.offset is None else event.offset - offset,
            )
        )
    return shifted


# ── Record discovery ─────────────────────────────────────────────────────────

def discover_records(qtdb_root: Path, annotator: str) -> list[str]:
    records_file = qtdb_root / "RECORDS"
    if records_file.exists():
        candidates = [
            line.strip()
            for line in records_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        candidates = sorted(path.stem for path in qtdb_root.glob("*.hea"))
    return [
        record
        for record in candidates
        if (qtdb_root / f"{record}.{annotator}").exists()
    ]


# ── Signal windowing and the sparse-lead adapter ─────────────────────────────

def _fill_nonfinite(channel: np.ndarray) -> tuple[np.ndarray, float]:
    """Linearly interpolate non-finite samples; report how many there were."""

    values = np.asarray(channel, dtype=float).copy()
    bad = ~np.isfinite(values)
    if not bad.any():
        return values, 0.0
    if bad.all():
        return np.zeros_like(values), 1.0
    good_index = np.flatnonzero(~bad)
    values[bad] = np.interp(np.flatnonzero(bad), good_index, values[good_index])
    return values, float(bad.mean())


def load_annotated_window(
    record_path: Path,
    gt_waves: Mapping[str, Sequence[WaveEvent]],
    *,
    window_pad_sec: float,
) -> tuple[np.ndarray, int, int, list[str], float]:
    """Read the annotated window and lay it out as a sparse 12-lead matrix.

    Returns ``(ecg_12lead, fs, window_start, signal_names, missing_fraction)``.
    """

    import wfdb

    header = wfdb.rdheader(str(record_path))
    fs = int(round(float(header.fs)))
    marks = [
        value
        for events in gt_waves.values()
        for event in events
        for value in (event.onset, event.peak, event.offset)
        if value is not None
    ]
    if not marks:
        raise ValueError(f"no usable annotations in {record_path.name}")

    pad = int(round(window_pad_sec * fs))
    start = max(0, min(marks) - pad)
    end = min(int(header.sig_len), max(marks) + pad)
    if end - start < max(int(round(4.0 * fs)), 32):
        raise ValueError(
            f"annotated window too short for {record_path.name}: "
            f"{end - start} samples"
        )

    record = wfdb.rdrecord(str(record_path), sampfrom=start, sampto=end)
    physical = np.asarray(record.p_signal, dtype=float)
    if physical.ndim != 2 or not 1 <= physical.shape[1] <= len(SPARSE_ADAPTER_SLOTS):
        raise ValueError(
            f"expected 1-{len(SPARSE_ADAPTER_SLOTS)} channels, got shape "
            f"{physical.shape} for {record_path.name}"
        )

    from ecgfeat.models import STANDARD_12_LEADS

    ecg_12lead = np.zeros((12, physical.shape[0]), dtype=float)
    missing_fractions = []
    for source_index in range(physical.shape[1]):
        clean, missing_fraction = _fill_nonfinite(physical[:, source_index])
        slot = SPARSE_ADAPTER_SLOTS[source_index]
        ecg_12lead[STANDARD_12_LEADS.index(slot)] = clean
        missing_fractions.append(missing_fraction)

    signal_names = [str(name) for name in record.sig_name]
    return (
        ecg_12lead,
        fs,
        start,
        signal_names,
        float(missing_fractions[ANNOTATED_CHANNEL]),
    )


# ── Per-record evaluation ────────────────────────────────────────────────────

def _failure_row(
    record_id: str,
    method: str,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "record": record_id,
        "lead": SCORED_SLOT,
        "method": method,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _failed_rows(
    record_id: str,
    method: str,
    gt_waves: Mapping[str, Sequence[WaveEvent]],
    config: QTDBConfig,
    exc: BaseException,
) -> list[dict[str, Any]]:
    rows = []
    for wave in WAVES:
        tolerance = (
            config.r_tolerance_ms if wave == "QRS" else config.wave_tolerance_ms
        )
        detection, _ = compare_wave(
            record_id=record_id,
            lead=SCORED_SLOT,
            method=method,
            wave=wave,
            gt_events=gt_waves.get(wave, []),
            detected_events=[],
            fs=250,
            tolerance_ms=tolerance,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        rows.append(detection)
    return rows


def evaluate_record(record_id: str, config: QTDBConfig) -> RecordResult:
    qtdb_root = Path(config.qtdb_root)
    record_path = qtdb_root / record_id

    detection_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    try:
        gt_absolute = parse_qtdb_waves(record_path, config.annotator)
    except Exception as exc:  # noqa: BLE001 - one bad record must not stop the run
        failure_rows.append(_failure_row(record_id, "-", "read_annotation", exc))
        return RecordResult([], [], [], failure_rows)

    try:
        ecg_12lead, fs, window_start, signal_names, missing_fraction = (
            load_annotated_window(
                record_path,
                gt_absolute,
                window_pad_sec=config.window_pad_sec,
            )
        )
    except Exception as exc:  # noqa: BLE001
        failure_rows.append(_failure_row(record_id, "-", "read_signal", exc))
        return RecordResult([], [], [], failure_rows)

    gt_waves = {
        wave: _shift_events(events, window_start)
        for wave, events in gt_absolute.items()
    }
    annotated_channel_name = (
        signal_names[ANNOTATED_CHANNEL]
        if ANNOTATED_CHANNEL < len(signal_names)
        else "?"
    )

    detections: dict[str, dict[str, list[WaveEvent]]] = {}

    if "ecgfeat" in config.methods:
        try:
            with warnings.catch_warnings():
                # The zero-padded adapter slots make some correlation terms
                # degenerate; that is expected for sparse-lead input.
                warnings.simplefilter("ignore")
                per_lead, elapsed = detect_ecgfeat(
                    ecg_12lead,
                    fs,
                    (SCORED_SLOT,),
                    config.mains_freq,
                    qrs_peak_source=config.ecgfeat_r_source,
                    wave_peak_source=config.ecgfeat_wave_source,
                )
            detections["ecgfeat"] = per_lead[SCORED_SLOT]
            runtime_rows.append(
                {
                    "record": record_id,
                    "method": "ecgfeat",
                    "seconds": elapsed,
                    "signal_seconds": ecg_12lead.shape[1] / fs,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(_failure_row(record_id, "ecgfeat", "detect", exc))
            detection_rows.extend(
                _failed_rows(record_id, "ecgfeat", gt_waves, config, exc)
            )

    if "neurokit2" in config.methods:
        try:
            from ecgfeat.models import STANDARD_12_LEADS

            channel = ecg_12lead[STANDARD_12_LEADS.index(SCORED_SLOT)]
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detections["neurokit2"] = detect_neurokit2(
                    channel,
                    fs,
                    config.neurokit_peak_method,
                    config.neurokit_delineate_method,
                )
            runtime_rows.append(
                {
                    "record": record_id,
                    "method": "neurokit2",
                    "seconds": time.perf_counter() - started,
                    "signal_seconds": ecg_12lead.shape[1] / fs,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(_failure_row(record_id, "neurokit2", "detect", exc))
            detection_rows.extend(
                _failed_rows(record_id, "neurokit2", gt_waves, config, exc)
            )

    max_gap_samples = int(round(config.annotation_cluster_gap_sec * fs))
    for method, per_wave in detections.items():
        for wave in WAVES:
            tolerance = (
                config.r_tolerance_ms if wave == "QRS" else config.wave_tolerance_ms
            )
            gt_events = gt_waves.get(wave, [])
            detected_events = per_wave.get(wave, [])
            # Cluster-aware restriction happens here rather than inside
            # compare_wave, so the LUDB harness keeps its validated behaviour.
            outside_span = 0
            n_clusters = 0
            if config.restrict_to_annotated_span:
                tolerance_samples = max(1, int(round(tolerance * fs / 1000.0)))
                detected_events, outside_span, n_clusters = (
                    restrict_to_annotated_clusters(
                        gt_events,
                        detected_events,
                        tolerance_samples,
                        max_gap_samples,
                    )
                )
            detection, matches = compare_wave(
                record_id=record_id,
                lead=SCORED_SLOT,
                method=method,
                wave=wave,
                gt_events=gt_events,
                detected_events=detected_events,
                fs=fs,
                tolerance_ms=tolerance,
                restrict_span=False,
            )
            detection.update(
                {
                    "annotator": config.annotator,
                    "source_channel": annotated_channel_name,
                    "fs": fs,
                    "window_start": window_start,
                    "window_samples": int(ecg_12lead.shape[1]),
                    "input_missing_fraction": missing_fraction,
                    "det_outside_annotated_span": outside_span,
                    "annotated_clusters": n_clusters,
                }
            )
            detection_rows.append(detection)
            for row in matches:
                row.update(
                    {
                        "annotator": config.annotator,
                        "source_channel": annotated_channel_name,
                    }
                )
            match_rows.extend(matches)

    return RecordResult(detection_rows, match_rows, runtime_rows, failure_rows)


def _evaluate_record_task(payload: tuple[str, QTDBConfig]) -> RecordResult:
    record_id, config = payload
    return evaluate_record(record_id, config)


# ── Reporting ────────────────────────────────────────────────────────────────

def cse_conformance(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare each boundary's error SD with the CSE 2-sigma tolerance."""

    rows = []
    by_key = {
        (str(row["method"]), str(row["wave"])): row for row in summary_rows
    }
    for (wave, boundary), limit in CSE_2SD_TOLERANCE_MS.items():
        for method in sorted({str(row["method"]) for row in summary_rows}):
            row = by_key.get((method, wave))
            if row is None:
                continue
            sd = row.get(f"{boundary}_sd_ms")
            bias = row.get(f"{boundary}_bias_ms")
            count = row.get(f"{boundary}_n")
            rows.append(
                {
                    "method": method,
                    "wave": wave,
                    "boundary": boundary,
                    "n": count,
                    "bias_ms": bias,
                    "sd_ms": sd,
                    "cse_2sd_limit_ms": limit,
                    "within_cse_limit": (
                        None if sd is None else bool(float(sd) <= limit)
                    ),
                }
            )
    return rows


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def print_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    cse_rows: Sequence[Mapping[str, Any]],
    n_records: int,
    annotator: str,
) -> None:
    line = "─" * 96
    print()
    print("=" * 96)
    print(
        f"  QTDB DELINEATION  —  {n_records} records  /  annotator {annotator}  "
        f"/  channel 0 via adapter slot {SCORED_SLOT}"
    )
    print("=" * 96)
    print()
    header = (
        f"  {'method':<10s} {'wave':<4s} {'GT':>5s} {'TP':>5s} {'FP':>4s} "
        f"{'FN':>4s} {'Se%':>6s} {'PPV%':>6s} "
        f"{'on bias±SD':>16s} {'off bias±SD':>16s}"
    )
    print(header)
    print(f"  {line}")
    for row in sorted(
        summary_rows, key=lambda item: (str(item["method"]), str(item["wave"]))
    ):
        sensitivity = row.get("sensitivity")
        precision = row.get("precision")
        print(
            f"  {str(row['method']):<10s} {str(row['wave']):<4s} "
            f"{int(row['gt_count']):>5d} {int(row['tp']):>5d} "
            f"{int(row['fp']):>4d} {int(row['fn']):>4d} "
            f"{_fmt(None if sensitivity is None else sensitivity * 100, 1):>6s} "
            f"{_fmt(None if precision is None else precision * 100, 1):>6s} "
            f"{_fmt(row.get('onset_bias_ms'), 1) + '±' + _fmt(row.get('onset_sd_ms'), 1):>16s} "
            f"{_fmt(row.get('offset_bias_ms'), 1) + '±' + _fmt(row.get('offset_sd_ms'), 1):>16s}"
        )
    print()
    print("  CSE 2-sigma tolerance conformance (SD of boundary error):")
    print(f"  {line}")
    print(
        f"  {'method':<10s} {'boundary':<14s} {'n':>6s} {'bias':>8s} "
        f"{'SD':>8s} {'limit':>8s}  {'within':<6s}"
    )
    for row in sorted(
        cse_rows, key=lambda item: (str(item["method"]), str(item["wave"]))
    ):
        print(
            f"  {str(row['method']):<10s} "
            f"{str(row['wave']) + ' ' + str(row['boundary']):<14s} "
            f"{_fmt(row.get('n'), 0):>6s} {_fmt(row.get('bias_ms'), 1):>8s} "
            f"{_fmt(row.get('sd_ms'), 1):>8s} "
            f"{_fmt(row.get('cse_2sd_limit_ms'), 1):>8s}  "
            f"{_fmt(row.get('within_cse_limit')):<6s}"
        )
    print()
    print("  Errors in ms, sign = detected − reference (positive: detected later).")
    print("=" * 96)


def write_report(
    path: Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    cse_rows: Sequence[Mapping[str, Any]],
    per_record_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    n_records = int(metadata["records_evaluated"])
    annotator = str(metadata["annotator"])
    lines: list[str] = []
    lines.append("# ecgfeat 在 QTDB 上的 P/QRS/T 描记性能评估")
    lines.append("")
    lines.append(
        f"数据集 QT Database（QTDB），{n_records} 条记录，标注者 `{annotator}`，"
        f"采样率 250 Hz，仅通道 0 有人工边界标注。"
    )
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append(
        "- **稀疏导联适配**：QTDB 只有 2 个通道，而 `ECGFeatureExtractor` 只接受 12 导联矩阵。"
        f"沿用本仓库 `analyze_physionet_st.py` 对 EDB/LTSTDB 已确立的约定，把真实通道放进 "
        f"ecgfeat 的 QRS 检测槽位 {', '.join(SPARSE_ADAPTER_SLOTS[:2])}，其余槽位置零。"
        f"槽位只是适配位置，不代表解剖导联；标注在通道 0，因此评分读取槽位 {SCORED_SLOT}。"
    )
    lines.append(
        "- **参考真值**：`.q1c`/`.q2c` 人工边界。解析以峰值符号为锚点，`(` 与 `)` 均视为可选，"
        "因为 QTDB 常把 T 波标成 `t)`（无起点）。若像 LUDB 那样以 `(` 为锚，会静默丢弃这些 T 波。"
    )
    lines.append(
        "- **评分口径**：匹配、误差统计与聚合直接复用 `compare_ludb_detectors` 的函数，"
        "因此 QTDB 与既有 LUDB 结果可直接对比。"
    )
    lines.append(
        "- **限制在标注区间内（按簇）**：每条 15 分钟记录只有约 30 拍被标注，"
        "若不限制，其余真实心拍都会被记成假阳性，PPV 衡量的就变成标注覆盖率而非检测精度。"
        "此外 QTDB 与 LUDB 不同：约四分之一的记录标注被切成多个不连续的簇"
        "（如 `sel102` 的 85 拍散布在 28 个簇中，簇间最大间隔 62 s），"
        "因此不能用单一 `[min, max]` 区间，否则簇间的正常心拍会被全部计成假阳性。"
        f"本评估按参考心拍聚簇（间隔 > {metadata['annotation_cluster_gap_sec']} s 即断簇），"
        "只在簇内评分；当只有一个簇时该规则与 LUDB 的单区间做法完全等价。"
    )
    lines.append(
        f"- **匹配容差**：QRS {metadata['r_tolerance_ms']} ms，P/T "
        f"{metadata['wave_tolerance_ms']} ms（与 LUDB 基准一致）。"
    )
    lines.append("")
    lines.append("## 检测与描记总表")
    lines.append("")
    lines.append(
        "| 方法 | 波 | 参考数 | TP | FP | FN | 灵敏度 % | PPV % | "
        "起点 bias±SD (ms) | 峰 bias±SD (ms) | 终点 bias±SD (ms) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(
        summary_rows, key=lambda item: (str(item["method"]), str(item["wave"]))
    ):
        sensitivity = row.get("sensitivity")
        precision = row.get("precision")
        lines.append(
            f"| {row['method']} | {row['wave']} | {int(row['gt_count'])} | "
            f"{int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} | "
            f"{_fmt(None if sensitivity is None else sensitivity * 100, 1)} | "
            f"{_fmt(None if precision is None else precision * 100, 1)} | "
            f"{_fmt(row.get('onset_bias_ms'), 1)}±{_fmt(row.get('onset_sd_ms'), 1)} | "
            f"{_fmt(row.get('peak_bias_ms'), 1)}±{_fmt(row.get('peak_sd_ms'), 1)} | "
            f"{_fmt(row.get('offset_bias_ms'), 1)}±{_fmt(row.get('offset_sd_ms'), 1)} |"
        )
    lines.append("")
    lines.append("## CSE 2σ 容差符合性")
    lines.append("")
    lines.append(
        "CSE Working Party 的 2σ 容差限制的是边界误差的**标准差**，是 QTDB 描记类工作"
        "惯用的判据。这里是研究性对照，不是正式 CSE/IEC 符合性结论。"
    )
    lines.append("")
    lines.append("| 方法 | 边界 | n | bias (ms) | SD (ms) | CSE 2σ 限 (ms) | 是否达标 |")
    lines.append("|---|---|---:|---:|---:|---:|:--:|")
    for row in sorted(
        cse_rows, key=lambda item: (str(item["method"]), str(item["wave"]))
    ):
        lines.append(
            f"| {row['method']} | {row['wave']} {row['boundary']} | "
            f"{_fmt(row.get('n'), 0)} | {_fmt(row.get('bias_ms'), 1)} | "
            f"{_fmt(row.get('sd_ms'), 1)} | {_fmt(row.get('cse_2sd_limit_ms'), 1)} | "
            f"{'✅' if row.get('within_cse_limit') else '❌'} |"
        )
    lines.append("")

    if runtime_rows:
        by_method: dict[str, list[float]] = defaultdict(list)
        signal_seconds: dict[str, list[float]] = defaultdict(list)
        for row in runtime_rows:
            by_method[str(row["method"])].append(float(row["seconds"]))
            signal_seconds[str(row["method"])].append(float(row["signal_seconds"]))
        lines.append("## 运行时间")
        lines.append("")
        lines.append("| 方法 | 记录数 | 每条中位耗时 (s) | 实时倍率 (信号秒/耗时秒) |")
        lines.append("|---|---:|---:|---:|")
        for method in sorted(by_method):
            seconds = np.asarray(by_method[method], dtype=float)
            signal = np.asarray(signal_seconds[method], dtype=float)
            lines.append(
                f"| {method} | {seconds.size} | {np.median(seconds):.2f} | "
                f"{np.sum(signal) / max(np.sum(seconds), 1e-9):.1f}× |"
            )
        lines.append("")

    worst = [
        row
        for row in per_record_rows
        if row.get("method") == "ecgfeat"
        and row.get("sensitivity") is not None
    ]
    worst.sort(key=lambda row: (float(row["sensitivity"]), str(row["record"])))
    if worst:
        lines.append("## ecgfeat 灵敏度最低的 15 条记录×波")
        lines.append("")
        lines.append("| 记录 | 通道 | 波 | 参考数 | TP | FP | FN | 灵敏度 % | PPV % |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in worst[:15]:
            precision = row.get("precision")
            lines.append(
                f"| {row['record']} | {row.get('source_channel', '-')} | "
                f"{row['wave']} | {int(row['gt_count'])} | {int(row['tp'])} | "
                f"{int(row['fp'])} | {int(row['fn'])} | "
                f"{float(row['sensitivity']) * 100:.1f} | "
                f"{_fmt(None if precision is None else precision * 100, 1)} |"
            )
        lines.append("")

    lines.append("## 说明与边界条件")
    lines.append("")
    lines.append(
        "- 误差符号为 `检测 − 参考`，正值表示检测偏晚。"
    )
    lines.append(
        "- QTDB 大量记录为动态心电（Holter）导联（MLII、ECG1/ECG2、D3/D4、CM5 等），"
        "与 12 导联标准体系不同；本评估只做边界描记，不产生任何 12 导联电轴或诊断结论。"
    )
    lines.append(
        "- U 波在 QTDB 中有标注，但 ecgfeat 的逐拍特征未导出 U 波边界，因此不在评分范围内。"
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ecgfeat delineation on the QT Database.",
    )
    parser.add_argument("--qtdb-root", type=Path, default=DEFAULT_QTDB_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--annotator",
        default="q1c",
        help="manual annotation extension: q1c (cardiologist 1) or q2c",
    )
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--n", type=int, default=None, help="evaluate first N records")
    parser.add_argument(
        "--methods",
        default="all",
        help=f"comma-separated subset of {METHODS} or 'all'",
    )
    parser.add_argument("--r-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--wave-tolerance-ms", type=float, default=150.0)
    parser.add_argument(
        "--window-pad-sec",
        type=float,
        default=5.0,
        help="signal context read on each side of the annotated span",
    )
    parser.add_argument("--ecgfeat-r-source", default="global")
    parser.add_argument("--ecgfeat-wave-source", default="native")
    parser.add_argument("--neurokit-peak-method", default="neurokit")
    parser.add_argument("--neurokit-delineate-method", default="dwt")
    parser.add_argument(
        "--mains-freq",
        type=float,
        default=60.0,
        help="QTDB is largely North-American; ecgfeat still auto-resolves mains",
    )
    parser.add_argument(
        "--annotation-cluster-gap-sec",
        type=float,
        default=3.0,
        help=(
            "reference beats separated by more than this start a new annotated "
            "cluster; the gap between clusters is excluded from scoring"
        ),
    )
    parser.add_argument(
        "--no-restrict-to-annotated-span",
        action="store_true",
        help=(
            "diagnostic only: count detections outside the ~30 s annotated "
            "window as false positives"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser


def _parse_methods(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return METHODS
    chosen = tuple(
        item.strip() for item in raw.split(",") if item.strip()
    )
    invalid = [item for item in chosen if item not in METHODS]
    if invalid:
        raise SystemExit(f"unsupported methods: {invalid}; allowed: {list(METHODS)}")
    return chosen


def run(args: argparse.Namespace) -> int:
    qtdb_root = args.qtdb_root.resolve()
    if not qtdb_root.exists():
        raise SystemExit(f"QTDB root not found: {qtdb_root}")

    methods = _parse_methods(args.methods)
    all_records = discover_records(qtdb_root, args.annotator)
    if not all_records:
        raise SystemExit(
            f"no records with .{args.annotator} annotations under {qtdb_root}"
        )
    records = list(all_records)
    if args.records:
        unknown = [item for item in args.records if item not in set(all_records)]
        if unknown:
            raise SystemExit(f"unknown records: {unknown}")
        records = list(args.records)
    if args.n is not None:
        records = records[: max(0, args.n)]

    config = QTDBConfig(
        qtdb_root=str(qtdb_root),
        annotator=args.annotator,
        methods=methods,
        r_tolerance_ms=float(args.r_tolerance_ms),
        wave_tolerance_ms=float(args.wave_tolerance_ms),
        window_pad_sec=float(args.window_pad_sec),
        ecgfeat_r_source=str(args.ecgfeat_r_source),
        ecgfeat_wave_source=str(args.ecgfeat_wave_source),
        neurokit_peak_method=str(args.neurokit_peak_method),
        neurokit_delineate_method=str(args.neurokit_delineate_method),
        mains_freq=float(args.mains_freq),
        restrict_to_annotated_span=not bool(args.no_restrict_to_annotated_span),
        annotation_cluster_gap_sec=float(args.annotation_cluster_gap_sec),
    )

    print(
        f"QTDB evaluation: {len(records)} records, methods={list(methods)}, "
        f"annotator={config.annotator}, workers={args.workers}"
    )

    detection_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    if args.workers > 1:
        payloads = [(record, config) for record in records]
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            for index, result in enumerate(
                pool.map(_evaluate_record_task, payloads), start=1
            ):
                detection_rows.extend(result.detection_rows)
                match_rows.extend(result.match_rows)
                runtime_rows.extend(result.runtime_rows)
                failure_rows.extend(result.failure_rows)
                print(f"  [{index}/{len(records)}] done", flush=True)
    else:
        for index, record in enumerate(records, start=1):
            result = evaluate_record(record, config)
            detection_rows.extend(result.detection_rows)
            match_rows.extend(result.match_rows)
            runtime_rows.extend(result.runtime_rows)
            failure_rows.extend(result.failure_rows)
            print(f"  [{index}/{len(records)}] {record}", flush=True)
    elapsed = time.perf_counter() - started

    summary_rows = aggregate_metrics(
        detection_rows, match_rows, ("method", "wave")
    )
    per_record_rows = aggregate_metrics(
        detection_rows, match_rows, ("method", "wave", "record")
    )
    for row in per_record_rows:
        source = next(
            (
                item.get("source_channel")
                for item in detection_rows
                if item["record"] == row["record"]
            ),
            None,
        )
        row["source_channel"] = source
    cse_rows = cse_conformance(summary_rows)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "detection_by_record.csv", detection_rows)
    write_csv(out_dir / "matched_events.csv", match_rows)
    write_csv(out_dir / "summary_by_method_wave.csv", summary_rows)
    write_csv(out_dir / "summary_by_record.csv", per_record_rows)
    write_csv(out_dir / "cse_conformance.csv", cse_rows)
    if failure_rows:
        write_csv(out_dir / "failures.csv", failure_rows)

    metadata = {
        "dataset": "QT Database (qtdb)",
        "qtdb_root": str(qtdb_root),
        "annotator": config.annotator,
        "records_evaluated": len(records),
        "records_available": len(all_records),
        "methods": list(methods),
        "scored_channel": ANNOTATED_CHANNEL,
        "adapter": (
            "real channels in slots "
            f"{'/'.join(SPARSE_ADAPTER_SLOTS[:2])}, remaining slots zero"
        ),
        "scored_slot": SCORED_SLOT,
        "r_tolerance_ms": config.r_tolerance_ms,
        "wave_tolerance_ms": config.wave_tolerance_ms,
        "window_pad_sec": config.window_pad_sec,
        "restrict_to_annotated_span": config.restrict_to_annotated_span,
        "annotation_cluster_gap_sec": config.annotation_cluster_gap_sec,
        "ecgfeat_r_source": config.ecgfeat_r_source,
        "ecgfeat_wave_source": config.ecgfeat_wave_source,
        "neurokit_peak_method": config.neurokit_peak_method,
        "neurokit_delineate_method": config.neurokit_delineate_method,
        "mains_freq": config.mains_freq,
        "wall_clock_seconds": elapsed,
        "failures": len(failure_rows),
        "package_versions": {
            name: _package_version(name)
            for name in ("wfdb", "neurokit2", "numpy", "scipy")
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "metadata": metadata,
                    "summary": summary_rows,
                    "cse_conformance": cse_rows,
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(
        out_dir / "report.md",
        summary_rows=summary_rows,
        cse_rows=cse_rows,
        per_record_rows=per_record_rows,
        runtime_rows=runtime_rows,
        metadata=metadata,
    )

    print_summary(summary_rows, cse_rows, len(records), config.annotator)
    if failure_rows:
        print(f"  {len(failure_rows)} failures written to failures.csv")
    print(f"  wall clock: {elapsed:.1f}s   outputs: {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

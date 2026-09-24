#!/usr/bin/env python3
"""Compare ECG delineation methods against LUDB expert annotations.

The comparison intentionally separates two capabilities:

* R-peak detection: ecgfeat, NeuroKit2, and BioSPPy.
* P/QRS/T delineation: ecgfeat and NeuroKit2. BioSPPy's public ``ecg``
  pipeline does not expose P/T or QRS-boundary delineation.

LUDB annotations are lead-specific. NeuroKit2 and BioSPPy are therefore run
independently on every selected lead. ecgfeat keeps its native multilead QRS
detection and supplies lead-specific boundaries from its beat features.

Examples
--------
Smoke test on record 1, lead II:

    python compare_ludb_detectors.py --records 1 --leads II

First 10 records, all leads:

    python compare_ludb_detectors.py --limit 10 --workers 2

Full LUDB:

    python compare_ludb_detectors.py --workers 4 \
        --out-dir ludb_three_way_comparison
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import math
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_EXTRACTION_ROOT = PROJECT_ROOT / "feature_extraction"
DEFAULT_LUDB_ROOT = (
    PROJECT_ROOT
    / "data"
    / "lobachevsky-university-electrocardiography-database-1.0.1"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ludb_three_way_comparison"

STANDARD_12_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
LEAD_TO_ANN_EXT = {lead: lead.lower() for lead in STANDARD_12_LEADS}
WFDB_NAME_TO_LEAD = {lead.lower(): lead for lead in STANDARD_12_LEADS}
METHODS = ("ecgfeat", "neurokit2", "biosppy")
DELINEATION_METHODS = frozenset({"ecgfeat", "neurokit2"})
WAVES = ("QRS", "P", "T")


@dataclass(frozen=True)
class WaveEvent:
    """One waveform with sample-index boundaries and a peak/fiducial."""

    onset: int | None
    peak: int | None
    offset: int | None


@dataclass(frozen=True)
class CompareConfig:
    dataset_data_dir: str
    leads: tuple[str, ...]
    methods: tuple[str, ...]
    r_tolerance_ms: float
    wave_tolerance_ms: float
    ecgfeat_r_source: str
    ecgfeat_wave_source: str
    neurokit_peak_method: str
    neurokit_delineate_method: str
    mains_freq: float
    fail_fast: bool
    restrict_to_annotated_span: bool = False


@dataclass
class RecordResult:
    detection_rows: list[dict[str, Any]]
    match_rows: list[dict[str, Any]]
    runtime_rows: list[dict[str, Any]]
    failure_rows: list[dict[str, Any]]


def _ensure_feature_extraction_path() -> None:
    path = str(FEATURE_EXTRACTION_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _record_sort_key(record_id: str) -> tuple[int, int | str]:
    return (0, int(record_id)) if record_id.isdigit() else (1, record_id)


def resolve_dataset_data_dir(dataset_dir: Path) -> Path:
    """Accept either the LUDB root or its nested ``data`` directory."""

    dataset_dir = dataset_dir.expanduser().resolve()
    candidates = (dataset_dir, dataset_dir / "data")
    for candidate in candidates:
        if any(candidate.glob("*.hea")):
            return candidate
    raise FileNotFoundError(
        "No LUDB .hea records found. Expected files such as data/1.hea and "
        f"data/1.dat under: {dataset_dir}"
    )


def discover_records(data_dir: Path) -> list[str]:
    records = sorted(
        {path.stem for path in data_dir.glob("*.hea")},
        key=_record_sort_key,
    )
    if not records:
        raise FileNotFoundError(f"No .hea records found in {data_dir}")
    return records


def _finite_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric):
        return None
    return int(round(numeric))


def _at(values: Sequence[Any] | None, index: int) -> int | None:
    if values is None or index >= len(values):
        return None
    return _finite_index(values[index])


def _event_quality(event: WaveEvent) -> tuple[int, int]:
    return (
        int(event.onset is not None) + int(event.offset is not None),
        -abs((event.offset or event.peak or 0) - (event.onset or event.peak or 0)),
    )


def _deduplicate_events(events: Iterable[WaveEvent]) -> list[WaveEvent]:
    """Keep the most complete event when two detections share one anchor."""

    by_peak: dict[int, WaveEvent] = {}
    for event in events:
        if event.peak is None:
            continue
        previous = by_peak.get(event.peak)
        if previous is None or _event_quality(event) > _event_quality(previous):
            by_peak[event.peak] = event
    return [by_peak[peak] for peak in sorted(by_peak)]


def parse_ludb_waves(record_path: Path, annotator: str) -> dict[str, list[WaveEvent]]:
    """Read one LUDB lead annotation into independent P/QRS/T events."""

    import wfdb

    annotation = wfdb.rdann(str(record_path), annotator)
    symbols = list(annotation.symbol)
    samples = list(annotation.sample)
    waves: dict[str, list[WaveEvent]] = {wave: [] for wave in WAVES}
    symbol_to_wave = {"N": "QRS", "p": "P", "t": "T"}

    index = 0
    while index < len(symbols):
        if symbols[index] != "(" or index + 1 >= len(symbols):
            index += 1
            continue
        peak_symbol = symbols[index + 1]
        wave = symbol_to_wave.get(peak_symbol)
        if wave is None:
            index += 1
            continue
        offset = (
            int(samples[index + 2])
            if index + 2 < len(symbols) and symbols[index + 2] == ")"
            else None
        )
        waves[wave].append(
            WaveEvent(
                onset=int(samples[index]),
                peak=int(samples[index + 1]),
                offset=offset,
            )
        )
        index += 3 if offset is not None else 2

    return waves


def _lead_indices(signal_names: Sequence[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, name in enumerate(signal_names):
        normalized = str(name).strip().lower()
        lead = WFDB_NAME_TO_LEAD.get(normalized)
        if lead is not None:
            indices[lead] = index
    missing = [lead for lead in STANDARD_12_LEADS if lead not in indices]
    if missing:
        raise ValueError(f"WFDB record is missing standard leads: {missing}")
    return indices


def _select_ecgfeat_qrs_peak(
    beat_feature: Any,
    global_r: int,
    source: str,
) -> int:
    """Choose the plotted/evaluated QRS marker without changing beat count."""

    if source == "lead-fiducial":
        selected = _finite_index(beat_feature.qrs.peak)
    elif source == "positive-r":
        selected = _finite_index(getattr(beat_feature, "r_peak_index", None))
        if selected is None:
            selected = _finite_index(beat_feature.qrs.peak)
    elif source == "hybrid-prominence":
        selected = _finite_index(
            getattr(beat_feature, "r_localized_index", None)
        )
        if selected is None:
            selected = _finite_index(beat_feature.qrs.peak)
    elif source == "global":
        selected = int(global_r)
    else:
        raise ValueError(f"unsupported ecgfeat QRS peak source: {source!r}")
    return int(global_r) if selected is None else int(selected)


def _select_ecgfeat_wave_event(
    beat_feature: Any,
    wave: str,
    source: str,
) -> WaveEvent:
    native = getattr(beat_feature, wave)
    onset = _finite_index(native.onset)
    peak = _finite_index(native.peak)
    offset = _finite_index(native.offset)
    if source == "native":
        return WaveEvent(onset, peak, offset)
    if source != "hybrid-peaks":
        raise ValueError(f"unsupported ecgfeat wave source: {source!r}")
    localized_peak = _finite_index(
        getattr(beat_feature, f"{wave}_localized_index", None)
    )
    # Native ecgfeat P timing is already highly accurate on LUDB. The bipolar
    # branch remains a morphology side annotation; only T timing is replaced in
    # the currently validated evaluation mode.
    if wave == "p":
        localized_peak = peak
    return WaveEvent(onset, localized_peak, offset)


def detect_ecgfeat(
    ecg_12lead: np.ndarray,
    fs: int,
    leads: Sequence[str],
    mains_freq: float,
    qrs_peak_source: str = "global",
    wave_peak_source: str = "native",
) -> tuple[dict[str, dict[str, list[WaveEvent]]], float]:
    """Run the project's native multilead detector once for the record.

    ``qrs_peak_source`` controls only the QRS marker placed into each lead's
    ``WaveEvent``:

    - ``global`` keeps the multilead beat fiducial used by the evaluator;
    - ``lead-fiducial`` uses that lead's locally refined ``qrs.peak``;
    - ``positive-r`` uses the maximum positive R deflection in that lead.

    Beat presence/count remains determined by the multilead detector in all
    three modes.
    """

    valid_qrs_peak_sources = {
        "global",
        "lead-fiducial",
        "positive-r",
        "hybrid-prominence",
    }
    if qrs_peak_source not in valid_qrs_peak_sources:
        raise ValueError(
            "qrs_peak_source must be one of "
            f"{sorted(valid_qrs_peak_sources)}, got {qrs_peak_source!r}"
        )
    if wave_peak_source not in {"native", "hybrid-peaks"}:
        raise ValueError(
            "wave_peak_source must be one of ['hybrid-peaks', 'native'], "
            f"got {wave_peak_source!r}"
        )

    _ensure_feature_extraction_path()
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    started = time.perf_counter()
    features = ECGFeatureExtractor(
        fs_internal=int(fs),
        mains_freq=float(mains_freq),
    ).extract(ecg_12lead, fs=float(fs))
    elapsed = time.perf_counter() - started

    global_r = sorted(
        {
            int(beat.r_index)
            for beat in features.beats
            if getattr(beat, "r_index", None) is not None
        }
    )
    beat_features_by_lead: dict[str, list[Any]] = {lead: [] for lead in leads}
    for beat_feature in features.beat_features:
        if beat_feature.lead in beat_features_by_lead:
            beat_features_by_lead[beat_feature.lead].append(beat_feature)

    result: dict[str, dict[str, list[WaveEvent]]] = {}
    association_tolerance = max(1, int(round(0.020 * fs)))
    for lead in leads:
        lead_features = beat_features_by_lead[lead]
        qrs_events: list[WaveEvent] = []
        used_features: set[int] = set()
        for r_peak in global_r:
            candidates = [
                (abs(int(item.qrs.peak) - r_peak), index, item)
                for index, item in enumerate(lead_features)
                if index not in used_features and item.qrs.peak is not None
            ]
            if candidates:
                distance, feature_index, item = min(candidates, key=lambda row: row[0])
            else:
                distance, feature_index, item = association_tolerance + 1, -1, None
            if item is not None and distance <= association_tolerance:
                used_features.add(feature_index)
                event_peak = _select_ecgfeat_qrs_peak(
                    item,
                    r_peak,
                    qrs_peak_source,
                )
                qrs_events.append(
                    WaveEvent(
                        onset=_finite_index(item.qrs.onset),
                        peak=event_peak,
                        offset=_finite_index(item.qrs.offset),
                    )
                )
            else:
                qrs_events.append(WaveEvent(None, r_peak, None))

        p_events = _deduplicate_events(
            _select_ecgfeat_wave_event(item, "p", wave_peak_source)
            for item in lead_features
        )
        t_events = _deduplicate_events(
            _select_ecgfeat_wave_event(item, "t", wave_peak_source)
            for item in lead_features
        )
        result[lead] = {
            "QRS": qrs_events,
            "P": p_events,
            "T": t_events,
        }
    return result, elapsed


def detect_neurokit2(
    signal: np.ndarray,
    fs: int,
    peak_method: str,
    delineate_method: str,
) -> dict[str, list[WaveEvent]]:
    """Run NeuroKit2 peak detection and delineation on one lead."""

    import neurokit2 as nk

    clean = nk.ecg_clean(
        np.asarray(signal, dtype=float),
        sampling_rate=int(fs),
        method=peak_method,
    )
    _, r_info = nk.ecg_peaks(
        clean,
        sampling_rate=int(fs),
        method=peak_method,
    )
    r_peaks = [_finite_index(value) for value in r_info.get("ECG_R_Peaks", [])]
    r_peaks = [value for value in r_peaks if value is not None]
    if not r_peaks:
        return {"QRS": [], "P": [], "T": []}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, waves = nk.ecg_delineate(
            clean,
            {"ECG_R_Peaks": np.asarray(r_peaks, dtype=int)},
            sampling_rate=int(fs),
            method=delineate_method,
            check=True,
        )

    qrs_events = [
        WaveEvent(
            onset=_at(waves.get("ECG_R_Onsets"), index),
            peak=r_peak,
            offset=_at(waves.get("ECG_R_Offsets"), index),
        )
        for index, r_peak in enumerate(r_peaks)
    ]

    def aligned_wave(prefix: str) -> list[WaveEvent]:
        peaks = waves.get(f"ECG_{prefix}_Peaks", [])
        onsets = waves.get(f"ECG_{prefix}_Onsets", [])
        offsets = waves.get(f"ECG_{prefix}_Offsets", [])
        length = max(len(peaks), len(onsets), len(offsets))
        return _deduplicate_events(
            WaveEvent(
                onset=_at(onsets, index),
                peak=_at(peaks, index),
                offset=_at(offsets, index),
            )
            for index in range(length)
        )

    return {
        "QRS": qrs_events,
        "P": aligned_wave("P"),
        "T": aligned_wave("T"),
    }


def detect_biosppy(signal: np.ndarray, fs: int) -> dict[str, list[WaveEvent]]:
    """Run BioSPPy's public ECG pipeline on one lead.

    BioSPPy exposes corrected R peaks but not P/T or QRS onset/offset from this
    API, so only QRS events with an R fiducial are returned.
    """

    from biosppy.signals import ecg as biosppy_ecg

    output = biosppy_ecg.ecg(
        signal=np.asarray(signal, dtype=float),
        sampling_rate=float(fs),
        show=False,
    )
    r_peaks = output["rpeaks"]
    return {
        "QRS": [
            WaveEvent(None, _finite_index(value), None)
            for value in r_peaks
            if _finite_index(value) is not None
        ],
        "P": [],
        "T": [],
    }


def match_events(
    ground_truth: Sequence[WaveEvent],
    detected: Sequence[WaveEvent],
    tolerance_samples: int,
) -> tuple[list[tuple[WaveEvent, WaveEvent]], int, int]:
    """Order-preserving one-to-one matching inside a fixed tolerance.

    Dynamic programming first maximizes the number of matched events and then
    minimizes total absolute peak error. This avoids the under-counting that a
    globally greedy nearest-neighbour matcher can cause for adjacent events.
    """

    gt = sorted(
        (event for event in ground_truth if event.peak is not None),
        key=lambda event: int(event.peak or -1),
    )
    det = sorted(
        (event for event in detected if event.peak is not None),
        key=lambda event: int(event.peak or -1),
    )
    n_gt = len(gt)
    n_det = len(det)
    # Each state is (matched_count, total_absolute_error). Higher match count
    # wins; for ties, lower total error wins.
    scores = [[(0, 0) for _ in range(n_det + 1)] for _ in range(n_gt + 1)]
    choices = [["" for _ in range(n_det + 1)] for _ in range(n_gt + 1)]

    def better(
        candidate: tuple[int, int],
        current: tuple[int, int],
    ) -> bool:
        return candidate[0] > current[0] or (
            candidate[0] == current[0] and candidate[1] < current[1]
        )

    for gt_index in range(1, n_gt + 1):
        choices[gt_index][0] = "skip_gt"
    for det_index in range(1, n_det + 1):
        choices[0][det_index] = "skip_det"

    for gt_index in range(1, n_gt + 1):
        for det_index in range(1, n_det + 1):
            best = scores[gt_index - 1][det_index]
            choice = "skip_gt"
            skip_det = scores[gt_index][det_index - 1]
            if better(skip_det, best):
                best = skip_det
                choice = "skip_det"

            distance = abs(
                int(gt[gt_index - 1].peak) - int(det[det_index - 1].peak)
            )
            if distance <= tolerance_samples:
                previous = scores[gt_index - 1][det_index - 1]
                matched = (previous[0] + 1, previous[1] + distance)
                if better(matched, best):
                    best = matched
                    choice = "match"
            scores[gt_index][det_index] = best
            choices[gt_index][det_index] = choice

    pairs: list[tuple[WaveEvent, WaveEvent]] = []
    gt_index, det_index = n_gt, n_det
    while gt_index > 0 or det_index > 0:
        choice = choices[gt_index][det_index]
        if choice == "match":
            pairs.append((gt[gt_index - 1], det[det_index - 1]))
            gt_index -= 1
            det_index -= 1
        elif choice == "skip_gt":
            gt_index -= 1
        else:
            det_index -= 1
    pairs.reverse()
    return pairs, len(gt) - len(pairs), len(det) - len(pairs)


def _error_ms(gt_sample: int | None, det_sample: int | None, fs: int) -> float | None:
    if gt_sample is None or det_sample is None:
        return None
    return (int(det_sample) - int(gt_sample)) * 1000.0 / float(fs)


def restrict_to_annotated_span(
    gt_events: Sequence[WaveEvent],
    detected_events: Sequence[WaveEvent],
    tolerance_samples: int,
) -> tuple[list[WaveEvent], int]:
    """Drop detections that lie outside the annotated span.

    LUDB annotates only a middle window of each 10 s record (record 1, for
    instance, is annotated from 1.32 s to 7.94 s), so the beats before and
    after that window are real beats with no reference to match against.
    Counting them as false positives measures annotation coverage, not
    detector precision, and it penalises every method for being correct.

    The margin is the match tolerance itself, which makes the filter exact
    rather than a judgement call: a detection farther than the tolerance from
    the span could not have matched any ground-truth event even in principle,
    so it is dropped; every detection that had a chance to match is kept.
    """
    peaks = [int(event.peak) for event in gt_events if event.peak is not None]
    if not peaks:
        # Nothing annotated in this lead: no detection here is scoreable.
        return [], sum(event.peak is not None for event in detected_events)
    low = min(peaks) - tolerance_samples
    high = max(peaks) + tolerance_samples
    kept = [
        event
        for event in detected_events
        if event.peak is None or low <= int(event.peak) <= high
    ]
    dropped = sum(event.peak is not None for event in detected_events) - sum(
        event.peak is not None for event in kept
    )
    return kept, dropped


def compare_wave(
    *,
    record_id: str,
    lead: str,
    method: str,
    wave: str,
    gt_events: Sequence[WaveEvent],
    detected_events: Sequence[WaveEvent],
    fs: int,
    tolerance_ms: float,
    status: str = "ok",
    error: str = "",
    restrict_span: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tolerance_samples = max(1, int(round(tolerance_ms * fs / 1000.0)))
    outside_span = 0
    if restrict_span:
        detected_events, outside_span = restrict_to_annotated_span(
            gt_events, detected_events, tolerance_samples
        )
    pairs, false_negatives, false_positives = match_events(
        gt_events,
        detected_events,
        tolerance_samples,
    )
    true_positives = len(pairs)
    gt_count = sum(event.peak is not None for event in gt_events)
    det_count = sum(event.peak is not None for event in detected_events)
    detection = {
        "record": record_id,
        "lead": lead,
        "method": method,
        "wave": wave,
        "status": status,
        "error": error,
        "tolerance_ms": tolerance_ms,
        "gt_count": gt_count,
        "det_count": det_count,
        "det_outside_annotated_span": outside_span,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "sensitivity": (
            true_positives / gt_count if gt_count else None
        ),
        "precision": (
            true_positives / det_count if det_count else None
        ),
        "f1": (
            2 * true_positives
            / (2 * true_positives + false_positives + false_negatives)
            if 2 * true_positives + false_positives + false_negatives
            else None
        ),
    }

    matched_rows = []
    for gt_event, det_event in pairs:
        matched_rows.append(
            {
                "record": record_id,
                "lead": lead,
                "method": method,
                "wave": wave,
                "gt_onset_sample": gt_event.onset,
                "det_onset_sample": det_event.onset,
                "onset_error_ms": _error_ms(gt_event.onset, det_event.onset, fs),
                "gt_peak_sample": gt_event.peak,
                "det_peak_sample": det_event.peak,
                "peak_error_ms": _error_ms(gt_event.peak, det_event.peak, fs),
                "gt_offset_sample": gt_event.offset,
                "det_offset_sample": det_event.offset,
                "offset_error_ms": _error_ms(gt_event.offset, det_event.offset, fs),
            }
        )
    return detection, matched_rows


def _failure(
    record_id: str,
    lead: str,
    method: str,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "record": record_id,
        "lead": lead,
        "method": method,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _failed_method_rows(
    *,
    record_id: str,
    lead: str,
    method: str,
    gt_waves: Mapping[str, Sequence[WaveEvent]],
    fs: int,
    config: CompareConfig,
    message: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supported = WAVES if method in DELINEATION_METHODS else ("QRS",)
    detection_rows = []
    for wave in supported:
        tolerance = (
            config.r_tolerance_ms if wave == "QRS" else config.wave_tolerance_ms
        )
        row, _matches = compare_wave(
            record_id=record_id,
            lead=lead,
            method=method,
            wave=wave,
            gt_events=gt_waves.get(wave, []),
            detected_events=[],
            fs=fs,
            tolerance_ms=tolerance,
            status="failed",
            error=message,
        )
        detection_rows.append(row)
    return detection_rows, []


def evaluate_record(record_id: str, config: CompareConfig) -> RecordResult:
    """Evaluate all selected methods for one LUDB record."""

    import wfdb

    record_path = Path(config.dataset_data_dir) / record_id
    record = wfdb.rdrecord(str(record_path))
    fs = int(round(float(record.fs)))
    signal = np.asarray(record.p_signal, dtype=float)
    indices = _lead_indices(record.sig_name)
    ecg_12lead = signal[
        :,
        [indices[lead] for lead in STANDARD_12_LEADS],
    ].T

    gt_by_lead = {
        lead: parse_ludb_waves(record_path, LEAD_TO_ANN_EXT[lead])
        for lead in config.leads
    }
    detection_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    if "ecgfeat" in config.methods:
        started = time.perf_counter()
        try:
            detected_by_lead, elapsed = detect_ecgfeat(
                ecg_12lead,
                fs,
                config.leads,
                config.mains_freq,
                qrs_peak_source=config.ecgfeat_r_source,
                wave_peak_source=config.ecgfeat_wave_source,
            )
            runtime_rows.append(
                {
                    "record": record_id,
                    "lead": "ALL",
                    "method": "ecgfeat",
                    "seconds": elapsed,
                    "status": "ok",
                    "error": "",
                }
            )
            for lead in config.leads:
                for wave in WAVES:
                    tolerance = (
                        config.r_tolerance_ms
                        if wave == "QRS"
                        else config.wave_tolerance_ms
                    )
                    detection, matches = compare_wave(
                        record_id=record_id,
                        lead=lead,
                        method="ecgfeat",
                        wave=wave,
                        gt_events=gt_by_lead[lead][wave],
                        detected_events=detected_by_lead[lead][wave],
                        fs=fs,
                        tolerance_ms=tolerance,
                        restrict_span=config.restrict_to_annotated_span,
                    )
                    detection_rows.append(detection)
                    match_rows.extend(matches)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failure_rows.append(_failure(record_id, "ALL", "ecgfeat", "record", exc))
            runtime_rows.append(
                {
                    "record": record_id,
                    "lead": "ALL",
                    "method": "ecgfeat",
                    "seconds": elapsed,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            for lead in config.leads:
                failed, matches = _failed_method_rows(
                    record_id=record_id,
                    lead=lead,
                    method="ecgfeat",
                    gt_waves=gt_by_lead[lead],
                    fs=fs,
                    config=config,
                    message=str(exc),
                )
                detection_rows.extend(failed)
                match_rows.extend(matches)
            if config.fail_fast:
                raise

    for lead in config.leads:
        lead_signal = signal[:, indices[lead]]
        if "neurokit2" in config.methods:
            started = time.perf_counter()
            try:
                detected = detect_neurokit2(
                    lead_signal,
                    fs,
                    config.neurokit_peak_method,
                    config.neurokit_delineate_method,
                )
                elapsed = time.perf_counter() - started
                runtime_rows.append(
                    {
                        "record": record_id,
                        "lead": lead,
                        "method": "neurokit2",
                        "seconds": elapsed,
                        "status": "ok",
                        "error": "",
                    }
                )
                for wave in WAVES:
                    tolerance = (
                        config.r_tolerance_ms
                        if wave == "QRS"
                        else config.wave_tolerance_ms
                    )
                    detection, matches = compare_wave(
                        record_id=record_id,
                        lead=lead,
                        method="neurokit2",
                        wave=wave,
                        gt_events=gt_by_lead[lead][wave],
                        detected_events=detected[wave],
                        fs=fs,
                        tolerance_ms=tolerance,
                        restrict_span=config.restrict_to_annotated_span,
                    )
                    detection_rows.append(detection)
                    match_rows.extend(matches)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                failure_rows.append(
                    _failure(record_id, lead, "neurokit2", "lead", exc)
                )
                runtime_rows.append(
                    {
                        "record": record_id,
                        "lead": lead,
                        "method": "neurokit2",
                        "seconds": elapsed,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                failed, matches = _failed_method_rows(
                    record_id=record_id,
                    lead=lead,
                    method="neurokit2",
                    gt_waves=gt_by_lead[lead],
                    fs=fs,
                    config=config,
                    message=str(exc),
                )
                detection_rows.extend(failed)
                match_rows.extend(matches)
                if config.fail_fast:
                    raise

        if "biosppy" in config.methods:
            started = time.perf_counter()
            try:
                detected = detect_biosppy(lead_signal, fs)
                elapsed = time.perf_counter() - started
                runtime_rows.append(
                    {
                        "record": record_id,
                        "lead": lead,
                        "method": "biosppy",
                        "seconds": elapsed,
                        "status": "ok",
                        "error": "",
                    }
                )
                detection, matches = compare_wave(
                    record_id=record_id,
                    lead=lead,
                    method="biosppy",
                    wave="QRS",
                    gt_events=gt_by_lead[lead]["QRS"],
                    detected_events=detected["QRS"],
                    fs=fs,
                    tolerance_ms=config.r_tolerance_ms,
                    restrict_span=config.restrict_to_annotated_span,
                )
                detection_rows.append(detection)
                match_rows.extend(matches)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                failure_rows.append(
                    _failure(record_id, lead, "biosppy", "lead", exc)
                )
                runtime_rows.append(
                    {
                        "record": record_id,
                        "lead": lead,
                        "method": "biosppy",
                        "seconds": elapsed,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                failed, matches = _failed_method_rows(
                    record_id=record_id,
                    lead=lead,
                    method="biosppy",
                    gt_waves=gt_by_lead[lead],
                    fs=fs,
                    config=config,
                    message=str(exc),
                )
                detection_rows.extend(failed)
                match_rows.extend(matches)
                if config.fail_fast:
                    raise

    return RecordResult(
        detection_rows=detection_rows,
        match_rows=match_rows,
        runtime_rows=runtime_rows,
        failure_rows=failure_rows,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _error_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "bias_ms": None,
            "sd_ms": None,
            "mae_ms": None,
            "median_ae_ms": None,
            "p95_ae_ms": None,
        }
    array = np.asarray(values, dtype=float)
    absolute = np.abs(array)
    return {
        "n": int(array.size),
        "bias_ms": float(np.mean(array)),
        "sd_ms": float(np.std(array)),
        "mae_ms": float(np.mean(absolute)),
        "median_ae_ms": float(np.median(absolute)),
        "p95_ae_ms": float(np.percentile(absolute, 95)),
    }


def aggregate_metrics(
    detection_rows: Sequence[Mapping[str, Any]],
    match_rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in detection_rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    match_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in match_rows:
        key = tuple(row[field] for field in group_fields)
        match_groups.setdefault(key, []).append(row)

    aggregated = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        rows = groups[key]
        matches = match_groups.get(key, [])
        gt_count = sum(int(row["gt_count"]) for row in rows)
        det_count = sum(int(row["det_count"]) for row in rows)
        true_positives = sum(int(row["tp"]) for row in rows)
        false_positives = sum(int(row["fp"]) for row in rows)
        false_negatives = sum(int(row["fn"]) for row in rows)
        result = {field: value for field, value in zip(group_fields, key)}
        result.update(
            {
                "records": len({str(row["record"]) for row in rows}),
                "lead_records": len(rows),
                "failed_lead_records": sum(row["status"] != "ok" for row in rows),
                "gt_count": gt_count,
                "det_count": det_count,
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "sensitivity": _ratio(true_positives, gt_count),
                "precision": _ratio(true_positives, det_count),
                "f1": _ratio(
                    2 * true_positives,
                    2 * true_positives + false_positives + false_negatives,
                ),
            }
        )
        for boundary in ("onset", "peak", "offset"):
            stats = _error_stats(
                [
                    float(row[f"{boundary}_error_ms"])
                    for row in matches
                    if row.get(f"{boundary}_error_ms") is not None
                ]
            )
            for statistic, value in stats.items():
                result[f"{boundary}_{statistic}"] = value
        aggregated.append(result)
    return aggregated


def aggregate_runtime(
    runtime_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in runtime_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    output = []
    for method in METHODS:
        rows = grouped.get(method, [])
        if not rows:
            continue
        values = [float(row["seconds"]) for row in rows]
        output.append(
            {
                "method": method,
                "calls": len(rows),
                "failed_calls": sum(row["status"] != "ok" for row in rows),
                "total_seconds": float(sum(values)),
                "mean_seconds": float(np.mean(values)),
                "p95_seconds": float(np.percentile(values, 95)),
                "timing_unit": (
                    "record_12lead" if method == "ecgfeat" else "single_lead"
                ),
            }
        )
    return output


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_metadata(
    *,
    args: argparse.Namespace,
    data_dir: Path,
    records: Sequence[str],
    config: CompareConfig,
) -> dict[str, Any]:
    return {
        "schema_version": "ludb_detector_comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "LUDB",
            "version": "1.0.1",
            "data_dir": str(data_dir),
            "records": list(records),
            "record_count": len(records),
            "sampling_rate_hz": 500,
            "annotation_scope": "lead_specific",
        },
        "comparison": {
            **asdict(config),
            "workers": args.workers,
            "qrs_anchor": "R_peak",
            "matching": "one_to_one_nearest_within_tolerance",
            "error_sign": "detected_minus_ground_truth",
            "aggregate_unit": "wave_events_across_record_lead_pairs",
            "biosppy_capability": "R_peaks_only",
            "ecgfeat_detection_scope": (
                "multilead_beat_detection_with_"
                f"{config.ecgfeat_r_source}_qrs_markers"
            ),
            "ecgfeat_qrs_peak_source": config.ecgfeat_r_source,
            "ecgfeat_wave_peak_source": config.ecgfeat_wave_source,
            "neurokit2_detection_scope": "independent_single_lead",
            "biosppy_detection_scope": "independent_single_lead",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "wfdb": _package_version("wfdb"),
            "neurokit2": _package_version("neurokit2"),
            "biosppy": _package_version("biosppy"),
            "ecgfeat": _package_version("ecgfeat-dxl-inspired"),
        },
    }


def plot_summary(summary: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_by_key = {
        (str(row["method"]), str(row["wave"])): row for row in summary
    }
    methods = [method for method in METHODS if any(key[0] == method for key in summary_by_key)]
    waves = list(WAVES)
    colors = {
        "ecgfeat": "#2166ac",
        "neurokit2": "#4daf4a",
        "biosppy": "#d95f02",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(waves), dtype=float)
    width = 0.24
    for method_index, method in enumerate(methods):
        values = [
            summary_by_key.get((method, wave), {}).get("f1")
            for wave in waves
        ]
        heights = [float(value) if value is not None else np.nan for value in values]
        offset = (method_index - (len(methods) - 1) / 2.0) * width
        axes[0].bar(
            x + offset,
            heights,
            width,
            label=method,
            color=colors[method],
        )
    axes[0].set_xticks(x, waves)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Micro F1")
    axes[0].set_title("Wave detection")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    boundaries = [
        ("QRS", "onset", "QRS on"),
        ("QRS", "peak", "R peak"),
        ("QRS", "offset", "QRS off"),
        ("P", "onset", "P on"),
        ("P", "peak", "P peak"),
        ("P", "offset", "P off"),
        ("T", "onset", "T on"),
        ("T", "peak", "T peak"),
        ("T", "offset", "T off"),
    ]
    x2 = np.arange(len(boundaries), dtype=float)
    for method_index, method in enumerate(methods):
        heights = []
        for wave, boundary, _label in boundaries:
            value = summary_by_key.get((method, wave), {}).get(
                f"{boundary}_mae_ms"
            )
            heights.append(float(value) if value is not None else np.nan)
        offset = (method_index - (len(methods) - 1) / 2.0) * width
        axes[1].bar(
            x2 + offset,
            heights,
            width,
            label=method,
            color=colors[method],
        )
    axes[1].set_xticks(
        x2,
        [label for _wave, _boundary, label in boundaries],
        rotation=35,
        ha="right",
    )
    axes[1].set_ylabel("MAE (ms)")
    axes[1].set_title("Localization error on matched waves")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("LUDB detector comparison", fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def print_summary(
    summary: Sequence[Mapping[str, Any]],
    runtime_summary: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    print()
    print("LUDB detector comparison")
    print(
        f"{'Method':<12} {'Wave':<5} {'GT':>7} {'Det':>7} "
        f"{'Sens':>7} {'PPV':>7} {'F1':>7} "
        f"{'On MAE':>9} {'Peak MAE':>10} {'Off MAE':>9}"
    )
    for row in summary:
        print(
            f"{str(row['method']):<12} {str(row['wave']):<5} "
            f"{int(row['gt_count']):>7} {int(row['det_count']):>7} "
            f"{_fmt_metric(row['sensitivity']):>7} "
            f"{_fmt_metric(row['precision']):>7} "
            f"{_fmt_metric(row['f1']):>7} "
            f"{_fmt_metric(row['onset_mae_ms'], 1):>9} "
            f"{_fmt_metric(row['peak_mae_ms'], 1):>10} "
            f"{_fmt_metric(row['offset_mae_ms'], 1):>9}"
        )
    print()
    print("Runtime")
    for row in runtime_summary:
        print(
            f"  {row['method']:<12} total={row['total_seconds']:.2f}s "
            f"mean={row['mean_seconds']:.3f}s/{row['timing_unit']} "
            f"failed={row['failed_calls']}/{row['calls']}"
        )
    print()
    print(f"Results: {output_dir}")


def _parse_csv_choice(raw: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return tuple(allowed)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid {label}: {invalid}. Allowed: {', '.join(allowed)}"
        )
    if not values:
        raise ValueError(f"At least one {label} must be selected")
    return values


def _check_dependencies(methods: Sequence[str]) -> None:
    required = {"wfdb": "wfdb"}
    if "neurokit2" in methods:
        required["neurokit2"] = "neurokit2"
    if "biosppy" in methods:
        required["biosppy"] = "biosppy"
        required["peakutils"] = "peakutils"
    missing = []
    for package, module in required.items():
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "Missing comparison dependencies: "
            + ", ".join(missing)
            + ". Install with: python -m pip install -r "
            "requirements-ludb-compare.txt"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ecgfeat, NeuroKit2, and BioSPPy detections against "
            "lead-specific LUDB expert annotations."
        )
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
        help="Directory for CSV, JSON, and PNG results.",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Specific record IDs. Default: all discovered records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N selected records.",
    )
    parser.add_argument(
        "--leads",
        default="all",
        help="Comma-separated standard lead names or 'all'.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated ecgfeat,neurokit2,biosppy or 'all'.",
    )
    parser.add_argument(
        "--r-tolerance-ms",
        type=float,
        default=75.0,
        help="One-to-one R-peak match tolerance (default: 75 ms).",
    )
    parser.add_argument(
        "--wave-tolerance-ms",
        type=float,
        default=150.0,
        help="One-to-one P/T peak match tolerance (default: 150 ms).",
    )
    parser.add_argument(
        "--ecgfeat-r-source",
        choices=("global", "lead-fiducial", "positive-r", "hybrid-prominence"),
        default="global",
        help=(
            "ecgfeat QRS marker used for matching. 'global' preserves the "
            "historical evaluation; 'lead-fiducial' evaluates each lead's "
            "locally refined QRS point; 'hybrid-prominence' uses the parallel "
            "positive-prominence localizer with a negative fallback."
        ),
    )
    parser.add_argument(
        "--ecgfeat-wave-source",
        choices=("native", "hybrid-peaks"),
        default="native",
        help=(
            "ecgfeat P/T markers used for matching. 'native' preserves the "
            "historical evaluation; 'hybrid-peaks' uses the parallel bipolar "
            "prominence T peaks while retaining native P timing and all native "
            "onset/offset boundaries."
        ),
    )
    parser.add_argument(
        "--neurokit-peak-method",
        default="neurokit",
        choices=(
            "neurokit",
            "pantompkins1985",
            "hamilton2002",
            "elgendi2010",
            "engzeemod2012",
        ),
        help="NeuroKit2 cleaning/R-peak method.",
    )
    parser.add_argument(
        "--neurokit-delineate-method",
        default="dwt",
        choices=("peak", "prominence", "cwt", "dwt"),
        help="NeuroKit2 delineation method.",
    )
    parser.add_argument(
        "--restrict-to-annotated-span",
        action="store_true",
        help=(
            "Score only inside each lead's annotated span. LUDB annotates a "
            "middle window of every 10 s record, so beats outside it are real "
            "but unreferenced and otherwise count as false positives against "
            "every method. Sensitivity and boundary errors are unaffected; "
            "precision and F1 become interpretable."
        ),
    )
    parser.add_argument(
        "--mains-freq",
        type=float,
        default=50.0,
        help="Mains frequency passed to ecgfeat.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel record workers. Use 1 for the most reproducible run.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate comparison.png.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first detector failure instead of recording it.",
    )
    return parser


def _select_records(
    available: Sequence[str],
    requested: Sequence[str] | None,
    limit: int | None,
) -> list[str]:
    if requested:
        available_set = set(available)
        missing = [record for record in requested if record not in available_set]
        if missing:
            raise ValueError(f"Requested LUDB records not found: {missing}")
        selected = sorted(set(requested), key=_record_sort_key)
    else:
        selected = list(available)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No LUDB records selected")
    return selected


def _merge_record_result(
    target: RecordResult,
    source: RecordResult,
) -> None:
    target.detection_rows.extend(source.detection_rows)
    target.match_rows.extend(source.match_rows)
    target.runtime_rows.extend(source.runtime_rows)
    target.failure_rows.extend(source.failure_rows)


def run(args: argparse.Namespace) -> int:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.r_tolerance_ms <= 0 or args.wave_tolerance_ms <= 0:
        raise ValueError("Matching tolerances must be positive")

    leads = _parse_csv_choice(args.leads, STANDARD_12_LEADS, "leads")
    methods = _parse_csv_choice(args.methods, METHODS, "methods")
    _check_dependencies(methods)
    data_dir = resolve_dataset_data_dir(args.dataset_dir)
    records = _select_records(
        discover_records(data_dir),
        args.records,
        args.limit,
    )
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = CompareConfig(
        dataset_data_dir=str(data_dir),
        leads=leads,
        methods=methods,
        r_tolerance_ms=float(args.r_tolerance_ms),
        wave_tolerance_ms=float(args.wave_tolerance_ms),
        ecgfeat_r_source=args.ecgfeat_r_source,
        ecgfeat_wave_source=args.ecgfeat_wave_source,
        neurokit_peak_method=args.neurokit_peak_method,
        neurokit_delineate_method=args.neurokit_delineate_method,
        mains_freq=float(args.mains_freq),
        fail_fast=bool(args.fail_fast),
        restrict_to_annotated_span=bool(args.restrict_to_annotated_span),
    )
    combined = RecordResult([], [], [], [])
    print(
        f"LUDB records={len(records)} leads={len(leads)} "
        f"methods={','.join(methods)} workers={args.workers} "
        f"ecgfeat_r_source={args.ecgfeat_r_source} "
        f"ecgfeat_wave_source={args.ecgfeat_wave_source}"
    )

    if args.workers == 1:
        for index, record_id in enumerate(records, start=1):
            print(f"[{index}/{len(records)}] record {record_id}", flush=True)
            _merge_record_result(combined, evaluate_record(record_id, config))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_record = {
                executor.submit(evaluate_record, record_id, config): record_id
                for record_id in records
            }
            completed = 0
            for future in as_completed(future_to_record):
                record_id = future_to_record[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    combined.failure_rows.append(
                        _failure(record_id, "ALL", "worker", "record", exc)
                    )
                    print(
                        f"[{completed}/{len(records)}] record {record_id} FAILED: {exc}",
                        flush=True,
                    )
                    continue
                _merge_record_result(combined, result)
                print(
                    f"[{completed}/{len(records)}] record {record_id}",
                    flush=True,
                )

    sort_fields = ("record", "lead", "method", "wave")
    combined.detection_rows.sort(
        key=lambda row: tuple(str(row.get(field, "")) for field in sort_fields)
    )
    combined.match_rows.sort(
        key=lambda row: (
            str(row["record"]),
            str(row["lead"]),
            str(row["method"]),
            str(row["wave"]),
            int(row.get("gt_peak_sample") or -1),
        )
    )
    combined.runtime_rows.sort(
        key=lambda row: (str(row["record"]), str(row["method"]), str(row["lead"]))
    )

    summary = aggregate_metrics(
        combined.detection_rows,
        combined.match_rows,
        ("method", "wave"),
    )
    summary_by_lead = aggregate_metrics(
        combined.detection_rows,
        combined.match_rows,
        ("method", "wave", "lead"),
    )
    record_summary = aggregate_metrics(
        combined.detection_rows,
        combined.match_rows,
        ("record", "method", "wave"),
    )
    runtime_summary = aggregate_runtime(combined.runtime_rows)
    metadata = build_metadata(
        args=args,
        data_dir=data_dir,
        records=records,
        config=config,
    )

    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "summary_by_lead.csv", summary_by_lead)
    write_csv(output_dir / "record_summary.csv", record_summary)
    write_csv(output_dir / "detection_rows.csv", combined.detection_rows)
    write_csv(output_dir / "matched_events.csv", combined.match_rows)
    write_csv(output_dir / "runtime.csv", combined.runtime_rows)
    write_csv(output_dir / "runtime_summary.csv", runtime_summary)
    write_csv(output_dir / "failures.csv", combined.failure_rows)

    json_payload = {
        "metadata": metadata,
        "summary": summary,
        "summary_by_lead": summary_by_lead,
        "runtime_summary": runtime_summary,
        "failure_count": len(combined.failure_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(json_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.no_plot:
        plot_summary(summary, output_dir / "comparison.png")

    print_summary(summary, runtime_summary, output_dir)
    return 0 if not combined.failure_rows else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

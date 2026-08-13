#!/usr/bin/env python3
"""Split LUDB P-duration error into P-onset and P-offset components.

Primary analysis uses the multilead-consensus P boundaries that feed the
record-level P-duration measurement. A secondary analysis compares both the
raw lead-local and consensus-corrected lead-local boundaries with direct LUDB
per-lead annotations.

All boundary errors are algorithm minus LUDB:

* positive onset error = algorithm onset is late;
* negative offset error = algorithm offset is early;
* duration error = offset error - onset error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from compare_annotations import (
    ECGFeatureExtractor,
    MATCH_TOLERANCE_MS,
    REPORT_INTERNAL_FS,
    REPORT_MAINS_FREQ,
    STANDARD_12_LEADS,
    _build_patient_meta,
    build_annotation_derived_result,
    discover_record_ids,
    load_all_gt_annotations,
    load_record,
    parse_ludb_header,
)
from evaluate_ludb_cse import (
    _finite_or_none,
    _stats,
    record_p_duration_ms,
    trim_largest_deviations,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ludb_p_boundary_analysis"
PRIMARY_OUTLIER_COUNT = 8


def _median_or_none(values: Iterable[object]) -> Optional[float]:
    finite = [
        value
        for item in values
        if (value := _finite_or_none(item)) is not None
    ]
    return float(np.median(finite)) if finite else None


def _median_int_or_none(values: Iterable[object]) -> Optional[int]:
    value = _median_or_none(values)
    return int(round(value)) if value is not None else None


def _beat_consensus_points(result) -> List[Dict[str, object]]:
    """Collapse repeated lead-level consensus fields to one row per beat."""
    features_by_beat: Dict[int, list] = defaultdict(list)
    for feature in result.beat_features:
        features_by_beat[int(feature.beat_id)].append(feature)
    r_by_beat = {int(beat.beat_id): int(beat.r_index) for beat in result.beats}

    rows: List[Dict[str, object]] = []
    for beat_id, items in sorted(features_by_beat.items()):
        onset = _median_int_or_none(
            item.p_onset_consensus_index for item in items
        )
        offset = _median_int_or_none(
            item.p_offset_consensus_index for item in items
        )
        duration = (
            (offset - onset) * 1000.0 / float(result.fs)
            if onset is not None and offset is not None
            else None
        )
        rows.append(
            {
                "beat_id": beat_id,
                "r_index": r_by_beat.get(
                    beat_id,
                    _median_int_or_none(item.qrs.peak for item in items),
                ),
                "p_onset": onset,
                "p_offset": offset,
                "p_duration_ms": duration,
                "p_onset_support": _median_or_none(
                    item.p_onset_consensus_support for item in items
                ),
                "p_offset_support": _median_or_none(
                    item.p_offset_consensus_support for item in items
                ),
                "p_duration_guard_target_ms": _median_or_none(
                    item.p_duration_guard_target_ms for item in items
                ),
                "p_duration_guard_delta_ms": _median_or_none(
                    item.p_duration_guard_delta_ms for item in items
                ),
                "p_duration_guard_support": _median_or_none(
                    item.p_duration_guard_support for item in items
                ),
                "p_duration_guard_source": next(
                    (
                        item.p_duration_guard_source
                        for item in items
                        if item.p_duration_guard_source
                    ),
                    None,
                ),
            }
        )
    return rows


def _match_by_r(
    reference_rows: Sequence[Mapping[str, object]],
    algorithm_rows: Sequence[Mapping[str, object]],
    fs: int,
) -> List[tuple[Mapping[str, object], Mapping[str, object]]]:
    tolerance = int(round(MATCH_TOLERANCE_MS * fs / 1000.0))
    candidates = [
        row for row in algorithm_rows
        if row.get("r_index") is not None
    ]
    used: set[int] = set()
    pairs: List[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for reference in reference_rows:
        reference_r = reference.get("r_index")
        if reference_r is None:
            continue
        ranked = sorted(
            (
                (abs(int(candidate["r_index"]) - int(reference_r)), index)
                for index, candidate in enumerate(candidates)
                if index not in used
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked or ranked[0][0] > tolerance:
            continue
        _, index = ranked[0]
        used.add(index)
        pairs.append((reference, candidates[index]))
    return pairs


def _consensus_pair_rows(
    record_id: str,
    algorithm_result,
    reference_result,
) -> List[Dict[str, object]]:
    reference_rows = _beat_consensus_points(reference_result)
    algorithm_rows = _beat_consensus_points(algorithm_result)
    pairs = _match_by_r(reference_rows, algorithm_rows, algorithm_result.fs)

    rows: List[Dict[str, object]] = []
    for reference, algorithm in pairs:
        row: Dict[str, object] = {
            "record_id": record_id,
            "reference_beat_id": reference["beat_id"],
            "algorithm_beat_id": algorithm["beat_id"],
            "reference_r_index": reference["r_index"],
            "algorithm_r_index": algorithm["r_index"],
            "r_error_ms": (
                (int(algorithm["r_index"]) - int(reference["r_index"]))
                * 1000.0
                / algorithm_result.fs
            ),
            "reference_p_onset": reference["p_onset"],
            "algorithm_p_onset": algorithm["p_onset"],
            "reference_p_offset": reference["p_offset"],
            "algorithm_p_offset": algorithm["p_offset"],
            "reference_p_duration_ms": reference["p_duration_ms"],
            "algorithm_p_duration_ms": algorithm["p_duration_ms"],
            "reference_p_onset_support": reference["p_onset_support"],
            "algorithm_p_onset_support": algorithm["p_onset_support"],
            "reference_p_offset_support": reference["p_offset_support"],
            "algorithm_p_offset_support": algorithm["p_offset_support"],
            "reference_p_duration_guard_target_ms": reference[
                "p_duration_guard_target_ms"
            ],
            "algorithm_p_duration_guard_target_ms": algorithm[
                "p_duration_guard_target_ms"
            ],
            "reference_p_duration_guard_delta_ms": reference[
                "p_duration_guard_delta_ms"
            ],
            "algorithm_p_duration_guard_delta_ms": algorithm[
                "p_duration_guard_delta_ms"
            ],
            "reference_p_duration_guard_support": reference[
                "p_duration_guard_support"
            ],
            "algorithm_p_duration_guard_support": algorithm[
                "p_duration_guard_support"
            ],
            "reference_p_duration_guard_source": reference[
                "p_duration_guard_source"
            ],
            "algorithm_p_duration_guard_source": algorithm[
                "p_duration_guard_source"
            ],
        }
        for boundary in ("onset", "offset"):
            algorithm_value = algorithm[f"p_{boundary}"]
            reference_value = reference[f"p_{boundary}"]
            row[f"{boundary}_error_ms"] = (
                (int(algorithm_value) - int(reference_value))
                * 1000.0
                / algorithm_result.fs
                if algorithm_value is not None and reference_value is not None
                else None
            )
        onset_error = row["onset_error_ms"]
        offset_error = row["offset_error_ms"]
        row["duration_error_ms"] = (
            float(offset_error) - float(onset_error)
            if onset_error is not None and offset_error is not None
            else None
        )
        row["onset_contraction_ms"] = (
            -float(onset_error) if onset_error is not None else None
        )
        row["offset_contraction_ms"] = (
            float(offset_error) if offset_error is not None else None
        )
        rows.append(row)
    return rows


def _direct_reference_p_events(
    gt_by_lead: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, List[Dict[str, int]]]:
    """Map each LUDB post-T P annotation to the following QRS in that lead."""
    events: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for lead, beats in gt_by_lead.items():
        qrs_rows = [
            beat for beat in beats
            if beat.get("r_sample") is not None
        ]
        for beat in beats:
            p_on = beat.get("p_on")
            p_peak = beat.get("p_peak")
            p_off = beat.get("p_off")
            if p_on is None or p_peak is None or p_off is None:
                continue
            following = [
                candidate for candidate in qrs_rows
                if int(candidate["r_sample"]) > int(p_off)
            ]
            if not following:
                continue
            target = min(following, key=lambda candidate: int(candidate["r_sample"]))
            events[lead].append(
                {
                    "r_index": int(target["r_sample"]),
                    "p_onset": int(p_on),
                    "p_offset": int(p_off),
                }
            )
    return events


def _per_lead_pair_rows(
    record_id: str,
    algorithm_result,
    gt_by_lead: Mapping[str, Sequence[Mapping[str, object]]],
) -> List[Dict[str, object]]:
    reference_by_lead = _direct_reference_p_events(gt_by_lead)
    algorithm_by_lead: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for feature in algorithm_result.beat_features:
        if feature.qrs.peak is None:
            continue
        lead_quality = algorithm_result.quality.get(feature.lead)
        algorithm_by_lead[feature.lead].append(
            {
                "r_index": int(feature.qrs.peak),
                "p_onset": feature.p_onset_corrected_index,
                "p_offset": feature.p_offset_corrected_index,
                "p_onset_raw_index": feature.p_onset_raw_index,
                "p_offset_raw_index": feature.p_offset_raw_index,
                "p_onset_corrected_index": feature.p_onset_corrected_index,
                "p_offset_corrected_index": feature.p_offset_corrected_index,
                "p_boundary_source": feature.p_boundary_source,
                "p_confidence": feature.p_confidence,
                "p_onset_confidence": feature.p_onset_confidence,
                "p_offset_confidence": feature.p_offset_confidence,
                "reliable_for_p": bool(
                    getattr(lead_quality, "reliable_for_p", False)
                ),
                "p_onset_consensus_index": feature.p_onset_consensus_index,
                "p_offset_consensus_index": feature.p_offset_consensus_index,
                "p_onset_consensus_support": feature.p_onset_consensus_support,
                "p_offset_consensus_support": feature.p_offset_consensus_support,
                "p_onset_corrected": "p_onset_corrected_by_consensus" in feature.flags,
                "p_offset_corrected": "p_offset_corrected_by_consensus" in feature.flags,
                "p_suppressed": "p_candidate_suppressed_by_consensus" in feature.flags,
            }
        )

    rows: List[Dict[str, object]] = []
    for lead in STANDARD_12_LEADS:
        references = reference_by_lead.get(lead, [])
        algorithms = algorithm_by_lead.get(lead, [])
        pairs = _match_by_r(references, algorithms, algorithm_result.fs)
        for reference, algorithm in pairs:
            row: Dict[str, object] = {
                "record_id": record_id,
                "lead": lead,
                "reference_r_index": reference["r_index"],
                "algorithm_r_index": algorithm["r_index"],
                "reference_p_onset": reference["p_onset"],
                "algorithm_p_onset": algorithm["p_onset"],
                "reference_p_offset": reference["p_offset"],
                "algorithm_p_offset": algorithm["p_offset"],
                "algorithm_p_onset_raw_index": algorithm["p_onset_raw_index"],
                "algorithm_p_offset_raw_index": algorithm["p_offset_raw_index"],
                "algorithm_p_onset_corrected_index": algorithm[
                    "p_onset_corrected_index"
                ],
                "algorithm_p_offset_corrected_index": algorithm[
                    "p_offset_corrected_index"
                ],
                "algorithm_p_boundary_source": algorithm["p_boundary_source"],
                "algorithm_p_confidence": algorithm["p_confidence"],
                "algorithm_p_onset_confidence": algorithm["p_onset_confidence"],
                "algorithm_p_offset_confidence": algorithm["p_offset_confidence"],
                "algorithm_reliable_for_p": algorithm["reliable_for_p"],
                "algorithm_p_onset_consensus_index": algorithm[
                    "p_onset_consensus_index"
                ],
                "algorithm_p_offset_consensus_index": algorithm[
                    "p_offset_consensus_index"
                ],
                "algorithm_p_onset_consensus_support": algorithm[
                    "p_onset_consensus_support"
                ],
                "algorithm_p_offset_consensus_support": algorithm[
                    "p_offset_consensus_support"
                ],
                "algorithm_p_onset_corrected": algorithm["p_onset_corrected"],
                "algorithm_p_offset_corrected": algorithm["p_offset_corrected"],
                "algorithm_p_suppressed": algorithm["p_suppressed"],
            }
            for boundary in ("onset", "offset"):
                algorithm_value = algorithm[f"p_{boundary}"]
                reference_value = reference[f"p_{boundary}"]
                row[f"{boundary}_error_ms"] = (
                    (int(algorithm_value) - int(reference_value))
                    * 1000.0
                    / algorithm_result.fs
                    if algorithm_value is not None
                    else None
                )
            onset_error = row["onset_error_ms"]
            offset_error = row["offset_error_ms"]
            row["duration_error_ms"] = (
                float(offset_error) - float(onset_error)
                if onset_error is not None and offset_error is not None
                else None
            )
            for layer in ("raw", "corrected"):
                for boundary in ("onset", "offset"):
                    algorithm_value = algorithm[f"p_{boundary}_{layer}_index"]
                    reference_value = reference[f"p_{boundary}"]
                    row[f"{layer}_{boundary}_error_ms"] = (
                        (int(algorithm_value) - int(reference_value))
                        * 1000.0
                        / algorithm_result.fs
                        if algorithm_value is not None
                        else None
                    )
                layer_onset_error = row[f"{layer}_onset_error_ms"]
                layer_offset_error = row[f"{layer}_offset_error_ms"]
                row[f"{layer}_duration_error_ms"] = (
                    float(layer_offset_error) - float(layer_onset_error)
                    if layer_onset_error is not None and layer_offset_error is not None
                    else None
                )
            for boundary in ("onset", "offset"):
                raw_index = algorithm[f"p_{boundary}_raw_index"]
                corrected_index = algorithm[f"p_{boundary}_corrected_index"]
                row[f"{boundary}_correction_shift_ms"] = (
                    (int(corrected_index) - int(raw_index))
                    * 1000.0
                    / algorithm_result.fs
                    if raw_index is not None and corrected_index is not None
                    else None
                )
            onset_shift = row["onset_correction_shift_ms"]
            offset_shift = row["offset_correction_shift_ms"]
            row["duration_correction_delta_ms"] = (
                float(offset_shift) - float(onset_shift)
                if onset_shift is not None and offset_shift is not None
                else None
            )
            rows.append(row)
    return rows


def evaluate_record(record_id: str) -> Dict[str, object]:
    try:
        ecg, fs = load_record(record_id)
        patient_meta = _build_patient_meta(parse_ludb_header(record_id))
        extractor = ECGFeatureExtractor(
            fs_internal=REPORT_INTERNAL_FS,
            mains_freq=REPORT_MAINS_FREQ,
        )
        algorithm_result = extractor.extract(ecg, float(fs), meta=patient_meta)
        gt_by_lead = load_all_gt_annotations(
            record_id,
            int(fs),
            REPORT_INTERNAL_FS,
        )
        reference_result = build_annotation_derived_result(
            ecg,
            int(fs),
            gt_by_lead,
            preferred_lead="II",
            fs_internal=REPORT_INTERNAL_FS,
            patient_meta=patient_meta,
        )
        algorithm_duration, algorithm_source = record_p_duration_ms(algorithm_result)
        reference_duration, reference_source = record_p_duration_ms(reference_result)
        duration_difference = (
            algorithm_duration - reference_duration
            if algorithm_duration is not None and reference_duration is not None
            else None
        )
        return {
            "record_id": record_id,
            "error": None,
            "record": {
                "record_id": record_id,
                "algorithm_p_duration_ms": algorithm_duration,
                "reference_p_duration_ms": reference_duration,
                "duration_difference_ms": duration_difference,
                "algorithm_p_duration_source": algorithm_source,
                "reference_p_duration_source": reference_source,
            },
            "consensus_pairs": _consensus_pair_rows(
                record_id,
                algorithm_result,
                reference_result,
            ),
            "per_lead_pairs": _per_lead_pair_rows(
                record_id,
                algorithm_result,
                gt_by_lead,
            ),
        }
    except Exception as exc:
        return {
            "record_id": record_id,
            "error": f"{type(exc).__name__}: {exc}",
            "record": {
                "record_id": record_id,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "consensus_pairs": [],
            "per_lead_pairs": [],
        }


def _record_boundary_rows(
    consensus_rows: Sequence[Mapping[str, object]],
    record_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_record: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in consensus_rows:
        by_record[str(row["record_id"])].append(row)
    duration_by_record = {
        str(row["record_id"]): row for row in record_rows
    }

    output: List[Dict[str, object]] = []
    for record_id in sorted(
        duration_by_record,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        pairs = by_record.get(record_id, [])
        complete = [
            row for row in pairs
            if _finite_or_none(row.get("onset_error_ms")) is not None
            and _finite_or_none(row.get("offset_error_ms")) is not None
        ]
        onset_median = _median_or_none(
            row.get("onset_error_ms") for row in complete
        )
        offset_median = _median_or_none(
            row.get("offset_error_ms") for row in complete
        )
        pair_duration_median = _median_or_none(
            row.get("duration_error_ms") for row in complete
        )
        duration_row = duration_by_record[record_id]
        output.append(
            {
                "record_id": record_id,
                "matched_qrs_beats": len(pairs),
                "complete_consensus_p_pairs": len(complete),
                "median_onset_error_ms": onset_median,
                "median_offset_error_ms": offset_median,
                "median_paired_duration_error_ms": pair_duration_median,
                "onset_contraction_component_ms": (
                    -onset_median if onset_median is not None else None
                ),
                "offset_contraction_component_ms": offset_median,
                "algorithm_record_p_duration_ms": duration_row.get(
                    "algorithm_p_duration_ms"
                ),
                "reference_record_p_duration_ms": duration_row.get(
                    "reference_p_duration_ms"
                ),
                "record_p_duration_difference_ms": duration_row.get(
                    "duration_difference_ms"
                ),
            }
        )
    return output


def _error_stats(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> Dict[str, Optional[float]]:
    return _stats(
        [
            value
            for row in rows
            if (value := _finite_or_none(row.get(field))) is not None
        ]
    )


def summarize(
    results: Sequence[Mapping[str, object]],
) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    record_rows = [dict(result["record"]) for result in results]
    consensus_rows = [
        dict(row)
        for result in results
        for row in result["consensus_pairs"]
    ]
    per_lead_rows = [
        dict(row)
        for result in results
        for row in result["per_lead_pairs"]
    ]
    record_boundary_rows = _record_boundary_rows(consensus_rows, record_rows)

    complete_consensus = [
        row for row in consensus_rows
        if _finite_or_none(row.get("onset_error_ms")) is not None
        and _finite_or_none(row.get("offset_error_ms")) is not None
    ]
    onset_stats = _error_stats(complete_consensus, "onset_error_ms")
    offset_stats = _error_stats(complete_consensus, "offset_error_ms")
    duration_stats = _error_stats(complete_consensus, "duration_error_ms")
    onset_mean = float(onset_stats["mean_ms"])
    offset_mean = float(offset_stats["mean_ms"])
    duration_mean = float(duration_stats["mean_ms"])
    reference_onset_rows = [
        row for row in consensus_rows
        if _finite_or_none(row.get("reference_p_onset")) is not None
    ]
    reference_offset_rows = [
        row for row in consensus_rows
        if _finite_or_none(row.get("reference_p_offset")) is not None
    ]
    reference_complete_rows = [
        row for row in consensus_rows
        if _finite_or_none(row.get("reference_p_onset")) is not None
        and _finite_or_none(row.get("reference_p_offset")) is not None
    ]
    paired_onset_rows = [
        row for row in reference_onset_rows
        if _finite_or_none(row.get("algorithm_p_onset")) is not None
    ]
    paired_offset_rows = [
        row for row in reference_offset_rows
        if _finite_or_none(row.get("algorithm_p_offset")) is not None
    ]

    record_complete = [
        row for row in record_boundary_rows
        if _finite_or_none(row.get("median_onset_error_ms")) is not None
        and _finite_or_none(row.get("median_offset_error_ms")) is not None
    ]
    record_onset_stats = _error_stats(record_complete, "median_onset_error_ms")
    record_offset_stats = _error_stats(record_complete, "median_offset_error_ms")
    record_duration_stats = _error_stats(
        record_complete,
        "median_paired_duration_error_ms",
    )

    duration_record_values = [
        (str(row["record_id"]), value)
        for row in record_rows
        if (value := _finite_or_none(row.get("duration_difference_ms"))) is not None
    ]
    kept, removed = trim_largest_deviations(
        duration_record_values,
        PRIMARY_OUTLIER_COUNT,
    )

    per_lead_complete = [
        row for row in per_lead_rows
        if _finite_or_none(row.get("onset_error_ms")) is not None
        and _finite_or_none(row.get("offset_error_ms")) is not None
    ]
    per_lead_raw_complete = [
        row for row in per_lead_rows
        if _finite_or_none(row.get("raw_onset_error_ms")) is not None
        and _finite_or_none(row.get("raw_offset_error_ms")) is not None
    ]
    per_lead_correction_complete = [
        row for row in per_lead_rows
        if _finite_or_none(row.get("onset_correction_shift_ms")) is not None
        and _finite_or_none(row.get("offset_correction_shift_ms")) is not None
    ]
    lead_stats: Dict[str, object] = {}
    for lead in STANDARD_12_LEADS:
        lead_rows = [row for row in per_lead_complete if row["lead"] == lead]
        lead_stats[lead] = {
            "n": len(lead_rows),
            "onset": _error_stats(lead_rows, "onset_error_ms"),
            "offset": _error_stats(lead_rows, "offset_error_ms"),
            "duration": _error_stats(lead_rows, "duration_error_ms"),
        }

    onset_contraction = -onset_mean
    offset_contraction = offset_mean
    negative_magnitude = abs(min(0.0, onset_contraction)) + abs(
        min(0.0, offset_contraction)
    )
    onset_share = (
        abs(min(0.0, onset_contraction)) / negative_magnitude
        if negative_magnitude
        else None
    )
    offset_share = (
        abs(min(0.0, offset_contraction)) / negative_magnitude
        if negative_magnitude
        else None
    )
    direction_counts = {
        "both_late_onset_and_early_offset": 0,
        "late_onset_without_early_offset": 0,
        "early_offset_without_late_onset": 0,
        "neither": 0,
    }
    for row in record_complete:
        onset = float(row["median_onset_error_ms"])
        offset = float(row["median_offset_error_ms"])
        if onset > 0.0 and offset < 0.0:
            direction_counts["both_late_onset_and_early_offset"] += 1
        elif onset > 0.0:
            direction_counts["late_onset_without_early_offset"] += 1
        elif offset < 0.0:
            direction_counts["early_offset_without_late_onset"] += 1
        else:
            direction_counts["neither"] += 1

    support = {
        "algorithm_onset_median": _median_or_none(
            row.get("algorithm_p_onset_support") for row in complete_consensus
        ),
        "reference_onset_median": _median_or_none(
            row.get("reference_p_onset_support") for row in complete_consensus
        ),
        "algorithm_offset_median": _median_or_none(
            row.get("algorithm_p_offset_support") for row in complete_consensus
        ),
        "reference_offset_median": _median_or_none(
            row.get("reference_p_offset_support") for row in complete_consensus
        ),
    }

    summary = {
        "schema_version": "ecgfeat_ludb_p_boundary_analysis.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "difference_definition": "algorithm minus LUDB",
        "requested_records": len(results),
        "successful_records": sum(not result.get("error") for result in results),
        "failed_records": [
            {
                "record_id": result["record_id"],
                "error": result["error"],
            }
            for result in results
            if result.get("error")
        ],
        "consensus_beat_level": {
            "matched_qrs_beats": len(consensus_rows),
            "reference_onset_available": len(reference_onset_rows),
            "paired_onset_available": len(paired_onset_rows),
            "onset_coverage": (
                len(paired_onset_rows) / len(reference_onset_rows)
                if reference_onset_rows
                else None
            ),
            "reference_offset_available": len(reference_offset_rows),
            "paired_offset_available": len(paired_offset_rows),
            "offset_coverage": (
                len(paired_offset_rows) / len(reference_offset_rows)
                if reference_offset_rows
                else None
            ),
            "reference_complete_p_boundaries": len(reference_complete_rows),
            "complete_p_boundary_pairs": len(complete_consensus),
            "complete_boundary_coverage": (
                len(complete_consensus) / len(reference_complete_rows)
                if reference_complete_rows
                else None
            ),
            "all_available_onset_error": _error_stats(
                paired_onset_rows,
                "onset_error_ms",
            ),
            "all_available_offset_error": _error_stats(
                paired_offset_rows,
                "offset_error_ms",
            ),
            "onset_error": onset_stats,
            "offset_error": offset_stats,
            "duration_error": duration_stats,
            "support": support,
            "paired_duration_guard": {
                "algorithm_triggered_beats": sum(
                    bool(row.get("algorithm_p_duration_guard_source"))
                    for row in consensus_rows
                ),
                "reference_triggered_beats": sum(
                    bool(row.get("reference_p_duration_guard_source"))
                    for row in consensus_rows
                ),
                "algorithm_delta_ms": _error_stats(
                    consensus_rows,
                    "algorithm_p_duration_guard_delta_ms",
                ),
                "reference_delta_ms": _error_stats(
                    consensus_rows,
                    "reference_p_duration_guard_delta_ms",
                ),
            },
            "decomposition": {
                "onset_contraction_component_ms": onset_contraction,
                "offset_contraction_component_ms": offset_contraction,
                "reconstructed_duration_error_ms": (
                    onset_contraction + offset_contraction
                ),
                "direct_duration_error_ms": duration_mean,
                "onset_share_of_negative_contraction": onset_share,
                "offset_share_of_negative_contraction": offset_share,
            },
        },
        "consensus_record_level": {
            "records_with_complete_p_boundary_pairs": len(record_complete),
            "median_onset_error_across_records": record_onset_stats,
            "median_offset_error_across_records": record_offset_stats,
            "median_paired_duration_error_across_records": record_duration_stats,
            "direction_counts": direction_counts,
        },
        "record_p_duration": {
            "paired_records": len(duration_record_values),
            "raw": _stats([value for _, value in duration_record_values]),
            "trimmed": _stats([value for _, value in kept]),
            "outliers_removed": [
                {"record_id": record_id, "difference_ms": difference}
                for record_id, difference in removed
            ],
        },
        "per_lead_final_boundaries": {
            "expert_p_events": len(per_lead_rows),
            "complete_p_boundary_pairs": len(per_lead_complete),
            "onset_coverage": (
                sum(
                    _finite_or_none(row.get("onset_error_ms")) is not None
                    for row in per_lead_rows
                )
                / len(per_lead_rows)
                if per_lead_rows
                else None
            ),
            "offset_coverage": (
                sum(
                    _finite_or_none(row.get("offset_error_ms")) is not None
                    for row in per_lead_rows
                )
                / len(per_lead_rows)
                if per_lead_rows
                else None
            ),
            "onset_error": _error_stats(per_lead_complete, "onset_error_ms"),
            "offset_error": _error_stats(per_lead_complete, "offset_error_ms"),
            "duration_error": _error_stats(per_lead_complete, "duration_error_ms"),
            "onset_corrected_count": sum(
                str(row.get("algorithm_p_onset_corrected")).lower() == "true"
                for row in per_lead_rows
            ),
            "offset_corrected_count": sum(
                str(row.get("algorithm_p_offset_corrected")).lower() == "true"
                for row in per_lead_rows
            ),
            "suppressed_count": sum(
                str(row.get("algorithm_p_suppressed")).lower() == "true"
                for row in per_lead_rows
            ),
            "by_lead": lead_stats,
        },
        "per_lead_raw_boundaries": {
            "expert_p_events": len(per_lead_rows),
            "complete_p_boundary_pairs": len(per_lead_raw_complete),
            "onset_coverage": (
                sum(
                    _finite_or_none(row.get("raw_onset_error_ms")) is not None
                    for row in per_lead_rows
                )
                / len(per_lead_rows)
                if per_lead_rows
                else None
            ),
            "offset_coverage": (
                sum(
                    _finite_or_none(row.get("raw_offset_error_ms")) is not None
                    for row in per_lead_rows
                )
                / len(per_lead_rows)
                if per_lead_rows
                else None
            ),
            "onset_error": _error_stats(
                per_lead_raw_complete,
                "raw_onset_error_ms",
            ),
            "offset_error": _error_stats(
                per_lead_raw_complete,
                "raw_offset_error_ms",
            ),
            "duration_error": _error_stats(
                per_lead_raw_complete,
                "raw_duration_error_ms",
            ),
        },
        "per_lead_correction_effect": {
            "complete_raw_and_corrected_pairs": len(per_lead_correction_complete),
            "onset_shift": _error_stats(
                per_lead_correction_complete,
                "onset_correction_shift_ms",
            ),
            "offset_shift": _error_stats(
                per_lead_correction_complete,
                "offset_correction_shift_ms",
            ),
            "duration_delta": _error_stats(
                per_lead_correction_complete,
                "duration_correction_delta_ms",
            ),
        },
    }
    return summary, record_rows, consensus_rows, per_lead_rows


def _fmt(value: object, digits: int = 2) -> str:
    number = _finite_or_none(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def render_markdown(summary: Mapping[str, object]) -> str:
    beat = summary["consensus_beat_level"]
    decomposition = beat["decomposition"]
    record = summary["consensus_record_level"]
    duration = summary["record_p_duration"]
    raw_per_lead = summary["per_lead_raw_boundaries"]
    per_lead = summary["per_lead_final_boundaries"]
    correction = summary["per_lead_correction_effect"]
    onset_share = decomposition["onset_share_of_negative_contraction"]
    offset_share = decomposition["offset_share_of_negative_contraction"]
    duration_guard = beat["paired_duration_guard"]

    lines = [
        "# ecgfeat 在 LUDB 上的 P onset / P offset 误差分解",
        "",
        f"生成时间：{summary['generated_at_utc']}",
        "",
        "## 结论",
        "",
        (
            f"在 {beat['complete_p_boundary_pairs']} 个完整的多导联一致性 P 波配对中，"
            f"P onset 平均误差为 **{_fmt(beat['onset_error']['mean_ms'])} ms**，"
            f"P offset 平均误差为 **{_fmt(beat['offset_error']['mean_ms'])} ms**。"
        ),
        "",
        (
            f"在 LUDB 有完整一致性参考边界的 {beat['reference_complete_p_boundaries']} "
            f"个 P 波中，算法完整输出 {beat['complete_p_boundary_pairs']} 个，"
            f"覆盖率 {100.0 * float(beat['complete_boundary_coverage']):.1f}%。"
        ),
        (
            f"单独看端点：onset覆盖 {beat['paired_onset_available']}/"
            f"{beat['reference_onset_available']} "
            f"({100.0 * float(beat['onset_coverage']):.1f}%)，offset覆盖 "
            f"{beat['paired_offset_available']}/{beat['reference_offset_available']} "
            f"({100.0 * float(beat['offset_coverage']):.1f}%)。"
        ),
        "",
        (
            "因此 P 时限变化可分解为："
            f"onset 项 {_fmt(decomposition['onset_contraction_component_ms'])} ms，"
            f"offset 项 {_fmt(decomposition['offset_contraction_component_ms'])} ms，"
            f"合计 {_fmt(decomposition['reconstructed_duration_error_ms'])} ms。"
        ),
        "",
        (
            f"负向收缩中，onset 贡献约 {100.0 * float(onset_share):.1f}%，"
            f"offset 贡献约 {100.0 * float(offset_share):.1f}%。"
            if onset_share is not None and offset_share is not None
            else "无法计算 onset/offset 的负向收缩占比。"
        ),
        "",
        "## 多导联一致性边界",
        "",
        "| 层级 | 样本数 | Onset均差 | Onset SD | Offset均差 | Offset SD | 时限均差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Beat配对 | {beat['complete_p_boundary_pairs']} "
            f"| {_fmt(beat['onset_error']['mean_ms'])} "
            f"| {_fmt(beat['onset_error']['sd_ms'])} "
            f"| {_fmt(beat['offset_error']['mean_ms'])} "
            f"| {_fmt(beat['offset_error']['sd_ms'])} "
            f"| {_fmt(beat['duration_error']['mean_ms'])} |"
        ),
        (
            f"| 每记录中位误差 | {record['records_with_complete_p_boundary_pairs']} "
            f"| {_fmt(record['median_onset_error_across_records']['mean_ms'])} "
            f"| {_fmt(record['median_onset_error_across_records']['sd_ms'])} "
            f"| {_fmt(record['median_offset_error_across_records']['mean_ms'])} "
            f"| {_fmt(record['median_offset_error_across_records']['sd_ms'])} "
            f"| {_fmt(record['median_paired_duration_error_across_records']['mean_ms'])} |"
        ),
        "",
        (
            f"记录级 P 时限差：未剔除均差 {_fmt(duration['raw']['mean_ms'])} ms，"
            f"剔除8条后 {_fmt(duration['trimmed']['mean_ms'])} ms。"
        ),
        "",
        (
            f"{record['records_with_complete_p_boundary_pairs']}条具有完整记录级边界的记录中："
            f"{record['direction_counts']['both_late_onset_and_early_offset']}条同时存在"
            "onset偏晚和offset偏早；"
            f"{record['direction_counts']['late_onset_without_early_offset']}条仅呈onset偏晚方向；"
            f"{record['direction_counts']['early_offset_without_late_onset']}条仅呈offset偏早方向。"
        ),
        "",
        "## Raw 与修正后逐导联边界",
        "",
        (
            f"直接 LUDB 专家 P 事件共 {per_lead['expert_p_events']} 个；"
            f"raw 完整起止点配对 {raw_per_lead['complete_p_boundary_pairs']} 个，"
            f"修正后完整起止点配对 {per_lead['complete_p_boundary_pairs']} 个。"
        ),
        "",
        "| 层级 | 项目 | 均差 | SD | MAE | P95绝对误差 |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| Raw | P onset | {_fmt(raw_per_lead['onset_error']['mean_ms'])} "
            f"| {_fmt(raw_per_lead['onset_error']['sd_ms'])} "
            f"| {_fmt(raw_per_lead['onset_error']['mae_ms'])} "
            f"| {_fmt(raw_per_lead['onset_error']['p95_abs_ms'])} |"
        ),
        (
            f"| Raw | P offset | {_fmt(raw_per_lead['offset_error']['mean_ms'])} "
            f"| {_fmt(raw_per_lead['offset_error']['sd_ms'])} "
            f"| {_fmt(raw_per_lead['offset_error']['mae_ms'])} "
            f"| {_fmt(raw_per_lead['offset_error']['p95_abs_ms'])} |"
        ),
        (
            f"| Raw | P duration | {_fmt(raw_per_lead['duration_error']['mean_ms'])} "
            f"| {_fmt(raw_per_lead['duration_error']['sd_ms'])} "
            f"| {_fmt(raw_per_lead['duration_error']['mae_ms'])} "
            f"| {_fmt(raw_per_lead['duration_error']['p95_abs_ms'])} |"
        ),
        (
            f"| Corrected | P onset | {_fmt(per_lead['onset_error']['mean_ms'])} "
            f"| {_fmt(per_lead['onset_error']['sd_ms'])} "
            f"| {_fmt(per_lead['onset_error']['mae_ms'])} "
            f"| {_fmt(per_lead['onset_error']['p95_abs_ms'])} |"
        ),
        (
            f"| Corrected | P offset | {_fmt(per_lead['offset_error']['mean_ms'])} "
            f"| {_fmt(per_lead['offset_error']['sd_ms'])} "
            f"| {_fmt(per_lead['offset_error']['mae_ms'])} "
            f"| {_fmt(per_lead['offset_error']['p95_abs_ms'])} |"
        ),
        (
            f"| Corrected | P duration | {_fmt(per_lead['duration_error']['mean_ms'])} "
            f"| {_fmt(per_lead['duration_error']['sd_ms'])} "
            f"| {_fmt(per_lead['duration_error']['mae_ms'])} "
            f"| {_fmt(per_lead['duration_error']['p95_abs_ms'])} |"
        ),
        "",
        (
            f"Raw onset/offset覆盖率分别为 "
            f"{100.0 * float(raw_per_lead['onset_coverage']):.1f}%/"
            f"{100.0 * float(raw_per_lead['offset_coverage']):.1f}%；"
            f"修正后为 {100.0 * float(per_lead['onset_coverage']):.1f}%/"
            f"{100.0 * float(per_lead['offset_coverage']):.1f}%。"
        ),
        "",
        (
            f"在 {correction['complete_raw_and_corrected_pairs']} 个 raw 和修正后边界均完整的"
            f"事件中，共识修正平均令 onset 移动 "
            f"{_fmt(correction['onset_shift']['mean_ms'])} ms、offset 移动 "
            f"{_fmt(correction['offset_shift']['mean_ms'])} ms，P 时限净变化 "
            f"{_fmt(correction['duration_delta']['mean_ms'])} ms。"
        ),
        "",
        "## 融合层观察",
        "",
        (
            f"- 完整配对中，算法 onset 支持度中位数为 "
            f"{_fmt(beat['support']['algorithm_onset_median'], 1)} 个导联，"
            f"参考为 {_fmt(beat['support']['reference_onset_median'], 1)}；"
            f"算法 offset 支持度中位数为 "
            f"{_fmt(beat['support']['algorithm_offset_median'], 1)}，"
            f"参考为 {_fmt(beat['support']['reference_offset_median'], 1)}。"
        ),
        (
            "- 逐导联最终边界的平均偏差为 onset "
            f"{_fmt(per_lead['onset_error']['mean_ms'])} ms、offset "
            f"{_fmt(per_lead['offset_error']['mean_ms'])} ms；融合后完整配对变为 "
            f"{_fmt(beat['onset_error']['mean_ms'])} ms 和 "
            f"{_fmt(beat['offset_error']['mean_ms'])} ms。"
            "两层样本权重并不完全相同，但方向表明多导联边界选择会放大时限收缩。"
        ),
        (
            f"- 最终逐导联事件中，onset共识修正触发 "
            f"{per_lead['onset_corrected_count']} 次，offset共识修正触发 "
            f"{per_lead['offset_corrected_count']} 次。"
        ),
        (
            f"- 成对时限保护在算法侧触发 {duration_guard['algorithm_triggered_beats']} 拍，"
            f"平均延长 {_fmt(duration_guard['algorithm_delta_ms']['mean_ms'])} ms；"
            f"LUDB参考重建侧触发 {duration_guard['reference_triggered_beats']} 拍，"
            f"平均延长 {_fmt(duration_guard['reference_delta_ms']['mean_ms'])} ms。"
        ),
        "",
        "## 解释",
        "",
        "- onset 为正表示算法起点晚于专家标注，会缩短 P 时限。",
        "- offset 为负表示算法终点早于专家标注，也会缩短 P 时限。",
        "- `duration error = offset error − onset error`，因此两部分可直接相加解释时限收缩。",
        "- Raw 是单导联初始测量；Corrected 是多导联共识修正/抑制后的单导联测量；Consensus 是记录对齐后的全局边界。",
        "",
    ]
    return "\n".join(lines)


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, object],
    record_rows: Sequence[Mapping[str, object]],
    consensus_rows: Sequence[Mapping[str, object]],
    per_lead_rows: Sequence[Mapping[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record_boundary_rows = _record_boundary_rows(consensus_rows, record_rows)
    _write_csv(record_rows, output_dir / "record_p_duration.csv")
    _write_csv(record_boundary_rows, output_dir / "record_boundary_summary.csv")
    _write_csv(consensus_rows, output_dir / "consensus_beat_pairs.csv")
    _write_csv(per_lead_rows, output_dir / "per_lead_boundary_pairs.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )


def run(record_ids: Sequence[str], workers: int) -> List[Dict[str, object]]:
    worker_count = max(1, min(int(workers), len(record_ids)))
    if worker_count == 1:
        return [evaluate_record(record_id) for record_id in record_ids]

    results_by_id: Dict[str, Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(evaluate_record, record_id): record_id
            for record_id in record_ids
        }
        completed = 0
        for future in as_completed(futures):
            record_id = futures[future]
            completed += 1
            results_by_id[record_id] = future.result()
            if completed == 1 or completed % 10 == 0 or completed == len(record_ids):
                print(f"[{completed:>3}/{len(record_ids)}] completed", flush=True)
    return [
        results_by_id[record_id]
        for record_id in record_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split LUDB P-duration error into onset and offset errors."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--records", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, int(os.cpu_count() or 1)),
    )
    args = parser.parse_args()

    record_ids = list(args.records or discover_record_ids())
    if args.limit is not None:
        record_ids = record_ids[: max(0, args.limit)]
    if not record_ids:
        parser.error("No LUDB records selected.")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    print(
        f"Analyzing P boundaries in {len(record_ids)} LUDB records with "
        f"{min(args.workers, len(record_ids))} worker(s)."
    )
    results = run(record_ids, args.workers)
    summary, record_rows, consensus_rows, per_lead_rows = summarize(results)
    write_outputs(
        args.out_dir.resolve(),
        summary,
        record_rows,
        consensus_rows,
        per_lead_rows,
    )
    print(render_markdown(summary))
    print(f"Outputs: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

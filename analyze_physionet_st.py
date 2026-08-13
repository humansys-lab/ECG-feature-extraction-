#!/usr/bin/env python3
"""Evaluate ecgfeat ST measurements on EDB and LTSTDB without changing ecgfeat.

This is a standalone research evaluator for:

* European ST-T Database (EDB): event start/peak/end annotations in ``.atr``.
* Long-Term ST Database (LTSTDB): event annotations in ``.sta/.stb/.stc`` and
  continuous ST level/reference/deviation values in ``.stf``.

The existing :class:`ecgfeat.api.ECGFeatureExtractor` accepts only 12-lead
input.  These databases contain two or three channels.  This script therefore
uses an external sparse-lead adapter:

* real channels are placed in the ecgfeat QRS detector slots II, V2, and V3;
* unused channels are zero and are rejected by ecgfeat's quality layer;
* results are mapped back to the original physical signal names;
* no 12-lead territory or diagnostic interpretation is reported.

Sparse-lead reliability is recomputed independently of ecgfeat's ordered
single-reason flag.  It requires at least two real-lead anchors, checks J-point
agreement and signal/baseline confidence, and normalizes the consensus term by
the number of real channels.  Because these databases define the target at a
fixed post-J offset, early T overlap and an unavailable auxiliary ST slope are
reported as warnings rather than deleting an otherwise measured J+40/J+80
amplitude.

Long recordings are read in short event-centred/control windows.  The script
never edits files under ``feature_extraction/ecgfeat``.

Important comparison semantics
------------------------------

EDB peak amplitudes are changes relative to the subject reference waveform.
For EDB, the evaluator estimates an ecgfeat reference level from the first
30 seconds, then subtracts it from each event-window ST measurement.

LTSTDB ``.stf`` gives expert ST level, time-varying ST reference, and their
difference.  The primary measurement metric compares ecgfeat ST level with
the expert ST level.  A second, explicitly labelled measurement-only metric
subtracts the *expert* ``.stf`` reference from the ecgfeat level.  It evaluates
ST amplitude extraction, not an autonomous long-term reference tracker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.models import PatientMeta, STANDARD_12_LEADS


# All three slots participate in ecgfeat's default multilead QRS detector.
# They are adapter slots, not claims about the physical lead anatomy.
SPARSE_ADAPTER_SLOTS = ("II", "V2", "V3")
SPARSE_ADAPTER_NAME = "real_2_or_3_channels_in_II_V2_V3_zero_padded"
PRIMARY_ST_THRESHOLD_MV = 0.10
CONTROL_ST_LIMIT_MV = 0.05

_EDB_BEGIN_RE = re.compile(r"^\(ST(?P<lead>\d)(?P<direction>[+-])$")
_EDB_PEAK_RE = re.compile(
    r"^AST(?P<lead>\d)(?P<direction>[+-])(?P<uv>\d+)$"
)
_EDB_END_RE = re.compile(r"^ST(?P<lead>\d)(?P<direction>[+-])\)$")

_LTST_BEGIN_RE = re.compile(
    r"^\((?P<kind>rtst|st)(?P<lead>\d)(?P<direction>[+-])(?P<uv>\d+)$"
)
_LTST_PEAK_RE = re.compile(
    r"^a(?P<kind>rtst|st)(?P<lead>\d)(?P<direction>[+-])(?P<uv>\d+)$"
)
_LTST_END_RE = re.compile(
    r"^(?P<kind>rtst|st)(?P<lead>\d)(?P<direction>[+-])(?P<uv>\d+)\)$"
)


@dataclass
class STEpisode:
    dataset: str
    record: str
    lead_index: int
    direction: str
    kind: str
    start_sample: int
    peak_sample: int | None = None
    end_sample: int | None = None
    peak_deviation_mv: float | None = None
    start_deviation_mv: float | None = None
    end_deviation_mv: float | None = None


@dataclass(frozen=True)
class SparseHybridAssessment:
    """Reliability result recomputed consistently for two/three real leads."""

    usable: bool
    reason: str
    adjusted_confidence: float | None
    consensus_support: int
    required_support: int
    warnings: tuple[str, ...] = ()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _signed_mv(direction: str, magnitude_uv: str | int | float) -> float:
    magnitude = abs(float(magnitude_uv)) / 1000.0
    return magnitude if direction == "+" else -magnitude


def _clean_aux(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _parse_patient_meta(comments: Sequence[str]) -> PatientMeta:
    age: float | None = None
    sex: str | None = None
    for raw in comments or ():
        line = str(raw).strip().lstrip("#").strip()
        age_match = re.search(r"\bAge:\s*([0-9.]+)", line, flags=re.IGNORECASE)
        sex_match = re.search(r"\bSex:\s*([A-Za-z]+)", line, flags=re.IGNORECASE)
        if age is None and age_match:
            age = _finite(age_match.group(1))
        if sex is None and sex_match:
            candidate = sex_match.group(1).upper()
            if candidate in {"M", "F", "MALE", "FEMALE"}:
                sex = candidate
    return PatientMeta(age=age, sex=sex)


def parse_edb_episodes(record_path: Path) -> list[STEpisode]:
    """Parse real ST episodes; lower-case axis-shift comments are excluded."""
    import wfdb

    annotation = wfdb.rdann(str(record_path), "atr")
    active: dict[tuple[int, str], STEpisode] = {}
    episodes: list[STEpisode] = []
    for sample, symbol, aux in zip(
        annotation.sample,
        annotation.symbol,
        annotation.aux_note,
    ):
        if symbol != "s":
            continue
        note = _clean_aux(aux)
        if match := _EDB_BEGIN_RE.match(note):
            lead = int(match.group("lead"))
            direction = match.group("direction")
            active[(lead, direction)] = STEpisode(
                dataset="edb",
                record=record_path.name,
                lead_index=lead,
                direction=direction,
                kind="ischemic_st",
                start_sample=int(sample),
            )
            continue
        if match := _EDB_PEAK_RE.match(note):
            key = (int(match.group("lead")), match.group("direction"))
            event = active.get(key)
            if event is not None:
                event.peak_sample = int(sample)
                event.peak_deviation_mv = _signed_mv(
                    event.direction,
                    match.group("uv"),
                )
            continue
        if match := _EDB_END_RE.match(note):
            key = (int(match.group("lead")), match.group("direction"))
            event = active.pop(key, None)
            if event is not None:
                event.end_sample = int(sample)
                if event.peak_sample is not None and event.peak_deviation_mv is not None:
                    episodes.append(event)
    # A small number of EDB events continue through the end of the two-hour
    # excerpt and intentionally have no explicit end marker. Their annotated
    # peak remains valid for amplitude analysis.
    sig_len = int(wfdb.rdheader(str(record_path)).sig_len)
    for event in active.values():
        if event.peak_sample is not None and event.peak_deviation_mv is not None:
            event.end_sample = sig_len - 1
            episodes.append(event)
    return sorted(episodes, key=lambda item: (item.peak_sample or item.start_sample))


def parse_ltstdb_episodes(
    record_path: Path,
    annotation_extension: str,
) -> list[STEpisode]:
    """Parse ischemic (st) and rate-related non-ischemic (rtst) episodes."""
    import wfdb

    annotation = wfdb.rdann(str(record_path), annotation_extension)
    active: dict[tuple[str, int, str], STEpisode] = {}
    episodes: list[STEpisode] = []
    for sample, symbol, aux in zip(
        annotation.sample,
        annotation.symbol,
        annotation.aux_note,
    ):
        if symbol != "s":
            continue
        note = _clean_aux(aux).lower()
        if match := _LTST_BEGIN_RE.match(note):
            kind_code = match.group("kind")
            lead = int(match.group("lead"))
            direction = match.group("direction")
            key = (kind_code, lead, direction)
            active[key] = STEpisode(
                dataset="ltstdb",
                record=record_path.name,
                lead_index=lead,
                direction=direction,
                kind="ischemic_st" if kind_code == "st" else "rate_related_st",
                start_sample=int(sample),
                start_deviation_mv=_signed_mv(direction, match.group("uv")),
            )
            continue
        if match := _LTST_PEAK_RE.match(note):
            key = (
                match.group("kind"),
                int(match.group("lead")),
                match.group("direction"),
            )
            event = active.get(key)
            if event is not None:
                event.peak_sample = int(sample)
                event.peak_deviation_mv = _signed_mv(
                    event.direction,
                    match.group("uv"),
                )
            continue
        if match := _LTST_END_RE.match(note):
            key = (
                match.group("kind"),
                int(match.group("lead")),
                match.group("direction"),
            )
            event = active.pop(key, None)
            if event is not None:
                event.end_sample = int(sample)
                event.end_deviation_mv = _signed_mv(
                    event.direction,
                    match.group("uv"),
                )
                if event.peak_sample is not None and event.peak_deviation_mv is not None:
                    episodes.append(event)
    sig_len = int(wfdb.rdheader(str(record_path)).sig_len)
    for event in active.values():
        if event.peak_sample is not None and event.peak_deviation_mv is not None:
            event.end_sample = sig_len - 1
            episodes.append(event)
    return sorted(episodes, key=lambda item: (item.peak_sample or item.start_sample))


def _select_evenly(items: Sequence[Any], limit: int) -> list[Any]:
    values = list(items)
    if limit <= 0 or len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return [values[int(index)] for index in sorted(set(indices.tolist()))]


def _fill_nonfinite(signal: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(signal, dtype=float).copy()
    finite = np.isfinite(values)
    missing_fraction = 1.0 - float(np.mean(finite)) if values.size else 1.0
    if np.all(finite):
        return values, missing_fraction
    if not np.any(finite):
        return np.zeros_like(values), missing_fraction
    indices = np.arange(values.size)
    values[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return values, missing_fraction


def _median_iqr(values: Iterable[Any]) -> tuple[float | None, float | None, int]:
    finite_values = [
        float(value)
        for value in values
        if _finite(value) is not None
    ]
    if not finite_values:
        return None, None, 0
    array = np.asarray(finite_values, dtype=float)
    iqr = (
        float(np.percentile(array, 75) - np.percentile(array, 25))
        if array.size >= 2
        else 0.0
    )
    return float(np.median(array)), iqr, int(array.size)


def _protocol_value(
    st40_mv: Any,
    st80_mv: Any,
    heart_rate_bpm: float | None,
) -> float | None:
    st40 = _finite(st40_mv)
    st80 = _finite(st80_mv)
    if heart_rate_bpm is not None and heart_rate_bpm > 120.0:
        if st40 is not None and st80 is not None:
            return 0.5 * (st40 + st80)  # J+60 ms interpolation
        return st80 if st80 is not None else st40
    return st80 if st80 is not None else st40


def _assess_sparse_hybrid(
    feature: Any,
    *,
    physical_lead_count: int,
    fs: float,
) -> SparseHybridAssessment:
    """Re-evaluate hybrid ST quality without the core single-reason masking.

    ``localize_st_robust`` emits one reason using a fixed priority order. With
    two physical leads, ``insufficient_consensus_support`` is emitted before
    checks such as early-T overlap and low confidence. Accepting that reason
    directly therefore hides other problems. This adapter reconstructs all
    observable gates and normalizes the support-confidence term by the number
    of real channels rather than by six 12-lead channels.
    """

    n_physical = max(1, int(physical_lead_count))
    required_support = min(2, n_physical)
    support = int(
        round(_finite(getattr(feature, "st_hybrid_consensus_support", None)) or 0.0)
    )
    core_reason = getattr(feature, "st_hybrid_unreliable_reason", None)
    flags = set(getattr(feature, "flags", []) or ())

    def result(
        usable: bool,
        reason: str,
        confidence: float | None = None,
        warnings: Sequence[str] = (),
    ) -> SparseHybridAssessment:
        return SparseHybridAssessment(
            usable=usable,
            reason=reason,
            adjusted_confidence=confidence,
            consensus_support=support,
            required_support=required_support,
            warnings=tuple(warnings),
        )

    if core_reason == "paced_beat" or "paced_beat" in flags:
        return result(False, "paced_beat")
    if core_reason == "qrs_unreliable" or "qrs_unreliable" in flags:
        return result(False, "qrs_unreliable")
    if core_reason == "lead_qrs_quality":
        return result(False, "lead_qrs_quality")
    if support < required_support:
        return result(False, "insufficient_real_lead_support")

    j_method = str(getattr(feature, "st_hybrid_j_method", "") or "")
    if (
        core_reason == "j_consensus_disagreement"
        or j_method == "native_consensus_disagreement"
    ):
        return result(False, "j_consensus_disagreement")

    warnings: list[str] = []
    j_index = getattr(feature, "st_hybrid_j_index", None)
    t_wave = getattr(feature, "t", None)
    t_onset = getattr(t_wave, "onset", None)
    t_peak = getattr(t_wave, "peak", None)
    t_offset = getattr(t_wave, "offset", None)
    if j_index is not None and t_onset is not None and fs > 0:
        st80_index = int(j_index) + int(round(0.080 * float(fs)))
        radius = max(1, int(round(0.006 * float(fs))))
        if int(t_onset) <= st80_index + radius:
            warnings.append("st80_t_overlap")

    # Sparse ambulatory leads frequently produce an implausibly early or
    # internally inconsistent T delineation. EDB/LTSTDB nevertheless define
    # their amplitude target at a fixed post-J offset (normally J+80 ms).
    # Therefore T overlap is recorded as a contamination warning instead of
    # deciding whether the fixed-protocol ST sample exists. The comparison
    # against the expert amplitude remains the final validity check.
    if t_onset is not None and t_peak is not None and int(t_peak) < int(t_onset):
        warnings.append("implausible_t_delineation")
    if t_peak is not None and t_offset is not None and int(t_offset) <= int(t_peak):
        warnings.append("implausible_t_delineation")
    warnings = list(dict.fromkeys(warnings))

    # This evaluator's primary endpoint is the fixed-offset ST amplitude.
    # A slope may be unavailable when an early T marker shortens the fitting
    # interval, even though the J, J+40 and J+80 medians were all measured.
    # Keep slope as an ancillary warning instead of discarding valid amplitude
    # observations.
    required_protocol_values = (
        getattr(feature, "st_hybrid_j_mv", None),
        getattr(feature, "st_hybrid_40ms_mv", None),
        getattr(feature, "st_hybrid_80ms_mv", None),
    )
    if any(_finite(value) is None for value in required_protocol_values):
        return result(
            False,
            "incomplete_fixed_protocol_st",
            warnings=warnings,
        )
    if _finite(getattr(feature, "st_hybrid_slope_mv_per_ms", None)) is None:
        warnings.append("missing_st_slope")
        warnings = list(dict.fromkeys(warnings))

    baseline_confidence = (
        _finite(getattr(feature, "st_hybrid_baseline_confidence", None)) or 0.0
    )
    if baseline_confidence < 0.50:
        return result(False, "low_baseline_confidence", warnings=warnings)

    core_confidence = _finite(
        getattr(feature, "st_hybrid_j_confidence", None)
    )
    if core_confidence is None:
        return result(False, "missing_confidence", warnings=warnings)
    core_support_score = float(np.clip(support / 6.0, 0.0, 1.0))
    sparse_support_score = float(
        np.clip(support / float(n_physical), 0.0, 1.0)
    )
    adjusted_confidence = float(
        np.clip(
            core_confidence
            + 0.20 * (sparse_support_score - core_support_score),
            0.0,
            1.0,
        )
    )
    if adjusted_confidence < 0.50:
        return result(
            False,
            "low_sparse_adjusted_confidence",
            adjusted_confidence,
            warnings,
        )

    known_core_reasons = {
        None,
        "insufficient_consensus_support",
        "low_confidence",
        "early_t_overlap",
        "incomplete_st_window",
        "low_baseline_confidence",
        "j_consensus_disagreement",
    }
    if core_reason not in known_core_reasons:
        return result(
            False,
            f"unhandled_core_reason_{core_reason}",
            adjusted_confidence,
            warnings,
        )
    return result(True, "usable", adjusted_confidence, warnings)


def _native_usable(feature: Any) -> bool:
    flags = set(getattr(feature, "flags", []) or ())
    return bool(
        "qrs_unreliable" not in flags
        and _finite(getattr(feature, "st_mid_mv", None)) is not None
        and _finite(getattr(feature, "st_80ms_mv", None)) is not None
    )


def _aggregate_channel(
    features: Sequence[Any],
    *,
    source_index: int,
    source_lead: str,
    adapter_slot: str,
    heart_rate_bpm: float | None,
    fs: float,
    center_index: int,
    physical_lead_count: int,
) -> dict[str, Any]:
    native_items = [item for item in features if _native_usable(item)]
    hybrid_assessments = {
        id(item): _assess_sparse_hybrid(
            item,
            physical_lead_count=physical_lead_count,
            fs=fs,
        )
        for item in features
    }
    hybrid_items = [
        item
        for item in features
        if hybrid_assessments[id(item)].usable
    ]
    rejection_counts = Counter(
        assessment.reason
        for assessment in hybrid_assessments.values()
        if not assessment.usable
    )
    warning_counts = Counter(
        warning
        for assessment in hybrid_assessments.values()
        for warning in assessment.warnings
    )
    row: dict[str, Any] = {
        "source_lead_index": source_index,
        "source_lead": source_lead,
        "adapter_slot": adapter_slot,
        "adapter_mode": SPARSE_ADAPTER_NAME,
        "heart_rate_bpm": heart_rate_bpm,
        "beats_total": len(features),
        "native_beats_usable": len(native_items),
        "hybrid_beats_usable": len(hybrid_items),
        "hybrid_core_reliable_beats": sum(
            bool(getattr(item, "st_hybrid_reliable", False))
            for item in features
        ),
        "hybrid_sparse_policy": (
            "v2_fixed_protocol_t_overlap_warning_real_lead_normalized"
        ),
        "physical_lead_count": int(physical_lead_count),
        "hybrid_sparse_rejection_counts": json.dumps(
            dict(sorted(rejection_counts.items())),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "hybrid_sparse_warning_counts": json.dumps(
            dict(sorted(warning_counts.items())),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    support_minimum = max(2, int(math.ceil(0.20 * len(features))))
    row["window_support_minimum"] = support_minimum
    row["native_window_support_pass"] = len(native_items) >= support_minimum
    row["hybrid_window_support_pass"] = len(hybrid_items) >= support_minimum
    adjusted_confidences = [
        assessment.adjusted_confidence
        for assessment in hybrid_assessments.values()
        if assessment.usable and assessment.adjusted_confidence is not None
    ]
    (
        row["hybrid_sparse_adjusted_confidence"],
        row["hybrid_sparse_adjusted_confidence_iqr"],
        row["hybrid_sparse_adjusted_confidence_n"],
    ) = _median_iqr(adjusted_confidences)
    fields = {
        "native_stj_mv": (native_items, "st_on_mv"),
        "native_st40_mv": (native_items, "st_mid_mv"),
        "native_st80_mv": (native_items, "st_80ms_mv"),
        "hybrid_stj_mv": (hybrid_items, "st_hybrid_j_mv"),
        "hybrid_st40_mv": (hybrid_items, "st_hybrid_40ms_mv"),
        "hybrid_st80_mv": (hybrid_items, "st_hybrid_80ms_mv"),
        "hybrid_st_mean_mv": (hybrid_items, "st_hybrid_mean_mv"),
        "hybrid_st_area_mv_ms": (hybrid_items, "st_hybrid_area_mv_ms"),
        "hybrid_st_slope_mv_per_ms": (
            hybrid_items,
            "st_hybrid_slope_mv_per_ms",
        ),
        "hybrid_j_confidence": (hybrid_items, "st_hybrid_j_confidence"),
        "hybrid_baseline_confidence": (
            hybrid_items,
            "st_hybrid_baseline_confidence",
        ),
        "hybrid_consensus_support": (
            hybrid_items,
            "st_hybrid_consensus_support",
        ),
    }
    for name, (items, attribute) in fields.items():
        median, iqr, count = _median_iqr(
            getattr(item, attribute, None) for item in items
        )
        row[name] = median
        row[f"{name}_iqr"] = iqr
        row[f"{name}_n"] = count

    native_protocol = [
        _protocol_value(item.st_mid_mv, item.st_80ms_mv, heart_rate_bpm)
        for item in native_items
    ]
    hybrid_protocol = [
        _protocol_value(
            item.st_hybrid_40ms_mv,
            item.st_hybrid_80ms_mv,
            heart_rate_bpm,
        )
        for item in hybrid_items
    ]
    row["native_protocol_st_mv"], row["native_protocol_st_mv_iqr"], _ = (
        _median_iqr(native_protocol)
    )
    row["hybrid_protocol_st_mv"], row["hybrid_protocol_st_mv_iqr"], _ = (
        _median_iqr(hybrid_protocol)
    )
    for method in ("native", "hybrid"):
        row[f"{method}_evaluable_protocol_st_mv"] = (
            row.get(f"{method}_protocol_st_mv")
            if row[f"{method}_window_support_pass"]
            else None
        )
    row["protocol_offset_ms"] = (
        60.0 if heart_rate_bpm is not None and heart_rate_bpm > 120.0 else 80.0
    )

    def feature_r_index(item: Any) -> int | None:
        value = getattr(item, "r_hybrid_index", None)
        if value is None:
            value = getattr(getattr(item, "qrs", None), "peak", None)
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    for method, items, st40_attr, st80_attr in (
        ("native", native_items, "st_mid_mv", "st_80ms_mv"),
        (
            "hybrid",
            hybrid_items,
            "st_hybrid_40ms_mv",
            "st_hybrid_80ms_mv",
        ),
    ):
        candidates = [
            (abs(int(r_index) - int(center_index)), int(r_index), item)
            for item in items
            if (r_index := feature_r_index(item)) is not None
        ]
        # EDB/LTST peak-change annotations are placed immediately before the
        # beat carrying the marked extreme. Prefer the first following beat;
        # fall back to absolute-nearest if the detector has no nearby follower.
        following = [
            candidate
            for candidate in candidates
            if 0 <= candidate[1] - int(center_index) <= int(round(1.5 * fs))
        ]
        nearest = (
            min(following, key=lambda candidate: candidate[1])
            if following
            else (
                min(candidates, key=lambda candidate: candidate[0])
                if candidates
                else None
            )
        )
        if nearest is None:
            row[f"{method}_peakbeat_r_index"] = None
            row[f"{method}_peakbeat_offset_ms"] = None
            row[f"{method}_peakbeat_st40_mv"] = None
            row[f"{method}_peakbeat_st80_mv"] = None
            row[f"{method}_peakbeat_protocol_st_mv"] = None
        else:
            _, r_index, item = nearest
            st40 = _finite(getattr(item, st40_attr, None))
            st80 = _finite(getattr(item, st80_attr, None))
            row[f"{method}_peakbeat_r_index"] = r_index
            row[f"{method}_peakbeat_offset_ms"] = (
                (r_index - int(center_index)) * 1000.0 / float(fs)
            )
            row[f"{method}_peakbeat_st40_mv"] = st40
            row[f"{method}_peakbeat_st80_mv"] = st80
            row[f"{method}_peakbeat_protocol_st_mv"] = _protocol_value(
                st40,
                st80,
                heart_rate_bpm,
            )
        row[f"{method}_peakbeat_evaluable_protocol_st_mv"] = (
            row.get(f"{method}_peakbeat_protocol_st_mv")
            if row[f"{method}_window_support_pass"]
            else None
        )

    markers: list[dict[str, Any]] = []
    for item in features:
        r_index = getattr(item, "r_hybrid_index", None)
        if r_index is None:
            r_index = getattr(getattr(item, "qrs", None), "peak", None)
        native_j = getattr(item, "st_j_remeasured_index", None)
        if native_j is None:
            native_j = getattr(getattr(item, "qrs", None), "offset", None)
        hybrid_j = getattr(item, "st_hybrid_j_index", None)
        markers.append(
            {
                "r_index": r_index,
                "native_j_index": native_j,
                "native_st80_index": (
                    None
                    if native_j is None
                    else int(native_j) + int(round(0.080 * fs))
                ),
                "hybrid_j_index": hybrid_j,
                "hybrid_st80_index": (
                    None
                    if hybrid_j is None
                    else int(hybrid_j) + int(round(0.080 * fs))
                ),
                "hybrid_sparse_usable": hybrid_assessments[id(item)].usable,
                "hybrid_sparse_reason": hybrid_assessments[id(item)].reason,
                "hybrid_sparse_warnings": list(
                    hybrid_assessments[id(item)].warnings
                ),
            }
        )
    row["_markers"] = markers
    return row


def extract_sparse_window(
    record_path: Path,
    *,
    center_sample: int,
    window_sec: float,
    mains_freq: int,
) -> dict[str, Any]:
    """Read a short window and run the unchanged 12-lead ecgfeat extractor."""
    import wfdb

    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs)
    half = max(1, int(round(window_sec * fs / 2.0)))
    start = max(0, int(center_sample) - half)
    end = min(int(header.sig_len), int(center_sample) + half)
    if end - start < max(int(round(4.0 * fs)), 32):
        raise ValueError(
            f"window too short for {record_path.name}: {end - start} samples"
        )
    record = wfdb.rdrecord(str(record_path), sampfrom=start, sampto=end)
    physical = np.asarray(record.p_signal, dtype=float)
    if physical.ndim != 2 or physical.shape[1] not in {2, 3}:
        raise ValueError(
            f"expected 2 or 3 channels, got shape {physical.shape} "
            f"for {record_path.name}"
        )

    ecg12 = np.zeros((12, physical.shape[0]), dtype=float)
    missing_fractions: list[float] = []
    for source_index in range(physical.shape[1]):
        clean, missing_fraction = _fill_nonfinite(physical[:, source_index])
        slot = SPARSE_ADAPTER_SLOTS[source_index]
        ecg12[STANDARD_12_LEADS.index(slot)] = clean
        missing_fractions.append(missing_fraction)

    extractor = ECGFeatureExtractor(
        fs_internal=int(round(fs)),
        mains_freq=int(mains_freq),
        enable_pacing=False,
        enable_lead_reversal=False,
        compute_grouping=False,
        enable_hybrid_r_localization=True,
        enable_hybrid_wave_localization=True,
        enable_hybrid_st_measurement=True,
    )
    result = extractor.extract(
        ecg12,
        fs,
        meta=_parse_patient_meta(getattr(header, "comments", [])),
    )
    heart_rate = _finite(result.global_features.heart_rate_bpm)
    by_slot = {
        slot: [
            item
            for item in result.beat_features
            if str(getattr(item, "lead", "")) == slot
        ]
        for slot in SPARSE_ADAPTER_SLOTS[: physical.shape[1]]
    }
    channels: dict[int, dict[str, Any]] = {}
    for source_index, source_lead in enumerate(record.sig_name):
        slot = SPARSE_ADAPTER_SLOTS[source_index]
        channels[source_index] = _aggregate_channel(
            by_slot[slot],
            source_index=source_index,
            source_lead=str(source_lead),
            adapter_slot=slot,
            heart_rate_bpm=heart_rate,
            fs=fs,
            center_index=int(center_sample) - start,
            physical_lead_count=int(physical.shape[1]),
        )
        channels[source_index]["input_missing_fraction"] = missing_fractions[
            source_index
        ]
    return {
        "record": record_path.name,
        "fs": fs,
        "start_sample": start,
        "end_sample": end,
        "center_sample": int(center_sample),
        "signals": physical.T,
        "signal_names": [str(name) for name in record.sig_name],
        "channels": channels,
        "detected_beats": len(result.beats),
        "heart_rate_bpm": heart_rate,
    }


def _public_channel_row(channel: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in channel.items() if not key.startswith("_")}


def _event_base_row(
    event: STEpisode,
    *,
    lead_name: str,
    annotation_extension: str,
) -> dict[str, Any]:
    return {
        **asdict(event),
        "lead_name": lead_name,
        "annotation_extension": annotation_extension,
        "event_duration_sec": None,
    }


def _add_event_duration(row: dict[str, Any], fs: float) -> None:
    start = row.get("start_sample")
    end = row.get("end_sample")
    if start is not None and end is not None:
        row["event_duration_sec"] = (int(end) - int(start)) / float(fs)


def _candidate_control_centers(
    *,
    sig_len: int,
    fs: float,
    episodes: Sequence[STEpisode],
    count: int,
    edge_sec: float,
) -> list[int]:
    if count <= 0:
        return []
    margin = int(round(edge_sec * fs))
    if sig_len <= 2 * margin:
        return []
    candidates = np.linspace(
        margin,
        sig_len - margin - 1,
        num=max(200, count * 40),
        dtype=int,
    )
    exclusion = int(round(30.0 * fs))
    valid = [
        int(sample)
        for sample in candidates
        if all(
            sample < int(event.start_sample) - exclusion
            or sample > int(event.end_sample or event.start_sample) + exclusion
            for event in episodes
        )
    ]
    return _select_evenly(valid, count)


def _analyse_edb_record(
    record_path: Path,
    *,
    args: argparse.Namespace,
    plot_state: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import wfdb

    header = wfdb.rdheader(str(record_path))
    episodes = parse_edb_episodes(record_path)
    selected = _select_evenly(episodes, args.max_events_per_record)
    reference_center = int(round(0.5 * args.edb_reference_sec * float(header.fs)))
    reference = extract_sparse_window(
        record_path,
        center_sample=reference_center,
        window_sec=args.edb_reference_sec,
        mains_freq=args.mains_freq,
    )

    event_rows: list[dict[str, Any]] = []
    for event in selected:
        if event.lead_index >= int(header.n_sig) or event.peak_sample is None:
            continue
        window = extract_sparse_window(
            record_path,
            center_sample=event.peak_sample,
            window_sec=args.window_sec,
            mains_freq=args.mains_freq,
        )
        current = window["channels"][event.lead_index]
        baseline = reference["channels"][event.lead_index]
        row = {
            **_event_base_row(
                event,
                lead_name=str(header.sig_name[event.lead_index]),
                annotation_extension="atr",
            ),
            **_public_channel_row(current),
            "window_start_sample": window["start_sample"],
            "window_end_sample": window["end_sample"],
            "reference_window_start_sample": reference["start_sample"],
            "reference_window_end_sample": reference["end_sample"],
            "reference_native_protocol_st_mv": baseline.get(
                "native_evaluable_protocol_st_mv"
            ),
            "reference_hybrid_protocol_st_mv": baseline.get(
                "hybrid_evaluable_protocol_st_mv"
            ),
            "event_measurement_source": (
                "ecgfeat_beat_nearest_expert_peak_annotation"
            ),
            "reference_measurement_source": (
                "ecgfeat_window_median_first_30_seconds"
            ),
        }
        for method in ("native", "hybrid"):
            current_value = _finite(
                row.get(f"{method}_peakbeat_evaluable_protocol_st_mv")
            )
            reference_value = _finite(
                row.get(f"reference_{method}_protocol_st_mv")
            )
            predicted = (
                current_value - reference_value
                if current_value is not None and reference_value is not None
                else None
            )
            row[f"{method}_predicted_deviation_mv"] = predicted
            truth = _finite(event.peak_deviation_mv)
            row[f"{method}_deviation_error_mv"] = (
                predicted - truth
                if predicted is not None and truth is not None
                else None
            )
        _add_event_duration(row, float(header.fs))
        event_rows.append(row)
        if plot_state["edb"] < args.plot_examples:
            _plot_event_example(
                out_dir=args.out_dir,
                dataset="edb",
                record_path=record_path,
                event=event,
                window=window,
                row=row,
            )
            plot_state["edb"] += 1

    control_rows: list[dict[str, Any]] = []
    centers = _candidate_control_centers(
        sig_len=int(header.sig_len),
        fs=float(header.fs),
        episodes=episodes,
        count=args.controls_per_record,
        edge_sec=max(args.window_sec, args.edb_reference_sec) / 2.0 + 1.0,
    )
    for control_index, center in enumerate(centers):
        window = extract_sparse_window(
            record_path,
            center_sample=center,
            window_sec=args.window_sec,
            mains_freq=args.mains_freq,
        )
        for lead_index in range(int(header.n_sig)):
            current = window["channels"][lead_index]
            baseline = reference["channels"][lead_index]
            row = {
                "dataset": "edb",
                "record": record_path.name,
                "control_index": control_index,
                "center_sample": center,
                "lead_index": lead_index,
                "lead_name": str(header.sig_name[lead_index]),
                "truth_control_definition": (
                    "outside_all_annotated_ST_events_by_at_least_30s"
                ),
                "truth_deviation_mv": 0.0,
                **_public_channel_row(current),
            }
            for method in ("native", "hybrid"):
                current_value = _finite(
                    row.get(f"{method}_evaluable_protocol_st_mv")
                )
                reference_value = _finite(
                    baseline.get(f"{method}_evaluable_protocol_st_mv")
                )
                row[f"reference_{method}_protocol_st_mv"] = reference_value
                row[f"{method}_predicted_deviation_mv"] = (
                    current_value - reference_value
                    if current_value is not None and reference_value is not None
                    else None
                )
            control_rows.append(row)
    return event_rows, control_rows


def _load_stf(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    if values.ndim != 2 or values.shape[1] not in {7, 10}:
        raise ValueError(f"unexpected LTSTDB .stf shape {values.shape}: {path}")
    return values


def _nearest_stf_values(
    stf: np.ndarray,
    *,
    sample: int,
    lead_index: int,
) -> dict[str, float]:
    row_index = int(np.argmin(np.abs(stf[:, 0] - float(sample))))
    row = stf[row_index]
    column = 1 + 3 * int(lead_index)
    scale_mv = 0.005  # 20 .stf units = 100 microvolts
    return {
        "stf_sample": int(round(float(row[0]))),
        "stf_level_mv": float(row[column]) * scale_mv,
        "stf_reference_mv": float(row[column + 1]) * scale_mv,
        "stf_deviation_mv": float(row[column + 2]) * scale_mv,
    }


def _ltst_control_centers(
    *,
    stf: np.ndarray,
    n_sig: int,
    sig_len: int,
    fs: float,
    episodes: Sequence[STEpisode],
    count: int,
    window_sec: float,
) -> list[int]:
    if count <= 0:
        return []
    margin = int(round((window_sec / 2.0 + 1.0) * fs))
    deviation_columns = [3 + 3 * lead for lead in range(n_sig)]
    candidates = []
    for row in stf:
        sample = int(round(float(row[0])))
        if sample < margin or sample >= sig_len - margin:
            continue
        deviations_mv = np.abs(row[deviation_columns] * 0.005)
        if np.any(deviations_mv >= 0.025):
            continue
        if any(
            int(event.start_sample) - int(30 * fs)
            <= sample
            <= int(event.end_sample or event.start_sample) + int(30 * fs)
            for event in episodes
        ):
            continue
        candidates.append(sample)
    return _select_evenly(candidates, count)


def _analyse_ltstdb_record(
    record_path: Path,
    *,
    args: argparse.Namespace,
    plot_state: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import wfdb

    header = wfdb.rdheader(str(record_path))
    episodes = parse_ltstdb_episodes(record_path, args.ltst_annotation)
    selected = _select_evenly(episodes, args.max_events_per_record)
    stf = _load_stf(record_path.with_suffix(".stf"))
    event_rows: list[dict[str, Any]] = []
    for event in selected:
        if event.lead_index >= int(header.n_sig) or event.peak_sample is None:
            continue
        window = extract_sparse_window(
            record_path,
            center_sample=event.peak_sample,
            window_sec=args.window_sec,
            mains_freq=args.mains_freq,
        )
        current = window["channels"][event.lead_index]
        truth = _nearest_stf_values(
            stf,
            sample=event.peak_sample,
            lead_index=event.lead_index,
        )
        row = {
            **_event_base_row(
                event,
                lead_name=str(header.sig_name[event.lead_index]),
                annotation_extension=args.ltst_annotation,
            ),
            **truth,
            **_public_channel_row(current),
            "window_start_sample": window["start_sample"],
            "window_end_sample": window["end_sample"],
            "deviation_reference_source": "expert_stf_reference",
        }
        for method in ("native", "hybrid"):
            predicted_level = _finite(
                row.get(f"{method}_evaluable_protocol_st_mv")
            )
            stf_level = _finite(row.get("stf_level_mv"))
            stf_reference = _finite(row.get("stf_reference_mv"))
            peak_truth = _finite(row.get("peak_deviation_mv"))
            row[f"{method}_level_error_mv"] = (
                predicted_level - stf_level
                if predicted_level is not None and stf_level is not None
                else None
            )
            predicted_deviation = (
                predicted_level - stf_reference
                if predicted_level is not None and stf_reference is not None
                else None
            )
            row[f"{method}_predicted_deviation_mv"] = predicted_deviation
            row[f"{method}_deviation_error_vs_peak_mv"] = (
                predicted_deviation - peak_truth
                if predicted_deviation is not None and peak_truth is not None
                else None
            )
            stf_deviation = _finite(row.get("stf_deviation_mv"))
            row[f"{method}_deviation_error_vs_stf_mv"] = (
                predicted_deviation - stf_deviation
                if predicted_deviation is not None and stf_deviation is not None
                else None
            )
        _add_event_duration(row, float(header.fs))
        event_rows.append(row)
        if plot_state["ltstdb"] < args.plot_examples:
            _plot_event_example(
                out_dir=args.out_dir,
                dataset="ltstdb",
                record_path=record_path,
                event=event,
                window=window,
                row=row,
            )
            plot_state["ltstdb"] += 1

    control_rows: list[dict[str, Any]] = []
    centers = _ltst_control_centers(
        stf=stf,
        n_sig=int(header.n_sig),
        sig_len=int(header.sig_len),
        fs=float(header.fs),
        episodes=episodes,
        count=args.controls_per_record,
        window_sec=args.window_sec,
    )
    for control_index, center in enumerate(centers):
        window = extract_sparse_window(
            record_path,
            center_sample=center,
            window_sec=args.window_sec,
            mains_freq=args.mains_freq,
        )
        for lead_index in range(int(header.n_sig)):
            current = window["channels"][lead_index]
            truth = _nearest_stf_values(
                stf,
                sample=center,
                lead_index=lead_index,
            )
            row = {
                "dataset": "ltstdb",
                "record": record_path.name,
                "control_index": control_index,
                "center_sample": center,
                "lead_index": lead_index,
                "lead_name": str(header.sig_name[lead_index]),
                "truth_control_definition": (
                    "all_leads_abs_stf_deviation_below_0.025mV_and_outside_events"
                ),
                **truth,
                **_public_channel_row(current),
                "deviation_reference_source": "expert_stf_reference",
            }
            for method in ("native", "hybrid"):
                predicted_level = _finite(
                    row.get(f"{method}_evaluable_protocol_st_mv")
                )
                stf_reference = _finite(row.get("stf_reference_mv"))
                row[f"{method}_predicted_deviation_mv"] = (
                    predicted_level - stf_reference
                    if predicted_level is not None and stf_reference is not None
                    else None
                )
            control_rows.append(row)
    return event_rows, control_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen and not key.startswith("_"):
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _regression_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    dataset: str,
    cohort: str,
    comparison: str,
    method: str,
    truth_key: str,
    prediction_key: str,
) -> dict[str, Any]:
    pairs = [
        (_finite(row.get(truth_key)), _finite(row.get(prediction_key)))
        for row in rows
    ]
    clean = [(truth, pred) for truth, pred in pairs if truth is not None and pred is not None]
    if not clean:
        return {
            "dataset": dataset,
            "cohort": cohort,
            "comparison": comparison,
            "method": method,
            "truth_key": truth_key,
            "prediction_key": prediction_key,
            "n": 0,
        }
    truth = np.asarray([pair[0] for pair in clean], dtype=float)
    predicted = np.asarray([pair[1] for pair in clean], dtype=float)
    error = predicted - truth
    correlation = (
        float(np.corrcoef(truth, predicted)[0, 1])
        if len(clean) >= 2
        and float(np.std(truth)) > 0
        and float(np.std(predicted)) > 0
        else None
    )
    nonzero = np.abs(truth) >= 0.05
    direction_accuracy = (
        float(np.mean(np.sign(truth[nonzero]) == np.sign(predicted[nonzero])))
        if np.any(nonzero)
        else None
    )
    return {
        "dataset": dataset,
        "cohort": cohort,
        "comparison": comparison,
        "method": method,
        "truth_key": truth_key,
        "prediction_key": prediction_key,
        "n": len(clean),
        "mae_mv": float(np.mean(np.abs(error))),
        "rmse_mv": float(np.sqrt(np.mean(error**2))),
        "bias_mv": float(np.mean(error)),
        "median_error_mv": float(np.median(error)),
        "within_0p05_mv": float(np.mean(np.abs(error) <= 0.05)),
        "pearson_r": correlation,
        "direction_accuracy": direction_accuracy,
    }


def _classification_metrics(
    event_rows: Sequence[dict[str, Any]],
    control_rows: Sequence[dict[str, Any]],
    *,
    dataset: str,
    method: str,
) -> dict[str, Any]:
    values: list[tuple[bool, bool]] = []
    for row in event_rows:
        truth = _finite(
            row.get("peak_deviation_mv")
            if dataset == "edb"
            else row.get("stf_deviation_mv")
        )
        predicted = _finite(row.get(f"{method}_predicted_deviation_mv"))
        if truth is not None and predicted is not None:
            values.append(
                (
                    abs(truth) >= PRIMARY_ST_THRESHOLD_MV,
                    abs(predicted) >= PRIMARY_ST_THRESHOLD_MV,
                )
            )
    for row in control_rows:
        truth = (
            0.0
            if dataset == "edb"
            else _finite(row.get("stf_deviation_mv"))
        )
        predicted = _finite(row.get(f"{method}_predicted_deviation_mv"))
        if truth is not None and predicted is not None:
            values.append(
                (
                    abs(truth) >= PRIMARY_ST_THRESHOLD_MV,
                    abs(predicted) >= PRIMARY_ST_THRESHOLD_MV,
                )
            )
    tp = sum(truth and pred for truth, pred in values)
    fp = sum(not truth and pred for truth, pred in values)
    fn = sum(truth and not pred for truth, pred in values)
    tn = sum(not truth and not pred for truth, pred in values)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity > 0
        else None
    )
    return {
        "dataset": dataset,
        "cohort": "selected_event_peaks_plus_controls",
        "comparison": "absolute_ST_deviation_threshold_0p10mV",
        "method": method,
        "n": len(values),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def build_metrics(
    *,
    edb_events: Sequence[dict[str, Any]],
    edb_controls: Sequence[dict[str, Any]],
    ltst_events: Sequence[dict[str, Any]],
    ltst_controls: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for method in ("native", "hybrid"):
        metrics.append(
            _regression_metrics(
                edb_events,
                dataset="edb",
                cohort="event_peaks",
                comparison="peak_ST_deviation",
                method=method,
                truth_key="peak_deviation_mv",
                prediction_key=f"{method}_predicted_deviation_mv",
            )
        )
        metrics.append(
            _classification_metrics(
                edb_events,
                edb_controls,
                dataset="edb",
                method=method,
            )
        )
        ltst_all = [*ltst_events, *ltst_controls]
        metrics.append(
            _regression_metrics(
                ltst_all,
                dataset="ltstdb",
                cohort="event_peaks_plus_controls",
                comparison="absolute_ST_level",
                method=method,
                truth_key="stf_level_mv",
                prediction_key=f"{method}_evaluable_protocol_st_mv",
            )
        )
        metrics.append(
            _regression_metrics(
                ltst_events,
                dataset="ltstdb",
                cohort="event_peaks",
                comparison="ST_deviation_using_expert_reference",
                method=method,
                truth_key="stf_deviation_mv",
                prediction_key=f"{method}_predicted_deviation_mv",
            )
        )
        metrics.append(
            _classification_metrics(
                ltst_events,
                ltst_controls,
                dataset="ltstdb",
                method=method,
            )
        )
    return metrics


def _plot_event_example(
    *,
    out_dir: Path,
    dataset: str,
    record_path: Path,
    event: STEpisode,
    window: dict[str, Any],
    row: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lead_index = event.lead_index
    signal = np.asarray(window["signals"][lead_index], dtype=float)
    display_signal = signal - float(np.nanmedian(signal))
    fs = float(window["fs"])
    time = np.arange(signal.size, dtype=float) / fs
    channel = window["channels"][lead_index]
    markers = channel.get("_markers", [])
    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(time, display_signal, color="black", linewidth=0.75, label="ECG")
    first_native = True
    first_hybrid = True
    for marker in markers:
        native = marker.get("native_st80_index")
        if native is not None and 0 <= int(native) < signal.size:
            index = int(native)
            axis.scatter(
                index / fs,
                display_signal[index],
                marker="x",
                color="#7f7f7f",
                s=26,
                label="native J+80" if first_native else None,
                zorder=3,
            )
            first_native = False
        hybrid = marker.get("hybrid_st80_index")
        if hybrid is not None and 0 <= int(hybrid) < signal.size:
            index = int(hybrid)
            color = (
                "#d62728"
                if marker.get("hybrid_sparse_usable")
                else "#ff9896"
            )
            axis.scatter(
                index / fs,
                display_signal[index],
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=30,
                label="hybrid J+80" if first_hybrid else None,
                zorder=3,
            )
            first_hybrid = False
    peak_time = (
        int(event.peak_sample) - int(window["start_sample"])
    ) / fs
    axis.axvline(
        peak_time,
        color="#2166ac",
        linestyle="--",
        linewidth=1.2,
        label="expert event peak",
    )
    truth = _finite(row.get("peak_deviation_mv"))
    native = _finite(row.get("native_predicted_deviation_mv"))
    hybrid = _finite(row.get("hybrid_predicted_deviation_mv"))
    axis.set_title(
        f"{dataset.upper()} {record_path.name} | "
        f"{row.get('lead_name')} | {event.kind} | "
        f"truth={_fmt_mv(truth)}, native={_fmt_mv(native)}, "
        f"hybrid={_fmt_mv(hybrid)}"
    )
    axis.set_xlabel("Window time (s)")
    axis.set_ylabel("Amplitude (mV; window median removed)")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper right", ncol=4, fontsize=8)
    figure.tight_layout()
    plot_dir = out_dir / "plots" / "examples"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        plot_dir
        / (
            f"{dataset}_{record_path.name}_lead{lead_index}_"
            f"sample{event.peak_sample}.png"
        ),
        dpi=150,
    )
    plt.close(figure)


def _plot_scatter(
    *,
    out_dir: Path,
    dataset: str,
    rows: Sequence[dict[str, Any]],
    truth_key: str,
) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 7))
    colors = {"native": "#7f7f7f", "hybrid": "#d62728"}
    available: list[float] = []
    for method in ("native", "hybrid"):
        points = [
            (_finite(row.get(truth_key)), _finite(row.get(f"{method}_predicted_deviation_mv")))
            for row in rows
        ]
        clean = [(truth, pred) for truth, pred in points if truth is not None and pred is not None]
        if not clean:
            continue
        truth = np.asarray([item[0] for item in clean], dtype=float)
        predicted = np.asarray([item[1] for item in clean], dtype=float)
        available.extend(truth.tolist())
        available.extend(predicted.tolist())
        axis.scatter(
            truth,
            predicted,
            s=25,
            alpha=0.72,
            color=colors[method],
            label=f"{method} (n={len(clean)})",
        )
    if not available:
        plt.close(figure)
        return
    bound = max(0.15, float(np.percentile(np.abs(available), 99)) * 1.10)
    axis.plot([-bound, bound], [-bound, bound], color="black", linestyle="--")
    axis.axhline(PRIMARY_ST_THRESHOLD_MV, color="#bdbdbd", linewidth=0.8)
    axis.axhline(-PRIMARY_ST_THRESHOLD_MV, color="#bdbdbd", linewidth=0.8)
    axis.axvline(PRIMARY_ST_THRESHOLD_MV, color="#bdbdbd", linewidth=0.8)
    axis.axvline(-PRIMARY_ST_THRESHOLD_MV, color="#bdbdbd", linewidth=0.8)
    axis.set_xlim(-bound, bound)
    axis.set_ylim(-bound, bound)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Expert ST deviation (mV)")
    axis.set_ylabel("ecgfeat predicted ST deviation (mV)")
    axis.set_title(f"{dataset.upper()} event-peak ST deviation")
    axis.grid(alpha=0.18)
    axis.legend()
    figure.tight_layout()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / f"{dataset}_event_scatter.png", dpi=170)
    plt.close(figure)


def _fmt_mv(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.3f} mV"


def _fmt_metric(value: Any, digits: int = 3) -> str:
    finite = _finite(value)
    return "NA" if finite is None else f"{finite:.{digits}f}"


def _write_results_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
    metrics: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# PhysioNet ST analysis",
        "",
        "This report was generated by the standalone `analyze_physionet_st.py` "
        "adapter. No source file under `feature_extraction/ecgfeat` was changed.",
        "",
        "## Run",
        "",
        f"- Dataset selection: `{args.dataset}`",
        f"- Sparse-lead adapter: `{SPARSE_ADAPTER_NAME}`",
        f"- Analysis window: {args.window_sec:g} s",
        f"- LTSTDB event annotation: `.{args.ltst_annotation}`",
        f"- Maximum events per record: {args.max_events_per_record or 'all'}",
        f"- Controls per record: {args.controls_per_record}",
        "",
        "## Processed data",
        "",
        f"- EDB records processed: {summary['edb_records_processed']}",
        f"- EDB selected event peaks: {summary['edb_event_rows']}",
        f"- EDB control lead-windows: {summary['edb_control_rows']}",
        f"- LTSTDB records processed: {summary['ltstdb_records_processed']}",
        f"- LTSTDB selected event peaks: {summary['ltstdb_event_rows']}",
        f"- LTSTDB control lead-windows: {summary['ltstdb_control_rows']}",
        f"- Failed records: {len(summary['failures'])}",
        "",
        "## Metrics",
        "",
        "| Dataset | Cohort | Comparison | Method | n | MAE (mV) | "
        "Bias (mV) | r | Sensitivity | Specificity | F1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('cohort', '')} | "
            f"{row.get('comparison', '')} | {row.get('method', '')} | "
            f"{row.get('n', 0)} | {_fmt_metric(row.get('mae_mv'))} | "
            f"{_fmt_metric(row.get('bias_mv'))} | "
            f"{_fmt_metric(row.get('pearson_r'))} | "
            f"{_fmt_metric(row.get('sensitivity'))} | "
            f"{_fmt_metric(row.get('specificity'))} | "
            f"{_fmt_metric(row.get('f1'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- EDB and LTSTDB have only two or three physical channels. The "
            "adapter supports per-channel ST measurement but cannot validate "
            "12-lead contiguous-territory diagnostic rules.",
            "- EDB truth is deviation from a subject reference waveform. The "
            "script estimates the ecgfeat reference from the first 30 seconds.",
            "- LTSTDB absolute-level MAE is the cleanest feature-extraction "
            "comparison. Its deviation/event screen uses the expert `.stf` "
            "reference and is therefore not an end-to-end baseline-tracking test.",
            "- A control window outside annotated episodes is not guaranteed to "
            "be perfectly normal; the CSV retains its exact selection definition.",
            "- The 0.10 mV screen is a research comparison threshold, not a "
            "clinical diagnostic conclusion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _record_paths(
    directory: Path,
    *,
    requested: set[str],
    max_records: int,
) -> list[Path]:
    records_file = directory / "RECORDS"
    if records_file.is_file():
        names = [
            line.strip()
            for line in records_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        names = sorted(path.stem for path in directory.glob("*.hea"))
    if requested:
        names = [name for name in names if name in requested]
    if max_records > 0:
        names = names[:max_records]
    return [directory / name for name in names]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=("edb", "ltstdb", "both"),
        default="both",
    )
    parser.add_argument(
        "--edb-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "edb",
    )
    parser.add_argument(
        "--ltstdb-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ltstdb",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "physionet_st_analysis",
    )
    parser.add_argument(
        "--records",
        default="",
        help="Optional comma-separated record names, such as e0103,s20011.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Limit records per selected dataset; 0 means all.",
    )
    parser.add_argument(
        "--max-events-per-record",
        type=int,
        default=3,
        help="Evenly sample this many events per record; 0 means all events.",
    )
    parser.add_argument("--controls-per-record", type=int, default=1)
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--edb-reference-sec", type=float, default=30.0)
    parser.add_argument(
        "--ltst-annotation",
        choices=("sta", "stb", "stc"),
        default="stb",
        help="LTSTDB event protocol; stb matches the EDB 0.10 mV/30 s rule.",
    )
    parser.add_argument("--mains-freq", type=int, choices=(50, 60), default=50)
    parser.add_argument(
        "--plot-examples",
        type=int,
        default=6,
        help="Maximum event waveform plots per selected dataset.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.edb_dir = args.edb_dir.resolve()
    args.ltstdb_dir = args.ltstdb_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.no_plots:
        args.plot_examples = 0
    if args.window_sec < 4.0:
        raise SystemExit("--window-sec must be at least 4 seconds")
    if args.edb_reference_sec < 8.0:
        raise SystemExit("--edb-reference-sec must be at least 8 seconds")
    if args.max_events_per_record < 0 or args.controls_per_record < 0:
        raise SystemExit("event/control limits must be non-negative")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = {
        item.strip()
        for item in str(args.records).split(",")
        if item.strip()
    }
    edb_paths = (
        _record_paths(
            args.edb_dir,
            requested=requested,
            max_records=args.max_records,
        )
        if args.dataset in {"edb", "both"}
        else []
    )
    ltst_paths = (
        _record_paths(
            args.ltstdb_dir,
            requested=requested,
            max_records=args.max_records,
        )
        if args.dataset in {"ltstdb", "both"}
        else []
    )
    if args.dataset in {"edb", "both"} and not edb_paths:
        raise SystemExit(f"no EDB records selected under {args.edb_dir}")
    if args.dataset in {"ltstdb", "both"} and not ltst_paths:
        raise SystemExit(f"no LTSTDB records selected under {args.ltstdb_dir}")

    edb_events: list[dict[str, Any]] = []
    edb_controls: list[dict[str, Any]] = []
    ltst_events: list[dict[str, Any]] = []
    ltst_controls: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    processed = {"edb": 0, "ltstdb": 0}
    plot_state = {"edb": 0, "ltstdb": 0}

    total = len(edb_paths) + len(ltst_paths)
    completed = 0
    for dataset, paths in (("edb", edb_paths), ("ltstdb", ltst_paths)):
        for path in paths:
            try:
                if dataset == "edb":
                    event_rows, control_rows = _analyse_edb_record(
                        path,
                        args=args,
                        plot_state=plot_state,
                    )
                    edb_events.extend(event_rows)
                    edb_controls.extend(control_rows)
                else:
                    event_rows, control_rows = _analyse_ltstdb_record(
                        path,
                        args=args,
                        plot_state=plot_state,
                    )
                    ltst_events.extend(event_rows)
                    ltst_controls.extend(control_rows)
                processed[dataset] += 1
                status = (
                    f"events={len(event_rows)} controls={len(control_rows)}"
                )
            except Exception as exc:
                failures.append(
                    {
                        "dataset": dataset,
                        "record": path.name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                status = f"FAILED: {exc}"
            completed += 1
            print(
                f"[{completed}/{total}] {dataset}:{path.name} {status}",
                flush=True,
            )

    _write_csv(args.out_dir / "edb_event_measurements.csv", edb_events)
    _write_csv(args.out_dir / "edb_control_measurements.csv", edb_controls)
    _write_csv(args.out_dir / "ltstdb_event_measurements.csv", ltst_events)
    _write_csv(args.out_dir / "ltstdb_control_measurements.csv", ltst_controls)
    _write_csv(args.out_dir / "failures.csv", failures)

    metrics = build_metrics(
        edb_events=edb_events,
        edb_controls=edb_controls,
        ltst_events=ltst_events,
        ltst_controls=ltst_controls,
    )
    _write_csv(args.out_dir / "metrics.csv", metrics)
    if not args.no_plots:
        _plot_scatter(
            out_dir=args.out_dir,
            dataset="edb",
            rows=edb_events,
            truth_key="peak_deviation_mv",
        )
        _plot_scatter(
            out_dir=args.out_dir,
            dataset="ltstdb",
            rows=ltst_events,
            truth_key="stf_deviation_mv",
        )

    summary = {
        "configuration": {
            "dataset": args.dataset,
            "edb_dir": str(args.edb_dir),
            "ltstdb_dir": str(args.ltstdb_dir),
            "out_dir": str(args.out_dir),
            "window_sec": args.window_sec,
            "edb_reference_sec": args.edb_reference_sec,
            "ltst_annotation": args.ltst_annotation,
            "max_records": args.max_records,
            "max_events_per_record": args.max_events_per_record,
            "controls_per_record": args.controls_per_record,
            "mains_freq": args.mains_freq,
            "adapter_mode": SPARSE_ADAPTER_NAME,
            "ecgfeat_source_tree_modified": False,
        },
        "edb_records_selected": len(edb_paths),
        "edb_records_processed": processed["edb"],
        "edb_event_rows": len(edb_events),
        "edb_control_rows": len(edb_controls),
        "ltstdb_records_selected": len(ltst_paths),
        "ltstdb_records_processed": processed["ltstdb"],
        "ltstdb_event_rows": len(ltst_events),
        "ltstdb_control_rows": len(ltst_controls),
        "metrics": metrics,
        "failures": failures,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_results_markdown(
        args.out_dir / "RESULTS.md",
        args=args,
        summary=summary,
        metrics=metrics,
    )
    print(f"Results: {args.out_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

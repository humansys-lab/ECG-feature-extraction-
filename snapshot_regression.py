from __future__ import annotations

from math import isfinite
from statistics import fmean
from typing import Iterable, Mapping, Sequence


TRACKED_FIELDS = (
    "p_on",
    "p_off",
    "pr_ms",
    "qrs_on",
    "qrs_off",
    "qrs_ms",
    "t_off",
    "qt_ms",
)

DEFAULT_TOLERANCE = {
    "p_on": 10.0,
    "p_off": 10.0,
    "pr_ms": 10.0,
    "qrs_on": 8.0,
    "qrs_off": 8.0,
    "qrs_ms": 10.0,
    "t_off": 12.0,
    "qt_ms": 12.0,
}


def _row_key(row: Mapping[str, object], key_fields: Sequence[str]) -> tuple[object, ...]:
    return tuple(row.get(field) for field in key_fields)


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def compare_snapshots(
    baseline: Iterable[Mapping[str, object]],
    current: Iterable[Mapping[str, object]],
    *,
    key_fields: Sequence[str] = ("record", "lead", "beat_id"),
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | int | bool]]:
    """Compare aligned ECG measurement rows and flag material field drift."""

    baseline_by_key = {
        _row_key(row, key_fields): row
        for row in baseline
    }
    current_by_key = {
        _row_key(row, key_fields): row
        for row in current
    }
    shared_keys = sorted(
        baseline_by_key.keys() & current_by_key.keys(),
        key=repr,
    )
    thresholds = dict(DEFAULT_TOLERANCE)
    if tolerances:
        thresholds.update({str(key): float(value) for key, value in tolerances.items()})

    result: dict[str, dict[str, float | int | bool]] = {}
    for field in TRACKED_FIELDS:
        changes: list[float] = []
        for key in shared_keys:
            before = _finite_float(baseline_by_key[key].get(field))
            after = _finite_float(current_by_key[key].get(field))
            if before is None or after is None:
                continue
            changes.append(after - before)

        absolute_changes = [abs(value) for value in changes]
        tolerance = float(thresholds.get(field, 0.0))
        result[field] = {
            "n": len(changes),
            "mean_change": fmean(changes) if changes else 0.0,
            "mean_abs_change": fmean(absolute_changes) if absolute_changes else 0.0,
            "max_abs_change": max(absolute_changes, default=0.0),
            "tolerance": tolerance,
            "regression": bool(
                absolute_changes and max(absolute_changes) > tolerance
            ),
        }
    return result

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def binary_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = unavailable = 0
    total = 0
    for row in rows:
        total += 1
        prediction = row.get("predicted")
        if prediction is None:
            unavailable += 1
            continue
        truth = bool(row.get("truth"))
        predicted = bool(prediction)
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    assessed = tp + tn + fp + fn
    return {
        "n": total,
        "assessed": assessed,
        "unavailable": unavailable,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "coverage": _ratio(assessed, total),
        "unavailable_rate": _ratio(unavailable, total),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def stratified_metrics(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return {
        group: binary_metrics(group_rows)
        for group, group_rows in groups.items()
    }

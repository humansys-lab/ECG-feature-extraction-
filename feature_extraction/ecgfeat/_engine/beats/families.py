from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np


@dataclass(frozen=True)
class BeatFamilyAssignment:
    beat_index: int
    family_id: int
    paced: bool = False
    rr_context: str = "unknown"


def _flat_norm(beat: np.ndarray) -> np.ndarray:
    x = np.asarray(beat, dtype=float).ravel()
    if x.size == 0:
        return x
    x = x - float(np.mean(x))
    denom = float(np.linalg.norm(x))
    if denom < 1e-8:
        return np.zeros_like(x)
    return x / denom


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    av = _flat_norm(a)
    bv = _flat_norm(b)
    if av.size == 0 or bv.size == 0 or av.shape != bv.shape:
        return 0.0
    return float(np.dot(av, bv))


def _rr_context(rr_ms: Optional[float], median_rr: Optional[float]) -> str:
    if rr_ms is None or median_rr is None or median_rr <= 0.0:
        return "unknown"
    rr_value = float(rr_ms)
    if not np.isfinite(rr_value) or rr_value <= 0.0:
        return "unknown"
    ratio = rr_value / float(median_rr)
    # Keep a neutral band around the clinical +/-20% boundary so otherwise
    # identical beats do not split families because of tiny RR jitter.
    if ratio < 0.75:
        return "short_rr"
    if ratio > 1.25:
        return "long_rr"
    return "regular_rr"


def assign_morphology_families(
    beat_windows: Sequence[np.ndarray],
    rr_prev_ms: Optional[Sequence[Optional[float]]] = None,
    paced_beat_ids: Optional[Iterable[int]] = None,
    threshold: float = 0.88,
) -> List[BeatFamilyAssignment]:
    """
    Assign morphology families inside an already selected measurement group.

    This is intentionally narrower than grouping.cluster_beats: it is used by
    representative-beat construction to pick a medoid from a local pool. The
    hard partition keys are paced state and coarse RR context; morphology
    correlation then clusters beats within each partition.
    """
    paced_set: Set[int] = {int(beat_id) for beat_id in paced_beat_ids} if paced_beat_ids is not None else set()
    rr_values = list(rr_prev_ms) if rr_prev_ms is not None else []
    finite_rr = [
        float(rr)
        for rr in rr_values
        if rr is not None and np.isfinite(float(rr)) and float(rr) > 0.0
    ]
    median_rr = float(np.median(finite_rr)) if finite_rr else None

    families: List[dict[str, object]] = []
    assignments: List[BeatFamilyAssignment] = []
    for beat_index, beat in enumerate(beat_windows):
        paced = beat_index in paced_set
        rr = rr_values[beat_index] if beat_index < len(rr_values) else None
        context = _rr_context(rr, median_rr)

        best_i: Optional[int] = None
        best_corr = -1.0
        for family_index, family in enumerate(families):
            if bool(family["paced"]) != paced or family["rr_context"] != context:
                continue
            score = _corr(beat, np.asarray(family["prototype"]))
            if score > best_corr:
                best_corr = score
                best_i = family_index

        if best_i is None or best_corr < threshold:
            families.append({
                "prototype": np.asarray(beat, dtype=float).copy(),
                "members": [beat_index],
                "paced": paced,
                "rr_context": context,
            })
            family_id = len(families)
        else:
            family = families[best_i]
            members = family["members"]
            assert isinstance(members, list)
            members.append(beat_index)
            prototype = np.asarray(family["prototype"], dtype=float)
            family["prototype"] = prototype + (np.asarray(beat, dtype=float) - prototype) / len(members)
            family_id = best_i + 1

        assignments.append(BeatFamilyAssignment(
            beat_index=beat_index,
            family_id=family_id,
            paced=paced,
            rr_context=context,
        ))

    return assignments


def select_family_medoid(beat_windows: Sequence[np.ndarray], members: Iterable[int]) -> Optional[int]:
    member_list = [int(member) for member in members]
    if not member_list:
        return None
    if len(member_list) == 1:
        return member_list[0]

    best_member = member_list[0]
    best_score = -1.0
    for member in member_list:
        scores = [
            _corr(beat_windows[member], beat_windows[other])
            for other in member_list
            if other != member
        ]
        score = float(np.mean(scores)) if scores else 1.0
        if score > best_score:
            best_score = score
            best_member = member
    return best_member

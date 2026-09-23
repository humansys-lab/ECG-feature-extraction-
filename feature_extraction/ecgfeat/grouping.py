from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .models import BeatAnnotation


@dataclass
class BeatCluster:
    members:   List[int]
    prototype: np.ndarray
    is_wide:   bool = False   # T008: narrow vs wide QRS flag


# ── Local vector magnitude (avoids circular import with qrs.py) ──────────────

def _zscore(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return (x - np.mean(x)) / (s + 1e-8)


def _local_vm(ecg: np.ndarray) -> np.ndarray:
    """I, II, V1-V6 vector magnitude."""
    leads = [i for i in (0, 1, 6, 7, 8, 9, 10, 11) if i < ecg.shape[0]]
    return np.sqrt(np.mean(np.vstack([_zscore(ecg[i]) for i in leads]) ** 2, axis=0))


# ── T007: beat descriptor helpers ────────────────────────────────────────────

def _estimate_qrs_width_ms(vm: np.ndarray, r: int, fs: int) -> float:
    """Rough QRS width from vector magnitude energy (T007)."""
    energy = vm ** 2
    r = int(np.clip(r, 0, len(energy) - 1))
    thr = max(0.15 * energy[r], 1e-8)
    left = r
    while left > 0 and energy[left] > thr:
        left -= 1
    right = r
    while right < len(energy) - 1 and energy[right] > thr:
        right += 1
    return (right - left) * 1000.0 / fs


def beat_template_vector(
    ecg: np.ndarray,
    r: int,
    fs: int,
    qrs_win_ms: Tuple[int, int] = (-80, 120),
) -> np.ndarray:
    lo = max(0, int(r + qrs_win_ms[0] * fs / 1000))
    hi = min(ecg.shape[1], int(r + qrs_win_ms[1] * fs / 1000))
    seg = ecg[:, lo:hi]
    if seg.shape[1] == 0:
        return np.zeros(ecg.shape[0] * 32)
    target = 32
    xs = []
    for lead in seg:
        xi = np.interp(np.linspace(0, len(lead) - 1, target), np.arange(len(lead)), lead)
        xi = xi - np.mean(xi)
        denom = np.linalg.norm(xi)
        if denom > 0:
            xi = xi / denom
        xs.append(xi)
    return np.concatenate(xs)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── T008: two-stage clusterer ────────────────────────────────────────────────

def _morphology_cluster(
    beat_indices: List[int],
    templates: List[np.ndarray],
    max_groups: int,
    threshold: float = 0.88,
    preserve_outliers: bool = False,
) -> List[BeatCluster]:
    """Online morphology clustering within one narrow/wide bin."""
    clusters: List[BeatCluster] = []
    for beat_idx in beat_indices:
        vec = templates[beat_idx]
        if not clusters:
            clusters.append(BeatCluster(members=[beat_idx], prototype=vec))
            continue
        scores = [_corr(vec, c.prototype) for c in clusters]
        best_i = int(np.argmax(scores))
        if preserve_outliers and len(clusters) >= max_groups and scores[best_i] < 0.65:
            # A distinct singleton is explicit evidence, not contamination of
            # the dominant template. max_groups remains the regular budget.
            clusters.append(BeatCluster(members=[beat_idx], prototype=vec))
        elif scores[best_i] >= threshold or len(clusters) >= max_groups:
            c = clusters[best_i]
            c.members.append(beat_idx)
            c.prototype = 0.8 * c.prototype + 0.2 * vec
        else:
            clusters.append(BeatCluster(members=[beat_idx], prototype=vec))
    return clusters


def cluster_beats(
    ecg: np.ndarray,
    r_locs: np.ndarray,
    fs: int,
    max_groups: int = 5,
    wide_qrs_threshold_ms: float = 120.0,
    paced_beat_ids: List[int] | None = None,
    preserve_outliers: bool = False,
) -> Dict[int, List[int]]:
    """
    Measurement-group beat grouper (T008).

    This is the early pipeline grouping used to choose which beat family drives
    global measurements. It is deliberately different from
    family_representative.assign_morphology_families, which runs later inside a
    selected group only to choose a representative medoid.

    Stage 1 — rule-based split by rough QRS width (T007):
        narrow  < wide_qrs_threshold_ms
        wide   >= wide_qrs_threshold_ms

    Stage 2 — morphology clustering within each bin:
        Online prototype clustering (cosine similarity >= 0.88).

    Returns {group_id: [beat_idx, ...]} sorted by size descending.
    Group 1 is always the dominant group.
    """
    if len(r_locs) == 0:
        return {}

    vm = _local_vm(ecg)

    # Pre-compute templates and rough QRS widths
    templates: List[np.ndarray] = []
    widths_ms: List[float] = []
    for r in r_locs:
        templates.append(beat_template_vector(ecg, int(r), fs))
        widths_ms.append(_estimate_qrs_width_ms(vm, int(r), fs))

    # Stage 1: split into narrow / wide
    narrow_idx = [i for i, w in enumerate(widths_ms) if w < wide_qrs_threshold_ms]
    wide_idx   = [i for i, w in enumerate(widths_ms) if w >= wide_qrs_threshold_ms]

    # Allocate group budget (wide bin gets at most 2)
    n_wide   = min(2, max_groups - 1) if wide_idx else 0
    n_narrow = max(1, max_groups - n_wide)

    all_clusters: List[BeatCluster] = []

    if narrow_idx:
        for c in _morphology_cluster(narrow_idx, templates, n_narrow, preserve_outliers=preserve_outliers):
            all_clusters.append(c)

    if wide_idx:
        for c in _morphology_cluster(wide_idx, templates, n_wide, preserve_outliers=preserve_outliers):
            c.is_wide = True
            all_clusters.append(c)

    if not all_clusters:
        return {}

    # Sort by count → Group 1 = dominant
    all_clusters.sort(key=lambda c: len(c.members), reverse=True)
    groups = {gid: c.members for gid, c in enumerate(all_clusters, start=1)}

    paced_set = {int(beat_id) for beat_id in (paced_beat_ids or [])}
    if not paced_set:
        return groups

    paced_members = sorted(beat_id for beat_id in paced_set if 0 <= beat_id < len(r_locs))
    if not paced_members:
        return groups

    regrouped = [
        [beat_id for beat_id in members if beat_id not in paced_set]
        for members in groups.values()
    ]
    regrouped = [members for members in regrouped if members]
    regrouped.append(paced_members)
    regrouped.sort(key=len, reverse=True)
    return {gid: members for gid, members in enumerate(regrouped, start=1)}


# ── Beat annotation builder ───────────────────────────────────────────────────

def build_beat_annotations(
    r_locs: np.ndarray,
    fs: int,
    groups: Dict[int, List[int]],
    paced_beat_ids: List[int] | None = None,
) -> List[BeatAnnotation]:
    beat_to_group: Dict[int, int] = {m: gid for gid, members in groups.items() for m in members}
    paced_set = set(paced_beat_ids or [])
    ann: List[BeatAnnotation] = []
    for i, r in enumerate(r_locs):
        rr_prev = None if i == 0 else (r - r_locs[i - 1]) * 1000.0 / fs
        rr_next = None if i == len(r_locs) - 1 else (r_locs[i + 1] - r) * 1000.0 / fs
        ann.append(BeatAnnotation(
            beat_id=i,
            r_index=int(r),
            paced=i in paced_set,
            group_id=int(beat_to_group.get(i, 1)),
            rr_prev_ms=rr_prev,
            rr_next_ms=rr_next,
        ))
    return ann

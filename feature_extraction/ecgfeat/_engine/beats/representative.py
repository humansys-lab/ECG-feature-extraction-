from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import correlate

from .families import assign_morphology_families, select_family_medoid


@dataclass
class RepBeatMeta:
    """Metadata about representative beat construction (T009)."""
    group_id:          int
    member_count:      int
    outlier_count:     int
    mean_template_corr: float
    used_count:        int   # members actually averaged (after outlier removal)


def _extract_window(ecg: np.ndarray, center: int, left: int, right: int) -> np.ndarray:
    lo = max(0, center - left)
    hi = min(ecg.shape[1], center + right)
    out = np.zeros((ecg.shape[0], left + right), dtype=float)
    seg = ecg[:, lo:hi]
    dst_lo = left - (center - lo)
    dst_hi = dst_lo + seg.shape[1]
    out[:, dst_lo:dst_hi] = seg
    return out


def _vm_1d(beat: np.ndarray) -> np.ndarray:
    """Vector magnitude of a [12, N] beat snippet."""
    return np.sqrt(np.mean(beat ** 2, axis=0))


def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _align_beat(
    beat: np.ndarray,
    ref_vm: np.ndarray,
    max_lag_ms: int,
    fs: int,
    bounded_search: bool = False,
) -> np.ndarray:
    """Cross-correlation alignment, lag capped at max_lag_ms.

    Uses zero-padded shift instead of np.roll to avoid wrap-around
    contamination at beat window edges when |lag| is non-trivial.
    """
    vm = _vm_1d(beat)
    c = correlate(vm - np.mean(vm), ref_vm - np.mean(ref_vm), mode="full")
    if bounded_search:
        radius = int(max_lag_ms * fs / 1000)
        center = len(vm) - 1
        lag = int(np.argmax(c[max(0, center - radius):center + radius + 1]) - min(radius, center))
    else:
        lag = int(np.argmax(c) - (len(vm) - 1))
    lag = int(np.clip(lag, -int(max_lag_ms * fs / 1000), int(max_lag_ms * fs / 1000)))
    if lag == 0:
        return beat.copy()
    n = beat.shape[1]
    shifted = np.zeros_like(beat)
    if lag > 0:
        shifted[:, :n - lag] = beat[:, lag:]
    else:
        shifted[:, -lag:] = beat[:, :n + lag]
    return shifted


def build_representative_beats(
    ecg: np.ndarray,
    r_locs: np.ndarray,
    groups: Dict[int, List[int]],
    fs: int,
    left_ms: int = 300,
    right_ms: int = 500,
    outlier_corr_threshold: float = 0.70,
    max_lag_ms: int = 30,
) -> Dict[int, np.ndarray]:
    """
    Build one representative beat per group (T009 enhanced).

    Improvements over the original:
    - Outlier exclusion: beats with template_corr < outlier_corr_threshold
      vs the reference are excluded from the average.
    - Fallback: if all beats are outliers, keep all (avoids empty group).
    - Cross-correlation alignment within ±max_lag_ms (unchanged).
    """
    left  = int(left_ms  * fs / 1000)
    right = int(right_ms * fs / 1000)
    reps: Dict[int, np.ndarray] = {}

    for gid, members in groups.items():
        if not members:
            continue
        beats = [_extract_window(ecg, int(r_locs[m]), left, right) for m in members]
        ref     = beats[0]
        ref_vm  = _vm_1d(ref)

        # Step 1: align all beats to reference
        aligned = [ref]
        for b in beats[1:]:
            aligned.append(_align_beat(b, ref_vm, max_lag_ms, fs))

        # Step 2 (T009): outlier exclusion by template correlation
        corrs = [_corr_1d(_vm_1d(b), ref_vm) for b in aligned]
        clean = [b for b, c in zip(aligned, corrs) if c >= outlier_corr_threshold]
        if not clean:
            clean = aligned   # fallback: keep all

        # Step 3: per-sample median beat (12SL). Median suppresses outlier noise
        # better than the mean: e.g. [0,5,10,15,100] -> median 10 vs mean 26.
        reps[gid] = np.median(np.stack(clean, axis=0), axis=0)

    return reps


def build_representative_beats_with_meta(
    ecg: np.ndarray,
    r_locs: np.ndarray,
    groups: Dict[int, List[int]],
    fs: int,
    left_ms: int = 300,
    right_ms: int = 500,
    outlier_corr_threshold: float = 0.70,
    max_lag_ms: int = 30,
    paced_beat_ids: List[int] | set[int] | None = None,
    robust_alignment: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, RepBeatMeta]]:
    """
    Same as build_representative_beats but also returns per-group metadata (T009).
    """
    left  = int(left_ms  * fs / 1000)
    if robust_alignment and len(r_locs) > 1:
        # Extend slow-rhythm templates without crossing the typical next beat.
        right_ms = max(right_ms, min(800, int(.70 * np.median(np.diff(r_locs)) * 1000 / fs)))
    right = int(right_ms * fs / 1000)
    reps:  Dict[int, np.ndarray]  = {}
    metas: Dict[int, RepBeatMeta] = {}

    for gid, members in groups.items():
        if not members:
            continue

        # For irregular rhythms (AF, arrhythmia) filter members to beats whose
        # RR interval is within ±20 % of the group median before averaging.
        # Rate-aberrant beats have different morphology (rate-dependent BBB,
        # altered ST-T) and pollute the template when included.
        # Criterion: RR coefficient of variation > 0.15 (clearly irregular).
        # Fallback: if fewer than 3 beats survive, revert to all members.
        active_members = list(members)
        if len(members) >= 6:
            beat_rrs: List[Optional[float]] = []
            for m in members:
                m_int = int(m)
                if 0 < m_int < len(r_locs) - 1:
                    rr = (float(r_locs[m_int]) - float(r_locs[m_int - 1])) * 1000.0 / fs
                    beat_rrs.append(rr)
                else:
                    beat_rrs.append(None)
            valid_rrs = [rr for rr in beat_rrs if rr is not None]
            if len(valid_rrs) >= 4:
                median_rr = float(np.median(valid_rrs))
                rr_cv = float(np.std(valid_rrs) / (median_rr + 1e-6))
                if rr_cv > 0.15:
                    filtered = [
                        m for m, rr in zip(members, beat_rrs)
                        if rr is not None
                        and abs(rr - median_rr) / (median_rr + 1e-6) <= 0.20
                    ]
                    # Only apply when filtering genuinely removes ≥ 10 % of beats.
                    # Sinus tachycardia with mild RR variability (CV 0.15–0.20) loses
                    # almost no beats through the ±20 % gate, so the fraction condition
                    # blocks filtering there while still allowing it for true AF where
                    # many outlier beats are removed.
                    if len(filtered) >= 3 and len(filtered) < int(0.90 * len(members)):
                        active_members = filtered

        beats  = [_extract_window(ecg, int(r_locs[m]), left, right) for m in active_members]
        eligible = [i for i, member in enumerate(active_members)
                    if int(r_locs[member]) >= left and int(r_locs[member]) + right <= ecg.shape[1]]
        reference_index = (select_family_medoid(beats, eligible or list(range(len(beats))))
                           if robust_alignment else 0)
        ref    = beats[int(reference_index or 0)]
        ref_vm = _vm_1d(ref)

        if robust_alignment:
            aligned = [_align_beat(b, ref_vm, max_lag_ms, fs, bounded_search=True) for b in beats]
        else:
            aligned = [ref]
            for b in beats[1:]:
                aligned.append(_align_beat(b, ref_vm, max_lag_ms, fs))

        rr_prev_ms = [
            None if member <= 0 else (
                float(r_locs[member]) - float(r_locs[member - 1])
            ) * 1000.0 / fs
            for member in active_members
        ]
        paced_set = {int(beat_id) for beat_id in paced_beat_ids} if paced_beat_ids is not None else set()
        local_paced_ids = {
            idx for idx, member in enumerate(active_members)
            if int(member) in paced_set
        }
        assignments = assign_morphology_families(
            aligned,
            rr_prev_ms=rr_prev_ms,
            paced_beat_ids=local_paced_ids,
        )
        family_counts: Dict[int, int] = {}
        for assignment in assignments:
            family_counts[assignment.family_id] = family_counts.get(assignment.family_id, 0) + 1
        dominant_family = max(family_counts, key=family_counts.get) if family_counts else 1
        family_local_indices = [
            assignment.beat_index
            for assignment in assignments
            if assignment.family_id == dominant_family
        ]
        preliminary_medoid = select_family_medoid(aligned, family_local_indices)
        family_ref_vm = (
            _vm_1d(aligned[int(preliminary_medoid)])
            if preliminary_medoid is not None
            else ref_vm
        )
        corrs = [_corr_1d(_vm_1d(b), family_ref_vm) for b in aligned]
        clean_indices = [
            idx for idx in family_local_indices
            if corrs[idx] >= outlier_corr_threshold
        ]
        n_out = len(aligned) - len(clean_indices)
        if not clean_indices:
            clean_indices = list(family_local_indices or range(len(aligned)))
            n_out = len(aligned) - len(clean_indices)
        medoid_local = select_family_medoid(aligned, clean_indices)
        # Boundary delineation uses the family medoid because regression
        # testing shows that point-wise medians can shift low-slope QRS/P/T
        # boundaries in paced and wide-complex records. The companion
        # build_representative_beats() path remains a true point-wise median
        # and pooled lead measurements use robust medians downstream.
        representative = (
            aligned[int(medoid_local)].copy()
            if medoid_local is not None
            else np.median(
                np.stack([aligned[idx] for idx in clean_indices], axis=0),
                axis=0,
            )
        )

        reps[gid] = representative
        metas[gid] = RepBeatMeta(
            group_id=gid,
            member_count=len(members),
            outlier_count=n_out,
            mean_template_corr=float(np.mean(corrs)),
            used_count=1 if medoid_local is not None else len(clean_indices),
        )

    return reps, metas

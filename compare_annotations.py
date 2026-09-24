#!/usr/bin/env python3
"""
Visual comparison of LUDB expert annotations vs algorithm detections.

For each beat the plot shows the ECG signal with two sets of markers:
    ▐ Expert (GT)   — solid lines / filled markers
    ▐ Algorithm     — dashed lines / open markers

Usage:
    python compare_annotations.py 3               # record 3, lead II, all beats
    python compare_annotations.py 3 --lead V2     # specific lead
    python compare_annotations.py 3 --beat 2      # single beat zoomed in
    python compare_annotations.py 3 --all-leads   # 12-lead grid for one beat
    python compare_annotations.py 3 --out fig.png # save to file instead of show
    python compare_annotations.py 3 --reports --out-dir out/
    python compare_annotations.py --batch-reports --limit 5 --out-dir out/
    python compare_annotations.py --random5       # 5 random records, one PNG each
    python compare_annotations.py --random5 --lead V5  # specific lead, 5 random
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# np.trapezoid was added in numpy 2.0; np.trapz is the legacy name (removed in 2.1+).
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "feature_extraction"))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import wfdb

from demo_feature_extraction import generate_report
from ecgfeat.compat.api_v0 import ECGFeatureExtractor
from ecgfeat.delineate import (
    _apply_multilead_consensus,
    _beat_quality,
    _classify_st_morphology,
    _fqrs_score,
    _ptf_v1,
    _qrs_components,
    _qrs_points,
)
from ecgfeat.features import (
    _select_reliable_qt_leads,
    build_representative_lead_features,
    compute_global_features,
    compute_group_features,
)
from ecgfeat.compat.export_v0 import build_morphology_inputs, to_dict
from ecgfeat.grouping import build_beat_annotations
from ecginterpret.interpret import interpret
from ecgfeat.models import (
    ECGFeatures,
    GlobalFeatures,
    LeadBeatFeatures,
    PatientMeta,
    RepresentativeLeadFeatures,
    STANDARD_12_LEADS,
    WaveBounds,
)
from ecgfeat.preprocess import analysis_signal, resample_ecg
from ecgfeat.quality import compute_quality, detect_limb_lead_reversal
from evaluate_ludb import ANN_EXT_TO_LEAD, LEAD_TO_ANN_EXT, MATCH_TOLERANCE_MS, parse_ludb_annotations

_LUDB_CANDIDATES = [
    PROJECT_ROOT / "data" / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
    PROJECT_ROOT / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
]
LUDB_DIR = next((path for path in _LUDB_CANDIDATES if path.exists()), _LUDB_CANDIDATES[0])
REPORT_INTERNAL_FS = 500
REPORT_MAINS_FREQ = 50

GLOBAL_METRIC_FIELDS = [
    ("HR", "heart_rate_bpm"),
    ("PR", "pr_ms"),
    ("QRS", "qrs_ms"),
    ("QT", "qt_ms"),
    ("QTcB", "qtc_bazett_ms"),
    ("QTcF", "qtc_fridericia_ms"),
    ("P Axis", "p_axis_deg"),
    ("QRS Axis", "qrs_axis_deg"),
    ("T Axis", "t_axis_deg"),
    ("QT Dispersion", "qt_dispersion_ms"),
]

PROFILE_GLOBAL_METRIC_FIELDS = [
    ("HR", "heart_rate_bpm"),
    ("PR", "pr_ms"),
    ("QRS", "qrs_duration_ms"),
    ("QT", "qt_ms"),
    ("QTcB", "qtc_bazett_ms"),
    ("QTcF", "qtc_fridericia_ms"),
    ("P Axis", "p_axis_frontal_deg"),
    ("QRS Axis", "qrs_axis_frontal_deg"),
    ("T Axis", "t_axis_frontal_deg"),
    ("QT Dispersion", "qt_dispersion_ms"),
]

MEASUREMENT_PROFILE_CHOICES = ("native", "robust", "12sl", "twelve_sl", "hybrid")

LEAD_METRIC_FIELDS = [
    ("PR", "pr_ms"),
    ("QRS", "qrs_ms"),
    ("QT", "qt_ms"),
    ("QTcB", "qtc_bazett_ms"),
    ("R", "r_amp_mv"),
    ("T", "t_amp_mv"),
    ("ST-J", "st_on_mv"),
]

# ── Colour scheme ─────────────────────────────────────────────────────────────
# Each wave has one hue; GT = saturated/solid, DET = lighter/dashed
GT_STYLE  = dict(lw=1.5,  ls="-",  alpha=0.90, zorder=4)
DET_STYLE = dict(lw=1.5,  ls="--", alpha=0.85, zorder=4)

COLOURS = {
    "qrs":   ("#1a6e1a", "#5bc45b"),   # (GT green, DET green-light)
    "t":     ("#c0392b", "#e87c72"),   # (GT red,   DET salmon)
    "p":     ("#1a4fa0", "#6aabf7"),   # (GT blue,  DET sky-blue)
    "r":     ("#333333", "#999999"),   # R-peak markers
}


# ── Report-comparison helpers ────────────────────────────────────────────────

def _record_sort_key(record_id: str):
    return (0, int(record_id)) if record_id.isdigit() else (1, record_id)


def discover_record_ids() -> List[str]:
    return sorted({path.stem for path in LUDB_DIR.glob("*.hea")}, key=_record_sort_key)


def parse_ludb_header(record_id: str) -> Dict[str, object]:
    hea_path = LUDB_DIR / f"{record_id}.hea"
    lines = hea_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty header file: {hea_path}")

    first = lines[0].split()
    header: Dict[str, object] = {
        "record": record_id,
        "fs": int(first[2]),
        "n_samples": int(first[3]),
        "age": None,
        "sex": None,
        "dx": [],
        "rx": "Unknown",
        "hx": "dataset=LUDB",
        "sx": "annotation-compare",
    }

    diagnoses: List[str] = []
    in_diagnosis_block = False
    for raw_line in lines[1:]:
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("#<age>:"):
            age_text = line.split(":", 1)[1].strip()
            header["age"] = int(age_text) if age_text.isdigit() else None
            in_diagnosis_block = False
        elif lower.startswith("#<sex>:"):
            header["sex"] = line.split(":", 1)[1].strip() or None
            in_diagnosis_block = False
        elif lower.startswith("#<diagnoses>:"):
            in_diagnosis_block = True
        elif in_diagnosis_block and line.startswith("#"):
            diag = line[1:].strip()
            if diag:
                diagnoses.append(diag.rstrip("."))
        else:
            in_diagnosis_block = False

    if diagnoses:
        header["dx"] = diagnoses
        header["hx"] = f"{header['hx']}; diagnoses={' | '.join(diagnoses)}"

    return header


def _with_report_source(header: Dict[str, object], source_label: str) -> Dict[str, object]:
    out = dict(header)
    hx = str(out.get("hx", "Unknown"))
    out["hx"] = f"{hx}; feature_source={source_label}"
    out["sx"] = source_label
    return out


def choose_reference_gt_lead(gt_by_lead: Dict[str, List[Dict]], preferred: str = "II") -> str:
    if gt_by_lead.get(preferred):
        return preferred
    available = [lead for lead in STANDARD_12_LEADS if gt_by_lead.get(lead)]
    if not available:
        raise ValueError("No LUDB annotations available to choose a reference lead")
    return max(available, key=lambda lead: (len(gt_by_lead[lead]), -STANDARD_12_LEADS.index(lead)))


def _scale_sample(sample: Optional[int], scale: float) -> Optional[int]:
    if sample is None:
        return None
    return int(round(sample * scale))


def _scale_annotation_beats(beats: Sequence[Dict], scale: float) -> List[Dict]:
    if abs(scale - 1.0) < 1e-9:
        return [dict(beat) for beat in beats]
    out: List[Dict] = []
    for beat in beats:
        out.append(
            {
                key: (_scale_sample(value, scale) if isinstance(value, int) or value is None else value)
                for key, value in beat.items()
            }
        )
    return out


def load_all_gt_annotations(record_id: str, fs_in: int, fs_out: int) -> Dict[str, List[Dict]]:
    scale = float(fs_out) / float(fs_in)
    gt_by_lead: Dict[str, List[Dict]] = {}
    for lead in STANDARD_12_LEADS:
        beats = get_gt_for_lead(record_id, lead)
        if beats:
            gt_by_lead[lead] = _scale_annotation_beats(beats, scale)
    return gt_by_lead


def _match_canonical_beats(
    canonical_beats: Sequence[Dict],
    lead_beats: Sequence[Dict],
    tolerance_samples: int,
) -> List[Optional[Dict]]:
    if not lead_beats:
        return [None for _ in canonical_beats]

    matched: List[Optional[Dict]] = []
    used: set[int] = set()
    lead_rs = np.array([beat["r_sample"] for beat in lead_beats], dtype=int)
    for canonical in canonical_beats:
        dists = np.abs(lead_rs - int(canonical["r_sample"]))
        order = np.argsort(dists)
        choice: Optional[Dict] = None
        for idx in order:
            idx_int = int(idx)
            if idx_int in used:
                continue
            if int(dists[idx_int]) <= tolerance_samples:
                choice = dict(lead_beats[idx_int])
                used.add(idx_int)
            break
        matched.append(choice)
    return matched


def _median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.median(finite))


_HEXAXIAL_ANGLES = {
    "I": 0.0,
    "II": 60.0,
    "III": 120.0,
    "aVR": -150.0,
    "aVL": -30.0,
    "aVF": 90.0,
}


def _finite_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _representative_param(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    lead: str,
    key: str,
) -> Optional[float]:
    rep = representative_leads.get(lead)
    if rep is None:
        return None
    return _finite_or_none(rep.params.get(key))


def _axis_from_annotation_values(values: Dict[str, Optional[float]]) -> Optional[float]:
    xs: List[float] = []
    ys: List[float] = []
    for lead, angle_deg in _HEXAXIAL_ANGLES.items():
        value = _finite_or_none(values.get(lead))
        if value is None:
            continue
        angle = np.deg2rad(angle_deg)
        xs.append(float(value) * float(np.cos(angle)))
        ys.append(float(value) * float(np.sin(angle)))

    if not xs:
        return None
    x = float(np.sum(xs))
    y = float(np.sum(ys))
    if abs(x) + abs(y) < 1e-9:
        return None
    return float(np.rad2deg(np.arctan2(y, x)))


def _annotation_signed_area_axis_values(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    area_key: str,
    amp_key: str,
    signed_area_key: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {}
    for lead in _HEXAXIAL_ANGLES:
        signed = _representative_param(representative_leads, lead, signed_area_key or "")
        if signed is not None:
            values[lead] = signed
            continue

        area = _representative_param(representative_leads, lead, area_key)
        amp = _representative_param(representative_leads, lead, amp_key)
        if area is None or amp is None or abs(amp) < 1e-9:
            values[lead] = None
        else:
            values[lead] = float(np.sign(amp) * area)
    return values


def _annotation_qrs_net_axis_values(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {}
    for lead in _HEXAXIAL_ANGLES:
        q_amp = _representative_param(representative_leads, lead, "q_amp_mv")
        r_amp = _representative_param(representative_leads, lead, "r_amp_mv")
        s_amp = _representative_param(representative_leads, lead, "s_amp_mv")
        parts = [value for value in (q_amp, r_amp, s_amp) if value is not None]
        values[lead] = float(np.sum(parts)) if parts else None
    return values


def _compute_annotation_global_features(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    beat_features: List[LeadBeatFeatures],
    r_locs: np.ndarray,
    fs: int,
) -> GlobalFeatures:
    rr_ms = np.diff(r_locs) * 1000.0 / fs if len(r_locs) > 1 else np.array([])
    rr_sec = float(np.median(rr_ms) / 1000.0) if len(rr_ms) else None
    hr = float(60000.0 / np.median(rr_ms)) if len(rr_ms) else None

    pr = _median_or_none(rep.params.get("pr_ms") for rep in representative_leads.values())
    qrs = _median_or_none(rep.params.get("qrs_ms") for rep in representative_leads.values())
    qt = _median_or_none(rep.params.get("qt_ms") for rep in representative_leads.values())
    p_duration_source = "representative_multilead_consensus"
    p_duration_pairs = [
        (lead, value)
        for lead, rep in representative_leads.items()
        if (value := _finite_or_none(rep.params.get("p_dur_consensus_ms"))) is not None
    ]
    if p_duration_pairs and any(
        representative_leads[lead].params.get("p_duration_guard_source")
        for lead, _ in p_duration_pairs
    ):
        p_duration_source = (
            "representative_multilead_consensus_paired_duration_guard"
        )
    if not p_duration_pairs:
        p_duration_source = "reliable_lead_median_fallback"
        p_duration_pairs = [
            (lead, value)
            for lead, rep in representative_leads.items()
            if (value := _finite_or_none(rep.params.get("p_dur_ms"))) is not None
        ]
    p_duration = _median_or_none(value for _, value in p_duration_pairs)
    p_duration_used_leads = [lead for lead, _ in p_duration_pairs]
    p_duration_support = len(p_duration_pairs)
    p_duration_spread = (
        float(
            np.max([value for _, value in p_duration_pairs])
            - np.min([value for _, value in p_duration_pairs])
        )
        if p_duration_pairs
        else None
    )
    if p_duration is None:
        p_duration_source = None
        p_duration_reliability = "unavailable"
    elif p_duration_source == "representative_multilead_consensus":
        p_duration_reliability = "reliable" if p_duration_support >= 3 else "low_support"
    else:
        p_duration_reliability = "fallback" if p_duration_support >= 2 else "low_support"
    qt_values = [
        value
        for rep in representative_leads.values()
        if (value := _finite_or_none(rep.params.get("qt_ms"))) is not None
    ]
    qt_dispersion = float(np.max(qt_values) - np.min(qt_values)) if len(qt_values) >= 2 else None
    qtc_b = float(qt / np.sqrt(rr_sec)) if qt is not None and rr_sec and rr_sec > 0 else None
    qtc_f = float(qt / np.cbrt(rr_sec)) if qt is not None and rr_sec and rr_sec > 0 else None

    p_axis = _axis_from_annotation_values(
        _annotation_signed_area_axis_values(
            representative_leads,
            "p_area",
            "p_amp_mv",
            signed_area_key="p_signed_area",
        )
    )
    qrs_axis = _axis_from_annotation_values(_annotation_qrs_net_axis_values(representative_leads))
    t_axis = _axis_from_annotation_values(
        _annotation_signed_area_axis_values(
            representative_leads,
            "t_area",
            "t_amp_mv",
            signed_area_key="t_signed_area",
        )
    )
    st_axis = _axis_from_annotation_values(
        {lead: _representative_param(representative_leads, lead, "st_mid_mv") for lead in _HEXAXIAL_ANGLES}
    )

    ptf_v1 = _representative_param(representative_leads, "V1", "ptf_v1_mv_ms")

    return GlobalFeatures(
        heart_rate_bpm=hr,
        atrial_rate_bpm=hr,
        pr_ms=pr,
        qrs_ms=qrs,
        qt_ms=qt,
        qtc_bazett_ms=qtc_b,
        qtc_fridericia_ms=qtc_f,
        p_axis_deg=p_axis,
        qrs_axis_deg=qrs_axis,
        t_axis_deg=t_axis,
        st_axis_deg=st_axis,
        qt_dispersion_ms=qt_dispersion,
        p_duration_ms=p_duration,
        p_duration_source=p_duration_source,
        p_duration_used_leads=p_duration_used_leads,
        p_duration_support=p_duration_support,
        p_duration_spread_ms=p_duration_spread,
        p_duration_reliability=p_duration_reliability,
        ptf_v1_mv_ms=ptf_v1,
    )


def _annotation_flags(
    qrs_on: Optional[int],
    qrs_off: Optional[int],
    p_peak: Optional[int],
    t_peak: Optional[int],
    beat_reliable: bool,
) -> List[str]:
    flags: List[str] = []
    if p_peak is None:
        flags.append("p_unreliable")
    if t_peak is None:
        flags.append("t_unreliable")
    if qrs_on is None or qrs_off is None:
        flags.append("qrs_unreliable")
    if not beat_reliable:
        flags.append("beat_unreliable")
    return flags


def _promote_quality_with_annotations(
    quality: Dict[str, object],
    gt_by_lead: Dict[str, Sequence[Dict]],
) -> None:
    for lead, beats in gt_by_lead.items():
        if not beats or lead not in quality:
            continue
        lead_quality = quality[lead]
        has_p = any(beat.get("p_peak") is not None for beat in beats)
        has_qrs = any(beat.get("qrs_on") is not None and beat.get("qrs_off") is not None for beat in beats)
        has_t = any(beat.get("t_peak") is not None for beat in beats)
        has_qt = any(beat.get("qrs_on") is not None and beat.get("t_off") is not None for beat in beats)
        lead_quality.reliable = True
        lead_quality.reliable_for_p = has_p
        lead_quality.reliable_for_qrs = has_qrs
        lead_quality.reliable_for_t = has_t
        lead_quality.reliable_for_qt = has_qt


def _build_annotation_lead_feature(
    sig: np.ndarray,
    fs: int,
    lead: str,
    beat_id: int,
    canonical_r: int,
    annotation: Optional[Dict],
    p_annotation: Optional[Dict] = None,
) -> LeadBeatFeatures:
    r_index = int(annotation["r_sample"]) if annotation and annotation.get("r_sample") is not None else int(canonical_r)
    r_index = int(np.clip(r_index, 0, len(sig) - 1))

    qrs_on = annotation.get("qrs_on") if annotation else None
    qrs_off = annotation.get("qrs_off") if annotation else None
    p_source = p_annotation
    p_on = p_source.get("p_on") if p_source else None
    p_peak = p_source.get("p_peak") if p_source else None
    p_off = p_source.get("p_off") if p_source else None
    t_on = annotation.get("t_on") if annotation else None
    t_peak = annotation.get("t_peak") if annotation else None
    t_off = annotation.get("t_off") if annotation else None

    bl_lo = max(0, r_index - int(0.22 * fs))
    bl_hi = max(bl_lo + 1, r_index - int(0.08 * fs))
    baseline = float(np.median(sig[bl_lo:bl_hi])) if bl_hi > bl_lo else 0.0

    pr_ms = None
    qrs_ms = None
    qt_ms = None
    jt_ms = None
    qrs_area = None
    qrs_signed_area = None
    j_index = None
    st_on_mv = None
    st_mid_mv = None
    st_80ms_mv = None
    p_amp = None
    t_amp = None

    if p_peak is not None:
        p_amp = float(sig[p_peak] - baseline)
    if t_peak is not None:
        t_amp = float(sig[t_peak] - baseline)

    if qrs_on is not None and qrs_off is not None and qrs_off > qrs_on:
        qrs_ms = (qrs_off - qrs_on) * 1000.0 / fs
        qrs_segment = sig[qrs_on : qrs_off + 1] - baseline
        qrs_area = float(_trapezoid(np.abs(qrs_segment)))
        qrs_signed_area = float(_trapezoid(qrs_segment))
        j_index = qrs_off
        st_on_mv = float(sig[qrs_off] - baseline)
        st_mid = min(len(sig) - 1, qrs_off + int(0.04 * fs))
        st_80 = min(len(sig) - 1, qrs_off + int(0.08 * fs))
        st_mid_mv = float(sig[st_mid] - baseline)
        st_80ms_mv = float(sig[st_80] - baseline)

    if p_on is not None and qrs_on is not None:
        pr_ms = (qrs_on - p_on) * 1000.0 / fs
    if qrs_on is not None and t_off is not None:
        qt_ms = (t_off - qrs_on) * 1000.0 / fs
    if qrs_off is not None and t_off is not None:
        jt_ms = (t_off - qrs_off) * 1000.0 / fs

    q_amp, r_amp, s_amp, _ = _qrs_points(sig, qrs_on, r_index, qrs_off)
    r_prime_amp, s_prime_amp, qrs_num_peaks, qrs_notch_count, qrs_component_durations = _qrs_components(
        sig, qrs_on, r_index, qrs_off, baseline, fs
    )
    vat_ms = (r_index - qrs_on) * 1000.0 / fs if qrs_on is not None else None

    p_dur_ms = None
    p_area = None
    p_signed_area = None
    if p_on is not None and p_off is not None and p_off > p_on:
        p_dur_ms = (p_off - p_on) * 1000.0 / fs
        p_segment = sig[p_on : p_off + 1] - baseline
        p_area = float(_trapezoid(np.abs(p_segment)))
        p_signed_area = float(_trapezoid(p_segment))

    t_dur_ms = None
    t_area = None
    t_signed_area = None
    if t_on is not None and t_off is not None and t_off > t_on:
        t_dur_ms = (t_off - t_on) * 1000.0 / fs
        t_segment = sig[t_on : t_off + 1] - baseline
        t_area = float(_trapezoid(np.abs(t_segment)))
        t_signed_area = float(_trapezoid(t_segment))

    t_polarity = 0
    if t_amp is not None:
        if t_amp > 0.05:
            t_polarity = 1
        elif t_amp < -0.05:
            t_polarity = -1

    u_wave_flag = False
    if t_off is not None:
        u_lo = t_off + 1
        u_hi = min(len(sig), t_off + int(0.20 * fs))
        if u_hi > u_lo + 2:
            u_wave_flag = float(np.max(np.abs(sig[u_lo:u_hi] - baseline))) > 0.05

    qrs_slur_flag = qrs_notch_count >= 2

    st_slope_mv_per_ms = None
    if qrs_off is not None:
        j_80 = qrs_off + int(0.08 * fs)
        if j_80 < len(sig):
            st_slope_mv_per_ms = (float(sig[j_80]) - float(sig[qrs_off])) / 80.0

    tpe_ms = (
        (t_off - t_peak) * 1000.0 / fs
        if t_peak is not None and t_off is not None and t_off > t_peak
        else None
    )
    st_morphology = _classify_st_morphology(st_slope_mv_per_ms)
    fqrs_score = _fqrs_score(qrs_notch_count, qrs_num_peaks)
    ptf_v1 = _ptf_v1(sig, p_on, p_off, baseline, fs) if lead == "V1" and p_peak is not None else None

    beat_noise_score, beat_baseline_shift, beat_template_corr, beat_reliable = _beat_quality(
        sig, r_index, qrs_on, qrs_off, baseline, fs, rep_qrs_snippet=None
    )
    p_confidence = 1.0 if p_peak is not None else 0.0
    qrs_confidence = 1.0 if qrs_on is not None and qrs_off is not None else 0.0
    qrs_on_confidence = 1.0 if qrs_on is not None else 0.0
    qrs_off_confidence = 1.0 if qrs_off is not None else 0.0
    qt_confidence = 1.0 if t_off is not None else 0.0
    flags = _annotation_flags(qrs_on, qrs_off, p_peak, t_peak, beat_reliable)

    return LeadBeatFeatures(
        lead=lead,
        beat_id=int(beat_id),
        p=WaveBounds(onset=p_on, peak=p_peak, offset=p_off),
        qrs=WaveBounds(onset=qrs_on, peak=r_index, offset=qrs_off),
        t=WaveBounds(onset=t_on, peak=t_peak, offset=t_off),
        qt_ms=qt_ms,
        pr_ms=pr_ms,
        qrs_ms=qrs_ms,
        p_amp_mv=p_amp,
        qrs_area=qrs_area,
        qrs_signed_area=qrs_signed_area,
        q_amp_mv=q_amp,
        r_amp_mv=r_amp,
        s_amp_mv=s_amp,
        st_on_mv=st_on_mv,
        st_mid_mv=st_mid_mv,
        st_80ms_mv=st_80ms_mv,
        t_amp_mv=t_amp,
        j_index=j_index,
        delta_present=False,
        qrs_notch_sign=0,
        flags=flags,
        jt_ms=jt_ms,
        qt_confidence=qt_confidence,
        t_end_method="annotation" if t_off is not None else "missing",
        beat_noise_score=beat_noise_score,
        beat_baseline_shift=beat_baseline_shift,
        beat_template_corr=beat_template_corr,
        beat_measurement_reliable=beat_reliable,
        p_confidence=p_confidence,
        qrs_confidence=qrs_confidence,
        qrs_on_confidence=qrs_on_confidence,
        qrs_off_confidence=qrs_off_confidence,
        r_prime_amp_mv=r_prime_amp,
        s_prime_amp_mv=s_prime_amp,
        r_duration_ms=qrs_component_durations["r_duration_ms"],
        r_prime_duration_ms=qrs_component_durations["r_prime_duration_ms"],
        s_duration_ms=qrs_component_durations["s_duration_ms"],
        s_prime_duration_ms=qrs_component_durations["s_prime_duration_ms"],
        qrs_num_peaks=qrs_num_peaks,
        qrs_notch_count=qrs_notch_count,
        vat_ms=vat_ms,
        p_dur_ms=p_dur_ms,
        p_area=p_area,
        p_signed_area=p_signed_area,
        t_dur_ms=t_dur_ms,
        t_area=t_area,
        t_signed_area=t_signed_area,
        t_polarity=t_polarity,
        u_wave_flag=u_wave_flag,
        qrs_slur_flag=qrs_slur_flag,
        st_slope_mv_per_ms=st_slope_mv_per_ms,
        tpe_ms=tpe_ms,
        st_morphology=st_morphology,
        fqrs_score=fqrs_score,
        ptf_v1_mv_ms=ptf_v1,
        qt_consensus_ms=None,
        pr_consensus_ms=None,
    )


def build_annotation_derived_result(
    ecg: np.ndarray,
    fs: int,
    gt_by_lead: Dict[str, List[Dict]],
    preferred_lead: str = "II",
    fs_internal: int = REPORT_INTERNAL_FS,
    patient_meta: Optional[PatientMeta] = None,
) -> ECGFeatures:
    ecg_arr = np.asarray(ecg, dtype=float)
    if ecg_arr.ndim != 2:
        raise ValueError("ecg must have shape [12, n_samples]")
    if ecg_arr.shape[0] != 12:
        if ecg_arr.shape[1] == 12:
            ecg_arr = ecg_arr.T
        else:
            raise ValueError("expected 12 leads")

    fs_run = int(fs_internal)
    ecg_rs = resample_ecg(ecg_arr, fs, fs_run)
    ecg_an = analysis_signal(ecg_rs, fs_run, mains_hz=REPORT_MAINS_FREQ)
    quality = compute_quality(ecg_an, fs_run, mains_hz=REPORT_MAINS_FREQ)
    scale = float(fs_run) / float(fs)
    scaled_gt = {lead: _scale_annotation_beats(beats, scale) for lead, beats in gt_by_lead.items()}
    _promote_quality_with_annotations(quality, scaled_gt)

    reference_lead = choose_reference_gt_lead(scaled_gt, preferred=preferred_lead)
    canonical_beats = sorted(scaled_gt[reference_lead], key=lambda beat: int(beat["r_sample"]))
    if not canonical_beats:
        raise ValueError("No canonical LUDB beats available for report generation")

    tolerance_samples = int(round(MATCH_TOLERANCE_MS * fs_run / 1000.0))
    matched_by_lead = {
        lead: _match_canonical_beats(canonical_beats, scaled_gt.get(lead, []), tolerance_samples)
        for lead in STANDARD_12_LEADS
    }

    beat_features: List[LeadBeatFeatures] = []
    for beat_id, canonical in enumerate(canonical_beats):
        canonical_r = int(canonical["r_sample"])
        for lead_idx, lead in enumerate(STANDARD_12_LEADS):
            annotation = matched_by_lead[lead][beat_id]
            p_annotation = matched_by_lead[lead][beat_id - 1] if beat_id > 0 else None
            beat_features.append(
                _build_annotation_lead_feature(
                    ecg_an[lead_idx],
                    fs_run,
                    lead,
                    beat_id,
                    canonical_r,
                    annotation,
                    p_annotation=p_annotation,
                )
            )

    beat_features = _apply_multilead_consensus(beat_features, fs_run, quality)
    r_locs = np.array([int(beat["r_sample"]) for beat in canonical_beats], dtype=int)
    beat_groups = {1: list(range(len(r_locs)))}
    beats = build_beat_annotations(r_locs, fs_run, beat_groups)
    representative_leads = build_representative_lead_features(
        beat_features,
        quality,
        beat_groups=beat_groups,
        representative_beat_features=None,
        dominant_group_id=1,
    )
    groups = compute_group_features(beat_features, beat_groups, len(r_locs), fs_run, r_locs=r_locs)
    global_features = _compute_annotation_global_features(representative_leads, beat_features, r_locs, fs_run)
    reliable_qt_leads = _select_reliable_qt_leads(representative_leads)

    metadata: Dict[str, object] = {
        "input_fs": fs,
        "internal_fs": fs_run,
        "lead_order": STANDARD_12_LEADS,
        "lead_reversal": detect_limb_lead_reversal(ecg_an),
        "n_beats": int(len(r_locs)),
        "reliable_qt_leads": reliable_qt_leads,
        "n_reliable_qt_leads": len(reliable_qt_leads),
        "measurement_source": "ludb_expert_annotations",
        "global_feature_source": "ludb_annotation_only",
        "reference_gt_lead": reference_lead,
    }
    if patient_meta is not None:
        metadata["patient_meta"] = patient_meta

    result = ECGFeatures(
        fs=fs_run,
        quality=quality,
        beats=beats,
        beat_features=beat_features,
        representative_leads=representative_leads,
        groups=groups,
        global_features=global_features,
        metadata=metadata,
    )
    result.interpretation = interpret(result)
    return result


def _result_missing_counts(result: ECGFeatures) -> Dict[str, int]:
    return {
        "total": len(result.beat_features),
        "p_missing": sum(1 for beat in result.beat_features if "p_unreliable" in beat.flags),
        "t_missing": sum(1 for beat in result.beat_features if "t_unreliable" in beat.flags),
        "qrs_missing": sum(1 for beat in result.beat_features if "qrs_unreliable" in beat.flags),
        "beat_unreliable": sum(1 for beat in result.beat_features if not beat.beat_measurement_reliable),
    }


def _normalize_measurement_profile(measurement_profile: str = "native") -> str:
    profile = str(measurement_profile or "native").strip().lower()
    if profile == "robust":
        return "native"
    if profile == "twelve_sl":
        return "12sl"
    if profile in {"native", "12sl", "hybrid"}:
        return profile
    raise ValueError(
        f"Unsupported measurement profile {measurement_profile!r}; "
        f"expected one of {', '.join(MEASUREMENT_PROFILE_CHOICES)}"
    )


def _profile_global_measurements(
    result: ECGFeatures,
    measurement_profile: str,
) -> Dict[str, Optional[float]]:
    profile_name = _normalize_measurement_profile(measurement_profile)
    morphology_inputs = build_morphology_inputs(result)
    profiles = morphology_inputs.get("measurement_profiles", {}).get("available", {})
    profile = profiles.get(profile_name, {})
    global_measurements = (
        profile.get("global_measurements", {})
        if isinstance(profile, dict)
        else {}
    )
    return {
        label: _finite_or_none(global_measurements.get(key))
        for label, key in PROFILE_GLOBAL_METRIC_FIELDS
    }


def _result_global_metrics(
    result: ECGFeatures,
    measurement_profile: str = "native",
) -> Dict[str, Optional[float]]:
    profile_name = _normalize_measurement_profile(measurement_profile)
    if profile_name != "native":
        return _profile_global_measurements(result, profile_name)
    gf = result.global_features
    return {label: getattr(gf, attr) for label, attr in GLOBAL_METRIC_FIELDS}


def build_global_comparison_rows(
    algorithm_metrics: Dict[str, Optional[float]],
    gt_metrics: Dict[str, Optional[float]],
) -> List[Tuple[str, Optional[float], Optional[float], Optional[float]]]:
    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = []
    keys = list(dict.fromkeys(list(algorithm_metrics.keys()) + list(gt_metrics.keys())))
    for key in keys:
        algorithm_value = algorithm_metrics.get(key)
        gt_value = gt_metrics.get(key)
        diff = None
        if algorithm_value is not None and gt_value is not None:
            diff = float(algorithm_value) - float(gt_value)
        rows.append((key, algorithm_value, gt_value, diff))
    return rows


def _axis_circular_diff(
    algorithm_value: Optional[float],
    gt_value: Optional[float],
) -> Optional[float]:
    if algorithm_value is None or gt_value is None:
        return None
    if not np.isfinite(float(algorithm_value)) or not np.isfinite(float(gt_value)):
        return None
    return float(abs((float(algorithm_value) - float(gt_value) + 180.0) % 360.0 - 180.0))


def _representative_qtcb(result: ECGFeatures, lead: str) -> Optional[float]:
    rep = result.representative_leads.get(lead)
    if rep is None:
        return None
    qt_ms = rep.params.get("qt_consensus_ms") or rep.params.get("qt_ms")
    rr_vals = [beat.rr_prev_ms for beat in result.beats if beat.rr_prev_ms is not None]
    if qt_ms is None or not rr_vals:
        return None
    rr_sec = float(np.median(rr_vals) / 1000.0)
    if rr_sec <= 0:
        return None
    return float(qt_ms / np.sqrt(rr_sec))


def _representative_metric(result: ECGFeatures, lead: str, attr: str) -> Optional[float]:
    rep = result.representative_leads.get(lead)
    if rep is None:
        return None
    if attr == "qtc_bazett_ms":
        return _representative_qtcb(result, lead)
    # For QRS and QT, prefer consensus (multilead-fused) values over raw per-lead
    # measurements, which suffer from lead-specific onset/offset detection noise.
    consensus_key = {"qrs_ms": "qrs_consensus_ms", "qt_ms": "qt_consensus_ms"}.get(attr)
    if consensus_key is not None:
        v = rep.params.get(consensus_key)
        if v is not None and np.isfinite(float(v)):
            return float(v)
    value = rep.params.get(attr)
    if value is None:
        return None
    return float(value)


def _build_per_lead_comparison_rows(
    algorithm_result: ECGFeatures,
    gt_result: ECGFeatures,
) -> List[Tuple[str, str, Optional[float], Optional[float], Optional[float]]]:
    rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[float]]] = []
    for lead in STANDARD_12_LEADS:
        for metric_name, attr in LEAD_METRIC_FIELDS:
            algorithm_value = _representative_metric(algorithm_result, lead, attr)
            gt_value = _representative_metric(gt_result, lead, attr)
            diff = None
            if algorithm_value is not None and gt_value is not None:
                diff = algorithm_value - gt_value
            rows.append((lead, metric_name, algorithm_value, gt_value, diff))
    return rows


def _format_metric(value: Optional[float], precision: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def render_comparison_text(
    record_id: str,
    algorithm_result: ECGFeatures,
    gt_result: ECGFeatures,
    measurement_profile: str = "native",
) -> str:
    profile_name = _normalize_measurement_profile(measurement_profile)
    global_rows = build_global_comparison_rows(
        _result_global_metrics(algorithm_result, profile_name),
        _result_global_metrics(gt_result, profile_name),
    )
    lead_rows = _build_per_lead_comparison_rows(algorithm_result, gt_result)
    algorithm_counts = _result_missing_counts(algorithm_result)
    gt_counts = _result_missing_counts(gt_result)

    lines = [
        "=" * 72,
        f"LUDB REPORT COMPARISON  —  Record {record_id}",
        "=" * 72,
        "",
        f"Measurement profile: {profile_name}",
        "",
        "Global Measurements",
        "Metric           Algorithm       GroundTruth     Diff(A-GT)",
        "---------------  --------------  --------------  ------------",
    ]
    for metric, algorithm_value, gt_value, diff in global_rows:
        lines.append(
            f"{metric:<15}  {_format_metric(algorithm_value):<14}  "
            f"{_format_metric(gt_value):<14}  {_format_metric(diff):<12}"
        )

    lines.extend(
        [
            "",
            "Beat / Missing Summary",
            f"Algorithm beats     : {len(algorithm_result.beats)}",
            f"Ground-truth beats  : {len(gt_result.beats)}",
            f"Algorithm missing   : P={algorithm_counts['p_missing']}  T={algorithm_counts['t_missing']}  "
            f"QRS={algorithm_counts['qrs_missing']}  Unreliable={algorithm_counts['beat_unreliable']}",
            f"Ground-truth missing: P={gt_counts['p_missing']}  T={gt_counts['t_missing']}  "
            f"QRS={gt_counts['qrs_missing']}  Unreliable={gt_counts['beat_unreliable']}",
            "",
            "Per-Lead Representative Measurements",
            "Lead  Metric  Algorithm    GroundTruth  Diff(A-GT)",
            "----  ------  -----------  -----------  ----------",
        ]
    )
    for lead, metric, algorithm_value, gt_value, diff in lead_rows:
        lines.append(
            f"{lead:<4}  {metric:<6}  {_format_metric(algorithm_value, 2):<11}  "
            f"{_format_metric(gt_value, 2):<11}  {_format_metric(diff, 2):<10}"
        )

    return "\n".join(lines) + "\n"


def build_batch_summary_row(
    record_id: str,
    algorithm_result: ECGFeatures,
    gt_result: ECGFeatures,
    measurement_profile: str = "native",
) -> Dict[str, object]:
    profile_name = _normalize_measurement_profile(measurement_profile)
    algorithm_record_quality = algorithm_result.metadata.get("record_quality", {})
    algorithm_pacing_state = algorithm_result.metadata.get(
        "measurement_pacing_state",
        algorithm_result.metadata.get("pacing_state"),
    )
    row: Dict[str, object] = {
        "record_id": record_id,
        "algorithm_beats": len(algorithm_result.beats),
        "ground_truth_beats": len(gt_result.beats),
        "reference_gt_lead": gt_result.metadata.get("reference_gt_lead"),
        "record_grade": algorithm_record_quality.get("record_grade"),
        "pacing_state": algorithm_pacing_state,
        "pacing_detection_state": algorithm_result.metadata.get(
            "pacing_detection_state",
            algorithm_result.metadata.get("pacing_state"),
        ),
        "pacing_measurement_effect": algorithm_result.metadata.get("pacing_measurement_effect"),
        "pacing_capture_confirmed": algorithm_result.metadata.get("pacing_capture_confirmed"),
        "measurement_profile": profile_name,
    }
    algorithm_counts = _result_missing_counts(algorithm_result)
    gt_counts = _result_missing_counts(gt_result)
    for key, value in algorithm_counts.items():
        row[f"algorithm_{key}"] = value
    for key, value in gt_counts.items():
        row[f"ground_truth_{key}"] = value

    for metric, algorithm_value, gt_value, diff in build_global_comparison_rows(
        _result_global_metrics(algorithm_result, profile_name),
        _result_global_metrics(gt_result, profile_name),
    ):
        safe_metric = metric.lower().replace(" ", "_")
        row[f"algorithm_{safe_metric}"] = algorithm_value
        row[f"ground_truth_{safe_metric}"] = gt_value
        row[f"diff_{safe_metric}"] = diff
        if safe_metric in {"p_axis", "qrs_axis", "t_axis"}:
            row[f"abs_{safe_metric}_circular_diff"] = _axis_circular_diff(
                algorithm_value,
                gt_value,
            )

    return row


def _write_summary_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_patient_meta(header: Dict[str, object]) -> PatientMeta:
    return PatientMeta(age=header.get("age"), sex=header.get("sex"))


def _pick_visual_beat_index(result: ECGFeatures, requested_beat_idx: int) -> Optional[int]:
    if not result.beats:
        return None
    return max(0, min(int(requested_beat_idx), len(result.beats) - 1))


def write_visual_comparison_bundle(
    record_id: str,
    out_dir: Path,
    lead: str = "II",
    beat_idx: int = 1,
    context_ms: int = 80,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_record_plot_path = out_dir / f"{record_id}_{lead}_compare.png"
    all_leads_full_record_plot_path = out_dir / f"{record_id}_all_leads_compare.png"
    beat_plot_path = out_dir / f"{record_id}_{lead}_beat{beat_idx}_compare.png"
    grid_plot_path = out_dir / f"{record_id}_beat{beat_idx}_all_leads_compare.png"

    plot_single_lead(record_id, lead, str(full_record_plot_path))
    plot_all_leads_full_record(record_id, str(all_leads_full_record_plot_path))
    plot_single_beat(record_id, lead, beat_idx, context_ms, str(beat_plot_path))
    plot_all_leads_one_beat(record_id, beat_idx, context_ms, str(grid_plot_path))

    return {
        "full_record_plot_path": full_record_plot_path,
        "all_leads_full_record_plot_path": all_leads_full_record_plot_path,
        "beat_plot_path": beat_plot_path,
        "grid_plot_path": grid_plot_path,
    }


def generate_report_pair(
    record_id: str,
    out_dir: Path,
    plot_lead: str = "II",
    plot_beat_idx: int = 1,
    plot_context_ms: int = 80,
    include_visuals: bool = True,
    measurement_profile: str = "native",
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ecg, fs = load_record(record_id)
    header = parse_ludb_header(record_id)
    patient_meta = _build_patient_meta(header)

    extractor = ECGFeatureExtractor(fs_internal=REPORT_INTERNAL_FS, mains_freq=REPORT_MAINS_FREQ)
    algorithm_result = extractor.extract(ecg, float(fs), meta=patient_meta)
    gt_by_lead = load_all_gt_annotations(record_id, fs, REPORT_INTERNAL_FS)
    gt_result = build_annotation_derived_result(
        ecg,
        fs,
        gt_by_lead,
        preferred_lead="II",
        fs_internal=REPORT_INTERNAL_FS,
        patient_meta=patient_meta,
    )

    algorithm_report_path = out_dir / f"{record_id}_algorithm_report.txt"
    gt_report_path = out_dir / f"{record_id}_ground_truth_report.txt"
    comparison_path = out_dir / f"{record_id}_comparison.txt"

    generate_report(
        record_id,
        _with_report_source(header, "algorithm"),
        ecg,
        algorithm_result,
        algorithm_report_path,
    )
    generate_report(
        record_id,
        _with_report_source(header, "expert_annotations"),
        ecg,
        gt_result,
        gt_report_path,
    )
    comparison_path.write_text(
        render_comparison_text(
            record_id,
            algorithm_result,
            gt_result,
            measurement_profile=measurement_profile,
        ),
        encoding="utf-8",
    )

    features_json_path = out_dir / f"{record_id}_features.json"
    with features_json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_dict(algorithm_result), handle,
            indent=2, default=str, ensure_ascii=False,
        )

    artifacts: Dict[str, object] = {
        "record_id": record_id,
        "algorithm_result": algorithm_result,
        "ground_truth_result": gt_result,
        "algorithm_report_path": algorithm_report_path,
        "ground_truth_report_path": gt_report_path,
        "comparison_path": comparison_path,
        "features_json_path": features_json_path,
    }

    if include_visuals:
        beat_for_plots = _pick_visual_beat_index(algorithm_result, plot_beat_idx)
        if beat_for_plots is not None:
            artifacts.update(
                write_visual_comparison_bundle(
                    record_id=record_id,
                    out_dir=out_dir,
                    lead=plot_lead,
                    beat_idx=beat_for_plots,
                    context_ms=plot_context_ms,
                )
            )

    return artifacts


def run_single_report_compare(
    record_id: str,
    out_dir: Path,
    plot_lead: str = "II",
    plot_beat_idx: int = 1,
    plot_context_ms: int = 80,
    include_visuals: bool = True,
    measurement_profile: str = "native",
) -> Dict[str, object]:
    artifact_dir = out_dir / str(record_id)
    artifacts = generate_report_pair(
        record_id,
        artifact_dir,
        plot_lead=plot_lead,
        plot_beat_idx=plot_beat_idx,
        plot_context_ms=plot_context_ms,
        include_visuals=include_visuals,
        measurement_profile=measurement_profile,
    )
    print(f"Saved: {artifacts['algorithm_report_path']}")
    print(f"Saved: {artifacts['ground_truth_report_path']}")
    print(f"Saved: {artifacts['comparison_path']}")
    print(f"Saved: {artifacts['features_json_path']}")
    if "full_record_plot_path" in artifacts:
        print(f"Saved: {artifacts['full_record_plot_path']}")
        print(f"Saved: {artifacts['all_leads_full_record_plot_path']}")
        print(f"Saved: {artifacts['beat_plot_path']}")
        print(f"Saved: {artifacts['grid_plot_path']}")
    return artifacts


def _resolve_batch_workers(workers: Optional[int], total_records: int) -> int:
    if total_records <= 1:
        return 1
    requested = os.cpu_count() if workers is None or workers <= 0 else int(workers)
    return max(1, min(int(requested or 1), total_records))


def _run_batch_report_job(
    record_id: str,
    out_dir: Path,
    plot_lead: str,
    plot_beat_idx: int,
    plot_context_ms: int,
    include_visuals: bool,
    measurement_profile: str,
) -> Dict[str, object]:
    artifacts = generate_report_pair(
        record_id,
        out_dir / str(record_id),
        plot_lead=plot_lead,
        plot_beat_idx=plot_beat_idx,
        plot_context_ms=plot_context_ms,
        include_visuals=include_visuals,
        measurement_profile=measurement_profile,
    )
    return {
        "record_id": record_id,
        "summary_row": build_batch_summary_row(
            record_id,
            artifacts["algorithm_result"],
            artifacts["ground_truth_result"],
            measurement_profile=measurement_profile,
        ),
        "comparison_path": artifacts["comparison_path"],
    }


def run_batch_report_compare(
    out_dir: Path,
    record_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    plot_lead: str = "II",
    plot_beat_idx: int = 1,
    plot_context_ms: int = 80,
    include_visuals: bool = True,
    workers: Optional[int] = 0,
    measurement_profile: str = "native",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = list(record_ids) if record_ids else discover_record_ids()
    if limit is not None:
        selected_ids = selected_ids[:limit]

    worker_count = _resolve_batch_workers(workers, len(selected_ids))
    print(f"Processing {len(selected_ids)} record(s) with {worker_count} worker(s).")

    rows_by_index: Dict[int, Dict[str, object]] = {}
    if worker_count == 1:
        for idx, record_id in enumerate(selected_ids):
            print(f"\n── Report compare: {record_id} ──────────────────")
            try:
                result = _run_batch_report_job(
                    record_id,
                    out_dir,
                    plot_lead,
                    plot_beat_idx,
                    plot_context_ms,
                    include_visuals,
                    measurement_profile,
                )
                rows_by_index[idx] = result["summary_row"]
                print(f"  Saved: {result['comparison_path']}")
            except Exception as exc:
                print(f"  SKIP {record_id}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_record = {
                executor.submit(
                    _run_batch_report_job,
                    record_id,
                    out_dir,
                    plot_lead,
                    plot_beat_idx,
                    plot_context_ms,
                    include_visuals,
                    measurement_profile,
                ): (idx, record_id)
                for idx, record_id in enumerate(selected_ids)
            }
            for future in as_completed(future_to_record):
                idx, record_id = future_to_record[future]
                try:
                    result = future.result()
                    rows_by_index[idx] = result["summary_row"]
                    print(f"  DONE {record_id}: {result['comparison_path']}")
                except Exception as exc:
                    print(f"  SKIP {record_id}: {exc}")

    summary_rows = [
        rows_by_index[idx]
        for idx in range(len(selected_ids))
        if idx in rows_by_index
    ]
    if summary_rows:
        summary_path = out_dir / "summary.csv"
        _write_summary_csv(summary_rows, summary_path)
        print(f"\nSaved: {summary_path.resolve()}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_record(record_id: str):
    rec = wfdb.rdrecord(str(LUDB_DIR / record_id))
    ecg = rec.p_signal.T   # [12, N]
    return ecg, rec.fs


# ── Build GT annotation dict ──────────────────────────────────────────────────

def get_gt_for_lead(record_id: str, lead: str) -> List[Dict]:
    ext = LEAD_TO_ANN_EXT.get(lead)
    if ext is None:
        return []
    return parse_ludb_annotations(str(LUDB_DIR / record_id), ext)


# ── Build detected annotation dict ───────────────────────────────────────────

def get_det_for_lead(feat, lead: str) -> List:
    return [bf for bf in feat.beat_features if bf.lead == lead]


# ── Single-lead plot with both annotation sets ────────────────────────────────

def plot_lead_comparison(
    ax,
    sig: np.ndarray,
    fs: int,
    gt_beats: List[Dict],
    det_beats: List,
    lead_name: str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    show_xaxis: bool = True,
    highlight_beat: Optional[int] = None,   # det beat index to highlight
):
    """
    Plot ECG + GT boundaries (solid) + detected boundaries (dashed).
    """
    n = len(sig)
    if t_end is None:
        t_end = n / fs
    i0 = max(0, int(t_start * fs))
    i1 = min(n, int(t_end   * fs) + 1)
    t_ax   = np.arange(i0, i1) / fs
    sig_win = sig[i0:i1]

    # ── Background ──────────────────────────────────────────────────────────
    ax.set_facecolor("#FFFAF8")
    sig_range = np.ptp(sig_win) if len(sig_win) else 1.0
    pad = max(sig_range * 0.30, 0.4)
    ylo = sig_win.min() - pad if len(sig_win) else -1
    yhi = sig_win.max() + pad if len(sig_win) else  1
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(ylo, yhi)

    # ECG paper grid
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which="major", color="#F5C6C6", lw=0.5, zorder=1)
    ax.grid(which="minor", color="#FAE0E0", lw=0.2, zorder=1)

    def _vline(x_s, color, style):
        if x_s is None:
            return
        if t_start <= x_s <= t_end:
            ax.axvline(x_s, color=color, **style)

    def _dot(idx, color, marker="o", ms=5, fill=True):
        if idx is None or not (0 <= idx < n):
            return
        t = idx / fs
        if t_start <= t <= t_end:
            mec = color if not fill else "none"
            mfc = color if fill else "none"
            ax.plot(t, sig[idx], marker, color=color,
                    markersize=ms, markerfacecolor=mfc,
                    markeredgecolor=color, markeredgewidth=1.2, zorder=6)

    # ── GT annotations ──────────────────────────────────────────────────────
    gt_c = {w: COLOURS[w][0] for w in ("qrs", "t", "p")}
    for b in gt_beats:
        # QRS
        _vline(b["qrs_on"]  / fs if b["qrs_on"]  else None, gt_c["qrs"], GT_STYLE)
        _vline(b["qrs_off"] / fs if b["qrs_off"] else None, gt_c["qrs"], GT_STYLE)
        _dot(b["r_sample"], COLOURS["r"][0], "v", ms=7, fill=True)
        # T
        _vline(b["t_on"]  / fs if b["t_on"]  else None, gt_c["t"], GT_STYLE)
        _vline(b["t_off"] / fs if b["t_off"] else None, gt_c["t"], GT_STYLE)
        _dot(b["t_peak"], gt_c["t"], "^", ms=5, fill=True)
        # P
        _vline(b["p_on"]  / fs if b["p_on"]  else None, gt_c["p"], GT_STYLE)
        _vline(b["p_off"] / fs if b["p_off"] else None, gt_c["p"], GT_STYLE)
        _dot(b["p_peak"], gt_c["p"], "o", ms=5, fill=True)

    # ── Detected annotations ─────────────────────────────────────────────────
    det_c = {w: COLOURS[w][1] for w in ("qrs", "t", "p")}
    for i, d in enumerate(det_beats):
        hl = (highlight_beat is not None and d.beat_id == highlight_beat)
        alpha_extra = dict(alpha=1.0) if hl else {}
        # QRS
        _vline(d.qrs.onset  / fs if d.qrs.onset  else None, det_c["qrs"],
               {**DET_STYLE, **alpha_extra})
        _vline(d.qrs.offset / fs if d.qrs.offset else None, det_c["qrs"],
               {**DET_STYLE, **alpha_extra})
        _dot(d.qrs.peak, COLOURS["r"][1], "v", ms=6, fill=False)
        # T
        _vline(d.t.onset  / fs if d.t.onset  else None, det_c["t"],
               {**DET_STYLE, **alpha_extra})
        _vline(d.t.offset / fs if d.t.offset else None, det_c["t"],
               {**DET_STYLE, **alpha_extra})
        _dot(d.t.peak, det_c["t"], "^", ms=5, fill=False)
        # P
        _vline(d.p.onset  / fs if d.p.onset  else None, det_c["p"],
               {**DET_STYLE, **alpha_extra})
        _vline(d.p.offset / fs if d.p.offset else None, det_c["p"],
               {**DET_STYLE, **alpha_extra})
        _dot(d.p.peak, det_c["p"], "o", ms=5, fill=False)

    # ── ECG signal on top ────────────────────────────────────────────────────
    ax.plot(t_ax, sig_win, color="#111111", lw=0.9, zorder=5)

    # ── Lead label ───────────────────────────────────────────────────────────
    ax.text(t_start + (t_end - t_start) * 0.01, yhi - (yhi - ylo) * 0.04,
            lead_name, fontsize=9, fontweight="bold",
            va="top", ha="left", color="#111111", zorder=8)

    ax.tick_params(left=False, labelleft=False)
    if show_xaxis:
        ax.tick_params(bottom=True, labelbottom=True, labelsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)
    else:
        ax.tick_params(bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _legend_handles():
    gt_c  = {w: COLOURS[w][0] for w in ("qrs", "t", "p")}
    det_c = {w: COLOURS[w][1] for w in ("qrs", "t", "p")}
    items = [
        mpatches.Patch(color="none", label="── Expert (GT) ──"),
        plt.Line2D([0],[0], color=gt_c["qrs"], lw=1.5, ls="-",  label="GT QRS bounds"),
        plt.Line2D([0],[0], color=gt_c["t"],   lw=1.5, ls="-",  label="GT T bounds"),
        plt.Line2D([0],[0], color=gt_c["p"],   lw=1.5, ls="-",  label="GT P bounds"),
        plt.Line2D([0],[0], color=COLOURS["r"][0], marker="v", ls="none",
                   markersize=7, label="GT R peak"),
        mpatches.Patch(color="none", label="── Algorithm ──"),
        plt.Line2D([0],[0], color=det_c["qrs"], lw=1.5, ls="--", label="Det QRS bounds"),
        plt.Line2D([0],[0], color=det_c["t"],   lw=1.5, ls="--", label="Det T bounds"),
        plt.Line2D([0],[0], color=det_c["p"],   lw=1.5, ls="--", label="Det P bounds"),
        plt.Line2D([0],[0], color=COLOURS["r"][1], marker="v", ls="none",
                   fillstyle="none", markersize=6, label="Det R peak"),
    ]
    return items


# ── Mode 1: single lead, full recording ──────────────────────────────────────

def plot_single_lead(record_id: str, lead: str, out: Optional[str]) -> None:
    ecg, fs = load_record(record_id)
    li = STANDARD_12_LEADS.index(lead)
    sig = ecg[li]

    extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
    feat = extractor.extract(ecg, float(fs))

    gt_beats  = get_gt_for_lead(record_id, lead)
    det_beats = get_det_for_lead(feat, lead)

    dur = len(sig) / fs
    fig_w = max(20, dur * 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    plot_lead_comparison(ax, sig, fs, gt_beats, det_beats, lead,
                         t_start=0.0, t_end=dur, show_xaxis=True)

    fig.legend(handles=_legend_handles(), loc="upper right",
               fontsize=7.5, ncol=2, framealpha=0.95,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        f"Record {record_id}  —  Lead {lead}  |  "
        f"Solid = Expert GT   Dashed = Algorithm",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, out or f"{record_id}_{lead}_compare.png")


# ── Mode 1b: all 12 leads, full recording (one row per lead) ──────────────────

def plot_all_leads_full_record(record_id: str, out: Optional[str]) -> None:
    ecg, fs = load_record(record_id)
    extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
    feat = extractor.extract(ecg, float(fs))

    dur = len(ecg[0]) / fs
    fig_w = max(20, dur * 2.5)
    n_leads = len(STANDARD_12_LEADS)
    fig, axes = plt.subplots(
        n_leads, 1, figsize=(fig_w, 2.0 * n_leads), sharex=True
    )

    for idx, lead in enumerate(STANDARD_12_LEADS):
        li = STANDARD_12_LEADS.index(lead)
        gt_beats  = get_gt_for_lead(record_id, lead)
        det_beats = get_det_for_lead(feat, lead)
        plot_lead_comparison(
            axes[idx], ecg[li], fs, gt_beats, det_beats, lead,
            t_start=0.0, t_end=dur, show_xaxis=(idx == n_leads - 1),
        )

    fig.legend(handles=_legend_handles(), loc="upper right",
               fontsize=7.5, ncol=2, framealpha=0.95,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(
        f"Record {record_id}  —  All 12 leads, full recording  |  "
        f"Solid = Expert GT   Dashed = Algorithm",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    _save_or_show(fig, out or f"{record_id}_all_leads_compare.png")


# ── Mode 2: single beat, all 12 leads ────────────────────────────────────────

def plot_all_leads_one_beat(
    record_id: str, beat_idx: int, context_ms: int, out: Optional[str]
) -> None:
    ecg, fs = load_record(record_id)
    extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
    feat = extractor.extract(ecg, float(fs))

    # Find R-peak of the chosen beat
    if beat_idx >= len(feat.beats):
        sys.exit(f"Beat index {beat_idx} out of range (record has {len(feat.beats)} beats)")
    r_samp = feat.beats[beat_idx].r_index
    ctx = int(context_ms / 1000 * fs)
    t_start = max(0.0, (r_samp - int(0.35 * fs) - ctx) / fs)
    t_end   = min(len(ecg[0]) / fs, (r_samp + int(0.55 * fs) + ctx) / fs)

    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    for idx, lead in enumerate(STANDARD_12_LEADS):
        li = STANDARD_12_LEADS.index(lead)
        gt_beats  = get_gt_for_lead(record_id, lead)
        det_beats = get_det_for_lead(feat, lead)
        plot_lead_comparison(
            axes_flat[idx], ecg[li], fs,
            gt_beats, det_beats, lead,
            t_start=t_start, t_end=t_end,
            show_xaxis=(idx >= 8),
            highlight_beat=beat_idx,
        )

    fig.legend(handles=_legend_handles(), loc="upper right",
               fontsize=8, ncol=2, framealpha=0.95,
               bbox_to_anchor=(0.99, 0.995))

    # Per-lead QT errors for this beat
    qt_errors = {}
    for lead in STANDARD_12_LEADS:
        det = next((d for d in feat.beat_features
                    if d.lead == lead and d.beat_id == beat_idx), None)
        gt_list = get_gt_for_lead(record_id, lead)
        # find matching GT beat by R proximity
        if det and det.qrs.peak and gt_list:
            dists = [abs(g["r_sample"] - det.qrs.peak) for g in gt_list]
            best = gt_list[int(np.argmin(dists))]
            if min(dists) < int(0.075 * fs) and best["t_off"] and best["qrs_on"] and det.qt_ms:
                qt_gt = (best["t_off"] - best["qrs_on"]) * 1000 / fs
                qt_errors[lead] = det.qt_ms - qt_gt

    err_str = "  ".join(
        f"{l}:{v:+.0f}ms" for l, v in qt_errors.items()
    ) if qt_errors else "N/A"

    fig.suptitle(
        f"Record {record_id}  —  Beat {beat_idx}  (R @ sample {r_samp})\n"
        f"Solid = Expert GT   Dashed = Algorithm   |   QT errors: {err_str}",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, out or f"{record_id}_beat{beat_idx}_all_leads_compare.png")


# ── Mode 3: specific beat, single lead (zoomed) ──────────────────────────────

def plot_single_beat(
    record_id: str, lead: str, beat_idx: int, context_ms: int, out: Optional[str]
) -> None:
    ecg, fs = load_record(record_id)
    li = STANDARD_12_LEADS.index(lead)
    extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
    feat = extractor.extract(ecg, float(fs))

    if beat_idx >= len(feat.beats):
        sys.exit(f"Beat index {beat_idx} out of range")
    r_samp = feat.beats[beat_idx].r_index
    ctx    = int(context_ms / 1000 * fs)
    t_start = max(0.0, (r_samp - int(0.35 * fs) - ctx) / fs)
    t_end   = min(len(ecg[0]) / fs, (r_samp + int(0.55 * fs) + ctx) / fs)

    gt_beats  = get_gt_for_lead(record_id, lead)
    det_beats = get_det_for_lead(feat, lead)

    # Find matched GT beat
    gt_match = None
    if gt_beats:
        dists = [abs(g["r_sample"] - r_samp) for g in gt_beats]
        best_i = int(np.argmin(dists))
        if dists[best_i] < int(0.075 * fs):
            gt_match = gt_beats[best_i]

    det_match = next((d for d in det_beats if d.beat_id == beat_idx), None)

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_lead_comparison(
        ax, ecg[li], fs,
        [gt_match] if gt_match else [], det_beats,
        lead, t_start=t_start, t_end=t_end,
        show_xaxis=True, highlight_beat=beat_idx,
    )

    # Annotation table
    lines = []
    labels = [
        ("QRS onset",  gt_match["qrs_on"]  if gt_match else None,
                       det_match.qrs.onset if det_match else None),
        ("QRS offset", gt_match["qrs_off"] if gt_match else None,
                       det_match.qrs.offset if det_match else None),
        ("T onset",    gt_match["t_on"]    if gt_match else None,
                       det_match.t.onset   if det_match else None),
        ("T offset",   gt_match["t_off"]   if gt_match else None,
                       det_match.t.offset  if det_match else None),
        ("P onset",    gt_match["p_on"]    if gt_match else None,
                       det_match.p.onset   if det_match else None),
        ("P offset",   gt_match["p_off"]   if gt_match else None,
                       det_match.p.offset  if det_match else None),
    ]
    for lbl, gt_v, det_v in labels:
        if gt_v is not None and det_v is not None:
            err = (det_v - gt_v) * 1000 / fs
            lines.append(f"{lbl:12s}  GT={gt_v:5d}  Det={det_v:5d}  err={err:+.0f}ms")
        else:
            lines.append(f"{lbl:12s}  GT={'N/A':5s}  Det={'N/A':5s}")
    info = "\n".join(lines)

    fig.text(0.01, 0.01, info, fontsize=7, va="bottom", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#EEF4FF",
                       edgecolor="#AACCEE", alpha=0.9))

    fig.legend(handles=_legend_handles(), loc="upper right",
               fontsize=7.5, ncol=2, framealpha=0.95,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        f"Record {record_id}  Lead {lead}  Beat {beat_idx}  |  "
        f"Solid = Expert GT   Dashed = Algorithm",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    _save_or_show(fig, out or f"{record_id}_{lead}_beat{beat_idx}_compare.png")


# ── Mode 4: 5 random individuals ─────────────────────────────────────────────

def plot_random5(lead: str = "II", context_ms: int = 80,
                 out_dir: Optional[str] = None) -> None:
    """
    Randomly select 5 records, run the extractor on each, and save one
    comparison PNG per record.  Each PNG shows lead *lead* for the full
    recording with GT (solid) and detected (dashed) boundaries overlaid.
    """
    import random

    # Discover all valid record IDs in the LUDB data directory
    all_ids = sorted(
        {p.stem for p in LUDB_DIR.glob("*.hea")},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    if len(all_ids) < 5:
        raise RuntimeError(f"Only {len(all_ids)} records found in {LUDB_DIR}")

    chosen = random.sample(all_ids, 5)
    print(f"Randomly selected records: {chosen}")

    save_dir = Path(out_dir) if out_dir else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)

    extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)

    for rec_id in chosen:
        print(f"\n── Record {rec_id} ──────────────────")
        try:
            ecg, fs = load_record(rec_id)
            li = STANDARD_12_LEADS.index(lead)
            sig = ecg[li]

            feat = extractor.extract(ecg, float(fs))
            gt_beats  = get_gt_for_lead(rec_id, lead)
            det_beats = get_det_for_lead(feat, lead)

            dur = len(sig) / fs
            fig_w = max(20, dur * 2.5)
            fig, ax = plt.subplots(figsize=(fig_w, 4))

            plot_lead_comparison(ax, sig, fs, gt_beats, det_beats, lead,
                                 t_start=0.0, t_end=dur, show_xaxis=True)

            fig.legend(handles=_legend_handles(), loc="upper right",
                       fontsize=7.5, ncol=2, framealpha=0.95,
                       bbox_to_anchor=(0.99, 0.99))
            fig.suptitle(
                f"Record {rec_id}  —  Lead {lead}  |  "
                f"Solid = Expert GT   Dashed = Algorithm",
                fontsize=10, fontweight="bold",
            )
            plt.tight_layout()
            out_path = save_dir / f"{rec_id}_{lead}_compare.png"
            _save_or_show(fig, str(out_path))

        except Exception as e:
            print(f"  SKIP record {rec_id}: {e}")


# ── Save / show ───────────────────────────────────────────────────────────────

def _save_or_show(fig, path: str) -> None:
    out = Path(path)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Compare LUDB expert annotations vs algorithm detections/reports"
    )
    parser.add_argument("record_id",   nargs="?", default=None,
                        help="LUDB record ID (e.g. 3). Omit when using --random5 or --batch-reports.")
    parser.add_argument("--lead",       default="II",
                        help="Lead to plot (default: II)")
    parser.add_argument("--beat",       type=int, default=None,
                        help="Beat index to zoom into (0-based). "
                             "Omit for full recording.")
    parser.add_argument("--all-leads",  action="store_true",
                        help="Show all 12 leads for --beat (requires --beat)")
    parser.add_argument("--all-leads-full", action="store_true",
                        help="Show all 12 leads over the full recording, one lead per row.")
    parser.add_argument("--context",    type=int, default=80,
                        help="Extra context around beat in ms (default: 80)")
    parser.add_argument("--out",        default=None,
                        help="Output filename or directory (PNG). Auto-named if omitted.")
    parser.add_argument("--out-dir",    default=None,
                        help="Output directory for --reports / --batch-reports.")
    parser.add_argument("--random5",    action="store_true",
                        help="Randomly pick 5 records and generate one PNG each.")
    parser.add_argument("--reports",    action="store_true",
                        help="Generate algorithm report, expert-annotation report, comparison text, and visual comparison plots for one record.")
    parser.add_argument("--batch-reports", action="store_true",
                        help="Generate report comparisons plus visual plots for multiple LUDB records and save summary.csv.")
    parser.add_argument("--records",    nargs="+", default=None,
                        help="Explicit record IDs for --batch-reports.")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Only process the first N records in --batch-reports.")
    parser.add_argument("--workers",    type=int, default=0,
                        help="Parallel workers for --batch-reports. 0=auto, 1=serial.")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Skip visual comparison PNG generation in --reports / --batch-reports.")
    parser.add_argument("--measurement-profile", default="native", choices=MEASUREMENT_PROFILE_CHOICES,
                        help="Global measurement profile for report comparisons and summary.csv. "
                             "native/robust keeps current behavior; hybrid uses 12SL HR/intervals when available.")
    args = parser.parse_args()

    if args.batch_reports:
        run_batch_report_compare(
            out_dir=Path(args.out_dir or "ludb_report_compare_batch"),
            record_ids=args.records,
            limit=args.limit,
            plot_lead=args.lead,
            plot_beat_idx=args.beat if args.beat is not None else 1,
            plot_context_ms=args.context,
            include_visuals=not args.no_visuals,
            workers=args.workers,
            measurement_profile=args.measurement_profile,
        )
        return

    if args.reports:
        if args.record_id is None:
            parser.error("record_id is required when using --reports")
        run_single_report_compare(
            args.record_id,
            Path(args.out_dir or "ludb_report_compare"),
            plot_lead=args.lead,
            plot_beat_idx=args.beat if args.beat is not None else 1,
            plot_context_ms=args.context,
            include_visuals=not args.no_visuals,
            measurement_profile=args.measurement_profile,
        )
        return

    if args.random5:
        plot_random5(lead=args.lead, context_ms=args.context,
                     out_dir=args.out or ".")
        return

    if args.record_id is None:
        parser.error("record_id is required unless --random5 is used")

    if args.all_leads_full:
        plot_all_leads_full_record(args.record_id, args.out)
    elif args.all_leads:
        beat = args.beat if args.beat is not None else 1
        plot_all_leads_one_beat(args.record_id, beat, args.context, args.out)
    elif args.beat is not None:
        plot_single_beat(args.record_id, args.lead, args.beat, args.context, args.out)
    else:
        plot_single_lead(args.record_id, args.lead, args.out)


if __name__ == "__main__":
    main()
#python /home/chtmedgemma/projects/ecg_gemma/compare_annotations.py --random5

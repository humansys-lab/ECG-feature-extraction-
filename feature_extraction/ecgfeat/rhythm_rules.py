from __future__ import annotations

from collections import Counter
from statistics import median, pstdev
from math import isfinite
from typing import Any, Dict, Iterable, List, Optional

from .clinical_rules.config import DEFAULT_DIAGNOSTIC_CONFIG
from .clinical_rules.rhythm import (
    av_dissociation_corroborated,
    disorganized_atrial_activity,
)


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def assess_pacing_evidence_quality(
    *,
    pacing_result: Dict[str, Any],
    spike_times: Iterable[Any],
    qrs_count: int,
    pacing_spike_beat_ids: Iterable[Any],
    spike_offsets_samples: Iterable[Any],
    pacing_failures: Dict[str, Any],
    fs: int,
) -> Dict[str, Any]:
    """Audit whether pacing candidates can safely alter ECG measurements.

    The spike detector and the spike-to-QRS matcher are parts of one
    algorithmic chain, not independent corroboration.  This audit keeps every
    candidate available as evidence while preventing a small cherry-picked
    subset of QRS-adjacent spikes from routing the whole record through the
    paced measurement path.
    """
    spikes = sorted(
        {
            value
            for item in (spike_times or [])
            if (value := _safe_int(item)) is not None
        }
    )
    associated_beats = sorted(
        {
            value
            for item in (pacing_spike_beat_ids or [])
            if (value := _safe_int(item)) is not None and value >= 0
        }
    )
    spike_count = len(spikes)
    qrs_count = max(0, int(qrs_count or 0))
    associated_count = len(associated_beats)
    spikes_per_qrs = (
        float(spike_count) / float(qrs_count)
        if qrs_count
        else None
    )
    qrs_associated_spike_fraction = (
        float(associated_count) / float(spike_count)
        if spike_count
        else 0.0
    )
    associated_beat_fraction = (
        float(associated_count) / float(qrs_count)
        if qrs_count
        else 0.0
    )

    offsets_ms: List[float] = []
    if fs > 0:
        for item in spike_offsets_samples or []:
            offset = _finite_float(item)
            if offset is not None:
                offsets_ms.append(float(offset) * 1000.0 / float(fs))
    absolute_offsets_ms = [abs(value) for value in offsets_ms]
    median_abs_offset_ms = (
        float(median(absolute_offsets_ms))
        if absolute_offsets_ms
        else None
    )
    offset_center_ms = float(median(offsets_ms)) if offsets_ms else None
    offset_mad_ms = (
        float(median(abs(value - offset_center_ms) for value in offsets_ms))
        if offset_center_ms is not None
        else None
    )

    capture_alignment_fraction = _finite_float(
        (pacing_failures or {}).get("capture_alignment_fraction")
    )
    if capture_alignment_fraction is None:
        # Older/mocked producers may only expose the legacy alias.  If neither
        # exists, the strict QRS-neighbourhood support is the safest fallback.
        capture_alignment_fraction = _finite_float(
            (pacing_failures or {}).get("artifact_confidence")
        )
    if capture_alignment_fraction is None:
        capture_alignment_fraction = qrs_associated_spike_fraction

    detector_positive = bool(pacing_result.get("paced", False))
    conflict_reasons: List[str] = []
    if detector_positive and spike_count:
        if spikes_per_qrs is not None and spikes_per_qrs > 2.5:
            conflict_reasons.append("excessive_spike_burden_per_qrs")
        if spike_count >= 4 and capture_alignment_fraction < 0.25:
            conflict_reasons.append("low_global_capture_alignment")
        if (
            spike_count >= 4
            and qrs_associated_spike_fraction < 0.20
            and (
                spikes_per_qrs is None
                or spikes_per_qrs > 1.5
                or capture_alignment_fraction < 0.25
            )
        ):
            conflict_reasons.append("sparse_qrs_proximal_spike_support")

    evidence_conflicted = bool(conflict_reasons)
    supports_measurement_routing = bool(
        detector_positive
        and not evidence_conflicted
        and associated_count >= 2
        and median_abs_offset_ms is not None
        and median_abs_offset_ms <= 55.0
    )
    if not spike_count:
        confidence_state = "off"
    elif evidence_conflicted:
        confidence_state = "conflicted"
    elif supports_measurement_routing:
        confidence_state = "supported"
    else:
        confidence_state = "indeterminate"

    return {
        "confidence_state": confidence_state,
        "evidence_conflicted": evidence_conflicted,
        "conflict_reasons": conflict_reasons,
        "supports_measurement_routing": supports_measurement_routing,
        "detector_positive": detector_positive,
        "spike_count": spike_count,
        "qrs_count": qrs_count,
        "qrs_associated_spike_count": associated_count,
        "spikes_per_qrs": spikes_per_qrs,
        "qrs_associated_spike_fraction": qrs_associated_spike_fraction,
        "associated_beat_fraction": associated_beat_fraction,
        "capture_alignment_fraction": capture_alignment_fraction,
        "median_abs_spike_qrs_offset_ms": median_abs_offset_ms,
        "spike_qrs_offset_mad_ms": offset_mad_ms,
    }


def classify_pacing_context(
    beats: Iterable[Dict[str, Any]],
    atrial_events: Iterable[Dict[str, Any]],
    spike_times: Iterable[Any],
    evidence_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize pacing context for rhythm interpretation policy decisions."""
    beat_rows = list(beats or [])
    spikes = list(spike_times or [])
    quality = dict(evidence_quality or {})
    evidence_conflicted = bool(quality.get("evidence_conflicted", False))

    paced_count = sum(1 for beat in beat_rows if bool(beat.get("paced", False)))
    beat_count = len(beat_rows)
    paced_fraction = float(paced_count) / float(beat_count) if beat_count else 0.0

    ventricular_pacing_present = bool(
        not evidence_conflicted
        and any(
            bool(beat.get("paced", False))
            and (_finite_float(beat.get("qrs_duration_ms")) or 0.0) >= 120.0
            for beat in beat_rows
        )
    )
    # A retrograde P wave is ventricular-to-atrial activation, not evidence of
    # an atrial pacing spike.  Until a spike-to-atrial-event matcher exists,
    # atrial/dual-chamber pacing must remain unavailable rather than inferred
    # from the direction of AV association.
    atrial_pacing_present = False
    continuous_pacing = bool(beat_count and paced_count == beat_count)

    context = {
        "enabled": bool(spikes or paced_count),
        "spike_count": len(spikes),
        "paced_fraction": paced_fraction,
        "continuous_pacing": continuous_pacing,
        "intermittent_pacing": bool(0 < paced_count < beat_count),
        "ventricular_pacing_present": ventricular_pacing_present,
        "atrial_pacing_present": atrial_pacing_present,
        "dual_chamber_pacing_present": bool(ventricular_pacing_present and atrial_pacing_present),
        "suppress_further_rhythm_interpretation": bool(
            not evidence_conflicted
            and continuous_pacing
            and ventricular_pacing_present
        ),
    }
    if quality:
        context.update(quality)
    return context


def detect_pacing_failures(
    *,
    spike_times: List[int],
    qrs_times: List[int],
    fs: int,
) -> Dict[str, object]:
    """Detect repeated capture failure only after a pacing context is established.

    A single sharp edge is common in ordinary ECGs and must not be promoted to
    a device emergency.  The detector therefore requires recurrent spikes,
    at least one captured event, at least two uncaptured events, and no more
    than half of all candidates to be uncaptured.  More ambiguous patterns
    remain available as raw evidence for manual review.
    """
    spikes = sorted({int(spike) for spike in (spike_times or [])})
    qrs_candidates = [int(qrs) for qrs in (qrs_times or [])]
    capture_failures: List[int] = []
    for spike_idx in spikes:
        capture_hi = int(spike_idx + 0.160 * fs)
        if not any(spike_idx <= qrs <= capture_hi for qrs in qrs_candidates):
            capture_failures.append(spike_idx)
    spike_count = len(spikes)
    failure_count = len(capture_failures)
    captured_count = max(0, spike_count - failure_count)
    failure_fraction = (
        float(failure_count) / float(spike_count)
        if spike_count
        else 0.0
    )
    minimum_evidence_met = bool(
        spike_count >= 3
        and captured_count >= 1
        and failure_count >= 2
        and failure_fraction <= 0.50
    )
    return {
        "capture_failure_suspected": minimum_evidence_met,
        "failure_spike_times": capture_failures,
        "spike_count": spike_count,
        "captured_spike_count": captured_count,
        "capture_failure_count": failure_count,
        "capture_failure_fraction": failure_fraction,
        "capture_alignment_fraction": (
            float(captured_count) / float(spike_count)
            if spike_count
            else 0.0
        ),
        "minimum_evidence_met": minimum_evidence_met,
        "sensing_failure_suspected": {
            "available": False,
            "value": None,
            "reason": "sensing_failure_detector_not_implemented",
        },
        "artifact_confidence": (
            float(captured_count) / float(spike_count)
            if spike_count
            else 0.0
        ),
    }


def select_measurement_beat_ids(beats: Iterable[Dict[str, Any]]) -> List[int]:
    """Select beat IDs from the dominant measurable morphology family."""
    beat_rows = list(beats or [])
    if not beat_rows:
        return []

    valid_rows = [
        beat for beat in beat_rows
        if _safe_int(beat.get("beat_id")) is not None
    ]
    if not valid_rows:
        return []
    paced_count = sum(1 for beat in valid_rows if bool(beat.get("paced", False)))
    paced_majority = paced_count > (len(valid_rows) / 2.0)
    if paced_majority:
        candidates = [
            beat for beat in valid_rows
            if bool(beat.get("paced", False))
            and not bool(beat.get("is_ventricular_ectopic", False))
        ]
        if not candidates:
            candidates = [beat for beat in valid_rows if bool(beat.get("paced", False))]
    else:
        candidates = [
            beat for beat in valid_rows
            if not bool(beat.get("paced", False))
            and not bool(beat.get("is_ventricular_ectopic", False))
        ]
        if not candidates:
            candidates = [beat for beat in valid_rows if not bool(beat.get("paced", False))]
    if not candidates:
        candidates = valid_rows

    group_counts = Counter(beat.get("group_id") for beat in candidates)
    dominant_group_id = max(group_counts, key=group_counts.get)
    selected: List[int] = []
    for beat in candidates:
        if beat.get("group_id") != dominant_group_id:
            continue
        beat_id = _safe_int(beat.get("beat_id"))
        if beat_id is not None:
            selected.append(beat_id)
    return selected


def detect_preexcitation(
    short_pr_interval: bool,
    short_pr_segment: bool,
    delta_leads: Iterable[str],
    mean_qrs_duration_ms: Optional[float],
    initial_qrs_axis_deg: Optional[float],
) -> Dict[str, Any]:
    """Summarize WPW/preexcitation rule evidence from measured features."""
    delta_leads_list = sorted({str(lead) for lead in (delta_leads or []) if lead})
    delta_lead_count = len(delta_leads_list)
    qrs_ms = _finite_float(mean_qrs_duration_ms)
    axis_deg = _finite_float(initial_qrs_axis_deg)
    # A short PR segment alone is vulnerable to P-offset/QRS-onset error.  It
    # may replace a measurable short PR only when delta morphology is
    # exceptionally widespread; ordinary calls require an explicit PR <120 ms.
    explicit_short_pr = bool(short_pr_interval)
    strong_multilead_fallback = bool(
        not explicit_short_pr
        and short_pr_segment
        and delta_lead_count >= 8
        and qrs_ms is not None
        and 100.0 <= qrs_ms <= 180.0
    )
    short_pr_evidence = bool(explicit_short_pr or strong_multilead_fallback)
    delta_threshold = 2 if explicit_short_pr else 8
    wpw_pattern = bool(
        short_pr_evidence
        and delta_lead_count >= delta_threshold
        and qrs_ms is not None
        and 100.0 <= qrs_ms <= 180.0
    )

    accessory_pathway_side: Optional[str]
    if not wpw_pattern or axis_deg is None:
        accessory_pathway_side = None
    elif axis_deg < -30.0:
        accessory_pathway_side = "left"
    elif axis_deg > 90.0:
        accessory_pathway_side = "right"
    else:
        accessory_pathway_side = "unknown"

    return {
        "available": True,
        "short_pr_interval": bool(short_pr_interval),
        "short_pr_segment": bool(short_pr_segment),
        "explicit_short_pr": explicit_short_pr,
        "strong_multilead_fallback": strong_multilead_fallback,
        "delta_threshold": delta_threshold,
        "delta_lead_count": delta_lead_count,
        "delta_leads": delta_leads_list,
        "mean_qrs_duration_ms": qrs_ms,
        "initial_qrs_axis_deg": axis_deg,
        "accessory_pathway_side": accessory_pathway_side,
        "wpw_pattern": wpw_pattern,
    }


# Minimum beat-to-beat PR increment that counts as Wenckebach lengthening
# rather than measurement jitter.
MOBITZ_I_MIN_PR_INCREMENT_MS = 10.0
# Maximum spread across the conducted PR intervals bracketing a dropped beat
# for the conduction to be called constant (Mobitz II).
MOBITZ_II_MAX_PR_SPREAD_MS = 20.0


def _classify_second_degree_type(
    preceding_pr_ms: List[float],
    following_pr_ms: List[float],
) -> tuple[str, Dict[str, Any]]:
    """Separate Mobitz I from Mobitz II around a single dropped beat.

    The discriminator is the behaviour of the conducted PR intervals: Mobitz I
    lengthens progressively into the drop, Mobitz II keeps PR constant and
    drops without warning. Mobitz II is the high-risk pattern (infranodal,
    pacemaker indication even when asymptomatic), so it must not stay folded
    into the generic label.
    """
    basis: Dict[str, Any] = {
        "preceding_pr_ms": list(preceding_pr_ms),
        "following_pr_ms": list(following_pr_ms),
        "mobitz_i_min_increment_ms": MOBITZ_I_MIN_PR_INCREMENT_MS,
        "mobitz_ii_max_spread_ms": MOBITZ_II_MAX_PR_SPREAD_MS,
    }
    if (
        len(preceding_pr_ms) >= 2
        and preceding_pr_ms[-1] - preceding_pr_ms[-2]
        >= MOBITZ_I_MIN_PR_INCREMENT_MS
    ):
        basis["discriminator"] = "progressive_pr_lengthening"
        basis["pr_increment_ms"] = preceding_pr_ms[-1] - preceding_pr_ms[-2]
        return "mobitz_i", basis
    # Constant PR is only meaningful when it is observed on both sides of the
    # drop, or across at least two conducted beats before it.
    bracketing = list(preceding_pr_ms) + list(following_pr_ms)
    spans_drop = bool(preceding_pr_ms and following_pr_ms)
    if len(bracketing) >= 2 and (spans_drop or len(preceding_pr_ms) >= 2):
        spread = max(bracketing) - min(bracketing)
        basis["pr_spread_ms"] = spread
        if spread <= MOBITZ_II_MAX_PR_SPREAD_MS:
            basis["discriminator"] = (
                "constant_pr_across_drop" if spans_drop
                else "constant_pr_before_drop"
            )
            return "mobitz_ii", basis
    basis["discriminator"] = "insufficient_pr_evidence"
    return "second_degree_av_block", basis


def detect_pauses_and_av_block(
    rr_ms: Iterable[Optional[float]],
    pr_series_ms: Iterable[Optional[float]],
    atrial_events_per_rr: Iterable[int],
    qrs_duration_ms: Optional[float],
) -> Dict[str, Any]:
    """Detect pause and simple second-degree AV-block evidence."""
    rr_values = [
        float(value)
        for value in (_finite_float(v) for v in (rr_ms or []))
        if value is not None and value > 0.0
    ]
    enough_rr_for_pause = len(rr_values) >= 3
    baseline_rr_ms = float(median(rr_values)) if rr_values else None
    longest_pause_ms = max(rr_values) if rr_values else None
    pause_threshold_ms = baseline_rr_ms * 1.40 if baseline_rr_ms is not None else None
    pauses_detected = bool(
        enough_rr_for_pause
        and longest_pause_ms is not None
        and pause_threshold_ms is not None
        and longest_pause_ms > pause_threshold_ms
    )

    atrial_counts = [
        int(value)
        for value in (atrial_events_per_rr or [])
        if _safe_int(value) is not None
    ]
    pr_values = list(pr_series_ms or [])
    second_degree_avb: Optional[str] = None
    dropped_p_interval_indices: List[int] = []
    atrial_event_excess_indices = [
        idx for idx, count in enumerate(atrial_counts) if count > 1
    ]
    localized_atrial_event_excess = bool(
        atrial_event_excess_indices
        and any(count <= 1 for count in atrial_counts)
    )
    second_degree_basis: Dict[str, Any] = {}
    for idx, pr_value in enumerate(pr_values):
        if pr_value is not None:
            continue
        if idx >= len(atrial_counts) or atrial_counts[idx] <= 1:
            continue
        if not localized_atrial_event_excess:
            continue
        # A true non-conducted P wave should span a prolonged ventricular
        # interval. Extra atrial detections inside a normal RR interval are
        # commonly T-wave/ectopy residuals and are not sufficient evidence.
        if (
            pause_threshold_ms is None
            or idx >= len(rr_values)
            or rr_values[idx] <= pause_threshold_ms
        ):
            continue
        dropped_p_interval_indices.append(idx)
        previous = [
            _finite_float(value)
            for value in pr_values[max(0, idx - 2):idx]
        ]
        previous = [value for value in previous if value is not None]
        following = [
            _finite_float(value)
            for value in pr_values[idx + 1: idx + 3]
        ]
        following = [value for value in following if value is not None]
        second_degree_avb, second_degree_basis = _classify_second_degree_type(
            previous, following
        )
        if second_degree_avb in {"mobitz_i", "mobitz_ii"}:
            break

    if second_degree_avb is None and pause_threshold_ms is not None:
        for idx, count in enumerate(atrial_counts):
            if count <= 1 or idx >= len(rr_values):
                continue
            if not localized_atrial_event_excess:
                continue
            if rr_values[idx] <= pause_threshold_ms:
                continue
            if idx not in dropped_p_interval_indices:
                dropped_p_interval_indices.append(idx)
            second_degree_avb = "second_degree_av_block"
            break

    qrs_ms = _finite_float(qrs_duration_ms)
    if pauses_detected and qrs_ms is not None and qrs_ms >= 120.0:
        escape_origin: Optional[str] = "ventricular"
    elif pauses_detected:
        escape_origin = "supraventricular"
    else:
        escape_origin = None

    # Every ventricular cycle carrying exactly two atrial events is 2:1
    # conduction, where Mobitz I and II are indistinguishable on the surface
    # ECG because no consecutive conducted PR intervals exist to compare.
    two_to_one_conduction = bool(
        len(atrial_counts) >= 3 and all(count == 2 for count in atrial_counts)
    )
    # Wide QRS with constant PR points below the His bundle, which is the
    # combination that carries the pacemaker indication.
    if second_degree_avb == "mobitz_ii":
        block_level_hint = (
            "infranodal_suspected"
            if qrs_ms is not None and qrs_ms >= 120.0
            else "infranodal_probable"
        )
    elif second_degree_avb == "mobitz_i":
        block_level_hint = (
            "av_nodal_probable"
            if qrs_ms is not None and qrs_ms < 120.0
            else "indeterminate"
        )
    elif two_to_one_conduction:
        block_level_hint = (
            "infranodal_suspected"
            if qrs_ms is not None and qrs_ms >= 120.0
            else "indeterminate"
        )
    else:
        block_level_hint = None

    return {
        "available": True,
        "pauses_detected": pauses_detected,
        "pause_longest_ms": round(float(longest_pause_ms), 1) if pauses_detected and longest_pause_ms is not None else None,
        "baseline_rr_ms": baseline_rr_ms,
        "pause_threshold_ms": pause_threshold_ms,
        "second_degree_avb": second_degree_avb,
        "second_degree_avb_basis": second_degree_basis,
        "two_to_one_conduction_suspected": two_to_one_conduction,
        "av_block_level_hint": block_level_hint,
        "av_block_evidence": {
            "atrial_events_per_rr": atrial_counts,
            "pr_series_ms": pr_values,
            "atrial_event_excess_indices": atrial_event_excess_indices,
            "localized_atrial_event_excess": localized_atrial_event_excess,
            "constant_multiple_atrial_events_requires_validation": bool(
                atrial_counts and all(count > 1 for count in atrial_counts)
            ),
            "dropped_p_interval_indices": dropped_p_interval_indices,
            "dropped_p_evidence": bool(dropped_p_interval_indices),
            "requires_prolonged_ventricular_interval": True,
        },
        "atrial_events_per_rr_max": max(atrial_counts) if atrial_counts else None,
        "escape_origin": escape_origin,
    }


def classify_post_pause_or_interpolated_beats(
    beats: List[Dict[str, object]],
    background_rr_ms: Optional[float],
) -> List[Dict[str, object]]:
    """Classify simple escape/interpolated beat candidates from RR context."""
    background_rr = _finite_float(background_rr_ms)
    if background_rr is None or background_rr <= 0.0:
        return []

    events: List[Dict[str, object]] = []
    for beat in beats or []:
        beat_id = _safe_int(_field_value(beat, "beat_id"))
        if beat_id is None:
            continue
        rr_prev = _finite_float(_field_value(beat, "rr_prev_ms"))
        rr_next = _finite_float(_field_value(beat, "rr_next_ms"))
        if rr_prev is not None and rr_prev > 1.4 * background_rr:
            events.append({"beat_id": beat_id, "type": "escape_candidate"})
        if (
            rr_prev is not None
            and rr_next is not None
            and abs((rr_prev + rr_next) - background_rr) <= 0.15 * background_rr
        ):
            events.append({"beat_id": beat_id, "type": "interpolated_candidate"})
    return events


def build_statement_evidence(
    rhythm_summary: Dict[str, Any],
    pacing_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the rule-engine statement evidence contract used by export."""
    primary_statement = rhythm_summary.get("primary_statement")
    additional_statements = list(rhythm_summary.get("additional_statements") or [])
    statements = list(rhythm_summary.get("statements") or [])
    return {
        "available": True,
        "primary_statement": primary_statement,
        "additional_statements": additional_statements,
        "statements": statements,
        "stop_further_interpretation": bool(
            pacing_context.get("suppress_further_rhythm_interpretation", False)
        ),
        "bypass_remaining_algorithm": bool(
            rhythm_summary.get("bypass_remaining_algorithm", False)
        ),
    }


# Reasons that suppress atrial-rhythm interpretation but need corroboration
# before they may also suppress the PR interval.  See the rationale in
# `build_measurement_availability`.
_PR_SOFT_REASONS = frozenset({"av_dissociation", "af_afl_indeterminate"})


def build_measurement_availability(
    pacing_context: Dict[str, Any],
    af_afl_summary: Dict[str, Any],
    rule_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """State when PR, P-axis, and atrial-rhythm inputs should be treated as unavailable."""
    pacing_context = pacing_context if isinstance(pacing_context, dict) else {}
    af_afl_summary = af_afl_summary if isinstance(af_afl_summary, dict) else {}
    rule_summary = rule_summary if isinstance(rule_summary, dict) else {}

    reasons: List[str] = []
    if bool(pacing_context.get("continuous_pacing")):
        reasons.append("continuous_pacing")
    if bool(pacing_context.get("wide_qrs_pacing_like_context")):
        reasons.append("wide_qrs_pacing_like_context")
    if bool(af_afl_summary.get("probable_af")):
        reasons.append("probable_af")
    if bool(af_afl_summary.get("probable_flutter")):
        reasons.append("probable_flutter")
    if bool(af_afl_summary.get("af_afl_indeterminate")):
        reasons.append("af_afl_indeterminate")
    if bool(rule_summary.get("complete_av_block")):
        reasons.append("complete_av_block")
    if bool(rule_summary.get("av_dissociation")):
        reasons.append("av_dissociation")
    if bool(rule_summary.get("atrial_measurements_invalid")):
        reasons.append("atrial_measurements_unavailable")

    reason_set = set(reasons)
    available = not reasons
    p_axis_available = available or reason_set == {"wide_qrs_pacing_like_context"}

    # PR reportability is not the same question as atrial-rhythm reportability.
    # Two of the reasons above do not, on their own, invalidate a PR interval:
    #
    #  * `av_dissociation` as computed by `detect_av_block_availability_flags`
    #    is derived purely from PR scatter (range > 80 ms and sd > 30 ms across
    #    per-beat medians).  On a record where one lead's P is associated to the
    #    wrong cycle that is a statement about measurement noise, not about AV
    #    conduction.  The corroborating criterion lives in the same dict
    #    (`atrial_faster_than_ventricular`), and genuine complete block adds its
    #    own `complete_av_block` reason, so requiring corroboration here cannot
    #    let a real dissociation through.
    #  * `af_afl_indeterminate` fires on borderline F-wave evidence even when
    #    organized P waves are present and RR is regular.  What actually makes a
    #    P-derived measurement untrustworthy is atrial activity being
    #    disorganized, which is measured separately by `organized_p_ratio`.
    #
    # Measured on the PTB-XL 09000 shard before this split: PR was withheld on
    # 202/1000 records, 155 of which carried neither an AFIB/AFLT nor a PACE
    # label, and 11 of the 44 records labelled 1AVB lost their PR class this
    # way.  The numeric PR was already retained for both reasons (they are not
    # in api.py's `hard_pr_unavailable` set), so suppressing only the class also
    # produced records reporting a numeric PR next to `pr_class=indeterminate`.
    pr_blocking = reason_set - _PR_SOFT_REASONS
    if "av_dissociation" in reason_set and av_dissociation_corroborated(rule_summary):
        pr_blocking.add("av_dissociation")
    if "af_afl_indeterminate" in reason_set and disorganized_atrial_activity(
        af_afl_summary,
        threshold=DEFAULT_DIAGNOSTIC_CONFIG.rhythm.organized_p_ratio_min_for_morphology,
    ) is not None:
        pr_blocking.add("af_afl_indeterminate")
    return {
        "atrial_rhythm_available": available,
        "pr_available": not pr_blocking,
        "p_axis_available": p_axis_available,
        "reasons": reasons,
        # Only meaningful when PR survived: which suppression reasons were
        # present but judged insufficient to withhold the PR interval.
        "pr_soft_reasons": (
            sorted(reason_set & _PR_SOFT_REASONS) if not pr_blocking else []
        ),
    }


def detect_av_block_availability_flags(
    beats: Iterable[Any],
    beat_features: Iterable[Any],
    heart_rate_bpm: Optional[float],
    atrial_rate_bpm: Optional[float] = None,
) -> Dict[str, bool]:
    """Conservatively mirror interpret-side AV block/dissociation availability flags."""
    hr = _finite_float(heart_rate_bpm)

    pr_by_beat: Dict[int, List[float]] = {}
    for feature in beat_features or []:
        beat_id = _safe_int(_field_value(feature, "beat_id"))
        pr_ms = _finite_float(_field_value(feature, "pr_ms"))
        if beat_id is None or pr_ms is None or not 60.0 < pr_ms < 600.0:
            continue
        pr_by_beat.setdefault(beat_id, []).append(pr_ms)

    ordered_pr: List[float] = []
    for beat in beats or []:
        beat_id = _safe_int(_field_value(beat, "beat_id"))
        if beat_id is None:
            continue
        values = pr_by_beat.get(beat_id) or []
        if values:
            ordered_pr.append(float(median(values)))

    av_dissociation = False
    if len(ordered_pr) >= 3:
        pr_range = max(ordered_pr) - min(ordered_pr)
        pr_sd = pstdev(ordered_pr)
        av_dissociation = bool(pr_range > 80.0 and pr_sd > 30.0)

    # Classic distinguishing feature of complete heart block vs. a merely
    # slow rhythm: the atrial rate runs distinctly faster than the
    # ventricular rate. A small margin excludes a trivial fallback where
    # atrial-rate estimation failed and defaulted to the ventricular rate
    # itself (in which case the two would be exactly equal, not "faster").
    atrial_rate = _finite_float(atrial_rate_bpm)
    atrial_faster_than_ventricular = bool(
        hr is not None and atrial_rate is not None and atrial_rate > hr + 5.0
    )

    complete_av_block = bool(
        hr is not None
        and hr < 45.0
        and (av_dissociation or atrial_faster_than_ventricular)
    )

    return {
        "complete_av_block": complete_av_block,
        "av_dissociation": av_dissociation,
        "atrial_faster_than_ventricular": atrial_faster_than_ventricular,
    }

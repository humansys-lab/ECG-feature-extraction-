"""Ventricular stage: QRS detection and pacing-aware QRS decisions.

Acquisition-chain QC and channel-delay compensation execute here because the legacy
call order runs them after QRS detection (they need the R peaks and may re-detect).

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List

import numpy as np

from ..._engine.detection.qrs import detect_qrs_multilead_with_meta
from ..._engine.preprocess import lowpass_filter
from ..._engine.quality.acquisition import (
    apply_channel_delay_compensation,
    assess_acquisition_chain,
)
from ..._engine.quality.signal import remove_pacing_spikes, validate_pacing_spikes_against_qrs
from ...rhythm_rules import assess_pacing_evidence_quality, detect_pacing_failures
from ..context import (
    PipelineContext,
    PolicyEvent,
    StageResult,
    record_decision,
    require,
    with_policy_events,
)
from ..policies.pacing import (
    ATTENUATED_PACING_RESCUE,
    CLEANED_SINGLE_LEAD_QRS_RESCUE,
    PACING_CAPTURE_ALIGNMENT,
    PACING_QRS_RESCUE_EVIDENCE,
    _PACING_QRS_RESCUE_MIN_CAPTURE_BEATS,
)


@dataclass(frozen=True, slots=True)
class VentricularBundle:
    """Engine state produced by the ventricular stage (legacy variable names).

    Fields reference live engine objects; see the ownership-transfer rule in
    ``ecgfeat.pipeline.context``.
    """

    ecg_measure: Any
    ecg_detect: Any
    qrs_result: Any
    r_locs: Any
    pacing_qrs_rescue: Any
    pacing_result: Any
    acquisition_qc: Any
    approved_delays: Any
    pacing_failures: Any
    pacing_spike_beat_ids: Any
    pacing_spike_offsets_samples: Any
    pacing_evidence_quality: Any
    pacing_capture_confirmed: Any
    paced_beat_ids: Any


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 1885-2077 (``self`` is ``context.config``)."""
    context = require(context, 'ventricular', 'config', 'input', 'quality')
    settings = context.config
    events: List[PolicyEvent] = []
    ecg_an = context.input.ecg_an
    ecg_det = context.input.ecg_det
    pacing_result = context.quality.pacing_result
    fs_run = context.input.fs_run
    qrs_options = context.quality.qrs_options
    quality = context.quality.quality
    ecg_rs = context.input.ecg_rs
    pacing_detection_cache = context.quality.pacing_detection_cache
    input_contract = context.input.input_contract
    ecg_measure = ecg_an
    ecg_detect = ecg_det
    pre_despike_qrs_result = None
    if pacing_result["spike_times"]:
        if (
            bool(pacing_result.get("paced", False))
            and str(pacing_result.get("state") or "off") == "on"
            and len(pacing_result["spike_times"]) >= _PACING_QRS_RESCUE_MIN_CAPTURE_BEATS
        ):
            pre_despike_qrs_result = detect_qrs_multilead_with_meta(ecg_det, fs_run, **qrs_options)
        ecg_measure = remove_pacing_spikes(ecg_an, pacing_result["spike_times"], fs_run)
        ecg_detect = lowpass_filter(ecg_measure, fs_run, cutoff_hz=settings.lp_hz, order=4)

    qrs_result = detect_qrs_multilead_with_meta(ecg_detect, fs_run, **qrs_options)
    r_locs = np.asarray(qrs_result.r_locs, dtype=int)
    pacing_qrs_rescue = record_decision(events, PACING_QRS_RESCUE_EVIDENCE.decide(
        pacing_result=pacing_result,
        pre_despike_qrs_result=pre_despike_qrs_result,
        despiked_qrs_result=qrs_result,
        fs=fs_run,
    ))
    if pacing_qrs_rescue["applied"]:
        rescued_qrs_result, rescue_detail = record_decision(events, CLEANED_SINGLE_LEAD_QRS_RESCUE.decide(
            ecg_detect=ecg_detect,
            fs=fs_run,
            quality=quality,
            pacing_result=pacing_result,
            pre_despike_qrs_result=pre_despike_qrs_result,
        ))
        if rescued_qrs_result is None:
            pacing_qrs_rescue = {
                **pacing_qrs_rescue,
                "applied": False,
                "reason": "no_regular_cleaned_single_lead_confirmation",
            }
        else:
            pacing_qrs_rescue = {**pacing_qrs_rescue, **rescue_detail}
            qrs_result = rescued_qrs_result
            r_locs = np.asarray(qrs_result.r_locs, dtype=int)
    if pacing_result["spike_times"]:
        raw_spike_times = [int(s) for s in pacing_result["spike_times"]]
        pacing_result = validate_pacing_spikes_against_qrs(
            ecg_rs,
            fs_run,
            raw_spike_times,
            r_locs,
            pacing_result,
        )
        validated_spike_times = [int(s) for s in pacing_result.get("spike_times", [])]
        if (
            pacing_result.get("rejection_reason") == "qrs_edge_artifact"
            and validated_spike_times != raw_spike_times
        ):
            ecg_measure = ecg_an
            ecg_detect = ecg_det
            if validated_spike_times:
                ecg_measure = remove_pacing_spikes(ecg_an, validated_spike_times, fs_run)
                ecg_detect = lowpass_filter(ecg_measure, fs_run, cutoff_hz=settings.lp_hz, order=4)
            qrs_result = detect_qrs_multilead_with_meta(ecg_detect, fs_run, **qrs_options)
            r_locs = np.asarray(qrs_result.r_locs, dtype=int)
            if not validated_spike_times:
                pacing_qrs_rescue = {
                    **pacing_qrs_rescue,
                    "applied": False,
                    "reason": "pacing_validation_rejected_qrs_edge_artifact",
                }
    if settings.enable_pacing:
        (
            pacing_result,
            ecg_measure,
            ecg_detect,
            qrs_result,
            r_locs,
        ) = record_decision(events, ATTENUATED_PACING_RESCUE.decide(
            ecg_rs=ecg_rs,
            ecg_an=ecg_an,
            fs_run=fs_run,
            lp_hz=settings.lp_hz,
            current_pacing_result=pacing_result,
            current_ecg_measure=ecg_measure,
            current_ecg_detect=ecg_detect,
            current_qrs_result=qrs_result,
            current_r_locs=r_locs,
            pacing_detection_cache=pacing_detection_cache,
        ))
    acquisition_qc = assess_acquisition_chain(
        ecg_rs,
        fs_run,
        r_locs,
        quality,
        # The swap detector has no `severely_inconsistent` key, so passing
        # it here silently disabled the LEAD_CONFIGURATION_INVALID reject.
        # The limb-equation residual lives in the input contract.
        limb_lead_consistency=input_contract.get("limb_lead_consistency"),
    )
    acquisition_qc["automatic_compensation_applied"] = False
    acquisition_qc["automatic_compensation_rejected_reason"] = None
    approved_delays = dict(
        acquisition_qc.get("approved_delay_samples") or {}
    )
    if approved_delays and not pacing_result.get("spike_times"):
        compensated_measure, applied_measure = apply_channel_delay_compensation(
            ecg_measure,
            approved_delays,
        )
        compensated_detect, _ = apply_channel_delay_compensation(
            ecg_detect,
            approved_delays,
        )
        compensated_qrs = detect_qrs_multilead_with_meta(
            compensated_detect,
            fs_run,
            **qrs_options,
        )
        compensated_r = np.asarray(compensated_qrs.r_locs, dtype=int)
        qrs_alignment_ms = None
        if compensated_r.size == r_locs.size and r_locs.size:
            qrs_alignment_ms = float(
                np.percentile(
                    np.abs(compensated_r - r_locs),
                    95,
                )
                * 1000.0
                / float(fs_run)
            )
        if (
            applied_measure
            and compensated_r.size == r_locs.size
            and qrs_alignment_ms is not None
            and qrs_alignment_ms <= 12.0
        ):
            ecg_measure = compensated_measure
            ecg_detect = compensated_detect
            qrs_result = compensated_qrs
            r_locs = compensated_r
            acquisition_qc["automatic_compensation_applied"] = True
            acquisition_qc["automatic_compensation_leads"] = sorted(
                applied_measure
            )
            acquisition_qc["post_compensation_qrs_p95_shift_ms"] = (
                qrs_alignment_ms
            )
        else:
            acquisition_qc["automatic_compensation_rejected_reason"] = (
                "qrs_detection_not_stable_after_compensation"
            )
            acquisition_qc["post_compensation_qrs_p95_shift_ms"] = (
                qrs_alignment_ms
            )
    elif approved_delays and pacing_result.get("spike_times"):
        acquisition_qc["automatic_compensation_rejected_reason"] = (
            "pacing_spikes_present"
        )
    pacing_failures = detect_pacing_failures(
        spike_times=[int(s) for s in pacing_result["spike_times"]],
        qrs_times=[int(r) for r in r_locs],
        fs=fs_run,
    )
    pacing_spike_beat_ids = []
    pacing_spike_offsets_samples = []
    if pacing_result["spike_times"]:
        for beat_id, r in enumerate(r_locs):
            spike_lo = int(r - 0.08 * fs_run)
            spike_hi = int(r + 0.02 * fs_run)
            nearby_spikes = [
                int(s)
                for s in pacing_result["spike_times"]
                if spike_lo <= int(s) <= spike_hi
            ]
            if nearby_spikes:
                nearest_spike = min(nearby_spikes, key=lambda s: abs(int(s) - int(r)))
                pacing_spike_beat_ids.append(int(beat_id))
                pacing_spike_offsets_samples.append(int(nearest_spike) - int(r))
    pacing_evidence_quality = assess_pacing_evidence_quality(
        pacing_result=pacing_result,
        spike_times=pacing_result["spike_times"],
        qrs_count=len(r_locs),
        pacing_spike_beat_ids=pacing_spike_beat_ids,
        spike_offsets_samples=pacing_spike_offsets_samples,
        pacing_failures=pacing_failures,
        fs=fs_run,
    )
    pacing_capture_confirmed = record_decision(events, PACING_CAPTURE_ALIGNMENT.decide(
        pacing_result,
        pacing_spike_offsets_samples,
        fs_run,
        pacing_evidence_quality,
    ))
    paced_beat_ids = (
        list(pacing_spike_beat_ids)
        if pacing_capture_confirmed or not bool(pacing_result.get("paced", False))
        else []
    )
    bundle = VentricularBundle(
        ecg_measure=ecg_measure,
        ecg_detect=ecg_detect,
        qrs_result=qrs_result,
        r_locs=r_locs,
        pacing_qrs_rescue=pacing_qrs_rescue,
        pacing_result=pacing_result,
        acquisition_qc=acquisition_qc,
        approved_delays=approved_delays,
        pacing_failures=pacing_failures,
        pacing_spike_beat_ids=pacing_spike_beat_ids,
        pacing_spike_offsets_samples=pacing_spike_offsets_samples,
        pacing_evidence_quality=pacing_evidence_quality,
        pacing_capture_confirmed=pacing_capture_confirmed,
        paced_beat_ids=paced_beat_ids,
    )
    context = replace(context, ventricular=bundle)
    context = with_policy_events(context, events)
    return StageResult(context)


__all__ = ['VentricularBundle', 'run']

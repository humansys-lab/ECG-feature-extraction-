"""Quality stage: signal quality, lead-reversal evidence and pacing pre-analysis.

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..._engine.quality.signal import (
    compute_quality,
    detect_limb_lead_reversal,
    detect_pacing_spikes,
    prepare_pacing_detection_cache,
    summarize_record_quality,
)
from ...models import STANDARD_12_LEADS
from ..context import PipelineContext, StageResult, require


@dataclass(frozen=True, slots=True)
class QualityBundle:
    """Engine state produced by the quality stage (legacy variable names).

    Fields reference live engine objects; see the ownership-transfer rule in
    ``ecgfeat.pipeline.context``.
    """

    raw_record_quality: Any
    quality: Any
    record_quality: Any
    refinement_kwargs: Any
    qrs_options: Any
    rep_left_ms: Any
    representative_options: Any
    lead_reversal: Any
    pacing_detection_cache: Any
    pacing_result: Any


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 1839-1884 (``self`` is ``context.config``)."""
    context = require(context, 'quality', 'config', 'request', 'input')
    settings = context.config
    meta = context.request.meta
    ecg_rs = context.input.ecg_rs
    fs_run = context.input.fs_run
    mains_freq_run = context.input.mains_freq_run
    quality_options = context.input.quality_options
    ecg_an = context.input.ecg_an
    available_leads = context.input.available_leads
    adc_full_scale_mv = (
        getattr(meta, "adc_full_scale_mv", None)
        if meta is not None
        else None
    )
    raw_quality = compute_quality(
        ecg_rs,
        fs_run,
        mains_hz=mains_freq_run,
        adc_full_scale_mv=adc_full_scale_mv,
    )
    raw_record_quality = summarize_record_quality(raw_quality, **quality_options)
    quality = compute_quality(
        ecg_an,
        fs_run,
        mains_hz=mains_freq_run,
        adc_full_scale_mv=adc_full_scale_mv,
    )
    record_quality = summarize_record_quality(quality, **quality_options)
    refinement_kwargs = {"refinement": settings.refinement} if settings.refinement.enabled else {}
    qrs_options = ({"quality": quality, "quality_reference": True}
                   if settings.refinement.qrs_quality_reference else {})
    if available_leads is not None:
        qrs_options["leads"] = tuple(STANDARD_12_LEADS.index(name) for name in available_leads)
    if settings.refinement.qrs_adaptive_consensus:
        qrs_options["adaptive_consensus"] = True
        qrs_options["quality"] = quality
    rep_left_ms = 450 if settings.refinement.representative_robust else 300
    representative_options = ({"left_ms": rep_left_ms, "robust_alignment": True}
                              if settings.refinement.representative_robust else {})
    lead_reversal = detect_limb_lead_reversal(ecg_an) if settings.enable_lead_reversal and available_leads is None else {}
    pacing_detection_cache = (
        prepare_pacing_detection_cache(ecg_rs, fs_run)
        if settings.enable_pacing
        else None
    )
    pacing_result = detect_pacing_spikes(
        ecg_rs,
        fs_run,
        detection_cache=pacing_detection_cache,
    ) if settings.enable_pacing else {
        "spike_times": [],
        "paced": False,
        "state": "off",
        "lead_vote_count": 0,
    }
    bundle = QualityBundle(
        raw_record_quality=raw_record_quality,
        quality=quality,
        record_quality=record_quality,
        refinement_kwargs=refinement_kwargs,
        qrs_options=qrs_options,
        rep_left_ms=rep_left_ms,
        representative_options=representative_options,
        lead_reversal=lead_reversal,
        pacing_detection_cache=pacing_detection_cache,
        pacing_result=pacing_result,
    )
    context = replace(context, quality=bundle)
    return StageResult(context)


__all__ = ['QualityBundle', 'run']

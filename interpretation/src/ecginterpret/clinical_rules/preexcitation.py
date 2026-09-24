from __future__ import annotations

from .models import RuleEvaluation
from .sources import SOURCE_AHA_RHYTHM


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def evaluate_preexcitation(context) -> RuleEvaluation:
    metadata = getattr(context.features, "metadata", {})
    rhythm = _mapping(metadata.get("rhythm_analysis"))
    summary = _mapping(_mapping(rhythm.get("rule_summary")).get("preexcitation"))
    pacing = _mapping(rhythm.get("pacing_context"))
    code = "ventricular_preexcitation_pattern"

    if not summary:
        return RuleEvaluation(
            rule_id="CLIN-RHYTHM-PREEXCITATION-01",
            domain="preexcitation",
            status="unavailable",
            required_inputs=["rhythm.rule_summary.preexcitation"],
            missing_inputs=["rhythm.rule_summary.preexcitation"],
            evidence={"evaluates_code": code},
            source=dict(SOURCE_AHA_RHYTHM),
            normality_required=False,
            normality_role="optional_screen",
        )

    suppressor = summary.get("suppressed_by")
    if not suppressor and (
        pacing.get("suppress_further_rhythm_interpretation")
        or pacing.get("continuous_pacing")
    ):
        suppressor = "pacing_context"

    qrs_ms = summary.get("mean_qrs_duration_ms")
    global_value = getattr(context, "global_value", None)
    pr_ms = global_value("pr_ms") if callable(global_value) else None
    pr_segment_ms = summary.get("pr_segment_ms")
    delta_leads = sorted({str(lead) for lead in summary.get("delta_leads", [])})
    delta_beat_ids = sorted(
        {
            int(beat_id)
            for beat_id in summary.get("delta_beat_ids", [])
            if isinstance(beat_id, (int, float))
        }
    )
    explicit_short_pr = bool(
        summary.get("short_pr_interval")
        or (isinstance(pr_ms, (int, float)) and float(pr_ms) < 120.0)
    )
    strong_multilead_fallback = bool(
        summary.get("strong_multilead_fallback")
        or (
            not explicit_short_pr
            and summary.get("short_pr_segment")
            and len(delta_leads) >= 8
            and len(delta_beat_ids) >= 3
            and isinstance(qrs_ms, (int, float))
            and 100.0 <= float(qrs_ms) <= 180.0
        )
    )
    validated_short_pr_evidence = bool(
        explicit_short_pr or strong_multilead_fallback
    )
    beat_features = list(getattr(context.features, "beat_features", []) or [])
    reliable_qrs_leads = [
        lead
        for lead, quality in getattr(context.features, "quality", {}).items()
        if bool(getattr(quality, "reliable_for_qrs", False))
    ]
    detector_coverage = bool(beat_features and len(reliable_qrs_leads) >= 3)
    matched = bool(
        summary.get("wpw_pattern")
        and validated_short_pr_evidence
        and len(delta_leads) >= (2 if explicit_short_pr else 8)
        and isinstance(qrs_ms, (int, float))
        and 100.0 <= float(qrs_ms) <= 180.0
    )
    weak_candidate = bool(
        summary.get("wpw_pattern")
        and not matched
        and summary.get("short_pr_segment")
        and len(delta_leads) >= 2
    )
    if strong_multilead_fallback and suppressor in {
        "wide_qrs_pacing_like_context",
        "pacing_context",
    }:
        suppressor = None
    evidence = {
        **summary,
        "evaluates_code": code,
        "pr_interval_ms": pr_ms,
        "pr_segment_ms": pr_segment_ms,
        "explicit_short_pr": explicit_short_pr,
        "strong_multilead_fallback": strong_multilead_fallback,
        "validated_short_pr_evidence": validated_short_pr_evidence,
        "reliable_qrs_leads": sorted(reliable_qrs_leads),
        "manual_confirmation_required": bool(matched or weak_candidate),
    }

    if suppressor:
        status = "suppressed" if matched else "indeterminate"
        missing = []
        suppressed_by = [str(suppressor)]
        coverage = "partial"
    elif matched:
        status = "matched"
        missing = []
        suppressed_by = []
        coverage = "full" if detector_coverage else "partial"
    elif weak_candidate:
        status = "indeterminate"
        missing = ["explicit_short_pr_or_strong_multilead_delta_evidence"]
        suppressed_by = []
        coverage = "partial"
    else:
        missing = []
        if qrs_ms is None:
            missing.append("global.qrs_ms")
        if pr_ms is None and pr_segment_ms is None:
            missing.append("global.pr_ms_or_pr_segment_ms")
        if not detector_coverage:
            missing.append("multilead_delta_wave_detection_coverage")
        status = "not_matched" if not missing else "unavailable"
        suppressed_by = []
        coverage = "full" if not missing else "unavailable"

    return RuleEvaluation(
        rule_id="CLIN-RHYTHM-PREEXCITATION-01",
        domain="preexcitation",
        status=status,
        statement_code=code if status in {"matched", "suppressed"} else None,
        statement=(
            "Ventricular pre-excitation ECG pattern; confirm morphology and clinical context"
            if status in {"matched", "suppressed"}
            else None
        ),
        severity="abnormal" if matched else "normal",
        confidence="moderate" if matched else None,
        coverage=coverage,
        required_inputs=[
            "global.pr_ms_or_pr_segment_ms",
            "global.qrs_ms",
            "multilead.delta_wave_evidence",
        ],
        missing_inputs=missing,
        evidence=evidence,
        thresholds={
            "short_pr_interval_lt_ms": 120.0,
            "minimum_delta_leads_with_short_pr": 2,
            "minimum_delta_leads_without_measurable_pr": 8,
            "qrs_duration_min_ms": 100.0,
            "qrs_duration_max_ms": 180.0,
            "short_pr_evidence_present": validated_short_pr_evidence,
            "observed_delta_lead_count": len(delta_leads),
        },
        suppressed_by=suppressed_by,
        source=dict(SOURCE_AHA_RHYTHM),
        normality_required=False,
        normality_role="optional_screen",
    )

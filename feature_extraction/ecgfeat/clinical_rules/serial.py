from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .models import RuleEvaluation


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _rhythm_signature(features: Any) -> str:
    metadata = getattr(features, "metadata", {})
    rhythm = metadata.get("rhythm_analysis", {}) if isinstance(metadata, dict) else {}
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    af = rhythm.get("af_afl_summary", {})
    af = af if isinstance(af, dict) else {}
    if af.get("probable_af"):
        return "atrial_fibrillation"
    if af.get("probable_flutter"):
        return "atrial_flutter"
    if getattr(getattr(features, "global_features", None), "paced_rhythm", False):
        return "paced"
    p_axis = _finite(getattr(getattr(features, "global_features", None), "p_axis_deg", None))
    return "sinus_candidate" if p_axis is not None and 0.0 <= p_axis <= 75.0 else "other"


def _median_st(features: Any) -> Optional[float]:
    representatives = getattr(features, "representative_leads", {}) or {}
    values = []
    for representative in representatives.values():
        value = _finite(getattr(representative, "params", {}).get("st_on_mv"))
        if value is not None:
            values.append(value)
    return float(np.median(values)) if values else None


def _t_polarity_changes(current: Any, prior: Any) -> list[str]:
    changed = []
    for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"):
        current_rep = (getattr(current, "representative_leads", {}) or {}).get(lead)
        prior_rep = (getattr(prior, "representative_leads", {}) or {}).get(lead)
        current_value = _finite(
            getattr(current_rep, "params", {}).get("t_amp_mv")
            if current_rep is not None else None
        )
        prior_value = _finite(
            getattr(prior_rep, "params", {}).get("t_amp_mv")
            if prior_rep is not None else None
        )
        if (
            current_value is not None
            and prior_value is not None
            and abs(current_value) >= 0.05
            and abs(prior_value) >= 0.05
            and np.sign(current_value) != np.sign(prior_value)
        ):
            changed.append(lead)
    return changed


def evaluate_serial_comparison(current: Any, prior: Any) -> list[RuleEvaluation]:
    if prior is None:
        return []
    current_global = getattr(current, "global_features", None)
    prior_global = getattr(prior, "global_features", None)
    metrics = {
        "pr_ms": (
            _finite(getattr(current_global, "pr_ms", None)),
            _finite(getattr(prior_global, "pr_ms", None)),
            20.0,
        ),
        "qrs_ms": (
            _finite(getattr(current_global, "qrs_ms", None)),
            _finite(getattr(prior_global, "qrs_ms", None)),
            10.0,
        ),
        "qt_ms": (
            _finite(getattr(current_global, "qt_ms", None)),
            _finite(getattr(prior_global, "qt_ms", None)),
            25.0,
        ),
        "qrs_axis_deg": (
            _finite(getattr(current_global, "qrs_axis_deg", None)),
            _finite(getattr(prior_global, "qrs_axis_deg", None)),
            30.0,
        ),
        "st_j_median_mv": (_median_st(current), _median_st(prior), 0.10),
    }
    significant = {
        name: {
            "current": current_value,
            "prior": prior_value,
            "delta": (
                current_value - prior_value
                if current_value is not None and prior_value is not None
                else None
            ),
            "threshold": threshold,
        }
        for name, (current_value, prior_value, threshold) in metrics.items()
        if current_value is not None
        and prior_value is not None
        and abs(current_value - prior_value) >= threshold
    }
    current_rhythm = _rhythm_signature(current)
    prior_rhythm = _rhythm_signature(prior)
    rhythm_changed = current_rhythm != prior_rhythm
    t_changes = _t_polarity_changes(current, prior)
    changed = bool(significant or rhythm_changed or t_changes)
    missing = []
    if all(
        current_value is None or prior_value is None
        for current_value, prior_value, _ in metrics.values()
    ):
        missing.append("comparable_current_and_prior_measurements")
    return [
        RuleEvaluation(
            rule_id="CLIN-SERIAL-01",
            domain="serial_comparison",
            status="matched" if changed else (
                "unavailable" if missing else "not_matched"
            ),
            statement_code="significant_change_from_prior" if changed else None,
            statement=(
                "Significant ECG change compared with prior tracing"
                if changed else None
            ),
            severity="high" if rhythm_changed else "observation" if changed else "normal",
            confidence="high" if changed else None,
            priority="P1" if rhythm_changed else "P2" if changed else None,
            missing_inputs=missing,
            evidence={
                "significant_metric_changes": significant,
                "current_rhythm": current_rhythm,
                "prior_rhythm": prior_rhythm,
                "rhythm_changed": rhythm_changed,
                "t_polarity_changed_leads": t_changes,
                "evaluates_code": "significant_change_from_prior",
            },
            thresholds={
                name: threshold
                for name, (_, _, threshold) in metrics.items()
            },
            human_review_required=changed,
            normality_role="supporting",
        )
    ]

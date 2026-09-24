from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Dict, Optional

from .glasgow_rules.context import build_context
from .glasgow_rules.engine import analyze_glasgow
from .glasgow_rules.measurement_matrix import build_measurement_matrix as _build_matrix
from .glasgow_rules.summary import SUMMARY_CODE_LABELS
from ecgfeat.compat.models_v0 import ECGFeatures


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def build_qtc_statement_guard(features: ECGFeatures) -> Dict[str, Any]:
    heart_rate = _finite(features.global_features.heart_rate_bpm)
    qrs_duration = _finite(features.global_features.qrs_ms)
    omitted_by = []
    if heart_rate is not None and heart_rate > 125.0:
        omitted_by.append("heart_rate_gt_125")
    if qrs_duration is not None and qrs_duration >= 120.0:
        omitted_by.append("qrs_duration_ge_120")
    return {
        "qtc_statement_allowed": not omitted_by,
        "omitted_by": omitted_by,
        "heart_rate_bpm": heart_rate,
        "qrs_duration_ms": qrs_duration,
        "source": "glasgow_qtc_omission_rule",
    }


def build_measurement_matrix(features: ECGFeatures) -> Dict[str, object]:
    """Compatibility wrapper for callers that previously imported this helper."""

    return _build_matrix(build_context(features))


def build_statement_catalog() -> Dict[str, Any]:
    return {
        "source": "Glasgow Physician's Guide chapter 20 statement families",
        "summary_codes": [
            {"code": code, "label": label}
            for code, label in SUMMARY_CODE_LABELS.items()
        ],
        "statement_families": [
            "preliminary",
            "intervals",
            "atrial_abnormalities",
            "critical_values",
            "qrs_axis_deviation",
            "conduction_defects",
            "wpw_pattern",
            "brugada_pattern",
            "hypertrophy",
            "myocardial_infarction",
            "st_abnormalities",
            "st_t_changes",
            "miscellaneous",
            "dominant_rhythm",
            "supplementary_rhythm",
            "summary",
        ],
        "foundation_complete_chapters": [3, 4, 5, 6, 18, 19],
        "pending_chapters": list(range(7, 18)),
    }


def build_glasgow_payload(
    features: ECGFeatures,
    *,
    statement_engine: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cached = features.metadata.get("glasgow_analysis")
    if isinstance(cached, dict) and cached.get("schema_version") == "glasgow_rules.v2":
        payload = deepcopy(cached)
    else:
        payload = analyze_glasgow(features).to_dict()

    payload["qtc_statement_guard"] = build_qtc_statement_guard(features)
    payload["statement_catalog"] = build_statement_catalog()
    payload["summary_code"] = deepcopy(payload["statement_resolution"]["summary_code"])
    compatibility = payload.setdefault("compatibility", {})
    compatibility.update(
        {
            "legacy_schema": "glasgow_inspired.v1",
            "legacy_projection_keys": [
                "qtc_statement_guard",
                "statement_catalog",
                "summary_code",
            ],
            "statement_engine_summary": (
                deepcopy(statement_engine.get("summary_code"))
                if isinstance(statement_engine, dict)
                and isinstance(statement_engine.get("summary_code"), dict)
                else None
            ),
        }
    )
    return payload


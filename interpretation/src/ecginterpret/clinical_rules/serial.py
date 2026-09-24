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


_SERIAL_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
_ST_TERRITORIES = {
    "inferior": {"II", "III", "aVF"},
    "lateral": {"I", "aVL", "V5", "V6"},
    "anterior": {"V1", "V2", "V3", "V4"},
}


def _metadata(features: Any) -> dict[str, Any]:
    value = getattr(features, "metadata", {})
    return value if isinstance(value, dict) else {}


def _patient_id(features: Any) -> Optional[str]:
    metadata = _metadata(features)
    patient = metadata.get("patient_meta")
    value = (
        getattr(patient, "patient_id", None)
        if patient is not None and not isinstance(patient, dict)
        else patient.get("patient_id") if isinstance(patient, dict) else None
    )
    value = value or metadata.get("patient_id")
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


def _serial_compatibility(current: Any, prior: Any) -> dict[str, Any]:
    """Check facts that can make two tracings unsafe to compare.

    Patient identity is only enforceable when callers provide it.  Older
    callers remain supported, but the unverified state is explicit evidence.
    """
    current_meta = _metadata(current)
    prior_meta = _metadata(prior)
    current_id = _patient_id(current)
    prior_id = _patient_id(prior)
    reasons: list[str] = []
    if current_id is not None and prior_id is not None and current_id != prior_id:
        reasons.append("patient_id_mismatch")
    current_order = current_meta.get("lead_order")
    prior_order = prior_meta.get("lead_order")
    if current_order and prior_order and list(current_order) != list(prior_order):
        reasons.append("lead_order_mismatch")
    for label, metadata in (("current", current_meta), ("prior", prior_meta)):
        gate = metadata.get("diagnostic_gate", {})
        if isinstance(gate, dict) and gate.get("state") == "stop":
            reasons.append(f"{label}_diagnostic_gate_stop")
    current_contract = current_meta.get("input_contract", {})
    prior_contract = prior_meta.get("input_contract", {})
    current_unit = current_contract.get("amplitude_output_unit") if isinstance(current_contract, dict) else None
    prior_unit = prior_contract.get("amplitude_output_unit") if isinstance(prior_contract, dict) else None
    if current_unit and prior_unit and current_unit != prior_unit:
        reasons.append("amplitude_unit_mismatch")
    amplitude_comparable = True
    for contract in (current_contract, prior_contract):
        calibration = contract.get("amplitude_calibration", {}) if isinstance(contract, dict) else {}
        if isinstance(calibration, dict) and calibration.get("amplitudes_diagnostic") is False:
            amplitude_comparable = False
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "identity_verified": bool(current_id and prior_id and current_id == prior_id),
        "current_patient_id": current_id,
        "prior_patient_id": prior_id,
        "amplitude_comparable": amplitude_comparable,
    }


def _axis_delta_deg(current: float, prior: float) -> float:
    """Return the signed shortest angular difference in [-180, 180)."""
    return float((float(current) - float(prior) + 180.0) % 360.0 - 180.0)


def _lead_st_values(features: Any) -> dict[str, float]:
    representatives = getattr(features, "representative_leads", {}) or {}
    values: dict[str, float] = {}
    for lead in _SERIAL_LEADS:
        representative = representatives.get(lead)
        if representative is None:
            continue
        params = getattr(representative, "params", {})
        if not isinstance(params, dict):
            continue
        if params.get("reliable_for_global") is False:
            continue
        value = _finite(params.get("st_on_mv"))
        if value is not None:
            values[lead] = value
    return values


def _regional_st_changes(current: Any, prior: Any) -> dict[str, Any]:
    current_values = _lead_st_values(current)
    prior_values = _lead_st_values(prior)
    deltas = {
        lead: float(current_values[lead] - prior_values[lead])
        for lead in _SERIAL_LEADS
        if lead in current_values and lead in prior_values
    }
    changed_leads = {
        lead for lead, delta in deltas.items() if abs(delta) >= 0.10
    }
    territories = {
        name: sorted(changed_leads & leads)
        for name, leads in _ST_TERRITORIES.items()
        if len(changed_leads & leads) >= 2
    }
    marked_single_leads = sorted(
        lead for lead, delta in deltas.items() if abs(delta) >= 0.20
    )
    significant_leads = sorted(
        set(marked_single_leads).union(
            *(set(leads) for leads in territories.values())
        )
    )
    return {
        "significant": bool(significant_leads),
        "deltas_mv": deltas,
        "significant_leads": significant_leads,
        "territories": territories,
        "threshold_mv": 0.10,
        "marked_single_lead_threshold_mv": 0.20,
    }


def _t_polarity_changes(current: Any, prior: Any) -> list[str]:
    changed = []
    for lead in _SERIAL_LEADS:
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
    compatibility = _serial_compatibility(current, prior)
    if not compatibility["compatible"]:
        return [
            RuleEvaluation(
                rule_id="CLIN-SERIAL-01",
                domain="serial_comparison",
                status="unavailable",
                severity="normal",
                missing_inputs=list(compatibility["reasons"]),
                evidence={
                    "comparison_contract": compatibility,
                    "evaluates_code": "significant_change_from_prior",
                },
                thresholds={},
                human_review_required=False,
                normality_role="supporting",
            )
        ]
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
        "st_j_median_mv": (
            _median_st(current) if compatibility["amplitude_comparable"] else None,
            _median_st(prior) if compatibility["amplitude_comparable"] else None,
            0.10,
        ),
    }
    significant = {}
    for name, (current_value, prior_value, threshold) in metrics.items():
        if current_value is None or prior_value is None:
            continue
        delta = (
            _axis_delta_deg(current_value, prior_value)
            if name == "qrs_axis_deg"
            else float(current_value - prior_value)
        )
        if abs(delta) < threshold:
            continue
        significant[name] = {
            "current": current_value,
            "prior": prior_value,
            "delta": delta,
            "threshold": threshold,
        }
    regional_st = (
        _regional_st_changes(current, prior)
        if compatibility["amplitude_comparable"]
        else {
            "significant": False,
            "reason": "amplitude_calibration_not_comparable",
            "deltas_mv": {},
            "significant_leads": [],
            "territories": {},
        }
    )
    if regional_st["significant"]:
        significant["regional_st_j_mv"] = {
            "leads": regional_st["significant_leads"],
            "territories": regional_st["territories"],
            "deltas_mv": {
                lead: regional_st["deltas_mv"][lead]
                for lead in regional_st["significant_leads"]
            },
            "threshold": regional_st["threshold_mv"],
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
                "regional_st_change": regional_st,
                "comparison_contract": compatibility,
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

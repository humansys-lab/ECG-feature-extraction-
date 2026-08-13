from __future__ import annotations

from .models import RuleEvaluation


def _available_boolean(value) -> bool | None:
    """Unwrap tri-state detector output without treating a dict as truthy."""
    if isinstance(value, dict):
        if value.get("available") is False:
            return None
        nested = value.get("value")
        return nested if isinstance(nested, bool) else None
    return value if isinstance(value, bool) else None


def evaluate_pacing(context) -> list[RuleEvaluation]:
    metadata = context.features.metadata
    rhythm = metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    failures = rhythm.get("pacing_failures", {})
    failures = failures if isinstance(failures, dict) else {}
    paced_fraction = metadata.get("paced_beat_fraction")
    state = str(metadata.get("measurement_pacing_state") or "off")
    spike_present = bool(
        getattr(context.features.global_features, "pacing_spikes", []) or []
    )
    pacing_evidence_present = bool(spike_present and state in {"on", "unknown"})
    capture_failure = _available_boolean(
        failures.get("capture_failure_suspected")
    )
    sensing_failure = _available_boolean(
        failures.get("sensing_failure_suspected")
    )
    spike_count = int(
        failures.get("spike_count")
        or pacing.get("spike_count")
        or len(getattr(context.features.global_features, "pacing_spikes", []) or [])
    )
    failure_times = list(failures.get("failure_spike_times") or [])
    failure_count = int(
        failures.get("capture_failure_count")
        if failures.get("capture_failure_count") is not None
        else len(failure_times)
    )
    failure_fraction = failures.get("capture_failure_fraction")
    if failure_fraction is None and spike_count:
        failure_fraction = float(failure_count) / float(spike_count)
    alignment_fraction = failures.get("capture_alignment_fraction")
    if alignment_fraction is None and failure_fraction is not None:
        alignment_fraction = max(0.0, 1.0 - float(failure_fraction))
    paced_fraction_value = (
        float(paced_fraction) if paced_fraction is not None else 0.0
    )
    established_pacing_context = bool(
        state == "on"
        and spike_count >= 3
        and bool(pacing.get("ventricular_pacing_present"))
        and paced_fraction_value >= 0.20
    )
    capture_alert_supported = bool(
        capture_failure is True
        and established_pacing_context
        and bool(failures.get("minimum_evidence_met"))
        and failure_count >= 2
        and failure_fraction is not None
        and 0.0 < float(failure_fraction) <= 0.50
    )
    aligned_pacing_supported = bool(
        established_pacing_context
        and alignment_fraction is not None
        and float(alignment_fraction) >= 0.60
    )

    definitions = [
        {
            "rule_id": "CLIN-PACING-V-01",
            "code": "ventricular_paced_rhythm",
            "statement": "Ventricular paced rhythm",
            "matched": bool(
                context.features.global_features.paced_rhythm
                and paced_fraction_value > 0.80
                and aligned_pacing_supported
            ),
            "candidate": bool(
                context.features.global_features.paced_rhythm
                and paced_fraction_value > 0.80
                and not aligned_pacing_supported
            ),
            "detector_available": True,
            "missing": ["recurrent_capture_aligned_pacing"]
            if not aligned_pacing_supported else [],
        },
        {
            "rule_id": "CLIN-PACING-I-01",
            "code": "intermittent_pacing",
            "statement": "Intermittent pacing",
            "matched": bool(
                0.20 <= paced_fraction_value <= 0.80
                and aligned_pacing_supported
            ),
            "candidate": bool(
                0.20 <= paced_fraction_value <= 0.80
                and not aligned_pacing_supported
            ),
            "detector_available": True,
            "missing": ["recurrent_capture_aligned_pacing"]
            if not aligned_pacing_supported else [],
        },
        {
            "rule_id": "CLIN-PACING-CAPTURE-01",
            "code": "pacing_failure_to_capture_suspected",
            "statement": "Possible pacemaker failure to capture; urgent device review",
            "matched": capture_alert_supported,
            "candidate": bool(capture_failure is True and not capture_alert_supported),
            "detector_available": (
                not pacing_evidence_present or capture_failure is not None
            ),
            "missing": ["recurrent_confirmed_pacing_context"]
            if capture_failure is True and not capture_alert_supported else [],
        },
        {
            "rule_id": "CLIN-PACING-SENSE-01",
            "code": "pacing_sensing_failure_suspected",
            "statement": "Possible pacemaker sensing abnormality; device review required",
            "matched": bool(
                pacing_evidence_present
                and established_pacing_context
                and sensing_failure is True
            ),
            "candidate": bool(
                sensing_failure is True and not established_pacing_context
            ),
            "detector_available": (
                not pacing_evidence_present or sensing_failure is not None
            ),
            "missing": ["recurrent_confirmed_pacing_context"]
            if sensing_failure is True and not established_pacing_context else [],
        },
    ]
    rows: list[RuleEvaluation] = []
    for definition in definitions:
        rule_id = definition["rule_id"]
        code = definition["code"]
        statement = definition["statement"]
        matched = bool(definition["matched"])
        candidate = bool(definition["candidate"])
        detector_available = bool(definition["detector_available"])
        device_alert = "CAPTURE" in rule_id or "SENSE" in rule_id
        if matched:
            status = "matched"
        elif candidate:
            status = "indeterminate"
        elif state == "unknown" or not detector_available:
            status = "unavailable"
        else:
            status = "not_matched"
        rows.append(
            RuleEvaluation(
                rule_id=rule_id,
                domain="rhythm",
                status=status,
                statement_code=code if matched else None,
                statement=statement if matched else None,
                severity="high" if device_alert else (
                    "observation" if matched else "normal"
                ),
                confidence="medium" if matched else None,
                coverage="partial" if candidate else None,
                priority="P1" if device_alert else None,
                human_review_required=device_alert,
                normality_role="supporting" if not device_alert else "core",
                missing_inputs=list(definition["missing"]),
                evidence={
                    "measurement_pacing_state": state,
                    "pacing_spike_present": spike_present,
                    "paced_beat_fraction": paced_fraction,
                    "pacing_context": pacing,
                    "pacing_failures": failures,
                    "established_pacing_context": established_pacing_context,
                    "capture_alert_supported": capture_alert_supported,
                    "capture_alignment_fraction": alignment_fraction,
                    "evaluates_code": code,
                },
            )
        )
    return rows

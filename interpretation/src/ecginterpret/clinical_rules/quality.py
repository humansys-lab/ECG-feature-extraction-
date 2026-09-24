from __future__ import annotations

from .models import RuleEvaluation


def evaluate_quality(context) -> list[RuleEvaluation]:
    record_quality = context.features.metadata.get("record_quality", {})
    grade = record_quality.get("record_grade") if isinstance(record_quality, dict) else None
    missing = [] if grade is not None else ["metadata.record_quality.record_grade"]
    gate = context.features.metadata.get("diagnostic_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    gate_state = str(gate.get("state") or "")
    limited = grade in {"Q2", "Q3"} or gate_state == "stop"
    rows = [
        RuleEvaluation(
            rule_id="CLIN-QUALITY-01",
            domain="quality",
            status="matched" if limited else "not_matched",
            statement_code="technically_limited" if limited else None,
            statement="Technically limited ECG" if limited else None,
            severity="technical" if limited else "normal",
            required_inputs=["metadata.record_quality.record_grade"],
            missing_inputs=missing,
            evidence={
                "record_grade": grade,
                "diagnostic_gate": gate,
                "excluded_leads": sorted(context.excluded_leads),
            },
            human_review_required=limited,
        )
    ]
    if grade == "Q1" and not limited:
        rows.append(
            RuleEvaluation(
                rule_id="CLIN-QUALITY-OBS-01",
                domain="quality",
                status="matched",
                statement_code="minor_signal_quality_issue",
                statement="Minor signal-quality issue; interpretation remains available",
                severity="observation",
                confidence="high",
                normality_role="supporting",
                evidence={"record_grade": grade},
            )
        )
    if gate_state == "stop":
        rows.append(
            RuleEvaluation(
                rule_id="CLIN-QUALITY-GATE-01",
                domain="quality",
                status="matched",
                statement_code="repeat_ecg_required",
                statement="Diagnostic interpretation withheld; repeat ECG acquisition required",
                severity="technical",
                confidence="high",
                priority="P3",
                evidence=gate,
                human_review_required=True,
            )
        )
    elif gate_state == "partial":
        rows.append(
            RuleEvaluation(
                rule_id="CLIN-QUALITY-GATE-02",
                domain="quality",
                status="matched",
                statement_code="diagnostic_coverage_limited",
                statement="Diagnostic coverage is limited; see acquisition and metadata reasons",
                severity="technical",
                confidence="high",
                priority="P3",
                evidence=gate,
                human_review_required=True,
            )
        )
    if bool(gate.get("morphology_complex")):
        rows.append(
            RuleEvaluation(
                rule_id="CLIN-QUALITY-MORPH-01",
                domain="quality",
                status="matched",
                statement_code="complex_morphology_limited_interpretation",
                statement="Complex beat morphology; representative-beat diagnoses withheld",
                severity="technical",
                confidence="high",
                priority="P3",
                evidence=gate,
                human_review_required=True,
            )
        )
    if getattr(context, "precordial_reversal_state", "not_suspected") == "possible":
        rows.append(
            RuleEvaluation(
                rule_id="CLIN-QUALITY-LEADMAP-01",
                domain="quality",
                status="matched",
                statement_code="possible_precordial_lead_reversal",
                statement="Possible precordial lead-placement anomaly; verify lead placement",
                severity="observation",
                confidence="low",
                normality_role="supporting",
                evidence={
                    "state": "possible",
                    "implicated_leads": sorted(
                        getattr(context, "precordial_reversal_leads", ())
                    ),
                },
            )
        )
    return rows

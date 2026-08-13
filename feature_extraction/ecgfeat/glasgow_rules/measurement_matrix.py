from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List

from ..models import STANDARD_12_LEADS
from .context import GlasgowContext
from .models import RuleEvaluation, RuleSpec


EXPECTED_MATRIX_ROWS = [
    1, 2, 3, 4, 6, 7, 8, 9, 10, 14, 16, 18, 19, 20, 21, 23, 24,
    25, 26, 27, 30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 47, 51, 52, 53,
]


@dataclass(frozen=True)
class MatrixRow:
    row: int
    content: str
    description: str
    unit: str
    key: str
    scale: float = 1.0


MATRIX_ROWS = (
    MatrixRow(1, "P onset", "Time from representative beat start to P onset", "ms", "p_onset_ms"),
    MatrixRow(2, "P duration", "P wave duration", "ms", "p_duration_ms"),
    MatrixRow(3, "QRS onset", "Time from representative beat start to QRS onset", "ms", "qrs_onset_ms"),
    MatrixRow(4, "QRS duration", "QRS complex duration", "ms", "qrs_duration_ms"),
    MatrixRow(6, "Q duration", "Q wave duration", "ms", "q_duration_ms"),
    MatrixRow(7, "R duration", "R wave duration", "ms", "r_duration_ms"),
    MatrixRow(8, "S duration", "S wave duration", "ms", "s_duration_ms"),
    MatrixRow(9, "R' duration", "R' wave duration", "ms", "r_prime_duration_ms"),
    MatrixRow(10, "S' duration", "S' wave duration", "ms", "s_prime_duration_ms"),
    MatrixRow(14, "T onset", "Time from representative beat start to T onset", "ms", "t_onset_ms"),
    MatrixRow(16, "P+ duration", "Positive P component duration", "ms", "p_positive_duration_ms"),
    MatrixRow(18, "QRS intrinsicoid deflection", "Ventricular activation time", "ms", "vat_ms"),
    MatrixRow(19, "P+ amplitude", "Positive P component amplitude", "uV", "p_positive_amp_mv", 1000.0),
    MatrixRow(20, "P- amplitude", "Negative P component amplitude", "uV", "p_negative_amp_mv", 1000.0),
    MatrixRow(21, "Peak to peak QRS", "Peak-to-peak QRS amplitude", "uV", "qrs_peak_to_peak_mv", 1000.0),
    MatrixRow(23, "Q amplitude", "Q wave amplitude", "uV", "q_amp_mv", 1000.0),
    MatrixRow(24, "R amplitude", "R wave amplitude", "uV", "r_amp_mv", 1000.0),
    MatrixRow(25, "S amplitude", "S wave amplitude", "uV", "s_amp_mv", 1000.0),
    MatrixRow(26, "R' amplitude", "R' wave amplitude", "uV", "r_prime_amp_mv", 1000.0),
    MatrixRow(27, "S' amplitude", "S' wave amplitude", "uV", "s_prime_amp_mv", 1000.0),
    MatrixRow(30, "ST amplitude", "ST amplitude at the J point", "uV", "st_amp_mv", 1000.0),
    MatrixRow(31, "2/8 ST-T amp", "Amplitude at 2/8 of the J-to-T-end interval", "uV", "st_2_8_amp_mv", 1000.0),
    MatrixRow(32, "3/8 ST-T amp", "Amplitude at 3/8 of the J-to-T-end interval", "uV", "st_3_8_amp_mv", 1000.0),
    MatrixRow(33, "T+ amplitude", "Positive T component amplitude", "uV", "t_positive_amp_mv", 1000.0),
    MatrixRow(34, "T- amplitude", "Negative T component amplitude", "uV", "t_negative_amp_mv", 1000.0),
    MatrixRow(35, "QRS area", "Absolute QRS area scaled down by 20", "uV*ms/20", "qrs_area_matrix"),
    MatrixRow(39, "T morphology", "T wave morphology code", "code", "t_morphology"),
    MatrixRow(40, "R wave notch", "R wave notch count", "count", "r_wave_notch_count"),
    MatrixRow(41, "Delta wave confidence", "Probability of delta wave presence", "%", "delta_confidence_pct"),
    MatrixRow(42, "ST slope", "ST slope from J point to 3/8 ST-T point", "degrees", "st_slope_deg"),
    MatrixRow(47, "QT interval", "QT interval duration", "ms", "qt_ms"),
    MatrixRow(51, "QRS notch/slur", "Amplitude at start/end QRS notch or slur", "uV", "qrs_notch_slur_amp_mv", 1000.0),
    MatrixRow(52, "PR amplitude", "Amplitude difference from P onset to QRS onset", "uV", "pr_amp_mv", 1000.0),
    MatrixRow(53, "ST adjusted amplitude", "ST amplitude adjusted for PR amplitude", "uV", "st_adjusted_amp_mv", 1000.0),
)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(result):
        return None
    return int(result) if isinstance(value, int) else result


def build_measurement_matrix(context: GlasgowContext) -> Dict[str, object]:
    rows = []
    for definition in MATRIX_ROWS:
        cells: Dict[str, Dict[str, object]] = {}
        for lead in STANDARD_12_LEADS:
            fact = context.leads.get(lead)
            measurements = fact.measurements if fact is not None else {}
            raw_value = _number(measurements.get(definition.key))
            unavailable = measurements.get("unavailable_reasons")
            unavailable = unavailable if isinstance(unavailable, dict) else {}
            excluded_reasons = context.excluded_leads.get(lead, [])
            representative_id = measurements.get("representative_beat_id")
            source = measurements.get("measurement_source", "representative_profile")
            if raw_value is None:
                cells[lead] = {
                    "status": "unavailable",
                    "value": None,
                    "source": source,
                    "reliability": "excluded" if excluded_reasons else "unknown",
                    "representative_beat_id": representative_id,
                    "reason": unavailable.get(definition.key, f"{definition.key}_unavailable"),
                }
            else:
                cells[lead] = {
                    "status": "available",
                    "value": raw_value * definition.scale,
                    "source": source,
                    "reliability": "excluded" if excluded_reasons else "usable",
                    "representative_beat_id": representative_id,
                    "reason": ";".join(excluded_reasons) if excluded_reasons else None,
                }
        rows.append(
            {
                "row": definition.row,
                "content": definition.content,
                "description": definition.description,
                "unit": definition.unit,
                "profile_key": definition.key,
                "cells": cells,
            }
        )
    return {
        "schema": "glasgow_measurement_matrix.v1",
        "reference": "QRS onset horizontal reference",
        "leads": list(STANDARD_12_LEADS),
        "rows": rows,
    }


def _measurement_evaluator(definition: MatrixRow):
    def evaluate(context: GlasgowContext) -> RuleEvaluation:
        values = {
            lead: _number(context.leads[lead].measurements.get(definition.key))
            for lead in STANDARD_12_LEADS
            if lead in context.leads
        }
        available = [lead for lead, value in values.items() if value is not None]
        missing = [lead for lead, value in values.items() if value is None]
        fidelity = (
            "existing_dxl_approximation"
            if definition.row in {1, 3, 14, 39, 41, 51, 53}
            else "glasgow_explicit"
        )
        approximation = None
        if fidelity != "glasgow_explicit":
            approximation = {
                "source": "existing_dxl",
                "reason": {
                    1: "representative_lead_local_boundary_selection",
                    3: "representative_lead_local_boundary_selection",
                    14: "representative_lead_local_boundary_selection",
                    39: "existing_component_classifier_mapped_to_glasgow_morphology",
                    41: "existing_delta_evidence_used_as_confidence_seed",
                    51: "existing_qrs_extrema_used_for_notch_slur_location",
                    53: "guide_does_not_disclose_complete_st_adjustment",
                }[definition.row],
                "original_rule_reproducible": False,
            }
        return RuleEvaluation(
            rule_id=f"GAN-19-R{definition.row:02d}",
            evaluation_status="matched" if available else "unavailable",
            resolution_status="no_statement",
            statement=definition.content,
            evidence={
                "row": definition.row,
                "profile_key": definition.key,
                "unit": definition.unit,
                "available_leads": available,
                "unavailable_leads": missing,
            },
            required_inputs=[f"lead.*.{definition.key}"],
            missing_inputs=[] if available else [f"lead.*.{definition.key}"],
            fidelity=fidelity,
            source={
                "kind": "existing_dxl" if fidelity != "glasgow_explicit" else "glasgow_guide",
                "reference": "Physician's Guide sections 3.1 and 19, PDF pages 9 and 73-74",
            },
            approximation=approximation,
        )

    return evaluate


def measurement_rule_specs() -> List[RuleSpec]:
    rules: List[RuleSpec] = []
    for definition in MATRIX_ROWS:
        approximated = definition.row in {1, 3, 14, 39, 41, 51, 53}
        rules.append(
            RuleSpec(
                rule_id=f"GAN-19-R{definition.row:02d}",
                chapter="19",
                pdf_page=73 if definition.row <= 41 else 74,
                category="measurement",
                order=190000 + definition.row,
                statement=definition.content,
                required_inputs=(f"lead.*.{definition.key}",),
                fidelity="existing_dxl_approximation" if approximated else "glasgow_explicit",
                source_kind="existing_dxl" if approximated else "glasgow_guide",
                source_reference="Physician's Guide sections 3.1 and 19, PDF pages 9 and 73-74",
                evaluator=_measurement_evaluator(definition),
                source_chapters=(3, 19),
            )
        )
    return rules


assert [definition.row for definition in MATRIX_ROWS] == EXPECTED_MATRIX_ROWS

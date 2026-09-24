from __future__ import annotations

from typing import Any, Dict, Sequence

from .context import GlasgowContext
from .models import RuleEvaluation, RuleSpec


SUMMARY_CODE_LABELS = {
    1: "Normal ECG",
    2: "Normal ECG except for rate",
    3: "Normal ECG based on available leads",
    4: "Borderline ECG",
    5: "Abnormal ECG",
    6: "Technical error",
}


def resolve_summary_code(
    final: Sequence[RuleEvaluation],
    *,
    core_error: bool,
    lead_limited: bool,
) -> Dict[str, Any]:
    if core_error:
        code = 6
    else:
        codes = [
            int(item.summary_code)
            for item in final
            if item.summary_code in SUMMARY_CODE_LABELS
        ]
        code = max(codes) if codes else (3 if lead_limited else 1)
    return {
        "code": code,
        "label": SUMMARY_CODE_LABELS[code],
        "source": "glasgow_rule_resolution",
    }


def _summary_placeholder(_context: GlasgowContext) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id="GAN-18-01",
        evaluation_status="not_applicable",
        resolution_status="no_statement",
        statement="Summary code resolution",
        evidence={"reason": "resolved_after_all_foundation_rules"},
        fidelity="glasgow_explicit",
        source={
            "kind": "glasgow_guide",
            "reference": "Physician's Guide chapter 18, PDF page 72",
        },
    )


def summary_rule() -> RuleSpec:
    return RuleSpec(
        rule_id="GAN-18-01",
        chapter="18",
        pdf_page=72,
        category="summary",
        order=180001,
        statement="Summary code resolution",
        required_inputs=("resolved_statements",),
        fidelity="glasgow_explicit",
        source_kind="glasgow_guide",
        source_reference="Physician's Guide chapter 18, PDF page 72",
        evaluator=_summary_placeholder,
    )


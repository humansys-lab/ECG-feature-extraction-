"""Atrial stage: P-wave assessments, atrial events and measurement-side AV evidence.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from ..context import PipelineContext, StageResult


def _av_block_evidence(rule_summary: Dict[str, Any]) -> Dict[str, Any]:
    pauses = rule_summary.get("pauses") if isinstance(rule_summary, dict) else {}
    evidence = pauses.get("av_block_evidence", {}) if isinstance(pauses, dict) else {}
    return evidence if isinstance(evidence, dict) else {}



@dataclass(frozen=True, slots=True)
class AtrialBundle:
    p_assessments: tuple[Any, ...] = ()
    atrial_events: tuple[Any, ...] = ()
    organized_p_ratio: float | None = None
    pr_dispersion_ms: float | None = None
    av_measurement_evidence: Mapping[str, Any] = field(default_factory=dict)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent atrial stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

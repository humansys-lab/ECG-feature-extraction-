from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import PipelineContext, StageResult


@dataclass(frozen=True, slots=True)
class AtrialBundle:
    p_assessments: tuple[Any, ...] = ()
    atrial_events: tuple[Any, ...] = ()
    organized_p_ratio: float | None = None
    pr_dispersion_ms: float | None = None
    av_measurement_evidence: Mapping[str, Any] = field(default_factory=dict)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent atrial stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

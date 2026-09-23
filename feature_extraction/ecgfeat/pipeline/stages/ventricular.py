from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import PipelineContext, StageResult


@dataclass(frozen=True, slots=True)
class VentricularBundle:
    primary: Any = None
    adaptive: Any = None
    selected_r_peaks: tuple[int, ...] = ()
    pacing_decision: Any = None
    detector_agreement: Mapping[str, Any] = field(default_factory=dict)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent ventricular stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

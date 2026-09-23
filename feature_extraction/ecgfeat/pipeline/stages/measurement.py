from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import PipelineContext, StageResult


@dataclass(frozen=True, slots=True)
class MeasurementBundle:
    representative: Mapping[int, Any] = field(default_factory=dict)
    groups: Mapping[int, Any] = field(default_factory=dict)
    global_features: Any = None
    qt_decision: Any = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent measurement stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

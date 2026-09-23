from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import PipelineContext, StageResult


@dataclass(frozen=True, slots=True)
class DelineationBundle:
    beat_boundaries: Mapping[int, Any] = field(default_factory=dict)
    representative_boundaries: Mapping[str, Any] = field(default_factory=dict)
    candidates: Mapping[str, Any] = field(default_factory=dict)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent delineation stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

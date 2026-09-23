from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import PipelineContext, StageResult


@dataclass(frozen=True, slots=True)
class QualityBundle:
    lead_quality: tuple[Any, ...] = ()
    record_summary: Mapping[str, Any] = field(default_factory=dict)
    acquisition: Any = None
    pacing: Any = None
    limb_reversal: Any = None
    precordial_reversal: Any = None


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent quality stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

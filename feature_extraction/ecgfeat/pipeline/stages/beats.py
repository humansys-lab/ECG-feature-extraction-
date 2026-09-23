from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import PipelineContext, StageResult


@dataclass(frozen=True, slots=True)
class BeatBundle:
    annotations: tuple[Any, ...] = ()
    clusters: tuple[Any, ...] = ()
    families: tuple[Any, ...] = ()
    representatives: Mapping[int, Any] = field(default_factory=dict)
    representative_meta: Mapping[int, Any] = field(default_factory=dict)
    measurement_group_id: int | None = None
    measurement_beat_ids: tuple[int, ...] = ()


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent beats stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

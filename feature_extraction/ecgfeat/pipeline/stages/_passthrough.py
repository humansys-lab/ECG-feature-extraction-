from __future__ import annotations

from ..context import PipelineContext, StageResult


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent _passthrough stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

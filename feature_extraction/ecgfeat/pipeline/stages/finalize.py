from __future__ import annotations

from ..context import PipelineContext, StageResult


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent finalize stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")


def build_record(context: PipelineContext):
    if context.measurements is None:
        raise ValueError("finalize requires measurement-stage output")
    # The compatibility-backed extractor stores the complete typed measurement
    # carrier here.  A future native staged engine can provide the same carrier
    # without changing this finalize seam.
    if hasattr(context.measurements, "prepared") and hasattr(context.measurements, "legacy_features"):
        from ..extractor import ecg_emit
        return ecg_emit(context.measurements)
    raise ValueError("measurement bundle does not contain a record-emission carrier")

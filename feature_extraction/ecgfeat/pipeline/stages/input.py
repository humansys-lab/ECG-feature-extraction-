"""Input-stage helpers and the shared stage seam."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...errors import SamplingRateError
from ..context import PipelineContext, StageResult


def resolve_mains_frequency(signal: np.ndarray, fs_hz: float, configured_hz: Literal[50, 60] | None) -> Literal[50, 60]:
    from ...api import _resolve_mains_frequency
    return _resolve_mains_frequency(signal, fs_hz, configured_hz)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent input stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

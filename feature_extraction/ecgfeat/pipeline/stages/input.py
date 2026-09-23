"""Input stage: input contract, resampling and mains-frequency resolution.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...errors import SamplingRateError
from ..context import PipelineContext, StageResult


def _resolve_mains_frequency(
    ecg: np.ndarray,
    fs: int,
    configured: int | str | None,
) -> int:
    if configured not in {None, "auto"}:
        value = int(configured)
        if value not in {50, 60}:
            raise ValueError("mains_freq must be 50, 60, 'auto', or None")
        return value
    if fs <= 122:
        return 50
    sample = np.asarray(ecg, dtype=float)
    spectrum = np.abs(np.fft.rfft(sample - np.mean(sample, axis=1, keepdims=True), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(sample.shape[1], d=1.0 / float(fs))
    powers = {}
    for candidate in (50, 60):
        mask = np.abs(frequencies - candidate) <= 1.0
        powers[candidate] = float(np.median(np.sum(spectrum[:, mask], axis=1)))
    return max(powers, key=powers.get)



def resolve_mains_frequency(signal: np.ndarray, fs_hz: float, configured_hz: Literal[50, 60] | None) -> Literal[50, 60]:
    return _resolve_mains_frequency(signal, fs_hz, configured_hz)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent input stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")

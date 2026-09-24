"""Forwarding module for the raw-input contract.

The implementation moved to :mod:`ecgfeat.pipeline.stages.input` in Phase 3 of the
library migration (docs/library_design/04_module_implementation_guide.md, input
stage).  Every name that was importable here is re-exported with identical object
identity, including ``ECGInputError`` (defined in :mod:`ecgfeat.errors`) and the
private helpers some callers import.
"""

from __future__ import annotations

from .pipeline.stages.input import (
    INDEPENDENT_8_LEADS,
    MIN_RECORD_DURATION_SECONDS,
    MIN_SAMPLING_RATE_HZ,
    STANDARD_12_LEADS,
    ECGInputError,
    _assess_amplitude_calibration,
    _finite_sampling_rate,
    validate_ecg_input,
)

__all__ = [
    "INDEPENDENT_8_LEADS", "MIN_RECORD_DURATION_SECONDS", "MIN_SAMPLING_RATE_HZ", "STANDARD_12_LEADS",
    "ECGInputError", "_assess_amplitude_calibration", "_finite_sampling_rate", "validate_ecg_input",
]

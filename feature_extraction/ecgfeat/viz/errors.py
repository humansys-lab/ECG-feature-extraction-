"""Exceptions raised by the plotting API."""

from __future__ import annotations

from ecgfeat.errors import ECGInputError


class VisualizationInputError(ECGInputError):
    """Signal data do not match the acquisition identity declared by the ECG Record.

    Also raised for lead, beat, time-window and annotation selectors that the
    record cannot satisfy, and for sidecar-backed fields that cannot be read
    from the object that was passed.  It subclasses
    :class:`ecgfeat.errors.ECGInputError` (and therefore ``ValueError``), so
    callers that already catch the public root keep catching it.  ``code``
    carries a stable machine-readable reason.
    """

    def __init__(self, message: str, *, code: str = "visualization_input_error", **details: object) -> None:
        super().__init__(message, code=code, **details)


__all__ = ["VisualizationInputError"]

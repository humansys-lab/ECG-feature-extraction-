"""Public exceptions for the versioned ECG Record API.

The legacy package already exposes ``validation.ECGInputError``.  New
exceptions inherit from it so callers that catch the old root continue to
work while the record API can expose a more useful taxonomy.
"""

from __future__ import annotations

class ECGInputError(ValueError):
    """Base class for public ECG Record and extraction errors."""

    def __init__(self, *args: str, code: str = "ecg_input_error", **details: object) -> None:
        # Preserve legacy ECGInputError(code, message, **details) as well as
        # the new message/keyword-code convention, with one exception identity.
        if len(args) == 2:
            code, message = args
        elif len(args) <= 1:
            message = args[0] if args else "ECG input is invalid"
        else:
            raise TypeError("expected message or (code, message)")
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": dict(self.details)}


class SignalShapeError(ECGInputError):
    """Signal rank, orientation, length, or channel count is invalid."""


class LeadNameError(ECGInputError):
    """Lead names are missing, duplicated, or inconsistent with rows."""


class SamplingRateError(ECGInputError):
    """Sampling rate or resampling configuration is invalid."""


class AmplitudeUnitError(ECGInputError):
    """The declared amplitude unit is unsupported."""


class ConfigurationError(ECGInputError):
    """A method or configuration value is unsupported or inconsistent."""


class RecordValidationError(ECGInputError):
    """An ECG Record violates the supported schema or invariants."""


class RecordIdentityError(ECGInputError):
    """A canonical address targets another record or schema version."""


class AddressSyntaxError(ECGInputError):
    """An ECG Record address or JSON Pointer is malformed."""


class AddressNotFoundError(ECGInputError):
    """A valid JSON Pointer does not exist in the supplied record."""


class MeasurementNotFoundError(ECGInputError):
    """A requested published measurement is not present."""


class MeasurementSelectorError(ECGInputError):
    """A lead or beat selector is invalid for the record axes."""


class ComputationInvariantError(ECGInputError):
    """A computation invariant prevents trustworthy record construction."""


class ECGWarning(UserWarning):
    """Base class for actionable non-fatal ECG API warnings."""


class ECGCompatibilityWarning(ECGWarning):
    """A legacy compatibility behavior was used."""


class ECGDeprecationWarning(FutureWarning, ECGWarning):
    """A legacy API is scheduled for removal."""


__all__ = [
    "ECGInputError", "SignalShapeError", "LeadNameError", "SamplingRateError",
    "AmplitudeUnitError", "ConfigurationError", "RecordValidationError",
    "RecordIdentityError", "AddressSyntaxError", "AddressNotFoundError",
    "MeasurementNotFoundError", "MeasurementSelectorError",
    "ComputationInvariantError", "ECGWarning", "ECGCompatibilityWarning",
    "ECGDeprecationWarning",
]

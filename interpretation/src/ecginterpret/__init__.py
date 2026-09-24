"""ecginterpret: rule-based interpretation over ecg-records measurements.

Research and engineering software; not a medical device.  The output is a
separately versioned Interpretation document (see :mod:`ecginterpret.document`),
never part of the ECG Record measurement contract.
"""

from ._version import __version__
from .document import (
    INTERPRETATION_SCHEMA,
    INTERPRETATION_SCHEMA_VERSION,
    SUPPORTED_RECORD_SCHEMA_MAJORS,
    InterpretationDocument,
    InterpretationInputError,
    interpret_features,
    interpret_record,
)

__all__ = [
    "__version__",
    "INTERPRETATION_SCHEMA",
    "INTERPRETATION_SCHEMA_VERSION",
    "SUPPORTED_RECORD_SCHEMA_MAJORS",
    "InterpretationDocument",
    "InterpretationInputError",
    "interpret_features",
    "interpret_record",
]

"""Legacy extractor forwarding adapter."""

from __future__ import annotations

import warnings
from typing import Any

from ..api import ECGFeatureExtractor as _LegacyExtractor
from ..errors import ECGDeprecationWarning


class ECGFeatureExtractor(_LegacyExtractor):
    """Deprecated legacy object-return extractor.

    The original constructor/call shape is preserved for the migration window.
    New code should use ``ecgfeat.ecg_record`` or ``ECGRecordExtractor``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ecgfeat.compat.ECGFeatureExtractor is deprecated; use ecgfeat.ecg_record or ECGRecordExtractor",
            ECGDeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "legacy ECGFeatureExtractor.extract remains for compatibility; migrate to ecg_record",
            ECGDeprecationWarning,
            stacklevel=2,
        )
        return super().extract(*args, **kwargs)


__all__ = ["ECGFeatureExtractor"]

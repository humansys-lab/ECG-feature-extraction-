"""Deprecated legacy entry point: ``ecgfeat.api.ECGFeatureExtractor``.

The legacy object-return extractor now lives in ``ecgfeat.compat.api_v0``
(which does not warn: importing from ``ecgfeat.compat`` is an explicit opt-in
to the legacy contract).  This spelling, also re-exported as
``ecgfeat.ECGFeatureExtractor``, warns on construction and will be removed no
earlier than ecg-records 0.3.0.  New code uses ``ecgfeat.ecg_record``.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

from ._moved import REMOVAL_RELEASE
from .compat.api_v0 import ECGFeatureExtractor as _CompatExtractor
from .errors import ECGDeprecationWarning


class ECGFeatureExtractor(_CompatExtractor):
    """Deprecated spelling of :class:`ecgfeat.compat.api_v0.ECGFeatureExtractor`."""

    @functools.wraps(_CompatExtractor.__init__)  # keeps the documented signature for introspection
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ecgfeat.ECGFeatureExtractor / ecgfeat.api.ECGFeatureExtractor is deprecated and will be removed no "
            f"earlier than ecg-records {REMOVAL_RELEASE}. Use ecgfeat.ecg_record() for versioned ECG Records, or "
            "ecgfeat.compat.api_v0.ECGFeatureExtractor while you still need the legacy ECGFeatures object.",
            ECGDeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["ECGFeatureExtractor"]

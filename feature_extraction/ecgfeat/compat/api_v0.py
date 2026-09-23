"""Legacy extractor forwarding adapter.

``extract_legacy_features`` is the single routing point for the legacy object-return
extraction (``ecgfeat.api.ECGFeatureExtractor.extract`` delegates here):

* default: the staged pipeline, ``ecgfeat.pipeline.extractor.run_legacy_pipeline``,
  with :data:`~ecgfeat.compat.interpretation_hooks.LEGACY_INTERPRETATION_HOOKS`;
* ``ECGFEAT_LEGACY_ORCHESTRATION=1``: the preserved pre-decomposition orchestration in
  ``ecgfeat.compat._api_v0_legacy_orchestration`` (Phase 3 rollback path).
"""

from __future__ import annotations

import warnings
from typing import Any

from ..api import ECGFeatureExtractor as _LegacyExtractor
from ..errors import ECGDeprecationWarning
from ..pipeline.extractor import legacy_orchestration_requested, run_legacy_pipeline
from .interpretation_hooks import LEGACY_INTERPRETATION_HOOKS


def extract_legacy_features(
    extractor: Any,
    ecg_12lead: Any,
    fs: Any,
    meta: Any = None,
    *,
    lead_names: Any = None,
    amplitude_unit: Any = None,
    gain_uv_per_lsb: Any = None,
    prior_features: Any = None,
) -> Any:
    """Legacy ``ECGFeatures`` for one extraction, configured by ``extractor``'s attributes."""
    if legacy_orchestration_requested():
        from ._api_v0_legacy_orchestration import ECGFeatureExtractor as _PreDecompositionExtractor

        return _PreDecompositionExtractor.extract(
            extractor,
            ecg_12lead,
            fs,
            meta,
            lead_names=lead_names,
            amplitude_unit=amplitude_unit,
            gain_uv_per_lsb=gain_uv_per_lsb,
            prior_features=prior_features,
        )
    return run_legacy_pipeline(
        extractor,
        ecg_12lead,
        fs,
        meta,
        lead_names=lead_names,
        amplitude_unit=amplitude_unit,
        gain_uv_per_lsb=gain_uv_per_lsb,
        prior_features=prior_features,
        hooks=LEGACY_INTERPRETATION_HOOKS,
    ).features


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


__all__ = ["ECGFeatureExtractor", "extract_legacy_features"]

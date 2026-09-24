"""Legacy extractor (compatibility contract) and its routing point.

``extract_legacy_features`` is the single routing point for the legacy object-return
extraction (``ecgfeat.api.ECGFeatureExtractor.extract`` delegates here):

* default: the staged pipeline, ``ecgfeat.pipeline.extractor.run_legacy_pipeline``,
  with :data:`~ecgfeat.compat.interpretation_hooks.LEGACY_INTERPRETATION_HOOKS`;
* ``ECGFEAT_LEGACY_ORCHESTRATION=1``: the preserved pre-decomposition orchestration in
  ``ecgfeat.compat._api_v0_legacy_orchestration`` (Phase 3 rollback path).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..models import ECGFeatures, PatientMeta
from ..pipeline.extractor import legacy_orchestration_requested, run_legacy_pipeline
from ..refinement import RefinementConfig
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


class ECGFeatureExtractor:
    """Legacy object-return extractor (explicit compatibility contract).

    Importing it from ``ecgfeat.compat`` is the acknowledgement that the caller
    needs the legacy ``ECGFeatures`` object; it therefore does not warn.  The
    whole ``ecgfeat.compat`` namespace is removed no earlier than ecg-records
    0.3.0.  New code uses ``ecgfeat.ecg_record``.

    Research/engineering software, not a validated medical device: signal
    standardization -> quality -> multilead QRS -> beat grouping ->
    representative beats -> measurements -> global features.
    """

    def __init__(
        self,
        fs_internal: int | None = None,
        mains_freq: int | str | None = 50,
        lp_hz: float = 40.0,
        enable_pacing: bool = True,
        enable_lead_reversal: bool = True,
        compute_grouping: bool = True,
        enable_hybrid_r_localization: bool = True,
        enable_hybrid_wave_localization: bool = True,
        enable_hybrid_st_measurement: bool = True,
        enable_t_wave_refinement: bool = True,
        st_amplitude_source: str = "analysis",
        refinement: Optional[RefinementConfig] = None,
        input_mode: str = "standard",
    ) -> None:
        if input_mode not in {"standard", "limited"}:
            raise ValueError("input_mode must be 'standard' or 'limited'")
        self.input_mode = input_mode
        if st_amplitude_source not in {"analysis", "calibrated_pr", "adaptive_pr_tp"}:
            raise ValueError("st_amplitude_source must be 'analysis', 'calibrated_pr' or 'adaptive_pr_tp'")
        if refinement is not None and not isinstance(refinement, RefinementConfig):
            raise TypeError("refinement must be a RefinementConfig")
        self.refinement = refinement or RefinementConfig()
        if st_amplitude_source != "analysis" and not enable_hybrid_st_measurement:
            raise ValueError("calibrated_pr requires enable_hybrid_st_measurement=True")
        self.st_amplitude_source = st_amplitude_source
        self.fs_internal = fs_internal  # None = use native input fs
        self.mains_freq = mains_freq
        self.lp_hz = lp_hz
        self.enable_pacing = enable_pacing
        self.enable_lead_reversal = enable_lead_reversal
        self.compute_grouping = compute_grouping
        self.enable_hybrid_r_localization = enable_hybrid_r_localization
        self.enable_hybrid_wave_localization = enable_hybrid_wave_localization
        self.enable_hybrid_st_measurement = enable_hybrid_st_measurement
        self.enable_t_wave_refinement = enable_t_wave_refinement

    def extract(
        self,
        ecg_12lead: np.ndarray,
        fs: float,
        meta: Optional[PatientMeta] = None,
        *,
        lead_names: Optional[List[str]] = None,
        amplitude_unit: Optional[str] = None,
        gain_uv_per_lsb: Optional[float] = None,
        prior_features: Optional[ECGFeatures] = None,
    ) -> ECGFeatures:
        return extract_legacy_features(
            self,
            ecg_12lead,
            fs,
            meta,
            lead_names=lead_names,
            amplitude_unit=amplitude_unit,
            gain_uv_per_lsb=gain_uv_per_lsb,
            prior_features=prior_features,
        )


__all__ = ["ECGFeatureExtractor", "extract_legacy_features"]

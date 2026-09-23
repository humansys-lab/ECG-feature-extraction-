"""Input stage: raw-input contract, resampling, mains frequency and signal domains.

Owns the input contract formerly implemented in ``ecgfeat/validation.py`` (that module now
forwards here and every name keeps its identity) and ``_resolve_mains_frequency``.

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.

Private helpers below were moved verbatim from ``ecgfeat/api.py``
(docs/library_design/01_architecture.md, "Destination of all 49 private helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np

from ..._engine.preprocess import analysis_signal, lowpass_filter, resample_ecg
from ...errors import ECGInputError
from ..context import (
    PipelineContext,
    SignalView,
    StageResult,
    require,
)

MIN_SAMPLING_RATE_HZ = 100.0

MIN_RECORD_DURATION_SECONDS = 1.0

STANDARD_12_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

INDEPENDENT_8_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")



def _finite_sampling_rate(value: object, *, field: str) -> float:
    try:
        sampling_rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ECGInputError(
            "invalid_sampling_rate",
            f"{field} must be a finite number",
            field=field,
            value=repr(value),
        ) from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ECGInputError(
            "invalid_sampling_rate",
            f"{field} must be finite and greater than zero",
            field=field,
            value=sampling_rate,
        )
    if sampling_rate < MIN_SAMPLING_RATE_HZ:
        raise ECGInputError(
            "unsupported_sampling_rate",
            f"{field} must be at least {MIN_SAMPLING_RATE_HZ:g} Hz",
            field=field,
            value=sampling_rate,
            minimum_hz=MIN_SAMPLING_RATE_HZ,
        )
    return sampling_rate



def validate_ecg_input(
    ecg_12lead: object,
    fs: object,
    fs_internal: Optional[int] = None,
    *,
    lead_names: Optional[Iterable[str]] = None,
    amplitude_unit: str = "mV",
    gain_uv_per_lsb: Optional[float] = None,
    minimum_duration_seconds: float = MIN_RECORD_DURATION_SECONDS,
    allow_limited_leads: bool = False,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Validate, reorder and normalize a conventional 12-lead ECG.

    In addition to a complete 12-lead matrix, the eight independent leads
    I, II and V1-V6 are accepted when ``lead_names`` is supplied.  The four
    augmented/derived limb leads are reconstructed from I and II.

    Sampling below 500 Hz remains supported intentionally.  The input
    contract records resampling, but does not require it.
    """

    input_fs = _finite_sampling_rate(fs, field="fs")
    if fs_internal is None:
        internal_fs = int(round(input_fs))
    else:
        internal_value = _finite_sampling_rate(fs_internal, field="fs_internal")
        if not float(internal_value).is_integer():
            raise ECGInputError(
                "invalid_internal_sampling_rate",
                "fs_internal must be an integer sampling rate",
                field="fs_internal",
                value=internal_value,
            )
        internal_fs = int(internal_value)

    try:
        ecg = np.asarray(ecg_12lead, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ECGInputError(
            "invalid_signal_type",
            "ecg_12lead must be a rectangular numeric array",
        ) from exc

    if ecg.ndim != 2:
        raise ECGInputError(
            "invalid_signal_shape",
            "ecg_12lead must have shape [12, n_samples] or [n_samples, 12]",
            shape=tuple(int(v) for v in ecg.shape),
        )

    original_shape = tuple(int(v) for v in ecg.shape)
    normalized_lead_names = (
        [str(name).strip() for name in lead_names]
        if lead_names is not None
        else None
    )
    if allow_limited_leads and (not normalized_lead_names or not 1 <= len(normalized_lead_names) <= 8
                                or any(not name for name in normalized_lead_names)):
        raise ECGInputError("limited_lead_names_required", "limited input requires 1-8 explicit channel names")
    lead_slot_map = None
    expected_count = len(normalized_lead_names) if normalized_lead_names else 12
    transposed = False
    if ecg.shape[0] != expected_count:
        if ecg.shape[1] == expected_count:
            ecg = ecg.T
            transposed = True
        else:
            raise ECGInputError(
                "invalid_lead_count",
                f"expected {expected_count} ECG leads",
                shape=original_shape,
                expected_lead_count=expected_count,
            )
    limb_consistency: Optional[Dict[str, float | bool]] = None
    if normalized_lead_names is not None:
        if len(set(normalized_lead_names)) != len(normalized_lead_names):
            raise ECGInputError(
                "duplicate_lead_names",
                "lead_names contains duplicate entries",
                lead_names=normalized_lead_names,
            )
        lead_map = {
            name: np.asarray(ecg[index], dtype=float)
            for index, name in enumerate(normalized_lead_names)
        }
        supplied = set(lead_map)
        if allow_limited_leads:
            # Slots are computational coordinates only. Unknown channels do
            # not acquire anatomical meaning and no limb leads are derived.
            preferred = ["II", "V2", "V5", "I", "V1", "V3", "V4", "V6", "III", "aVR", "aVL", "aVF"]
            used = supplied & set(STANDARD_12_LEADS)
            free = iter(name for name in preferred if name not in used)
            lead_slot_map = {name: name if name in STANDARD_12_LEADS else next(free)
                             for name in normalized_lead_names}
            output = np.zeros((12, ecg.shape[1]), dtype=float)
            for name, slot in lead_slot_map.items():
                output[STANDARD_12_LEADS.index(slot)] = lead_map[name]
            ecg = output
            lead_mode = "limited_explicit"
        elif set(STANDARD_12_LEADS).issubset(supplied):
            ecg = np.stack([lead_map[name] for name in STANDARD_12_LEADS])
            lead_mode = "complete_12_reordered"
        elif set(INDEPENDENT_8_LEADS).issubset(supplied):
            lead_i = lead_map["I"]
            lead_ii = lead_map["II"]
            derived = {
                "III": lead_ii - lead_i,
                "aVR": -(lead_i + lead_ii) / 2.0,
                "aVL": lead_i - lead_ii / 2.0,
                "aVF": lead_ii - lead_i / 2.0,
            }
            lead_map.update(derived)
            ecg = np.stack([lead_map[name] for name in STANDARD_12_LEADS])
            lead_mode = "independent_8_with_derived_limb_leads"
        else:
            missing = sorted(set(INDEPENDENT_8_LEADS) - supplied)
            raise ECGInputError(
                "unsupported_lead_set",
                "lead_names must contain all 12 standard leads or I, II and V1-V6",
                lead_names=normalized_lead_names,
                missing_independent_leads=missing,
            )
    else:
        lead_mode = "assumed_standard_12"

    n_samples = int(ecg.shape[1])
    if n_samples == 0:
        raise ECGInputError(
            "empty_signal",
            "ecg_12lead must contain at least one sample",
            shape=original_shape,
        )

    finite_mask = np.isfinite(ecg)
    if not bool(np.all(finite_mask)):
        raise ECGInputError(
            "nonfinite_signal",
            "ecg_12lead contains NaN or infinite samples",
            nonfinite_count=int(finite_mask.size - np.count_nonzero(finite_mask)),
            sample_count=int(finite_mask.size),
        )

    try:
        minimum_duration = float(minimum_duration_seconds)
    except (TypeError, ValueError) as exc:
        raise ECGInputError(
            "invalid_minimum_duration",
            "minimum_duration_seconds must be a finite non-negative number",
        ) from exc
    if not np.isfinite(minimum_duration) or minimum_duration < 0.0:
        raise ECGInputError(
            "invalid_minimum_duration",
            "minimum_duration_seconds must be a finite non-negative number",
            value=minimum_duration,
        )
    duration_sec = float(n_samples) / input_fs
    if duration_sec + 1e-12 < minimum_duration:
        raise ECGInputError(
            "record_too_short",
            f"ECG duration must be at least {minimum_duration:g} second",
            duration_sec=duration_sec,
            minimum_duration_sec=minimum_duration,
            n_samples=n_samples,
            fs=input_fs,
        )

    unit = str(amplitude_unit or "mV").strip().lower().replace("μ", "u").replace("µ", "u")
    unit_scale = {
        "mv": 1.0,
        "uv": 1e-3,
        "v": 1e3,
    }.get(unit)
    if unit in {"digital", "adc", "lsb"}:
        try:
            gain = float(gain_uv_per_lsb)
        except (TypeError, ValueError) as exc:
            raise ECGInputError(
                "missing_digital_gain",
                "digital ECG samples require gain_uv_per_lsb",
            ) from exc
        if not np.isfinite(gain) or gain <= 0.0:
            raise ECGInputError(
                "invalid_digital_gain",
                "gain_uv_per_lsb must be finite and greater than zero",
                value=gain,
            )
        unit_scale = gain / 1000.0
    if unit_scale is None:
        raise ECGInputError(
            "unsupported_amplitude_unit",
            "amplitude_unit must be mV, uV, V, or digital",
            amplitude_unit=amplitude_unit,
        )
    ecg = np.asarray(ecg * unit_scale, dtype=float)
    amplitude_calibration = _assess_amplitude_calibration(ecg)
    if lead_mode not in {"independent_8_with_derived_limb_leads", "limited_explicit"}:
        lead_i, lead_ii, lead_iii, avr, avl, avf = ecg[:6]
        reference = max(
            float(np.sqrt(np.mean(lead_i**2) + np.mean(lead_ii**2))),
            1e-6,
        )
        einthoven = float(np.sqrt(np.mean((lead_ii - lead_i - lead_iii) ** 2)))
        goldberger = float(np.sqrt(np.mean(
            np.stack(
                [
                    avr + (lead_i + lead_ii) / 2.0,
                    avl - (lead_i - lead_ii / 2.0),
                    avf - (lead_ii - lead_i / 2.0),
                ]
            ) ** 2
        )))
        normalized_error = max(einthoven, goldberger) / reference
        limb_consistency = {
            "einthoven_rms_mv": einthoven,
            "goldberger_rms_mv": goldberger,
            "normalized_error": float(normalized_error),
            "consistent": bool(normalized_error <= 0.35),
            "severely_inconsistent": bool(normalized_error > 1.0),
        }
    elif lead_mode == "independent_8_with_derived_limb_leads":
        limb_consistency = {
            "einthoven_rms_mv": 0.0,
            "goldberger_rms_mv": 0.0,
            "normalized_error": 0.0,
            "consistent": True,
            "severely_inconsistent": False,
        }

    contract = {
        "valid": True,
        "original_shape": original_shape,
        "normalized_shape": tuple(int(v) for v in ecg.shape),
        "transposed": transposed,
        "duration_sec": duration_sec,
        "input_fs": input_fs,
        "internal_fs": internal_fs,
        "resampling_required": not bool(
            np.isclose(input_fs, float(internal_fs), rtol=0.0, atol=1e-12)
        ),
        "amplitude_input_unit": str(amplitude_unit),
        "amplitude_output_unit": "mV",
        "amplitude_scale_to_mv": float(unit_scale),
        "lead_mode": lead_mode,
        "supplied_lead_names": normalized_lead_names,
        "derived_leads": (
            ["III", "aVR", "aVL", "aVF"]
            if lead_mode == "independent_8_with_derived_limb_leads"
            else []
        ),
        "minimum_sampling_rate_hz": MIN_SAMPLING_RATE_HZ,
        "requires_original_500_hz": False,
        "sampling_capabilities": {
            "interval_quantization_ms": 1000.0 / input_fs,
            "pacing_spike_sensitivity_limited": bool(input_fs < 500.0),
            "sub_500_hz_is_diagnostic_stop": False,
        },
        "limb_lead_consistency": limb_consistency,
        "amplitude_calibration": amplitude_calibration,
    }
    if allow_limited_leads:
        contract.update(lead_slot_map=lead_slot_map, available_leads=list(lead_slot_map.values()),
                        anatomical_names_known=all(name in STANDARD_12_LEADS for name in normalized_lead_names),
                        unavailable_leads=[name for name in STANDARD_12_LEADS if name not in lead_slot_map.values()])
    return ecg, internal_fs, contract



def _assess_amplitude_calibration(ecg: np.ndarray) -> Dict[str, object]:
    """Detect a record whose leads were each rescaled to a common amplitude.

    Some public databases normalise every lead independently to a fixed
    peak-to-peak span. The waveform shape survives, so delineation stays valid,
    but absolute millivolts, inter-lead amplitude ratios and everything derived
    from them - voltage criteria, chamber enlargement, quantitative axis - no
    longer describe the patient. Detecting it once at ingest is what lets the
    diagnostic gate suppress exactly those domains instead of either trusting
    fabricated amplitudes or discarding the whole record.

    The signature is unmistakable: twelve independent physiological leads do
    not land within a fraction of a percent of the same peak-to-peak value.
    """
    ranges = [float(np.ptp(np.asarray(row, dtype=float))) for row in ecg]
    finite = [value for value in ranges if np.isfinite(value) and value > 0.05]
    if len(finite) < 8:
        return {
            "per_lead_normalized": False,
            "reason": "too_few_leads_with_signal",
            "peak_to_peak_spread": None,
            "amplitudes_diagnostic": True,
        }
    smallest = min(finite)
    largest = max(finite)
    spread = (largest - smallest) / largest
    normalized = bool(spread < 0.01)
    return {
        "per_lead_normalized": normalized,
        "reason": "identical_peak_to_peak_across_leads" if normalized else None,
        "peak_to_peak_spread": float(spread),
        "peak_to_peak_mv": float(largest),
        "amplitudes_diagnostic": not normalized,
    }



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


@dataclass(frozen=True, slots=True)
class InputBundle:
    """Engine state produced by the input stage (legacy variable names).

    Fields reference live engine objects; see the ownership-transfer rule in
    ``ecgfeat.pipeline.context``.
    """

    fs_run: Any
    input_contract: Any
    available_leads: Any
    quality_options: Any
    ecg_rs: Any
    mains_freq_run: Any
    ecg_an: Any
    ecg_det: Any


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 1808-1837 (``self`` is ``context.config``)."""
    context = require(context, 'input', 'config', 'request')
    settings = context.config
    amplitude_unit = context.request.amplitude_unit
    meta = context.request.meta
    gain_uv_per_lsb = context.request.gain_uv_per_lsb
    ecg_12lead = context.request.signal
    fs = context.request.fs
    lead_names = context.request.lead_names
    input_unit = (
        amplitude_unit
        or (getattr(meta, "amplitude_unit", None) if meta is not None else None)
        or "mV"
    )
    input_gain = (
        gain_uv_per_lsb
        if gain_uv_per_lsb is not None
        else (getattr(meta, "gain_uv_per_lsb", None) if meta is not None else None)
    )
    ecg, fs_run, input_contract = validate_ecg_input(
        ecg_12lead,
        fs,
        settings.fs_internal,
        lead_names=lead_names,
        amplitude_unit=input_unit,
        gain_uv_per_lsb=input_gain,
        **({"allow_limited_leads": True} if settings.input_mode == "limited" else {}),
    )
    available_leads = input_contract.get("available_leads")
    quality_options = {"available_leads": available_leads} if available_leads is not None else {}

    ecg_rs = resample_ecg(ecg, fs, fs_run)
    mains_freq_run = _resolve_mains_frequency(ecg_rs, fs_run, settings.mains_freq)
    # Measurement signal: baseline removal + mains notch (no LP).
    # Used for all amplitude/boundary measurements to preserve peak heights.
    ecg_an = analysis_signal(ecg_rs, fs_run, mains_hz=mains_freq_run)
    # Detection signal: ecg_an + low-pass filter.
    # Used only for QRS R-peak detection to suppress muscle noise.
    ecg_det = lowpass_filter(ecg_an, fs_run, cutoff_hz=settings.lp_hz, order=4)
    bundle = InputBundle(
        fs_run=fs_run,
        input_contract=input_contract,
        available_leads=available_leads,
        quality_options=quality_options,
        ecg_rs=ecg_rs,
        mains_freq_run=mains_freq_run,
        ecg_an=ecg_an,
        ecg_det=ecg_det,
    )
    context = replace(context, input=bundle, native=SignalView(ecg, input_contract["input_fs"], STANDARD_12_LEADS, "native"))
    return StageResult(context)


__all__ = ['InputBundle', 'run', 'ECGInputError', 'validate_ecg_input', 'resolve_mains_frequency', 'MIN_SAMPLING_RATE_HZ', 'MIN_RECORD_DURATION_SECONDS', 'STANDARD_12_LEADS', 'INDEPENDENT_8_LEADS']

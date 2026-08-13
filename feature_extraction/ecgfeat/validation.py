from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


MIN_SAMPLING_RATE_HZ = 100.0
MIN_RECORD_DURATION_SECONDS = 1.0
STANDARD_12_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
INDEPENDENT_8_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


class ECGInputError(ValueError):
    """A machine-readable input-contract violation."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": dict(self.details),
        }


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
        if set(STANDARD_12_LEADS).issubset(supplied):
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
    if lead_mode != "independent_8_with_derived_limb_leads":
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
    else:
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

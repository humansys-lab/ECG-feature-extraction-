from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from types import MappingProxyType
import re
from typing import Any, Mapping, TextIO

import numpy as np
from scipy.io import loadmat


_COMMENT_FIELDS = {
    "Sex": "sex",
    "Rx": "rx",
    "Hx": "hx",
    "Sx": "sx",
}


@dataclass(frozen=True, slots=True)
class SignalData:
    values: np.ndarray
    sampling_rate: float
    lead_names: tuple[str, ...]
    amplitude_unit: str


@dataclass(frozen=True, slots=True)
class WFDBHeader:
    fs_hz: float
    channel_names: tuple[str, ...]
    units: tuple[str, ...]
    gains: tuple[float | None, ...] = ()
    baseline: tuple[float | None, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    sample_count: int = 0
    signal_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sampling_rate(self) -> float:
        return self.fs_hz

    @property
    def lead_names(self) -> tuple[str, ...]:
        return self.channel_names

    @property
    def amplitude_units(self) -> tuple[str, ...]:
        return self.units


def _sampling_rate(token: str, path: Path) -> int | float:
    """Parse the sampling-frequency part of a WFDB frequency token."""
    value = token.split("/", 1)[0]
    try:
        sampling_rate = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{path}: invalid sampling-frequency token {token!r}"
        ) from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(f"{path}: sampling frequency must be positive")
    return int(sampling_rate) if sampling_rate.is_integer() else sampling_rate


def parse_wfdb_header(path: str | Path, *, text: bool = False) -> dict[str, Any] | WFDBHeader:
    """Read record metadata from a local PhysioNet-style WFDB header.

    This intentionally handles only the record line and comment metadata.
    Callers that need calibration or lead definitions should use a complete
    WFDB parser.
    """
    header_path = Path("<text>") if text else Path(path)
    lines = str(path).splitlines() if text else header_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{header_path}: empty WFDB header")

    record_fields = lines[0].split()
    if len(record_fields) < 4:
        raise ValueError(
            f"{header_path}: first line must contain record, lead count, "
            "sampling frequency, and sample count"
        )

    try:
        lead_count = int(record_fields[1])
        sample_count = int(record_fields[3])
    except ValueError as exc:
        raise ValueError(f"{header_path}: invalid lead or sample count") from exc

    info: dict[str, Any] = {
        "record": record_fields[0],
        "n_leads": lead_count,
        "fs": _sampling_rate(record_fields[2], header_path),
        "n_samples": sample_count,
    }

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line.startswith("#") or ":" not in line:
            continue
        label, raw_value = line[1:].split(":", 1)
        label = label.strip()
        value = raw_value.strip()

        if label == "Age":
            try:
                info["age"] = int(value)
            except ValueError:
                info["age"] = None
        elif label == "Dx":
            info["dx"] = [
                code.strip() for code in value.split(",") if code.strip()
            ]
        elif label in _COMMENT_FIELDS:
            info[_COMMENT_FIELDS[label]] = value

    if text:
        return _typed_header(str(path), "<text>")
    return info


def load_wfdb_mat(
    path: str | Path,
    gain: float = 1000.0,
    *,
    signal_key: str = "val",
) -> np.ndarray:
    """Load a WFDB MATLAB signal matrix as float64 values in millivolts."""
    mat_path = Path(path)
    if not np.isfinite(gain) or gain == 0:
        raise ValueError("gain must be finite and non-zero")

    payload = loadmat(str(mat_path))
    if signal_key not in payload:
        raise KeyError(f"{mat_path}: MATLAB payload has no {signal_key!r} array")

    ecg = np.asarray(payload[signal_key], dtype=np.float64)
    if ecg.ndim != 2:
        raise ValueError(
            f"{mat_path}: expected a 2D ECG matrix, got shape {ecg.shape}"
        )
    return ecg / gain


def read_wfdb_header(source: str | Path | TextIO) -> WFDBHeader:
    """Read the stable subset of a WFDB header used by ``ecg_record``.

    The legacy ``parse_wfdb_header`` mapping remains unchanged for existing
    callers; this typed adapter is the new public I/O boundary.
    """
    content = source.read() if hasattr(source, "read") else Path(source).read_text(encoding="utf-8")
    return _typed_header(content, str(getattr(source, "name", source)))


def _typed_header(content: str, source: str) -> WFDBHeader:
    """Parse a calibrated, named, synchronous single-segment MAT header.

    This deliberately supports a bounded subset of HEADER(5). Unsupported
    skew/multi-frequency layouts must be decoded with a full WFDB reader.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{source}: empty WFDB header")
    record = lines[0].split()
    if len(record) < 4 or "/" in record[0]:
        raise ValueError(f"{source}: explicit single-segment sampling rate and length are required")
    count, samples = int(record[1]), int(record[3])
    fs = float(_sampling_rate(record[2], Path(source)))
    if count < 1 or samples < 1 or len(lines) != count + 1:
        raise ValueError(f"{source}: signal specifications do not match the declared dimensions")
    names, units, gains, baselines, files = [], [], [], [], []
    for line in lines[1:]:
        parts = line.split(maxsplit=8)
        if len(parts) != 9 or not parts[8]:
            raise ValueError(f"{source}: every signal requires an explicit description/lead name")
        if not re.fullmatch(r"(?:16|32|80)(?:\+\d+)?", parts[1]):
            raise ValueError(f"{source}: unsupported MAT signal format, skew, or samples-per-frame")
        gain_token = re.fullmatch(r"([^()/]+)(?:\(([+-]?\d+)\))?/([^/]+)", parts[2])
        if gain_token is None:
            raise ValueError(f"{source}: explicit gain and physical unit are required")
        gain = float(gain_token[1])
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError(f"{source}: signal gain must be finite and positive")
        # HEADER(5): absent baseline equals the explicitly supplied ADC zero.
        baseline = int(gain_token[2]) if gain_token[2] is not None else int(parts[4])
        names.append(parts[8])
        units.append(gain_token[3])
        gains.append(gain)
        baselines.append(float(baseline))
        files.append(parts[0])
    if len(set(names)) != len(names):
        raise ValueError(f"{source}: duplicate signal descriptions/lead names")
    metadata = {}
    for line in content.splitlines():
        if line.lstrip().startswith("#") and ":" in line:
            key, value = line.lstrip()[1:].split(":", 1)
            metadata[key.strip()] = value.strip()
    return WFDBHeader(fs, tuple(names), tuple(units), tuple(gains), tuple(baselines), metadata, samples, tuple(files))


def read_wfdb(signal_path: str | Path, *, header_path: str | Path | None = None) -> SignalData:
    """Load a MAT/header pair without resampling or feature extraction."""
    header_path = Path(header_path) if header_path is not None else Path(signal_path).with_suffix(".hea")
    header = read_wfdb_header(header_path)
    for filename in header.signal_files:
        if (header_path.parent / filename).resolve() != Path(signal_path).resolve():
            raise ValueError("header references a different signal file or multiple signal files")
    signal = load_wfdb_mat(signal_path, gain=1.0)
    if signal.shape != (len(header.lead_names), header.sample_count):
        raise ValueError("MAT signal shape does not match WFDB header")
    scales = {"mV": 1.0, "uV": 0.001, "µV": 0.001, "μV": 0.001, "V": 1000.0}
    if any(unit not in scales for unit in header.units):
        raise ValueError("unsupported WFDB amplitude unit")
    values = (signal - np.asarray(header.baseline)[:, None]) / np.asarray(header.gains)[:, None]
    values *= np.asarray([scales[unit] for unit in header.units])[:, None]
    if not np.all(np.isfinite(values)):
        raise ValueError("MAT signal contains nonfinite calibrated samples")
    return SignalData(values, header.sampling_rate, header.lead_names, "mV")


__all__ = [
    "SignalData", "WFDBHeader", "parse_wfdb_header", "load_wfdb_mat",
    "read_wfdb", "read_wfdb_header",
]

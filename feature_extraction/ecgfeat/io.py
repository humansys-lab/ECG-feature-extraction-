from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


_COMMENT_FIELDS = {
    "Sex": "sex",
    "Rx": "rx",
    "Hx": "hx",
    "Sx": "sx",
}


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


def parse_wfdb_header(path: str | Path) -> dict[str, Any]:
    """Read record metadata from a local PhysioNet-style WFDB header.

    This intentionally handles only the record line and comment metadata.
    Callers that need calibration or lead definitions should use a complete
    WFDB parser.
    """
    header_path = Path(path)
    lines = header_path.read_text(encoding="utf-8").splitlines()
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

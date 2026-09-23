"""Deterministic provenance and input fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class InputFingerprint:
    sha256: str
    n_channels: int
    n_samples: int
    fs_hz: float
    channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlgorithmFingerprint:
    name: str
    version: str
    parameters_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecisionRecord:
    policy: str
    decision: str
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    affected_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Provenance:
    library_version: str
    schema_version: str
    resolved_config: Mapping[str, JsonValue]
    input: InputFingerprint
    algorithms: tuple[AlgorithmFingerprint, ...]
    policy_decisions: tuple[PolicyDecisionRecord, ...]
    created_by: str = "ecgfeat"


def fingerprint_input(signal: np.ndarray, *, fs_hz: float, channels: Sequence[str], amplitude_unit: str = "mV") -> InputFingerprint:
    import numpy as np

    array = np.asarray(signal, dtype="<f8", order="C")
    if array.ndim != 2:
        raise ValueError("signal fingerprint requires a channel-major 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal fingerprint requires finite samples")
    if len(channels) != array.shape[0] or len(set(channels)) != len(channels):
        raise ValueError("signal fingerprint requires one unique name per channel")
    if not np.isfinite(fs_hz) or fs_hz <= 0 or amplitude_unit not in {"mV", "uV"}:
        raise ValueError("invalid sampling frequency or amplitude unit")
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    metadata = _canonical({"channels": list(channels), "fs_hz": float(fs_hz), "shape": list(array.shape), "amplitude_unit": amplitude_unit})
    combined = hashlib.sha256(digest.encode("ascii") + b"\0" + metadata).hexdigest()
    return InputFingerprint(combined, int(array.shape[0]), int(array.shape[1]), float(fs_hz), tuple(str(x) for x in channels))


def clinical_fingerprint(values: Mapping[str, JsonValue], *, algorithm: str, version: str) -> str:
    """Return a stable SHA-256 identity for normalized values and algorithm metadata."""
    payload = {"algorithm": algorithm, "version": version, "values": values}
    return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


def build_provenance(
    *,
    library_version: str,
    schema_version: str,
    resolved_config: Mapping[str, JsonValue],
    input_fingerprint: InputFingerprint,
    algorithms: Sequence[AlgorithmFingerprint],
    policy_decisions: Sequence[PolicyDecisionRecord],
) -> Provenance:
    return Provenance(
        library_version=str(library_version),
        schema_version=str(schema_version),
        resolved_config=dict(resolved_config),
        input=input_fingerprint,
        algorithms=tuple(algorithms),
        policy_decisions=tuple(policy_decisions),
    )


__all__ = [
    "InputFingerprint", "AlgorithmFingerprint", "PolicyDecisionRecord", "Provenance",
    "fingerprint_input", "clinical_fingerprint", "build_provenance",
]

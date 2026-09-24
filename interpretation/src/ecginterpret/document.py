"""Versioned Interpretation document (separate from the ECG Record contract).

``interpret_features`` runs the legacy interpreter and clinical rule engine on
the legacy measurement object; ``interpret_record`` does the same for an ECG
Record plus its raw signal.  The two document members ``interpretation`` and
``clinical`` are exactly what the legacy export carried in its
``interpretation`` and ``metadata.clinical_interpretation`` sections, which
is what the interpretation parity gate compares.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping

from ._version import __version__

INTERPRETATION_SCHEMA = "ecg-interpretation"
INTERPRETATION_SCHEMA_VERSION = "1.0.0"
SUPPORTED_RECORD_SCHEMA_MAJORS = ("1",)
INTENDED_USE = "research_and_engineering_only"
LIMITED_INPUT_STATUS = "not_applicable_limited_lead_measurements"


class InterpretationInputError(ValueError):
    """The input cannot be interpreted as requested (wrong type, mismatch, missing signal)."""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item) and getattr(value, "shape", None) == ():
        return _json_safe(value.item())  # NumPy scalar
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return str(value)


@dataclass(frozen=True)
class InterpretationDocument:
    """Separately versioned interpretation output (JSON-compatible members)."""

    input: Mapping[str, Any]
    interpretation: Mapping[str, Any] | None  # None: not applicable to the input contract
    clinical: Mapping[str, Any]
    schema_version: str = INTERPRETATION_SCHEMA_VERSION
    interpreter_version: str = __version__
    intended_use: str = INTENDED_USE
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": INTERPRETATION_SCHEMA,
            "schema_version": self.schema_version,
            "interpreter_version": self.interpreter_version,
            "intended_use": self.intended_use,
            "input": dict(self.input),
            "interpretation": None if self.interpretation is None else dict(self.interpretation),
            "clinical": dict(self.clinical),
            **({"extensions": dict(self.extensions)} if self.extensions else {}),
        }


def interpret_features(features: Any, *, prior_features: Any = None,
                       input_description: Mapping[str, Any] | None = None) -> InterpretationDocument:
    """Interpret the legacy ``ECGFeatures`` measurement object (compatibility input)."""

    from ecgfeat.compat.models_v0 import ECGFeatures

    from .clinical_rules.engine import analyze_clinical
    from .interpret import interpret

    if not isinstance(features, ECGFeatures):
        raise InterpretationInputError("interpret_features expects the legacy ECGFeatures measurement object")
    description = dict(input_description or {"kind": "legacy_measurement_object"})
    if isinstance(features.metadata.get("limited_lead_capabilities"), Mapping):
        # Limited-lead input is a measurement contract only: no diagnostic rules run.
        return InterpretationDocument(input=description, interpretation=None,
                                      clinical={"status": LIMITED_INPUT_STATUS})
    legacy = interpret(features)
    clinical = analyze_clinical(features, prior_features=prior_features).to_dict()
    return InterpretationDocument(input=description, interpretation=_json_safe(legacy), clinical=_json_safe(clinical))


def _record_mapping(record: Any) -> dict[str, Any]:
    if hasattr(record, "as_dict"):
        return record.as_dict()
    if isinstance(record, Mapping):
        return dict(record)
    raise InterpretationInputError("record must be an ecgfeat ECGRecord or a record mapping")


def interpret_record(record: Any, *, signal: Any = None, prior_features: Any = None) -> InterpretationDocument:
    """Interpret an ECG Record together with the raw signal it was measured from.

    ECG Record schema 1.0 does not carry every quantity the rule engines read,
    so the measurements are re-derived from ``signal`` with the record's own
    resolved configuration, after verifying that ``signal`` is exactly the
    record's raw signal (SHA-256 of little-endian float64 C-order bytes).
    """

    import numpy as np

    from ecgfeat import RefinementConfig
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    document = _record_mapping(record)
    schema_version = str(document.get("schema_version", ""))
    if schema_version.split(".")[0] not in SUPPORTED_RECORD_SCHEMA_MAJORS:
        raise InterpretationInputError(f"unsupported ECG Record schema version {schema_version!r}")
    if signal is None:
        raise InterpretationInputError(
            "interpret_record needs the raw signal: ECG Record 1.0 does not carry the representative-lead, "
            "rhythm and P-wave evidence the rule engines read; pass signal=<the array the record was measured from>")
    samples = np.ascontiguousarray(np.asarray(signal, dtype="<f8"))
    expected = str(document.get("artifacts", {}).get("raw_signal", {}).get("sha256", "")).removeprefix("sha256:")
    if hashlib.sha256(samples.tobytes(order="C")).hexdigest() != expected:
        raise InterpretationInputError("signal does not match the record's raw-signal SHA-256")
    acquisition = document["acquisition"]
    resolved = document["provenance"]["config"]["resolved"]
    limited = acquisition["input_mode"] == "limited"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        extractor = ECGFeatureExtractor(
            fs_internal=None if resolved.get("fs_internal") is None else int(resolved["fs_internal"]),
            mains_freq=resolved.get("mains_frequency_hz"),
            st_amplitude_source=resolved.get("st_amplitude_source", "analysis"),
            refinement=RefinementConfig(**resolved.get("refinement", {})),
            input_mode="limited" if limited else "standard",
        )
    patient = (document.get("metadata") or {}).get("patient")
    meta = None
    if patient:
        from ecgfeat.compat.models_v0 import PatientMeta

        meta = PatientMeta(**{key: patient[key] for key in ("age", "age_days", "sex", "patient_id") if key in patient})
    with warnings.catch_warnings():
        # The compatibility extractor warns its callers; this internal use is not theirs.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        features = extractor.extract(samples, float(acquisition["sample_rate_hz"]), meta,
                                     lead_names=list(acquisition["leads"]), amplitude_unit=acquisition["amplitude_unit"])
    return interpret_features(features, prior_features=prior_features, input_description={
        "kind": "ecg_record",
        "record_id": document["record_id"],
        "record_schema_version": schema_version,
        "raw_signal_sha256": "sha256:" + expected,
    })


__all__ = [
    "INTERPRETATION_SCHEMA", "INTERPRETATION_SCHEMA_VERSION", "SUPPORTED_RECORD_SCHEMA_MAJORS", "INTENDED_USE",
    "InterpretationDocument", "InterpretationInputError", "interpret_features", "interpret_record",
]

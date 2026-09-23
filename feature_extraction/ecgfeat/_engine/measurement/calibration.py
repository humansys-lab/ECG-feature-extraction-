"""Record-weighted empirical QT error risk, separate from detector confidence.

This model describes the reference cohort and pipeline that fitted it. It must
not be interpreted as an externally validated clinical probability. Missing QT
outcomes are excluded from fitting and must be reported separately by callers.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


def confidence_bin(confidence: Any) -> str:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(value) or not 0 <= value <= 1:
        return "missing"
    return str(min(4, int(value * 5)))


@dataclass(frozen=True)
class QTErrorCalibration:
    threshold_ms: float
    risk_by_bin: dict[str, float]
    global_risk: float
    records_by_bin: dict[str, int]
    training_records: tuple[str, ...]
    provenance: dict[str, Any]
    schema_version: int = 1

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]], *, threshold_ms: float = 40.,
            prior_record_weight: float = 3., provenance: Mapping[str, Any] | None = None):
        if not math.isfinite(threshold_ms) or threshold_ms <= 0:
            raise ValueError("threshold_ms must be finite and positive")
        if not math.isfinite(prior_record_weight) or prior_record_weight < 0:
            raise ValueError("prior_record_weight must be finite and nonnegative")
        observations = []
        for row in rows:
            record = str(row["record"])
            try:
                error = float(row["qt_err"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(error):
                observations.append((record, confidence_bin(row.get("qt_conf")), abs(error) > threshold_ms))
        if not observations:
            raise ValueError("at least one finite annotated QT error is required")
        counts = Counter(record for record, _, _ in observations)
        # Each record has unit weight, regardless of beat/lead count.
        global_risk = sum(float(bad) / counts[record] for record, _, bad in observations) / len(counts)
        risks, support = {}, {}
        for key in ["missing", "0", "1", "2", "3", "4"]:
            subset = [(r, bad) for r, b, bad in observations if b == key]
            weight = sum(1 / counts[r] for r, _ in subset)
            bad_weight = sum(float(bad) / counts[r] for r, bad in subset)
            risks[key] = ((bad_weight + prior_record_weight * global_risk) / (weight + prior_record_weight)
                          if weight + prior_record_weight else global_risk)
            support[key] = len({r for r, _ in subset})
        return cls(float(threshold_ms), risks, global_risk, support,
                   tuple(sorted(counts)), dict(provenance or {}))

    def predict(self, confidence: Any) -> dict[str, Any]:
        key = confidence_bin(confidence)
        return {"probability_abs_qt_error_gt_threshold": self.risk_by_bin.get(key, self.global_risk),
                "threshold_ms": self.threshold_ms, "reference_record_count": self.records_by_bin.get(key, 0),
                "confidence_bin": key, "status": "cohort_calibration_requires_external_validation"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if data.get("schema_version") != 1:
            raise ValueError("unsupported calibration schema")
        result = cls(**{**data, "training_records": tuple(data["training_records"])})
        if not math.isfinite(result.threshold_ms) or result.threshold_ms <= 0:
            raise ValueError("invalid threshold")
        if any(not math.isfinite(p) or not 0 <= p <= 1
               for p in [result.global_risk, *result.risk_by_bin.values()]):
            raise ValueError("invalid risk probability")
        return result

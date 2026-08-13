"""Versioned minimal clinical invariants used to block impossible positives.

This is explicitly a safety policy, not a hidden diagnosis producer.  It can
reject or downgrade a model conclusion, but it never adds one.  Thresholds are
centralized here so prompts, guards and future rule adapters cannot silently
drift apart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CLINICAL_SAFETY_POLICY_VERSION = "ecgagent.clinical-safety.v2"


@dataclass(frozen=True)
class ClinicalSafetyPolicy:
    version: str = CLINICAL_SAFETY_POLICY_VERSION
    population: str = "adult"
    bradycardia_upper_exclusive_bpm: float = 60.0
    tachycardia_lower_exclusive_bpm: float = 100.0
    atrial_tachycardia_lower_exclusive_bpm: float = 100.0
    first_degree_av_block_pr_lower_exclusive_ms: float = 200.0
    short_pr_upper_exclusive_ms: float = 120.0
    complete_bundle_hard_exclusion_qrs_ms: float = 115.0
    nonspecific_ivcd_lower_exclusive_ms: float = 110.0
    prolonged_qt_hard_exclusion_qtc_ms: float = 400.0
    short_qt_hard_exclusion_qtc_ms: float = 400.0
    retrograde_min_beats: int = 2
    retrograde_min_leads: int = 2
    urgent_bradycardia_below_bpm: float = 40.0
    urgent_tachycardia_above_bpm: float = 150.0
    urgent_wide_qrs_at_least_ms: float = 120.0
    urgent_qtc_bazett_above_ms: float = 500.0
    sources: tuple[str, ...] = (
        "AHA/ACCF/HRS ECG standardization recommendations",
        "ACC/AHA/HRS bradycardia and conduction-delay definitions",
        "Project diagnostic reference corpus; versioned review required before change",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CLINICAL_SAFETY_POLICY = ClinicalSafetyPolicy()


__all__ = [
    "CLINICAL_SAFETY_POLICY_VERSION",
    "ClinicalSafetyPolicy",
    "DEFAULT_CLINICAL_SAFETY_POLICY",
]

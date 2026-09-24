"""Legacy model names kept outside the new record model.

These are the same classes the engine uses (``ecgfeat._engine.foundation.models``),
re-exported unchanged so existing ``isinstance`` checks and pickles keep
working during the compatibility window.  New code consumes ``ECGRecord``.
"""

from __future__ import annotations

from typing import Any

from .._engine.foundation.models import (
    STANDARD_12_LEADS,
    BeatAnnotation,
    ECGFeatures,
    ECGInterpretation,
    GlobalFeatures,
    GroupFeatures,
    LeadBeatFeatures,
    LeadQuality,
    PatientMeta,
    PWaveBeatAssessment,
    PWaveLeadBoundary,
    QRSCandidateWindow,
    QRSDetectorResult,
    RepresentativeLeadFeatures,
    ResolvedPatientAge,
    WaveBounds,
    resolve_patient_age,
)


def features_from_record(record: Any) -> ECGFeatures:
    """Reject lossy reconstruction of legacy intermediates from a record."""
    raise NotImplementedError(
        "ECGRecord does not retain legacy intermediate state; use the legacy extractor when ECGFeatures is required"
    )


__all__ = [
    "BeatAnnotation", "ECGFeatures", "ECGInterpretation", "GlobalFeatures", "GroupFeatures",
    "LeadBeatFeatures", "LeadQuality", "PWaveBeatAssessment", "PWaveLeadBoundary", "PatientMeta",
    "QRSCandidateWindow", "QRSDetectorResult", "RepresentativeLeadFeatures", "STANDARD_12_LEADS",
    "WaveBounds", "ResolvedPatientAge", "resolve_patient_age", "features_from_record",
]

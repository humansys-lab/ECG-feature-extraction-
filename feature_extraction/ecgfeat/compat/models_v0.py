"""Legacy model aliases kept outside the new record model."""

from __future__ import annotations

from typing import Any

from ..models import *
from ..models import ECGFeatures, GlobalFeatures


def features_from_record(record: Any) -> ECGFeatures:
    """Reject lossy reconstruction of legacy intermediates from a record."""
    raise NotImplementedError("ECGRecord does not retain legacy intermediate state; use the legacy extractor when ECGFeatures is required")


__all__ = ["ECGFeatures", "features_from_record"]

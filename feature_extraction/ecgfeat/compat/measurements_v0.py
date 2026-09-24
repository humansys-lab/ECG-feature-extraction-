"""Measurement helpers the legacy interpretation input still needs.

``ecginterpret`` consumes the legacy ``ECGFeatures`` object during the
compatibility window and recomputes one derived quantity from it.  It imports
that helper from here, never from ``ecgfeat._engine``.  Removed together with
the rest of ``ecgfeat.compat`` (no earlier than ecg-records 0.3.0).
"""

from __future__ import annotations

from .._engine.measurement.features import estimate_initial_qrs_axis_deg

__all__ = ["estimate_initial_qrs_axis_deg"]

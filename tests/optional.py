"""Markers for tests that need an optional distribution.

The legacy ``ECGFeatureExtractor.extract()`` path and the legacy JSON export
include the interpretation, which ships in the separate ``ecginterpret``
distribution.  Without it they raise a targeted ``ImportError`` by design, so
their tests only run where ``ecginterpret`` is installed.  The record path
(``ecgfeat.ecg_record``) never needs it and is tested unconditionally.
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_INTERPRETATION = importlib.util.find_spec("ecginterpret") is not None
needs_interpretation = pytest.mark.skipif(
    not HAS_INTERPRETATION,
    reason='legacy path includes the interpretation: needs ecginterpret (pip install "ecg-records[interpret]")',
)

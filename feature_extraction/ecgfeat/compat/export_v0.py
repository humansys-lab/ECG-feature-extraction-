"""Legacy dictionary export adapter."""

from __future__ import annotations

from typing import Any

from ..export import build_structured_payload as _legacy_structured
from ..export import prepare_json_export as _legacy_prepare
from ..export import to_dict as _legacy_to_dict
from ..record.model import ECGRecord
from ..record.serialize import record_to_dict, to_json_obj


def to_dict(value: Any, *, profile: str | None = None) -> dict[str, Any]:
    if isinstance(value, ECGRecord):
        raise TypeError("ECGRecord is not a legacy export; use record_to_dict or dumps_record")
    return _legacy_to_dict(value, profile=profile)


def prepare_json_export(value: Any, *, profile: str | None = None, **kwargs: Any) -> dict[str, Any]:
    if isinstance(value, ECGRecord):
        raise TypeError("ECGRecord is not a legacy export; use record_to_dict or dumps_record")
    return _legacy_prepare(value, profile=profile, **kwargs)


def build_structured_payload(value: Any, *, profile: str | None = None) -> dict[str, Any]:
    if isinstance(value, ECGRecord):
        raise TypeError("ECGRecord cannot reconstruct a structured legacy payload")
    return _legacy_structured(value)


__all__ = ["to_dict", "prepare_json_export", "build_structured_payload"]

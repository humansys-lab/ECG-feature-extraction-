"""Deprecated legacy export spelling (``ecgfeat.export``).

The historical exporter lives in ``ecgfeat.compat.export_v0``.  Importing this
module is silent (document 03); calling its documented entry points
(``to_dict``, ``prepare_json_export``, ``build_structured_payload``) warns
with the replacement.  Every other name forwards to ``ecgfeat.compat.export_v0``
unchanged.  Removed no earlier than ecg-records 0.3.0.
"""

from __future__ import annotations

from typing import Any

from ._deprecated import warn_legacy
from .compat import export_v0 as _impl

_REPLACEMENTS = {
    "to_dict": "ecgfeat.record_to_dict(record) / ecgfeat.dumps_record(record) for ECG Records, or "
               "ecgfeat.compat.export_v0.to_dict for the legacy payload",
    "prepare_json_export": "ecgfeat.dumps_record(record, profile=...) for ECG Records, or "
                           "ecgfeat.compat.export_v0.prepare_json_export for the legacy payload",
    "build_structured_payload": "ecgfeat.compat.export_v0.build_structured_payload",
}


def to_dict(*args: Any, **kwargs: Any) -> Any:
    warn_legacy("ecgfeat.export.to_dict", _REPLACEMENTS["to_dict"], stacklevel=3)
    return _impl.to_dict(*args, **kwargs)


def prepare_json_export(*args: Any, **kwargs: Any) -> Any:
    warn_legacy("ecgfeat.export.prepare_json_export", _REPLACEMENTS["prepare_json_export"], stacklevel=3)
    return _impl.prepare_json_export(*args, **kwargs)


def build_structured_payload(*args: Any, **kwargs: Any) -> Any:
    warn_legacy("ecgfeat.export.build_structured_payload", _REPLACEMENTS["build_structured_payload"], stacklevel=3)
    return _impl.build_structured_payload(*args, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)

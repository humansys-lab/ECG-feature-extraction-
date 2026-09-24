"""Use-time deprecation wrappers for legacy spellings (document 03, "Deprecation mechanism").

Plain imports stay silent; calling a legacy function emits
``ECGDeprecationWarning`` naming the replacement and the earliest removal
release.  ``ecgfeat.compat`` itself is the explicit, silent legacy contract.
"""

from __future__ import annotations

import functools
import importlib
import warnings
from typing import Any, Callable

from ._moved import REMOVAL_RELEASE
from .errors import ECGDeprecationWarning


def warn_legacy(spelling: str, replacement: str, *, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{spelling} is deprecated and will be removed no earlier than ecg-records {REMOVAL_RELEASE}; "
        f"use {replacement}.",
        ECGDeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated_function(spelling: str, replacement: str, resolve: Callable[[], Callable[..., Any]]):
    """Wrap a lazily resolved legacy callable with a use-time warning."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warn_legacy(spelling, replacement)
        return resolve()(*args, **kwargs)

    wrapper.__name__ = wrapper.__qualname__ = spelling.rsplit(".", 1)[-1]
    wrapper.__doc__ = f"Deprecated: {spelling}. Use {replacement}."
    return wrapper


def _attr(module: str, name: str, *, install_hint: str | None = None) -> Callable[[], Callable[..., Any]]:
    def resolve() -> Callable[..., Any]:
        try:
            return getattr(importlib.import_module(module), name)
        except ModuleNotFoundError as exc:
            if install_hint and (exc.name or "").split(".")[0] in {"ecginterpret", "ecgrecords_viz", "matplotlib"}:
                raise ImportError(install_hint, name=exc.name) from exc
            raise

    return resolve


_VIZ_HINT = 'the plotting helpers moved to the ecg-records-viz distribution; install it with: pip install "ecg-records[viz]"'
_INTERPRET_HINT = ('interpretation moved to the separate ecginterpret distribution; install it with '
                   'pip install "ecg-records[interpret]"')

LEGACY_FUNCTIONS = {
    "to_dict": deprecated_function(
        "ecgfeat.to_dict", "ecgfeat.record_to_dict(record) for ECG Records, or ecgfeat.compat.export_v0.to_dict "
        "for the legacy payload", _attr("ecgfeat.compat.export_v0", "to_dict")),
    "interpret": deprecated_function(
        "ecgfeat.interpret", "ecginterpret.interpret_record(record, signal=...) (pip install \"ecg-records[interpret]\")",
        _attr("ecginterpret.interpret", "interpret", install_hint=_INTERPRET_HINT)),
    "load_wfdb_mat": deprecated_function(
        "ecgfeat.load_wfdb_mat", "ecgfeat.io.read_wfdb()", _attr("ecgfeat.io", "load_wfdb_mat")),
    "parse_wfdb_header": deprecated_function(
        "ecgfeat.parse_wfdb_header", "ecgfeat.io.read_wfdb_header()", _attr("ecgfeat.io", "parse_wfdb_header")),
    "plot_beat": deprecated_function(
        "ecgfeat.plot_beat", "ecgfeat.viz.plot_beat(signal, record, ...)",
        _attr("ecgrecords_viz.legacy", "plot_beat", install_hint=_VIZ_HINT)),
    "plot_beat_all_leads": deprecated_function(
        "ecgfeat.plot_beat_all_leads", "ecgfeat.viz.plot_beat_all_leads(signal, record, ...)",
        _attr("ecgrecords_viz.legacy", "plot_beat_all_leads", install_hint=_VIZ_HINT)),
    "plot_rep_beat": deprecated_function(
        "ecgfeat.plot_rep_beat", "ecgfeat.viz.plot_representative_beat(signal, record, ...)",
        _attr("ecgrecords_viz.legacy", "plot_rep_beat", install_hint=_VIZ_HINT)),
    "plot_quality_summary": deprecated_function(
        "ecgfeat.plot_quality_summary", "ecgfeat.viz.plot_quality_summary(signal, record, ...)",
        _attr("ecgrecords_viz.legacy", "plot_quality_summary", install_hint=_VIZ_HINT)),
}

__all__ = ["warn_legacy", "deprecated_function", "LEGACY_FUNCTIONS"]

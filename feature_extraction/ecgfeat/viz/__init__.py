"""Matplotlib plots of ECG Records and the raw signals they were measured from.

Each function takes ``(signal, record)``: the exact channel-major array that
produced an :class:`ecgfeat.record.ECGRecord` (or its JSON mapping) and that
record. It checks that they belong together and returns ``(figure, axes)``.
Annotations are read from the record; nothing is measured.
``matplotlib.pyplot`` is never imported.

Matplotlib is the optional ``viz`` extra: ``pip install "ecg-records[viz]"``.
``import ecgfeat`` and ``import ecgfeat.viz`` never import it; the plotting
functions load on first access, and if Matplotlib is missing that access
raises ``ImportError`` with the installation hint.
:class:`VisualizationInputError` needs no Matplotlib.

The legacy helpers that plot ``ECGFeatures`` objects live in
:mod:`ecgfeat.viz.legacy` (deprecated) and are imported only on request.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .errors import VisualizationInputError

if TYPE_CHECKING:
    from .plots import (
        plot_beat as plot_beat,
        plot_beat_all_leads as plot_beat_all_leads,
        plot_quality_summary as plot_quality_summary,
        plot_record as plot_record,
        plot_representative_beat as plot_representative_beat,
    )

__all__ = [
    "plot_record",
    "plot_beat",
    "plot_beat_all_leads",
    "plot_representative_beat",
    "plot_quality_summary",
    "VisualizationInputError",
]

_INSTALL_HINT = 'plotting needs Matplotlib; install it with: pip install "ecg-records[viz]"'
_PLOTS = frozenset(__all__) - {"VisualizationInputError"}


def _import_viz_module(name: str = "ecgfeat.viz.plots") -> ModuleType:
    """Import a Matplotlib-backed module of this package, or explain how to install Matplotlib."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "matplotlib" or missing.startswith("matplotlib."):
            raise ImportError(_INSTALL_HINT, name=name) from exc
        raise


def __getattr__(name: str) -> Any:
    if name not in _PLOTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_viz_module(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

"""Plotting entry points for ECG Records (the optional ``viz`` extra).

The implementation ships separately as the ``ecg-records-viz`` distribution
(import name ``ecgrecords_viz``), installed by ``pip install "ecg-records[viz]"``.
This module re-exports its public names lazily: ``import ecgfeat`` and
``import ecgfeat.viz`` never import Matplotlib, and the first attribute access
imports ``ecgrecords_viz``. If that distribution or Matplotlib is missing, the
access raises ``ImportError`` with the installation hint.

Every function takes ``(signal, record)`` and returns ``(figure, axes)``; see
``ecgrecords_viz`` for details.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ecgrecords_viz import (
        VisualizationInputError as VisualizationInputError,
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

_INSTALL_HINT = (
    'the plotting helpers moved to the ecg-records-viz distribution; install it with: pip install "ecg-records[viz]"'
)
# A missing module under one of these roots means the viz extra is not installed.
_EXTRA_ROOTS = ("ecgrecords_viz", "matplotlib")


def _import_viz_module(name: str = "ecgrecords_viz") -> ModuleType:
    """Import a module of the viz distribution, or explain how to install it."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if any(missing == root or missing.startswith(root + ".") for root in _EXTRA_ROOTS):
            raise ImportError(_INSTALL_HINT, name=name) from exc
        raise


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_viz_module(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

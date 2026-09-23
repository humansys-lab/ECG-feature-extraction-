"""Matplotlib plots of ECG Records and the raw signals they were measured from.

Each function takes ``(signal, record)``: the exact channel-major array that
produced an :class:`ecgfeat.record.ECGRecord` (or its JSON mapping) and that
record. It checks that they belong together and returns
``(figure, axes)``. Annotations are read from the record; nothing is measured.
``matplotlib.pyplot`` is never imported. ``ecgfeat.viz`` re-exports these
names for callers of the core ``ecg-records`` distribution.

The legacy helpers that plot ``ECGFeatures`` objects live in
:mod:`ecgrecords_viz.legacy` and are imported only on request.
"""

from ._version import __version__
from .errors import VisualizationInputError
from .plots import (
    plot_beat,
    plot_beat_all_leads,
    plot_quality_summary,
    plot_record,
    plot_representative_beat,
)

__all__ = [
    "plot_record",
    "plot_beat",
    "plot_beat_all_leads",
    "plot_representative_beat",
    "plot_quality_summary",
    "VisualizationInputError",
    "__version__",
]

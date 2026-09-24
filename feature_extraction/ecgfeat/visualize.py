"""Deprecated path of the legacy ``ECGFeatures`` plotting helpers.

They moved unchanged to ``ecgrecords_viz.legacy`` in the separate
``ecg-records-viz`` distribution. Importing this module binds it to that
module object and warns once. If the distribution is not installed, the import
raises ``ImportError`` with the installation hint. New code plots
``(signal, record)`` with ``ecgfeat.viz``.
"""

import sys
import warnings

# ``alias_module`` must be a global here: _caller_stacklevel() then treats this
# module body as shim code and attributes the warning to the importer.
from ._moved import REMOVAL_RELEASE, _caller_stacklevel, alias_module  # noqa: F401
from .viz import _import_viz_module


def _forward() -> None:
    target = _import_viz_module("ecgrecords_viz.legacy")
    warnings.warn(
        f"{__name__} is deprecated: the legacy plotting helpers moved to ecgrecords_viz.legacy "
        f"(distribution ecg-records-viz), and this alias will be removed no earlier than ecg-records "
        f"{REMOVAL_RELEASE}. New code should plot (signal, record) with ecgfeat.viz or ecgrecords_viz.",
        DeprecationWarning,
        stacklevel=_caller_stacklevel(),
    )
    sys.modules[__name__] = target


_forward()

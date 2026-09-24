"""Deprecated path of the legacy ``ECGFeatures`` plotting helpers.

They moved unchanged to ``ecgfeat.viz.legacy``. Importing this module binds it
to that module object and warns once. New code plots ``(signal, record)`` with
``ecgfeat.viz`` (Matplotlib: ``pip install "ecg-records[viz]"``).
"""

import importlib
import sys
import warnings

# ``alias_module`` must be a global here: _caller_stacklevel() then treats this
# module body as shim code and attributes the warning to the importer.
from ._moved import REMOVAL_RELEASE, _caller_stacklevel, alias_module  # noqa: F401


def _forward() -> None:
    target = importlib.import_module("ecgfeat.viz.legacy")
    warnings.warn(
        f"{__name__} is deprecated: the legacy plotting helpers moved to ecgfeat.viz.legacy, and this "
        f"alias will be removed no earlier than ecg-records {REMOVAL_RELEASE}. New code should plot "
        f"(signal, record) with ecgfeat.viz.",
        DeprecationWarning,
        stacklevel=_caller_stacklevel(),
    )
    sys.modules[__name__] = target


_forward()

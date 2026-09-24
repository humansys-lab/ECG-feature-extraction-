"""Lazy access to the separately distributed interpretation package.

Only ``ecgfeat.compat`` may reach ``ecginterpret``, and only when a legacy
entry point that embeds interpretation (the legacy extractor or legacy JSON
export) is actually used.  Measurement and record APIs never need it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

INSTALL_HINT = (
    "interpretation moved to the separate ecginterpret distribution; install it with "
    'pip install "ecg-records[interpret]", or use ecgfeat.ecg_record() for measurements only'
)


def interpretation_module(name: str = "") -> ModuleType:
    target = "ecginterpret" + (f".{name}" if name else "")
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name == "ecginterpret" or (exc.name or "").startswith("ecginterpret."):
            raise ImportError(INSTALL_HINT, name="ecginterpret") from exc
        raise


def lazy_function(module: str, name: str):
    """A stand-in that imports ``ecginterpret.<module>.<name>`` on first call."""

    def proxy(*args, **kwargs):
        return getattr(interpretation_module(module), name)(*args, **kwargs)

    proxy.__name__ = proxy.__qualname__ = name
    proxy.__doc__ = f"Lazy forwarder to ecginterpret.{module}.{name}."
    return proxy


__all__ = ["INSTALL_HINT", "interpretation_module", "lazy_function"]

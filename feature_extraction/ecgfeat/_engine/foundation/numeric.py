from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def trapezoid(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1) -> Any:
    """NumPy 1.x/2.x compatible trapezoidal integration without warnings."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:  # pragma: no cover - NumPy < 2 compatibility
        implementation = np.trapz
    return implementation(y, x=x, dx=dx, axis=axis)


# Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
# (docs/library_design/01_architecture.md, "Destination of all 49 private
# ``api.py`` helpers").
def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None



def _median_or_none(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


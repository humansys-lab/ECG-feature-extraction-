from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1) -> Any:
    """NumPy 1.x/2.x compatible trapezoidal integration without warnings."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:  # pragma: no cover - NumPy < 2 compatibility
        implementation = np.trapz
    return implementation(y, x=x, dx=dx, axis=axis)

"""Construction of descriptive :class:`PolicyEvent` records.

Event construction only reads values; it never mutates engine objects and never
raises on odd inputs, so recording an event cannot change a measurement.  Details
are deeply immutable (``FrozenDetails`` mappings, tuples) and JSON-compatible.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from ..context import FrozenDetails, PolicyEvent


def json_safe(value: Any) -> Any:
    """Return an immutable, JSON-compatible copy of a small evidence value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return FrozenDetails({str(key): json_safe(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(json_safe(item) for item in value)
    return str(value)


def event(
    policy: str,
    decision: str,
    reason_code: str,
    source_ids: Iterable[Any] = (),
    **details: Any,
) -> PolicyEvent:
    return PolicyEvent(
        policy=policy,
        decision=decision,
        reason_code=reason_code,
        source_ids=tuple(str(item) for item in source_ids),
        details=FrozenDetails({key: json_safe(item) for key, item in details.items()}),
    )


__all__ = ["event", "json_safe"]

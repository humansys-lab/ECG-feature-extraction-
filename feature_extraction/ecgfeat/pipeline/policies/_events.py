"""Construction of descriptive :class:`PolicyEvent` records.

Event construction only reads values; it never mutates engine objects and never
raises on odd inputs, so recording an event cannot change a measurement.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from ..context import PolicyEvent


def json_safe(value: Any) -> Any:
    """Return a JSON-compatible copy of a small evidence value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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
        details=MappingProxyType({key: json_safe(item) for key, item in details.items()}),
    )


__all__ = ["event", "json_safe"]

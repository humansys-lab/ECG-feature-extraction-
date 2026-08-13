"""Runtime limits and phase-turn budgeting for bounded agents."""
from __future__ import annotations

import math
import os


TURN_SLACK = 3
LEGACY_TURN_SLACK = 4
DEFAULT_MAX_MODEL_TURNS = 28
DEFAULT_MAX_TOTAL_TOKENS = 400_000
DEFAULT_MAX_WALL_SECONDS = 1200.0
DEFAULT_MAX_CONSECUTIVE_MAX_TOKENS = 3
# One repair attempt put every phase one model slip away from discarding the
# whole record, and the slips clustered on the last permitted turn while token
# and turn budgets sat less than 60% used. The wall clock above absorbs the
# extra generations.
DEFAULT_PHASE_STATE_RETRIES = 2
DEFAULT_MAX_REVISIONS = 4


def phase_turn_limit(
    tool_budget: int,
    *,
    max_tools_per_turn: int = 1,
    phase_state_retries: int = 0,
) -> int:
    budget = max(0, int(tool_budget))
    retries = max(0, int(phase_state_retries))
    width = max(1, int(max_tools_per_turn))
    if budget == 0:
        return LEGACY_TURN_SLACK if width == 1 else 1 + retries
    if width == 1:
        return budget + LEGACY_TURN_SLACK + retries
    return math.ceil(budget / width) + TURN_SLACK + retries


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean flag")


def positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if value <= 0.0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_MAX_TOKENS",
    "DEFAULT_MAX_MODEL_TURNS",
    "DEFAULT_MAX_REVISIONS",
    "DEFAULT_MAX_TOTAL_TOKENS",
    "DEFAULT_MAX_WALL_SECONDS",
    "DEFAULT_PHASE_STATE_RETRIES",
    "bool_env",
    "phase_turn_limit",
    "positive_float_env",
    "positive_int_env",
]

"""Rhythm statement-evidence packaging (moved from the core rhythm measurement module)."""

from __future__ import annotations

from typing import Any, Dict


def build_statement_evidence(
    rhythm_summary: Dict[str, Any],
    pacing_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the rule-engine statement evidence contract used by export."""
    primary_statement = rhythm_summary.get("primary_statement")
    additional_statements = list(rhythm_summary.get("additional_statements") or [])
    statements = list(rhythm_summary.get("statements") or [])
    return {
        "available": True,
        "primary_statement": primary_statement,
        "additional_statements": additional_statements,
        "statements": statements,
        "stop_further_interpretation": bool(
            pacing_context.get("suppress_further_rhythm_interpretation", False)
        ),
        "bypass_remaining_algorithm": bool(
            rhythm_summary.get("bypass_remaining_algorithm", False)
        ),
    }


__all__ = ["build_statement_evidence"]

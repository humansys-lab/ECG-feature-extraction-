"""Deterministic verification of model output against the evidence store.

No LLM participates here.  Self-verification by the model that wrote the text
is not verification, so every check in this package resolves claims back to
ecgfeat's own numbers.
"""
from __future__ import annotations

from .audit import AuditReport, audit_untraceable_numbers, build_numeric_index
from .citations import (
    Finding,
    VerificationPolicy,
    VerificationReport,
    claim_is_qualified,
    verify_output,
    verify_structured,
)
from .numbers import Quantity, extract_quantities, quantities_match

__all__ = [
    "AuditReport",
    "Finding",
    "Quantity",
    "VerificationPolicy",
    "VerificationReport",
    "audit_untraceable_numbers",
    "build_numeric_index",
    "claim_is_qualified",
    "extract_quantities",
    "quantities_match",
    "verify_output",
    "verify_structured",
]

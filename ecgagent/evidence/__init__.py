"""Addressing, unit resolution and caveat attachment for ecgfeat payloads."""
from __future__ import annotations

from .store import EvidenceStore, EvidenceValue
from .diagnostic_contract import DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION
from .ledger import EvidenceLedger
from .pointer import Pointer, infer_unit, resolve_alias

__all__ = [
    "EvidenceStore",
    "EvidenceValue",
    "EvidenceLedger",
    "DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION",
    "Pointer",
    "infer_unit",
    "resolve_alias",
]

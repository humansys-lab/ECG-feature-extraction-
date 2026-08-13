"""The bounded phase loop that drives a model over the ecgfeat tool layer."""
from __future__ import annotations

from .loop import (
    DEFAULT_PHASES,
    AgentResult,
    ECGAgent,
    PhaseRecord,
    PhaseSpec,
    run_agent,
)
from .diagnostic import (
    DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
    DIAGNOSTIC_PROMPT_FINGERPRINT,
    ECGDiagnosticAgent,
    diagnostic_phases,
    run_diagnostic_agent,
)
from .protocol import DEFAULT_DIAGNOSTIC_PROTOCOL
from .diagnostic_ledger import DiagnosticLedger, LedgerPatchError
from .safety_policy import DEFAULT_CLINICAL_SAFETY_POLICY

__all__ = [
    "DEFAULT_PHASES",
    "AgentResult",
    "ECGAgent",
    "ECGDiagnosticAgent",
    "DIAGNOSTIC_AGENT_PROTOCOL_VERSION",
    "DIAGNOSTIC_PROMPT_FINGERPRINT",
    "DEFAULT_DIAGNOSTIC_PROTOCOL",
    "DiagnosticLedger",
    "LedgerPatchError",
    "DEFAULT_CLINICAL_SAFETY_POLICY",
    "PhaseRecord",
    "PhaseSpec",
    "run_agent",
    "run_diagnostic_agent",
    "diagnostic_phases",
]

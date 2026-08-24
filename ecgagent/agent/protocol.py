"""Single versioned configuration surface for the diagnosis Agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..evidence.diagnostic_contract import DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION
from .runtime import (
    DEFAULT_MAX_CONSECUTIVE_MAX_TOKENS,
    DEFAULT_MAX_MODEL_TURNS,
    DEFAULT_MAX_REVISIONS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MAX_WALL_SECONDS,
    DEFAULT_PHASE_STATE_RETRIES,
)
from .safety_policy import CLINICAL_SAFETY_POLICY_VERSION

DIAGNOSTIC_AGENT_PROTOCOL_VERSION = "ecgagent.diagnostic.v41"

DIAGNOSTIC_DOMAINS: tuple[str, ...] = (
    "quality",
    "rhythm_rate",
    "p_av",
    "intervals",
    "axis",
    "conduction_preexcitation",
    "ectopy_pauses",
    "voltage_chamber_r_progression",
    "q_st_t_u",
    "pacing_high_risk",
)

DIAGNOSTIC_TOOLS: tuple[str, ...] = (
    "get_diagnostic_overview",
    "get_global_table",
    "get_measurement",
    "get_lead_table",
    "get_beat_table",
    "search_measurements",
    "get_rhythm_profile",
    "get_atrial_event_table",
    "get_p_assessment_table",
    "get_morphology_groups",
    "get_morphology_map",
    "get_qrs_measurement_bundle",
    "get_interval_waveform_context",
    "get_pacing_profile",
    "get_native_beat_profile",
)

DOMAIN_TOOL_MAP: dict[str, tuple[str, ...]] = {
    "quality": ("get_morphology_map", "get_measurement"),
    "rhythm_rate": ("get_rhythm_profile", "get_beat_table"),
    "p_av": (
        "get_rhythm_profile",
        "get_p_assessment_table",
        "get_atrial_event_table",
    ),
    "intervals": ("get_global_table", "get_interval_waveform_context"),
    "axis": ("get_global_table", "get_lead_table"),
    "conduction_preexcitation": (
        "get_qrs_measurement_bundle",
        "get_morphology_map",
        "get_morphology_groups",
        "get_native_beat_profile",
    ),
    "ectopy_pauses": (
        "get_beat_table",
        "get_morphology_groups",
        "get_rhythm_profile",
    ),
    "voltage_chamber_r_progression": (
        "get_qrs_measurement_bundle",
        "get_morphology_map",
        "get_lead_table",
        "get_native_beat_profile",
    ),
    "q_st_t_u": (
        "get_morphology_map",
        "get_native_beat_profile",
        "get_lead_table",
    ),
    "pacing_high_risk": (
        "get_pacing_profile",
        "get_native_beat_profile",
        "get_morphology_groups",
    ),
}


@dataclass(frozen=True)
class RuntimeLimits:
    max_model_turns: int = DEFAULT_MAX_MODEL_TURNS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS
    max_consecutive_max_tokens: int = DEFAULT_MAX_CONSECUTIVE_MAX_TOKENS
    phase_state_retries: int = DEFAULT_PHASE_STATE_RETRIES
    max_revisions: int = DEFAULT_MAX_REVISIONS


@dataclass(frozen=True)
class PhasePolicy:
    # Default v40 deterministic-fact/pathway-gated small-model workflow. Legacy five-stage
    # ceilings remain
    # below for reproducibility with older traces.
    # Disease-dense records can legitimately fill all six plan candidates.
    # In the 200-record enriched replay, two plans were cut in the final JSON
    # field at 1,200 tokens and one adjudication was cut at 2,200.  The extra
    # headroom is cheaper than repeating the complete record after a
    # deterministic tail truncation. A later disease-dense trace still hit
    # the 2,800-token adjudication ceiling twice on the same record after the
    # exact per-plan schema made every model-owned step mandatory, so v40
    # raises only that terminal structured-output budget.
    compact_plan_max_tokens: int = 1600
    compact_adjudicate_max_tokens: int = 4096
    # This is an absolute ceiling, not a call target. The compact runtime
    # derives the effective budget from the named candidate checks, so simple
    # records may stop after only a few high-information views while complex
    # records can expand when their coverage contract requires it.
    # Candidate-specific narrow views are cheap local reads, and this ceiling
    # now governs two things at once: how many views are prefetched, and how
    # many candidates survive `_compact_bound_candidates_by_view_budget`.
    # The earlier 50-record diverse replay required at most 13 distinct local
    # views, but 13 views dropped eight reference-positive, rule-raised ectopy/IVCD
    # candidates in the disease-enriched replay; their complete shared-view
    # pathways required fourteen or fifteen.  Local evidence reads are cheap,
    # and the model still sees the existing bounded evidence packet.
    compact_tool_ceiling: int = 15
    # Only views feeding owner=model nodes enter this shared round-robin packet;
    # deterministic-only views stay in the program ledger. Eight thousand
    # characters therefore leaves enough cross-lead depth for morphology while
    # avoiding a 12K packet merely because fewer model views share the budget.
    compact_model_evidence_chars: int = 8000
    survey_budget: int = 1
    # Budgets are hard ceilings, not completion targets.  Simple records can
    # finish after the named coverage contract is satisfied; complex records
    # have room to expand without silently discarding unresolved domains.
    investigate_floor: int = 4
    investigate_default: int = 8
    investigate_ceiling: int = 12
    challenge_floor: int = 2
    challenge_default: int = 4
    challenge_ceiling: int = 6
    # Retained for configuration compatibility. Diagnosis phases use named
    # coverage requirements and therefore keep this at zero.
    challenge_minimum_calls: int = 0
    survey_max_tokens: int = 1800
    hypothesize_max_tokens: int = 2600
    investigate_max_tokens: int = 2800
    challenge_max_tokens: int = 2200
    synthesize_max_tokens: int = 5120
    max_planned_tools: int = 8


@dataclass(frozen=True)
class KnowledgePolicy:
    default_max_chunks: int = 8
    navigation_ceiling: int = 4
    challenge_ceiling: int = 12


@dataclass(frozen=True)
class DiagnosticProtocolConfig:
    version: str = DIAGNOSTIC_AGENT_PROTOCOL_VERSION
    evidence_contract_version: str = DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION
    safety_policy_version: str = CLINICAL_SAFETY_POLICY_VERSION
    runtime: RuntimeLimits = field(default_factory=RuntimeLimits)
    phases: PhasePolicy = field(default_factory=PhasePolicy)
    knowledge: KnowledgePolicy = field(default_factory=KnowledgePolicy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_DIAGNOSTIC_PROTOCOL = DiagnosticProtocolConfig()


__all__ = [
    "DEFAULT_DIAGNOSTIC_PROTOCOL",
    "DIAGNOSTIC_AGENT_PROTOCOL_VERSION",
    "DIAGNOSTIC_DOMAINS",
    "DIAGNOSTIC_TOOLS",
    "DOMAIN_TOOL_MAP",
    "DiagnosticProtocolConfig",
    "KnowledgePolicy",
    "PhasePolicy",
    "RuntimeLimits",
]

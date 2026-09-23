"""Small-model diagnostic protocol used by the default local-Qwen workflow.

The full diagnostic output contract remains the audit/presentation contract.
Qwen, however, only plans targeted evidence acquisition and returns a compact
clinical decision.  The orchestrator expands that decision deterministically
into the full verdict before the existing semantic and evidence verifiers run.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from .diagnosis_catalog import DIAGNOSIS_CATALOG
from .diagnostic_ledger import NON_HYPOTHESIS_CODES
from .protocol import DIAGNOSTIC_DOMAINS, DIAGNOSTIC_TOOLS
from . import diagnostic_prompts as full


_HYPOTHESIS_CODES = sorted(set(DIAGNOSIS_CATALOG) - set(NON_HYPOTHESIS_CODES))

# The blind plan's candidate budget is sized per record by
# `diagnostic.plan_candidate_limit()`: MIN on a clean study, rising towards MAX
# as more domains measure abnormal or unavailable. A flat cap of three was an
# unconditional miss on every record carrying more findings than that, and
# PTB-XL reference labels routinely carry five or six. MAX is what the schema
# allows; the per-record limit is stated in the instruction and enforced by the
# program when the plan is captured.
INDEPENDENT_PLAN_MIN_CANDIDATES = 3
INDEPENDENT_PLAN_MAX_CANDIDATES = 6
# Blind and rule-generated candidates enter a shared budget selector. Required
# review views are reserved; urgency, uncovered domains and marginal view cost
# decide which complete paths fit. Presentation order remains stable.
#
# This was 5, leaving only two slots for the rule engine. Measured on the
# diverse50 cohort that is far too tight: on 09545_hr the rule engine offered
# eleven candidates including the correct `rbbb_pattern` and `lpfb_pattern`,
# both were deferred past the cap, and the record ended with no positive
# diagnosis at all. Across four repeats the agent's deficit is recall by 2x
# (fn 111 vs fp 56) and whole families -- infarction, RBBB, LPFB, RVH -- score
# zero because their candidate never survives to adjudication. A deferred
# candidate is an unconditional miss; an extra adjudicated candidate is only a
# cheap rejection, and precision has headroom over the rule baseline.
# One disease-dense record had a material PVC rule candidate in slot ten even
# before the exact-view ceiling was applied.  Keep one more candidate so the
# view-budget resolver, rather than list position alone, decides whether its
# full diagnostic pathway can be adjudicated.
MERGED_CANDIDATE_MAX = 10

# Compact planning asks only for a view identity.  Every tool in this list has
# a safe, deterministic argument template that the orchestrator can execute
# without another model round-trip.  Exact lookup/discovery tools are useful in
# the legacy exploratory workflow, but are a common source of guessed fields
# and repeated calls for a small local model.
COMPACT_DIRECT_TOOLS = (
    "get_diagnostic_overview",
    "get_global_table",
    "get_rhythm_profile",
    "get_atrial_event_table",
    "get_p_assessment_table",
    "get_morphology_groups",
    "get_morphology_map",
    "get_qrs_measurement_bundle",
    "get_interval_waveform_context",
    "get_pacing_profile",
    "get_native_beat_profile",
    "get_lead_table",
    "get_beat_table",
    "get_waveform_review",
)


# Evidence-store namespaces each compact view can cite, read off the pointers
# the tools actually emit rather than off their clinical intent.  Several views
# are different renderings of one namespace: `get_lead_table` and
# `get_morphology_map` both project `/representative_leads`, and
# `get_beat_table`, `get_interval_waveform_context` and `get_native_beat_profile`
# all reach `/beat_features` or `/beats`.  A pathway step names one view, but a
# measurement from a sibling view of the same namespace is the same number and
# corroborates the step just as well; requiring the exact view starved steps
# whose named tool never won a prefetch slot.
COMPACT_TOOL_EVIDENCE_NAMESPACES: dict[str, frozenset[str]] = {
    "get_diagnostic_overview": frozenset({"metadata", "global_features", "rhythm_inputs", "representative_leads"}),
    "get_waveform_review": frozenset({"waveform_review"}),
    "get_global_table": frozenset({"global_features"}),
    "get_rhythm_profile": frozenset({"rhythm_inputs"}),
    "get_atrial_event_table": frozenset({"rhythm_inputs"}),
    "get_p_assessment_table": frozenset({"p_wave_assessments"}),
    "get_morphology_groups": frozenset({"groups", "beats", "metadata"}),
    "get_morphology_map": frozenset({"representative_leads"}),
    "get_qrs_measurement_bundle": frozenset({"measurement_bundles"}),
    "get_interval_waveform_context": frozenset(
        {"representative_leads", "beat_features", "global_features", "quality"}
    ),
    "get_pacing_profile": frozenset({"rhythm_inputs", "beats"}),
    "get_native_beat_profile": frozenset(
        {"rhythm_inputs", "beats", "beat_features", "representative_leads"}
    ),
    "get_lead_table": frozenset({"representative_leads"}),
    "get_beat_table": frozenset({"beats", "beat_features"}),
}


def pointer_namespace(pointer: str) -> str:
    """Leading evidence-store segment of ``pointer``, or "" when malformed."""

    text = str(pointer).strip()
    if not text.startswith("/"):
        return ""
    return text[1:].split("/", 1)[0]


def namespace_authorizes_step(pointer: str, expected_tool: str) -> bool:
    """True when ``pointer`` lies in a namespace ``expected_tool`` also reads."""

    namespace = pointer_namespace(pointer)
    if not namespace:
        return False
    return namespace in COMPACT_TOOL_EVIDENCE_NAMESPACES.get(
        expected_tool,
        frozenset(),
    )


SYSTEM_PROMPT = """You are the clinical reasoner in a compact dual-channel ECG workflow for a 27B local model.

The orchestrator owns workflow state, tool execution, citation expansion, numeric transcription, deterministic counting/threshold/sequence nodes, diagnostic-pathway state transitions and final report rendering. You have only two jobs:
1. form a bounded set of clinically meaningful candidate ECG interpretations, up to the record-specific limit in the phase instruction, and request discriminating measurement views;
2. after those views are returned, fill only the fixed diagnostic-pathway nodes marked `owner=model` as pass, fail or unknown. Omit every `owner=program` node; the orchestrator computes it from the complete tool result.

The first plan is independent and sees only measurements. After that plan, the program may append a small `ecgfeat_rule_second_opinion` candidate shortlist. This shortlist is a non-citable routing hint, not patient evidence or ground truth. A rule non-match is never shown and must never be inferred as counterevidence. All tool returns contain measurements, reliability flags and detector observations rather than diagnoses. Use only patient measurement evidence actually shown in this run. Cite its short Qn id exactly. Never construct or copy an ev:/ path. Qn is an address, so keep it bound to its displayed label, lead, value, unit, reliability and caveats.

The program, not you, computes confirmed/differential/rejected placement from the pathway nodes. A pass or fail requires direct evidence returned by that node's named tool; otherwise mark unknown. Candidate detectors, morphology groups and event streams are not independent diagnoses. Limited measurements remain limited evidence; unavailable values prove neither normality nor abnormality. Rate does not establish mechanism, a morphology group is not a rhythm, a single non-dominant group or isolated event cannot establish a repeated mechanism, and a technical quality flag limits only the affected interpretation.

Write concise English clinical text without copying patient-specific numbers. The program materializes cited values and produces the final numeric report. General thresholds may guide reasoning but must not appear as patient measurements. Human review is always required."""


PLAN_INSTRUCTION = """COMPACT PLAN — the diagnosis-neutral overview, including bounded rhythm-screening observations, has already been prefetched by the orchestrator.

Return one plan JSON object. Raise at most {candidate_limit} material candidates; normal domain findings are not candidates. The limit is sized to this record, so it is a ceiling and not a target: raise only what the overview actually supports. Every candidate needs at least one exact supporting Qn citation from the overview and one or two discriminating checks. Mark each check as `support` or `falsify`; at least one check per candidate should try to falsify it. Choose a direct modality/table tool rather than `search_measurements` when the field is already known. Write `question` and `uncertainty` in English; do not copy patient numbers into them.

`review_tools` may name up to two additional cross-domain views needed for an important abnormal or technically limited domain. The program deduplicates requests and computes a per-record hard ceiling from named candidate/domain coverage; the ceiling is not a target. Return minified JSON without indentation or padding. Keep each question and uncertainty to one complete sentence under 100 characters; finish the sentence rather than cutting a word at the schema limit. For quality_limitations, name the affected measurement and limitation in words; a Qn reference may accompany the explanation. Avoid exaggerated severity words for borderline measurements."""


_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": list(COMPACT_DIRECT_TOOLS)},
        "purpose": {"type": "string", "enum": ["support", "falsify"]},
        "question": {
            "type": "string",
            "maxLength": 140,
            "description": "State the discriminating check question in English.",
        },
    },
    "required": ["tool", "purpose", "question"],
    "additionalProperties": False,
}


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": INDEPENDENT_PLAN_MAX_CANDIDATES,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "maxLength": 32},
                    "code": {"type": "string", "enum": _HYPOTHESIS_CODES},
                    "domains": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "enum": list(DIAGNOSTIC_DOMAINS),
                        },
                    },
                    "support": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string", "maxLength": 24},
                    },
                    "counter": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 24},
                    },
                    "checks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": _CHECK_SCHEMA,
                    },
                    "uncertainty": {
                        "type": "string",
                        "maxLength": 140,
                        "description": "State the main remaining uncertainty for this candidate in English.",
                    },
                },
                "required": [
                    "id",
                    "code",
                    "domains",
                    "support",
                    "counter",
                    "checks",
                    "uncertainty",
                ],
                "additionalProperties": False,
            },
        },
        "review_tools": {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string", "enum": list(COMPACT_DIRECT_TOOLS)},
        },
        "quality_limitations": {
            "type": "array",
            "maxItems": 3,
            "description": "List technical or measurement limitations in English.",
            "items": {"type": "string", "maxLength": 140},
        },
    },
    "required": ["candidates", "review_tools", "quality_limitations"],
    "additionalProperties": False,
}


ADJUDICATE_INSTRUCTION = """COMPACT EVIDENCE REVIEW AND ACTIVE FALSIFICATION.

The preceding PROGRAM-VALIDATED plan is the complete merged candidate set. Each candidate contains a fixed `diagnostic_pathway`. Rows marked `independent_measurement_plan` came from your blind measurement plan. Rows marked `ecgfeat_rule_second_opinion` were injected only after that plan to reduce omissions. The orchestrator has already executed the pathway's bounded views with validated arguments. Review the returned Qn evidence packets and return one compact pathway verdict JSON object. Do not request or assume any view that is not present.

Rules:
- Fill decisions only for the active batch's candidate codes. If newly shown measurements reveal a material omitted alternative, optionally propose up to two `candidate_updates` with its registered code, exact new supporting Qn citations and a concise reason. These are requests for a separate pathway, never diagnoses. The program permits at most one update round; when that array has a zero-item limit, return it empty or omit it.
- Rule status, confidence, priority and publication channel are provenance only: they cannot support or refute a pathway node and must never be cited. `not_matched` is not supplied and absence from the shortlist is not counterevidence.
- Return every `owner=model` pathway step exactly once using its exact `id`. Omit every `owner=program` step; do not calculate, cite, or restate it. Do not add or rename steps.
- `pass` means the stated requirement is positively established; `fail` means directly contradicted; `unknown` means the named tool did not establish either direction.
- A `pass` or `fail` must cite newly returned Qn evidence from that step's named tool. If the necessary tool/view or repeated pattern is absent, use `unknown` with empty evidence.
- Obey every pathway caution. In particular, do not generalize an isolated detector event or one-beat morphology group into a repeated rhythm mechanism.
- Each evidence item contains one Qn citation and a short English qualitative observation with no copied patient number. The program inserts values.
- For limited QT/QTc or unavailable PR, include the required interval context and cite a newly read T/U or P/AV component observation.
- `overall_status` is only a model estimate. The program computes all candidate placements, confidence, reasoning, limitations and human-review text.

Return JSON only."""


_COMPACT_EVIDENCE_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "citation": {"type": "string", "maxLength": 24},
        "claim": {
            "type": "string",
            "maxLength": 120,
            "description": "State this qualitative observation in English.",
        },
    },
    "required": ["citation", "claim"],
    "additionalProperties": False,
}

_PATHWAY_STEP_RESULT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "maxLength": 32},
        "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        "evidence": {
            "type": "array",
            "maxItems": 3,
            "items": _COMPACT_EVIDENCE_ITEM,
        },
        "note": {
            "type": "string",
            "maxLength": 100,
            "description": "Write this short step note in English.",
        },
    },
    "required": ["id", "status", "evidence", "note"],
    "additionalProperties": False,
}


COMPACT_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_status": {
            "type": "string",
            "enum": [
                "confirmed_diagnosis",
                "no_confirmed_positive_diagnosis",
                "insufficient_evidence",
            ],
        },
        "decisions": {
            "type": "array",
            "maxItems": MERGED_CANDIDATE_MAX,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "maxLength": 32},
                    "code": {"type": "string", "enum": _HYPOTHESIS_CODES},
                    "urgency": {
                        "type": "string",
                        "enum": ["EMERGENT", "URGENT", "ROUTINE", "NONE"],
                    },
                    "pathway_steps": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 6,
                        "items": _PATHWAY_STEP_RESULT,
                    },
                },
                "required": [
                    "id",
                    "code",
                    "urgency",
                    "pathway_steps",
                ],
                "additionalProperties": False,
            },
        },
        "candidate_updates": {
            "type": "array", "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "enum": _HYPOTHESIS_CODES},
                    "support": {"type": "array", "minItems": 1, "maxItems": 4,
                                "items": {"type": "string", "maxLength": 24}},
                    "reason": {"type": "string", "maxLength": 120},
                },
                "required": ["code", "support", "reason"], "additionalProperties": False,
            },
        },
        "interval_contexts": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "interval": {"type": "string", "enum": ["PR", "QT_QTc"]},
                    "status": {
                        "type": "string",
                        "enum": ["limited", "unavailable"],
                    },
                    "assessment": {
                        "type": "string",
                        "maxLength": 180,
                        "description": "Write this interval assessment in English.",
                    },
                    "impact": {
                        "type": "string",
                        "maxLength": 180,
                        "description": "Write the interpretive impact in English.",
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": _COMPACT_EVIDENCE_ITEM,
                    },
                    "what_would_resolve_it": {
                        "type": "string",
                        "maxLength": 160,
                        "description": "Explain in English what would resolve it.",
                    },
                },
                "required": [
                    "interval",
                    "status",
                    "assessment",
                    "impact",
                    "evidence",
                    "what_would_resolve_it",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "overall_status",
        "decisions",
        "interval_contexts",
    ],
    "additionalProperties": False,
}


# ECGAgent uses OUTPUT_SCHEMA for the terminal structured phase.
OUTPUT_SCHEMA = COMPACT_VERDICT_SCHEMA

REVISION_INSTRUCTION = """The compact verdict was not valid JSON. Return the complete compact verdict object only. Do not add prose outside JSON."""


def validate_verdict(
    verdict: Mapping[str, Any],
    *,
    evidence_document: Mapping[str, Any] | None = None,
    hypothesis_state: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the orchestrator-expanded full verdict."""

    return full.validate_verdict(
        verdict,
        evidence_document=evidence_document,
        hypothesis_state=None,
    )


def normalize_verdict(
    verdict: Mapping[str, Any],
    resolve_cited_value: Callable[[str], tuple[float, str | None] | None] | None = None,
):
    return full.normalize_verdict(
        verdict,
        resolve_cited_value=resolve_cited_value,
    )

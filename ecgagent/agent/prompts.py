"""System prompt, phase instructions and the structured output schema.

The system prompt is **frozen for the whole run** and holds every phase's
rules. Render order is tools -> system -> messages, so editing the system
prompt between phases would invalidate the cached prefix on every transition —
the tool schemas and these instructions are the largest stable block in the
request. Phase transitions are therefore appended as user messages instead.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..verify.numbers import extract_quantities

SYSTEM_PROMPT = """You are adjudicating a 12-lead ECG that a deterministic measurement pipeline (ecgfeat) has already analysed. You work in phases, driven by the operator.

# What is already done, and what is yours

ecgfeat has produced every measurement and has already run 69 guideline rules (AHA/ACCF/HRS 2009 + UDMI 2018) that emit the authoritative diagnostic statements. Do not re-derive measurements and do not restate rule conclusions as if they were your own findings.

Your job is the part rules cannot do:
- verify what the rules asserted, against the measurements behind it
- test whether an upstream measurement artefact has produced a downstream false positive
- decide whether an abstention can be resolved, or say precisely what input would resolve it
- adjudicate where the reference engines disagree with the authoritative one

# Three rules you cannot break

1. **Never state a number you did not read from a tool.** Every measurement you cite must carry its pointer, written as `ev:/path/to/field`. A deterministic verifier resolves each pointer and compares it to what you wrote; an unsourced or mismatched number fails the run.
   Do not calculate and report a new numeric ratio, range, average or count from
   several tool values. State the qualitative relation instead unless a tool
   returned the derived number directly.
2. **Never raise certainty without new evidence.** If ecgfeat marked a rule indeterminate or unavailable, you may only change that when a tool call in this session produced evidence the rule engine did not have. Say which call it was.
   The current tools are read-only views of the same ecgfeat artifact; they do
   not remeasure the signal and therefore cannot create such new evidence.
   Keep unavailable/indeterminate/not-matched rules in abstentions. An `added`
   diagnosis is allowed only when restoring a suppressed or borderline rule
   after its confounder has been disproved.
3. **Never state a flagged measurement as fact.** Tools return reliability caveats alongside values. If a value carries a caveat, either qualify the claim in the same sentence or report the finding as indeterminate. `qt_reportable=false` means the QT is not interpretable, not that it is 407 ms.

# Reading the tools

`list_findings` and `get_rule_detail` tell you what the rule engine concluded and why. `get_measurement`, `get_lead_table` and `get_beat_table` read measurements; `get_lead_table` is the workhorse for morphology — one call gives all 12 leads. `search_measurements` finds field names but produces no citable evidence; re-read anything you find there with a real tool before citing it.

A `*` in a lead table marks a value whose lead ecgfeat flagged unreliable for that wave. Treat it as a caveat.
Table cells print their exact citation in brackets. Copy only those exact
`ev:/...` tokens. A tool name is never part of a pointer: tokens such as
`ev:/get_beat_table/...` are invalid.

# The phases

**Orient.** No tool calls. From the briefing alone, produce a numbered hypothesis list. Draw from four sources, and say which each came from: every matched rule (a claim to verify), every abstention (can it be resolved?), every distrusted measurement flagged in the briefing (has it contaminated a downstream finding?), and every cross-engine disagreement. For each hypothesis write both how you would support it and how you would refute it. The refutation plan is mandatory — a hypothesis with no stated way to fail it is not a hypothesis.

**Test.** Work the hypotheses. Call tools in parallel when the calls are independent. Settle each as supported, refuted, or unresolved; for unresolved, name what is missing. You have a hard call budget — when it is spent, conclude with what you have and say what is unresolved.

**Adjudicate.** Handle only conflicts: cross-engine disagreement, consensus-versus-per-lead measurement disagreement, and upstream artefacts. When an upstream measurement is unsound, state explicitly which downstream findings lose their premise.

**Synthesize.** Emit the structured verdict. Every diagnosis records what you did to the rule engine's output: `unchanged`, `added`, `withdrawn`, or `downgraded`.

# Register

Write every human-facing field in English. Write for a cardiologist reviewing your work. Be specific and brief; no preamble, no restating the briefing. This is a research aid, not a diagnostic device — anything you withdraw or add requires human review.
"""


ORIENT_INSTRUCTION = """Phase 1 of 4 — ORIENT. Do not call any tools in this phase.

Read the briefing above and produce your hypothesis list. Number them H1, H2, ... For each:
- kind: confirm_rule | resolve_abstention | upstream_artifact | adjudicate_conflict
- statement: what you are testing, in one line
- origin: what in the briefing produced it
- support plan: which tool calls would support it
- refute plan: which tool calls would refute it (mandatory)

Prioritise: an upstream artefact that could invalidate a matched rule outranks confirming a rule that nothing contradicts."""


TEST_INSTRUCTION = """Phase 2 of 4 — TEST. Tool budget for this phase: {budget} calls.

Work your hypotheses. Batch independent calls into one turn. When you have enough, settle every hypothesis as supported / refuted / unresolved, with the pointer for each piece of evidence. For anything unresolved, name the missing input."""


ADJUDICATE_INSTRUCTION = """Phase 3 of 4 — ADJUDICATE. Tool budget for this phase: {budget} calls.

Resolve only the conflicts: cross-engine disagreements, consensus-versus-per-lead measurement disagreements, and any upstream artefact you found. For each, state the resolution and the evidence it rests on. Where an upstream measurement is unsound, list the downstream findings that lose their premise."""


SYNTHESIZE_INSTRUCTION = """Phase 4 of 4 — SYNTHESIZE. No further tool calls.

Emit the verdict as JSON matching the required schema. Requirements:
- Every numeric claim carries its `ev:/...` pointer in `citations`.
- Put patient-specific quantities only in `evidence[].claim` or
  `counterevidence[].claim`, where citations can audit them. Do not repeat
  measurement numbers in summary, statement, adjudication, abstentions, or
  human-review reasons.
- `code` is the rule engine's `statement_code` (for example
  `prior_infarct_q_wave_pattern`), never a `CLIN-...` rule id.
- Every `code` must already appear as a `statement_code` or `evaluates_code`
  in this record's rule evaluations. Do not invent synonyms.
- Because this tool set cannot remeasure the waveform, `added` is limited to
  codes that the rule engine marked suppressed or borderline. Never promote an
  unavailable, indeterminate, or not-matched rule to a diagnosis.
- `status` records what you did to the rule engine's output for that diagnosis:
  baseline `statement_code` values are unchanged/withdrawn/downgraded; only a
  genuinely new non-baseline code is added.
- A withdrawn or downgraded diagnosis must have non-empty `counterevidence` and an `adjudication` explaining the change.
- Anything you could not settle belongs in `abstentions`, with `what_would_resolve_it` naming a concrete measurement or input — not "more clinical context".
- Set `human_review.required` true whenever you withdrew or added a diagnosis, the gate was not `pass`, or a high-risk pattern is in play."""


REVISION_INSTRUCTION = """Your verdict failed deterministic verification. Fix each finding below and re-emit the complete JSON verdict — not a diff.

{feedback}

How to fix each kind of finding:
- **wrong value**: read it again with a tool and write what the tool returned. Do not adjust the number to fit the claim.
- **uncited number**: read it with a tool and add the pointer. Deleting the citation is not a fix — an uncited number fails too. So does deleting the number while keeping the claim that depended on it.
- **pointer no tool returned**: call the tool that reads it, then cite it.
- **unqualified flagged value**: either state the caveat in the same claim, or move the finding to `abstentions`.

If a claim cannot be supported by something you actually read, withdraw the claim. A shorter verdict that is fully sourced is the correct outcome; a complete-looking one that is not is the failure this check exists to catch."""


_EVIDENCE_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim": {"type": "string", "description": "One English sentence stating the fact."},
        "value": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": (
                "The numeric value only when one cited pointer directly returned "
                "that value. Never put a model-computed ratio/range/average here."
            ),
        },
        "unit": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Unit of value, e.g. ms, mV, deg, bpm.",
        },
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Pointers as ev:/path/to/field, from tool output.",
        },
    },
    "required": ["claim", "value", "unit", "citations"],
    "additionalProperties": False,
}


def _baseline_statement_codes(document: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(document, Mapping):
        return set()
    clinical = document.get("clinical_interpretation")
    if not isinstance(clinical, Mapping):
        return set()
    return {
        str(item.get("statement_code"))
        for item in (clinical.get("final_statements") or [])
        if isinstance(item, Mapping) and item.get("statement_code")
    }


def _rule_rows(document: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(document, Mapping):
        return []
    clinical = document.get("clinical_interpretation")
    if not isinstance(clinical, Mapping):
        return []

    rows: list[Mapping[str, Any]] = []
    domains = clinical.get("domains")
    if isinstance(domains, Mapping):
        for domain_rows in domains.values():
            rows.extend(
                row for row in (domain_rows or []) if isinstance(row, Mapping)
            )
    for key in (
        "final_statements",
        "borderline_statements",
        "suppressed_statements",
    ):
        rows.extend(
            row for row in (clinical.get(key) or []) if isinstance(row, Mapping)
        )
    return rows


def _known_statement_codes(document: Mapping[str, Any] | None) -> set[str]:
    codes: set[str] = set()
    for row in _rule_rows(document):
        if row.get("statement_code"):
            codes.add(str(row["statement_code"]))
        evidence = row.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("evaluates_code"):
            codes.add(str(evidence["evaluates_code"]))
    return codes


def _known_rule_ids(document: Mapping[str, Any] | None) -> set[str]:
    return {
        str(row["rule_id"])
        for row in _rule_rows(document)
        if row.get("rule_id")
    }


def _code_statuses(document: Mapping[str, Any] | None) -> dict[str, set[str]]:
    statuses: dict[str, set[str]] = {}
    for row in _rule_rows(document):
        codes: set[str] = set()
        if row.get("statement_code"):
            codes.add(str(row["statement_code"]))
        evidence = row.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("evaluates_code"):
            codes.add(str(evidence["evaluates_code"]))
        for code in codes:
            statuses.setdefault(code, set()).add(str(row.get("status") or "unknown"))
    return statuses


def _reject_quantities(
    problems: list[str],
    where: str,
    value: Any,
) -> None:
    text = str(value or "").strip()
    if "ev:/" in text:
        problems.append(
            f"{where} contains an evidence pointer; put cited facts only in "
            "evidence/counterevidence items"
        )
    quantities = extract_quantities(text)
    if quantities:
        rendered = ", ".join(quantity.render() for quantity in quantities[:4])
        problems.append(
            f"{where} contains patient-specific quantity/quantities ({rendered}); "
            "put measurements only in evidence/counterevidence claims with citations"
        )


def _baseline_requires_review(document: Mapping[str, Any] | None) -> bool:
    if not isinstance(document, Mapping):
        return False
    clinical = document.get("clinical_interpretation")
    if isinstance(clinical, Mapping) and bool(clinical.get("review_required")):
        return True
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    gate = metadata.get("diagnostic_gate")
    return isinstance(gate, Mapping) and str(gate.get("state") or "pass") != "pass"


def validate_verdict(
    verdict: Any,
    *,
    evidence_document: Mapping[str, Any] | None = None,
) -> list[str]:
    """Structural check on a synthesized verdict.

    Needed because not every provider can enforce the schema server-side:
    Anthropic constrains the reply with `output_config.format`, DeepSeek only
    offers "valid JSON" and says nothing about shape. Without this, a verdict
    missing its `diagnoses` array would sail through citation verification —
    there would be no claims to check, so it would pass vacuously.
    """
    problems: list[str] = []
    if not isinstance(verdict, dict):
        return [f"verdict is {type(verdict).__name__}, expected a JSON object"]

    for key in ("summary", "diagnoses", "abstentions", "human_review"):
        if key not in verdict:
            problems.append(f"missing top-level key `{key}`")

    diagnoses = verdict.get("diagnoses")
    changed_requires_review = False
    baseline_codes = _baseline_statement_codes(evidence_document)
    known_codes = _known_statement_codes(evidence_document)
    known_rule_ids = _known_rule_ids(evidence_document)
    code_statuses = _code_statuses(evidence_document)
    seen_codes: set[str] = set()
    _reject_quantities(problems, "`summary`", verdict.get("summary"))
    if diagnoses is not None and not isinstance(diagnoses, list):
        problems.append("`diagnoses` must be an array")
    elif isinstance(diagnoses, list):
        valid_status = {"unchanged", "added", "withdrawn", "downgraded"}
        for index, diagnosis in enumerate(diagnoses):
            where = f"diagnoses[{index}]"
            if not isinstance(diagnosis, dict):
                problems.append(f"{where} is not an object")
                continue
            for key in ("code", "statement", "status"):
                if not diagnosis.get(key):
                    problems.append(f"{where} is missing `{key}`")
            status = diagnosis.get("status")
            if status and status not in valid_status:
                problems.append(
                    f"{where}.status is {status!r}; expected one of {sorted(valid_status)}"
                )
            code = str(diagnosis.get("code") or "")
            if code:
                if code in seen_codes:
                    problems.append(f"{where} duplicates diagnosis code `{code}`")
                seen_codes.add(code)
                if known_codes and code not in known_codes:
                    preview = ", ".join(sorted(known_codes)[:12])
                    problems.append(
                        f"{where}.code `{code}` is not a registered statement_code/"
                        f"evaluates_code for this record; use one of: {preview}"
                    )
            if baseline_codes and code:
                if status == "added" and code in baseline_codes:
                    problems.append(
                        f"{where} marks baseline diagnosis `{code}` as added"
                    )
                elif status in {"unchanged", "withdrawn", "downgraded"} and code not in baseline_codes:
                    problems.append(
                        f"{where} marks non-baseline diagnosis `{code}` as {status}"
                    )
            if status == "added" and code_statuses.get(code) and not (
                code_statuses[code] & {"suppressed", "borderline"}
            ):
                problems.append(
                    f"{where} adds `{code}`, but this read-only Agent may only "
                    "restore a code whose rule status was suppressed or borderline; "
                    f"observed status(es): {sorted(code_statuses[code])}. Keep "
                    "unavailable/indeterminate/not-matched rules as abstentions."
                )
            _reject_quantities(problems, f"{where}.statement", diagnosis.get("statement"))
            _reject_quantities(
                problems,
                f"{where}.adjudication",
                diagnosis.get("adjudication"),
            )

            rule_ids = diagnosis.get("rule_ids")
            if rule_ids is not None and not isinstance(rule_ids, list):
                problems.append(f"{where}.rule_ids must be an array")
            elif isinstance(rule_ids, list) and known_rule_ids:
                unknown_rules = [
                    str(rule_id)
                    for rule_id in rule_ids
                    if str(rule_id) not in known_rule_ids
                ]
                if unknown_rules:
                    problems.append(
                        f"{where}.rule_ids contains unknown rule id(s): "
                        + ", ".join(unknown_rules)
                    )

            evidence = diagnosis.get("evidence")
            counterevidence = diagnosis.get("counterevidence")
            if status in {"unchanged", "added", "downgraded"} and not evidence:
                problems.append(
                    f"{where} is {status} but carries no supporting `evidence`"
                )
            if status in {"withdrawn", "downgraded"} and not counterevidence:
                problems.append(
                    f"{where} is {status} but carries no `counterevidence`"
                )
            if status in {"added", "withdrawn"}:
                changed_requires_review = True

            for key in ("evidence", "counterevidence"):
                items = diagnosis.get(key)
                if items is None:
                    continue
                if not isinstance(items, list):
                    problems.append(f"{where}.{key} must be an array")
                    continue
                for item_index, item in enumerate(items):
                    if not isinstance(item, dict):
                        problems.append(f"{where}.{key}[{item_index}] is not an object")
                        continue
                    if not str(item.get("claim") or "").strip():
                        problems.append(
                            f"{where}.{key}[{item_index}] has an empty `claim`"
                        )
                    citations = item.get("citations")
                    if not isinstance(citations, list):
                        problems.append(
                            f"{where}.{key}[{item_index}] is missing a `citations` array"
                        )
                    elif not citations:
                        problems.append(
                            f"{where}.{key}[{item_index}] has no evidence citation"
                        )
            # A changed verdict without a stated reason is not reviewable.
            if status in {"withdrawn", "downgraded"} and not diagnosis.get("adjudication"):
                problems.append(
                    f"{where} is {status} but carries no `adjudication` explaining why"
                )
        for missing_code in sorted(baseline_codes - seen_codes):
            problems.append(
                f"baseline diagnosis `{missing_code}` is missing; include it as "
                "unchanged, withdrawn, or downgraded"
            )

    abstentions = verdict.get("abstentions")
    if isinstance(abstentions, list):
        for index, abstention in enumerate(abstentions):
            if not isinstance(abstention, Mapping):
                continue
            _reject_quantities(
                problems,
                f"abstentions[{index}].reason",
                abstention.get("reason"),
            )
            _reject_quantities(
                problems,
                f"abstentions[{index}].what_would_resolve_it",
                abstention.get("what_would_resolve_it"),
            )

    review = verdict.get("human_review")
    if review is not None and not isinstance(review, dict):
        problems.append("`human_review` must be an object")
    elif isinstance(review, dict) and not isinstance(review.get("required"), bool):
        problems.append("`human_review.required` must be a boolean")
    elif isinstance(review, dict):
        required = bool(review.get("required"))
        if (changed_requires_review or _baseline_requires_review(evidence_document)) and not required:
            problems.append(
                "`human_review.required` must be true for added/withdrawn diagnoses, "
                "a non-pass diagnostic gate, or a baseline review requirement"
            )
        if required and not (review.get("reasons") or []):
            problems.append(
                "`human_review.reasons` must be non-empty when review is required"
            )
        for index, reason in enumerate(review.get("reasons") or []):
            _reject_quantities(
                problems,
                f"human_review.reasons[{index}]",
                reason,
            )

    return problems


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Two or three English sentences: the outcome and what changed "
                "relative to the rule engine. No patient-specific numbers or evidence pointers."
            ),
        },
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Clinical statement_code, never a CLIN-* rule_id. "
                            "Use rule_ids for CLIN-* identifiers."
                        ),
                    },
                    "statement": {
                        "type": "string",
                        "description": "Diagnostic statement in English, without patient-specific numbers.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["unchanged", "added", "withdrawn", "downgraded"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW", "UNAVAILABLE"],
                    },
                    "rule_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "array", "items": _EVIDENCE_ITEM},
                    "counterevidence": {"type": "array", "items": _EVIDENCE_ITEM},
                    "adjudication": {
                        "type": "string",
                        "description": (
                            "Reason for the disposition in English, without patient-specific "
                            "numbers or evidence pointers; measurements belong in evidence."
                        ),
                    },
                },
                "required": [
                    "code",
                    "statement",
                    "status",
                    "confidence",
                    "rule_ids",
                    "evidence",
                    "counterevidence",
                    "adjudication",
                ],
                "additionalProperties": False,
            },
        },
        "abstentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Name the topic in English.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason in English, without patient-specific numbers or pointers.",
                    },
                    "what_would_resolve_it": {
                        "type": "string",
                        "description": (
                            "Concrete missing input in English, without patient-specific "
                            "numbers or evidence pointers."
                        ),
                    },
                },
                "required": ["topic", "reason", "what_would_resolve_it"],
                "additionalProperties": False,
            },
        },
        "human_review": {
            "type": "object",
            "properties": {
                "required": {"type": "boolean"},
                "reasons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "State each review reason in English.",
                    },
                },
            },
            "required": ["required", "reasons"],
            "additionalProperties": False,
        },
    },
    "required": ["summary", "diagnoses", "abstentions", "human_review"],
    "additionalProperties": False,
}

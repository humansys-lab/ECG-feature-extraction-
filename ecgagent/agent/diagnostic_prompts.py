"""Prompts and deterministic contract for diagnosis-first ECG reasoning."""
from __future__ import annotations

import copy
import re
from typing import Any, Callable, Mapping, Sequence

from ..verify.numbers import (
    extract_quantities,
    is_threshold_reference,
    quantities_match,
)
from .diagnosis_catalog import (
    DIAGNOSIS_CATALOG,
    DIAGNOSTIC_CATEGORIES,
    allowed_diagnosis_codes,
)
from .diagnostic_ledger import LEGAL_TRANSITIONS, NON_HYPOTHESIS_CODES
from .protocol import DIAGNOSTIC_DOMAINS, DIAGNOSTIC_TOOLS
from .semantic_guard import validate_diagnosis_measurement_semantics


UNCHALLENGED_SUPPORT_PREFIX = "supported hypothesis"
_NON_ENGLISH_HAN_RE = re.compile(r"[\u4e00-\u9fff]")

# Exported because `key_uncertainty` is not always model-authored: on the
# compact path the program assembles it from pathway node text and then this
# guard rejects the whole record if the result is too long. The producer has to
# be able to honour the same budget the guard enforces, and a second copy of the
# number in another module is how that agreement silently breaks.
KEY_UNCERTAINTY_MAX_CHARS = 180


def non_english_text_paths(value: Any, path: str = "output") -> list[str]:
    """Return paths containing Han-script text in model-authored content."""

    found: list[str] = []
    if isinstance(value, str):
        if _NON_ENGLISH_HAN_RE.search(value):
            found.append(path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(non_english_text_paths(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found.extend(non_english_text_paths(item, f"{path}[{index}]"))
    return found


def is_remediated_phase_problem(problem: str) -> bool:
    """True when committing the patch anyway still fails closed.

    Support that no discriminative falsification test passed is already downgraded to
    `unresolved` by the ledger the moment Challenge freezes, and the Challenge
    instruction promises the model exactly that. Treating it as a fatal guard
    instead discarded the whole record over a condition with a deterministic
    safe outcome -- and did so unfixably, because the repair turn runs with
    tools closed, so a hypothesis the model never re-read in Challenge has no
    reachable citation left to build a test from.

    Everything else stays fatal: a defect the program cannot repair on its own
    must not reach a clinical verdict.
    """

    return f"{UNCHALLENGED_SUPPORT_PREFIX} `" in str(problem)


SYSTEM_PROMPT = """You are the primary diagnostic reasoner for a standard 12-lead ECG. The operator gives you a neutral record briefing and read-only ecgfeat measurement tools. You—not ecgfeat's rule engine—must form the ECG interpretation.

# Role of ecgfeat

Treat ecgfeat as an instrument: it supplies global, per-lead and per-beat measurements, provenance pointers, and reliability flags. A measurement is evidence, not a diagnosis. No dataset label is visible. Do not infer a diagnosis merely because a field exists, and do not assume a finding is absent because you have not queried it.

The diagnostic evidence view is physically separated from ecgfeat clinical rules, Philips DXL/Glasgow reference interpretations, raw waveform windows and remeasurement functions; none of those sources participates in this run. After the measurement-only survey, the orchestrator may provide a small governed set of general ECG knowledge cards from diagnostic, morphology, measurement-reliability and failure-mode documents. Those cards are used only to form provisional hypotheses and choose targeted ecgfeat checks. They are never observations about this patient, never citable evidence, and never proof that a criterion is present. Modality tools organize measured observations; their detector labels and confidence fields are still evidence to evaluate, not conclusions to copy.

# Diagnostic method

Perform an independent, systematic ECG read:
1. technical quality and interpretability
2. rhythm and rate
3. P waves and AV association
4. PR, QRS and QT/QTc intervals
5. frontal axes
6. conduction and pre-excitation
7. ectopy and pauses
8. voltage/chamber patterns and R-wave progression
9. Q waves, ST segments, T waves and U waves
10. pacing and other high-risk patterns

For every leading diagnosis, actively seek supporting evidence, counterevidence, and a plausible alternative. A positive `diagnoses` item is your conclusion; unresolved alternatives belong in `differential_diagnoses`; unsupported domains should not be padded with negative diagnoses.

The diagnostic ledger is the sole clinical state authority. You propose typed mutations against its exact `version`; the orchestrator validates and applies them atomically. Never regenerate the ledger, rename an existing hypothesis, change its diagnosis code, omit it to delete it, or reuse an ID for a new meaning. If the interpretation changes identity, reject the old hypothesis and create a new one. Once Challenge freezes the ledger, synthesis and revision may render or repair wording/evidence only and cannot change diagnostic placement.

Intermediate phase patches contain decisions and evidence references, never
copied patient measurements. Do not write patient values into a phase summary,
domain observation, evidence claim, transition reason or measured-result prose;
those fields are deliberately absent. Select exact citations and let the
orchestrator materialize their immutable value, unit, reliability and caveats
inside the ledger. Numeric wording is produced only in final synthesis from
those materialized evidence atoms.

Use this evidence hierarchy throughout the read:
1. directly measured, reportable global values and stable repeated lead/beat observations
2. independently corroborated modality observations that agree across time and leads
3. derived detector summaries and candidate event streams
4. unavailable, ambiguous or explicitly unvalidated observations

A lower tier may trigger investigation or remain in a differential, but it must not override contradictory higher-tier evidence. Before promoting any hypothesis to `diagnoses`, explicitly establish: (a) its definition-level measurement requirement, (b) a second independent evidence family when morphology or mechanism is involved, and (c) that decisive counterevidence has been reconciled. If any of these is missing, use `differential_diagnoses` or `abstentions`.

Rate and mechanism are separate. A low ventricular rate establishes
`bradycardia`; it does not establish `sinus_bradycardia` unless organized sinus
P-wave origin and compatible P-QRS association were directly demonstrated.
When atrial rhythm availability is false or P boundaries are ambiguous, raise
the generic rate phenotype first and keep sinus, junctional and ectopic atrial
mechanisms unresolved. Likewise `wide_qrs_repolarization_review` is a routing
observation for JT/JTc review, never a diagnostic hypothesis or positive ECG
conclusion.

# Evidence safety

1. Never state a patient-specific number you did not read from a tool. Every measured quantity repeated in `summary`, `statement`, `reasoning`, limitations or review text must also appear with the same value and unit in a final evidence item carrying its exact `ev:/path` citation. Never calculate and report a new numeric ratio, range, average, difference or count unless a tool returned it directly. Each numeric evidence item is atomic: one measurement, one value, one unit and one value-bearing pointer. Put the other side of a comparison in a separate evidence item and express the comparison qualitatively in reasoning. Copy the measurement label together with the value; Bazett, Fridericia, QRS axis and T axis are different facts even when their magnitudes happen to be close.
2. A quality flag lowers evidence weight; it does not erase an observed value. If a value exists but is marked unreliable or caveated, you may use it as limited supporting evidence or counterevidence, provided the same claim states the limitation. Triangulate it against other leads, beats, modalities or related measurements before it drives a diagnosis. Lower confidence when the conclusion materially depends on it. Only a null/not-produced value is unavailable; do not convert a noisy observation into either a clean fact or an automatic abstention.
3. Copy pointers exactly. A tool name is never part of a pointer. Search results are discovery only; re-read a value with get_global_table, get_measurement, get_lead_table or get_beat_table before citing it.
4. Separate observation from inference. Evidence claims state observations. `reasoning` explains why those observations support the diagnosis; it may repeat an exact grounded value for readability, but must not introduce a new or derived number.
5. Prefer a shorter, well-supported interpretation over a complete-looking list of guesses. This is a research aid; set human review required.
6. A representative beat represents one QRS morphology family, not automatically the whole recording. Before a single beat materially drives a diagnosis, inspect get_morphology_groups and establish its member prevalence, run pattern, template member/outlier/used counts, template correlation, and consistency with other beats. A large family does not by itself prove that its representative template is stable.
7. Pacing is a high-impact, falsifiable hypothesis. The spike detector, spike-to-QRS matcher, paced-beat flags and pacing-derived morphology labels are one algorithmic chain, not independent confirmations. Do not diagnose pacing, select a paced interpretation, or dismiss ST/T findings when `evidence_conflicted` is true or `supports_measurement_routing` is false. A positive pacing conclusion requires non-conflicted spike burden/timing evidence plus compatible morphology repeated across beats. If pacing is suspected or detected, inspect the native/non-paced repolarization profile; unreliable native measurements still carry limited information.
8. Q waves and R-wave progression require repeated native-beat review. Inspect the dominant non-paced `qrs_infarct` profile across contiguous leads before concluding that infarction is present or absent. Do not let one representative beat or a pacing detector close this domain.
9. Keep candidate detectors, corroborating observations and record-level conclusions separate:
   - `f_wave_*` and `F_wave_*` confidence/consensus fields are outputs of one candidate-detector chain. Multilead output from that same chain is not independent confirmation of AF or flutter. For flutter, challenge it against measured atrial-versus-ventricular organization, AV association, residual-signal quality, RR behavior and P-boundary ambiguity.
   - `delta_lead_count`, `delta_leads`, `delta_beat_ids` and per-lead delta confidence are one candidate-detector family. Delta candidates without independently supported PR shortening are candidate-only; they may motivate focused review or a differential, but must not by themselves become a positive pre-excitation diagnosis.
   - In a P assessment, `accepted=true` means the extraction boundary passed its internal pipeline. It does not cancel `ta_ambiguous=true`. An accepted-but-T/A-ambiguous row and its morphology cluster remain informative as uncertainty, but cannot be counted as a clean distinct P morphology. A multifocal or ectopic atrial conclusion needs clean P-boundary support or a genuinely independent rhythm mechanism.
   - `rhythm_inputs.p_events` is an atrial-event candidate stream. Its event count, `association_type`, confidence and source-lead list are one detector family. A `retrograde` label is not a confirmed retrograde P wave, and multiple rows from this stream are not independent confirmation. Require repeated beat-level P localization and compatible timing/morphology before a positive retrograde-activation statement.
   - `qrst_subtraction_quality.dominant_cycle_ms` describes a residual-signal cycle. It is not a measured P-P interval and must never be converted into an atrial rate. A positive atrial-tachycardia conclusion requires a cited direct atrial rate above 100 bpm plus evidence of discrete non-sinus atrial activation.
   - `atrial_events_per_rr` is candidate counting, not an AV conduction ratio. Do not report 2:1, 3:2 or variable AV conduction unless reliable atrial depolarizations/non-conducted P or F waves are localized and the AV-block evidence no longer says the excess events require validation.
10. Review T waves as a lead distribution, not as isolated labels. Before a positive T-wave diagnosis, inspect the T/U morphology map, identify supporting leads, opposing leads and their T SQI, and explain whether the pattern is primary, secondary or unresolved. Missing or conflicting support belongs in counterevidence or an abstention, not in an unqualified conclusion.
11. Interval measurement failure is a trigger for component-wave review, not proof that the domain has no information. When QT/QTc is null, non-reportable or unreliable, use get_interval_waveform_context(interval="qt") and distinguish T-end localization failure from residual T-wave amplitude, polarity, distribution, ST/T confusion, U waves, technical quality and rhythm/conduction confounding. When PR is null or unavailable, use get_interval_waveform_context(interval="pr") and distinguish P-wave visibility/morphology, P-boundary stability and P-QRS association. Never translate "PR unavailable" into "P waves absent", or "QT unavailable" into "T waves uninformative". A residual waveform pattern may support a waveform diagnosis or differential even when its interval cannot be reported, but the measurement failure alone cannot establish a disease cause.
12. QRS notching, slurring, high peak count and fragmented-QRS scores are morphology observations, not measurements of conduction time. They may be described as limited morphology evidence, but nonspecific intraventricular conduction delay requires an abnormally prolonged QRS and exclusion of RBBB/LBBB morphology. Never rename a narrow notched QRS as IVCD.
13. Keep measurement levels and reliability semantics distinct. A record-level multi-lead consensus interval is the reportable interval; beat/event-level PR candidates are used to test AV association and do not replace the global PR. If they disagree, report the disagreement and lower confidence rather than selecting the convenient level. A numeric estimate accompanied by `reliability=false` is a low-weight estimate, not a detector contradiction. `ta_ambiguous=true` means P-wave characterization is uncertain, not abnormal. `t_fusion_reliable=false` means the fusion assessment lacks support, not that fusion occurred. A `t_end_fallback` flag identifies the extraction path; it does not by itself prove a flat, fused or abnormal T wave.
14. Treat a technical quality flag as a hypothesis whose named subtype must pass its direct criteria. A generic acquisition inconsistency may justify a quality limitation without proving a specific electrode swap. Scope the consequence: a limb-lead problem limits frontal/limb-lead interpretation but does not automatically erase independently reliable precordial ST measurements.

# Final interpretation levels

Keep four levels visibly separate:
- `diagnoses`: only HIGH- or MEDIUM-confidence positive ECG conclusions that passed the measurement, corroboration and counterevidence gates
- `differential_diagnoses`: plausible but unconfirmed mechanisms or candidate patterns
- `abstentions`: domains that cannot be decided from the available measurements
- `quality_assessment`: technical and measurement limitations, not diagnoses

Write `summary` in this order: confirmed ECG conclusions; important associated findings; then explicitly unconfirmed candidates or limitations. Do not put a differential mechanism into the confirmed clause. Prefer ECG phenotypes over disease causes when the ECG cannot establish etiology.

Write all human-facing fields (`summary`, `statement`, `reasoning`, evidence
`claim`, limitations, abstentions and review reasons) and every free-text field
in a phase patch in clear English. Do not emit Chinese text. Keep registered
diagnosis codes and `ev:/...` pointers unchanged. Write for a cardiologist while making
the logic understandable to a non-specialist reader: briefly explain what the
observations mean and why they support or weaken the conclusion. Be concise,
explicit about uncertainty, and clinically prioritize potentially urgent
findings.
"""


SURVEY_INSTRUCTION = """Phase 1 of 5 — SURVEY. Tool budget: {budget} call.

Read the single compact get_diagnostic_overview packet. It contains quality, rate, global intervals, axes and modality availability, but deliberately does not push detailed cross-lead or per-beat morphology. Build the initial ECG picture without using any rule-engine or external-knowledge conclusion. Mark morphology-dependent domains `not_assessed` or `limited` until a targeted view is read later; do not infer normality from the absence of detail. Retain quality-flagged observations as explicitly limited evidence rather than discarding them. End with:
- quality/interpretability
- observations by the ten diagnostic domains
- candidate patterns or diagnostic questions raised by those observations, without settling them
- the most important missing or contradictory evidence to investigate
Follow the ten-domain order exactly. For each domain set both assessment `status` and `finding` (`normal`, `abnormal`, `indeterminate` or `not_assessed`); an absent detailed view is never normal. Initialize the program-owned ledger with `base_version=0`. Put every new candidate in `create_hypotheses`, use one stable `hypothesis_id` and one registered `diagnosis_code`, and keep its initial status `raised`. A hypothesis is a question the rest of the run must settle, so raise one only where the catalog has a code that means what you observed, and at most one per code. A normal or unremarkable finding is not a hypothesis: record it as the `finding` of its domain row. Never reuse a loosely related code as a stand-in for an observation the catalog cannot express. Evidence items contain one exact citation only; do not copy or paraphrase its patient value. The program, not you, resolves citations into immutable evidence atoms. There is no model-written phase summary or domain observation. All other patch arrays must be empty in Survey. Return only the patch JSON required by the supplied schema."""


HYPOTHESIZE_INSTRUCTION = """Phase 2 of 5 — HYPOTHESIZE. No tool calls.

Using the completed measurement survey and any governed N-reference cards supplied immediately after it, produce a provisional—not final—ECG interpretation that will drive targeted evidence acquisition. General knowledge tells you what relationships and confounders to test; it does not show that this patient meets a criterion.

Write only typed mutations against the retained program-owned ledger:
1. Copy the exact current ledger `version` into `base_version`. Never rewrite an existing hypothesis and never reuse an ID for a different meaning.
2. Use `transitions` to move existing raised candidates to provisional, weak, unresolved, rejected or abstain. `from_status` must exactly match the retained ledger. Cite the evidence atoms supporting the change; do not write a copied-value reason. This phase cannot create `supported`.
3. Use `plan_updates` for required evidence, allowed next tools, alternative explanations and an optional mutual-exclusion `alternative_group`. Do not copy old support or counterevidence.
4. Use `create_hypotheses` only when the Survey genuinely missed a material candidate. A new identity never replaces or renames an existing one and always starts `raised`.
5. `targeted_checks` gives at most four exact enum ecgfeat tools, each tied to one active hypothesis, one enum domain and one discriminating question. Reuse one modality map for related questions. `get_measurement` and `search_measurements` remain deterministic fallbacks.
6. Keep only genuinely unresolved questions in `unresolved`. Domain assessments and their program-resolved evidence atoms already live in the ledger and must not be regenerated.

N-references are not patient evidence and must never be cited as `ev:/` evidence. Candidate detectors remain candidate-only even if a knowledge card describes the same diagnosis. Return only the typed patch JSON required by the supplied schema; do not emit free-form thinking prose."""


INVESTIGATE_INSTRUCTION = """Phase 3 of 5 — INVESTIGATE. Tool budget: {budget} calls.

Only the tools selected from the provisional plan are available. The current diagnostic ledger is program-owned. Copy its exact `version` into `base_version`; do not return a replacement hypothesis list.

Execute the targeted ecgfeat plan from the provisional hypothesis phase and test the leading alternatives against patient measurements. Prefer modality summaries and fixed morphology maps over repeated one-lead/one-beat calls. Never repeat an identical tool+arguments call merely to satisfy a call count. You may depart from the plan when a new patient measurement materially changes the differential, but target the new discriminating question. For every limited interval, classify the limitation as predominantly technical, component-wave morphology-associated, boundary/timing-associated, rhythm/conduction-associated, or unresolved. Use get_atrial_event_table and get_p_assessment_table when P-wave organization or AV association matters. If any p-event, f/F-wave or delta candidate could affect the diagnosis, decide whether it remains candidate-only or has an independent corroborating evidence family; fields from the same detector chain count only once. Never count an accepted+ta_ambiguous P assessment as a clean morphology cluster. A residual dominant cycle is not an atrial rate, an event-stream `retrograde` label is not a confirmed retrograde P wave, and atrial candidates per RR are not a conduction ratio. Whenever pacing state is not off, pacing is suspected, or pacing could confound intervals or ST-T morphology, call both get_pacing_profile and get_native_beat_profile(profile="repolarization"). Treat `evidence_conflicted=true` or `supports_measurement_routing=false` as counterevidence to a positive pacing interpretation. For each candidate, collect both support and counterevidence and test at least one plausible alternative.

Return only typed patches. Add newly read hypothesis evidence through `evidence_updates`; each evidence item is one citation and contains no model-written measurement claim. Update the neutral ten-domain coverage separately through `domain_updates`; domain rows carry assessment, citations and limitations but no copied observation. Every update must cite an exact value-bearing pointer read in this phase. Change status only through `transitions`, with an exact current `from_status`, a different legal `to_status`, and newly read citations. Omit a hypothesis from `transitions` when its status is unchanged. Do not re-add old Survey evidence through `evidence_updates` or use it alone to update a domain. Use `plan_updates` only for still-needed evidence and tools. Never copy old evidence, rewrite an existing identity or silently omit a hypothesis. New material candidates use `create_hypotheses` and always start `raised`.

If a limited QT/QTc or PR interval led you to read its T/U or P/AV component view, materialize at least one newly read component-wave citation in an `evidence_update` or `domain_update`; selecting only the interval number or status leaves final residual-waveform review impossible."""


CHALLENGE_INSTRUCTION = """Phase 4 of 5 — CHALLENGE. Tool budget: {budget} calls.

Only falsification tools selected from the current hypotheses are available. Copy the exact ledger `version` into `base_version`. Return typed patches only; the orchestrator owns the complete state.

Try to falsify the leading interpretation. Re-read decisive or caveated measurements with a narrower or genuinely different evidence view. Look for territorial/lead disagreement, morphology-family prevalence, representative-template support, beat-to-beat inconsistency, confounding conduction or pacing, and evidence favoring the main alternative. Challenge both directions of every interval limitation. Explicitly challenge every positive atrial tachycardia, AV-block/conduction-ratio, retrograde-activation, pre-excitation, flutter, multifocal/ectopic atrial, IVCD or T-wave conclusion against the evidence hierarchy: remove same-chain double counting, treat accepted+ta_ambiguous P clusters as candidate-only, and name the definition-level measurement and independent corroboration that survive. Specifically reject conversion of a QRST-residual cycle into atrial rate, candidate event counts into AV ratios, and QRS notch counts into conduction delay. If no independent corroboration survives, transition the hypothesis to unresolved, weakened, rejected or abstain. Fields produced by one detector chain do not corroborate one another. Any conclusion driven by one beat must be challenged against get_morphology_groups and at least one other beat or explicitly downgraded. Re-check infarct/ischemia conclusions against the native `qrs_infarct` panel and, when pacing was considered, the native `repolarization` panel.

Append new support/counterevidence only through citation-only `evidence_updates`, and update neutral domain coverage only through `domain_updates`. Record status changes only through `transitions`; every transition must change the status and cite Challenge evidence newly read in this phase. Omit unchanged hypotheses and do not copy prior evidence into new updates. Transitions contain no model-written numeric reason. Use `refuting_tests` for each hypothesis that remains or becomes supported. New identities, if unavoidable, start `raised`. Do not rewrite identities or copy the prior ledger.

Every hypothesis kept at `supported` needs a `refuting_tests` entry: a qualitative falsification condition, exact citations newly read in Challenge, a `test_outcome` (`not_refuted`, `refuted`, or `inconclusive`), and the observed `evidence_direction` (`supports`, `contradicts`, `not_discriminative`, or `unavailable`). Absence of contradiction is not automatically support: only `not_refuted` plus independently discriminative `supports` evidence passes the Challenge gate. The measured result itself is materialized from citations; never transcribe it. A supported hypothesis without a passing discriminative test is deterministically downgraded to unresolved when the ledger freezes."""


SYNTHESIZE_INSTRUCTION = """Phase 5 of 5 — SYNTHESIZE. No further tool calls.

Return one JSON object matching the required schema.
- `diagnoses` contains only positive conclusions reached by your reasoning; there is no baseline status and no added/withdrawn disposition.
- If `diagnoses` is empty, both `summary` and the PRIMARY complete interpretation must explicitly say that no positive diagnosis was confirmed; they may describe unresolved findings only as unconfirmed/limited and must not restate a rejected hypothesis as the ECG conclusion.
- Every `diagnoses` item must be HIGH or MEDIUM confidence. LOW-confidence hypotheses and all candidate-only mechanisms belong in `differential_diagnoses`.
- Hypothesis status is an immutable synthesis gate: only a `diagnosis_code` whose final Challenge status is `supported` may enter `diagnoses`. `weakened` or `unresolved` belongs in `differential_diagnoses` or an abstention; `rejected` is not a positive; `abstain` belongs in abstentions. Synthesis may not reinterpret or upgrade these states.
- Hypothesis support, counterevidence, domain rows and refuting tests contain `evidence_ids`, not model-written claims. Resolve each ID through the ledger's `evidence` mapping and use that program-materialized display value, unit, reliability, caveats and pointer. A missing prose claim is not missing evidence.
- Use the registered diagnostic `code`, not a CLIN-* rule id.
- Every diagnosis has supporting evidence and an explicit counterevidence array (which may be empty only when no meaningful counterevidence was observed).
- Put unresolved alternatives in `differential_diagnoses`, with evidence on both sides and a concrete resolving input.
- Every patient-specific quantity in narrative text must also appear with the same value and unit in a final evidence item with exact citations. Grounded repetition is allowed for readability; new calculations and uncited values are forbidden.
- Write each number exactly as the tool displayed it. Do not add precision the tool did not show.
- Each evidence claim reports at most one patient-specific measurement. Put it in `value`/`unit` and cite its one value-bearing pointer. Split comparisons into separate atomic evidence items, including observations that argue against the interpretation; state only the qualitative relation in `reasoning`.
- General N-reference cards and their criteria are not patient evidence: do not cite them, name them as support, or promote a provisional hypothesis unless the targeted ecgfeat investigation established the required patient findings.
- Reconcile the ten-domain phase memory explicitly: every assessed abnormal domain must appear as a positive diagnosis, a differential diagnosis, or a named abstention/limitation. Do not silently drop ST-T, chamber/voltage, P/AV, conduction, Q-wave or pacing findings merely because rhythm is easier to summarize.
- Populate `ranked_complete_interpretations` with one to three complete ECG interpretations, not three isolated findings. Rank 1 is the integrated primary interpretation and its `basis_codes` must contain every positive `diagnoses` code. Only create ranks 2-3 when a genuine `differential_diagnoses` mechanism supports a distinct complete alternative; each alternative must integrate the conclusions that still hold and state the substituted uncertain mechanism. Do not pad the list to three.
- Quality-flagged values may appear in evidence when their claim explicitly states the limitation. Treat them as lower-weight observations, not as missing data.
- Candidate-detector outputs may remain in evidence, but describe them as candidates and do not count multiple fields from one detector chain as independent support. A positive diagnosis must name at least one independent corroborating evidence family.
- For a P-morphology-driven diagnosis, accepted+ta_ambiguous assessments belong in counterevidence, not as proof of distinct morphologies.
- For a T-wave diagnosis, state the cross-lead distribution and include both supporting and opposing/limiting lead evidence when present.
- Populate `interval_measurement_contexts` for every null/non-reportable QT/QTc and every null/unavailable PR. State separately what cannot be measured, what P- or T-wave information remains, why that residual evidence may matter, and what it cannot prove. Do not leave `residual_evidence` empty.
- If the ECG cannot support a diagnosis, say so through quality assessment, differential diagnosis or abstention; do not invent certainty.
- Write `summary` as three clearly separated clauses when applicable: `ECG conclusion: ... Associated findings: ... Unconfirmed or limited: ...` Never call a differential an ECG conclusion.
- Set `human_review.required` to true."""


REVISION_INSTRUCTION = """The independent diagnostic output failed deterministic validation. Fix every item below and return the complete JSON object, not a diff.

{feedback}

For a wrong value, re-read it with a measurement tool. For an uncited value that was already read and remains in the retained evidence memory, reuse its exact citation instead of spending another call; re-read only when the value or pointer is genuinely uncertain. Every patient-specific quantity in a claim must be carried by its own atomic evidence item: one `value`, one `unit`, one value-bearing citation. Split a comparison into two or more evidence items and describe only their qualitative relation in reasoning. Do not delete the countervailing side of a comparison. Write each number at the precision the tool displayed. Never calculate P-terminal force, a rate, ratio, difference, average or threshold result from component measurements; cite the returned components separately and describe their implication qualitatively. Do not copy a numeric comparison or derived difference from verifier feedback unless a tool directly returned and cited every number. For an unreliable measurement, state the caveat in the same claim or lower certainty. If an interval-context entry is missing or empty, call get_interval_waveform_context for that interval and preserve residual P/T evidence instead of merely adding an abstention. If candidate-only or T/A-ambiguous evidence cannot be independently corroborated, do not merely rephrase it: remove the positive diagnosis and place the unresolved P-morphology question in the differential or abstentions. If a diagnosis cannot be supported by evidence actually read in this session, remove it or move it to the differential. Never restore a diagnosis whose final Challenge hypothesis status is not `supported`."""


_PHASE_DOMAIN_KEYS: tuple[str, ...] = DIAGNOSTIC_DOMAINS

# Phase-specific citation-only mutation contracts. Survey initializes domain
# coverage; later phases emit bounded mutations against program-owned state.
# There is intentionally no legacy replacement-state schema in this module:
# keeping it importable made copied observations look like a supported path.
_COMPACT_DOMAIN_STATE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["assessed", "limited", "not_assessed"],
        },
        "finding": {
            "type": "string",
            "enum": ["normal", "abnormal", "indeterminate", "not_assessed"],
        },
        "citations": {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string"},
        },
        "limitations": {
            "type": "array",
            "maxItems": 1,
            "items": {"type": "string", "maxLength": 100},
        },
    },
    "required": ["status", "finding", "citations", "limitations"],
    "additionalProperties": False,
}


def _compact_domains_schema(*, require_all: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            domain: _COMPACT_DOMAIN_STATE for domain in _PHASE_DOMAIN_KEYS
        },
        "required": list(_PHASE_DOMAIN_KEYS) if require_all else [],
        "additionalProperties": False,
    }


_PATCH_EVIDENCE_CLAIM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {"type": "string", "maxLength": 180},
        },
    },
    "required": ["citations"],
    "additionalProperties": False,
}

_CREATE_HYPOTHESIS_PATCH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string", "maxLength": 80},
        "diagnosis_code": {
            "type": "string",
            "enum": sorted(set(DIAGNOSIS_CATALOG) - set(NON_HYPOTHESIS_CODES)),
        },
        "phenotype": {"type": "string", "maxLength": 120},
        "kind": {"type": "string", "enum": ["diagnosis", "phenotype"]},
        "domains": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {"type": "string", "enum": list(DIAGNOSTIC_DOMAINS)},
        },
        "status": {
            "type": "string",
            "enum": ["raised"],
        },
        "supporting_observations": {
            "type": "array",
            "maxItems": 4,
            "items": _PATCH_EVIDENCE_CLAIM,
        },
        "counterevidence": {
            "type": "array",
            "maxItems": 4,
            "items": _PATCH_EVIDENCE_CLAIM,
        },
        "required_evidence": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 180},
        },
        "next_tools": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "enum": list(DIAGNOSTIC_TOOLS)},
        },
        "alternative_explanations": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 160},
        },
        "alternative_group": {"type": "string", "maxLength": 80},
    },
    "required": [
        "hypothesis_id",
        "diagnosis_code",
        "phenotype",
        "kind",
        "domains",
        "status",
        "supporting_observations",
        "counterevidence",
        "required_evidence",
        "next_tools",
        "alternative_explanations",
        "alternative_group",
    ],
    "additionalProperties": False,
}

_EVIDENCE_UPDATE_PATCH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string", "maxLength": 80},
        "evidence_type": {
            "type": "string",
            "enum": ["support", "counterevidence"],
        },
        **_PATCH_EVIDENCE_CLAIM["properties"],
    },
    "required": ["hypothesis_id", "evidence_type", "citations"],
    "additionalProperties": False,
}


def _transition_patch(statuses: Sequence[str]) -> dict[str, Any]:
    # Survey forbids transitions by setting the containing array's maxItems to
    # zero.  Keep the unreachable item schema valid nevertheless: XGrammar
    # treats ``enum: []`` as a fatal compiler error and vLLM 0.24 consequently
    # tears down the whole EngineCore instead of rejecting only the request.
    # Runtime phase validation remains the authority for legal transitions.
    to_status: dict[str, Any] = {"type": "string", "maxLength": 32}
    if statuses:
        to_status["enum"] = list(statuses)
    return {
        "type": "object",
        "properties": {
            "hypothesis_id": {"type": "string", "maxLength": 80},
            "from_status": {"type": "string", "maxLength": 32},
            "to_status": to_status,
            "citations": {
                "type": "array",
                "maxItems": 4,
                "items": {"type": "string", "maxLength": 180},
            },
        },
        "required": [
            "hypothesis_id",
            "from_status",
            "to_status",
            "citations",
        ],
        "additionalProperties": False,
    }


_PLAN_UPDATE_PATCH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string", "maxLength": 80},
        "required_evidence": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 180},
        },
        "next_tools": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "enum": list(DIAGNOSTIC_TOOLS)},
        },
        "alternative_explanations": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 160},
        },
        "alternative_group": {"type": "string", "maxLength": 80},
    },
    "required": [
        "hypothesis_id",
        "required_evidence",
        "next_tools",
        "alternative_explanations",
        "alternative_group",
    ],
    "additionalProperties": False,
}

_REFUTING_TEST_PATCH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string", "maxLength": 80},
        "would_refute": {"type": "string", "maxLength": 200},
        "citations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 180},
        },
        "test_outcome": {
            "type": "string",
            "enum": ["not_refuted", "refuted", "inconclusive"],
        },
        "evidence_direction": {
            "type": "string",
            "enum": [
                "supports",
                "contradicts",
                "not_discriminative",
                "unavailable",
            ],
        },
    },
    "required": [
        "hypothesis_id",
        "would_refute",
        "citations",
        "test_outcome",
        "evidence_direction",
    ],
    "additionalProperties": False,
}

_TARGETED_CHECK_PATCH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string", "maxLength": 80},
        "domain": {"type": "string", "enum": list(DIAGNOSTIC_DOMAINS)},
        "tool": {"type": "string", "enum": list(DIAGNOSTIC_TOOLS)},
        "question": {"type": "string", "maxLength": 180},
    },
    "required": ["hypothesis_id", "domain", "tool", "question"],
    "additionalProperties": False,
}

_DOMAIN_UPDATE_PATCH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "domain": {"type": "string", "enum": list(DIAGNOSTIC_DOMAINS)},
        **_COMPACT_DOMAIN_STATE["properties"],
    },
    "required": [
        "domain",
        "status",
        "finding",
        "citations",
        "limitations",
    ],
    "additionalProperties": False,
}


def _bounded_patch_array(schema: Mapping[str, Any], max_items: int) -> dict[str, Any]:
    return {"type": "array", "maxItems": max_items, "items": dict(schema)}


def _phase_patch_contract(
    transition_statuses: Sequence[str],
    *,
    survey: bool = False,
    evidence_updates: bool = False,
    domain_updates: bool = False,
    plan_updates: bool = False,
    refuting_tests: bool = False,
    targeted_checks: bool = False,
) -> dict[str, Any]:
    create_schema = copy.deepcopy(_CREATE_HYPOTHESIS_PATCH)
    if survey:
        create_schema["properties"]["status"]["enum"] = ["raised"]
    properties: dict[str, Any] = {
        "base_version": {"type": "integer", "minimum": 0},
        "create_hypotheses": _bounded_patch_array(
            create_schema,
            3 if survey else 2,
        ),
        "evidence_updates": _bounded_patch_array(
            _EVIDENCE_UPDATE_PATCH,
            8 if evidence_updates else 0,
        ),
        "domain_updates": _bounded_patch_array(
            _DOMAIN_UPDATE_PATCH,
            8 if domain_updates else 0,
        ),
        "transitions": _bounded_patch_array(
            _transition_patch(transition_statuses),
            6 if transition_statuses else 0,
        ),
        "plan_updates": _bounded_patch_array(
            _PLAN_UPDATE_PATCH,
            6 if plan_updates else 0,
        ),
        "refuting_tests": _bounded_patch_array(
            _REFUTING_TEST_PATCH,
            4 if refuting_tests else 0,
        ),
        "targeted_checks": _bounded_patch_array(
            _TARGETED_CHECK_PATCH,
            4 if targeted_checks else 0,
        ),
        "detector_conflicts": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 220},
        },
        "unresolved": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 220},
        },
    }
    required = list(properties)
    if survey:
        properties["domains"] = _compact_domains_schema(require_all=True)
        required.append("domains")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


SURVEY_STATE_SCHEMA = _phase_patch_contract((), survey=True)
HYPOTHESIS_STATE_SCHEMA = _phase_patch_contract(
    ("provisional", "weak", "unresolved", "rejected", "abstain"),
    plan_updates=True,
    targeted_checks=True,
)
INVESTIGATION_STATE_SCHEMA = _phase_patch_contract(
    ("provisional", "supported", "weak", "weakened", "rejected", "unresolved", "abstain"),
    evidence_updates=True,
    domain_updates=True,
    plan_updates=True,
)
CHALLENGE_STATE_SCHEMA = _phase_patch_contract(
    ("supported", "weakened", "rejected", "unresolved", "abstain"),
    evidence_updates=True,
    domain_updates=True,
    plan_updates=True,
    refuting_tests=True,
)

# Backward-compatible import for callers that still request one generic phase
# contract.  Investigation is the richest non-final state.
PHASE_STATE_SCHEMA = INVESTIGATION_STATE_SCHEMA


def validate_phase_state(
    phase: str,
    state: Mapping[str, Any],
    *,
    phase_citations: Sequence[str] = (),
    ledger_state: Mapping[str, Any] | None = None,
) -> list[str]:
    """Guard patch semantics before the central reducer commits them."""

    problems: list[str] = []
    for path in non_english_text_paths(state, f"phase.{phase}"):
        problems.append(f"{path} must be English-only and must not contain Han-script text")
    phase_key = str(phase)
    ledger_state = ledger_state if isinstance(ledger_state, Mapping) else {}
    current_version = ledger_state.get("version")
    base_version: int | None = None
    if phase_key == "survey" or isinstance(current_version, int):
        try:
            base_version = int(state.get("base_version"))
        except (TypeError, ValueError):
            base_version = -1
            problems.append("base_version must be an integer")
    if isinstance(current_version, int) and base_version != current_version:
        problems.append(
            f"stale base_version `{base_version}`; current ledger version is `{current_version}`"
        )
    if phase_key == "survey" and base_version != 0:
        problems.append("Survey must initialize ledger base_version 0")

    existing: dict[str, Mapping[str, Any]] = {}
    for row in ledger_state.get("hypotheses") or []:
        if isinstance(row, Mapping):
            existing[str(row.get("id") or "")] = row
    created: dict[str, Mapping[str, Any]] = {}
    active_codes = {
        str(row.get("diagnosis_code") or "")
        for row in existing.values()
        if str(row.get("status") or "") not in {"rejected", "abstain"}
    }
    for index, row in enumerate(state.get("create_hypotheses") or []):
        if not isinstance(row, Mapping):
            problems.append(f"create_hypotheses[{index}] is not an object")
            continue
        hypothesis_id = str(row.get("hypothesis_id") or "")
        code = str(row.get("diagnosis_code") or "")
        if not hypothesis_id:
            problems.append(f"create_hypotheses[{index}] has no hypothesis_id")
        elif hypothesis_id in existing or hypothesis_id in created:
            problems.append(f"create_hypotheses[{index}] reuses hypothesis `{hypothesis_id}`")
        else:
            created[hypothesis_id] = row
        if code not in DIAGNOSIS_CATALOG:
            problems.append(f"create_hypotheses[{index}] has an unregistered diagnosis_code")
        elif code in NON_HYPOTHESIS_CODES:
            problems.append(
                f"create_hypotheses[{index}] uses quality/recommendation code `{code}` as a hypothesis"
            )
        elif code in active_codes:
            problems.append(f"create_hypotheses[{index}] duplicates active code `{code}`")
        else:
            active_codes.add(code)
        if row.get("status") != "raised":
            problems.append(
                f"create_hypotheses[{index}] must start as `raised`"
            )
        claims = [
            claim
            for field in ("supporting_observations", "counterevidence")
            for claim in (row.get(field) or [])
            if isinstance(claim, Mapping) and claim.get("citations")
        ]
        if not claims:
            problems.append(
                f"create_hypotheses[{index}] cites no patient evidence"
            )
        invalid_domains = [
            str(value)
            for value in (row.get("domains") or [])
            if str(value) not in DIAGNOSTIC_DOMAINS
        ]
        if invalid_domains:
            problems.append(f"create_hypotheses[{index}] contains invalid domains")
        invalid_tools = [
            str(value)
            for value in (row.get("next_tools") or [])
            if str(value) not in DIAGNOSTIC_TOOLS
        ]
        if invalid_tools:
            problems.append(f"create_hypotheses[{index}] contains unregistered tools")

    known_ids = set(existing) | set(created)
    for field in ("evidence_updates", "domain_updates", "transitions"):
        for index, row in enumerate(state.get(field) or []):
            if not isinstance(row, Mapping):
                problems.append(f"{field}[{index}] is not an object")
                continue
            hypothesis_id = str(row.get("hypothesis_id") or "")
            if hypothesis_id not in known_ids:
                if field != "domain_updates":
                    problems.append(f"{field}[{index}] references unknown hypothesis `{hypothesis_id}`")
            if field == "domain_updates" and str(row.get("domain") or "") not in DIAGNOSTIC_DOMAINS:
                problems.append(f"domain_updates[{index}] references an unknown domain")
            if field == "transitions" and hypothesis_id in existing:
                expected = str(existing[hypothesis_id].get("status") or "")
                if str(row.get("from_status") or "") != expected:
                    problems.append(
                        f"transitions[{index}] has stale from_status; expected `{expected}`"
                    )
                from_status = str(row.get("from_status") or "")
                to_status = str(row.get("to_status") or "")
                if from_status == to_status:
                    problems.append(
                        f"transitions[{index}] is a no-op `{from_status}` -> `{to_status}`; "
                        "omit unchanged hypotheses"
                    )
                elif to_status not in LEGAL_TRANSITIONS.get(from_status, frozenset()):
                    problems.append(
                        f"transitions[{index}] is illegal `{from_status}` -> `{to_status}`"
                    )
            elif field == "transitions" and hypothesis_id in created:
                problems.append(
                    f"transitions[{index}] promotes newly created hypothesis "
                    f"`{hypothesis_id}` in its creation phase"
                )

    for field in ("plan_updates", "targeted_checks", "refuting_tests"):
        for index, row in enumerate(state.get(field) or []):
            if not isinstance(row, Mapping):
                problems.append(f"{field}[{index}] is not an object")
                continue
            hypothesis_id = str(row.get("hypothesis_id") or "")
            if hypothesis_id not in known_ids:
                problems.append(f"{field}[{index}] references unknown hypothesis `{hypothesis_id}`")
            if field in {"plan_updates", "targeted_checks"}:
                tools = (
                    row.get("next_tools") or []
                    if field == "plan_updates"
                    else [row.get("tool")]
                )
                if any(str(tool) not in DIAGNOSTIC_TOOLS for tool in tools):
                    problems.append(f"{field}[{index}] contains an unregistered tool")

    if phase_key == "challenge":
        passed = {
            str(row.get("hypothesis_id") or "")
            for row in (state.get("refuting_tests") or [])
            if isinstance(row, Mapping)
            and row.get("test_outcome") == "not_refuted"
            and row.get("evidence_direction") == "supports"
        }
        for index, row in enumerate(state.get("refuting_tests") or []):
            if not isinstance(row, Mapping):
                continue
            outcome = str(row.get("test_outcome") or "")
            direction = str(row.get("evidence_direction") or "")
            if outcome == "refuted" and direction != "contradicts":
                problems.append(
                    f"refuting_tests[{index}] refuted outcome requires "
                    "evidence_direction `contradicts`"
                )
            if direction == "unavailable" and outcome != "inconclusive":
                problems.append(
                    f"refuting_tests[{index}] unavailable evidence must be inconclusive"
                )
        final_status = {
            hypothesis_id: str(row.get("status") or "")
            for hypothesis_id, row in existing.items()
        }
        for row in state.get("transitions") or []:
            if isinstance(row, Mapping):
                final_status[str(row.get("hypothesis_id") or "")] = str(
                    row.get("to_status") or ""
                )
        for hypothesis_id, status in final_status.items():
            if status == "supported" and hypothesis_id not in passed:
                problems.append(
                    f"{UNCHALLENGED_SUPPORT_PREFIX} `{hypothesis_id}` "
                    "has no discriminative Challenge falsification test"
                )

    def phase_texts(value: Any, path: str = "phase_patch"):
        if isinstance(value, str):
            yield path, value
        elif isinstance(value, Mapping):
            for key, child in value.items():
                # Citation tokens, stable IDs and registered codes may contain
                # digits but are addresses/identities, not clinical prose.
                if key in {
                    "citations",
                    "hypothesis_id",
                    "diagnosis_code",
                    "tool",
                    "from_status",
                    "to_status",
                    "status",
                    "finding",
                }:
                    continue
                yield from phase_texts(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from phase_texts(child, f"{path}[{index}]")

    for path, text in phase_texts(state):
        quantities = [
            quantity.render()
            for quantity in extract_quantities(text)
            if not is_threshold_reference(text, quantity)
        ]
        if quantities:
            problems.append(
                f"{path} copies patient-style numeric text ({', '.join(quantities[:3])}); "
                "intermediate patches must select citations and let the ledger "
                "materialize values"
            )
    return list(dict.fromkeys(problems))


def sanitize_phase_state(
    phase: str,
    state: Mapping[str, Any],
    *,
    phase_citations: Sequence[str] = (),
    active_diagnosis_codes: Sequence[str] = (),
    expected_base_version: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply a monotone fail-closed projection before atomic reduction.

    Copy-forward rows that cite no evidence read in the current evidence phase
    do not convey a new observation. No-op transitions do not convey a state
    change. Removing those rows can only preserve the previous ledger state;
    it can never create support, promote a hypothesis or invent evidence. The
    remaining patch is still committed atomically by :class:`DiagnosticLedger`.
    """

    normalized = copy.deepcopy(dict(state))
    notes: list[str] = []
    phase_key = str(phase)
    current = {
        str(token).removeprefix("ev:")
        for token in phase_citations
        if str(token).strip()
    }

    if (
        isinstance(expected_base_version, int)
        and normalized.get("base_version") != expected_base_version
    ):
        notes.append(
            "base_version replaced with the program-owned current ledger version "
            f"{expected_base_version}"
        )
        normalized["base_version"] = expected_base_version

    def has_new_citation(row: Mapping[str, Any]) -> bool:
        cited = {
            str(token).removeprefix("ev:")
            for token in (row.get("citations") or [])
        }
        return bool(cited & current)

    if phase_key in {"investigate", "challenge"}:
        for field in ("evidence_updates", "domain_updates", "refuting_tests"):
            kept: list[Any] = []
            for index, row in enumerate(normalized.get(field) or []):
                if isinstance(row, Mapping) and not has_new_citation(row):
                    notes.append(
                        f"{field}[{index}] removed: it cites no atomic evidence "
                        f"newly read in {phase_key}"
                    )
                    continue
                kept.append(row)
            normalized[field] = kept

    kept_transitions: list[Any] = []
    for index, row in enumerate(normalized.get("transitions") or []):
        if not isinstance(row, Mapping):
            kept_transitions.append(row)
            continue
        before = str(row.get("from_status") or "")
        after = str(row.get("to_status") or "")
        if before == after:
            notes.append(
                f"transitions[{index}] removed: unchanged status `{before}`"
            )
            continue
        if phase_key in {"investigate", "challenge"} and not has_new_citation(row):
            notes.append(
                f"transitions[{index}] removed: it cites no atomic evidence "
                f"newly read in {phase_key}"
            )
            continue
        kept_transitions.append(row)
    normalized["transitions"] = kept_transitions

    # One hypothesis per diagnosis code. A second identity for a code that is
    # already carried adds no candidate the ledger is not already tracking, so
    # dropping it preserves the differential exactly. This mostly fires where
    # the catalog has no code for what the model meant -- "normal intervals"
    # and "normal axis" both land on `sinus_rhythm` -- and those observations
    # belong to the ten-domain coverage rows, which are unaffected here.
    seen_codes = {
        str(code) for code in active_diagnosis_codes if str(code).strip()
    }
    kept_creates: list[Any] = []
    for index, row in enumerate(normalized.get("create_hypotheses") or []):
        if isinstance(row, Mapping):
            if phase_key == "survey" and current:
                cited = {
                    str(token).removeprefix("ev:")
                    for field in ("supporting_observations", "counterevidence")
                    for claim in (row.get(field) or [])
                    if isinstance(claim, Mapping)
                    for token in (claim.get("citations") or [])
                }
                if not cited.intersection(current):
                    notes.append(
                        f"create_hypotheses[{index}] removed: Survey candidate "
                        "has no citation from the neutral intake packet"
                    )
                    continue
            code = str(row.get("diagnosis_code") or "")
            if code and code in seen_codes:
                notes.append(
                    f"create_hypotheses[{index}] removed: `{code}` already has "
                    "an active hypothesis"
                )
                continue
            if code:
                seen_codes.add(code)
        kept_creates.append(row)
    normalized["create_hypotheses"] = kept_creates

    # All new identities begin as questions. Promotion, if justified, is a
    # separate evidence-backed transition and therefore auditable.
    for index, row in enumerate(kept_creates):
        if isinstance(row, dict) and row.get("status") != "raised":
            notes.append(
                f"create_hypotheses[{index}] status downgraded to `raised`"
            )
            row["status"] = "raised"
    return normalized, notes


def normalize_verdict(
    verdict: Mapping[str, Any],
    resolve_cited_value: Callable[[str], tuple[float, str | None] | None] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply lossless/conservative presentation repairs before model revision.

    Grounded numbers remain intact. A number that the model did not place in a
    cited structured evidence value is replaced by a readable non-numeric
    phrase; this removes an unsupported precision claim without deleting the
    surrounding clinical statement. Padded/duplicate Top-3 alternatives are
    also removed deterministically. Neither operation creates patient evidence
    or promotes a diagnosis, so spending another large model turn on them is
    unnecessary.

    `resolve_cited_value` maps a citation token to the value the model was
    shown for it. Given one, a number is grounded when *any* pointer the item
    cites carries it, rather than only the item's single `value` scalar. That
    difference decides whether comparative statements survive: "S -0.11 mV,
    R 0.614 mV" cites both amplitudes, and scrubbing the second one inverted
    the clinical meaning of the sentence while leaving it looking cited.
    """

    normalized = copy.deepcopy(dict(verdict))
    changed: list[str] = []

    def cited_values(items: Any) -> list[tuple[float, str | None]]:
        """Values reachable from an evidence item's own citations."""
        if resolve_cited_value is None or not isinstance(items, list):
            return []
        found: list[tuple[float, str | None]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            for token in item.get("citations") or []:
                resolved = resolve_cited_value(str(token))
                if resolved is not None:
                    found.append(resolved)
        return found

    def sanitize_text(
        value: Any,
        grounded_values: Sequence[tuple[float, str | None]],
    ) -> str:
        text = str(value or "")
        unsupported = [
            quantity
            for quantity in extract_quantities(text)
            if not is_threshold_reference(text, quantity)
            if not (
                re.search(
                    r"\d+(?:\.\d+)?\s*[-–—~\u81f3\u5230]\s*$",
                    text[max(0, quantity.start - 24) : quantity.start],
                )
                or (
                    quantity.text.lstrip().startswith("-")
                    and re.search(
                        r"\d+(?:\.\d+)?\s*$",
                        text[max(0, quantity.start - 24) : quantity.start],
                    )
                )
            )
            if not any(
                quantities_match(quantity, evidence_value, evidence_unit)
                for evidence_value, evidence_unit in grounded_values
            )
        ]
        for quantity in reversed(unsupported):
            text = text[: quantity.start] + "corresponding measured value" + text[quantity.end :]
        return text

    def rewrite(
        parent: dict[str, Any],
        key: str,
        grounded_values: Sequence[tuple[float, str | None]],
        path: str,
    ) -> None:
        if key not in parent or not isinstance(parent.get(key), str):
            return
        before = str(parent[key])
        after = sanitize_text(before, grounded_values)
        if after != before:
            parent[key] = after
            changed.append(path)

    def rewrite_evidence(items: Any, path: str) -> None:
        if not isinstance(items, list):
            return
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            rewrite(
                item,
                "claim",
                [*_evidence_values([item]), *cited_values([item])],
                f"{path}[{index}].claim",
            )

    all_values = [
        *_verdict_evidence_values(normalized),
        *cited_values(_all_evidence_items(normalized)),
    ]
    rewrite(normalized, "summary", all_values, "summary")

    interpretations = normalized.get("ranked_complete_interpretations")
    if isinstance(interpretations, list):
        positive_rows = _items_for_contract(normalized.get("diagnoses"))
        differential_rows = _items_for_contract(
            normalized.get("differential_diagnoses")
        )
        positive_codes = {
            str(row.get("code") or "") for row in positive_rows if row.get("code")
        }
        differential_codes = {
            str(row.get("code") or "")
            for row in differential_rows
            if row.get("code")
        }
        kept: list[dict[str, Any]] = []
        seen_diagnoses: set[str] = set()
        for original_index, item in enumerate(interpretations):
            if not isinstance(item, dict):
                continue
            rewrite(
                item,
                "complete_diagnosis",
                all_values,
                f"ranked_complete_interpretations[{original_index}].complete_diagnosis",
            )
            rewrite(
                item,
                "key_uncertainty",
                all_values,
                f"ranked_complete_interpretations[{original_index}].key_uncertainty",
            )
            rendered = str(item.get("complete_diagnosis") or "").strip()
            basis = {
                str(code or "") for code in (item.get("basis_codes") or [])
            }
            if rendered in seen_diagnoses:
                changed.append(f"ranked_complete_interpretations[{original_index}]")
                continue
            if kept and not (basis & differential_codes):
                changed.append(f"ranked_complete_interpretations[{original_index}]")
                continue
            seen_diagnoses.add(rendered)
            kept.append(item)
            if len(kept) >= 3:
                break
        if kept:
            for index, item in enumerate(kept):
                expected_rank = index + 1
                expected_type = "PRIMARY" if index == 0 else "ALTERNATIVE"
                if item.get("rank") != expected_rank:
                    item["rank"] = expected_rank
                    changed.append(
                        f"ranked_complete_interpretations[{index}].rank"
                    )
                if item.get("interpretation_type") != expected_type:
                    item["interpretation_type"] = expected_type
                    changed.append(
                        f"ranked_complete_interpretations[{index}].interpretation_type"
                    )
            primary = kept[0]
            if (
                primary.get("confidence") == "HIGH"
                and any(row.get("confidence") == "MEDIUM" for row in positive_rows)
            ):
                primary["confidence"] = "MEDIUM"
                changed.append("ranked_complete_interpretations[0].confidence")
        if kept != interpretations:
            normalized["ranked_complete_interpretations"] = kept

    diagnoses = normalized.get("diagnoses")
    if isinstance(diagnoses, list):
        for index, diagnosis in enumerate(diagnoses):
            if not isinstance(diagnosis, dict):
                continue
            # `category` is a pure function of `code`, so asking the model to
            # restate it can only introduce disagreement -- and did, fatally.
            # Stamp the catalog's answer instead of validating a copy of it.
            code = str(diagnosis.get("code") or "")
            if code in DIAGNOSIS_CATALOG:
                expected_category = DIAGNOSIS_CATALOG[code][1]
                if diagnosis.get("category") != expected_category:
                    diagnosis["category"] = expected_category
                    changed.append(f"diagnoses[{index}].category")
            values = [
                *_evidence_values(diagnosis.get("evidence")),
                *_evidence_values(diagnosis.get("counterevidence")),
                *cited_values(diagnosis.get("evidence")),
                *cited_values(diagnosis.get("counterevidence")),
            ]
            rewrite(diagnosis, "statement", values, f"diagnoses[{index}].statement")
            rewrite(diagnosis, "reasoning", values, f"diagnoses[{index}].reasoning")
            rewrite_evidence(diagnosis.get("evidence"), f"diagnoses[{index}].evidence")
            rewrite_evidence(
                diagnosis.get("counterevidence"),
                f"diagnoses[{index}].counterevidence",
            )

    differentials = normalized.get("differential_diagnoses")
    if isinstance(differentials, list):
        for index, differential in enumerate(differentials):
            if not isinstance(differential, dict):
                continue
            values = [
                *_evidence_values(differential.get("supporting_evidence")),
                *_evidence_values(differential.get("counterevidence")),
                *cited_values(differential.get("supporting_evidence")),
                *cited_values(differential.get("counterevidence")),
            ]
            for key in ("statement", "what_would_resolve_it"):
                rewrite(
                    differential,
                    key,
                    values,
                    f"differential_diagnoses[{index}].{key}",
                )
            rewrite_evidence(
                differential.get("supporting_evidence"),
                f"differential_diagnoses[{index}].supporting_evidence",
            )
            rewrite_evidence(
                differential.get("counterevidence"),
                f"differential_diagnoses[{index}].counterevidence",
            )

    contexts = normalized.get("interval_measurement_contexts")
    if isinstance(contexts, list):
        for index, context in enumerate(contexts):
            if not isinstance(context, dict):
                continue
            values = [
                *_evidence_values(context.get("residual_evidence")),
                *cited_values(context.get("residual_evidence")),
            ]
            for key in (
                "interval_conclusion",
                "component_waveform_assessment",
                "interpretive_impact",
                "what_would_resolve_it",
            ):
                rewrite(
                    context,
                    key,
                    values,
                    f"interval_measurement_contexts[{index}].{key}",
                )
            rewrite_evidence(
                context.get("residual_evidence"),
                f"interval_measurement_contexts[{index}].residual_evidence",
            )

    for index, abstention in enumerate(normalized.get("abstentions") or []):
        if not isinstance(abstention, dict):
            continue
        for key in ("topic", "reason", "what_would_resolve_it"):
            rewrite(abstention, key, all_values, f"abstentions[{index}].{key}")
    quality = normalized.get("quality_assessment")
    if isinstance(quality, dict) and isinstance(quality.get("limitations"), list):
        for index, limitation in enumerate(quality["limitations"]):
            if not isinstance(limitation, str):
                continue
            after = sanitize_text(limitation, all_values)
            if after != limitation:
                quality["limitations"][index] = after
                changed.append(f"quality_assessment.limitations[{index}]")
    review = normalized.get("human_review")
    if isinstance(review, dict) and isinstance(review.get("reasons"), list):
        for index, reason in enumerate(review["reasons"]):
            if not isinstance(reason, str):
                continue
            after = sanitize_text(reason, all_values)
            if after != reason:
                review["reasons"][index] = after
                changed.append(f"human_review.reasons[{index}]")

    return normalized, list(dict.fromkeys(changed))


_EVIDENCE_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim": {
            "type": "string",
            "description": "One English sentence stating the observation.",
        },
        "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claim", "value", "unit", "citations"],
    "additionalProperties": False,
}


def _evidence_values(items: Any) -> list[tuple[float, str | None]]:
    values: list[tuple[float, str | None]] = []
    if not isinstance(items, list):
        return values
    for item in items:
        if not isinstance(item, Mapping):
            continue
        value = item.get("value")
        citations = item.get("citations")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not isinstance(citations, list) or not citations:
            continue
        unit = item.get("unit")
        values.append((float(value), str(unit) if unit is not None else None))
    return values


def _all_evidence_items(verdict: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every evidence item in the verdict, wherever it lives."""
    items: list[Mapping[str, Any]] = []
    for diagnosis in verdict.get("diagnoses") or []:
        if isinstance(diagnosis, Mapping):
            for key in ("evidence", "counterevidence"):
                items.extend(
                    row for row in (diagnosis.get(key) or []) if isinstance(row, Mapping)
                )
    for differential in verdict.get("differential_diagnoses") or []:
        if isinstance(differential, Mapping):
            for key in ("supporting_evidence", "counterevidence"):
                items.extend(
                    row
                    for row in (differential.get(key) or [])
                    if isinstance(row, Mapping)
                )
    for context in verdict.get("interval_measurement_contexts") or []:
        if isinstance(context, Mapping):
            items.extend(
                row
                for row in (context.get("residual_evidence") or [])
                if isinstance(row, Mapping)
            )
    return items


def _verdict_evidence_values(verdict: Mapping[str, Any]) -> list[tuple[float, str | None]]:
    values: list[tuple[float, str | None]] = []
    diagnoses = verdict.get("diagnoses")
    if isinstance(diagnoses, list):
        for diagnosis in diagnoses:
            if not isinstance(diagnosis, Mapping):
                continue
            values.extend(_evidence_values(diagnosis.get("evidence")))
            values.extend(_evidence_values(diagnosis.get("counterevidence")))
    differentials = verdict.get("differential_diagnoses")
    if isinstance(differentials, list):
        for differential in differentials:
            if not isinstance(differential, Mapping):
                continue
            values.extend(_evidence_values(differential.get("supporting_evidence")))
            values.extend(_evidence_values(differential.get("counterevidence")))
    contexts = verdict.get("interval_measurement_contexts")
    if isinstance(contexts, list):
        for context in contexts:
            if isinstance(context, Mapping):
                values.extend(_evidence_values(context.get("residual_evidence")))
    return values


def _required_interval_contexts(
    evidence_document: Mapping[str, Any] | None,
) -> set[str]:
    """Return limited intervals that require explicit component-wave review."""
    if not isinstance(evidence_document, Mapping):
        return set()
    metadata = _as_mapping(evidence_document.get("metadata"))
    gate = _as_mapping(metadata.get("diagnostic_gate"))
    if str(gate.get("state") or "").lower() == "stop":
        return set()
    global_features = _as_mapping(evidence_document.get("global_features"))
    rhythm_inputs = _as_mapping(evidence_document.get("rhythm_inputs"))
    rhythm_record = _as_mapping(rhythm_inputs.get("record"))
    availability = _as_mapping(rhythm_record.get("availability"))
    required: set[str] = set()
    qt_reliability = str(
        global_features.get("qt_reliability") or "unavailable"
    ).lower()
    if (
        global_features.get("qt_ms") is None
        or global_features.get("qt_reportable") is False
        or qt_reliability
        in {"fallback", "low_confidence", "unreliable", "unavailable"}
    ):
        required.add("QT_QTc")
    if (
        global_features.get("pr_ms") is None
        or availability.get("pr_available") is False
    ):
        required.add("PR")
    return required


def _has_component_wave_citation(interval: str, items: Any) -> bool:
    if not isinstance(items, list):
        return False
    citations = [
        str(citation).removeprefix("ev:")
        for item in items
        if isinstance(item, Mapping)
        for citation in (item.get("citations") or [])
    ]
    if interval == "QT_QTc":
        return any(
            (
                pointer.startswith("/representative_leads/")
                and "/params/t_" in pointer
            )
            or (
                pointer.startswith("/quality/")
                and pointer.rsplit("/", 1)[-1] in {"reliable_for_t", "reliable_for_qt"}
            )
            or (
                pointer.startswith("/beat_features/")
                and any(token in pointer for token in ("/t_", "/qt_", "/flags", "/u_"))
            )
            or (
                pointer.startswith("/global_features/")
                and pointer.rsplit("/", 1)[-1].startswith("t_")
            )
            or (
                pointer.startswith("/rhythm_inputs/native_beat_profiles/")
                and any(token in pointer for token in ("/t_", "/qt_", "/st_"))
            )
            for pointer in citations
        )
    if interval == "PR":
        return any(
            (
                pointer.startswith("/representative_leads/")
                and "/params/p_" in pointer
            )
            or pointer.startswith("/p_wave_assessments/")
            or pointer.startswith("/rhythm_inputs/p_events/")
            or pointer.startswith("/rhythm_inputs/av_block/")
            or (
                pointer.startswith("/rhythm_inputs/record/availability/")
                and pointer.rsplit("/", 1)[-1]
                in {
                    "atrial_rhythm_available",
                    "pr_available",
                    "p_axis_available",
                    "reasons",
                    # Which of `reasons` were present but judged insufficient to
                    # withhold PR; without it the pair (pr_available=true,
                    # reasons=[av_dissociation]) reads as a contradiction.
                    "pr_soft_reasons",
                }
            )
            or (
                pointer.startswith("/quality/")
                and pointer.endswith("/reliable_for_p")
            )
            or (
                pointer.startswith("/beat_features/")
                and any(token in pointer for token in ("/p_", "/pr_", "/flags"))
            )
            or (
                pointer.startswith("/global_features/")
                and pointer.rsplit("/", 1)[-1].startswith("p_")
            )
            for pointer in citations
        )
    return False


def _reject_quantities(
    problems: list[str],
    where: str,
    value: Any,
    *,
    grounded_values: Sequence[tuple[float, str | None]] = (),
) -> None:
    text = str(value or "").strip()
    if "ev:/" in text:
        problems.append(
            f"{where} contains a pointer; put citable facts in an evidence item"
        )
    unsupported = [
        quantity
        for quantity in extract_quantities(text)
        if not is_threshold_reference(text, quantity)
        if not (
            re.search(
                r"\d+(?:\.\d+)?\s*[-–—~\u81f3\u5230]\s*$",
                text[max(0, quantity.start - 24) : quantity.start],
            )
            or (
                quantity.text.lstrip().startswith("-")
                and re.search(
                    r"\d+(?:\.\d+)?\s*$",
                    text[max(0, quantity.start - 24) : quantity.start],
                )
            )
        )
        if not any(
            quantities_match(quantity, evidence_value, evidence_unit)
            for evidence_value, evidence_unit in grounded_values
        )
    ]
    if unsupported:
        rendered = ", ".join(item.render() for item in unsupported[:4])
        problems.append(
            f"{where} contains ungrounded patient-specific quantity/quantities "
            f"({rendered}); add the same value to a cited evidence item or "
            "remove it from the narrative"
        )


def _validate_evidence_items(
    problems: list[str],
    items: Any,
    where: str,
    *,
    require_nonempty: bool,
) -> None:
    if not isinstance(items, list):
        problems.append(f"{where} must be an array")
        return
    if require_nonempty and not items:
        problems.append(f"{where} must contain supporting evidence")
    for index, item in enumerate(items):
        item_where = f"{where}[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{item_where} is not an object")
            continue
        extra_keys = sorted(
            str(key)
            for key in set(item) - {"claim", "value", "unit", "citations"}
        )
        if extra_keys:
            problems.append(
                f"{item_where} contains non-atomic field(s) {', '.join(extra_keys)}; "
                "split every measurement into its own evidence item"
            )
        if not str(item.get("claim") or "").strip():
            problems.append(f"{item_where} has an empty `claim`")
        citations = item.get("citations")
        if not isinstance(citations, list):
            problems.append(f"{item_where} is missing a `citations` array")
        elif not citations:
            problems.append(f"{item_where} has no evidence citation")
        value = item.get("value")
        unit = item.get("unit")
        grounded_values: list[tuple[float, str | None]] = []
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            grounded_values.append(
                (float(value), str(unit) if unit is not None else None)
            )
        asserted_quantities = [
            quantity
            for quantity in extract_quantities(str(item.get("claim") or ""))
            if not is_threshold_reference(str(item.get("claim") or ""), quantity)
        ]
        if len(asserted_quantities) > 1:
            rendered = ", ".join(
                quantity.render() for quantity in asserted_quantities[:4]
            )
            problems.append(
                f"{item_where}.claim contains multiple patient measurements "
                f"({rendered}); evidence claims must be atomic"
            )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(citations, list)
            and len(citations) != 1
        ):
            problems.append(
                f"{item_where} carries a numeric value but has {len(citations)} "
                "citations; one atomic numeric fact must cite exactly one pointer"
            )
        _reject_quantities(
            problems,
            f"{item_where}.claim",
            item.get("claim"),
            grounded_values=grounded_values,
        )
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            problems.append(f"{item_where}.value must be numeric or null")
        if unit is not None and not isinstance(unit, str):
            problems.append(f"{item_where}.unit must be a string or null")


_CLEAR_MORPHOLOGY_RE = re.compile(
    r"(?:\bclear(?:ly)?\b|\bclean\b|\bdistinct\b|\bunambiguous\b|"
    r"\u6e05\u6670|\u660e\u786e|\u53ef\u9760\s*[Pp]\s*\u6ce2)",
    re.IGNORECASE,
)
_BAZETT_RE = re.compile(r"(?:bazett|qtcb|\u5df4\u6cfd\u7279)", re.IGNORECASE)
_FRIDERICIA_RE = re.compile(
    r"(?:fridericia|fredericia|qtcf|\u5f17\u91cc\u5fb7\u91cc\u5e0c\u4e9a)",
    re.IGNORECASE,
)
_LEAD_AFTER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<lead>aVR|aVL|aVF|V[1-6]|III|II|I)\s*"
    r"(?:\u5bfc\u8054|lead\b)",
    re.IGNORECASE,
)
_LEAD_BEFORE_RE = re.compile(
    r"(?:\u5bfc\u8054|lead)\s*(?P<lead>aVR|aVL|aVF|V[1-6]|III|II|I)\b",
    re.IGNORECASE,
)
_CANONICAL_LEADS = {
    "i": "I",
    "ii": "II",
    "iii": "III",
    "avr": "aVR",
    "avl": "aVL",
    "avf": "aVF",
    **{f"v{index}": f"V{index}" for index in range(1, 7)},
}


def _resolve_document_pointer(
    document: Mapping[str, Any] | None,
    token: Any,
) -> Any:
    """Resolve a final, expanded JSON pointer without authorizing new evidence."""

    if not isinstance(document, Mapping):
        return None
    pointer = str(token or "").removeprefix("ev:")
    if not pointer.startswith("/"):
        return None
    current: Any = document
    try:
        for raw_part in pointer[1:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, Mapping):
                current = current[part]
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return current


def _validate_evidence_semantics(
    problems: list[str],
    verdict: Mapping[str, Any],
    evidence_document: Mapping[str, Any] | None,
) -> None:
    """Block field-label drift and ambiguity-to-certainty conversions."""

    for index, item in enumerate(_all_evidence_items(verdict)):
        claim = str(item.get("claim") or "")
        citations = [str(value) for value in (item.get("citations") or [])]
        where = f"evidence_item[{index}]"
        claimed_leads = {
            _CANONICAL_LEADS[match.group("lead").lower()]
            for pattern in (_LEAD_AFTER_RE, _LEAD_BEFORE_RE)
            for match in pattern.finditer(claim)
        }
        cited_leads = {
            canonical
            for citation in citations
            for part in citation.removeprefix("ev:").split("/")
            if (canonical := _CANONICAL_LEADS.get(part.lower())) is not None
        }
        if claimed_leads and cited_leads and not claimed_leads.issubset(cited_leads):
            problems.append(
                f"{where} names lead(s) {sorted(claimed_leads)}, but its pointer "
                f"belongs to lead(s) {sorted(cited_leads)}; keep each lead label "
                "bound to its cited measurement"
            )
        for citation in citations:
            pointer = citation.removeprefix("ev:")
            if pointer.endswith("/qtc_bazett_ms") and _FRIDERICIA_RE.search(claim):
                problems.append(
                    f"{where} labels a Bazett pointer as Fridericia; keep the "
                    "measurement name bound to its canonical field"
                )
            if pointer.endswith("/qtc_fridericia_ms") and _BAZETT_RE.search(claim):
                problems.append(
                    f"{where} labels a Fridericia pointer as Bazett; keep the "
                    "measurement name bound to its canonical field"
                )
            lowered = claim.lower()
            if pointer.endswith("/qrs_axis_deg") and re.search(
                r"(?:\bt\s*axis\b|t\u8f74)", lowered
            ):
                problems.append(f"{where} labels a QRS-axis pointer as T axis")
            if pointer.endswith("/t_axis_deg") and re.search(
                r"(?:qrs\s*axis|qrs\u8f74)", lowered
            ):
                problems.append(f"{where} labels a T-axis pointer as QRS axis")

            if "/p_wave_assessments/" not in pointer:
                continue
            parts = pointer.split("/")
            try:
                row_index = int(parts[parts.index("p_wave_assessments") + 1])
                rows = (
                    evidence_document.get("p_wave_assessments")
                    if isinstance(evidence_document, Mapping)
                    else None
                )
                row = rows[row_index] if isinstance(rows, list) else None
            except (ValueError, IndexError, TypeError):
                row = None
            if (
                isinstance(row, Mapping)
                and row.get("ta_ambiguous") is True
                and _CLEAR_MORPHOLOGY_RE.search(claim)
            ):
                problems.append(
                    f"{where} describes a T/A-ambiguous P assessment as clear or "
                    "unambiguous; ambiguity may only lower certainty"
                )


_PREEXCITATION_CODES = frozenset({"ventricular_preexcitation_pattern"})
_FLUTTER_CODES = frozenset(
    {"atrial_flutter_pattern", "possible_atrial_flutter_pattern"}
)
_MULTIFOCAL_P_CODES = frozenset(
    {"multifocal_atrial_rhythm", "multifocal_atrial_tachycardia"}
)
_ECTOPIC_P_CODES = frozenset({"ectopic_atrial_rhythm_pattern"})
_ATRIAL_MORPHOLOGY_CODES = frozenset(
    {
        "p_wave_abnormality",
        "left_atrial_abnormality",
        "right_atrial_abnormality",
        "biatrial_abnormality",
        "pediatric_left_atrial_abnormality",
        "pediatric_right_atrial_abnormality",
    }
)
_SINUS_ORIGIN_CODES = frozenset(
    {"sinus_rhythm", "sinus_mechanism", "sinus_bradycardia", "sinus_tachycardia"}
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _p_boundary_summary(
    evidence_document: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in evidence_document.get("p_wave_assessments") or []
        if isinstance(row, Mapping)
    ]
    ambiguity_available = any("ta_ambiguous" in row for row in rows)
    clean = [
        row
        for row in rows
        if row.get("accepted") is True and row.get("ta_ambiguous") is False
    ]
    clean_clusters = {
        str(row["morphology_cluster_id"])
        for row in clean
        if row.get("morphology_cluster_id") is not None
    }
    ambiguous = sum(row.get("ta_ambiguous") is True for row in rows)
    return {
        "available": bool(rows) and ambiguity_available,
        "row_count": len(rows),
        "clean_accepted_count": len(clean),
        "clean_cluster_count": len(clean_clusters),
        "predominantly_ambiguous": bool(rows) and (2 * ambiguous >= len(rows)),
    }


def _rates_match(
    evidence_document: Mapping[str, Any],
) -> bool:
    rhythm = _as_mapping(evidence_document.get("rhythm_inputs"))
    background = _as_mapping(rhythm.get("background"))
    global_features = _as_mapping(evidence_document.get("global_features"))
    atrial = background.get("background_atrial_rate_bpm")
    ventricular = background.get("background_ventricular_rate_bpm")
    if not isinstance(atrial, (int, float)) or isinstance(atrial, bool):
        atrial = global_features.get("atrial_rate_bpm")
    if not isinstance(ventricular, (int, float)) or isinstance(ventricular, bool):
        ventricular = global_features.get("heart_rate_bpm")
    if (
        not isinstance(atrial, (int, float))
        or isinstance(atrial, bool)
        or not isinstance(ventricular, (int, float))
        or isinstance(ventricular, bool)
    ):
        return False
    tolerance = max(5.0, abs(float(ventricular)) * 0.10)
    return abs(float(atrial) - float(ventricular)) <= tolerance


def _validate_candidate_evidence_hierarchy(
    problems: list[str],
    diagnoses: Any,
    evidence_document: Mapping[str, Any] | None,
) -> None:
    """Block promotion of explicitly ambiguous detector evidence.

    This is an evidence-validity gate, not a hidden diagnostic rule engine.
    It does not add a diagnosis or compare with a dataset label. It only
    prevents a positive conclusion when the artifact itself says that the
    decisive observation is candidate-only or boundary-ambiguous. Such
    hypotheses remain legal in ``differential_diagnoses``.
    """
    if not isinstance(evidence_document, Mapping) or not isinstance(diagnoses, list):
        return

    rhythm = _as_mapping(evidence_document.get("rhythm_inputs"))
    preexcitation = _as_mapping(rhythm.get("preexcitation"))
    af_afl = _as_mapping(rhythm.get("af_afl"))
    av_block = _as_mapping(rhythm.get("av_block"))
    av_evidence = _as_mapping(av_block.get("evidence"))
    qrst_quality = _as_mapping(af_afl.get("qrst_subtraction_quality"))
    p_summary = _p_boundary_summary(evidence_document)

    for index, diagnosis in enumerate(diagnoses):
        if not isinstance(diagnosis, Mapping):
            continue
        code = str(diagnosis.get("code") or "")
        where = f"diagnoses[{index}] `{code}`"
        support_citations = {
            str(citation)
            for item in (diagnosis.get("evidence") or [])
            if isinstance(item, Mapping)
            for citation in (item.get("citations") or [])
        }

        if code in _PREEXCITATION_CODES and preexcitation:
            delta_count = preexcitation.get("delta_lead_count")
            has_delta_candidates = (
                isinstance(delta_count, (int, float))
                and not isinstance(delta_count, bool)
                and float(delta_count) > 0
            )
            pr_shortening = bool(
                preexcitation.get("short_pr_interval")
                or preexcitation.get("short_pr_segment")
            )
            if has_delta_candidates and not pr_shortening:
                problems.append(
                    f"{where} promotes delta detector candidates despite absent "
                    "independent PR-shortening corroboration; delta lead/beat/"
                    "confidence fields are one candidate-only evidence family. "
                    "Remove the positive diagnosis or move it to the differential."
                )

        if code in _FLUTTER_CODES:
            f_confidence = af_afl.get("F_wave_confidence")
            f_candidate = bool(af_afl.get("F_wave_multilead_consensus")) and (
                isinstance(f_confidence, (int, float))
                and not isinstance(f_confidence, bool)
                and float(f_confidence) >= 0.5
            )
            organized_power = qrst_quality.get("organized_spectral_power_ratio")
            weak_residual_organization = (
                isinstance(organized_power, (int, float))
                and not isinstance(organized_power, bool)
                and float(organized_power) < 0.20
            )
            if f_candidate and weak_residual_organization:
                problems.append(
                    f"{where} promotes the flutter candidate detector while the "
                    "residual signal lacks corroborating organization; keep it as "
                    "a differential unless another independent evidence family "
                    "supports the conclusion."
                )
            if (
                f_candidate
                and _rates_match(evidence_document)
                and p_summary["available"]
                and p_summary["predominantly_ambiguous"]
            ):
                problems.append(
                    f"{where} promotes a same-chain flutter candidate despite "
                    "matching measured atrial/ventricular rates and predominantly "
                    "T/A-ambiguous P boundaries. Extra atrial events or F-wave "
                    "confidence from these ambiguous detector paths are not "
                    "independent confirmation; move the hypothesis to the "
                    "differential unless clean corroboration exists."
                )

        if code in _MULTIFOCAL_P_CODES and p_summary["available"]:
            if p_summary["clean_cluster_count"] < 3:
                problems.append(
                    f"{where} requires distinct clean P morphologies, but fewer "
                    "than three morphology clusters remain after excluding "
                    "accepted+ta_ambiguous boundaries. Ambiguous cluster ids may "
                    "inform uncertainty, not establish a multifocal rhythm."
                )

        if code in _ECTOPIC_P_CODES and p_summary["available"]:
            if p_summary["clean_accepted_count"] < 2:
                problems.append(
                    f"{where} depends on P direction/morphology, but fewer than "
                    "two accepted non-T/A-ambiguous P assessments are available. "
                    "Retain the observations as limited evidence and move the "
                    "mechanism to the differential."
                )

        if code in _ATRIAL_MORPHOLOGY_CODES and p_summary["available"]:
            if p_summary["clean_accepted_count"] < 2:
                problems.append(
                    f"{where} promotes a P-morphology atrial-abnormality "
                    "diagnosis, but fewer than two accepted non-T/A-ambiguous "
                    "P assessments are available. Ambiguous P duration, "
                    "notching, biphasic shape or terminal-component candidates "
                    "may support uncertainty, but cannot establish a positive "
                    "atrial abnormality; move it to the differential or an "
                    "abstention."
                )
            if code == "p_wave_abnormality" and any(
                str(citation).removeprefix("ev:").endswith("/ta_ambiguous")
                and _resolve_document_pointer(evidence_document, citation) is True
                for citation in support_citations
            ):
                problems.append(
                    f"{where} uses T/A ambiguity as positive P-wave morphology "
                    "evidence. Ambiguity is a measurement limitation, not an "
                    "abnormal morphology; move it to counterevidence or abstain."
                )

        if code in _SINUS_ORIGIN_CODES and p_summary["available"]:
            if (
                p_summary["clean_accepted_count"] < 2
                or p_summary["predominantly_ambiguous"]
            ):
                problems.append(
                    f"{where} asserts sinus origin without repeated clean P-wave "
                    "support. Regular rate can support an organized rhythm or "
                    "bradycardia phenotype, but predominantly T/A-ambiguous P "
                    "boundaries cannot establish sinus mechanism; use the broader "
                    "rate phenotype or a differential."
                )
            if (
                av_evidence.get("localized_atrial_event_excess") is True
                and av_evidence.get(
                    "constant_multiple_atrial_events_requires_validation"
                )
                is True
            ):
                problems.append(
                    f"{where} asserts sinus origin while the atrial candidate stream "
                    "contains unresolved localized excess events. Candidate over-"
                    "detection is possible, but it must be reconciled before a "
                    "positive sinus-mechanism conclusion."
                )


def candidate_evidence_hierarchy_problems(
    diagnoses: list[Mapping[str, Any]],
    evidence_document: Mapping[str, Any] | None,
) -> list[str]:
    """Return candidate-only evidence defects before positive placement.

    Compact pathway placement is program-owned.  Running this narrow gate at
    that boundary keeps a candidate in the differential instead of first
    promoting it and then failing the complete verdict contract.  It neither
    adds a diagnosis nor weakens any existing evidence rule.
    """

    problems: list[str] = []
    _validate_candidate_evidence_hierarchy(
        problems,
        diagnoses,
        evidence_document,
    )
    return problems


def _validate_diagnosis_exclusivity(
    problems: list[str], diagnoses: Any
) -> None:
    """Reject internally contradictory positive diagnosis combinations."""
    if not isinstance(diagnoses, list):
        return
    codes = {
        str(item.get("code") or "")
        for item in diagnoses
        if isinstance(item, Mapping)
    }
    incompatible = sorted(codes - {"normal_ecg", "sinus_rhythm", ""})
    if "normal_ecg" in codes and incompatible:
        problems.append(
            "diagnosis section combines `normal_ecg` with positive abnormal "
            f"diagnosis code(s): {', '.join(incompatible)}; remove "
            "`normal_ecg` or move unconfirmed abnormalities out of "
            "`diagnoses`"
        )


_NO_CONFIRMED_POSITIVE_RE = re.compile(
    r"(?:\u672a\u5f62\u6210(?:\u8fbe\u5230[^\uff0c\u3002\uff1b;]{0,24})?(?:\u9633\u6027|\u786e\u8bca|\u786e\u8ba4)|"
    r"\u65e0(?:\u5df2)?\u786e\u8ba4(?:\u7684)?(?:\u9633\u6027)?\u8bca\u65ad|\u5c1a\u65e0(?:\u5df2)?\u786e\u8ba4|"
    r"\u672a(?:\u80fd)?\u786e\u8ba4(?:\u660e\u786e\u7684?)?(?:\u9633\u6027)?\u8bca\u65ad|"
    r"no\s+confirmed|no\s+positive|unconfirmed|indeterminate)",
    re.IGNORECASE,
)


def _validate_complete_interpretations(
    problems: list[str],
    verdict: Mapping[str, Any],
    *,
    grounded_values: Sequence[tuple[float, str | None]],
) -> None:
    """Ensure Top-3 entries are complete interpretations, not finding snippets."""
    interpretations = verdict.get("ranked_complete_interpretations")
    if not isinstance(interpretations, list):
        problems.append("`ranked_complete_interpretations` must be an array")
        return
    if not interpretations:
        problems.append(
            "`ranked_complete_interpretations` must contain the primary complete "
            "ECG interpretation"
        )
        return
    if len(interpretations) > 3:
        problems.append("`ranked_complete_interpretations` may contain at most 3 items")

    diagnoses = _items_for_contract(verdict.get("diagnoses"))
    differentials = _items_for_contract(verdict.get("differential_diagnoses"))
    positive_codes = {
        str(item.get("code") or "") for item in diagnoses if item.get("code")
    }
    differential_codes = {
        str(item.get("code") or "") for item in differentials if item.get("code")
    }
    conclusion_codes = positive_codes | differential_codes
    positive_confidences = {
        str(item.get("confidence") or "") for item in diagnoses
    }
    if not positive_codes:
        summary = str(verdict.get("summary") or "")
        if not _NO_CONFIRMED_POSITIVE_RE.search(summary):
            problems.append(
                "`summary` must explicitly state that no positive diagnosis was "
                "confirmed when `diagnoses` is empty; do not promote a rejected "
                "or unresolved hypothesis through summary prose"
            )
    rendered_diagnoses: set[str] = set()
    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    previous_confidence_order = -1

    for index, interpretation in enumerate(interpretations):
        where = f"ranked_complete_interpretations[{index}]"
        if not isinstance(interpretation, Mapping):
            problems.append(f"{where} is not an object")
            continue
        expected_rank = index + 1
        if interpretation.get("rank") != expected_rank:
            problems.append(f"{where}.rank must be {expected_rank}")
        expected_type = "PRIMARY" if index == 0 else "ALTERNATIVE"
        if interpretation.get("interpretation_type") != expected_type:
            problems.append(
                f"{where}.interpretation_type must be `{expected_type}`"
            )
        complete_diagnosis = str(
            interpretation.get("complete_diagnosis") or ""
        ).strip()
        if not complete_diagnosis:
            problems.append(f"{where}.complete_diagnosis must be non-empty")
        elif len(complete_diagnosis) > 320:
            problems.append(
                f"{where}.complete_diagnosis must be concise (at most 320 characters)"
            )
        elif complete_diagnosis in rendered_diagnoses:
            problems.append(f"{where}.complete_diagnosis duplicates an earlier rank")
        rendered_diagnoses.add(complete_diagnosis)
        _reject_quantities(
            problems,
            f"{where}.complete_diagnosis",
            complete_diagnosis,
            grounded_values=grounded_values,
        )
        uncertainty = str(interpretation.get("key_uncertainty") or "").strip()
        if not uncertainty:
            problems.append(f"{where}.key_uncertainty must be non-empty")
        elif len(uncertainty) > KEY_UNCERTAINTY_MAX_CHARS:
            problems.append(
                f"{where}.key_uncertainty must be concise "
                f"(at most {KEY_UNCERTAINTY_MAX_CHARS} characters)"
            )
        _reject_quantities(
            problems,
            f"{where}.key_uncertainty",
            uncertainty,
            grounded_values=grounded_values,
        )
        confidence = interpretation.get("confidence")
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            problems.append(f"{where}.confidence is invalid")
        else:
            current_confidence_order = confidence_order[str(confidence)]
            if current_confidence_order < previous_confidence_order:
                problems.append(
                    f"{where}.confidence is higher than the preceding complete "
                    "interpretation; ranks must run from most to least confident"
                )
            previous_confidence_order = current_confidence_order
        basis = interpretation.get("basis_codes")
        if not isinstance(basis, list):
            problems.append(f"{where}.basis_codes must be an array")
            basis_codes: set[str] = set()
        else:
            basis_rows = [str(code or "") for code in basis]
            basis_codes = set(basis_rows)
            if len(basis_codes) != len(basis_rows):
                problems.append(f"{where}.basis_codes contains duplicates")
            unknown = sorted(basis_codes - conclusion_codes)
            if unknown:
                problems.append(
                    f"{where}.basis_codes references code(s) absent from the "
                    "positive and differential sections: " + ", ".join(unknown)
                )
        if index == 0 and basis_codes != positive_codes:
            missing = sorted(positive_codes - basis_codes)
            extra = sorted(basis_codes - positive_codes)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unconfirmed " + ", ".join(extra))
            problems.append(
                f"{where} must integrate exactly every positive diagnosis code "
                f"({'; '.join(details)})"
            )
        if (
            index == 0
            and not positive_codes
            and not _NO_CONFIRMED_POSITIVE_RE.search(complete_diagnosis)
        ):
            problems.append(
                f"{where}.complete_diagnosis must explicitly frame the primary "
                "interpretation as unconfirmed/indeterminate because `diagnoses` "
                "is empty"
            )
        if index == 0 and confidence == "HIGH" and "MEDIUM" in positive_confidences:
            problems.append(
                f"{where}.confidence cannot be HIGH because the integrated "
                "primary interpretation contains a MEDIUM-confidence component"
            )
        if index > 0:
            if not differential_codes:
                problems.append(
                    f"{where} is padded despite there being no differential "
                    "mechanism; return fewer than three complete interpretations"
                )
            elif not (basis_codes & differential_codes):
                problems.append(
                    f"{where} must include at least one registered differential "
                    "diagnosis code to define a genuine alternative"
                )
            if confidence == "HIGH":
                problems.append(
                    f"{where}.confidence must be MEDIUM or LOW for an unresolved "
                    "alternative interpretation"
                )


def _items_for_contract(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _validate_hypothesis_output_gate(
    problems: list[str],
    verdict: Mapping[str, Any],
    hypothesis_state: Mapping[str, Any] | None,
) -> None:
    """Make the final Challenge disposition impossible to bypass in synthesis."""

    if not isinstance(hypothesis_state, Mapping):
        return
    rows = hypothesis_state.get("hypotheses")
    if not isinstance(rows, list):
        return
    program_owned_ledger = str(hypothesis_state.get("ledger_schema") or "").startswith(
        "diagnostic-ledger."
    )
    statuses: dict[str, set[str]] = {}
    challenge_passed: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        code = str(row.get("diagnosis_code") or "")
        status = str(row.get("status") or "")
        if code in DIAGNOSIS_CATALOG and status:
            statuses.setdefault(code, set()).add(status)
            challenge_passed[code] = bool(
                challenge_passed.get(code, False)
                or row.get("challenge_passed")
                or (not program_owned_ledger and status == "supported")
            )
    placement = hypothesis_state.get("allowed_final_placement")
    # Compatibility with historical artifacts that predate diagnosis_code.
    # A current program-owned ledger may legitimately have no hypotheses and
    # still synthesize ``normal_ecg`` from complete normal domain coverage.
    if not statuses and not (
        program_owned_ledger and isinstance(placement, Mapping)
    ):
        return

    positive_rows = _items_for_contract(verdict.get("diagnoses"))
    differential_rows = _items_for_contract(verdict.get("differential_diagnoses"))
    positive_codes = {str(row.get("code") or "") for row in positive_rows}
    differential_codes = {str(row.get("code") or "") for row in differential_rows}
    if program_owned_ledger and isinstance(placement, Mapping):
        required_positive = {
            str(code) for code in (placement.get("diagnoses") or [])
        }
        required_differential = {
            str(code) for code in (placement.get("differential_diagnoses") or [])
        }
    else:
        required_positive = {
            code
            for code, observed in statuses.items()
            if "supported" in observed and challenge_passed.get(code, False)
        }
        required_differential = {
            code
            for code, observed in statuses.items()
            if observed & {"raised", "provisional", "weak", "weakened", "unresolved"}
            and code not in required_positive
        }
    for index, diagnosis in enumerate(positive_rows):
        code = str(diagnosis.get("code") or "")
        observed = statuses.get(code, set())
        if code in required_positive:
            continue
        disposition = ", ".join(sorted(observed)) if observed else "not tracked"
        problems.append(
            f"diagnoses[{index}] `{code}` bypasses the final hypothesis gate: "
            f"Challenge status is {disposition} and challenge_passed="
            f"{challenge_passed.get(code, False)}, not a challenged support. Keep weakened/"
            "unresolved hypotheses in differentials or abstentions and omit "
            "rejected hypotheses from positive conclusions."
        )
    missing_positive = sorted(required_positive - positive_codes)
    if missing_positive:
        problems.append(
            "diagnoses omits program-supported code(s): " + ", ".join(missing_positive)
        )
    missing_differential = sorted(required_differential - differential_codes)
    if missing_differential:
        problems.append(
            "differential_diagnoses omits unresolved ledger code(s): "
            + ", ".join(missing_differential)
        )
    misplaced_differential = sorted(differential_codes - required_differential)
    if misplaced_differential:
        problems.append(
            "differential_diagnoses contains code(s) whose frozen placement forbids it: "
            + ", ".join(misplaced_differential)
        )


def validate_verdict(
    verdict: Any,
    *,
    evidence_document: Mapping[str, Any] | None = None,
    hypothesis_state: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate diagnosis semantics without comparing to rule-engine output."""
    if not isinstance(verdict, dict):
        return [f"verdict is {type(verdict).__name__}, expected a JSON object"]
    problems: list[str] = []
    for path in non_english_text_paths(verdict, "verdict"):
        problems.append(f"{path} must be English-only and must not contain Han-script text")
    required_top = (
        "summary",
        "ranked_complete_interpretations",
        "diagnoses",
        "differential_diagnoses",
        "interval_measurement_contexts",
        "abstentions",
        "quality_assessment",
        "human_review",
    )
    for key in required_top:
        if key not in verdict:
            problems.append(f"missing top-level key `{key}`")
    all_grounded_values = _verdict_evidence_values(verdict)
    _reject_quantities(
        problems,
        "`summary`",
        verdict.get("summary"),
        grounded_values=all_grounded_values,
    )
    _validate_complete_interpretations(
        problems,
        verdict,
        grounded_values=all_grounded_values,
    )
    _validate_hypothesis_output_gate(problems, verdict, hypothesis_state)
    _validate_evidence_semantics(problems, verdict, evidence_document)

    allowed_codes = allowed_diagnosis_codes(evidence_document)
    seen: set[str] = set()
    diagnoses = verdict.get("diagnoses")
    if not isinstance(diagnoses, list):
        problems.append("`diagnoses` must be an array")
    else:
        for index, diagnosis in enumerate(diagnoses):
            where = f"diagnoses[{index}]"
            if not isinstance(diagnosis, Mapping):
                problems.append(f"{where} is not an object")
                continue
            for key in (
                "code",
                "statement",
                "category",
                "confidence",
                "urgency",
                "evidence",
                "counterevidence",
                "reasoning",
            ):
                if key not in diagnosis:
                    problems.append(f"{where} is missing `{key}`")
            code = str(diagnosis.get("code") or "")
            if not code:
                problems.append(f"{where} has an empty `code`")
            elif code not in allowed_codes:
                problems.append(
                    f"{where}.code `{code}` is not in the diagnostic vocabulary"
                )
            elif code in seen:
                problems.append(f"{where} duplicates diagnosis code `{code}`")
            seen.add(code)
            category = diagnosis.get("category")
            if category not in DIAGNOSTIC_CATEGORIES:
                problems.append(f"{where}.category is invalid: {category!r}")
            elif code in DIAGNOSIS_CATALOG:
                expected_category = DIAGNOSIS_CATALOG[code][1]
                if category != expected_category:
                    problems.append(
                        f"{where}.category `{category}` conflicts with code "
                        f"`{code}`; expected `{expected_category}`"
                    )
            if diagnosis.get("confidence") not in {"HIGH", "MEDIUM"}:
                problems.append(
                    f"{where}.confidence must be HIGH or MEDIUM; LOW-confidence "
                    "hypotheses belong in `differential_diagnoses`"
                )
            if diagnosis.get("urgency") not in {
                "EMERGENT",
                "URGENT",
                "ROUTINE",
                "NONE",
            }:
                problems.append(f"{where}.urgency is invalid")
            diagnosis_values = [
                *_evidence_values(diagnosis.get("evidence")),
                *_evidence_values(diagnosis.get("counterevidence")),
            ]
            _reject_quantities(
                problems,
                f"{where}.statement",
                diagnosis.get("statement"),
                grounded_values=diagnosis_values,
            )
            _reject_quantities(
                problems,
                f"{where}.reasoning",
                diagnosis.get("reasoning"),
                grounded_values=diagnosis_values,
            )
            combined_narrative = " ".join(
                str(diagnosis.get(key) or "")
                for key in ("statement", "reasoning")
            ).lower()
            if code in {
                "short_qt",
                "borderline_short_qt",
                "possible_short_qt_pattern",
            } and any(token in combined_narrative for token in ("\u5ef6\u957f", "prolong", "long qt")):
                problems.append(
                    f"{where}.code `{code}` conflicts with narrative describing "
                    "QT/QTc prolongation"
                )
            if code in {
                "prolonged_qt",
                "markedly_prolonged_qt",
            } and any(token in combined_narrative for token in ("\u7f29\u77ed", "short qt")):
                problems.append(
                    f"{where}.code `{code}` conflicts with narrative describing "
                    "QT/QTc shortening"
                )
            _validate_evidence_items(
                problems,
                diagnosis.get("evidence"),
                f"{where}.evidence",
                require_nonempty=True,
            )
            _validate_evidence_items(
                problems,
                diagnosis.get("counterevidence"),
                f"{where}.counterevidence",
                require_nonempty=False,
            )
        _validate_candidate_evidence_hierarchy(
            problems,
            diagnoses,
            evidence_document,
        )
        _validate_diagnosis_exclusivity(problems, diagnoses)
        problems.extend(
            validate_diagnosis_measurement_semantics(
                verdict,
                evidence_document,
            )
        )

    differentials = verdict.get("differential_diagnoses")
    if not isinstance(differentials, list):
        problems.append("`differential_diagnoses` must be an array")
    else:
        for index, differential in enumerate(differentials):
            where = f"differential_diagnoses[{index}]"
            if not isinstance(differential, Mapping):
                problems.append(f"{where} is not an object")
                continue
            for key in (
                "code",
                "statement",
                "confidence",
                "supporting_evidence",
                "counterevidence",
                "what_would_resolve_it",
            ):
                if key not in differential:
                    problems.append(f"{where} is missing `{key}`")
            code = str(differential.get("code") or "")
            if not code or code not in allowed_codes:
                problems.append(f"{where}.code `{code}` is not in the diagnostic vocabulary")
            if code in seen:
                problems.append(f"{where}.code `{code}` duplicates a positive diagnosis")
            if differential.get("confidence") not in {"MEDIUM", "LOW"}:
                problems.append(f"{where}.confidence must be MEDIUM or LOW")
            differential_values = [
                *_evidence_values(differential.get("supporting_evidence")),
                *_evidence_values(differential.get("counterevidence")),
            ]
            _reject_quantities(
                problems,
                f"{where}.statement",
                differential.get("statement"),
                grounded_values=differential_values,
            )
            _reject_quantities(
                problems,
                f"{where}.what_would_resolve_it",
                differential.get("what_would_resolve_it"),
                grounded_values=differential_values,
            )
            _validate_evidence_items(
                problems,
                differential.get("supporting_evidence"),
                f"{where}.supporting_evidence",
                require_nonempty=True,
            )
            _validate_evidence_items(
                problems,
                differential.get("counterevidence"),
                f"{where}.counterevidence",
                require_nonempty=False,
            )

    contexts = verdict.get("interval_measurement_contexts")
    seen_contexts: set[str] = set()
    if not isinstance(contexts, list):
        problems.append("`interval_measurement_contexts` must be an array")
    else:
        for index, context in enumerate(contexts):
            where = f"interval_measurement_contexts[{index}]"
            if not isinstance(context, Mapping):
                problems.append(f"{where} is not an object")
                continue
            for key in (
                "interval",
                "status",
                "interval_conclusion",
                "component_waveform_assessment",
                "residual_evidence",
                "interpretive_impact",
                "what_would_resolve_it",
            ):
                if key not in context:
                    problems.append(f"{where} is missing `{key}`")
            interval = str(context.get("interval") or "")
            if interval not in {"PR", "QT_QTc"}:
                problems.append(f"{where}.interval is invalid: {interval!r}")
            elif interval in seen_contexts:
                problems.append(f"{where} duplicates interval `{interval}`")
            seen_contexts.add(interval)
            if context.get("status") not in {"limited", "unavailable"}:
                problems.append(f"{where}.status must be limited or unavailable")
            context_values = _evidence_values(context.get("residual_evidence"))
            for key in (
                "interval_conclusion",
                "component_waveform_assessment",
                "interpretive_impact",
                "what_would_resolve_it",
            ):
                if not str(context.get(key) or "").strip():
                    problems.append(f"{where}.{key} must be non-empty")
                _reject_quantities(
                    problems,
                    f"{where}.{key}",
                    context.get(key),
                    grounded_values=context_values,
                )
            _validate_evidence_items(
                problems,
                context.get("residual_evidence"),
                f"{where}.residual_evidence",
                require_nonempty=True,
            )
            if interval in {"PR", "QT_QTc"} and not _has_component_wave_citation(
                interval,
                context.get("residual_evidence"),
            ):
                component = "P/AV" if interval == "PR" else "T/U"
                problems.append(
                    f"{where}.residual_evidence does not cite any {component} "
                    "component-wave observation; an interval-status pointer alone "
                    "does not satisfy residual waveform review"
                )
        for interval in sorted(
            _required_interval_contexts(evidence_document) - seen_contexts
        ):
            component = "T/U waveform" if interval == "QT_QTc" else "P wave and AV association"
            problems.append(
                f"missing interval_measurement_contexts entry for limited `{interval}`; "
                f"review residual {component} evidence rather than treating the "
                "interval as an information-free missing value"
            )

    abstentions = verdict.get("abstentions")
    if not isinstance(abstentions, list):
        problems.append("`abstentions` must be an array")
    else:
        for index, abstention in enumerate(abstentions):
            where = f"abstentions[{index}]"
            if not isinstance(abstention, Mapping):
                problems.append(f"{where} is not an object")
                continue
            for key in ("topic", "reason", "what_would_resolve_it"):
                if not str(abstention.get(key) or "").strip():
                    problems.append(f"{where} is missing `{key}`")
                _reject_quantities(
                    problems,
                    f"{where}.{key}",
                    abstention.get(key),
                    grounded_values=all_grounded_values,
                )

    quality = verdict.get("quality_assessment")
    if not isinstance(quality, Mapping):
        problems.append("`quality_assessment` must be an object")
    else:
        if quality.get("interpretability") not in {
            "adequate",
            "limited",
            "non_diagnostic",
        }:
            problems.append("`quality_assessment.interpretability` is invalid")
        limitations = quality.get("limitations")
        if not isinstance(limitations, list):
            problems.append("`quality_assessment.limitations` must be an array")
        else:
            for index, limitation in enumerate(limitations):
                _reject_quantities(
                    problems,
                    f"quality_assessment.limitations[{index}]",
                    limitation,
                    grounded_values=all_grounded_values,
                )

    metadata = _as_mapping(
        evidence_document.get("metadata")
        if isinstance(evidence_document, Mapping)
        else None
    )
    diagnostic_gate = _as_mapping(metadata.get("diagnostic_gate"))
    gate_state = str(diagnostic_gate.get("state") or "pass").lower()
    if gate_state in {"partial", "stop"} and isinstance(quality, Mapping):
        if gate_state == "partial" and quality.get("interpretability") == "adequate":
            problems.append(
                "quality gate is partial; quality_assessment.interpretability "
                "must be limited or non_diagnostic"
            )
        if gate_state == "stop" and quality.get("interpretability") != "non_diagnostic":
            problems.append(
                "quality gate is stop; quality_assessment.interpretability must "
                "be non_diagnostic"
            )
        if not isinstance(quality.get("limitations"), list) or not quality.get("limitations"):
            problems.append(
                f"quality gate is {gate_state}; quality_assessment.limitations "
                "must be non-empty"
            )

    diagnosis_rows = _items_for_contract(verdict.get("diagnoses"))
    if gate_state == "stop":
        if diagnosis_rows or _items_for_contract(verdict.get("differential_diagnoses")):
            problems.append(
                "quality gate is stop; positive and differential ECG conclusions "
                "must both be empty"
            )
    elif gate_state == "partial":
        allowed = {
            str(value) for value in (diagnostic_gate.get("allowed_domains") or [])
        }
        allowed_categories: set[str] | None = None
        if allowed and "all" not in allowed:
            allowed_categories = set()
            if "quality" in allowed:
                allowed_categories.add("quality")
            if "rhythm" in allowed:
                allowed_categories.update(("rhythm", "rate"))
            if "ectopy" in allowed:
                allowed_categories.add("ectopy")
        suppressed_category_map = {
            "axis": {"axis"},
            "axis_quantitative": {"axis"},
            "intervals": {"interval"},
            "conduction_preexcitation": {"conduction"},
            "voltage_chamber_r_progression": {"chamber"},
            "q_st_t_u": {"ischemia_repolarization"},
            "pacing_high_risk": {"pacing"},
        }
        suppressed_categories = {
            category
            for domain in (diagnostic_gate.get("suppressed_domains") or [])
            for category in suppressed_category_map.get(str(domain), set())
        }
        for index, diagnosis in enumerate(diagnosis_rows):
            category = str(diagnosis.get("category") or "")
            if allowed_categories is not None and category not in allowed_categories:
                problems.append(
                    f"diagnoses[{index}] category `{category}` is outside the "
                    "partial quality gate's allowed domains"
                )
            if category in suppressed_categories:
                problems.append(
                    f"diagnoses[{index}] category `{category}` is suppressed by "
                    "the partial quality gate"
                )

    review = verdict.get("human_review")
    if not isinstance(review, Mapping):
        problems.append("`human_review` must be an object")
    else:
        if review.get("required") is not True:
            problems.append("`human_review.required` must be true")
        reasons = review.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            problems.append("`human_review.reasons` must be a non-empty array")
        else:
            for index, reason in enumerate(reasons):
                _reject_quantities(
                    problems,
                    f"human_review.reasons[{index}]",
                    reason,
                    grounded_values=all_grounded_values,
                )
    return problems


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Use clear English. Summarize positive diagnoses first, followed by the key uncertainties.",
        },
        "ranked_complete_interpretations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "description": (
                "Provide one to three complete ECG interpretations, not three isolated findings. The first integrates all confirmed diagnoses. "
                "Provide a second or third complete alternative only when a genuine differential mechanism exists; do not pad the list."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                    "interpretation_type": {
                        "type": "string",
                        "enum": ["PRIMARY", "ALTERNATIVE"],
                    },
                    "complete_diagnosis": {
                        "type": "string",
                        "maxLength": 320,
                        "description": (
                            "In one concise English sentence, integrate rhythm, major conduction, morphology and repolarization abnormalities, "
                            "and important limitations; do not report only an isolated finding."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                    },
                    "basis_codes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": sorted(DIAGNOSIS_CATALOG),
                        },
                    },
                    "key_uncertainty": {
                        "type": "string",
                        "maxLength": 180,
                        "description": "State the most important uncertainty in this complete interpretation in concise English.",
                    },
                },
                "required": [
                    "rank",
                    "interpretation_type",
                    "complete_diagnosis",
                    "confidence",
                    "basis_codes",
                    "key_uncertainty",
                ],
                "additionalProperties": False,
            },
        },
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "enum": sorted(DIAGNOSIS_CATALOG),
                    },
                    "statement": {
                        "type": "string",
                        "description": "State a specific diagnosis in English; do not use a generic placeholder.",
                    },
                    "category": {
                        "type": "string",
                        "enum": sorted(DIAGNOSTIC_CATEGORIES),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM"],
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["EMERGENT", "URGENT", "ROUTINE", "NONE"],
                    },
                    "evidence": {"type": "array", "items": _EVIDENCE_ITEM},
                    "counterevidence": {"type": "array", "items": _EVIDENCE_ITEM},
                    "reasoning": {
                        "type": "string",
                        "description": "Explain in English why the evidence supports or weakens this diagnosis.",
                    },
                },
                "required": [
                    "code",
                    "statement",
                    "category",
                    "confidence",
                    "urgency",
                    "evidence",
                    "counterevidence",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
        },
        "differential_diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "enum": sorted(DIAGNOSIS_CATALOG),
                    },
                    "statement": {
                        "type": "string",
                        "description": "Describe the unconfirmed differential diagnosis in English.",
                    },
                    "confidence": {"type": "string", "enum": ["MEDIUM", "LOW"]},
                    "supporting_evidence": {
                        "type": "array",
                        "items": _EVIDENCE_ITEM,
                    },
                    "counterevidence": {"type": "array", "items": _EVIDENCE_ITEM},
                    "what_would_resolve_it": {
                        "type": "string",
                        "description": "Explain in English what evidence would resolve the differential.",
                    },
                },
                "required": [
                    "code",
                    "statement",
                    "confidence",
                    "supporting_evidence",
                    "counterevidence",
                    "what_would_resolve_it",
                ],
                "additionalProperties": False,
            },
        },
        "interval_measurement_contexts": {
            "type": "array",
            "description": (
                "List only unavailable or limited PR and QT/QTc intervals. Explain the evidence retained in the component waveforms; "
                "do not equate a missing interval with an uninformative waveform."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "interval": {
                        "type": "string",
                        "enum": ["PR", "QT_QTc"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["limited", "unavailable"],
                    },
                    "interval_conclusion": {
                        "type": "string",
                        "description": "Explain in English why this interval cannot be interpreted as a reliable numeric measurement.",
                    },
                    "component_waveform_assessment": {
                        "type": "string",
                        "description": "Explain in English what information remains in the P/AV or T/U morphology.",
                    },
                    "residual_evidence": {
                        "type": "array",
                        "minItems": 1,
                        "items": _EVIDENCE_ITEM,
                    },
                    "interpretive_impact": {
                        "type": "string",
                        "description": "Explain in English which interpretations the residual waveform evidence supports or weakens, and what it cannot prove.",
                    },
                    "what_would_resolve_it": {
                        "type": "string",
                        "description": "Explain in English what is needed to distinguish technical failure from a true morphologic abnormality.",
                    },
                },
                "required": [
                    "interval",
                    "status",
                    "interval_conclusion",
                    "component_waveform_assessment",
                    "residual_evidence",
                    "interpretive_impact",
                    "what_would_resolve_it",
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
                        "description": "Give the reason in English.",
                    },
                    "what_would_resolve_it": {
                        "type": "string",
                        "description": "State in English what would resolve it.",
                    },
                },
                "required": ["topic", "reason", "what_would_resolve_it"],
                "additionalProperties": False,
            },
        },
        "quality_assessment": {
            "type": "object",
            "properties": {
                "interpretability": {
                    "type": "string",
                    "enum": ["adequate", "limited", "non_diagnostic"],
                },
                "limitations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "State each limitation in English.",
                    },
                },
            },
            "required": ["interpretability", "limitations"],
            "additionalProperties": False,
        },
        "human_review": {
            "type": "object",
            "properties": {
                "required": {"type": "boolean", "const": True},
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
    "required": [
        "summary",
        "ranked_complete_interpretations",
        "diagnoses",
        "differential_diagnoses",
        "interval_measurement_contexts",
        "abstentions",
        "quality_assessment",
        "human_review",
    ],
    "additionalProperties": False,
}


# A failed final verdict is repaired section-by-section. The model receives
# the latest canonical candidate from the backend compactor and may replace
# only the top-level sections named in ``changed_sections``. This keeps valid
# sections byte-for-byte stable across revisions and prevents a repair from
# silently re-promoting a frozen differential.
_VERDICT_SECTION_NAMES = tuple(OUTPUT_SCHEMA["required"])
VERDICT_SECTION_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_candidate_hash": {"type": "string", "maxLength": 80},
        "changed_sections": {
            "type": "array",
            "minItems": 1,
            "maxItems": len(_VERDICT_SECTION_NAMES),
            "items": {"type": "string", "enum": list(_VERDICT_SECTION_NAMES)},
        },
        "replacement_sections": {
            "type": "object",
            "properties": {
                name: copy.deepcopy(OUTPUT_SCHEMA["properties"][name])
                for name in _VERDICT_SECTION_NAMES
            },
            "additionalProperties": False,
        },
    },
    "required": [
        "base_candidate_hash",
        "changed_sections",
        "replacement_sections",
    ],
    "additionalProperties": False,
}


VERDICT_SECTION_PATCH_INSTRUCTION = """The independent diagnostic output failed deterministic validation. Repair only the invalid top-level sections named by the feedback. Return a section patch matching the supplied schema, not a complete verdict.

Current candidate hash: {candidate_hash}

{feedback}

Copy the hash exactly into `base_candidate_hash`. List every section you replace in `changed_sections`, and put exactly those complete replacement values in `replacement_sections`. Omit all unchanged sections. The diagnostic ledger is frozen: diagnosis codes and their positive/differential/rejected placement cannot change except to restore the deterministic ledger placement named in the feedback. Re-read a measurement only when its value or pointer is genuinely uncertain. In any replacement evidence section, each patient-specific measurement remains one atomic item with one displayed value, one unit and one value-bearing citation. Do not introduce a diagnosis, number or citation that was not authorized. Write every replacement text field in English."""

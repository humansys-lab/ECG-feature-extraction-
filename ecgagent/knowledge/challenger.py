"""Out-of-band medical-knowledge challenge for a completed ECG verdict.

The diagnostic Agent may have used governed general references between its
measurement survey and provisional-hypothesis phase.  This module is a second,
separate review that runs only after a verdict has passed the patient-evidence
contract.  Its retrieved excerpts are not appended to the diagnostic
conversation: that conversation receives only neutral re-examination
questions and must return to ecgfeat patient measurements before any revision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ..agent.diagnosis_catalog import DIAGNOSIS_CATALOG
from ..backends.base import LLMBackend
from .index import KnowledgeBase, KnowledgeChunk, SearchResult
from .safety import sanitize_runtime_excerpt, sanitize_runtime_label


CHALLENGE_VERSION = "ecg-knowledge-challenge.v1"
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
RUNTIME_CATEGORIES: tuple[str, ...] = (
    "diagnostic_reference",
    "morphology_reference",
    "measurement_reliability_reference",
    "failure_modes",
)
EXCLUDED_RUNTIME_CATEGORIES: tuple[str, ...] = (
    "clinical_rules_reference",
    "philips_dxl_reference",
    "glasgow_reference",
    "capability_blueprint",
)

ISSUE_TYPES: tuple[str, ...] = (
    "omission",
    "confounder",
    "causal_jump",
    "measurement_limitation",
)

KNOWLEDGE_CHALLENGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 1000},
        "issues": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(ISSUE_TYPES)},
                    "priority": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                    },
                    "topic": {"type": "string", "maxLength": 160},
                    "why_current_reasoning_is_incomplete": {
                        "type": "string",
                        "maxLength": 700,
                    },
                    "evidence_review_question": {
                        "type": "string",
                        "maxLength": 500,
                    },
                    "knowledge_refs": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "type",
                    "priority",
                    "topic",
                    "why_current_reasoning_is_incomplete",
                    "evidence_review_question",
                    "knowledge_refs",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "issues"],
    "additionalProperties": False,
}

KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT = """You are an out-of-band medical-knowledge challenger for a completed 12-lead ECG interpretation.

The initial diagnostic Agent has already finished and passed patient-evidence verification. It may have used a governed subset of general references earlier to plan targeted checks, but it has not seen the excerpts supplied in this separate review turn. You do not diagnose the patient and you do not rewrite the verdict. Use these references only to identify clinically material omissions, confounders, measurement limitations, or causal jumps in the current reasoning.

Rules:
1. A knowledge excerpt is general reference material, never patient evidence. Do not claim that a patient feature exists unless it is already stated in the verified verdict.
2. Return at most three material issues. Do not manufacture a generic request for raw-waveform or human review; those limitations are already understood. If no material issue is supported, return an empty issues array.
3. Each issue must be phrased as a neutral evidence-review question that the diagnostic Agent can answer by re-reading ecgfeat measurement tools or by narrowing an unsupported etiologic claim to an ECG phenotype/limitation.
4. Do not provide a replacement diagnosis, a patient-specific conclusion, a reference threshold, or a patient-specific number in the review question.
5. Focus especially on primary versus secondary change, detector-chain dependence, cross-lead/cross-beat consistency, measurement-failure propagation, and whether ECG evidence can identify the stated cause.
6. Use only the supplied K references. The later diagnostic revision will not see the excerpts or your explanatory text; it will see only the neutral review questions and must return to patient measurements.
7. Write every human-facing field in concise English. Do not emit Chinese text."""


@dataclass(frozen=True)
class RetrievedKnowledge:
    ref: str
    result: SearchResult

    def prompt_row(self, *, excerpt_chars: int) -> dict[str, Any]:
        chunk = self.result.chunk
        sanitized = sanitize_runtime_excerpt(
            self.result.excerpt(max_chars=excerpt_chars)
        )
        return {
            "ref": self.ref,
            "category": chunk.category,
            "title": sanitize_runtime_label(chunk.title),
            "text": sanitized.text,
            "case_material_removed": sanitized.removed_lines,
            "record_identifiers_redacted": sanitized.redacted_identifiers,
        }

    def audit_row(self) -> dict[str, Any]:
        chunk = self.result.chunk
        sanitized = sanitize_runtime_excerpt(self.result.excerpt(max_chars=100000))
        return {
            "ref": self.ref,
            "chunk_id": chunk.chunk_id,
            "category": chunk.category,
            "source": chunk.source,
            "title": chunk.title,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "source_sha256": chunk.source_sha256,
            "score": round(float(self.result.score), 6),
            "patient_planning_safe": True,
            "case_material_removed": sanitized.removed_lines,
            "record_identifiers_redacted": sanitized.redacted_identifiers,
        }


def _compact_text(value: Any, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _query_rows(verdict: Mapping[str, Any]) -> list[tuple[str, bool]]:
    """Return ``(query, needs reliability sources)`` rows from the verdict."""
    rows: list[tuple[str, bool]] = []
    for diagnosis in verdict.get("diagnoses") or []:
        if not isinstance(diagnosis, Mapping):
            continue
        code = str(diagnosis.get("code") or "")
        label = DIAGNOSIS_CATALOG.get(code, (code, ""))[0]
        statement = _compact_text(diagnosis.get("statement"), 220)
        reasoning = _compact_text(diagnosis.get("reasoning"), 300)
        query = " ".join(part for part in (code, label, statement, reasoning) if part)
        if query:
            rows.append((query, True))
    for differential in verdict.get("differential_diagnoses") or []:
        if not isinstance(differential, Mapping):
            continue
        code = str(differential.get("code") or "")
        label = DIAGNOSIS_CATALOG.get(code, (code, ""))[0]
        statement = _compact_text(differential.get("statement"), 220)
        query = " ".join(part for part in (code, label, statement, "differential diagnosis counterevidence confounder") if part)
        if query:
            rows.append((query, False))
    for context in verdict.get("interval_measurement_contexts") or []:
        if not isinstance(context, Mapping):
            continue
        interval = str(context.get("interval") or "")
        assessment = _compact_text(context.get("component_waveform_assessment"), 260)
        if interval == "QT_QTc":
            rows.append((f"QT unavailable T-wave endpoint flat T wave T-U fusion repolarization {assessment}", True))
        elif interval == "PR":
            rows.append((f"PR unavailable P-wave boundary P-T overlap P-QRS association {assessment}", True))

    codes = {
        str(item.get("code") or "")
        for key in ("diagnoses", "differential_diagnoses")
        for item in (verdict.get(key) or [])
        if isinstance(item, Mapping)
    }
    if codes & {
        "st_depression",
        "st_elevation",
        "t_wave_abnormality",
        "flat_t_wave_pattern",
        "primary_t_wave_abnormality",
        "secondary_t_wave_abnormality",
    }:
        rows.append(("ST-T change primary secondary wide QRS hypertrophy pre-excitation pacing confounder", True))
    if codes & {
        "atrial_flutter_pattern",
        "possible_atrial_flutter_pattern",
        "atrial_fibrillation",
        "atrial_fibrillation_pattern",
        "probable_atrial_fibrillation_pattern",
    }:
        rows.append(("atrial fibrillation atrial flutter F wave P-QRS association RR candidate detector independent evidence", True))
    if "ventricular_preexcitation_pattern" in codes:
        rows.append(("delta wave short PR wide QRS pre-excitation independent evidence false positive", True))
    if codes & {
        "pathological_q_waves",
        "prior_infarct_q_wave_pattern",
        "poor_r_wave_progression",
        "reversed_r_wave_progression",
    }:
        rows.append(("Q wave R-wave progression representative beat multi-beat consensus lead placement confounder", True))

    if not rows:
        summary = _compact_text(verdict.get("summary"), 500)
        rows.append((f"standard 12-lead ECG systematic interpretation omission confounder {summary}", False))
    # Stable de-duplication protects retrieval and prompt size.
    seen: set[str] = set()
    unique: list[tuple[str, bool]] = []
    for query, reliability in rows:
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append((query, reliability))
    return unique[:8]


def _ranked_retrieval(
    base: KnowledgeBase,
    verdict: Mapping[str, Any],
    *,
    max_chunks: int,
) -> list[RetrievedKnowledge]:
    candidates: dict[str, tuple[float, SearchResult]] = {}
    for query_index, (query, needs_reliability) in enumerate(_query_rows(verdict)):
        category_groups: list[tuple[tuple[str, ...], float]] = [
            (("diagnostic_reference", "morphology_reference"), 1.0),
        ]
        if needs_reliability:
            category_groups.append(
                (("measurement_reliability_reference", "failure_modes"), 0.85)
            )
        for categories, category_weight in category_groups:
            available_categories = tuple(
                category for category in categories if category in base.categories
            )
            if not available_categories:
                continue
            for result in base.search(
                query,
                limit=4,
                categories=available_categories,
            ):
                score = (
                    float(result.score) * category_weight
                    + max(0.0, 1.5 - 0.15 * query_index)
                )
                current = candidates.get(result.chunk.chunk_id)
                if current is None or score > current[0]:
                    candidates[result.chunk.chunk_id] = (score, result)

    ordered = sorted(
        candidates.values(),
        key=lambda row: (
            -row[0],
            row[1].chunk.source,
            row[1].chunk.line_start,
        ),
    )
    selected: list[SearchResult] = []
    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for _, result in ordered:
        chunk = result.chunk
        # Prevent one long document or one category from monopolizing the
        # challenge context while still allowing the best two corroborating
        # sections from a source.
        if source_counts.get(chunk.source, 0) >= 2:
            continue
        if category_counts.get(chunk.category, 0) >= 4:
            continue
        selected.append(result)
        source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1
        category_counts[chunk.category] = category_counts.get(chunk.category, 0) + 1
        if len(selected) >= max(1, min(int(max_chunks), 12)):
            break
    return [
        RetrievedKnowledge(ref=f"K{index}", result=result)
        for index, result in enumerate(selected, start=1)
    ]


def _parse_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _sanitize_review(
    parsed: Mapping[str, Any],
    retrieved: Sequence[RetrievedKnowledge],
) -> dict[str, Any]:
    allowed_refs = {row.ref for row in retrieved}
    issues: list[dict[str, Any]] = []
    for row in parsed.get("issues") or []:
        if not isinstance(row, Mapping):
            continue
        issue_type = str(row.get("type") or "")
        priority = str(row.get("priority") or "")
        topic = _compact_text(row.get("topic"), 160)
        why = _compact_text(row.get("why_current_reasoning_is_incomplete"), 700)
        question = _compact_text(row.get("evidence_review_question"), 500)
        # Reference thresholds or citation tokens must never cross into the
        # diagnostic revision as if they were patient measurements.  A useful
        # challenge can always ask for the relation qualitatively and let the
        # Agent re-read the exact patient values from ecgfeat.
        unsafe_question = bool(
            re.search(r"(?:ev:/|kb:|\bK\d+\b|\d)", question, re.I)
        )
        non_english = any(_HAN_RE.search(text) for text in (topic, why, question))
        refs = [
            str(ref)
            for ref in (row.get("knowledge_refs") or [])
            if str(ref) in allowed_refs
        ][:4]
        if (
            issue_type not in ISSUE_TYPES
            or priority not in {"HIGH", "MEDIUM", "LOW"}
            or not topic
            or not why
            or not question
            or unsafe_question
            or non_english
            or not refs
        ):
            continue
        issues.append(
            {
                "type": issue_type,
                "priority": priority,
                "topic": topic,
                "why_current_reasoning_is_incomplete": why,
                "evidence_review_question": question,
                "knowledge_refs": list(dict.fromkeys(refs)),
            }
        )
        if len(issues) >= 3:
            break
    return {
        "summary": (
            f"The knowledge challenge raised {len(issues)} questions requiring review of patient measurement evidence."
            if issues
            else "No material knowledge challenge requiring patient-measurement reacquisition was found."
        ),
        "issues": issues,
    }


class KnowledgeChallenger:
    """Retrieve governed references and run one isolated challenge turn."""

    def __init__(
        self,
        base: KnowledgeBase | None = None,
        *,
        max_chunks: int = 8,
        excerpt_chars: int = 1400,
    ) -> None:
        self.base = base or KnowledgeBase.from_project()
        self.max_chunks = max(1, min(int(max_chunks), 12))
        self.excerpt_chars = max(400, min(int(excerpt_chars), 2400))

    def corpus_fingerprint(self) -> str:
        rows = sorted(
            (chunk.chunk_id, chunk.source_sha256)
            for chunk in self.base.chunks
            if chunk.category in RUNTIME_CATEGORIES
        )
        payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def review(
        self,
        verdict: Mapping[str, Any],
        *,
        backend: LLMBackend,
        max_tokens: int = 3072,
    ) -> dict[str, Any]:
        retrieved = _ranked_retrieval(
            self.base,
            verdict,
            max_chunks=self.max_chunks,
        )
        references = [row.audit_row() for row in retrieved]
        if not retrieved:
            return {
                "version": CHALLENGE_VERSION,
                "status": "no_knowledge_match",
                "summary": "No knowledge excerpt sufficient for an independent challenge was retrieved.",
                "issues": [],
                "references": [],
                "runtime_categories": list(RUNTIME_CATEGORIES),
                "excluded_categories": list(EXCLUDED_RUNTIME_CATEGORIES),
                "corpus_fingerprint": self.corpus_fingerprint(),
            }
        prompt = {
            "task": (
                "Review the already verified ECG verdict against the general "
                "knowledge excerpts. Return only the required JSON object."
            ),
            "verified_verdict": verdict,
            "general_knowledge_excerpts": [
                row.prompt_row(excerpt_chars=self.excerpt_chars)
                for row in retrieved
            ],
        }
        response = backend.complete(
            system=KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT,
            messages=[
                backend.user_turn(
                    json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
                )
            ],
            tools=[],
            max_tokens=max_tokens,
            response_schema=KNOWLEDGE_CHALLENGE_SCHEMA,
        )
        parsed = _parse_object(response.text)
        if parsed is None:
            return {
                "version": CHALLENGE_VERSION,
                "status": "error",
                "summary": "The knowledge-challenge model did not return a parseable structured result.",
                "issues": [],
                "references": references,
                "runtime_categories": list(RUNTIME_CATEGORIES),
                "excluded_categories": list(EXCLUDED_RUNTIME_CATEGORIES),
                "corpus_fingerprint": self.corpus_fingerprint(),
                "error": "unparseable challenge response",
            }
        review = _sanitize_review(parsed, retrieved)
        return {
            "version": CHALLENGE_VERSION,
            "status": "issues_found" if review["issues"] else "passed",
            **review,
            "references": references,
            "runtime_categories": list(RUNTIME_CATEGORIES),
            "excluded_categories": list(EXCLUDED_RUNTIME_CATEGORIES),
            "corpus_fingerprint": self.corpus_fingerprint(),
        }

    @staticmethod
    def neutral_revision_feedback(review: Mapping[str, Any]) -> str | None:
        issues = [
            issue
            for issue in (review.get("issues") or [])
            if isinstance(issue, Mapping)
            and str(issue.get("evidence_review_question") or "").strip()
        ]
        if not issues:
            return None
        lines = [
            "POST-HOC KNOWLEDGE CHALLENGE. The separate review excerpts and "
            "their conclusions are deliberately not shown here. Earlier "
            "provisional planning may have used governed general references, "
            "but those were never patient evidence. Treat the following as "
            "neutral questions, not facts or replacement diagnoses:",
        ]
        for index, issue in enumerate(issues[:3], start=1):
            lines.append(
                f"{index}. [{issue.get('type')}/{issue.get('priority')}] "
                f"{_compact_text(issue.get('topic'), 160)}: "
                f"{_compact_text(issue.get('evidence_review_question'), 500)}"
            )
        lines.extend(
            [
                "Re-read the relevant ecgfeat measurement tools before changing "
                "the verdict. Every retained or new patient claim must be supported "
                "by exact ev:/ citations read in this session. Do not cite or copy "
                "knowledge text, thresholds, rule-engine outputs, DXL/Glasgow "
                "interpretations, or dataset labels. If ecgfeat evidence cannot "
                "answer a question, narrow the claim to an ECG phenotype, move it "
                "to the differential, or state the limitation/abstention. Return "
                "the complete verdict JSON only."
            ]
        )
        return "\n".join(lines)


def chunk_is_runtime_governed(chunk: KnowledgeChunk) -> bool:
    """Public test/helper for the strict patient-review category boundary."""
    return chunk.category in RUNTIME_CATEGORIES

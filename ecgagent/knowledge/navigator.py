"""Governed medical-knowledge navigation between survey and investigation.

The navigator is deliberately not an ecgfeat tool and it never returns patient
facts.  It retrieves general diagnostic, morphology and measurement-reliability
references after the measurement-only survey has produced observations.  The
next model phase uses those references to form a provisional differential and
an ecgfeat checking plan; every final patient claim must still return to
``ev:/...`` measurement evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Sequence

from .challenger import EXCLUDED_RUNTIME_CATEGORIES, RUNTIME_CATEGORIES
from .index import KnowledgeBase, SearchResult
from .safety import sanitize_runtime_excerpt, sanitize_runtime_label


NAVIGATION_VERSION = "ecg-knowledge-navigation.v2"
_POINTER_RE = re.compile(r"(?:ev:)?/[A-Za-z0-9_./~-]+")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NavigationReference:
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


def _compact_survey(text: str, limit: int = 5000) -> str:
    cleaned = _POINTER_RE.sub(" ", str(text or ""))
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return cleaned[:limit]


class KnowledgeNavigator:
    """Retrieve diverse, governed references for provisional diagnosis planning."""

    def __init__(
        self,
        base: KnowledgeBase | None = None,
        *,
        max_chunks: int = 6,
        excerpt_chars: int = 900,
    ) -> None:
        self.base = base or KnowledgeBase.from_project()
        self.max_chunks = max(1, min(int(max_chunks), 8))
        self.excerpt_chars = max(400, min(int(excerpt_chars), 1800))

    def corpus_fingerprint(self) -> str:
        rows = sorted(
            (chunk.chunk_id, chunk.source_sha256)
            for chunk in self.base.chunks
            if chunk.category in RUNTIME_CATEGORIES
        )
        payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _retrieve(self, survey_text: str) -> list[NavigationReference]:
        query = _compact_survey(survey_text)
        if not query:
            query = "standard 12-lead ECG systematic interpretation rhythm conduction ST T wave differential diagnosis"

        candidate_rows: dict[str, tuple[float, SearchResult]] = {}
        category_groups: Sequence[tuple[tuple[str, ...], float]] = (
            (("diagnostic_reference", "morphology_reference"), 1.0),
            (("measurement_reliability_reference", "failure_modes"), 0.82),
        )
        for categories, weight in category_groups:
            available = tuple(
                category for category in categories if category in self.base.categories
            )
            if not available:
                continue
            for result in self.base.search(
                query,
                limit=max(6, self.max_chunks * 2),
                categories=available,
            ):
                score = float(result.score) * weight
                current = candidate_rows.get(result.chunk.chunk_id)
                if current is None or score > current[0]:
                    candidate_rows[result.chunk.chunk_id] = (score, result)

        ordered = sorted(
            candidate_rows.values(),
            key=lambda row: (
                -row[0],
                row[1].chunk.source,
                row[1].chunk.line_start,
            ),
        )
        clinical_categories = {"diagnostic_reference", "morphology_reference"}
        clinical = [
            result for _score, result in ordered
            if result.chunk.category in clinical_categories
        ]
        reliability = [
            result for _score, result in ordered
            if result.chunk.category not in clinical_categories
        ]
        # Planning cards should primarily answer "what clinical relationship
        # must be tested?" Internal measurement-audit material is useful as a
        # caveat, but letting it occupy three of four cards primed the model on
        # implementation details and displaced differential guidance.
        pools: Sequence[tuple[list[SearchResult], int]] = (
            (clinical, max(1, self.max_chunks - 1)),
            (reliability, 1 if self.max_chunks > 1 else 0),
        )
        selected: list[SearchResult] = []
        source_counts: dict[str, int] = {}
        for pool, limit in pools:
            admitted = 0
            for result in pool:
                if admitted >= limit or len(selected) >= self.max_chunks:
                    break
                chunk = result.chunk
                if source_counts.get(chunk.source, 0) >= 2:
                    continue
                selected.append(result)
                source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1
                admitted += 1
        return [
            NavigationReference(ref=f"N{index}", result=result)
            for index, result in enumerate(selected, start=1)
        ]

    def navigate(self, survey_text: str) -> tuple[str, dict[str, Any]]:
        references = self._retrieve(survey_text)
        audit = {
            "version": NAVIGATION_VERSION,
            "enabled": True,
            "status": "matched" if references else "no_knowledge_match",
            "max_chunks": self.max_chunks,
            "reference_count": len(references),
            "references": [row.audit_row() for row in references],
            "runtime_categories": list(RUNTIME_CATEGORIES),
            "excluded_categories": list(EXCLUDED_RUNTIME_CATEGORIES),
            "corpus_fingerprint": self.corpus_fingerprint(),
            "patient_evidence": False,
        }
        if not references:
            return (
                "INTERIM MEDICAL KNOWLEDGE NAVIGATION: no governed reference "
                "matched the survey. Form a provisional differential from the "
                "surveyed ecgfeat observations, then request targeted patient "
                "measurements. General knowledge is not patient evidence.",
                audit,
            )

        payload = {
            "purpose": (
                "Use these general references to form a provisional ECG "
                "differential and decide which ecgfeat measurements should be "
                "checked next. They are not patient evidence."
            ),
            "knowledge_cards": [
                row.prompt_row(excerpt_chars=self.excerpt_chars)
                for row in references
            ],
        }
        instructions = """
INTERIM MEDICAL KNOWLEDGE NAVIGATION — GENERAL REFERENCE ONLY

The measurement-only survey is complete. Before targeted investigation, use
the supplied cards to form a ranked, provisional ECG interpretation and an
evidence-acquisition plan.

Rules:
1. N-references are general medical knowledge, never observations about this patient.
2. Do not cite N-references in the final verdict and do not copy a general criterion as if the patient met it.
3. For each provisional hypothesis, name the surveyed observation that raised it, the defining or discriminating patient evidence still needed, important confounders, and the exact ecgfeat tool/view to check next.
4. Candidate detector output remains candidate-only. Knowledge does not upgrade it.
5. If subsequent ecgfeat measurements do not establish a criterion, remove the hypothesis, move it to the differential, or abstain.
6. Do not quote or reproduce the cards in the provisional plan. Convert only the relevant relationship into a concise patient-measurement question so the full excerpts can be discarded before investigation.

Governed knowledge payload:
""".strip()
        return (
            instructions
            + "\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            audit,
        )


__all__ = ["KnowledgeNavigator", "NAVIGATION_VERSION", "NavigationReference"]

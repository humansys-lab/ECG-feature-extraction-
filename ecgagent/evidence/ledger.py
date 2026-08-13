"""Model-visible evidence provenance.

Tool handlers know every pointer they touched, but a model may receive only a
rendered or compressed subset of that result.  This module deliberately keeps
those two sets separate.  A pointer becomes citable only after its exact
``ev:/...`` token (or a backend-owned exact alias) survives in the message that
will be sent to the model.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_ALIAS_BOUNDARY = r"(?<![A-Za-z0-9_]){token}(?![A-Za-z0-9_])"
_CITATION_BOUNDARY = (
    r"(?<![A-Za-z0-9_]){token}(?![A-Za-z0-9_~/.-])"
)


def evidence_set_id(citations: Iterable[str]) -> str | None:
    """Return a stable identifier without serialising every pointer inline."""

    pointers = sorted({str(pointer).removeprefix("ev:") for pointer in citations})
    if not pointers:
        return None
    canonical = json.dumps(
        pointers,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_set_summary(citations: Iterable[str]) -> dict[str, Any]:
    """Describe a pointer set compactly for result-file audit metadata."""

    pointers = {str(pointer).removeprefix("ev:") for pointer in citations}
    return {
        "count": len(pointers),
        "evidence_set_id": evidence_set_id(pointers),
    }


def message_text(messages: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str) -> str:
    """Return a stable textual representation of provider-shaped messages."""

    if isinstance(messages, str):
        return messages
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"), default=str)


def visible_citations(
    messages: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str,
    candidates: Iterable[str],
    *,
    aliases: Mapping[str, str] | None = None,
    allow_exact: bool = True,
) -> tuple[str, ...]:
    """Return candidate pointers whose citation token is model-visible.

    ``aliases`` is an alias-to-pointer map.  It is intentionally supplied by
    the backend after compression; the ledger never guesses or accepts a
    pointer-only manifest as proof that the corresponding value row survived.
    """

    rendered = message_text(messages)
    alias_by_pointer: dict[str, list[str]] = {}
    for alias, pointer in (aliases or {}).items():
        alias_by_pointer.setdefault(str(pointer), []).append(str(alias))

    found: list[str] = []
    for raw_pointer in candidates:
        pointer = str(raw_pointer)
        exact = f"ev:{pointer}"
        # A substring check would authorize ``ev:/a`` when only
        # ``ev:/atrial_rate`` was rendered. Citations are atomic tokens.
        if allow_exact and re.search(
            _CITATION_BOUNDARY.format(token=re.escape(exact)), rendered
        ):
            found.append(pointer)
            continue
        if any(
            re.search(_ALIAS_BOUNDARY.format(token=re.escape(alias)), rendered)
            for alias in alias_by_pointer.get(pointer, ())
        ):
            found.append(pointer)
    return tuple(dict.fromkeys(found))


@dataclass
class EvidenceLedger:
    """Session provenance split by what tools touched and what models saw."""

    touched: set[str] = field(default_factory=set)
    visible: set[str] = field(default_factory=set)
    visible_by_source: dict[str, set[str]] = field(default_factory=dict)
    program_authorized: set[str] = field(default_factory=set)
    program_by_source: dict[str, set[str]] = field(default_factory=dict)

    def record_touched(self, citations: Iterable[str]) -> None:
        self.touched.update(str(pointer) for pointer in citations)

    def authorize(self, citations: Iterable[str], *, source: str) -> tuple[str, ...]:
        accepted = tuple(dict.fromkeys(str(pointer) for pointer in citations))
        self.visible.update(accepted)
        self.visible_by_source.setdefault(str(source), set()).update(accepted)
        return accepted

    def authorize_program(
        self,
        citations: Iterable[str],
        *,
        source: str,
    ) -> tuple[str, ...]:
        """Authorize touched inputs used by a deterministic program decision.

        These pointers are intentionally kept separate from ``visible``: the
        model did not need to see or calculate them.  They are nevertheless
        valid final evidence because a successful, audited tool call returned
        them and the program used them in a versioned deterministic resolver.
        """

        accepted = tuple(
            dict.fromkeys(
                str(pointer)
                for pointer in citations
                if str(pointer) in self.touched
            )
        )
        self.program_authorized.update(accepted)
        self.program_by_source.setdefault(str(source), set()).update(accepted)
        return accepted

    @property
    def authorized(self) -> frozenset[str]:
        return frozenset(self.visible | self.program_authorized)

    @property
    def hidden(self) -> frozenset[str]:
        return frozenset(self.touched - self.visible)

    def audit(self) -> dict[str, Any]:
        """Return compact provenance metadata for the primary result JSON.

        Complete pointer manifests remain available in the standalone trace.
        Repeating them here made result files large without exposing values or
        clinical relevance, so the primary audit carries counts and
        content-addressed set identifiers only.
        """

        return {
            "touched_size": len(self.touched),
            "visible_size": len(self.visible),
            "hidden_after_render_size": len(self.hidden),
            "touched_evidence_set_id": evidence_set_id(self.touched),
            "visible_evidence_set_id": evidence_set_id(self.visible),
            "hidden_evidence_set_id": evidence_set_id(self.hidden),
            "visible_by_source": {
                source: evidence_set_summary(pointers)
                for source, pointers in sorted(self.visible_by_source.items())
            },
            "program_authorized_size": len(self.program_authorized),
            "program_authorized_evidence_set_id": evidence_set_id(
                self.program_authorized
            ),
            "program_by_source": {
                source: evidence_set_summary(pointers)
                for source, pointers in sorted(self.program_by_source.items())
            },
        }


__all__ = [
    "EvidenceLedger",
    "evidence_set_id",
    "evidence_set_summary",
    "message_text",
    "visible_citations",
]

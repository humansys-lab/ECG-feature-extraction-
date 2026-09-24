"""Read-only addressing layer over one ecgfeat features payload.

The payload is 2-4 MB of nested JSON, so it can never be handed to a model
wholesale.  `EvidenceStore` turns it into something a model can query one
pointer at a time, and turns every answer into a citable
`EvidenceValue` carrying its unit and its reliability context.

The store is immutable for the lifetime of a session: a pointer is both the
address of a value and the token used to cite it, so the two must never drift.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import caveats as caveat_rules
from .diagnostic_contract import build_diagnostic_document
from .pointer import (
    Pointer,
    PointerError,
    display_precision,
    infer_unit,
    parse_pointer,
    resolve_alias,
    resolve_in,
    walk_leaves,
)

STANDARD_LEAD_ORDER = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

# Largest container `resolve` will inline, in serialized characters. Roughly
# 200 tokens: big enough for a rule's criteria block, small enough that no
# single pointer can blow the context budget.
CONTAINER_CHAR_LIMIT = 800


@dataclass(frozen=True)
class EvidenceValue:
    """One resolved measurement with everything needed to quote it safely."""

    pointer: str
    value: Any
    unit: str | None = None
    caveats: tuple[str, ...] = ()
    companions: tuple[str, ...] = ()
    source: str | None = None

    @property
    def citation(self) -> str:
        return f"ev:{self.pointer}"

    @property
    def is_null(self) -> bool:
        return self.value is None

    @property
    def reliable(self) -> bool:
        """False when any caveat was raised. Deliberately conservative."""
        return not self.caveats

    def format_value(self) -> str:
        if self.value is None:
            return "null"
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        if isinstance(self.value, float):
            text = f"{self.value:.{display_precision(self.unit)}f}"
            # Only strip inside a fraction. An unguarded rstrip("0") turns an
            # integer-formatted 50 ms into "5" — silent corruption of every
            # value ending in zero.
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text or "0"
        if isinstance(self.value, (list, tuple, dict)):
            return json.dumps(self.value, ensure_ascii=False, default=str)
        return str(self.value)

    @property
    def display_value(self) -> Any:
        """The value at the precision the model is actually shown.

        Every tool renders through `format_value`, so this - not the stored
        float - is what a model can transcribe, cite and be held to. Verdict
        layers that reach past it to the raw float end up demanding digits
        nobody was ever given: a claim reading "50 bpm" was rewritten to carry
        50.08347245409015 and then failed for asserting an uncited number.
        """
        if isinstance(self.value, float):
            try:
                rendered = self.format_value()
                return float(rendered) if "." in rendered else int(rendered)
            except ValueError:  # pragma: no cover - non-finite floats
                return self.value
        return self.value

    def render(self, *, with_pointer: bool = True) -> str:
        """One-line rendering for a prompt. Caveats are never omitted."""
        parts = [self.format_value()]
        if self.unit and not isinstance(self.value, (bool, str, list, dict, type(None))):
            parts.append(self.unit)
        text = " ".join(parts)
        if with_pointer:
            text = f"{text}  [{self.citation}]"
        if self.caveats:
            text += "\n  ! " + "\n  ! ".join(self.caveats)
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "citation": self.citation,
            "value": self.value,
            "unit": self.unit,
            "caveats": list(self.caveats),
            "companions": list(self.companions),
            "source": self.source,
        }


@dataclass(frozen=True)
class SearchHit:
    pointer: str
    field: str
    value: Any
    unit: str | None


@dataclass
class EvidenceStore:
    """Immutable, pointer-addressable view of one features payload."""

    document: dict[str, Any]
    record_id: str = "unknown"
    access_profile: str = "full"
    access_contract: dict[str, Any] = field(default_factory=dict)
    _search_index: list[tuple[str, str]] = field(default_factory=list, repr=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path, record_id: str | None = None) -> "EvidenceStore":
        payload_path = Path(path)
        with payload_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        derived = record_id or payload_path.stem.replace("_features", "")
        return cls.from_dict(document, record_id=derived)

    @classmethod
    def from_dict(cls, document: dict[str, Any], record_id: str = "unknown") -> "EvidenceStore":
        if not isinstance(document, dict):
            raise TypeError("features payload must be a JSON object")
        store = cls(document=document, record_id=record_id)
        store._build_search_index()
        return store

    @classmethod
    def from_features(cls, features: Any, record_id: str = "unknown") -> "EvidenceStore":
        """Build from an in-memory `ecgfeat.ECGFeatures` via its export contract."""
        from ecgfeat.compat.export_v0 import to_dict as features_to_dict

        return cls.from_dict(features_to_dict(features), record_id=record_id)

    def diagnostic_view(self) -> "EvidenceStore":
        """Return the versioned, physically isolated diagnosis evidence DTO."""
        if self.access_profile == "diagnostic":
            return self
        document, contract_audit = build_diagnostic_document(self.document)

        store = type(self)(
            document=document,
            record_id=self.record_id,
            access_profile="diagnostic",
            access_contract=contract_audit.to_dict(),
        )
        store._build_search_index()
        return store

    def fingerprint(self) -> str:
        """Hash the exact immutable document exposed through this Store."""

        serialized = json.dumps(
            self.document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -- resolution ---------------------------------------------------------
    def resolve(self, reference: str) -> EvidenceValue:
        """Resolve a pointer, `ev:` citation, dotted path or clinical alias."""
        alias_target = resolve_alias(reference)
        if (
            alias_target is None
            and isinstance(reference, str)
            and reference.strip() in (self.document.get("global_features") or {})
        ):
            alias_target = f"/global_features/{reference.strip()}"
        pointer = parse_pointer(alias_target or reference)
        value = resolve_in(self.document, pointer)
        self._guard_container_size(pointer, value)
        unit = infer_unit(pointer.leaf)
        notes, companions = self._caveats_for(pointer, value)
        return EvidenceValue(
            pointer=pointer.raw,
            value=value,
            unit=unit,
            caveats=tuple(notes),
            companions=tuple(companions),
            source=self._source_for(pointer),
        )

    def try_resolve(self, reference: str) -> EvidenceValue | None:
        try:
            return self.resolve(reference)
        except (PointerError, TypeError):
            return None

    def resolve_many(self, references: Iterable[str]) -> list[EvidenceValue]:
        return [self.resolve(reference) for reference in references]

    def raw(self, reference: str, default: Any = None) -> Any:
        """Raw value without caveat machinery, for internal callers only."""
        try:
            return resolve_in(self.document, parse_pointer(resolve_alias(reference) or reference))
        except PointerError:
            return default

    @staticmethod
    def _guard_container_size(pointer: Pointer, value: Any) -> None:
        """Refuse to inline a container that would swamp the model's context.

        The binding constraint is rendered size, not key count: a 12-key node
        holding per-lead sub-objects costs far more than a 30-key node of
        scalars.  Refusing by size keeps small structured values (a rule's
        `criteria`, a `territory_results` entry) directly citable while still
        blocking `/representative_leads/<lead>/params` and friends.
        """
        if not isinstance(value, (dict, list)):
            return
        rendered = len(json.dumps(value, ensure_ascii=False, default=str))
        if rendered <= CONTAINER_CHAR_LIMIT:
            return
        size = f"{len(value)} keys" if isinstance(value, dict) else f"{len(value)} items"
        raise PointerError(
            f"{pointer.raw} addresses a container of {size} ({rendered} chars), over the "
            f"{CONTAINER_CHAR_LIMIT}-char inline limit. Address a specific field, or use "
            "get_lead_table / get_beat_table / get_rule_detail to read it in table form."
        )

    def _caveats_for(self, pointer: Pointer, value: Any) -> tuple[list[str], list[str]]:
        tokens = pointer.tokens
        notes: list[str] = []
        companions: list[str] = []

        if len(tokens) == 2 and tokens[0] == "global_features":
            notes, companions = caveat_rules.global_caveats(
                tokens[1], value, self.document.get("global_features") or {}
            )
        elif (
            len(tokens) == 4
            and tokens[0] == "representative_leads"
            and tokens[2] == "params"
        ):
            lead = tokens[1]
            notes, companions = caveat_rules.lead_caveats(
                lead,
                tokens[3],
                self._lead_params(lead),
                (self.document.get("quality") or {}).get(lead),
            )
        elif (
            len(tokens) == 3
            and tokens[:2] == ("measurement_bundles", "qrs_by_lead")
            and isinstance(value, dict)
        ):
            # A compact per-lead bundle must not erase the limitations carried
            # by its component measurements.  Apply the same caveat rules used
            # by the scalar representative-lead pointers and conservatively
            # mark the whole bundle limited when any component is limited.
            lead = tokens[2]
            params = self._lead_params(lead)
            quality = (self.document.get("quality") or {}).get(lead)
            for field_name, field_value in value.items():
                field_notes, field_companions = caveat_rules.lead_caveats(
                    lead,
                    str(field_name),
                    params,
                    quality,
                )
                notes.extend(
                    f"{field_name}: {note}" for note in field_notes
                )
                companions.extend(field_companions)
            notes = list(dict.fromkeys(notes))
            companions = list(dict.fromkeys(companions))

        if value is None:
            notes.insert(0, "value is null: this measurement was not produced for this record")
        return notes, companions

    def _source_for(self, pointer: Pointer) -> str | None:
        head = pointer.tokens[0] if pointer.tokens else ""
        return {
            "global_features": "ecgfeat.global",
            "representative_leads": "ecgfeat.representative_beat",
            "beat_features": "ecgfeat.per_beat",
            "beats": "ecgfeat.per_beat",
            "groups": "ecgfeat.morphology_groups",
            "p_wave_assessments": "ecgfeat.p_wave_engine",
            "rhythm_inputs": "ecgfeat.rhythm_modality",
            "morphology_inputs": "ecgfeat.morphology_modality",
            "measurement_bundles": "ecgagent.derived_measurement_bundle",
            "quality": "ecgfeat.quality",
            "metadata": "ecgfeat.metadata",
            "clinical_interpretation": "ecgfeat.clinical_rules",
            "interpretation": "ecgfeat.interpret(reference_only)",
            "statement_engine": "ecgfeat.statement_engine(reference_only)",
        }.get(head)

    # -- structure accessors ------------------------------------------------
    def _lead_params(self, lead: str) -> dict[str, Any]:
        leads = self.document.get("representative_leads") or {}
        entry = leads.get(lead) or {}
        return entry.get("params") or {}

    @property
    def leads(self) -> list[str]:
        present = set((self.document.get("representative_leads") or {}).keys())
        ordered = [lead for lead in STANDARD_LEAD_ORDER if lead in present]
        ordered.extend(sorted(present - set(ordered)))
        return ordered

    @property
    def n_beats(self) -> int:
        return len(self.document.get("beats") or [])

    def lead_param(self, lead: str, field_name: str) -> Any:
        return self._lead_params(lead).get(field_name)

    def lead_param_fields(self) -> list[str]:
        return sorted({field for lead in self.leads for field in self._lead_params(lead)})

    def global_fields(self) -> list[str]:
        return sorted((self.document.get("global_features") or {}).keys())

    def lead_quality(self, lead: str) -> dict[str, Any]:
        return (self.document.get("quality") or {}).get(lead) or {}

    def clinical(self) -> dict[str, Any]:
        return self.document.get("clinical_interpretation") or {}

    def rule_index(self) -> dict[str, tuple[str, dict[str, Any]]]:
        """Map rule_id to (pointer_base, row), preferring resolved rows.

        `clinical_interpretation.domains` holds each rule's *raw* evaluation,
        where `confidence` may still be a free-text basis such as
        `rr_and_p_wave_evidence` and `priority` may be unset.  The resolver
        republishes matched rules into `final_statements` with normalised
        HIGH/MEDIUM/LOW confidence and a priority.  Quoting the raw row while
        citing it as the resolved one would make the citation unverifiable, so
        the resolved list wins and the pointer base follows the value.
        """
        index: dict[str, tuple[str, dict[str, Any]]] = {}
        clinical = self.clinical()

        for domain, rows in (clinical.get("domains") or {}).items():
            for position, row in enumerate(rows or []):
                rule_id = row.get("rule_id")
                if rule_id and rule_id not in index:
                    index[rule_id] = (
                        f"/clinical_interpretation/domains/{domain}/{position}",
                        row,
                    )

        for list_name in ("suppressed_statements", "borderline_statements", "final_statements"):
            for position, row in enumerate(clinical.get(list_name) or []):
                rule_id = row.get("rule_id")
                if rule_id:
                    index[rule_id] = (
                        f"/clinical_interpretation/{list_name}/{position}",
                        row,
                    )
        return index

    def rule_evaluations(self) -> list[dict[str, Any]]:
        """Every rule evaluation, deduplicated by rule_id, resolved rows preferred."""
        return [row for _, row in self.rule_index().values()]

    def rule(self, rule_id: str) -> dict[str, Any] | None:
        located = self.rule_index().get(rule_id)
        return located[1] if located else None

    def abstentions(self) -> list[dict[str, Any]]:
        return list(self.clinical().get("abstentions") or [])

    def record_caveats(self) -> list[str]:
        return caveat_rules.record_caveats(self.document)

    # -- discovery ----------------------------------------------------------
    def _build_search_index(self) -> None:
        index: list[tuple[str, str]] = []
        for field_name in (self.document.get("global_features") or {}):
            index.append((f"/global_features/{field_name}", field_name))
        first_lead = self.leads[0] if self.leads else None
        if first_lead:
            for field_name in self._lead_params(first_lead):
                index.append((f"/representative_leads/<LEAD>/params/{field_name}", field_name))
        for pointer, _ in walk_leaves(self.document.get("metadata") or {}, "/metadata", max_depth=3):
            if pointer.startswith("/metadata/clinical_interpretation"):
                continue
            index.append((pointer, pointer.rsplit("/", 1)[-1]))
        self._search_index = index

    def search(self, fragment: str, limit: int = 20) -> list[SearchHit]:
        """Find pointers whose field name matches `fragment` (case-insensitive).

        Matching is per-token and order-independent, because ecgfeat field
        names compose the same words in different orders (`p_terminal_duration_ms`
        vs a query of `terminal p`).  A plain substring test misses those, and a
        model that cannot find a field will invent one.
        """
        cleaned = str(fragment).strip().lower().replace("-", "_").replace(" ", "_")
        needles = [token for token in cleaned.split("_") if token]
        if not needles:
            return []
        joined = "_".join(needles)

        scored: list[tuple[int, str, str]] = []
        for pointer, field_name in self._search_index:
            lowered = field_name.lower()
            if lowered == joined:
                score = 0
            elif lowered.startswith(joined):
                score = 1
            elif joined in lowered:
                score = 2
            elif all(token in lowered for token in needles):
                score = 3
            else:
                continue
            scored.append((score, pointer, field_name))
        scored.sort(key=lambda item: (item[0], len(item[2]), item[1]))

        hits: list[SearchHit] = []
        for _, pointer, field_name in scored[:limit]:
            if "<LEAD>" in pointer:
                sample_lead = self.leads[0] if self.leads else None
                value = self.lead_param(sample_lead, field_name) if sample_lead else None
            else:
                value = self.raw(pointer)
            hits.append(
                SearchHit(
                    pointer=pointer,
                    field=field_name,
                    value=value,
                    unit=infer_unit(field_name),
                )
            )
        return hits

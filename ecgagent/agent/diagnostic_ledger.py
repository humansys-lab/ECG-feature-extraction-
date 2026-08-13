"""Program-owned diagnostic hypothesis ledger.

The model may propose typed mutations, but it never replaces the clinical
state.  This module owns hypothesis identity, evidence materialisation, legal
status transitions and the append-only transition history shared by every
backend.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..evidence.store import EvidenceStore, EvidenceValue


LEDGER_VERSION = "diagnostic-ledger.v3"

HYPOTHESIS_STATUSES = frozenset(
    {
        "raised",
        "provisional",
        "weak",
        "supported",
        "weakened",
        "unresolved",
        "rejected",
        "abstain",
    }
)

UNCERTAIN_STATUSES = frozenset(
    {"raised", "provisional", "weak", "weakened", "unresolved"}
)

LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "raised": frozenset({"provisional", "weak", "unresolved", "rejected", "abstain"}),
    "provisional": frozenset({"supported", "weak", "weakened", "unresolved", "rejected", "abstain"}),
    "weak": frozenset({"provisional", "supported", "weakened", "unresolved", "rejected", "abstain"}),
    "supported": frozenset({"weakened", "unresolved", "rejected", "abstain"}),
    "weakened": frozenset({"provisional", "supported", "unresolved", "rejected", "abstain"}),
    "unresolved": frozenset({"provisional", "supported", "weakened", "rejected", "abstain"}),
    "rejected": frozenset(),
    "abstain": frozenset({"unresolved", "rejected"}),
}

_HYPOTHESIS_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
_CANDIDATE_POINTER_PREFIXES = (
    "/rhythm_inputs/af_afl/",
    "/rhythm_inputs/preexcitation/",
    "/rhythm_inputs/pacing/",
)

# These belong to domain coverage, quality assessment or recommendations.
# They must never masquerade as disease/phenotype hypotheses.
NON_HYPOTHESIS_CODES = frozenset(
    {
        "normal_ecg",
        "non_diagnostic_ecg",
        "technically_limited",
        "diagnostic_coverage_limited",
        "complex_morphology_limited_interpretation",
        "minor_signal_quality_issue",
        "repeat_ecg_required",
        "uncalibrated_amplitudes",
        # This is a measurement-routing observation, not a disease or ECG
        # phenotype that may be promoted into the positive diagnosis list.
        "wide_qrs_repolarization_review",
    }
)


class LedgerPatchError(ValueError):
    """Raised when a proposed patch cannot be committed atomically."""

    def __init__(self, problems: Sequence[str]):
        self.problems = tuple(str(problem) for problem in problems if str(problem))
        super().__init__("; ".join(self.problems))


def _json_safe(value: Any) -> Any:
    """Return an audit-safe JSON value without changing scalar precision."""

    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
    return copy.deepcopy(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _content_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def _evidence_family(pointer: str) -> str:
    """Approximate detector/measurement lineage for double-count protection."""

    parts = [part for part in str(pointer).split("/") if part]
    if not parts:
        return "unknown"
    if parts[0] == "rhythm_inputs" and len(parts) > 1:
        return f"rhythm_inputs:{parts[1]}"
    if parts[0] == "representative_leads" and len(parts) > 3:
        # Lead-specific values are distinct observations, but remain part of
        # one representative-template measurement family.
        return f"representative:{parts[3].split('_', 1)[0]}"
    if parts[0] == "global_features" and len(parts) > 1:
        return f"global:{parts[1]}"
    if parts[0] == "beat_features" and len(parts) > 2:
        return f"beat:{parts[2].split('_', 1)[0]}"
    return ":".join(parts[:3])


@dataclass(frozen=True)
class PlacementContract:
    positive_codes: tuple[str, ...]
    differential_codes: tuple[str, ...]
    rejected_codes: tuple[str, ...]
    abstained_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnoses": list(self.positive_codes),
            "differential_diagnoses": list(self.differential_codes),
            "rejected_audit_only": list(self.rejected_codes),
            "abstentions": list(self.abstained_codes),
        }


class DiagnosticLedger:
    """One versioned ledger for a complete diagnosis run."""

    def __init__(
        self,
        *,
        store: EvidenceStore,
        diagnosis_codes: Iterable[str],
        domains: Iterable[str],
        tools: Iterable[str],
        provenance_lookup: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self.record_id = store.record_id
        self.input_fingerprint = store.fingerprint()
        self.allowed_diagnosis_codes = frozenset(str(code) for code in diagnosis_codes)
        self.allowed_domains = frozenset(str(domain) for domain in domains)
        self.allowed_tools = frozenset(str(tool) for tool in tools)
        self.provenance_lookup = provenance_lookup
        self.version = 0
        self.current_phase = "not_started"
        self.frozen = False
        self.domains: dict[str, dict[str, Any]] = {}
        self.hypotheses: dict[str, dict[str, Any]] = {}
        self.evidence: dict[str, dict[str, Any]] = {}
        self.targeted_checks: list[dict[str, Any]] = []
        self.detector_conflicts: list[str] = []
        self.unresolved_questions: list[str] = []
        self.phase_summaries: dict[str, str] = {}
        self.history: list[dict[str, Any]] = []

    # -- immutable evidence -------------------------------------------------
    def resolve_citation(self, reference: str) -> EvidenceValue | None:
        """Resolve a store pointer or a ledger evidence id to its value.

        The snapshot sent to the model keys `evidence` by content id and fills
        every hypothesis claim with `evidence_ids`, so an `E<digest>` token is
        frequently the only handle the model can see -- Hypothesize reads no
        tools at all and therefore holds no fresh `ev:/` pointer. Accepting the
        id the ledger itself published closes that round trip. It grants no new
        reach: an id exists only because some earlier phase already resolved
        and authorized its pointer.
        """

        token = str(reference).strip()
        row = self.evidence.get(token)
        if row is not None:
            return self.store.try_resolve(str(row.get("pointer") or ""))
        return self.store.try_resolve(token)

    def _register_evidence(
        self,
        reference: str,
        *,
        phase: str,
        authorized: frozenset[str],
    ) -> str:
        resolved = self.resolve_citation(str(reference))
        if resolved is None:
            raise LedgerPatchError([f"unresolvable evidence citation `{reference}`"])
        if resolved.pointer not in authorized:
            raise LedgerPatchError(
                [f"evidence `{resolved.citation}` was not visible to the model"]
            )
        evidence_id = _content_id("E", resolved.pointer)
        existing = self.evidence.get(evidence_id)
        if existing is not None and existing.get("pointer") != resolved.pointer:
            raise LedgerPatchError([f"evidence id collision for `{resolved.pointer}`"])
        if existing is None:
            self.evidence[evidence_id] = self._evidence_row(resolved, phase=phase)
        return evidence_id

    def _evidence_row(self, value: EvidenceValue, *, phase: str) -> dict[str, Any]:
        parts = [part for part in value.pointer.split("/") if part]
        lead_scope: list[str] = []
        beat_scope: list[str] = []
        if len(parts) > 1 and parts[0] in {"representative_leads", "quality"}:
            lead_scope.append(parts[1])
        if len(parts) > 1 and parts[0] in {
            "beat_features",
            "p_wave_assessments",
        }:
            beat_scope.append(parts[1])
        metadata = self.store.document.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source_version = next(
            (
                metadata.get(key)
                for key in (
                    "feature_extraction_version",
                    "ecgfeat_version",
                    "algorithm_version",
                )
                if metadata.get(key) is not None
            ),
            None,
        )
        provenance = (
            dict(self.provenance_lookup(value.pointer))
            if callable(self.provenance_lookup)
            else {}
        )
        row = {
            "id": _content_id("E", value.pointer),
            "pointer": value.pointer,
            "raw_value": _json_safe(value.value),
            "display_value": _json_safe(value.display_value),
            "unit": value.unit,
            "caveats": list(value.caveats),
            "reliability": "usable" if value.reliable else "limited",
            "quality_status": (
                "unavailable"
                if value.value is None
                else "reliable"
                if value.reliable
                else "limited"
            ),
            "missing_reason": (
                (str(value.caveats[0]) if value.caveats else "source_value_null")
                if value.value is None
                else None
            ),
            "source": value.source,
            "source_algorithm_version": source_version,
            "evidence_contract_version": self.store.access_contract.get("version"),
            "scope": {
                "lead_scope": lead_scope,
                "beat_scope": beat_scope,
                "time_scope": None,
            },
            "evidence_family": _evidence_family(value.pointer),
            "input_fingerprint": self.input_fingerprint,
            "first_seen_phase": str(phase),
        }
        row.update(provenance)
        return row

    def _claim(
        self,
        row: Mapping[str, Any],
        *,
        phase: str,
        authorized: frozenset[str],
        require_new: bool,
        new_citations: frozenset[str],
    ) -> dict[str, Any]:
        citations = [str(value) for value in (row.get("citations") or [])]
        pointers: list[str] = []
        evidence_ids: list[str] = []
        for citation in citations:
            resolved = self.resolve_citation(citation)
            if resolved is None:
                raise LedgerPatchError([f"unresolvable evidence citation `{citation}`"])
            pointers.append(resolved.pointer)
            evidence_ids.append(
                self._register_evidence(
                    resolved.pointer,
                    phase=phase,
                    authorized=authorized,
                )
            )
        if require_new and not (set(pointers) & set(new_citations)):
            raise LedgerPatchError(
                [f"evidence update cites no evidence newly read in `{phase}`"]
            )
        return {
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "added_in_phase": str(phase),
        }

    # -- event-sourced reducer ---------------------------------------------
    def apply_phase_response(
        self,
        phase: str,
        response: Mapping[str, Any],
        *,
        authorized_citations: Iterable[str],
        new_phase_citations: Iterable[str],
    ) -> dict[str, Any]:
        """Validate and commit a phase patch as one atomic transaction."""

        if self.frozen:
            raise LedgerPatchError(["diagnostic ledger is frozen"])
        phase_key = str(phase)
        authorized = frozenset(
            resolved.pointer
            for value in authorized_citations
            if (resolved := self.store.try_resolve(str(value))) is not None
        )
        new_citations = frozenset(
            resolved.pointer
            for value in new_phase_citations
            if (resolved := self.store.try_resolve(str(value))) is not None
        )
        try:
            base_version = int(response.get("base_version"))
        except (TypeError, ValueError):
            raise LedgerPatchError(["base_version must be an integer"])
        if base_version != self.version:
            raise LedgerPatchError(
                [f"stale patch base_version={base_version}; current ledger version={self.version}"]
            )

        mutable_names = (
            "domains",
            "hypotheses",
            "evidence",
            "targeted_checks",
            "detector_conflicts",
            "unresolved_questions",
            "phase_summaries",
            "history",
        )
        backup = {name: copy.deepcopy(getattr(self, name)) for name in mutable_names}
        backup_version = self.version
        backup_phase = self.current_phase
        backup_frozen = self.frozen
        try:
            if phase_key == "survey":
                self._apply_domains(response.get("domains"), authorized=authorized)
            self._apply_domain_updates(
                phase_key,
                response.get("domain_updates"),
                authorized=authorized,
                new_citations=new_citations,
            )
            self._apply_creates(
                phase_key,
                response.get("create_hypotheses"),
                authorized=authorized,
                new_citations=new_citations,
            )
            self._apply_evidence_updates(
                phase_key,
                response.get("evidence_updates"),
                authorized=authorized,
                new_citations=new_citations,
            )
            self._apply_plan_updates(phase_key, response.get("plan_updates"))
            self._apply_refuting_tests(
                phase_key,
                response.get("refuting_tests"),
                authorized=authorized,
                new_citations=new_citations,
            )
            self._apply_transitions(
                phase_key,
                response.get("transitions"),
                authorized=authorized,
                new_citations=new_citations,
            )
            self._apply_phase_context(phase_key, response)
            if phase_key == "challenge":
                self._close_challenge_without_refutation()
                self.frozen = True
            self.current_phase = phase_key
            self._append_event(
                phase=phase_key,
                op="commit_phase",
                target=phase_key,
                before={"base_version": base_version},
                after={"frozen": self.frozen},
                evidence_ids=(),
                reason=self.phase_summaries.get(
                    phase_key,
                    "program-owned phase patch committed",
                ),
            )
        except Exception:
            for name, value in backup.items():
                setattr(self, name, value)
            self.version = backup_version
            self.current_phase = backup_phase
            self.frozen = backup_frozen
            raise
        return self.phase_state()

    def validate_phase_response(
        self,
        phase: str,
        response: Mapping[str, Any],
        *,
        authorized_citations: Iterable[str],
        new_phase_citations: Iterable[str],
    ) -> list[str]:
        """Dry-run one atomic patch without changing ledger state."""

        mutable_names = (
            "domains",
            "hypotheses",
            "evidence",
            "targeted_checks",
            "detector_conflicts",
            "unresolved_questions",
            "phase_summaries",
            "history",
        )
        backup = {name: copy.deepcopy(getattr(self, name)) for name in mutable_names}
        backup_version = self.version
        backup_phase = self.current_phase
        backup_frozen = self.frozen
        problems = self._citation_preflight(
            str(phase),
            response,
            authorized_citations=authorized_citations,
            new_phase_citations=new_phase_citations,
        )
        if problems:
            return problems
        try:
            self.apply_phase_response(
                phase,
                response,
                authorized_citations=authorized_citations,
                new_phase_citations=new_phase_citations,
            )
        except LedgerPatchError as exc:
            problems.extend(exc.problems)
        except Exception as exc:  # defensive: surface reducer bugs as hard guards
            problems.append(f"ledger reducer rejected patch: {type(exc).__name__}: {exc}")
        finally:
            for name, value in backup.items():
                setattr(self, name, value)
            self.version = backup_version
            self.current_phase = backup_phase
            self.frozen = backup_frozen
        return problems

    def _citation_preflight(
        self,
        phase: str,
        response: Mapping[str, Any],
        *,
        authorized_citations: Iterable[str],
        new_phase_citations: Iterable[str],
    ) -> list[str]:
        """Report every citation defect before attempting an atomic commit."""

        authorized = {
            resolved.pointer
            for token in authorized_citations
            if (resolved := self.store.try_resolve(str(token))) is not None
        }
        new_citations = {
            resolved.pointer
            for token in new_phase_citations
            if (resolved := self.store.try_resolve(str(token))) is not None
        }
        rows: list[tuple[str, Mapping[str, Any], bool]] = []
        if phase == "survey" and isinstance(response.get("domains"), Mapping):
            rows.extend(
                (f"domains.{domain}", row, False)
                for domain, row in response["domains"].items()
                if isinstance(row, Mapping)
            )
        # A new identity is itself the new state, so the copy-forward risk the
        # "newly read" rule exists to block does not apply to the claims that
        # motivate it. The decisive counterevidence to a late candidate is
        # routinely an interval already on the record -- QRS duration against a
        # fascicular block, say -- and demanding a same-phase re-read of it
        # rejects the most relevant citation the model could pick.
        for index, hypothesis in enumerate(response.get("create_hypotheses") or []):
            if not isinstance(hypothesis, Mapping):
                continue
            for field in ("supporting_observations", "counterevidence"):
                rows.extend(
                    (
                        f"create_hypotheses[{index}].{field}[{claim_index}]",
                        claim,
                        False,
                    )
                    for claim_index, claim in enumerate(hypothesis.get(field) or [])
                    if isinstance(claim, Mapping)
                )
        for field in ("evidence_updates", "domain_updates"):
            rows.extend(
                (f"{field}[{index}]", row, True)
                for index, row in enumerate(response.get(field) or [])
                if isinstance(row, Mapping)
            )
        rows.extend(
            (
                f"transitions[{index}]",
                row,
                phase in {"investigate", "challenge"},
            )
            for index, row in enumerate(response.get("transitions") or [])
            if isinstance(row, Mapping)
        )
        rows.extend(
            (f"refuting_tests[{index}]", row, True)
            for index, row in enumerate(response.get("refuting_tests") or [])
            if isinstance(row, Mapping)
        )

        problems: list[str] = []
        for path, row, require_new in rows:
            pointers: list[str] = []
            for token in row.get("citations") or []:
                evidence = self.resolve_citation(str(token))
                if evidence is None:
                    # Naming both accepted forms matters: the snapshot shows a
                    # pointer and a content id side by side, and blended
                    # spellings like `Ev0ff7...` -- the `ev:` prefix fused onto
                    # an `E<digest>` id -- are the single most common way a
                    # citation fails. Never guess which one was meant; a
                    # near-miss id would bind the claim to another measurement.
                    problems.append(
                        f"{path} has unresolvable citation `{token}`; cite "
                        "either the exact `ev:/...` pointer or the exact "
                        "`E<digest>` id shown in the ledger `evidence` map, "
                        "copied character for character"
                    )
                    continue
                pointers.append(evidence.pointer)
                if evidence.pointer not in authorized:
                    problems.append(
                        f"{path} cites `{evidence.citation}`, which was not present "
                        "in a model-visible atomic evidence view"
                    )
            if require_new and pointers and not (set(pointers) & new_citations):
                problems.append(
                    f"{path} cites no atomic evidence newly read in `{phase}`"
                )
        return list(dict.fromkeys(problems))

    def _apply_domains(
        self,
        value: Any,
        *,
        authorized: frozenset[str],
    ) -> None:
        if not isinstance(value, Mapping):
            raise LedgerPatchError(["survey must initialize the domain ledger"])
        problems: list[str] = []
        if set(value) != set(self.allowed_domains):
            missing = sorted(self.allowed_domains - set(value))
            extra = sorted(set(value) - self.allowed_domains)
            if missing:
                problems.append("missing survey domains: " + ", ".join(missing))
            if extra:
                problems.append("unknown survey domains: " + ", ".join(extra))
        if problems:
            raise LedgerPatchError(problems)
        for domain in sorted(self.allowed_domains):
            row = value.get(domain)
            if not isinstance(row, Mapping):
                raise LedgerPatchError([f"domain `{domain}` is not an object"])
            evidence_ids = [
                self._register_evidence(
                    str(citation),
                    phase="survey",
                    authorized=authorized,
                )
                for citation in (row.get("citations") or [])
            ]
            self.domains[domain] = {
                "status": str(row.get("status") or "not_assessed"),
                "finding": str(row.get("finding") or "not_assessed"),
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
                "limitations": [
                    str(item) for item in (row.get("limitations") or []) if str(item).strip()
                ],
            }
        self._append_event(
            phase="survey",
            op="initialize_domains",
            target="domains",
            before=None,
            after={"domain_count": len(self.domains)},
            evidence_ids=(
                evidence_id
                for row in self.domains.values()
                for evidence_id in row.get("evidence_ids") or []
            ),
            reason="systematic survey domain ledger initialized",
        )

    def _apply_domain_updates(
        self,
        phase: str,
        value: Any,
        *,
        authorized: frozenset[str],
        new_citations: frozenset[str],
    ) -> None:
        rows = value if isinstance(value, list) else []
        if rows and phase not in {"investigate", "challenge"}:
            raise LedgerPatchError([f"phase `{phase}` cannot update domain findings"])
        for row in rows:
            if not isinstance(row, Mapping):
                raise LedgerPatchError(["domain_updates contains a non-object"])
            domain = str(row.get("domain") or "")
            if domain not in self.allowed_domains or domain not in self.domains:
                raise LedgerPatchError([f"unknown domain `{domain}`"])
            citations = [str(value) for value in (row.get("citations") or [])]
            evidence_ids: list[str] = []
            pointers: list[str] = []
            for citation in citations:
                resolved = self.store.try_resolve(citation)
                if resolved is None:
                    raise LedgerPatchError([f"unresolvable evidence citation `{citation}`"])
                pointers.append(resolved.pointer)
                evidence_ids.append(
                    self._register_evidence(
                        resolved.pointer,
                        phase=phase,
                        authorized=authorized,
                    )
                )
            if not (set(pointers) & set(new_citations)):
                raise LedgerPatchError(
                    [f"domain update `{domain}` cites no evidence newly read in `{phase}`"]
                )
            before = copy.deepcopy(self.domains[domain])
            self.domains[domain] = {
                "status": str(row.get("status") or "not_assessed"),
                "finding": str(row.get("finding") or "indeterminate"),
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
                "limitations": [
                    str(item)
                    for item in (row.get("limitations") or [])
                    if str(item).strip()
                ],
            }
            self._append_event(
                phase=phase,
                op="update_domain_finding",
                target=domain,
                before=before,
                after=copy.deepcopy(self.domains[domain]),
                evidence_ids=evidence_ids,
                reason="domain assessment updated from program-resolved evidence",
            )

    def _apply_creates(
        self,
        phase: str,
        value: Any,
        *,
        authorized: frozenset[str],
        new_citations: frozenset[str],
    ) -> None:
        rows = value if isinstance(value, list) else []
        for row in rows:
            if not isinstance(row, Mapping):
                raise LedgerPatchError(["create_hypotheses contains a non-object"])
            hypothesis_id = str(row.get("hypothesis_id") or "")
            if not _HYPOTHESIS_ID_RE.fullmatch(hypothesis_id):
                raise LedgerPatchError([f"invalid hypothesis_id `{hypothesis_id}`"])
            if hypothesis_id in self.hypotheses:
                raise LedgerPatchError([f"hypothesis `{hypothesis_id}` already exists"])
            code = str(row.get("diagnosis_code") or "")
            if code not in self.allowed_diagnosis_codes:
                raise LedgerPatchError([f"unregistered diagnosis_code `{code}`"])
            if code in NON_HYPOTHESIS_CODES:
                raise LedgerPatchError(
                    [f"`{code}` is a quality/recommendation code, not a diagnostic hypothesis"]
                )
            if any(
                hypothesis.get("diagnosis_code") == code
                and hypothesis.get("status") not in {"rejected", "abstain"}
                for hypothesis in self.hypotheses.values()
            ):
                raise LedgerPatchError([f"active diagnosis_code `{code}` already has a hypothesis"])
            status = str(row.get("status") or "raised")
            # Identity creation and evidentiary promotion are deliberately
            # separate events.  Enforce this in the reducer itself rather
            # than relying on one backend's guided-JSON schema.
            allowed_create_statuses = {"raised"}
            if status not in allowed_create_statuses:
                raise LedgerPatchError(
                    [f"new hypothesis `{hypothesis_id}` cannot start as `{status}` in `{phase}`"]
                )
            domains = list(dict.fromkeys(str(item) for item in (row.get("domains") or [])))
            if not domains or any(domain not in self.allowed_domains for domain in domains):
                raise LedgerPatchError([f"hypothesis `{hypothesis_id}` has invalid domains"])
            next_tools = list(dict.fromkeys(str(item) for item in (row.get("next_tools") or [])))
            if any(tool not in self.allowed_tools for tool in next_tools):
                raise LedgerPatchError([f"hypothesis `{hypothesis_id}` requests an unknown tool"])
            # See validate_phase_response: the claims motivating a brand-new
            # identity are exempt from the "newly read" rule, which exists to
            # stop copy-forward mutation of state that already exists.
            supporting = [
                self._claim(
                    item,
                    phase=phase,
                    authorized=authorized,
                    require_new=False,
                    new_citations=new_citations,
                )
                for item in (row.get("supporting_observations") or [])
                if isinstance(item, Mapping)
            ]
            opposing = [
                self._claim(
                    item,
                    phase=phase,
                    authorized=authorized,
                    require_new=False,
                    new_citations=new_citations,
                )
                for item in (row.get("counterevidence") or [])
                if isinstance(item, Mapping)
            ]
            if not any(
                claim.get("evidence_ids")
                for claim in (*supporting, *opposing)
            ):
                raise LedgerPatchError(
                    [
                        f"new hypothesis `{hypothesis_id}` cites no atomic evidence "
                        f"available in `{phase}`"
                    ]
                )
            hypothesis = {
                "id": hypothesis_id,
                "diagnosis_code": code,
                "phenotype": str(row.get("phenotype") or "").strip(),
                "kind": str(row.get("kind") or "diagnosis"),
                "domains": domains,
                "status": status,
                "support": supporting,
                "counterevidence": opposing,
                "required_evidence": [
                    str(item) for item in (row.get("required_evidence") or []) if str(item).strip()
                ],
                "next_tools": next_tools,
                "alternative_explanations": [
                    str(item)
                    for item in (row.get("alternative_explanations") or [])
                    if str(item).strip()
                ],
                "alternative_group": str(row.get("alternative_group") or "").strip() or None,
                "refuting_test": None,
                "challenge_passed": False,
                "created_phase": phase,
                "last_updated_phase": phase,
            }
            self.hypotheses[hypothesis_id] = hypothesis
            self._append_event(
                phase=phase,
                op="create_hypothesis",
                target=hypothesis_id,
                before=None,
                after={
                    "diagnosis_code": code,
                    "phenotype": hypothesis["phenotype"],
                    "status": status,
                },
                evidence_ids=(
                    evidence_id
                    for claim in (*supporting, *opposing)
                    for evidence_id in claim.get("evidence_ids") or []
                ),
                reason="new diagnostic candidate",
            )

    def _require_hypothesis(self, hypothesis_id: str) -> dict[str, Any]:
        hypothesis = self.hypotheses.get(str(hypothesis_id))
        if hypothesis is None:
            raise LedgerPatchError([f"unknown hypothesis_id `{hypothesis_id}`"])
        return hypothesis

    def _apply_evidence_updates(
        self,
        phase: str,
        value: Any,
        *,
        authorized: frozenset[str],
        new_citations: frozenset[str],
    ) -> None:
        rows = value if isinstance(value, list) else []
        if rows and phase not in {"investigate", "challenge"}:
            raise LedgerPatchError([f"phase `{phase}` cannot append patient evidence"])
        for row in rows:
            if not isinstance(row, Mapping):
                raise LedgerPatchError(["evidence_updates contains a non-object"])
            hypothesis_id = str(row.get("hypothesis_id") or "")
            hypothesis = self._require_hypothesis(hypothesis_id)
            evidence_type = str(row.get("evidence_type") or "")
            if evidence_type not in {"support", "counterevidence"}:
                raise LedgerPatchError([f"invalid evidence_type `{evidence_type}`"])
            claim = self._claim(
                row,
                phase=phase,
                authorized=authorized,
                require_new=True,
                new_citations=new_citations,
            )
            target = hypothesis[evidence_type]
            signature = tuple(claim["evidence_ids"])
            if any(
                tuple(item.get("evidence_ids") or []) == signature
                for item in target
            ):
                continue
            target.append(claim)
            hypothesis["last_updated_phase"] = phase
            self._append_event(
                phase=phase,
                op=f"add_{evidence_type}",
                target=hypothesis_id,
                before=None,
                after={"evidence_ids": list(claim["evidence_ids"])},
                evidence_ids=claim["evidence_ids"],
                reason="program-resolved patient evidence linked to hypothesis",
            )

    def _apply_plan_updates(self, phase: str, value: Any) -> None:
        rows = value if isinstance(value, list) else []
        for row in rows:
            if not isinstance(row, Mapping):
                raise LedgerPatchError(["plan_updates contains a non-object"])
            hypothesis_id = str(row.get("hypothesis_id") or "")
            hypothesis = self._require_hypothesis(hypothesis_id)
            next_tools = list(dict.fromkeys(str(item) for item in (row.get("next_tools") or [])))
            if any(tool not in self.allowed_tools for tool in next_tools):
                raise LedgerPatchError([f"plan for `{hypothesis_id}` contains an unknown tool"])
            before = {
                "required_evidence": list(hypothesis.get("required_evidence") or []),
                "next_tools": list(hypothesis.get("next_tools") or []),
                "alternative_explanations": list(hypothesis.get("alternative_explanations") or []),
                "alternative_group": hypothesis.get("alternative_group"),
            }
            hypothesis.update(
                {
                    "required_evidence": [
                        str(item)
                        for item in (row.get("required_evidence") or [])
                        if str(item).strip()
                    ],
                    "next_tools": next_tools,
                    "alternative_explanations": [
                        str(item)
                        for item in (row.get("alternative_explanations") or [])
                        if str(item).strip()
                    ],
                    "alternative_group": str(row.get("alternative_group") or "").strip() or None,
                    "last_updated_phase": phase,
                }
            )
            self._append_event(
                phase=phase,
                op="set_hypothesis_plan",
                target=hypothesis_id,
                before=before,
                after={key: hypothesis.get(key) for key in before},
                evidence_ids=(),
                reason="diagnostic evidence plan updated",
            )

    def _apply_refuting_tests(
        self,
        phase: str,
        value: Any,
        *,
        authorized: frozenset[str],
        new_citations: frozenset[str],
    ) -> None:
        rows = value if isinstance(value, list) else []
        if rows and phase != "challenge":
            raise LedgerPatchError(["refuting_tests are accepted only in Challenge"])
        for row in rows:
            if not isinstance(row, Mapping):
                raise LedgerPatchError(["refuting_tests contains a non-object"])
            hypothesis_id = str(row.get("hypothesis_id") or "")
            hypothesis = self._require_hypothesis(hypothesis_id)
            measured = self._claim(
                {"citations": row.get("citations") or []},
                phase=phase,
                authorized=authorized,
                require_new=True,
                new_citations=new_citations,
            )
            test_outcome = str(row.get("test_outcome") or "inconclusive")
            evidence_direction = str(
                row.get("evidence_direction") or "not_discriminative"
            )
            if test_outcome not in {"not_refuted", "refuted", "inconclusive"}:
                raise LedgerPatchError(
                    [f"invalid falsification-test outcome `{test_outcome}`"]
                )
            if evidence_direction not in {
                "supports",
                "contradicts",
                "not_discriminative",
                "unavailable",
            }:
                raise LedgerPatchError(
                    [f"invalid falsification evidence direction `{evidence_direction}`"]
                )
            if test_outcome == "refuted" and evidence_direction != "contradicts":
                raise LedgerPatchError(
                    ["a refuted falsification test requires contradicting evidence"]
                )
            if evidence_direction == "unavailable" and test_outcome != "inconclusive":
                raise LedgerPatchError(
                    ["unavailable falsification evidence must be inconclusive"]
                )
            refuting_test = {
                "would_refute": str(row.get("would_refute") or "").strip(),
                "evidence_ids": measured["evidence_ids"],
                "test_outcome": test_outcome,
                "evidence_direction": evidence_direction,
                "phase": phase,
            }
            hypothesis["refuting_test"] = refuting_test
            hypothesis["challenge_passed"] = (
                test_outcome == "not_refuted"
                and evidence_direction == "supports"
            )
            hypothesis["last_updated_phase"] = phase
            self._append_event(
                phase=phase,
                op="record_refuting_test",
                target=hypothesis_id,
                before=None,
                after={
                    "test_outcome": test_outcome,
                    "evidence_direction": evidence_direction,
                    "would_refute": refuting_test["would_refute"],
                },
                evidence_ids=measured["evidence_ids"],
                reason="program-resolved Challenge evidence evaluated against falsification condition",
            )

    def _apply_transitions(
        self,
        phase: str,
        value: Any,
        *,
        authorized: frozenset[str],
        new_citations: frozenset[str],
    ) -> None:
        rows = value if isinstance(value, list) else []
        for row in rows:
            if not isinstance(row, Mapping):
                raise LedgerPatchError(["transitions contains a non-object"])
            hypothesis_id = str(row.get("hypothesis_id") or "")
            hypothesis = self._require_hypothesis(hypothesis_id)
            if hypothesis.get("created_phase") == phase:
                raise LedgerPatchError(
                    [
                        f"new hypothesis `{hypothesis_id}` cannot transition in its "
                        "creation phase; retain `raised` until a later evidence phase"
                    ]
                )
            before_status = str(row.get("from_status") or "")
            after_status = str(row.get("to_status") or "")
            current_status = str(hypothesis.get("status") or "")
            if before_status != current_status:
                raise LedgerPatchError(
                    [f"stale transition for `{hypothesis_id}`: expected `{current_status}`, got `{before_status}`"]
                )
            if after_status not in LEGAL_TRANSITIONS.get(current_status, frozenset()):
                raise LedgerPatchError(
                    [f"illegal transition `{current_status}` -> `{after_status}` for `{hypothesis_id}`"]
                )
            if phase == "hypothesize" and after_status == "supported":
                raise LedgerPatchError(["Hypothesize cannot produce a supported diagnosis"])
            claim = self._claim(
                {"citations": row.get("citations") or []},
                phase=phase,
                authorized=authorized,
                require_new=phase in {"investigate", "challenge"},
                new_citations=new_citations,
            )
            if after_status == "supported":
                # Challenge support with no passing falsification test is not
                # rejected here: `_close_challenge_without_refutation` runs at
                # the end of this same commit and turns it into `unresolved`,
                # so accepting the transition reaches the identical state and
                # records why. Rejecting it instead destroyed every other
                # mutation in an otherwise valid patch.
                self._require_non_candidate_support(hypothesis, claim)
                alternative_group = hypothesis.get("alternative_group")
                if alternative_group and any(
                    other_id != hypothesis_id
                    and other.get("alternative_group") == alternative_group
                    and other.get("status") == "supported"
                    for other_id, other in self.hypotheses.items()
                ):
                    raise LedgerPatchError(
                        [
                            f"mutually exclusive alternative group `{alternative_group}` "
                            "already contains a supported hypothesis"
                        ]
                    )
            hypothesis["status"] = after_status
            hypothesis["last_updated_phase"] = phase
            self._append_event(
                phase=phase,
                op="transition",
                target=hypothesis_id,
                before={"status": current_status},
                after={"status": after_status},
                evidence_ids=claim["evidence_ids"],
                reason=(
                    f"evidence-backed status transition `{current_status}` -> "
                    f"`{after_status}`"
                ),
            )

    def _require_non_candidate_support(
        self,
        hypothesis: Mapping[str, Any],
        transition_claim: Mapping[str, Any],
    ) -> None:
        pointers = [
            str(self.evidence.get(evidence_id, {}).get("pointer") or "")
            for claim in hypothesis.get("support") or []
            for evidence_id in claim.get("evidence_ids") or []
        ]
        pointers.extend(
            str(self.evidence.get(evidence_id, {}).get("pointer") or "")
            for evidence_id in transition_claim.get("evidence_ids") or []
        )
        if pointers and all(
            any(pointer.startswith(prefix) for prefix in _CANDIDATE_POINTER_PREFIXES)
            for pointer in pointers
        ):
            raise LedgerPatchError(
                ["candidate-detector fields from one chain cannot independently establish a supported mechanism"]
            )

    def _apply_phase_context(self, phase: str, response: Mapping[str, Any]) -> None:
        # Phase summaries are operational metadata, not model-authored
        # clinical prose. Patient values live exclusively in evidence atoms.
        self.phase_summaries[phase] = (
            f"{phase} committed: hypotheses={len(self.hypotheses)}, "
            f"domain_updates={len(response.get('domain_updates') or [])}, "
            f"evidence_updates={len(response.get('evidence_updates') or [])}, "
            f"transitions={len(response.get('transitions') or [])}"
        )
        self.detector_conflicts = [
            str(item) for item in (response.get("detector_conflicts") or []) if str(item).strip()
        ]
        self.unresolved_questions = [
            str(item) for item in (response.get("unresolved") or []) if str(item).strip()
        ]
        checks: list[dict[str, Any]] = []
        for row in response.get("targeted_checks") or []:
            if not isinstance(row, Mapping):
                continue
            hypothesis_id = str(row.get("hypothesis_id") or "")
            self._require_hypothesis(hypothesis_id)
            domain = str(row.get("domain") or "")
            tool = str(row.get("tool") or "")
            if domain not in self.allowed_domains or tool not in self.allowed_tools:
                raise LedgerPatchError([f"invalid targeted check for `{hypothesis_id}`"])
            checks.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "domain": domain,
                    "tool": tool,
                    "question": str(row.get("question") or "").strip(),
                }
            )
        if checks or phase == "hypothesize":
            self.targeted_checks = checks

    def _close_challenge_without_refutation(self) -> None:
        """Fail closed: unchallenged positives become unresolved, never final."""

        for hypothesis_id, hypothesis in self.hypotheses.items():
            if hypothesis.get("status") != "supported" or hypothesis.get("challenge_passed"):
                continue
            before = {"status": "supported", "challenge_passed": False}
            hypothesis["status"] = "unresolved"
            required = list(hypothesis.get("required_evidence") or [])
            reminder = "A valid falsification test citing new Challenge evidence is required"
            if reminder not in required:
                required.append(reminder)
            hypothesis["required_evidence"] = required[:4]
            hypothesis["last_updated_phase"] = "challenge"
            self._append_event(
                phase="challenge",
                op="automatic_safety_downgrade",
                target=hypothesis_id,
                before=before,
                after={"status": "unresolved", "challenge_passed": False},
                evidence_ids=(),
                reason=(
                    "supported hypothesis had no discriminative Challenge "
                    "falsification test"
                ),
            )

    def _append_event(
        self,
        *,
        phase: str,
        op: str,
        target: str,
        before: Any,
        after: Any,
        evidence_ids: Iterable[str],
        reason: str,
    ) -> None:
        base_version = self.version
        self.version += 1
        event = {
            "event_id": f"L{self.version:04d}",
            "base_version": base_version,
            "version": self.version,
            "phase": str(phase),
            "op": str(op),
            "target": str(target),
            "before": _json_safe(before),
            "after": _json_safe(after),
            "new_evidence_ids": list(dict.fromkeys(str(item) for item in evidence_ids)),
            "reason": str(reason or "").strip(),
        }
        self.history.append(event)

    # -- snapshots and final placement -------------------------------------
    def placement_contract(self) -> PlacementContract:
        positive: list[str] = []
        differential: list[str] = []
        rejected: list[str] = []
        abstained: list[str] = []
        for hypothesis in self.hypotheses.values():
            code = str(hypothesis.get("diagnosis_code") or "")
            status = str(hypothesis.get("status") or "")
            if status == "supported" and hypothesis.get("challenge_passed"):
                positive.append(code)
            elif status in UNCERTAIN_STATUSES:
                differential.append(code)
            elif status == "rejected":
                rejected.append(code)
            elif status == "abstain":
                abstained.append(code)
        if (
            not positive
            and not differential
            and not abstained
            and self.domains
            and all(
                row.get("status") == "assessed" and row.get("finding") == "normal"
                for row in self.domains.values()
            )
        ):
            # ``normal_ecg`` is never a model hypothesis. It is synthesized
            # only when the complete program-owned coverage ledger is normal.
            positive.append("normal_ecg")
        return PlacementContract(
            positive_codes=tuple(dict.fromkeys(positive)),
            differential_codes=tuple(dict.fromkeys(differential)),
            rejected_codes=tuple(dict.fromkeys(rejected)),
            abstained_codes=tuple(dict.fromkeys(abstained)),
        )

    def _active_evidence_ids(self) -> set[str]:
        active = {
            str(evidence_id)
            for row in self.domains.values()
            for evidence_id in (row.get("evidence_ids") or [])
        }
        for hypothesis in self.hypotheses.values():
            for field in ("support", "counterevidence"):
                active.update(
                    str(evidence_id)
                    for claim in (hypothesis.get(field) or [])
                    for evidence_id in (claim.get("evidence_ids") or [])
                )
            test = hypothesis.get("refuting_test")
            if isinstance(test, Mapping):
                active.update(
                    str(evidence_id) for evidence_id in (test.get("evidence_ids") or [])
                )
        return active

    def _snapshot_payload(self, *, include_raw_values: bool, include_history: bool) -> dict[str, Any]:
        evidence: dict[str, dict[str, Any]] = {}
        active_evidence = self._active_evidence_ids()
        for evidence_id, row in self.evidence.items():
            if not include_history and evidence_id not in active_evidence:
                continue
            public = dict(row)
            if not include_raw_values:
                public.pop("raw_value", None)
                public.pop("input_fingerprint", None)
            evidence[evidence_id] = public
        payload: dict[str, Any] = {
            "ledger_schema": LEDGER_VERSION,
            "record_id": self.record_id,
            "version": self.version,
            "current_phase": self.current_phase,
            "frozen": self.frozen,
            "snapshot_hash": "",
            "domains": copy.deepcopy(self.domains),
            "hypotheses": copy.deepcopy(list(self.hypotheses.values())),
            "evidence": evidence,
            "targeted_checks": copy.deepcopy(self.targeted_checks),
            "detector_conflicts": list(self.detector_conflicts),
            "unresolved": list(self.unresolved_questions),
            "phase_summary": self.phase_summaries.get(self.current_phase, ""),
            "allowed_final_placement": self.placement_contract().to_dict(),
            "last_event_id": self.history[-1]["event_id"] if self.history else None,
        }
        if include_history:
            payload["input_fingerprint"] = self.input_fingerprint
            payload["history"] = copy.deepcopy(self.history)
            payload["phase_summaries"] = dict(self.phase_summaries)
        digest_payload = dict(payload)
        digest_payload.pop("snapshot_hash", None)
        payload["snapshot_hash"] = "sha256:" + hashlib.sha256(
            _canonical_json(digest_payload).encode("utf-8")
        ).hexdigest()
        return payload

    def phase_state(self) -> dict[str, Any]:
        """Compact materialized snapshot sent to subsequent model phases."""

        return self._snapshot_payload(include_raw_values=False, include_history=False)

    def audit_state(self) -> dict[str, Any]:
        """Complete snapshot and immutable transition log for result audit."""

        return self._snapshot_payload(include_raw_values=True, include_history=True)

    def synthesis_instruction(self) -> str:
        contract = self.placement_contract().to_dict()
        return (
            "DETERMINISTIC VERDICT SKELETON. The program-owned diagnostic ledger is "
            "now frozen. Render clinical wording and atomic evidence items, but do "
            "not add, remove, promote, downgrade or rename a diagnosis code. The "
            "following placement is mandatory:\n"
            + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
        )


__all__ = [
    "DiagnosticLedger",
    "HYPOTHESIS_STATUSES",
    "LEGAL_TRANSITIONS",
    "LEDGER_VERSION",
    "LedgerPatchError",
    "PlacementContract",
    "UNCERTAIN_STATUSES",
]

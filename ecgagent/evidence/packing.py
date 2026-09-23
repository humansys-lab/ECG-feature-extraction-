"""Incremental JSON budgeting with indivisible required evidence bundles."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


EVIDENCE_PACKING_POLICY_VERSION = "ecgagent.evidence-packing.v2"
DEFAULT_MAX_RETAINED_EVIDENCE_CHARS = 8000
DEFAULT_MAX_REQUIRED_EVIDENCE_CHARS = 24000
DEFAULT_REQUIRED_EVIDENCE_HEADROOM_CHARS = 1024


@dataclass(frozen=True)
class EvidencePackingPolicy:
    """Shared JSON-packet budget, excluding provider message wrappers.

    Required atoms may grow the target up to the required-content ceiling.
    None disables growth; the retained target always remains a lower bound.
    Optional content alone never triggers growth. Budgets count Unicode
    characters in compact JSON, not bytes or model tokens.
    """

    max_retained_evidence_chars: int = DEFAULT_MAX_RETAINED_EVIDENCE_CHARS
    max_required_evidence_chars: int | None = DEFAULT_MAX_REQUIRED_EVIDENCE_CHARS
    required_evidence_headroom_chars: int = DEFAULT_REQUIRED_EVIDENCE_HEADROOM_CHARS

    def __post_init__(self) -> None:
        for name in ("max_retained_evidence_chars", "max_required_evidence_chars",
                     "required_evidence_headroom_chars"):
            value = getattr(self, name)
            if name == "max_required_evidence_chars" and value is None:
                continue
            minimum = 0 if name == "required_evidence_headroom_chars" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                kind = "non-negative" if minimum == 0 else "positive"
                suffix = " or None" if name == "max_required_evidence_chars" else ""
                raise ValueError(f"{name} must be a {kind} integer{suffix}")

    def effective_limit(self, rows: Sequence[Mapping[str, Any]]) -> int:
        target = self.max_retained_evidence_chars
        if self.max_required_evidence_chars is None:
            return target
        required = required_packet_size(rows)
        if required <= target or required == 2:  # Empty required packet: [].
            return target
        return max(target, min(self.max_required_evidence_chars,
                               required + self.required_evidence_headroom_chars))

    def audit_config(self, effective_evidence_chars: int | None = None) -> dict[str, Any]:
        return {
            "packing_policy_version": EVIDENCE_PACKING_POLICY_VERSION,
            **asdict(self),
            "effective_evidence_chars": (
                self.max_retained_evidence_chars
                if effective_evidence_chars is None else effective_evidence_chars
            ),
        }


def json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def evidence_citation_label(pointer: str) -> str:
    """Preserve the same measurement identity when either backend aliases it."""

    parts = [part for part in str(pointer).removeprefix("ev:").split("/") if part]
    if not parts:
        return "evidence"
    if len(parts) >= 4 and parts[0] == "representative_leads":
        return f"{parts[1]}.{parts[-1]}"
    if len(parts) >= 3 and parts[0] == "p_wave_assessments":
        return f"p_assessment[{parts[1]}].{parts[-1]}"
    if len(parts) >= 4 and parts[:2] == ["rhythm_inputs", "p_events"]:
        return f"p_event[{parts[2]}].{parts[-1]}"
    if len(parts) >= 3 and parts[0] == "beat_features":
        return f"beat[{parts[1]}].{parts[-1]}"
    if parts[0] == "waveform_review":
        if len(parts) >= 6 and parts[1] == "profiles" and parts[3] == "observations":
            return f"waveform.{parts[2]}[{parts[4]}].{parts[-1]}"
        return ".".join(("waveform", *parts[1:]))
    if parts[0] == "global_features":
        return f"global.{parts[-1]}"
    if len(parts) >= 2 and parts[0] == "quality":
        return ".".join(("quality", *parts[-2:]))
    if parts[0] == "rhythm_inputs":
        return ".".join(("rhythm", *parts[1:]))[-72:]
    if parts[0] == "metadata":
        return ".".join(parts)[-72:]
    return ".".join(parts[-3:])[-72:]


def required_packet_size(rows: Sequence[Mapping[str, Any]]) -> int:
    """Measure the complete mandatory JSON packet for scheduler overflow checks.

    Rows contain tool, arguments, and an atomic result with evidence atoms.
    Only views with required_by atoms contribute. The result is exact for the
    supplied citation representation (use aliased rows for backend sizing),
    excludes message wrappers, and is 2 for an empty required packet.
    """
    required = []
    for row in rows:
        result = row.get("result")
        if not _is_atomic_result(result):
            continue
        atoms = [a for a in result["evidence"] if isinstance(a, dict)]
        mandatory = [a for a in atoms if a.get("required_by")]
        if not mandatory:
            continue
        shell = {k: v for k, v in result.items()
                 if k not in {"evidence", "omitted_atom_count", "omission_policy", "tool", "arguments"}}
        shell.update(evidence=mandatory, omitted_atom_count=
                     max(0, int(result.get("omitted_atom_count") or 0)) + len(atoms) - len(mandatory))
        required.append({"tool": row.get("tool"), "arguments": row.get("arguments") or {}, "result": shell})
    return json_size(required)


def _is_atomic_result(result: Any) -> bool:
    return (isinstance(result, dict) and isinstance(result.get("evidence"), list)
            and str(result.get("contract", "")).startswith("ecgagent.model-evidence."))


def pack_evidence_rows(rows: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep each view's mandatory atoms together before optional round-robin.

    Sizes are computed once per atom; adding an atom changes only a comma and
    the omitted-count digits. Never expose pointer-only omitted manifests.
    """
    packed: list[dict[str, Any]] = []
    sources: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    deferred: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    raw_rows: list[dict[str, Any]] = []
    used = 2  # []
    for index, row in enumerate(rows):
        public = {"tool": row.get("tool"), "arguments": row.get("arguments") or {},
                  "result": row.get("result") or ""}
        result = public["result"]
        if not _is_atomic_result(result):
            raw_rows.append(public)
            continue
        atoms = [dict(atom) for atom in result["evidence"] if isinstance(atom, dict)]
        omitted = max(0, int(result.get("omitted_atom_count") or 0))
        shell = {k: v for k, v in result.items()
                 if k not in {"evidence", "omitted_atom_count", "omission_policy", "tool", "arguments"}}
        mandatory = [atom for atom in atoms if atom.get("required_by")]
        shell.update(evidence=mandatory, omitted_atom_count=len(atoms) + omitted - len(mandatory))
        public["result"] = shell
        # Reserve each required bundle together with its shell before optional
        # views can consume the budget. Withheld bundles never gain optional
        # atoms that could be mistaken for satisfying the required minimum.
        cost = json_size(public) + bool(packed)
        if not mandatory or used + cost > limit:
            shell.update(evidence=[], omitted_atom_count=len(atoms) + omitted)
            deferred.append((index, public, [] if mandatory else atoms))
            continue
        used += cost
        packed.append(public)
        sources.append((index, public, [a for a in atoms if not a.get("required_by")]))

    for index, public, atoms in deferred:
        cost = json_size(public) + bool(packed)
        if used + cost <= limit:
            used += cost
            packed.append(public)
            sources.append((index, public, atoms))
    sources.sort(key=lambda source: source[0])
    packed = [public for _, public, _ in sources]

    def admit(result: dict[str, Any], atoms: list[dict[str, Any]], sizes: list[int]) -> bool:
        nonlocal used
        selected = result["evidence"]
        old = result["omitted_atom_count"]
        new = old - len(atoms)
        cost = sum(sizes) + max(0, len(atoms) - (0 if selected else 1))
        cost += len(str(new)) - len(str(old))
        if used + cost > limit:
            return False
        selected.extend(atoms)
        result["omitted_atom_count"] = new
        used += cost
        return True

    optional: list[tuple[dict[str, Any], list[tuple[dict[str, Any], int]]]] = []
    for _, public, atoms in sources:
        optional.append((public["result"], [(a, json_size(a)) for a in atoms]))
    for index in range(max((len(atoms) for _, atoms in optional), default=0)):
        for result, atoms in reversed(optional):
            if index < len(atoms):
                atom, size = atoms[index]
                admit(result, [atom], [size])
    for public in reversed(raw_rows):
        cost = json_size(public) + bool(packed)
        if used + cost <= limit:
            packed.append(public)
            used += cost
    return packed

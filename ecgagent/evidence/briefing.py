"""The one-page record briefing pushed to the model before any tool call.

The briefing has to be sufficient for the ordinary record and honest about the
difficult one, in roughly 2k tokens.  It carries three things a bare feature
dump cannot: which measurements ecgfeat itself distrusts, which rules it
refused to evaluate and why, and where its reference engines disagree with the
authoritative one.

It deliberately does *not* list the reference engines' conclusions wholesale.
Those layers are marked reference-only in `reference_metadata`, and handing a
model a ready-made label list invites it to restate rather than reason.  Only
disagreements are surfaced, because a disagreement is a question, not an answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .store import EvidenceStore

# Global measurements every read starts from, in reading order.
_HEADLINE_FIELDS: tuple[tuple[str, str], ...] = (
    ("heart_rate_bpm", "HR"),
    ("atrial_rate_bpm", "atrial rate"),
    ("pr_ms", "PR"),
    ("qrs_ms", "QRS"),
    ("qt_ms", "QT"),
    ("qtc_bazett_ms", "QTcB"),
    ("qtc_fridericia_ms", "QTcF"),
)

_AXIS_FIELDS: tuple[tuple[str, str], ...] = (
    ("p_axis_deg", "P"),
    ("qrs_axis_deg", "QRS"),
    ("t_axis_deg", "T"),
    ("st_axis_deg", "ST"),
)

_RHYTHM_FIELDS: tuple[tuple[str, str], ...] = (
    ("rr_mean_ms", "RR mean"),
    ("rr_cv", "RR CV"),
    ("rmssd_ms", "RMSSD"),
    ("pnn50", "pNN50"),
)

# Curated cross-engine checks. Left: reference-only field in `interpretation`
# (Philips DXL lineage). Middle: statement-code substrings the authoritative
# layer uses for the same concept — several, because the two engines name
# things differently and a single fragment manufactures dissent that is not
# there. `bundle_branch_block` is the case that caught this: the reference
# layer reports "IVCD" while clinical_rules emits `nonspecific_ivcd`, so
# matching only `bundle_branch` reported a conflict where both agree.
# Only genuine mismatches are surfaced.
_CROSS_ENGINE_CHECKS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("probable_af", ("atrial_fibrillation",), "atrial fibrillation"),
    (
        "bundle_branch_block",
        ("bundle_branch", "rbbb", "lbbb", "ivcd", "conduction_delay"),
        "intraventricular conduction",
    ),
    ("wpw_pattern", ("preexcitation", "pre_excitation", "wpw"), "pre-excitation"),
    ("lae_definite", ("left_atrial",), "left atrial abnormality"),
    ("pathological_q_leads", ("q_wave", "infarct"), "pathological Q / prior infarct"),
)


@dataclass(frozen=True)
class Briefing:
    text: str
    citations: tuple[str, ...]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


def _fmt(value: Any, unit: str | None = None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        text = f"{value:.{digits}f}"
        return f"{text}{unit or ''}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "none"
    return f"{value}{unit or ''}"


def _headline_line(store: EvidenceStore, citations: list[str]) -> list[str]:
    lines: list[str] = []
    chunks: list[str] = []
    for field_name, label in _HEADLINE_FIELDS:
        evidence = store.try_resolve(f"/global_features/{field_name}")
        if evidence is None:
            continue
        citations.append(evidence.pointer)
        digits = 1 if field_name in {"heart_rate_bpm", "atrial_rate_bpm"} else 0
        marker = "!" if evidence.caveats else ""
        chunks.append(
            f"{label} {_fmt(evidence.value, evidence.unit and ' ' + evidence.unit, digits)}"
            f"{marker} [ev:{evidence.pointer}]"
        )
    lines.append("[intervals]  " + " | ".join(chunks))

    axes = []
    for field_name, label in _AXIS_FIELDS:
        evidence = store.try_resolve(f"/global_features/{field_name}")
        if evidence is None:
            continue
        citations.append(evidence.pointer)
        marker = "!" if evidence.caveats else ""
        axes.append(
            f"{label} {_fmt(evidence.value, ' deg')}{marker} [ev:{evidence.pointer}]"
        )
    if axes:
        lines.append("[axes]       " + " | ".join(axes))

    rhythm = []
    for field_name, label in _RHYTHM_FIELDS:
        evidence = store.try_resolve(f"/global_features/{field_name}")
        if evidence is None or evidence.value is None:
            continue
        citations.append(evidence.pointer)
        digits = 3 if field_name in {"rr_cv", "pnn50"} else 0
        rhythm.append(
            f"{label} {_fmt(evidence.value, None, digits)} [ev:{evidence.pointer}]"
        )
    if rhythm:
        lines.append("[rr]         " + " | ".join(rhythm))
    return lines


def _distrusted_measurements(store: EvidenceStore, citations: list[str]) -> list[str]:
    """Global measurements ecgfeat itself flagged. The '!' markers, explained."""
    notes: list[str] = []
    for field_name, _ in (*_HEADLINE_FIELDS, *_AXIS_FIELDS):
        evidence = store.try_resolve(f"/global_features/{field_name}")
        if evidence is None or not evidence.caveats:
            continue
        citations.extend(evidence.companions)
        notes.append(f"{field_name}: " + "; ".join(evidence.caveats))
    return notes


def _quality_block(store: EvidenceStore, citations: list[str]) -> list[str]:
    lines: list[str] = []
    grade = store.try_resolve("/metadata/record_quality/record_grade")
    if grade is not None:
        citations.append(grade.pointer)
    gate_state = store.try_resolve("/metadata/diagnostic_gate/state")
    gate_bits = []
    if gate_state is not None:
        citations.append(gate_state.pointer)
        gate_bits.append(f"gate={gate_state.value}")
    for key in ("stop_reasons", "partial_reasons", "allowed_domains"):
        entry = store.try_resolve(f"/metadata/diagnostic_gate/{key}")
        if entry is not None and entry.value:
            citations.append(entry.pointer)
            gate_bits.append(f"{key}={_fmt(entry.value)}")
    lines.append(
        f"[quality]    grade={_fmt(grade.value if grade else None)} | " + " | ".join(gate_bits)
    )

    per_modality: dict[str, list[str]] = {}
    unreliable: list[str] = []
    for lead in store.leads:
        quality = store.lead_quality(lead)
        for modality, key in (("P", "reliable_for_p"), ("QRS", "reliable_for_qrs"),
                              ("T", "reliable_for_t"), ("QT", "reliable_for_qt")):
            if quality.get(key) is False:
                per_modality.setdefault(modality, []).append(lead)
                citations.append(f"/quality/{lead}/{key}")
        if quality.get("reliable") is False:
            flags = ",".join(str(f) for f in (quality.get("flags") or []))
            unreliable.append(f"{lead}({flags})" if flags else lead)
            citations.append(f"/quality/{lead}/flags")
    if unreliable:
        lines.append("[leads]      unreliable: " + ", ".join(unreliable))
    if per_modality:
        lines.append(
            "[wave qc]    "
            + " | ".join(f"{m}-unreliable: {','.join(v)}" for m, v in sorted(per_modality.items()))
        )
    return lines


def _findings_block(store: EvidenceStore, citations: list[str], max_rows: int = 10) -> list[str]:
    matched: list[str] = []
    for base, row in sorted(
        store.rule_index().values(), key=lambda item: str(item[1].get("rule_id"))
    ):
        if row.get("status") != "matched":
            continue
        citations.extend([f"{base}/status", f"{base}/statement"])
        if row.get("statement_code"):
            citations.append(f"{base}/statement_code")
        if row.get("confidence"):
            citations.append(f"{base}/confidence")
        matched.append(
            f"  {str(row.get('rule_id') or ''):<34} "
            f"code={str(row.get('statement_code') or ''):<34} "
            f"{str(row.get('statement') or row.get('statement_code') or '')[:44]:<44} "
            f"{str(row.get('confidence') or ''):<7} {row.get('priority') or ''}"
        )
    lines = ["[matched]" + (f"    {len(matched)} rule(s)" if matched else "    none")]
    lines.extend(matched[:max_rows])
    if len(matched) > max_rows:
        lines.append(f"  ... {len(matched) - max_rows} more; use list_findings")
    return lines


def _abstention_block(store: EvidenceStore, citations: list[str], max_rows: int = 8) -> list[str]:
    """Render both abstention shapes ecgfeat emits, without conflating them.

    Rule-level entries carry `rule_id` plus the `missing_inputs` that blocked
    them.  Finding-layer entries carry `finding_id` and a three-state `UNKNOWN`
    from the v10 contract, meaning the measurement layer never projected the
    finding at all.  They need different follow-ups, so they are labelled apart
    and each cites the id key it actually has.
    """
    abstentions = store.abstentions()
    if not abstentions:
        return []

    rule_rows: list[tuple[int, dict[str, Any]]] = []
    finding_rows: list[tuple[int, dict[str, Any]]] = []
    for position, row in enumerate(abstentions):
        (finding_rows if row.get("finding_id") else rule_rows).append((position, row))

    lines: list[str] = []
    if rule_rows:
        lines.append(f"[abstained]  {len(rule_rows)} rule(s) could not be decided")
        for position, row in rule_rows[:max_rows]:
            base = f"/clinical_interpretation/abstentions/{position}"
            citations.append(f"{base}/rule_id")
            missing_inputs = row.get("missing_inputs") or []
            citations.append(f"{base}/missing_inputs" if missing_inputs else f"{base}/reason")
            detail = ", ".join(str(m) for m in missing_inputs) or str(row.get("reason") or "")
            lines.append(f"  {str(row.get('rule_id') or ''):<34} needs: {detail[:60]}")
        if len(rule_rows) > max_rows:
            lines.append(
                f"  ... {len(rule_rows) - max_rows} more; "
                "use list_findings(status=['unavailable','indeterminate'])"
            )

    if finding_rows:
        names = []
        for position, row in finding_rows[:12]:
            citations.append(f"/clinical_interpretation/abstentions/{position}/finding_id")
            names.append(str(row.get("finding_id")))
        suffix = "" if len(finding_rows) <= 12 else f", ... {len(finding_rows) - 12} more"
        lines.append(
            f"[unknown]    {len(finding_rows)} finding(s) not projected by the measurement "
            f"layer (three-state UNKNOWN, not a negative): {', '.join(names)}{suffix}"
        )
    return lines


_NEGATIVE_STRINGS = {"", "none", "normal", "no", "absent", "negative", "unknown"}


def _reference_asserts(raw: Any) -> bool:
    """Whether a reference-layer field is actually claiming the finding.

    Plain truthiness is wrong for these shapes. `pathological_q_leads` is a
    per-lead dict, and a dict of twelve `False` values is still truthy — which
    reported the reference layer as asserting a prior infarct on records where
    it asserted nothing in any lead.
    """
    if raw is None or raw is False:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict):
        return any(_reference_asserts(value) for value in raw.values())
    if isinstance(raw, (list, tuple, set)):
        return any(_reference_asserts(value) for value in raw)
    if isinstance(raw, str):
        return raw.strip().lower() not in _NEGATIVE_STRINGS
    return bool(raw)


def _render_reference(raw: Any) -> str:
    """Show what the reference layer claimed, not the container it came in."""
    if isinstance(raw, dict):
        positive = [key for key, value in raw.items() if _reference_asserts(value)]
        return ", ".join(positive) if positive else "none"
    if isinstance(raw, (list, tuple, set)):
        positive = [str(item) for item in raw if _reference_asserts(item)]
        return ", ".join(positive) if positive else "none"
    return _fmt(raw)


def _cross_engine_block(store: EvidenceStore, citations: list[str]) -> list[str]:
    """Report only where the reference engines dissent from clinical_rules."""
    reference = store.document.get("interpretation") or {}
    if not reference:
        return []
    matched_codes = " ".join(
        str(row.get("statement_code") or "")
        for row in store.rule_evaluations()
        if row.get("status") == "matched"
    ).lower()

    dissent: list[str] = []
    for field_name, code_fragments, label in _CROSS_ENGINE_CHECKS:
        if field_name not in reference:
            continue
        raw = reference[field_name]
        reference_positive = _reference_asserts(raw)
        authoritative_positive = any(
            fragment in matched_codes for fragment in code_fragments
        )
        if reference_positive == authoritative_positive:
            continue
        citations.append(f"/interpretation/{field_name}")
        if reference_positive:
            dissent.append(
                f"  {label}: reference layer says {_render_reference(raw)}, "
                "clinical_rules did not assert it"
            )
        else:
            dissent.append(
                f"  {label}: clinical_rules asserted it, reference layer did not"
            )
    if not dissent:
        return []
    return [
        "[dissent]    reference-only engines disagree with the authoritative layer "
        "(clinical_rules arbitrates; these are questions, not answers)",
        *dissent,
    ]


def build_chart_briefing(store: EvidenceStore) -> Briefing:
    """Render the one-page briefing and the pointers it cites."""
    citations: list[str] = []
    lines: list[str] = []

    age = store.try_resolve("/metadata/patient_meta/age")
    sex = store.try_resolve("/metadata/patient_meta/sex")
    fs = store.try_resolve("/metadata/input_fs")
    duration = store.try_resolve("/metadata/duration_sec")
    n_beats = store.try_resolve("/metadata/n_beats")
    for entry in (age, sex, fs, duration, n_beats):
        if entry is not None:
            citations.append(entry.pointer)
    lines.append(
        f"[record]     {store.record_id} | "
        f"age {_fmt(age.value if age else None)} | sex {_fmt(sex.value if sex else None)} | "
        f"fs {_fmt(fs.value if fs else None, ' Hz')} | "
        f"{_fmt(duration.value if duration else None, ' s', 1)} | "
        f"{_fmt(n_beats.value if n_beats else None)} beats | {len(store.leads)} leads"
    )

    lines.extend(_quality_block(store, citations))
    lines.extend(_headline_line(store, citations))

    distrusted = _distrusted_measurements(store, citations)
    if distrusted:
        lines.append("[distrusted] ecgfeat flagged these globals (marked ! above):")
        lines.extend(f"  {note}" for note in distrusted)

    lines.extend(_findings_block(store, citations))
    lines.extend(_abstention_block(store, citations))
    lines.extend(_cross_engine_block(store, citations))

    review = store.try_resolve("/clinical_interpretation/review_required")
    if review is not None and review.value:
        citations.append(review.pointer)
        reasons = store.raw("/clinical_interpretation/review_reasons") or []
        lines.append(f"[review]     required: {', '.join(str(r) for r in reasons[:8])}")

    # Every pointer counted as briefing provenance must be literally visible to
    # the model.  Several compact prose blocks summarize lists or rule rows and
    # cannot place a token beside every constituent without becoming unreadable,
    # so retain one bounded provenance ledger at the end.  This prevents a
    # value the model saw in the briefing from becoming uncitable after context
    # compaction while preserving the distinction between touched and visible
    # pointers.
    visible_pointers = tuple(dict.fromkeys(citations))
    if visible_pointers:
        lines.append(
            "[briefing evidence pointers] "
            + " ".join(f"ev:{pointer}" for pointer in visible_pointers)
        )

    return Briefing(text="\n".join(lines), citations=visible_pointers)

"""Versioned clinical safety policy for impossible positive conclusions.

This module is a minimal clinical invariant engine. It never adds a diagnosis
and never compares the model with a reference interpretation; it only blocks
positive conclusions that violate the explicitly versioned safety policy.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from typing import Any

from .safety_policy import (
    ClinicalSafetyPolicy,
    DEFAULT_CLINICAL_SAFETY_POLICY,
)


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = r"(?:bpm|beats?/min|\u6b21[/\uff0f]\u5206|ms|msec|\u6beb\u79d2|mV|mm)"
_COMPARISON_RE = re.compile(
    rf"(?P<left>{_NUMBER})\s*(?P<left_unit>{_UNIT})"
    rf"(?P<context>[^\d]{{0,48}}?)"
    rf"(?P<operator>\u4e0d\u4f4e\u4e8e|\u4e0d\u5c0f\u4e8e|\u4e0d\u5c11\u4e8e|\u4e0d\u9ad8\u4e8e|\u4e0d\u5927\u4e8e|\u4e0d\u8d85\u8fc7|"
    rf"\u4f4e\u4e8e|\u5c0f\u4e8e|\u5c11\u4e8e|\u9ad8\u4e8e|\u5927\u4e8e|\u8d85\u8fc7|\u81f3\u5c11|\u81f3\u591a|"
    rf"greater\s+than|less\s+than|at\s+least|at\s+most|"
    rf"above|below|under|over|exceeds?|>=|<=|>|<)"
    rf"(?P<label>[^\d]{{0,48}}?)"
    rf"(?P<right>{_NUMBER})\s*(?P<right_unit>{_UNIT})",
    re.IGNORECASE,
)

_LESS = {"\u4f4e\u4e8e", "\u5c0f\u4e8e", "\u5c11\u4e8e", "less than", "below", "under", "<"}
_LESS_EQUAL = {"\u4e0d\u9ad8\u4e8e", "\u4e0d\u5927\u4e8e", "\u4e0d\u8d85\u8fc7", "\u81f3\u591a", "at most", "<="}
_GREATER = {
    "\u9ad8\u4e8e",
    "\u5927\u4e8e",
    "\u8d85\u8fc7",
    "greater than",
    "above",
    "over",
    "exceed",
    "exceeds",
    ">",
}
_GREATER_EQUAL = {
    "\u4e0d\u4f4e\u4e8e",
    "\u4e0d\u5c0f\u4e8e",
    "\u4e0d\u5c11\u4e8e",
    "\u81f3\u5c11",
    "at least",
    ">=",
}

_BRADYCARDIA_CODES = frozenset({"bradycardia", "sinus_bradycardia"})
_TACHYCARDIA_CODES = frozenset({"tachycardia", "sinus_tachycardia"})
_ATRIAL_TACHYCARDIA_CODES = frozenset(
    {"atrial_tachycardia", "multifocal_atrial_tachycardia"}
)
_FIRST_DEGREE_CODES = frozenset(
    {"first_degree_av_block", "first_degree_av_delay"}
)
_SECOND_DEGREE_CODES = frozenset(
    {"second_degree_av_block", "second_degree_av_block_pattern"}
)
_COMPLETE_BUNDLE_CODES = frozenset(
    {
        "right_bundle_branch_block",
        "left_bundle_branch_block",
        "rbbb_pattern",
        "lbbb_pattern",
    }
)
_SHORT_QT_CODES = frozenset(
    {"short_qt", "borderline_short_qt", "possible_short_qt_pattern"}
)
_PROLONGED_QT_CODES = frozenset({"prolonged_qt", "markedly_prolonged_qt"})
_IVCD_CODES = frozenset({"nonspecific_ivcd", "probable_nonspecific_ivcd"})

_RETROGRADE_RE = re.compile(
    r"(?:\u9006\u4f20(?:P\u6ce2|\u5fc3\u623f|\u623f\u6027)|\u5fc3\u623f\u9006\u4f20|retrograde\s+(?:p|atrial))",
    re.IGNORECASE,
)
_VARIABLE_AV_RE = re.compile(
    r"(?:\b[2-9]\s*:\s*[1-9]\b|\u53ef\u53d8(?:\u6027)?\u623f\u5ba4\u4f20\u5bfc|"
    r"\u623f\u5ba4\u4f20\u5bfc(?:\u6bd4\u4f8b|\u6bd4\u7387)|variable\s+(?:a\s*[-/]?\s*v|atrioventricular)\s+conduction)",
    re.IGNORECASE,
)
_ATRIAL_TACHY_RE = re.compile(
    r"(?:\u623f\u6027\u5fc3\u52a8\u8fc7\u901f|\u623f\u901f|atrial\s+tachycardia)", re.IGNORECASE
)
_IVCD_RE = re.compile(
    r"(?:\u975e\u7279\u5f02\u6027(?:\u5fc3)?\u5ba4\u5185\u4f20\u5bfc\u5ef6\u8fdf|nonspecific\s+(?:ivcd|intraventricular\s+conduction\s+delay))",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"(?:\u65e0|\u672a\u89c1|\u672a\u68c0\u51fa|\u4e0d\u652f\u6301|\u4e0d\u80fd\u8bc1\u5b9e|\u5c1a\u672a\u8bc1\u5b9e|\u7f3a\u4e4f|\u6392\u9664|"
    r"\u672a\u786e\u8ba4|\u672a\u660e|\u53ef\u80fd|\u7591\u4f3c|\u662f\u5426|\u9274\u522b|\u5f85\u6392|"
    r"not\s+(?:seen|detected|supported|demonstrated|confirmed|established|verified)|no\s+evidence|"
    r"possible|suspected|differential|uncertain|unconfirmed|unavailable|unknown)",
    re.IGNORECASE,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _normalise_unit(value: str) -> str:
    unit = value.lower().replace(" ", "")
    if unit in {"msec", "\u6beb\u79d2"}:
        return "ms"
    if unit in {"beats/min", "beat/min", "\u6b21/\u5206", "\u6b21\uff0f\u5206"}:
        return "bpm"
    return unit


def _iter_strings(value: Any, path: str = "verdict") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_strings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_strings(child, f"{path}[{index}]")


def _comparison_holds(left: float, operator: str, right: float) -> bool:
    op = re.sub(r"\s+", " ", operator.strip().lower())
    if op in _LESS:
        return left < right
    if op in _LESS_EQUAL:
        return left <= right
    if op in _GREATER:
        return left > right
    if op in _GREATER_EQUAL:
        return left >= right
    return True


def _arithmetic_problems(verdict: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for path, text in _iter_strings(verdict):
        for match in _COMPARISON_RE.finditer(text):
            if _normalise_unit(match.group("left_unit")) != _normalise_unit(
                match.group("right_unit")
            ):
                continue
            left = float(match.group("left"))
            right = float(match.group("right"))
            operator = match.group("operator")
            if _comparison_holds(left, operator, right):
                continue
            identity = (path, match.start(), match.end())
            if identity in seen:
                continue
            seen.add(identity)
            excerpt = " ".join(match.group(0).split())
            problems.append(
                "[diagnosis_measurement_conflict] "
                f"{path} contains a false numeric comparison: `{excerpt}` "
                f"({_fmt(left)} {operator} {_fmt(right)} is false). Re-read the "
                "patient measurement with ecgfeat and remove the unsupported "
                "positive diagnosis or correct the explanation."
            )
    return problems


def _adult_patient(document: Mapping[str, Any]) -> bool:
    metadata = _mapping(document.get("metadata"))
    patient = _mapping(metadata.get("patient_meta"))
    age = _finite_number(patient.get("age"))
    return age is not None and age >= 18.0


def _diagnosis_evidence_numbers(
    diagnosis: Mapping[str, Any], unit: str
) -> list[float]:
    values: list[float] = []
    for key in ("evidence", "counterevidence"):
        items = diagnosis.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            # A qualifying measurement counts wherever the item carries it, or
            # a diagnosis could be cleared by moving its own numbers sideways.
            for extra in item.get("supporting_values") or []:
                if not isinstance(extra, Mapping):
                    continue
                if _normalise_unit(str(extra.get("unit") or "")) != unit:
                    continue
                extra_value = _finite_number(extra.get("value"))
                if extra_value is not None:
                    values.append(extra_value)
            if _normalise_unit(str(item.get("unit") or "")) != unit:
                continue
            value = _finite_number(item.get("value"))
            if value is not None:
                values.append(value)
    return values


def _diagnosis_citations(diagnosis: Mapping[str, Any]) -> set[str]:
    citations: set[str] = set()
    for key in ("evidence", "counterevidence"):
        items = diagnosis.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            for citation in item.get("citations") or []:
                if isinstance(citation, str):
                    citations.add(citation.removeprefix("ev:"))
            for extra in item.get("supporting_values") or []:
                if isinstance(extra, Mapping) and isinstance(
                    extra.get("citation"), str
                ):
                    citations.add(str(extra["citation"]).removeprefix("ev:"))
    return citations


def _positive_narrative_match(
    diagnosis: Mapping[str, Any], pattern: re.Pattern[str]
) -> bool:
    """Return true when statement/reasoning makes a non-negated claim."""

    for key in ("statement", "reasoning"):
        text = str(diagnosis.get(key) or "")
        for match in pattern.finditer(text):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            if not _NEGATION_RE.search(text[start:end]):
                return True
    return False


def _rhythm_sections(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    rhythm = _mapping(document.get("rhythm_inputs"))
    return (
        _mapping(rhythm.get("background")),
        _mapping(rhythm.get("av_block")),
        _mapping(rhythm.get("af_afl")),
    )


def _direct_atrial_rates(
    document: Mapping[str, Any],
) -> list[float]:
    global_features = _mapping(document.get("global_features"))
    background, _, _ = _rhythm_sections(document)
    rates: list[float] = []
    for value in (
        _finite_number(global_features.get("atrial_rate_bpm")),
        _finite_number(background.get("background_atrial_rate_bpm")),
    ):
        if value is not None and value > 0.0:
            rates.append(value)
    return rates


def _retrograde_localization_support(
    document: Mapping[str, Any],
    policy: ClinicalSafetyPolicy = DEFAULT_CLINICAL_SAFETY_POLICY,
) -> tuple[bool, int, int]:
    """Summarize beat/lead support independent of the atrial-event candidate stream."""

    rows = [
        row
        for row in document.get("beat_features") or []
        if isinstance(row, Mapping) and "p_localization_retrograde" in row
    ]
    beat_ids = {
        row.get("beat_id")
        for row in rows
        if row.get("p_localization_retrograde") is True
        and row.get("beat_id") is not None
    }
    leads = {
        str(row.get("lead"))
        for row in rows
        if row.get("p_localization_retrograde") is True and row.get("lead")
    }
    # Repetition across both time and space is required before a candidate
    # atrial-event association is promoted to a record-level statement.
    supported = (
        len(beat_ids) >= policy.retrograde_min_beats
        and len(leads) >= policy.retrograde_min_leads
    )
    return supported, len(beat_ids), len(leads)


def _av_conduction_support(
    document: Mapping[str, Any],
) -> tuple[bool, Mapping[str, Any]]:
    _, av_block, _ = _rhythm_sections(document)
    evidence = _mapping(av_block.get("evidence"))
    dropped = evidence.get("dropped_p_evidence") is True and bool(
        evidence.get("dropped_p_interval_indices")
    )
    localized = evidence.get("localized_atrial_event_excess") is True
    candidate_only = (
        evidence.get("constant_multiple_atrial_events_requires_validation") is True
    )
    return bool(dropped or (localized and not candidate_only)), evidence


def _definition_problems(
    verdict: Mapping[str, Any],
    document: Mapping[str, Any],
    policy: ClinicalSafetyPolicy = DEFAULT_CLINICAL_SAFETY_POLICY,
) -> list[str]:
    """Reject only definition-level contradictions; never infer positives."""

    if not _adult_patient(document):
        return []
    global_features = _mapping(document.get("global_features"))
    heart_rate = _finite_number(global_features.get("heart_rate_bpm"))
    pr_ms = _finite_number(global_features.get("pr_ms"))
    qrs_values = [
        value
        for value in (
            _finite_number(global_features.get("qrs_ms")),
            _finite_number(global_features.get("qrs_wide_ms")),
        )
        if value is not None
    ]
    qt_reportable = global_features.get("qt_reportable") is True
    qt_reliability = str(global_features.get("qt_reliability") or "").lower()
    reliable_qtc = qt_reportable and qt_reliability not in {
        "unreliable",
        "unavailable",
        "not_reportable",
    }
    qtc_values = [
        value
        for field in (
            "qtc_bazett_ms",
            "qtc_fridericia_ms",
            "qtc_framingham_ms",
            "qtc_hodges_ms",
        )
        if (value := _finite_number(global_features.get(field))) is not None
    ]

    diagnoses = verdict.get("diagnoses")
    if not isinstance(diagnoses, list):
        return []
    problems: list[str] = []
    diagnosis_codes = {
        str(row.get("code") or "")
        for row in diagnoses
        if isinstance(row, Mapping)
    }
    for index, diagnosis in enumerate(diagnoses):
        if not isinstance(diagnosis, Mapping):
            continue
        code = str(diagnosis.get("code") or "")
        where = f"diagnoses[{index}] `{code}`"

        if code in _BRADYCARDIA_CODES and heart_rate is not None:
            cited_rates = _diagnosis_evidence_numbers(diagnosis, "bpm")
            cutoff = policy.bradycardia_upper_exclusive_bpm
            if heart_rate >= cutoff and not any(rate < cutoff for rate in cited_rates):
                problems.append(
                    "[diagnosis_measurement_conflict] "
                    f"{where} conflicts with adult ecgfeat heart_rate_bpm="
                    f"{_fmt(heart_rate)} bpm: bradycardia requires a rate below "
                    f"{_fmt(cutoff)} bpm, and this diagnosis cites no qualifying slower rate. "
                    "Re-read the rate and remove the positive diagnosis or move "
                    "a genuinely unresolved hypothesis to the differential."
                )

        if code in _TACHYCARDIA_CODES and heart_rate is not None:
            cited_rates = _diagnosis_evidence_numbers(diagnosis, "bpm")
            cutoff = policy.tachycardia_lower_exclusive_bpm
            if heart_rate <= cutoff and not any(rate > cutoff for rate in cited_rates):
                problems.append(
                    "[diagnosis_measurement_conflict] "
                    f"{where} conflicts with adult ecgfeat heart_rate_bpm="
                    f"{_fmt(heart_rate)} bpm: tachycardia requires a rate above "
                    f"{_fmt(cutoff)} bpm, and this diagnosis cites no qualifying faster rate. "
                    "Re-read the rate and remove the unsupported positive diagnosis."
                )

        if code in _ATRIAL_TACHYCARDIA_CODES:
            atrial_rates = _direct_atrial_rates(document)
            citations = _diagnosis_citations(diagnosis)
            cites_direct_atrial_rate = any(
                pointer in citations
                for pointer in (
                    "/global_features/atrial_rate_bpm",
                    "/rhythm_inputs/background/background_atrial_rate_bpm",
                )
            )
            if (
                atrial_rates
                and max(atrial_rates)
                <= policy.atrial_tachycardia_lower_exclusive_bpm
            ):
                problems.append(
                    "[diagnosis_measurement_conflict] "
                    f"{where} conflicts with the direct measured atrial rate "
                    f"(maximum {_fmt(max(atrial_rates))} bpm), which is not "
                    "tachycardic. A QRST-residual dominant cycle or extra "
                    "atrial-event candidates cannot replace a validated atrial "
                    "rate. Remove the positive diagnosis or keep it as an "
                    "explicitly unresolved differential."
                )
            elif not cites_direct_atrial_rate:
                problems.append(
                    "[diagnosis_evidence_gate] "
                    f"{where} lacks a cited direct atrial-rate measurement. A "
                    "positive atrial-tachycardia conclusion requires a validated "
                    "atrial rate above "
                    f"{_fmt(policy.atrial_tachycardia_lower_exclusive_bpm)} bpm "
                    "plus non-sinus atrial activation "
                    "evidence; residual-signal cycle length is candidate-only."
                )

        if code in _FIRST_DEGREE_CODES:
            if pr_ms is None:
                problems.append(
                    "[diagnosis_measurement_conflict] "
                    f"{where} is definite, but ecgfeat did not produce a PR "
                    "interval. PR absence cannot establish first-degree AV delay; "
                    "re-read P/PR evidence and remove or downgrade the conclusion."
                )
            elif pr_ms <= policy.first_degree_av_block_pr_lower_exclusive_ms:
                problems.append(
                    "[diagnosis_measurement_conflict] "
                    f"{where} conflicts with ecgfeat PR={_fmt(pr_ms)} ms: a "
                    "definite adult first-degree AV delay requires PR above "
                    f"{_fmt(policy.first_degree_av_block_pr_lower_exclusive_ms)} ms. "
                    "Re-read PR and remove the unsupported positive diagnosis."
                )

        if (
            code == "short_pr_interval"
            and pr_ms is not None
            and pr_ms >= policy.short_pr_upper_exclusive_ms
        ):
            problems.append(
                "[diagnosis_measurement_conflict] "
                f"{where} conflicts with ecgfeat PR={_fmt(pr_ms)} ms: an adult "
                "short-PR conclusion requires PR below "
                f"{_fmt(policy.short_pr_upper_exclusive_ms)} ms. Re-read PR and "
                "remove the unsupported positive diagnosis."
            )

        if code in _SECOND_DEGREE_CODES:
            av_supported, av_evidence = _av_conduction_support(document)
            if av_evidence and not av_supported:
                problems.append(
                    "[diagnosis_evidence_gate] "
                    f"{where} lacks validated non-conducted atrial activity. "
                    "Extra atrial-event candidates inside ordinary RR intervals "
                    "do not establish second-degree AV block when localized "
                    "excess/dropped-P evidence is absent or explicitly requires "
                    "validation. Move the hypothesis to the differential."
                )

        # Use the widest ecgfeat global estimate and a conservative margin so
        # borderline/rounding cases are left to model reasoning and review.
        if (
            code in _COMPLETE_BUNDLE_CODES
            and qrs_values
            and max(qrs_values) < policy.complete_bundle_hard_exclusion_qrs_ms
        ):
            problems.append(
                "[diagnosis_measurement_conflict] "
                f"{where} conflicts with the widest ecgfeat global QRS estimate "
                f"({_fmt(max(qrs_values))} ms): this is too narrow for a definite "
                "complete bundle-branch block. Re-read QRS duration and morphology "
                "and remove or downgrade the conclusion."
            )

        if (
            code in _IVCD_CODES
            and qrs_values
            and max(qrs_values) <= policy.nonspecific_ivcd_lower_exclusive_ms
        ):
            problems.append(
                "[diagnosis_measurement_conflict] "
                f"{where} conflicts with the widest ecgfeat global QRS estimate "
                f"({_fmt(max(qrs_values))} ms). Adult nonspecific IVCD requires "
                "QRS prolongation beyond "
                f"{_fmt(policy.nonspecific_ivcd_lower_exclusive_ms)} ms and "
                "exclusion of RBBB/LBBB "
                "morphology; QRS notching or fragmentation alone cannot establish "
                "a conduction delay. Remove the positive diagnosis or describe "
                "the notch as an unconfirmed morphology observation."
            )

        if _positive_narrative_match(diagnosis, _VARIABLE_AV_RE):
            av_supported, av_evidence = _av_conduction_support(document)
            if av_evidence and not av_supported:
                problems.append(
                    "[diagnosis_evidence_gate] "
                    f"{where} positively asserts a conduction ratio/variable AV "
                    "conduction, but ecgfeat has no validated dropped-P or "
                    "localized excess-atrial-event evidence. Candidate counts "
                    "that explicitly require validation cannot be promoted into "
                    "the final diagnosis."
                )

        if _positive_narrative_match(diagnosis, _RETROGRADE_RE):
            supported, beat_count, lead_count = _retrograde_localization_support(
                document, policy
            )
            if not supported:
                problems.append(
                    "[diagnosis_evidence_gate] "
                    f"{where} positively asserts retrograde atrial activation, "
                    "but independent beat/lead localization does not show "
                    f"repeatable support ({beat_count} beat(s), {lead_count} "
                    "lead(s)). `p_events.association_type=retrograde` is a "
                    "candidate stream, not independent confirmation. Remove the "
                    "claim or move it to the differential."
                )

        # QT cutoffs vary by formula, sex and context.  These deliberately wide
        # exclusion bounds catch only gross polarity errors, and only when
        # ecgfeat says QT is reportable and reliable.
        if reliable_qtc and qtc_values:
            if (
                code in _PROLONGED_QT_CODES
                and max(qtc_values) <= policy.prolonged_qt_hard_exclusion_qtc_ms
            ):
                problems.append(
                    "[diagnosis_measurement_conflict] "
                    f"{where} conflicts with all reliable reportable ecgfeat QTc "
                    f"values (maximum {_fmt(max(qtc_values))} ms), which cannot "
                    "support a positive prolonged-QT conclusion."
                )
            if (
                code in _SHORT_QT_CODES
                and min(qtc_values) >= policy.short_qt_hard_exclusion_qtc_ms
            ):
                problems.append(
                    "[diagnosis_measurement_conflict] "
                    f"{where} conflicts with all reliable reportable ecgfeat QTc "
                    f"values (minimum {_fmt(min(qtc_values))} ms), which cannot "
                    "support a positive short-QT conclusion."
                )

    summary_claim = {"statement": verdict.get("summary")}
    if (
        _positive_narrative_match(summary_claim, _ATRIAL_TACHY_RE)
        and not (diagnosis_codes & _ATRIAL_TACHYCARDIA_CODES)
    ):
        problems.append(
            "[diagnosis_section_conflict] `summary` asserts atrial tachycardia, "
            "but no positive atrial-tachycardia diagnosis passed through the "
            "structured `diagnoses` list. Confirmed conclusions, differentials "
            "and limitations must remain separate."
        )
    if _positive_narrative_match(summary_claim, _VARIABLE_AV_RE):
        av_supported, av_evidence = _av_conduction_support(document)
        if av_evidence and not av_supported:
            problems.append(
                "[diagnosis_evidence_gate] `summary` positively asserts an AV "
                "conduction ratio or variable AV conduction without validated "
                "dropped-P/localized atrial-event support. Keep this candidate "
                "out of the confirmed summary."
            )
    if _positive_narrative_match(summary_claim, _RETROGRADE_RE):
        supported, beat_count, lead_count = _retrograde_localization_support(
            document, policy
        )
        if not supported:
            problems.append(
                "[diagnosis_evidence_gate] `summary` positively asserts "
                "retrograde atrial activation without repeatable independent "
                f"beat/lead support ({beat_count} beat(s), {lead_count} lead(s))."
            )
    if (
        _positive_narrative_match(summary_claim, _IVCD_RE)
        and not (diagnosis_codes & _IVCD_CODES)
    ):
        problems.append(
            "[diagnosis_section_conflict] `summary` asserts nonspecific IVCD, "
            "but no positive IVCD diagnosis passed through the structured "
            "`diagnoses` list. A morphology observation cannot bypass the "
            "diagnostic evidence gate through summary prose."
        )
    return problems


_LIMB_REVERSAL_CODES = frozenset(
    {"limb_lead_reversal_suspected", "possible_limb_lead_reversal"}
)
_LIMB_REVERSAL_RE = re.compile(
    r"(\u80a2\u4f53\u5bfc\u8054\u53cd\u63a5|\u5bfc\u8054\u53cd\u63a5|\u7535\u6781\u63a5\u9519|\u5de6\u53f3[\u81c2\u624b]\u4e92\u6362|LA\s*/\s*RA|RA\s*/\s*LA"
    r"|limb\s+lead\s+reversal|arm\s+lead\s+reversal)",
    re.IGNORECASE,
)


def _dominant_deflection(
    document: Mapping[str, Any], lead: str
) -> tuple[str | None, float | None, float | None]:
    """Return which way a lead's QRS mainly points, from R against S.

    Reading one amplitude alone is what turns a normal lead into evidence of a
    swap: on LUDB 133 lead I has S=-0.110 mV and R=0.614 mV, and quoting only
    the S wave made an upright lead look inverted.
    """
    leads = _mapping(document.get("representative_leads"))
    params = _mapping(_mapping(leads.get(lead)).get("params"))
    r_amp = _finite_number(params.get("r_amp_mv"))
    s_amp = _finite_number(params.get("s_amp_mv"))
    if r_amp is None and s_amp is None:
        return None, r_amp, s_amp
    positive = abs(r_amp) if r_amp is not None else 0.0
    negative = abs(s_amp) if s_amp is not None else 0.0
    if positive == negative:
        return None, r_amp, s_amp
    return ("positive" if positive > negative else "negative"), r_amp, s_amp


def _limb_reversal_problems(
    verdict: Mapping[str, Any],
    document: Mapping[str, Any],
) -> list[str]:
    """Reject an asserted limb-lead swap whose defining criteria do not hold.

    The criteria are not a matter of interpretation: an LA/RA swap inverts lead
    I and makes aVR upright. Checking them here rather than asking the model to
    argue them is the point - a quality flag arrives with the authority of the
    pipeline, and prose reasoning about it tends to end where the flag started.
    """
    diagnoses = verdict.get("diagnoses")
    if not isinstance(diagnoses, list):
        return []
    lead_i, r_i, s_i = _dominant_deflection(document, "I")
    avr, r_avr, s_avr = _dominant_deflection(document, "aVR")
    if lead_i is None or avr is None:
        return []
    failures: list[str] = []
    if lead_i != "negative":
        failures.append(
            f"lead I is dominantly positive (r_amp_mv={_fmt(r_i or 0.0)} mV vs "
            f"s_amp_mv={_fmt(s_i or 0.0)} mV)"
        )
    if avr != "positive":
        failures.append(
            f"aVR is dominantly negative (r_amp_mv={_fmt(r_avr or 0.0)} mV vs "
            f"s_amp_mv={_fmt(s_avr or 0.0)} mV)"
        )
    if not failures:
        return []

    problems: list[str] = []
    for index, diagnosis in enumerate(diagnoses):
        if not isinstance(diagnosis, Mapping):
            continue
        code = str(diagnosis.get("code") or "")
        asserts_reversal = code in _LIMB_REVERSAL_CODES or _positive_narrative_match(
            diagnosis, _LIMB_REVERSAL_RE
        )
        if not asserts_reversal:
            continue
        problems.append(
            "[diagnosis_criteria_conflict] "
            f"diagnoses[{index}] `{code}` asserts limb lead reversal, but the "
            "defining criteria fail: " + "; ".join(failures) + ". An LA/RA swap "
            "inverts lead I and makes aVR upright. Read the dominant deflection "
            "in both leads, not one wave of it, and either withdraw the "
            "assertion or report the recording as inconsistent without naming a "
            "swap."
        )
    return problems


def validate_diagnosis_measurement_semantics(
    verdict: Mapping[str, Any],
    evidence_document: Mapping[str, Any] | None,
    policy: ClinicalSafetyPolicy = DEFAULT_CLINICAL_SAFETY_POLICY,
) -> list[str]:
    """Return blocking contradictions between model logic and ecgfeat evidence."""

    problems = _arithmetic_problems(verdict)
    if isinstance(evidence_document, Mapping):
        problems.extend(_definition_problems(verdict, evidence_document, policy))
        problems.extend(_limb_reversal_problems(verdict, evidence_document))
    return problems


__all__ = ["validate_diagnosis_measurement_semantics"]

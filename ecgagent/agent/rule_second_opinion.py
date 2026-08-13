"""Bounded, non-citable ecgfeat rule hints for the compact workflow.

The compact Agent first plans from the diagnosis-safe measurement document.
Only after that blind plan is complete may this module expose a small list of
rule-generated *candidates*.  The rows intentionally contain no measurements,
thresholds or clinical statement prose: they are routing hints, not patient
evidence, and can never satisfy a diagnosis citation requirement.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..evidence.store import EvidenceStore
from .diagnosis_catalog import DIAGNOSIS_CATALOG
from .diagnostic_ledger import NON_HYPOTHESIS_CODES


RULE_SECOND_OPINION_VERSION = "ecgagent.rule-second-opinion.v3"

_PUBLICATION_RANK = {"final": 0, "borderline": 1, "suppressed": 2}
_CONFIDENCE_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_SEVERITY_RANK = {
    "critical": 0,
    "emergent": 0,
    "high": 1,
    "major": 1,
    "moderate": 2,
    "warning": 2,
    "low": 3,
    "minor": 3,
}


def _mechanism_rank(row: Mapping[str, Any]) -> int:
    """Prefer specific mechanisms over broad measurement observations."""

    code = str(row.get("code") or "")
    category = str(row.get("category") or "")
    if category in {"conduction", "rhythm", "ectopy", "pacing"}:
        return 0
    if code in {
        "acute_occlusion_pattern",
        "left_main_pattern",
        "de_winter_pattern",
        "sgarbossa_positive",
        "brugada_type1_screening",
    }:
        return 0
    if category in {"interval", "chamber"}:
        return 1
    return 2
_NON_MATERIAL_RULE_CODES = frozenset(
    {
        # The compact plan intentionally spends candidate slots on material
        # abnormalities. These normal/background labels remain visible in the
        # measurement overview and must not displace an abnormal rule hint.
        "sinus_rhythm",
        "sinus_mechanism",
        # This is an exclusion statement, not a positive ECG phenotype.
        "exclude_2_to_1_atrial_flutter",
    }
)


def _priority_rank(value: Any) -> int:
    token = str(value or "").strip().upper()
    if len(token) >= 2 and token[0] == "P" and token[1:].isdigit():
        return min(int(token[1:]), 99)
    return 99


def _candidate_sort_key(
    row: Mapping[str, Any],
) -> tuple[int, int, int, int, int, int, str]:
    priority = _priority_rank(row.get("priority"))
    # Within the non-emergent pool, preserve specific rhythm/conduction/ectopy
    # mechanisms before generic morphology observations.  The disease-enriched
    # replay showed P5 PAC/PVC candidates repeatedly displaced by P2/P4 broad
    # R-progression and repolarization observations even though the former are
    # the more specific mechanism to adjudicate.  Urgent/P1 routing remains
    # first, and rule priority still orders candidates within a mechanism tier.
    urgent = 0 if priority <= 1 else 1
    return (
        urgent,
        _mechanism_rank(row),
        priority,
        _SEVERITY_RANK.get(str(row.get("severity") or "").lower(), 9),
        _PUBLICATION_RANK.get(str(row.get("publication") or ""), 9),
        _CONFIDENCE_RANK.get(str(row.get("confidence") or "").upper(), 9),
        str(row.get("code") or ""),
    )


def _safe_strings(value: Any, *, limit: int = 6) -> list[str]:
    return [
        str(item).strip()
        for item in (value if isinstance(value, list) else [])
        if str(item).strip()
    ][:limit]


def _candidate_row(raw: Mapping[str, Any], publication: str) -> dict[str, Any] | None:
    code = str(raw.get("statement_code") or "").strip()
    if (
        not code
        or code not in DIAGNOSIS_CATALOG
        or code in NON_HYPOTHESIS_CODES
        or code in _NON_MATERIAL_RULE_CODES
    ):
        return None
    status = str(raw.get("status") or publication).strip().lower()
    # Negative rule evaluations are deliberately absent.  A strict rule that
    # did not match is never counterevidence against an independently raised
    # candidate.
    if status in {"not_matched", "not-matched", "negative"}:
        return None
    return {
        "code": code,
        "label": DIAGNOSIS_CATALOG[code][0],
        "category": DIAGNOSIS_CATALOG[code][1],
        "rule_id": str(raw.get("rule_id") or "").strip(),
        "rule_domain": str(raw.get("domain") or "").strip(),
        "publication": publication,
        "status": status,
        "confidence": str(raw.get("confidence") or "").strip().upper() or None,
        "priority": str(raw.get("priority") or "").strip().upper() or None,
        "severity": str(raw.get("severity") or "").strip().lower() or None,
        "missing_inputs": _safe_strings(raw.get("missing_inputs")),
        "suppressed_by": _safe_strings(raw.get("suppressed_by")),
    }


def build_rule_second_opinion(
    store: EvidenceStore,
    *,
    candidate_limit: int = 8,
    gap_limit: int = 4,
) -> dict[str, Any]:
    """Extract a small rule-candidate packet before diagnosis scrubbing.

    The returned object is safe to place in model context only after blind
    planning.  It includes rule identity/status for provenance but excludes
    statements, thresholds and evidence values so the model cannot mistake a
    rule conclusion for independently cited patient evidence.
    """

    clinical = store.document.get("clinical_interpretation")
    if not isinstance(clinical, Mapping):
        return {
            "version": RULE_SECOND_OPINION_VERSION,
            "candidate_rows": [],
            "coverage_gaps": [],
            "counts": {"eligible": 0, "published": 0, "coverage_gaps": 0},
        }

    collected: list[dict[str, Any]] = []
    publication_keys = (
        ("final_statements", "final"),
        ("borderline_statements", "borderline"),
        ("suppressed_statements", "suppressed"),
    )
    for key, publication in publication_keys:
        rows = clinical.get(key)
        for raw in rows if isinstance(rows, list) else []:
            if not isinstance(raw, Mapping):
                continue
            row = _candidate_row(raw, publication)
            if row is not None:
                collected.append(row)

    # Keep one best row per diagnosis code.  This prevents a rule published in
    # both a final and a borderline collection from consuming two slots.
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in sorted(collected, key=_candidate_sort_key):
        deduplicated.setdefault(str(row["code"]), row)
    eligible = sorted(deduplicated.values(), key=_candidate_sort_key)

    gaps: list[dict[str, Any]] = []
    for raw in clinical.get("abstentions") or []:
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status") or "unavailable").strip().lower()
        if status in {"not_matched", "not-matched", "negative"}:
            continue
        gaps.append(
            {
                "rule_id": str(raw.get("rule_id") or "").strip(),
                "rule_domain": str(raw.get("domain") or "").strip(),
                "status": status,
                "missing_inputs": _safe_strings(raw.get("missing_inputs")),
                "suppressed_by": _safe_strings(raw.get("suppressed_by")),
            }
        )

    bounded_candidates = eligible[: max(0, int(candidate_limit))]
    bounded_gaps = gaps[: max(0, int(gap_limit))]
    return {
        "version": RULE_SECOND_OPINION_VERSION,
        "candidate_rows": bounded_candidates,
        "coverage_gaps": bounded_gaps,
        "counts": {
            "eligible": len(eligible),
            "published": len(bounded_candidates),
            "coverage_gaps": len(gaps),
        },
    }


_CATEGORY_DOMAINS: dict[str, list[str]] = {
    "quality": ["quality"],
    "rhythm": ["rhythm_rate", "p_av"],
    "rate": ["rhythm_rate"],
    "interval": ["intervals"],
    "axis": ["axis"],
    "conduction": ["conduction_preexcitation"],
    "ectopy": ["ectopy_pauses"],
    "chamber": ["voltage_chamber_r_progression"],
    "ischemia_repolarization": ["q_st_t_u"],
    "pacing": ["pacing_high_risk"],
    "other": ["q_st_t_u"],
}


def rule_candidate_blueprint(code: str) -> dict[str, Any]:
    """Return deterministic, diagnosis-neutral views for a rule candidate."""

    category = DIAGNOSIS_CATALOG.get(code, ("", "other"))[1]
    domains = list(_CATEGORY_DOMAINS.get(category, ["q_st_t_u"]))
    checks: list[dict[str, str]]
    if code in {"first_degree_av_block", "first_degree_av_delay", "possible_first_degree_av_delay"}:
        domains = ["intervals", "p_av"]
        checks = [
            {
                "tool": "get_interval_waveform_context",
                "purpose": "support",
                "question": "Review PR-interval availability and its component-wave evidence",
            },
            {
                "tool": "get_atrial_event_table",
                "purpose": "falsify",
                "question": "Review P-wave to QRS association and seek evidence that refutes this candidate",
            },
        ]
    elif category == "rhythm":
        checks = [
            {
                "tool": "get_rhythm_profile",
                "purpose": "support",
                "question": "Review the background rhythm, atrial activity and AV association",
            },
            {
                "tool": "get_p_assessment_table",
                "purpose": "falsify",
                "question": "Seek reliable P-wave evidence that refutes this rhythm candidate",
            },
        ]
    elif category == "rate":
        checks = [
            {
                "tool": "get_rhythm_profile",
                "purpose": "falsify",
                "question": "Review whether the rate context and rhythm mechanism support this candidate",
            }
        ]
    elif category == "ectopy":
        checks = [
            {
                "tool": "get_beat_table",
                "purpose": "support",
                "question": "Review beat-to-beat timing in relation to the premature-complex candidate",
            },
            {
                "tool": "get_morphology_groups",
                "purpose": "falsify",
                "question": "Seek morphology-group evidence that refutes the ectopic-beat interpretation",
            },
        ]
    elif category == "conduction":
        checks = [
            {
                "tool": "get_morphology_groups",
                "purpose": "support",
                "question": "Review native QRS duration and morphology grouping",
            },
            {
                "tool": "get_morphology_map",
                "purpose": "falsify",
                "question": "Seek cross-lead QRS morphology that refutes the conduction pattern",
            },
        ]
    elif category == "interval":
        checks = [
            {
                "tool": "get_interval_waveform_context",
                "purpose": "falsify",
                "question": "Review interval reportability, component waves and counterevidence",
            }
        ]
    elif category == "axis":
        checks = [
            {
                "tool": "get_lead_table",
                "purpose": "falsify",
                "question": "Review whether limb-lead QRS morphology supports this axis candidate",
            }
        ]
    elif category == "chamber":
        checks = [
            {
                "tool": "get_lead_table",
                "purpose": "support",
                "question": "Review cross-lead voltage and waveform distribution",
            },
            {
                "tool": "get_morphology_map",
                "purpose": "falsify",
                "question": "Seek cross-lead evidence that refutes the voltage or chamber pattern",
            },
        ]
    elif category == "pacing":
        checks = [
            {
                "tool": "get_pacing_profile",
                "purpose": "support",
                "question": "Review pacing markers and the capture relationship",
            },
            {
                "tool": "get_native_beat_profile",
                "purpose": "falsify",
                "question": "Seek native beats or evidence that refutes the pacing interpretation",
            },
        ]
    else:
        checks = [
            {
                "tool": "get_native_beat_profile",
                "purpose": "support",
                "question": "Review the relevant waveforms in representative native beats",
            },
            {
                "tool": "get_morphology_map",
                "purpose": "falsify",
                "question": "Seek cross-lead morphology that refutes this candidate",
            },
        ]
    return {"domains": domains, "checks": checks}

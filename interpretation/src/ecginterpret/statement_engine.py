from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


SEVERITY_RANKS = {
    None: 0,
    "normal": 0,
    "borderline": 1,
    "possible": 1,
    "mild": 1,
    "abnormal": 2,
    "old_or_age_indeterminate": 2,
    "probable": 2,
    "suspected": 3,
    "severe": 3,
    "acute": 4,
    "critical": 5,
}

SUMMARY_CODE_LABELS = {
    1: "Normal ECG",
    2: "Normal ECG except for rate",
    3: "Normal ECG based on available leads",
    4: "Borderline ECG",
    5: "Abnormal ECG",
    6: "Technical error",
}

WPW_BYPASS_CATEGORIES = ("mi", "hypertrophy", "st_t", "pediatric_morphology")
PACING_BYPASS_CATEGORIES = ("rhythm", "af_afl", "mi", "hypertrophy", "st_t", "pediatric_morphology")
LEAD_STOP_BYPASS_CATEGORIES = ("mi", "hypertrophy", "st_t", "pediatric_morphology", "axis")
LBBB_SUPPRESSED_CATEGORIES = {"mi", "hypertrophy", "st_t", "prior"}
AF_P_DEPENDENT_CATEGORIES = {"sinus", "pr", "atrial_enlargement", "av_block"}


@dataclass
class CandidateStatement:
    code: str
    category: str
    severity: Any
    evidence: Dict[str, object]
    text: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    required_inputs: List[str] = field(default_factory=list)
    unavailable_inputs: List[str] = field(default_factory=list)
    suppressed_by: List[str] = field(default_factory=list)
    bypasses: List[str] = field(default_factory=list)
    status: str = "candidate"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        status = self.status or "candidate"
        if self.unavailable_inputs:
            status = "unavailable"
        elif self.suppressed_by:
            status = "suppressed"
        elif status == "candidate":
            status = "final"
        return {
            "code": self.code,
            "text": self.text,
            "category": self.category,
            "severity": self.severity,
            "severity_rank": _severity_rank(self.severity),
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
            "required_inputs": list(self.required_inputs),
            "unavailable_inputs": list(self.unavailable_inputs),
            "suppressed_by": list(self.suppressed_by),
            "bypasses": list(self.bypasses),
            "status": status,
            "reason": self.reason,
            "final": status == "final",
        }


@dataclass
class StatementResolution:
    candidates: List[Dict[str, object]]
    final: List[Dict[str, object]]
    suppressed: List[Dict[str, object]]
    bypassed: List[Dict[str, object]]
    unavailable: List[Dict[str, object]]
    summary_code: Dict[str, object]
    stop_further_interpretation: bool = False
    bypass_remaining_algorithm: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "available": bool(
                self.candidates or self.final or self.suppressed or self.bypassed or self.unavailable
            ),
            "candidates": list(self.candidates),
            "final": list(self.final),
            "final_statements": list(self.final),
            "suppressed": list(self.suppressed),
            "bypassed": list(self.bypassed),
            "unavailable": list(self.unavailable),
            "summary_code": dict(self.summary_code),
            "stop_further_interpretation": bool(self.stop_further_interpretation),
            "bypass_remaining_algorithm": bool(self.bypass_remaining_algorithm),
        }


def _severity_rank(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return int(SEVERITY_RANKS.get(str(value).lower() if value is not None else None, 0))


def _list_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _candidate_to_dict(candidate: Any) -> Dict[str, object]:
    if isinstance(candidate, CandidateStatement):
        return candidate.to_dict()
    if isinstance(candidate, dict):
        item = dict(candidate)
        item.setdefault("text", None)
        item.setdefault("category", "unspecified")
        item.setdefault("severity", 0)
        item.setdefault("severity_rank", _severity_rank(item.get("severity")))
        item.setdefault("confidence", item.get("probability"))
        item.setdefault("evidence", {})
        item.setdefault("source", None)
        item["required_inputs"] = _list_value(item.get("required_inputs"))
        item["unavailable_inputs"] = _list_value(item.get("unavailable_inputs"))
        item["suppressed_by"] = _list_value(item.get("suppressed_by"))
        item["bypasses"] = _list_value(item.get("bypasses"))
        status = str(item.get("status") or "candidate")
        if item["unavailable_inputs"]:
            status = "unavailable"
        elif item["suppressed_by"]:
            status = "suppressed"
        elif bool(item.get("final", False)):
            status = "final"
        item["status"] = status
        item["final"] = status == "final"
        return item
    raise TypeError(f"Unsupported statement candidate type: {type(candidate)!r}")


def _context_dict(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return context if isinstance(context, dict) else {}


def _pacing_is_continuous_ventricular(context: Dict[str, Any], candidates: Sequence[Dict[str, object]]) -> bool:
    pacing_context = context.get("pacing_context") if isinstance(context.get("pacing_context"), dict) else {}
    if bool(pacing_context.get("suppress_further_rhythm_interpretation")):
        return True
    if bool(pacing_context.get("continuous_ventricular_pacing")):
        return True
    if bool(pacing_context.get("continuous_pacing")) and (
        bool(pacing_context.get("ventricular_pacing_present")) or bool(pacing_context.get("dual_chamber_pacing_present"))
    ):
        return True
    for candidate in candidates:
        if candidate.get("code") != "paced_rhythm":
            continue
        evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
        if bool(evidence.get("continuous_pacing")) and (
            bool(evidence.get("ventricular_pacing_present")) or bool(evidence.get("dual_chamber_pacing_present"))
        ):
            return True
    return False


def _wpw_is_active(context: Dict[str, Any], candidates: Sequence[Dict[str, object]]) -> bool:
    preexcitation = context.get("preexcitation") if isinstance(context.get("preexcitation"), dict) else {}
    if bool(preexcitation.get("wpw_pattern")):
        return True
    return any(candidate.get("code") == "wpw_pattern" for candidate in candidates)


def _lbbb_is_active(context: Dict[str, Any], candidates: Sequence[Dict[str, object]]) -> bool:
    if str(context.get("bundle_branch_block") or "").upper() == "LBBB":
        return True
    return any(str(candidate.get("code") or "").lower() == "lbbb" for candidate in candidates)


def _af_suppresses_p_dependent(context: Dict[str, Any], candidates: Sequence[Dict[str, object]]) -> bool:
    af_afl = context.get("af_afl") if isinstance(context.get("af_afl"), dict) else {}
    availability = context.get("availability") if isinstance(context.get("availability"), dict) else {}
    af_active = bool(af_afl.get("probable_af")) or bool(
        af_afl.get("probable_flutter")
    ) or bool(af_afl.get("af_afl_indeterminate")) or any(
        str(candidate.get("code") or "") in {
            "probable_af",
            "atrial_fibrillation",
            "atrial_flutter",
            "af_afl_indeterminate",
        }
        for candidate in candidates
    )
    if not af_active:
        return False
    return not bool(availability.get("p_axis_available", True)) or not bool(
        availability.get("atrial_rhythm_available", True)
    )


def _lead_stop_active(context: Dict[str, Any]) -> Optional[str]:
    if bool(context.get("dextrocardia_suspected")):
        return "dextrocardia_suspected"
    lead_reversal = context.get("lead_reversal") if isinstance(context.get("lead_reversal"), dict) else {}
    if lead_reversal.get("limb") or lead_reversal.get("precordial"):
        return "lead_reversal_suspected"
    return None


def _add_bypass(
    records: List[Dict[str, object]],
    seen: set[tuple[str, str]],
    *,
    category: str,
    by: str,
    reason: str,
) -> None:
    key = (category, by)
    if key in seen:
        return
    seen.add(key)
    records.append({"category": category, "bypassed_by": by, "reason": reason})


def _status_sort_key(item: Dict[str, object]) -> tuple[int, str]:
    return (-_severity_rank(item.get("severity")), str(item.get("code") or ""))


def _summary_code_payload(code: int, source: str = "candidate_resolver_derived") -> Dict[str, object]:
    return {
        "code": int(code),
        "label": SUMMARY_CODE_LABELS[int(code)],
        "source": source,
    }


def derive_summary_code(
    final: Sequence[Dict[str, object]],
    *,
    bypassed: Sequence[Dict[str, object]] = (),
) -> Dict[str, object]:
    """Derive a generic record summary code from resolved final statements."""
    final_items = list(final or [])
    if any(str(item.get("category") or "") == "technical" for item in final_items):
        return _summary_code_payload(6)

    explicit_codes: List[int] = []
    for item in final_items:
        raw = item.get("summary_code")
        if raw is None and isinstance(item.get("evidence"), dict):
            raw = item["evidence"].get("summary_code")
        try:
            code = int(raw)
        except (TypeError, ValueError):
            continue
        if code in SUMMARY_CODE_LABELS:
            explicit_codes.append(code)
    if explicit_codes:
        return _summary_code_payload(max(explicit_codes), source="candidate_summary_code")

    if not final_items:
        lead_limited = any(
            str(item.get("bypassed_by") or "") in {"lead_reversal_suspected", "dextrocardia_suspected"}
            for item in bypassed or []
        )
        return _summary_code_payload(3 if lead_limited else 1)

    categories = {str(item.get("category") or "") for item in final_items}
    ranks = [_severity_rank(item.get("severity")) for item in final_items]
    max_rank = max(ranks) if ranks else 0
    if categories <= {"rate"} and max_rank > 0:
        return _summary_code_payload(2)
    if max_rank >= 2:
        return _summary_code_payload(5)
    if max_rank == 1:
        return _summary_code_payload(4)
    return _summary_code_payload(1)


def resolve_statement_candidates(
    candidates: Iterable[Any],
    context: Optional[Dict[str, Any]] = None,
) -> StatementResolution:
    """Resolve statement candidates into final, suppressed, bypassed, and unavailable buckets."""
    context = _context_dict(context)
    normalized = [_candidate_to_dict(candidate) for candidate in candidates]

    bypassed: List[Dict[str, object]] = []
    seen_bypass: set[tuple[str, str]] = set()
    for candidate in normalized:
        for category in _list_value(candidate.get("bypasses")):
            _add_bypass(
                bypassed,
                seen_bypass,
                category=category,
                by=str(candidate.get("code") or "unknown"),
                reason=f"{candidate.get('code')}_bypass",
            )

    wpw_active = _wpw_is_active(context, normalized)
    if wpw_active:
        for category in WPW_BYPASS_CATEGORIES:
            _add_bypass(bypassed, seen_bypass, category=category, by="wpw_pattern", reason="preexcitation_bypass")

    pacing_active = _pacing_is_continuous_ventricular(context, normalized)
    if pacing_active:
        for category in PACING_BYPASS_CATEGORIES:
            _add_bypass(
                bypassed,
                seen_bypass,
                category=category,
                by="continuous_ventricular_pacing",
                reason="paced_rhythm_bypass",
            )

    lead_stop_reason = _lead_stop_active(context)
    if lead_stop_reason:
        for category in LEAD_STOP_BYPASS_CATEGORIES:
            _add_bypass(
                bypassed,
                seen_bypass,
                category=category,
                by=lead_stop_reason,
                reason="lead_orientation_bypass",
            )

    lbbb_active = _lbbb_is_active(context, normalized)
    af_p_dependent_suppressed = _af_suppresses_p_dependent(context, normalized)
    bypass_by_category: Dict[str, List[str]] = {}
    for record in bypassed:
        bypass_by_category.setdefault(str(record["category"]), []).append(str(record["bypassed_by"]))

    resolved: List[Dict[str, object]] = []
    for candidate in normalized:
        item = dict(candidate)
        category = str(item.get("category") or "")
        code = str(item.get("code") or "")
        suppressed_by = _list_value(item.get("suppressed_by"))

        if wpw_active and code != "wpw_pattern" and category in WPW_BYPASS_CATEGORIES:
            suppressed_by.append("wpw_pattern")
        if pacing_active and category in PACING_BYPASS_CATEGORIES and category != "pacing":
            suppressed_by.append("continuous_ventricular_pacing")
        if lead_stop_reason and category in LEAD_STOP_BYPASS_CATEGORIES:
            suppressed_by.append(lead_stop_reason)
        if lbbb_active and code.lower() != "lbbb" and category in LBBB_SUPPRESSED_CATEGORIES:
            suppressed_by.append("lbbb_secondary_repolarization")
        if af_p_dependent_suppressed and category in AF_P_DEPENDENT_CATEGORIES:
            suppressed_by.append("af_afl_p_wave_dependent_rule")

        item["suppressed_by"] = sorted(set(suppressed_by))
        if item.get("unavailable_inputs"):
            item["status"] = "unavailable"
        elif item["suppressed_by"]:
            item["status"] = "suppressed"
        elif item.get("status") in {"bypassed", "unavailable", "suppressed"}:
            item["status"] = item["status"]
        else:
            item["status"] = "final"
        item["final"] = item["status"] == "final"
        item["bypassed_by"] = bypass_by_category.get(category, [])
        resolved.append(item)

    final = sorted([item for item in resolved if item["status"] == "final"], key=_status_sort_key)
    suppressed = sorted([item for item in resolved if item["status"] == "suppressed"], key=lambda item: str(item.get("code")))
    unavailable = sorted([item for item in resolved if item["status"] == "unavailable"], key=lambda item: str(item.get("code")))
    return StatementResolution(
        candidates=resolved,
        final=final,
        suppressed=suppressed,
        bypassed=bypassed,
        unavailable=unavailable,
        summary_code=derive_summary_code(final, bypassed=bypassed),
        stop_further_interpretation=bool(pacing_active or lead_stop_reason),
        bypass_remaining_algorithm=bool(wpw_active or pacing_active or lead_stop_reason),
    )


def finalize_statements(candidates: List[CandidateStatement]) -> Dict[str, object]:
    return resolve_statement_candidates(candidates).to_dict()


def build_morphology_statement_evidence(features: Any) -> Dict[str, object]:
    interpretation = getattr(features, "interpretation", None)
    if interpretation is None:
        return {
            "available": False,
            "reason": "interpretation_not_available",
            "candidates": [],
            "final_statements": [],
            "stop_further_interpretation": False,
        }

    candidates: List[Dict[str, object]] = []
    candidates.extend(getattr(interpretation, "mi_statement_candidates", []) or [])
    rhythm_analysis = getattr(features, "metadata", {}).get("rhythm_analysis", {})
    pacing_context = rhythm_analysis.get("pacing_context", {}) if isinstance(rhythm_analysis, dict) else {}
    context = {
        "bundle_branch_block": getattr(interpretation, "bundle_branch_block", None),
        "dextrocardia_suspected": bool(getattr(interpretation, "dextrocardia_suspected", False)),
        "lead_reversal": getattr(features, "metadata", {}).get("lead_reversal", {}),
        "pacing_context": pacing_context,
    }
    resolution = resolve_statement_candidates(candidates, context=context).to_dict()
    resolution["reason"] = None if candidates else "no_candidates"
    if not candidates:
        resolution["available"] = False
    return resolution

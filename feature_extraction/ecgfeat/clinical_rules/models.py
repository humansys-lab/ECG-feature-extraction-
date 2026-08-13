from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


VALID_STATUSES = {
    "matched",
    "not_matched",
    "indeterminate",
    "unavailable",
    "not_applicable",
    "suppressed",
}

VALID_COVERAGE = {"full", "partial", "unavailable"}
VALID_NORMALITY_ROLES = {"core", "supporting", "optional_screen"}
VALID_PRIORITIES = {"P0", "P1", "P2", "P3", "P4", "P5", "P6"}


@dataclass
class RuleEvaluation:
    rule_id: str
    domain: str
    status: str
    statement_code: Optional[str] = None
    statement: Optional[str] = None
    severity: str = "normal"
    confidence: Optional[str] = None
    coverage: Optional[str] = None
    required_inputs: List[str] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    suppressed_by: List[str] = field(default_factory=list)
    source: Dict[str, str] = field(default_factory=dict)
    normality_required: bool = True
    normality_role: Optional[str] = None
    priority: Optional[str] = None
    standard_codes: Dict[str, str] = field(default_factory=dict)
    human_review_required: bool = False

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid rule status: {self.status}")
        if self.suppressed_by and self.status == "matched":
            self.status = "suppressed"
        if self.coverage is None:
            if self.status in {"unavailable", "indeterminate"}:
                self.coverage = "unavailable"
            elif self.status == "suppressed":
                self.coverage = "partial"
            else:
                self.coverage = "full"
        if self.coverage not in VALID_COVERAGE:
            raise ValueError(f"invalid rule coverage: {self.coverage}")
        if self.confidence is None and self.coverage == "unavailable":
            self.confidence = "unavailable"
        if self.normality_role is None:
            self.normality_role = (
                "core" if self.normality_required else "optional_screen"
            )
        if self.normality_role not in VALID_NORMALITY_ROLES:
            raise ValueError(f"invalid normality role: {self.normality_role}")
        if self.priority is not None and self.priority not in VALID_PRIORITIES:
            raise ValueError(f"invalid priority: {self.priority}")

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "RuleEvaluation":
        data = dict(row)
        legacy_status = data.get("status")
        if legacy_status == "unavailable":
            data["status"] = "indeterminate"
            data.setdefault("coverage", "unavailable")
            data.setdefault("confidence", "unavailable")
        elif legacy_status == "suppressed":
            data["status"] = "indeterminate"
            data.setdefault("coverage", "partial")
            data.setdefault("confidence", "low")
        if "normality_role" not in data:
            data["normality_role"] = (
                "core"
                if data.get("normality_required", True)
                else "optional_screen"
            )
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalAnalysis:
    schema_version: str
    ruleset_version: str
    overall_status: str
    summary: Dict[str, Any]
    final_statements: List[Dict[str, Any]]
    borderline_statements: List[Dict[str, Any]]
    suppressed_statements: List[Dict[str, Any]]
    unavailable_domains: List[str]
    conflicts: List[Dict[str, Any]]
    domains: Dict[str, Any]
    reference_interpretations: Dict[str, Any]
    generated_at: str
    artifact_fingerprint: str = ""
    findings: Dict[str, Any] = field(default_factory=dict)
    abstentions: List[Dict[str, Any]] = field(default_factory=list)
    review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    disclaimer: str = (
        "Research decision support only; not a validated medical device. "
        "Clinical findings require qualified human review."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

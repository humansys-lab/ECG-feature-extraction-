from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


FIDELITIES = frozenset(
    {
        "glasgow_explicit",
        "public_standard_approximation",
        "existing_dxl_approximation",
        "not_reproducible_from_guide",
    }
)
SOURCE_KINDS = frozenset({"glasgow_guide", "public_standard", "existing_dxl", "none"})
EVALUATION_STATUSES = frozenset(
    {"matched", "not_matched", "unavailable", "not_applicable"}
)
RESOLUTION_STATUSES = frozenset({"final", "suppressed", "advisory", "no_statement"})


@dataclass(frozen=True)
class GlasgowConfig:
    qtc_formula: str = "hodges"
    adult_tachycardia_bpm: float = 100.0
    adult_bradycardia_bpm: float = 50.0
    paper_speed_mm_per_s: float = 25.0
    gain_mm_per_mv: float = 10.0
    strict: bool = False

    def __post_init__(self) -> None:
        if self.qtc_formula not in {"hodges", "bazett", "fridericia", "framingham"}:
            raise ValueError(f"unsupported QTc formula: {self.qtc_formula}")
        if self.adult_tachycardia_bpm <= 0.0 or self.adult_bradycardia_bpm <= 0.0:
            raise ValueError("adult rate thresholds must be positive")
        if self.paper_speed_mm_per_s <= 0.0 or self.gain_mm_per_mv <= 0.0:
            raise ValueError("ECG display scale must be positive")


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    chapter: str
    pdf_page: int
    category: str
    order: int
    statement: str
    required_inputs: Tuple[str, ...]
    fidelity: str
    source_kind: str
    source_reference: str
    evaluator: Callable[[Any], "RuleEvaluation"]
    summary_code: Optional[int] = None
    suppresses: Tuple[str, ...] = ()
    stops: Tuple[str, ...] = ()
    source_chapters: Tuple[int, ...] = ()


@dataclass
class RuleEvaluation:
    rule_id: str
    evaluation_status: str
    resolution_status: str = "no_statement"
    statement: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    required_inputs: List[str] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    suppressed_by: List[str] = field(default_factory=list)
    fidelity: str = "glasgow_explicit"
    source: Dict[str, Any] = field(default_factory=dict)
    approximation: Optional[Dict[str, Any]] = None
    evaluation_error: Optional[str] = None
    summary_code: Optional[int] = None

    def __post_init__(self) -> None:
        if self.evaluation_status not in EVALUATION_STATUSES:
            raise ValueError(f"invalid evaluation status: {self.evaluation_status}")
        if self.resolution_status not in RESOLUTION_STATUSES:
            raise ValueError(f"invalid resolution status: {self.resolution_status}")
        if self.fidelity not in FIDELITIES:
            raise ValueError(f"invalid fidelity: {self.fidelity}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlasgowAnalysis:
    schema_version: str
    source: str
    config: Dict[str, Any]
    patient_route: Dict[str, Any]
    measurement_matrix: Dict[str, Any]
    rule_evaluations: List[RuleEvaluation]
    statement_resolution: Dict[str, Any]
    coverage: Dict[str, Any]
    compatibility: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else dict(self.config),
            "patient_route": dict(self.patient_route),
            "measurement_matrix": dict(self.measurement_matrix),
            "rule_evaluations": [item.to_dict() for item in self.rule_evaluations],
            "statement_resolution": dict(self.statement_resolution),
            "coverage": dict(self.coverage),
            "compatibility": dict(self.compatibility),
        }

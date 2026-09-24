from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Dict, Iterable, Mapping, Optional

from .config import DEFAULT_DIAGNOSTIC_CONFIG, DiagnosticConfig


TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"
VALID_VALUES = {TRUE, FALSE, UNKNOWN}


@dataclass(frozen=True)
class FindingEvidence:
    lead: Optional[str]
    metric: str
    measured: Any
    threshold: Any
    clause: str


@dataclass(frozen=True)
class Finding:
    id: str
    value: str
    grade: Optional[str] = None
    evidence: tuple[FindingEvidence, ...] = field(default_factory=tuple)
    confidence: str = "HIGH"
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.value not in VALID_VALUES:
            raise ValueError(f"invalid finding value: {self.value}")

    def to_dict(self) -> dict:
        return asdict(self)


def tri_and(*values: str) -> str:
    if any(value == FALSE for value in values):
        return FALSE
    if all(value == TRUE for value in values):
        return TRUE
    return UNKNOWN


def tri_or(*values: str) -> str:
    if any(value == TRUE for value in values):
        return TRUE
    if all(value == FALSE for value in values):
        return FALSE
    return UNKNOWN


def tri_not(value: str) -> str:
    if value == TRUE:
        return FALSE
    if value == FALSE:
        return TRUE
    return UNKNOWN


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _available_boolean(value: Any) -> Optional[bool]:
    if isinstance(value, Mapping):
        if value.get("available") is False:
            return None
        nested = value.get("value")
        return nested if isinstance(nested, bool) else None
    return value if isinstance(value, bool) else None


def _comparison(
    *,
    finding_id: str,
    value: Optional[float],
    threshold: float,
    operator: str,
    metric: str,
    clause: str,
    lead: Optional[str] = None,
) -> Finding:
    evidence = (
        FindingEvidence(
            lead=lead,
            metric=metric,
            measured=value,
            threshold={operator: threshold},
            clause=clause,
        ),
    )
    if value is None:
        return Finding(
            id=finding_id,
            value=UNKNOWN,
            evidence=evidence,
            confidence="UNAVAILABLE",
            unavailable_reason=f"{metric}_unavailable",
        )
    matched = {
        "lt": value < threshold,
        "le": value <= threshold,
        "gt": value > threshold,
        "ge": value >= threshold,
    }[operator]
    boundary = threshold != 0 and abs(value - threshold) <= 0.10 * abs(threshold)
    return Finding(
        id=finding_id,
        value=TRUE if matched else FALSE,
        evidence=evidence,
        confidence="MEDIUM" if boundary else "HIGH",
    )


def _global(context: Any, name: str) -> Optional[float]:
    getter = getattr(context, "global_value", None)
    return _finite(getter(name)) if callable(getter) else None


def _lead(context: Any, lead: str, name: str, reliability: str = "reliable_for_qrs") -> Optional[float]:
    getter = getattr(context, "lead_value", None)
    return _finite(getter(lead, name, reliability)) if callable(getter) else None


def _boolean_finding(
    finding_id: str,
    value: Optional[bool],
    *,
    metric: str,
    clause: str,
    measured: Any = None,
) -> Finding:
    return Finding(
        id=finding_id,
        value=UNKNOWN if value is None else TRUE if value else FALSE,
        confidence="UNAVAILABLE" if value is None else "HIGH",
        unavailable_reason=f"{metric}_unavailable" if value is None else None,
        evidence=(
            FindingEvidence(
                lead=None,
                metric=metric,
                measured=measured if measured is not None else value,
                threshold=True,
                clause=clause,
            ),
        ),
    )


def _unknown(finding_id: str, reason: str, clause: str) -> Finding:
    return Finding(
        id=finding_id,
        value=UNKNOWN,
        confidence="UNAVAILABLE",
        unavailable_reason=reason,
        evidence=(
            FindingEvidence(
                lead=None,
                metric=reason,
                measured=None,
                threshold=None,
                clause=clause,
            ),
        ),
    )


def build_findings(
    context: Any,
    config: DiagnosticConfig = DEFAULT_DIAGNOSTIC_CONFIG,
) -> Dict[str, Finding]:
    """Project measurements into a stable three-valued finding layer.

    Unsupported or unavailable items are emitted explicitly as UNKNOWN; they
    are never silently converted to FALSE.
    """

    interval = config.intervals
    rhythm_cfg = config.rhythm
    findings: Dict[str, Finding] = {}

    hr = _global(context, "heart_rate_bpm")
    pr = _global(context, "pr_ms")
    qrs = _global(context, "qrs_ms")
    qtc = _global(context, "qtc_bazett_ms")
    p_duration = _lead(context, "II", "p_dur_consensus_ms", "reliable_for_p")
    axis = _global(context, "qrs_axis_deg")

    for item in (
        _comparison(
            finding_id="HR_BRADY",
            value=hr,
            threshold=interval.adult_bradycardia_bpm,
            operator="lt",
            metric="global.heart_rate_bpm",
            clause="8.2.HR_BRADY",
        ),
        _comparison(
            finding_id="HR_TACHY",
            value=hr,
            threshold=interval.adult_tachycardia_bpm,
            operator="gt",
            metric="global.heart_rate_bpm",
            clause="8.2.HR_TACHY",
        ),
        _comparison(
            finding_id="PR_PROLONGED",
            value=pr,
            threshold=interval.pr_prolonged_ms,
            operator="gt",
            metric="global.pr_ms",
            clause="8.2.PR_PROLONGED",
        ),
        _comparison(
            finding_id="PR_SHORT",
            value=pr,
            threshold=interval.pr_short_ms,
            operator="lt",
            metric="global.pr_ms",
            clause="8.2.PR_SHORT",
        ),
        _comparison(
            finding_id="QRS_WIDE",
            value=qrs,
            threshold=interval.qrs_wide_ms,
            operator="ge",
            metric="global.qrs_ms",
            clause="8.2.QRS_WIDE",
        ),
        _comparison(
            finding_id="QT_PROLONGED",
            value=qtc,
            threshold=(
                interval.qtc_prolonged_female_ms
                if str(getattr(context, "sex", "")).lower() in {"female", "f"}
                else interval.qtc_prolonged_male_ms
            ),
            operator="gt",
            metric="global.qtc_bazett_ms",
            clause="8.2.QT_PROLONGED",
        ),
        _comparison(
            finding_id="QT_SEVERE",
            value=qtc,
            threshold=interval.qtc_severe_ms,
            operator="gt",
            metric="global.qtc_bazett_ms",
            clause="8.2.QT_SEVERE",
        ),
        _comparison(
            finding_id="QT_SHORT",
            value=qtc,
            threshold=interval.qtc_short_ms,
            operator="lt",
            metric="global.qtc_bazett_ms",
            clause="8.2.QT_SHORT",
        ),
        _comparison(
            finding_id="P_DUR_PROLONGED",
            value=p_duration,
            threshold=interval.p_duration_prolonged_ms,
            operator="ge",
            metric="lead.II.p_dur_consensus_ms",
            clause="8.2.P_DUR_PROLONGED",
            lead="II",
        ),
    ):
        findings[item.id] = item

    if qrs is None:
        findings["QRS_BORDERLINE"] = _unknown(
            "QRS_BORDERLINE", "global.qrs_ms_unavailable", "8.2.QRS_BORDERLINE"
        )
    else:
        findings["QRS_BORDERLINE"] = Finding(
            id="QRS_BORDERLINE",
            value=(
                TRUE
                if interval.qrs_borderline_ms <= qrs < interval.qrs_wide_ms
                else FALSE
            ),
            evidence=(
                FindingEvidence(
                    lead=None,
                    metric="global.qrs_ms",
                    measured=qrs,
                    threshold={
                        "gte": interval.qrs_borderline_ms,
                        "lt": interval.qrs_wide_ms,
                    },
                    clause="8.2.QRS_BORDERLINE",
                ),
            ),
        )

    if axis is None:
        for finding_id in ("AXIS_NORMAL", "AXIS_LEFT", "AXIS_RIGHT", "AXIS_EXTREME"):
            findings[finding_id] = _unknown(
                finding_id, "global.qrs_axis_deg_unavailable", f"8.3.{finding_id}"
            )
    else:
        axis_values = {
            "AXIS_NORMAL": -30.0 <= axis <= 90.0,
            "AXIS_LEFT": -90.0 <= axis < -30.0,
            "AXIS_RIGHT": 90.0 < axis <= 180.0,
            "AXIS_EXTREME": -180.0 <= axis < -90.0,
        }
        for finding_id, matched in axis_values.items():
            findings[finding_id] = Finding(
                id=finding_id,
                value=TRUE if matched else FALSE,
                evidence=(
                    FindingEvidence(
                        lead=None,
                        metric="global.qrs_axis_deg",
                        measured=axis,
                        threshold=finding_id,
                        clause=f"8.3.{finding_id}",
                    ),
                ),
            )

    metadata = getattr(getattr(context, "features", None), "metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    quality_gate = metadata.get("diagnostic_gate")
    quality_gate = quality_gate if isinstance(quality_gate, Mapping) else {}
    quality_stop = (
        str(quality_gate.get("state") or "").lower() == "stop"
        if quality_gate
        else None
    )
    findings["SIGNAL_QUALITY_POOR"] = _boolean_finding(
        "SIGNAL_QUALITY_POOR",
        quality_stop,
        metric="metadata.diagnostic_gate.state",
        measured=quality_gate.get("state") if quality_gate else None,
        clause="4.3.SIGNAL_QUALITY_POOR",
    )

    rhythm_analysis = metadata.get("rhythm_analysis")
    rhythm_analysis = rhythm_analysis if isinstance(rhythm_analysis, Mapping) else {}
    af_afl = rhythm_analysis.get("af_afl_summary")
    af_afl = af_afl if isinstance(af_afl, Mapping) else {}
    rr_cv = _finite(af_afl.get("rr_cv"))
    organized_p = _finite(af_afl.get("organized_p_ratio"))
    findings["RR_IRREGULAR_ABSOLUTE"] = _comparison(
        finding_id="RR_IRREGULAR_ABSOLUTE",
        value=rr_cv,
        threshold=rhythm_cfg.rr_irregular_cv,
        operator="gt",
        metric="rhythm.rr_cv",
        clause="8.6.RR_IRREGULAR_ABSOLUTE",
    )
    findings["RR_REGULAR"] = _comparison(
        finding_id="RR_REGULAR",
        value=rr_cv,
        threshold=rhythm_cfg.rr_regular_cv,
        operator="lt",
        metric="rhythm.rr_cv",
        clause="8.6.RR_REGULAR",
    )
    findings["P_ABSENT"] = _comparison(
        finding_id="P_ABSENT",
        value=organized_p,
        threshold=rhythm_cfg.p_absent_ratio,
        operator="lt",
        metric="rhythm.organized_p_ratio",
        clause="8.6.P_ABSENT",
    )
    findings["F_WAVE_PRESENT"] = _boolean_finding(
        "F_WAVE_PRESENT",
        bool(af_afl.get("probable_flutter")) if af_afl else None,
        metric="rhythm.probable_flutter",
        clause="8.6.F_WAVE_PRESENT",
    )

    rule_summary = rhythm_analysis.get("rule_summary")
    rule_summary = rule_summary if isinstance(rule_summary, Mapping) else {}
    findings["AV_DISSOCIATION"] = _boolean_finding(
        "AV_DISSOCIATION",
        bool(rule_summary.get("av_dissociation")) if rule_summary else None,
        metric="rhythm.av_dissociation",
        clause="8.6.AV_DISSOCIATION",
    )
    pauses = rule_summary.get("pauses")
    pauses = pauses if isinstance(pauses, Mapping) else {}
    av_evidence = pauses.get("av_block_evidence")
    av_evidence = av_evidence if isinstance(av_evidence, Mapping) else {}
    findings["P_NONCONDUCTED"] = _boolean_finding(
        "P_NONCONDUCTED",
        bool(av_evidence.get("dropped_p_evidence")) if av_evidence else None,
        metric="rhythm.dropped_p_evidence",
        clause="8.6.P_NONCONDUCTED",
    )

    pacing = rhythm_analysis.get("pacing_context")
    pacing = pacing if isinstance(pacing, Mapping) else {}
    pacing_failures = rhythm_analysis.get("pacing_failures")
    pacing_failures = pacing_failures if isinstance(pacing_failures, Mapping) else {}
    pacing_values = {
        "PACING_SPIKE_PRESENT": (
            bool(getattr(getattr(context, "features", None).global_features, "pacing_spikes", []))
            if getattr(context, "features", None) is not None
            else None
        ),
        "PACED_VENTRICULAR": (
            bool(pacing.get("ventricular_pacing_present")) if pacing else None
        ),
        "PACED_ATRIAL": bool(pacing.get("atrial_pacing_present")) if pacing else None,
        "PACED_DUAL": bool(pacing.get("dual_chamber_pacing_present")) if pacing else None,
        "PACING_FAILURE_CAPTURE": (
            _available_boolean(
                pacing_failures.get("capture_failure_suspected")
            )
            if pacing_failures
            else None
        ),
        "PACING_UNDERSENSING": (
            _available_boolean(
                pacing_failures.get("sensing_failure_suspected")
            )
            if pacing_failures
            else None
        ),
    }
    for finding_id, value in pacing_values.items():
        findings[finding_id] = _boolean_finding(
            finding_id,
            value,
            metric=f"rhythm.{finding_id.lower()}",
            clause=f"8.5.{finding_id}",
        )

    # Emit every safety-critical checklist item even when the corresponding
    # measurement has not yet been made. This makes capability gaps visible.
    checklist = {
        "PR_PROGRESSIVE",
        "PR_CONSTANT",
        "RR_REGULARLY_IRREGULAR",
        "P_UNIFORM",
        "P_MULTIFORM",
        "P_RETROGRADE",
        "DELTA_WAVE",
        "TERMINAL_QRS_DISTORTION",
        "DE_WINTER_PATTERN",
        "WELLENS_PATTERN",
        "LEFT_MAIN_PATTERN",
        "POSTERIOR_MI_PATTERN",
        "SGARBOSSA_POSITIVE",
        "BRUGADA_TYPE1",
        "EARLY_REPOL",
        "PERICARDITIS_PATTERN",
        "ELECTRICAL_ALTERNANS",
        "HYPERKALEMIA_PATTERN",
        "HYPOKALEMIA_PATTERN",
        "HYPOCALCEMIA_PATTERN",
        "HYPERCALCEMIA_PATTERN",
        "DIGITALIS_EFFECT",
    }
    for finding_id in sorted(checklist):
        findings.setdefault(
            finding_id,
            _unknown(
                finding_id,
                "finding_not_projected_by_measurement_layer",
                f"appendix-A.{finding_id}",
            ),
        )
    return dict(sorted(findings.items()))


def serialize_findings(findings: Mapping[str, Finding] | Iterable[Finding]) -> dict:
    if isinstance(findings, Mapping):
        items = findings.items()
    else:
        items = ((finding.id, finding) for finding in findings)
    return {key: finding.to_dict() for key, finding in items}

"""QT reportability policy: reject gate and rescue decisions.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..._engine.foundation.numeric import _finite_float
from ..._engine.measurement.features import compute_global_features
from ...record.availability import Measured, NotApplicable, Unavailable
from ..context import PipelineContext, PolicyDecision, PolicyEvent

_PACING_INTERMITTENT_QT_NATIVE_RESCUE_MAX_PACED_FRACTION = 0.80
_PACING_INTERMITTENT_QT_NATIVE_RESCUE_MIN_DELTA_MS = 40.0
_QRS_TAIL_QT_RESCUE_MIN_LEADS = 6
_QRS_TAIL_QT_RESCUE_MAX_SPREAD_MS = 25.0
_QRS_TAIL_QT_RESCUE_MIN_JT_MS = 180.0
_QRS_TAIL_QT_RESCUE_MAX_JT_MS = 500.0


def _copy_qt_measurements(target: Any, source: Any) -> None:
    for attr in (
        "qt_ms",
        "qtc_bazett_ms",
        "qtc_fridericia_ms",
        "qt_dispersion_ms",
        "qt_dispersion_independent_ms",
        "qt_dispersion_p90_p10_ms",
        "qt_dispersion_source",
        "qt_dispersion_used_leads",
        "qt_dispersion_legacy_excluded_leads",
        "qt_source",
        "qt_used_leads",
        "qt_reliability",
        "qt_reportable",
        "qt_unreliable_reasons",
        "qt_path",
        "qt_confidence_reason",
        "qt_excluded_leads",
        "qt_lead_weights",
        "consensus_vs_independent_per_lead",
        "qt_robust_center_ms",
        "qt_latest_p85_ms",
        "t_fusion_support",
        "t_fusion_mad_ms",
        "t_fusion_ci_half_width_ms",
        "t_fusion_reliable",
        "t_fusion_cluster_count",
        "t_fusion_selected_cluster_score",
        "t_fusion_selected_cluster_support",
        "t_fusion_lead_groups",
        "t_fusion_methods",
        "t_fusion_reliability_reasons",
        "t_global_tpte_ms",
        "t_derived_disagreement_ms",
        "t_tail_incomplete_leads",
        "t_tail_incomplete_fraction",
        "t_systematic_early_risk",
        "t_systematic_early_fraction",
        "t_morphology_guard_pass",
    ):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


_QT_REJECT_RELIABILITY_TIERS = ("fallback", "low_confidence")
_QT_REJECT_RECORD_GRADES = ("Q2", "Q3")


def _apply_qt_reject_gate(global_features: Any, record_grade: Optional[str]) -> None:
    """Null QT/QTc rather than report a number when the evidence is too weak.

    quality.py's diagnostic gate deliberately does not suppress measurements
    (it only gates diagnostic *statements*), so a record can reach here with
    qt_reliability already at its weakest tiers ("fallback" — a single
    preferred-lead value with no cross-validation, or "low_confidence" — too
    few reliable QT leads). That alone doesn't justify rejecting a number
    (an otherwise-clean record can still fall back for incidental reasons),
    but combined with a poor overall record grade (Q2/Q3: reduced reliable
    lead coverage or other technical issues) it is — reject and record a
    reason rather than emit a plausible-looking but unsupported QT/QTc.
    """
    reliability = getattr(global_features, "qt_reliability", "unavailable")
    if reliability not in _QT_REJECT_RELIABILITY_TIERS:
        return
    if record_grade not in _QT_REJECT_RECORD_GRADES:
        return
    global_features.qt_rejected = True
    global_features.qt_reject_reason = f"qt_reliability={reliability},record_grade={record_grade}"
    for attr in ("qt_ms", "qtc_bazett_ms", "qtc_fridericia_ms", "qtc_framingham_ms", "qtc_hodges_ms"):
        setattr(global_features, attr, None)



def _rescue_intermittent_paced_qt_from_native(
    global_features: Any,
    representative_leads: Dict[str, Any],
    measurement_beat_features: List[Any],
    r_locs: np.ndarray,
    fs: int,
    paced_fraction: float,
) -> Optional[str]:
    if paced_fraction >= _PACING_INTERMITTENT_QT_NATIVE_RESCUE_MAX_PACED_FRACTION:
        return None
    if str(getattr(global_features, "qt_source", "") or "") != "irregular_raw_beat_qt_core":
        return None
    current_qt = _finite_float(getattr(global_features, "qt_ms", None))
    native_global = compute_global_features(
        representative_leads,
        measurement_beat_features,
        r_locs,
        fs,
        paced=False,
    )
    native_qt = _finite_float(getattr(native_global, "qt_ms", None))
    if native_qt is None:
        return None
    if current_qt is not None and native_qt - current_qt < _PACING_INTERMITTENT_QT_NATIVE_RESCUE_MIN_DELTA_MS:
        return None
    _copy_qt_measurements(global_features, native_global)
    return str(getattr(native_global, "qt_source", None) or "native_global")



def _rescue_qt_after_qrs_tail_settling(
    *,
    global_features: Any,
    representative_leads: Dict[str, Any],
    r_locs: np.ndarray,
    fs: int,
    qrs_tail_settling_rescue: Dict[str, Any],
) -> Optional[str]:
    if (
        not bool(qrs_tail_settling_rescue.get("applied", False))
        or getattr(global_features, "qt_ms", None) is not None
        or fs <= 0
        or len(r_locs) < 2
    ):
        return None

    values: List[Tuple[str, float]] = []
    for lead, representative in representative_leads.items():
        params = getattr(representative, "params", {}) or {}
        if not bool(params.get("reliable_for_global", False)):
            continue
        value = _finite_float(params.get("qt_consensus_ms"))
        if value is None or not (250.0 <= value <= 650.0):
            continue
        values.append((lead, value))
    if len(values) < _QRS_TAIL_QT_RESCUE_MIN_LEADS:
        return None

    qt_values = np.asarray([value for _lead, value in values], dtype=float)
    if float(np.ptp(qt_values)) > _QRS_TAIL_QT_RESCUE_MAX_SPREAD_MS:
        return None
    qt_ms = float(np.median(qt_values))
    qrs_ms = _finite_float(getattr(global_features, "qrs_ms", None))
    if qrs_ms is None or not (
        _QRS_TAIL_QT_RESCUE_MIN_JT_MS
        <= qt_ms - qrs_ms
        <= _QRS_TAIL_QT_RESCUE_MAX_JT_MS
    ):
        return None

    rr_sec = float(np.median(np.diff(np.asarray(r_locs, dtype=float)))) / float(fs)
    if not np.isfinite(rr_sec) or rr_sec <= 0.0:
        return None
    global_features.qt_ms = qt_ms
    global_features.qtc_bazett_ms = float(qt_ms / np.sqrt(rr_sec))
    global_features.qtc_fridericia_ms = float(qt_ms / np.cbrt(rr_sec))
    global_features.qt_source = "qrs_tail_settling_consensus_rescue"
    global_features.qt_used_leads = [lead for lead, _value in values]
    global_features.qt_reliability = "rescued"
    global_features.qt_reportable = True
    global_features.qt_unreliable_reasons = []
    global_features.qt_path = "qrs_tail_settling_consensus"
    global_features.qt_confidence_reason = (
        "systematic_multilead_qrs_tail_and_t_end_consensus"
    )
    return "qrs_tail_settling_consensus_rescue"



@dataclass(frozen=True, slots=True)
class QTEvidence:
    path: Any = None
    candidates_ms: Mapping[str, float] = field(default_factory=dict)
    lead_candidates_ms: Mapping[str, float] = field(default_factory=dict)
    selected_group_id: int | None = None
    native_group_id: int | None = None
    pacing: Any = None
    tail_settling_candidate_ms: float | None = None
    reliability_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QTDecision:
    value: Any
    source: str | None
    used_leads: tuple[str, ...]
    excluded_leads: tuple[str, ...]
    events: tuple[PolicyEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class QTPolicy:
    name: str = "default_qt_rescue_policy"
    version: str = "1"

    def decide(self, evidence: QTEvidence, *, context: PipelineContext) -> PolicyDecision:
        raise NotImplementedError("The qt policy is still embedded in the legacy engine; independent decisions require the migration equivalence gate")


__all__ = ["QTEvidence", "QTDecision", "QTPolicy"]

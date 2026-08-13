from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


WIDE_QRS_MS = 120.0
WIDE_MIN_JT_MS = 160.0
WIDE_MAX_JT_MS = 550.0

_NORMAL_MIN_QT_MS = 240.0
_NORMAL_MAX_QT_MS = 700.0


@dataclass
class QTPathDecision:
    path: str
    reliability: str
    used_leads: List[str] = field(default_factory=list)
    excluded_leads: Dict[str, str] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    reason: Optional[str] = None
    consensus_vs_independent_per_lead: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)


def _finite_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def qt_allowed_for_path(qt_ms: Optional[float], qrs_ms: Optional[float], path: str) -> bool:
    qt = _finite_float(qt_ms)
    if qt is None:
        return False
    if path in {"wide_qrs_jt", "paced_jt"}:
        qrs = _finite_float(qrs_ms)
        if qrs is None:
            return False
        jt = qt - qrs
        return WIDE_MIN_JT_MS <= jt <= WIDE_MAX_JT_MS
    return _NORMAL_MIN_QT_MS <= qt <= _NORMAL_MAX_QT_MS


def classify_qt_path(
    qrs_ms: Optional[float],
    *,
    paced: bool = False,
    qrs_offset_confidence: Optional[float] = None,
) -> str:
    if paced:
        return "paced_jt"
    qrs = _finite_float(qrs_ms)
    if qrs is not None and qrs >= WIDE_QRS_MS:
        return "wide_qrs_jt"
    return "normal_qrs"


def build_qt_path_decision(
    qrs_ms: Optional[float],
    reliable_qt_leads: List[str],
    lead_qt_values: Dict[str, Optional[float]],
    lead_qrs_values: Dict[str, Optional[float]],
    *,
    lead_qt_consensus_values: Optional[Dict[str, Optional[float]]] = None,
    paced: bool = False,
    qrs_offset_confidence: Optional[float] = None,
) -> QTPathDecision:
    path = classify_qt_path(
        qrs_ms,
        paced=paced,
        qrs_offset_confidence=qrs_offset_confidence,
    )
    min_support = 2 if path in {"wide_qrs_jt", "paced_jt"} else 3

    used_leads: List[str] = []
    excluded_leads: Dict[str, str] = {}
    weights: Dict[str, float] = {}
    per_lead: Dict[str, Dict[str, Optional[float]]] = {}

    for lead, qt_value in lead_qt_values.items():
        qrs_value = lead_qrs_values.get(lead, qrs_ms)
        independent_qt = _finite_float(qt_value)
        consensus_qt = _finite_float((lead_qt_consensus_values or {}).get(lead))
        consensus_qrs = _finite_float(qrs_value)
        per_lead[lead] = {
            "qt_ms": independent_qt,
            "qt_consensus_ms": consensus_qt,
            "qrs_ms": consensus_qrs,
            "independent_qt_ms": independent_qt,
            "consensus_qrs_ms": consensus_qrs,
            "jt_ms": (
                independent_qt - consensus_qrs
                if independent_qt is not None and consensus_qrs is not None
                else None
            ),
        }

    for lead in reliable_qt_leads:
        qt_value = lead_qt_values.get(lead)
        qrs_value = lead_qrs_values.get(lead, qrs_ms)
        if not qt_allowed_for_path(qt_value, qrs_value, path):
            excluded_leads[lead] = "qt_outside_path_bounds"
            continue
        used_leads.append(lead)
        weights[lead] = 1.0

    if len(used_leads) < min_support:
        return QTPathDecision(
            path="low_qt_support",
            reliability="low_confidence" if used_leads else "unavailable",
            used_leads=used_leads,
            excluded_leads=excluded_leads,
            weights=weights,
            reason="insufficient_reliable_qt_leads",
            consensus_vs_independent_per_lead=per_lead,
        )

    return QTPathDecision(
        path=path,
        reliability="reliable",
        used_leads=used_leads,
        excluded_leads=excluded_leads,
        weights=weights,
        reason=None,
        consensus_vs_independent_per_lead=per_lead,
    )

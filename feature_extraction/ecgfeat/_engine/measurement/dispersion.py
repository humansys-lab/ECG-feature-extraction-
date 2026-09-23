"""Explicit QT dispersion estimands over quality-approved independent leads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class QTDispersion:
    independent_ms: float | None
    p90_p10_ms: float | None
    legacy_ms: float | None
    legacy_source: str
    used_leads: tuple[str, ...]
    legacy_excluded_leads: tuple[str, ...]


def summarize_qt_dispersion(values: Mapping[str, float]) -> QTDispersion:
    """Keep the historical estimator explicit while exposing unclipped ranges.

    The legacy compact-cluster estimate remains available for compatibility.
    It must not be described as the full independent-lead range. Its excluded
    leads are statistical exclusions, not additional waveform quality rejects.
    """
    ordered = sorted((float(value), lead) for lead, value in values.items()
                     if np.isfinite(value))
    leads = tuple(lead for _, lead in ordered)
    if len(ordered) < 2:
        return QTDispersion(None, None, None, "unavailable", leads, ())
    qt = np.asarray([value for value, _ in ordered])
    independent = float(np.ptp(qt))
    p10, p90 = np.percentile(qt, [10, 90])
    legacy = independent
    excluded: tuple[str, ...] = ()
    source = "independent_reliable_lead_range"
    if independent >= 80.0:
        clusters = [
            [item for item in ordered[index:] if item[0] - first[0] <= 45.0]
            for index, first in enumerate(ordered)
        ]
        clusters = [cluster for cluster in clusters if len(cluster) >= 3]
        if clusters:
            core = max(clusters, key=lambda c: (len(c), -(c[-1][0] - c[0][0])))
            legacy = float(core[-1][0] - core[0][0])
            included = {lead for _, lead in core}
            excluded = tuple(lead for lead in leads if lead not in included)
            source = "legacy_compact_cluster_range"
    return QTDispersion(independent, float(p90 - p10), legacy, source, leads, excluded)

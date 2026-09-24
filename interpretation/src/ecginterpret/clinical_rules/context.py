from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional

from ecgfeat.compat.models_v0 import ECGFeatures, resolve_patient_age


LIMB_LEADS = frozenset({"I", "II", "III", "aVR", "aVL", "aVF"})
PRECORDIAL_LEADS = frozenset({"V1", "V2", "V3", "V4", "V5", "V6"})


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _field(container: Any, name: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


@dataclass(frozen=True)
class ClinicalContext:
    features: ECGFeatures
    age_years: Optional[float]
    sex: str
    excluded_leads: frozenset[str]
    precordial_reversal: bool
    precordial_reversal_state: str
    precordial_reversal_leads: frozenset[str]
    limb_reversal: bool

    def global_value(self, name: str) -> Optional[float]:
        return finite_float(getattr(self.features.global_features, name, None))

    def lead_available(self, lead: str, reliability: str) -> bool:
        quality = self.features.quality.get(lead)
        return bool(
            lead not in self.excluded_leads
            and quality is not None
            and getattr(quality, reliability, False)
        )

    def lead_value(
        self,
        lead: str,
        name: str,
        reliability: str = "reliable_for_qrs",
    ) -> Optional[float]:
        if not self.lead_available(lead, reliability):
            return None
        representative = self.features.representative_leads.get(lead)
        params = getattr(representative, "params", {}) if representative is not None else {}
        return finite_float(params.get(name))

    def lead_raw_value(
        self,
        lead: str,
        name: str,
        reliability: str = "reliable_for_qrs",
    ) -> Any:
        if not self.lead_available(lead, reliability):
            return None
        representative = self.features.representative_leads.get(lead)
        params = getattr(representative, "params", {}) if representative is not None else {}
        return params.get(name)

    def lead_variance_value(
        self,
        lead: str,
        name: str,
        reliability: str = "reliable_for_qrs",
    ) -> Optional[float]:
        if not self.lead_available(lead, reliability):
            return None
        representative = self.features.representative_leads.get(lead)
        variance = (
            getattr(representative, "variance", {})
            if representative is not None
            else {}
        )
        return finite_float(variance.get(name))


def build_context(features: ECGFeatures) -> ClinicalContext:
    patient = features.metadata.get("patient_meta")
    age = resolve_patient_age(patient).age_years
    sex = str(_field(patient, "sex", "unknown") or "unknown").strip().lower()
    interpretation = features.interpretation
    lead_reversal = features.metadata.get("lead_reversal", {})
    lead_reversal = lead_reversal if isinstance(lead_reversal, dict) else {}
    precordial_detail = lead_reversal.get("precordial", {})
    precordial_detail = (
        precordial_detail if isinstance(precordial_detail, dict) else {}
    )
    legacy_precordial = bool(
        _field(interpretation, "precordial_reversal_suspected", False)
    )
    precordial_state = str(precordial_detail.get("state") or "").strip().lower()
    if precordial_state not in {"not_suspected", "possible", "confirmed"}:
        precordial_state = "possible" if legacy_precordial else "not_suspected"
    implicated = {
        str(lead)
        for lead in (precordial_detail.get("implicated_leads") or [])
        if str(lead) in PRECORDIAL_LEADS
    }
    precordial = precordial_state == "confirmed"
    limb = bool(_field(interpretation, "limb_reversal_suspected", None))
    excluded: set[str] = set()
    if precordial:
        excluded.update(implicated)
    if limb:
        excluded.update(LIMB_LEADS)
    return ClinicalContext(
        features=features,
        age_years=age,
        sex=sex,
        excluded_leads=frozenset(excluded),
        precordial_reversal=precordial,
        precordial_reversal_state=precordial_state,
        precordial_reversal_leads=frozenset(implicated),
        limb_reversal=limb,
    )

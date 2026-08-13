from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any, Dict, List, Optional

from ..models import ECGFeatures, STANDARD_12_LEADS
from .models import GlasgowConfig


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _field(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _patient_value(features: ECGFeatures, key: str) -> Any:
    patient = features.metadata.get("patient_meta")
    return _field(patient, key)


def _normalize_sex(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"f", "female", "woman", "girl", "2"}:
        return "female"
    if text in {"m", "male", "man", "boy", "1"}:
        return "male"
    return "unknown"


@dataclass(frozen=True)
class PatientRoute:
    age_years: Optional[float]
    age_days: Optional[float]
    age_days_source: str
    age_approximation: bool
    sex: Optional[str]
    sex_normalized: str
    age_missing: bool
    sex_missing: bool
    pediatric: bool
    algorithm_age_group: str
    qtc_demographic_route: str
    meds: tuple[str, ...]
    clinical_classifications: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["meds"] = list(self.meds)
        payload["clinical_classifications"] = list(self.clinical_classifications)
        return payload


@dataclass(frozen=True)
class LeadFacts:
    lead: str
    available: bool
    reliable_for_p: bool
    reliable_for_qrs: bool
    reliable_for_t: bool
    reliable_for_qt: bool
    measurements: Dict[str, Any]


@dataclass(frozen=True)
class GlasgowContext:
    features: ECGFeatures
    config: GlasgowConfig
    patient: PatientRoute
    global_measurements: Dict[str, Any]
    leads: Dict[str, LeadFacts]
    excluded_leads: Dict[str, List[str]]
    rhythm: Dict[str, Any]
    record_quality: Dict[str, Any]

    def with_exclusions(self, exclusions: Dict[str, List[str]]) -> "GlasgowContext":
        merged = {lead: list(reasons) for lead, reasons in self.excluded_leads.items()}
        for lead, reasons in exclusions.items():
            current = merged.setdefault(str(lead), [])
            for reason in reasons:
                if reason not in current:
                    current.append(str(reason))
        return replace(self, excluded_leads=merged)

    def global_value(self, name: str) -> Any:
        return self.global_measurements.get(name)

    def lead_value(self, lead: str, name: str) -> Any:
        fact = self.leads.get(lead)
        if fact is None or lead in self.excluded_leads:
            return None
        return fact.measurements.get(name)

    def lead_is_available(self, lead: str) -> bool:
        fact = self.leads.get(lead)
        return bool(fact is not None and fact.available and lead not in self.excluded_leads)


def _patient_route(features: ECGFeatures) -> PatientRoute:
    age_years = _finite(_patient_value(features, "age"))
    if age_years is not None and age_years < 0.0:
        age_years = None
    provided_days = _finite(_patient_value(features, "age_days"))
    if provided_days is not None and provided_days < 0.0:
        provided_days = None

    if provided_days is not None:
        age_days = provided_days
        age_days_source = "provided"
        age_approximation = False
        if age_years is None:
            age_years = provided_days / 365.25
    elif age_years is not None:
        age_days = age_years * 365.25
        age_days_source = "derived_from_age_years"
        age_approximation = True
    else:
        age_days = None
        age_days_source = "missing_adult_fallback"
        age_approximation = True

    raw_sex = _patient_value(features, "sex")
    normalized_sex = _normalize_sex(raw_sex)
    age_missing = age_years is None and provided_days is None
    sex_missing = normalized_sex == "unknown"
    # NOTE ON CROSS-ENGINE DIVERGENCE: 18 years matches this engine's own
    # source, GAN Physician's Guide section 4.3.6 ("the patient is under 18
    # years of age"). ../interpret.py's PEDS_MAX_AGE_YEARS uses 16 years
    # instead, per the Philips DXL pediatric morphology spec. The two vendor
    # spec families define "pediatric" differently; this is a known,
    # deliberate divergence, not a bug -- see the note next to
    # PEDS_MAX_AGE_YEARS in interpret.py.
    pediatric = bool(age_days is not None and age_days < 18.0 * 365.25)

    meds = tuple(str(item) for item in (_patient_value(features, "meds") or []) if item is not None)
    classifications = tuple(
        str(item)
        for item in (_patient_value(features, "clinical_classifications") or [])
        if item is not None
    )
    return PatientRoute(
        age_years=age_years,
        age_days=age_days,
        age_days_source=age_days_source,
        age_approximation=age_approximation,
        sex=None if raw_sex is None else str(raw_sex),
        sex_normalized=normalized_sex,
        age_missing=age_missing,
        sex_missing=sex_missing,
        pediatric=pediatric,
        algorithm_age_group="pediatric" if pediatric else "adult",
        qtc_demographic_route=(
            "sex_neutral_existing_dxl"
            if age_missing or sex_missing
            else "glasgow_age_sex_specific"
        ),
        meds=meds,
        clinical_classifications=classifications,
    )


def _lead_facts(features: ECGFeatures) -> tuple[Dict[str, LeadFacts], Dict[str, List[str]]]:
    facts: Dict[str, LeadFacts] = {}
    exclusions: Dict[str, List[str]] = {}
    for lead in STANDARD_12_LEADS:
        quality = features.quality.get(lead)
        representative = features.representative_leads.get(lead)
        measurements = dict(getattr(representative, "params", {}) or {})
        glasgow = measurements.get("glasgow_measurements")
        if isinstance(glasgow, dict):
            measurements.update(glasgow)

        missing = quality is None or bool(getattr(quality, "missing", False))
        reliable = bool(quality is not None and getattr(quality, "reliable", False))
        available = bool(not missing and reliable)
        reasons: List[str] = []
        if missing:
            reasons.append("lead_missing")
        elif not reliable:
            reasons.append("lead_quality_unreliable")
        if reasons:
            exclusions[lead] = reasons

        facts[lead] = LeadFacts(
            lead=lead,
            available=available,
            reliable_for_p=bool(quality is not None and getattr(quality, "reliable_for_p", False)),
            reliable_for_qrs=bool(quality is not None and getattr(quality, "reliable_for_qrs", False)),
            reliable_for_t=bool(quality is not None and getattr(quality, "reliable_for_t", False)),
            reliable_for_qt=bool(quality is not None and getattr(quality, "reliable_for_qt", False)),
            measurements=measurements,
        )
    return facts, exclusions


def _rhythm_facts(features: ECGFeatures, leads: Dict[str, LeadFacts]) -> Dict[str, Any]:
    analysis = features.metadata.get("rhythm_analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    availability = analysis.get("availability")
    availability = availability if isinstance(availability, dict) else {}
    rule_summary = analysis.get("rule_summary")
    rule_summary = rule_summary if isinstance(rule_summary, dict) else {}
    pacing = analysis.get("pacing_context")
    pacing = pacing if isinstance(pacing, dict) else {}
    preexcitation = rule_summary.get("preexcitation")
    preexcitation = preexcitation if isinstance(preexcitation, dict) else {}
    interpretation = features.interpretation

    primary = rule_summary.get("primary_statement")
    probable_af = bool(_field(interpretation, "probable_af", False))
    paced = bool(
        getattr(features.global_features, "paced_rhythm", False)
        or pacing.get("continuous_pacing")
    )
    p_wave_flag = bool(
        availability.get("atrial_rhythm_available", True)
        and any(fact.reliable_for_p and fact.measurements.get("p_amp_mv") is not None for fact in leads.values())
    )
    sinus = bool(
        primary in {"sinus_rhythm", "sinus_bradycardia", "sinus_tachycardia"}
        or (
            primary is None
            and _field(interpretation, "p_axis_normal") is True
            and not probable_af
            and not paced
        )
    )
    bundle = _field(interpretation, "bundle_branch_block")
    return {
        "primary_statement": primary,
        "sinus_rhythm": sinus,
        "p_wave_flag": p_wave_flag,
        "wpw_pattern": bool(_field(interpretation, "wpw_pattern", False) or preexcitation.get("wpw_pattern")),
        "probable_af": probable_af,
        "paced_rhythm": paced,
        "bundle_branch_block": bundle,
        "abnormal_ventricular_conduction": bool(
            bundle
            or (_finite(getattr(features.global_features, "qrs_ms", None)) or 0.0) >= 120.0
            or _field(interpretation, "non_sustained_vt", False)
        ),
        "dominant_qrs_available": bool(
            features.metadata.get("measurement_beat_ids", features.beats)
        ),
        "availability": dict(availability),
        "pacing_context": dict(pacing),
    }


def build_context(
    features: ECGFeatures,
    config: Optional[GlasgowConfig] = None,
) -> GlasgowContext:
    selected_config = config or GlasgowConfig()
    leads, exclusions = _lead_facts(features)
    global_features = features.global_features
    global_measurements = {
        name: getattr(global_features, name, None)
        for name in (
            "heart_rate_bpm",
            "atrial_rate_bpm",
            "pr_ms",
            "qrs_ms",
            "qt_ms",
            "qtc_bazett_ms",
            "qtc_fridericia_ms",
            "p_axis_deg",
            "qrs_axis_deg",
            "t_axis_deg",
            "st_axis_deg",
            "qt_dispersion_ms",
        )
    }
    record_quality = features.metadata.get("record_quality")
    record_quality = dict(record_quality) if isinstance(record_quality, dict) else {}
    return GlasgowContext(
        features=features,
        config=selected_config,
        patient=_patient_route(features),
        global_measurements=global_measurements,
        leads=leads,
        excluded_leads=exclusions,
        rhythm=_rhythm_facts(features, leads),
        record_quality=record_quality,
    )

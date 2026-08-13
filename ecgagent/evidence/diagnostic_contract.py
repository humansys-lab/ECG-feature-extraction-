"""Versioned allowlist contract for diagnosis-first evidence.

The upstream feature payload is intentionally broader than the independent
diagnostic surface.  New upstream sections do not cross this boundary by
default: they must be classified here and advance the contract version.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping


DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION = "ecgagent.diagnosis-evidence.v3"

# A morphology map exposes every field as a separate evidence atom.  That is
# ideal for exact lookup, but a six-lead conduction pathway used to spend 66
# atoms before the model had seen one complete lead.  These bundles are built
# *inside* the diagnosis boundary from the already scrubbed representative
# measurements; an upstream payload cannot inject or override them.  They are
# diagnosis-neutral values, merely packed by lead so one citation carries a
# coherent QRS observation.
QRS_MEASUREMENT_BUNDLE_FIELDS = (
    "qrs_ms",
    "q_duration_ms",
    "q_amp_mv",
    "q_r_ratio",
    "r_amp_mv",
    "r_prime_amp_mv",
    "s_amp_mv",
    "s_prime_amp_mv",
    "qrs_notch_count",
    "qrs_slur_flag",
    "vat_ms",
)

ALLOWED_TOP_LEVEL = frozenset(
    {
        "fs",
        "global_features",
        "representative_leads",
        "beat_features",
        "beats",
        "groups",
        "p_wave_assessments",
        "rhythm_inputs",
        "morphology_inputs",
        "quality",
        "metadata",
    }
)

ALLOWED_RHYTHM_SECTIONS = frozenset(
    {
        "aberrancy",
        "af_afl",
        "av_block",
        "background",
        "beats",
        "native_beat_profiles",
        "p_events",
        "pacing",
        "preexcitation",
        "record",
    }
)
ALLOWED_MORPHOLOGY_SECTIONS = frozenset(
    {"global", "leads", "measurement_profiles", "record"}
)
ALLOWED_PATIENT_META = frozenset(
    {
        "age",
        "age_days",
        "sex",
        "acquisition_time",
        "amplitude_unit",
        "device",
        "gain_uv_per_lsb",
        "adc_full_scale_mv",
        "filter_highpass_hz",
        "filter_lowpass_hz",
        "muscle_filter_enabled",
    }
)

# These are acquisition, measurement-routing, signal-quality and provenance
# fields. The clinical/rhythm-analysis mirrors are deliberately absent.
ALLOWED_METADATA_FIELDS = frozenset(
    {
        "acquisition_qc",
        "diagnostic_gate",
        "algorithm_version",
        "duration_sec",
        "global_measurement_paced",
        "global_p_duration",
        "global_qt",
        "ecgfeat_version",
        "feature_extraction_version",
        "initial_measurement_beat_ids",
        "initial_measurement_group_id",
        "input_contract",
        "input_fs",
        "internal_fs",
        "lead_order",
        "lead_reversal",
        "mains_frequency_hz",
        "measurement_beat_ids",
        "measurement_group_paced",
        "measurement_group_reselect_reason",
        "measurement_group_reselected",
        "measurement_pacing_state",
        "measurement_representative_beat_meta",
        "metadata_contract",
        "n_beats",
        "n_reliable_qt_leads",
        "p_wave_contract",
        "paced_beat_fraction",
        "paced_beat_ids",
        "pacing_capture_confirmed",
        "pacing_detection_state",
        "pacing_evidence_by_beat",
        "pacing_evidence_quality",
        "pacing_measurement_effect",
        "pacing_qrs_rescue",
        "pacing_qt_rescue_source",
        "pacing_rescue_applied",
        "pacing_rescue_capture_fraction",
        "pacing_rescue_prominence_uv",
        "pacing_rescue_reason",
        "pacing_segmentation_effect",
        "pacing_spike_beat_ids",
        "pacing_spike_offsets_ms",
        "pacing_state",
        "patient_meta",
        "qrs_detector",
        "qrs_detector_agreement",
        "qrs_tail_settling_rescue",
        "quality_stages",
        "r_localization",
        "record_quality",
        "reliable_qt_leads",
        "representative_beat_meta",
        "representative_group_id",
        "st_measurement",
        "t_axis_profile_backfill",
        "t_wave_refinement",
        "twelve_sl_measurement_profile",
        "wave_localization",
    }
)

_FORBIDDEN_KEYS = frozenset(
    {
        "clinical_interpretation",
        "interpretation",
        "statement_engine",
        "statement_evidence",
        "reference_metadata",
        "glasgow_analysis",
        "glasgow_measurements",
        "rhythm_analysis",
        "clinical_classifications",
        "confirmed_pacing_context",
        "meds",
        "clinical_dx_codes",
        "rx_codes",
        "derived_facts",
        "p_state",
        "probable_af",
        "probable_flutter",
        "atrial_rhythm_classification",
        "diagnostic_confidence",
        "wpw_pattern",
        "accessory_pathway_side",
        "second_degree_avb",
    }
)


@dataclass(frozen=True)
class ContractAudit:
    version: str
    dropped_top_level: tuple[str, ...]
    unclassified_top_level: tuple[str, ...]
    dropped_metadata: tuple[str, ...]
    dropped_rhythm_sections: tuple[str, ...]
    dropped_morphology_sections: tuple[str, ...]
    dropped_sensitive_fields: tuple[str, ...]
    unclassified_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dropped_top_level": list(self.dropped_top_level),
            "unclassified_top_level": list(self.unclassified_top_level),
            "dropped_metadata": list(self.dropped_metadata),
            "dropped_rhythm_sections": list(self.dropped_rhythm_sections),
            "dropped_morphology_sections": list(
                self.dropped_morphology_sections
            ),
            "dropped_sensitive_fields": list(self.dropped_sensitive_fields),
            "unclassified_fields": list(self.unclassified_fields),
        }


def _scrub(
    node: Any,
    *,
    path: str = "",
    dropped: list[str] | None = None,
) -> Any:
    if isinstance(node, Mapping):
        result: dict[str, Any] = {}
        for key, value in node.items():
            name = str(key)
            child_path = f"{path}/{name}" if path else f"/{name}"
            if name in _FORBIDDEN_KEYS:
                if dropped is not None:
                    dropped.append(child_path)
                continue
            result[name] = _scrub(
                value,
                path=child_path,
                dropped=dropped,
            )
        return result
    if isinstance(node, list):
        return [
            _scrub(value, path=f"{path}/{index}", dropped=dropped)
            for index, value in enumerate(node)
        ]
    if isinstance(node, tuple):
        return [
            _scrub(value, path=f"{path}/{index}", dropped=dropped)
            for index, value in enumerate(node)
        ]
    return copy.deepcopy(node)


def build_diagnostic_document(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], ContractAudit]:
    """Construct the physical diagnosis DTO from explicit allowlists."""

    source_keys = set(str(key) for key in source)
    known_blocked = source_keys & _FORBIDDEN_KEYS
    unknown = source_keys - ALLOWED_TOP_LEVEL - _FORBIDDEN_KEYS
    document: dict[str, Any] = {}
    dropped_sensitive: list[str] = [f"/{key}" for key in known_blocked]
    for key in sorted(ALLOWED_TOP_LEVEL - {"metadata", "rhythm_inputs", "morphology_inputs"}):
        if key in source:
            document[key] = _scrub(
                source[key],
                path=f"/{key}",
                dropped=dropped_sensitive,
            )

    metadata_source = source.get("metadata")
    metadata_source = metadata_source if isinstance(metadata_source, Mapping) else {}
    dropped_sensitive.extend(
        f"/metadata/{key}"
        for key in map(str, metadata_source)
        if key in _FORBIDDEN_KEYS
    )
    metadata = {
        str(key): _scrub(
            value,
            path=f"/metadata/{key}",
            dropped=dropped_sensitive,
        )
        for key, value in metadata_source.items()
        if str(key) in ALLOWED_METADATA_FIELDS
    }
    patient = metadata_source.get("patient_meta")
    patient = patient if isinstance(patient, Mapping) else {}
    dropped_sensitive.extend(
        f"/metadata/patient_meta/{key}"
        for key in map(str, patient)
        if key in _FORBIDDEN_KEYS
    )
    metadata["patient_meta"] = {
        str(key): _scrub(
            value,
            path=f"/metadata/patient_meta/{key}",
            dropped=dropped_sensitive,
        )
        for key, value in patient.items()
        if str(key) in ALLOWED_PATIENT_META
    }
    document["metadata"] = metadata

    rhythm_source = source.get("rhythm_inputs")
    rhythm_source = rhythm_source if isinstance(rhythm_source, Mapping) else {}
    dropped_sensitive.extend(
        f"/rhythm_inputs/{key}"
        for key in map(str, rhythm_source)
        if key in _FORBIDDEN_KEYS
    )
    rhythm_document = {
        str(key): _scrub(
            value,
            path=f"/rhythm_inputs/{key}",
            dropped=dropped_sensitive,
        )
        for key, value in rhythm_source.items()
        if str(key) in ALLOWED_RHYTHM_SECTIONS
    }
    # Upstream availability can be disabled because a hidden rule-like
    # classifier fired (for example ``probable_flutter``). Keep the neutral
    # availability booleans, never the diagnosis-derived reason string.
    record = rhythm_document.get("record")
    if isinstance(record, dict):
        availability = record.get("availability")
        if isinstance(availability, dict) and "reasons" in availability:
            availability.pop("reasons", None)
            dropped_sensitive.append(
                "/rhythm_inputs/record/availability/reasons"
            )
    document["rhythm_inputs"] = rhythm_document
    morphology_source = source.get("morphology_inputs")
    morphology_source = (
        morphology_source if isinstance(morphology_source, Mapping) else {}
    )
    dropped_sensitive.extend(
        f"/morphology_inputs/{key}"
        for key in map(str, morphology_source)
        if key in _FORBIDDEN_KEYS
    )
    document["morphology_inputs"] = {
        str(key): _scrub(
            value,
            path=f"/morphology_inputs/{key}",
            dropped=dropped_sensitive,
        )
        for key, value in morphology_source.items()
        if str(key) in ALLOWED_MORPHOLOGY_SECTIONS
    }

    # Derive this only from the scrubbed DTO.  ``measurement_bundles`` is
    # deliberately not in ALLOWED_TOP_LEVEL, so a same-named source section is
    # reported as unclassified and cannot cross the contract boundary.
    qrs_by_lead: dict[str, dict[str, Any]] = {}
    representative_leads = document.get("representative_leads")
    if isinstance(representative_leads, Mapping):
        for lead, row in representative_leads.items():
            params = row.get("params") if isinstance(row, Mapping) else None
            if not isinstance(params, Mapping):
                continue
            bundle = {
                field: copy.deepcopy(params[field])
                for field in QRS_MEASUREMENT_BUNDLE_FIELDS
                if field in params
            }
            if bundle:
                qrs_by_lead[str(lead)] = bundle
    document["measurement_bundles"] = {"qrs_by_lead": qrs_by_lead}

    audit = ContractAudit(
        version=DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION,
        dropped_top_level=tuple(sorted(known_blocked | unknown)),
        unclassified_top_level=tuple(sorted(unknown)),
        dropped_metadata=tuple(
            sorted(set(map(str, metadata_source)) - ALLOWED_METADATA_FIELDS)
        ),
        dropped_rhythm_sections=tuple(
            sorted(set(map(str, rhythm_source)) - ALLOWED_RHYTHM_SECTIONS)
        ),
        dropped_morphology_sections=tuple(
            sorted(set(map(str, morphology_source)) - ALLOWED_MORPHOLOGY_SECTIONS)
        ),
        dropped_sensitive_fields=tuple(sorted(set(dropped_sensitive))),
        unclassified_fields=unclassified_source_fields(source),
    )
    return document, audit


def unclassified_source_fields(source: Mapping[str, Any]) -> tuple[str, ...]:
    """Contract-test hook: new high-risk families require explicit review."""

    unknown = {
        f"/{key}"
        for key in set(map(str, source)) - ALLOWED_TOP_LEVEL - _FORBIDDEN_KEYS
    }
    metadata = source.get("metadata")
    if isinstance(metadata, Mapping):
        unknown.update(
            f"/metadata/{key}"
            for key in set(map(str, metadata)) - ALLOWED_METADATA_FIELDS - _FORBIDDEN_KEYS
        )
        patient = metadata.get("patient_meta")
        if isinstance(patient, Mapping):
            unknown.update(
                f"/metadata/patient_meta/{key}"
                for key in set(map(str, patient)) - ALLOWED_PATIENT_META - _FORBIDDEN_KEYS
            )
    rhythm = source.get("rhythm_inputs")
    if isinstance(rhythm, Mapping):
        unknown.update(
            f"/rhythm_inputs/{key}"
            for key in set(map(str, rhythm)) - ALLOWED_RHYTHM_SECTIONS - _FORBIDDEN_KEYS
        )
    morphology = source.get("morphology_inputs")
    if isinstance(morphology, Mapping):
        unknown.update(
            f"/morphology_inputs/{key}"
            for key in set(map(str, morphology))
            - ALLOWED_MORPHOLOGY_SECTIONS
            - _FORBIDDEN_KEYS
        )
    return tuple(sorted(unknown))


__all__ = [
    "ALLOWED_TOP_LEVEL",
    "ContractAudit",
    "DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION",
    "QRS_MEASUREMENT_BUNDLE_FIELDS",
    "build_diagnostic_document",
    "unclassified_source_fields",
]

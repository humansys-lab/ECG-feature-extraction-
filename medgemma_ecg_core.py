from __future__ import annotations

import gzip
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from statistics import median, pstdev
from types import SimpleNamespace
from typing import Any, Callable

from feature_extraction.ecgfeat.compat.export_v0 import (
    _PEDS_QRS_NORMAL_MS,
    _peds_lookup,
    _peds_qrs_width_class,
    _peds_qtc_limits,
)
from ecginterpret.glasgow_rules.intervals import _pr_limits as _glasgow_pr_limits
from ecginterpret.glasgow_rules.models import GlasgowConfig
from ecginterpret.glasgow_rules.rate import bradycardia_limit, tachycardia_limit
from ecginterpret.interpret import _peds_classify_qrs_axis
from ecginterpret.pediatric_rules import (
    build_pediatric_hypertrophy_evidence,
    pediatric_age_bin,
    pediatric_voltage_threshold,
)


STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
REPORT_CHAR_LIMIT = 6000
REPORT_LAYER_CHAR_LIMIT = 2200
# Current ECG prompts are about 4k tokens; 8192 leaves enough room for a
# complete structured diagnostic response without over-allocating KV cache.
DEFAULT_MODEL_MAX_LEN = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DIAGNOSTIC_GATE_ENFORCEMENT_VERSION = "diagnostic_gate.v3"
MEDGEMMA_REASONING_PROTOCOL_VERSION = "json_report_no_dx_sequential_v4"

_PEDIATRIC_MORPHOLOGY_MAX_AGE_YEARS = 16.0
_RATE_ADULT_AGE_DAYS = 18.0 * 365.25
# Preserve the MedGemma adult 60/100 bpm convention while reusing the
# project's continuous Glasgow pediatric curves.
_MEDGEMMA_RATE_CONFIG = GlasgowConfig(
    adult_tachycardia_bpm=100.0,
    adult_bradycardia_bpm=60.0,
)

CANONICAL_LABELS = {
    "_avb": "atrioventricular block",
    "ami": "acute myocardial infarction",
    "clbbb": "complete left bundle branch block",
    "crbbb": "complete right bundle branch block",
    "ilbbb": "incomplete left bundle branch block",
    "imi": "inferior myocardial infarction",
    "irbbb": "incomplete right bundle branch block",
    "isc_": "ischemic st-t changes",
    "isca": "anterior ischemia",
    "isci": "inferior ischemia",
    "ivcd": "intraventricular conduction delay",
    "lafb_lpfb": "left fascicular block",
    "lao_lae": "left atrial abnormality",
    "lmi": "lateral myocardial infarction",
    "lvh": "left ventricular hypertrophy",
    "norm": "normal ecg",
    "nst_": "nonspecific st-t abnormality",
    "pmi": "posterior myocardial infarction",
    "rao_rae": "right atrial abnormality",
    "rvh": "right ventricular hypertrophy",
    "sehyp": "septal hypertrophy",
    "sttc": "st-t change",
    "wpw": "wolff-parkinson-white pattern",
}

DIAGNOSIS_ALIASES = {
    "ami": "acute myocardial infarction",
    "acute mi": "acute myocardial infarction",
    "acute myocardial infarction": "acute myocardial infarction",
    "myocardial infarction": "acute myocardial infarction",
    "acute coronary syndrome with infarction": "acute myocardial infarction",
    "crbbb": "complete right bundle branch block",
    "complete right bundle branch block": "complete right bundle branch block",
    "right bundle branch block": "complete right bundle branch block",
    "clbbb": "complete left bundle branch block",
    "complete left bundle branch block": "complete left bundle branch block",
    "left bundle branch block": "complete left bundle branch block",
    "irbbb": "incomplete right bundle branch block",
    "incomplete right bundle branch block": "incomplete right bundle branch block",
    "ilbbb": "incomplete left bundle branch block",
    "incomplete left bundle branch block": "incomplete left bundle branch block",
    "norm": "normal ecg",
    "normal ecg": "normal ecg",
    "normal sinus rhythm": "normal ecg",
    "sinus rhythm": "normal ecg",
    "lvh": "left ventricular hypertrophy",
    "left ventricular hypertrophy": "left ventricular hypertrophy",
    "rvh": "right ventricular hypertrophy",
    "right ventricular hypertrophy": "right ventricular hypertrophy",
    "ischemia": "ischemia",
    "anterior ischemia": "anterior ischemia",
    "inferior ischemia": "inferior ischemia",
    "st t change": "st-t change",
    "st-t change": "st-t change",
    "nonspecific st t abnormality": "nonspecific st-t abnormality",
    "nonspecific st-t abnormality": "nonspecific st-t abnormality",
    "old myocardial infarction": "old myocardial infarction",
    "posterior myocardial infarction": "posterior myocardial infarction",
    "inferior myocardial infarction": "inferior myocardial infarction",
    "lateral myocardial infarction": "lateral myocardial infarction",
    "atrioventricular block": "atrioventricular block",
    "av block": "atrioventricular block",
    "wpw": "wolff-parkinson-white pattern",
    "wolff parkinson white pattern": "wolff-parkinson-white pattern",
    "intraventricular conduction delay": "intraventricular conduction delay",
    "ivcd": "intraventricular conduction delay",
    "left atrial abnormality": "left atrial abnormality",
    "left atrial enlargement": "left atrial abnormality",
    "right atrial abnormality": "right atrial abnormality",
    "right atrial enlargement": "right atrial abnormality",
    "left fascicular block": "left fascicular block",
    "left anterior fascicular block": "left fascicular block",
    "left posterior fascicular block": "left fascicular block",
    "ischemic st t changes": "ischemic st-t changes",
    "ischemic st-t changes": "ischemic st-t changes",
    "septal hypertrophy": "septal hypertrophy",
}


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fmt_num(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return ("Yes" if value else "No") + unit
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _fmt_bool(value: Any, language: str = "en") -> str:
    if value is None:
        return "Unknown" if language == "en" else "不明"
    yes = "Yes" if language == "en" else "はい"
    no = "No" if language == "en" else "いいえ"
    return yes if value else no


def _fmt_list(values: list[Any] | tuple[Any, ...] | None, empty: str = "—") -> str:
    if not values:
        return empty
    return ", ".join(str(v) for v in values)


def _fmt_flagged_leads(quality: dict[str, Any], key: str) -> str:
    leads = [lead for lead, info in quality.items() if not info.get(key, True)]
    return _fmt_list(leads)


def _fmt_lead_map(mapping: dict[str, Any] | None, digits: int = 3, unit: str = " mV") -> str:
    if not mapping:
        return "—"
    parts = []
    for lead, value in mapping.items():
        if isinstance(value, (int, float)):
            parts.append(f"{lead}:{value:+.{digits}f}{unit}")
        else:
            parts.append(f"{lead}:{value}")
    return "; ".join(parts)


def is_current_features_schema(data: dict[str, Any]) -> bool:
    return isinstance(data, dict) and "global_features" in data and "interpretation" in data


def is_legacy_input_schema(data: dict[str, Any]) -> bool:
    legacy_keys = {"心率 (HR)", "R-R 间期 (RRI)", "P波形态与关系", "PR 间期", "QRS 波群"}
    return isinstance(data, dict) and any(key in data for key in legacy_keys)


def summarize_legacy_input(data: dict[str, Any], language: str = "en") -> str:
    header = "[Legacy ECG Input]" if language == "en" else "[旧形式 ECG 入力]"
    lines = [
        header,
        f"- Heart rate: {data.get('心率 (HR)', 'N/A')}",
        f"- R-R interval: {data.get('R-R 间期 (RRI)', 'N/A')}",
        f"- P-wave morphology/relationship: {data.get('P波形态与关系', 'N/A')}",
        f"- PR interval: {data.get('PR 间期', 'N/A')}",
        f"- QRS complex: {data.get('QRS 波群', 'N/A')}",
    ]
    return "\n".join(lines)


# Section headers in the generated report look like "  ───  NAME  ───...";
# table/row separators inside a section are dash-only ("  ──── ──── ────"),
# so requiring a letter right after the first dash-whitespace gap tells them apart.
_SECTION_HEADER_RE = re.compile(r"^\s*─+\s+[A-Za-z]")

# These sections contain pre-computed diagnostic conclusions from a separate
# rule engine (not raw measurements) — e.g. "CLIN-CONDUCTION-LAFB-01: ECG
# pattern consistent with left anterior fascicular block" or a fully worked
# "Step 1..5" walkthrough ending in a STEMI code. Feeding these to a model
# that is supposed to derive the diagnosis itself is answer leakage just like
# Dx Codes, so they are dropped even though they don't start with a simple
# line prefix.
_DROP_SECTION_MARKERS = (
    "UNIFIED CLINICAL INTERPRETATION",
    "DXL-INSPIRED REFERENCE DETAIL",
    # Backward compatibility: old reports may still contain this retired
    # interpretation/measurement appendix. Never pass it to the model.
    "GLASGOW MEASUREMENT MATRIX",
)


def sanitize_report_text(report_text: str, char_limit: int = REPORT_CHAR_LIMIT) -> str:
    """Return only diagnosis-free report measurements.

    Generated reports put the dataset record id in the document title as well
    as in the ``Record ID`` row.  Prefix-only filtering therefore is not a
    sufficient de-identification boundary.  When a structured report contains
    section headers, everything before the first section is treated as an
    untrusted title/preamble and discarded.  The field checks below are
    deliberately case-insensitive so alternative renderers cannot reintroduce
    a label via ``diagnosis:``, ``record id:``, etc.
    """
    cleaned_lines = []
    drop_fields = (
        "dx",
        "dx code",
        "dx codes",
        "dx detail",
        "diagnosis",
        "diagnostic label",
        "disease",
        "disease label",
        "ground truth",
        "ground truth label",
        "record",
        "record id",
        "record identifier",
        "sample id",
        "patient id",
        "hx",
        "sx",
        # Treatment fields can identify the target diagnosis as directly as a
        # label.  They are clinical context, not ECG measurements.
        "rx",
        "med",
        "meds",
        "medication",
        "medications",
        "medication list",
        "prescription",
        "prescriptions",
        "drug",
        "drugs",
        "drug therapy",
        "treatment",
        "treatments",
        "therapy",
        "therapies",
        "indication",
        "indications",
        "clinical history",
        "medical history",
        "past medical history",
        "pmh",
        "problem list",
        "condition",
        "conditions",
        "chief complaint",
        "presenting complaint",
        # This is an extractor diagnosis rather than a measurement.  Keep the
        # numeric AF/AFL evidence and F-wave validation in the rhythm section,
        # but withhold the already-decided class from independent reasoning.
        "atrial classification",
    )
    drop_substrings = (
        "disease=",
        "disease_label=",
        "ground_truth=",
        "ground_truth_label=",
        "dataset=",
        "source=",
        "batch_id=",
        "record_id=",
        "sample_id=",
    )
    drop_prefix_fields = set(drop_fields) - {"record"}
    treatment_field_tokens = {
        "rx",
        "med",
        "meds",
        "medication",
        "medications",
        "drug",
        "drugs",
        "prescription",
        "prescriptions",
        "therapy",
        "therapies",
        "treatment",
        "treatments",
        "history",
        "indication",
        "indications",
    }

    has_sections = any(_SECTION_HEADER_RE.match(line) for line in report_text.splitlines())
    reached_first_section = not has_sections

    skipping_section = False
    skipping_wrapped_field = False
    for line in report_text.splitlines():
        stripped = line.strip()
        is_section_header = bool(_SECTION_HEADER_RE.match(line))
        if is_section_header:
            reached_first_section = True
        if not reached_first_section:
            # Titles are presentation metadata, not ECG measurements.  In the
            # canonical report this is where ``... REPORT — <record_id>`` lives.
            continue
        if is_section_header and any(
            marker.casefold() in line.casefold() for marker in _DROP_SECTION_MARKERS
        ):
            skipping_section = True
            continue
        if skipping_section:
            if is_section_header:
                skipping_section = False
            else:
                continue
        if skipping_wrapped_field:
            # A dropped "Field : value" line can wrap onto indented
            # continuation lines with no " : " of their own (e.g. Dx Detail's
            # parenthetical continuation) — those must be dropped too, or the
            # label text leaks anyway.
            if not stripped or re.search(r"[:=]", line) or is_section_header:
                skipping_wrapped_field = False
            else:
                continue
        normalized = stripped.lstrip("#").strip().casefold()
        field_name = re.split(r"\s*[:=]\s*", normalized, maxsplit=1)[0].strip()
        field_tokens = set(re.findall(r"[a-z]+", field_name))
        if (
            field_name in drop_fields
            or any(field_name.startswith(field + " ") for field in drop_prefix_fields)
            or bool(field_tokens & treatment_field_tokens)
        ):
            skipping_wrapped_field = True
            continue
        if re.match(r"^(?:ecg\b.*\breport\b|report\b)", normalized):
            # Also protects short/bare reports which do not have section
            # headers but still embed their identifier in the title.
            continue
        if "note: this report is generated" in normalized:
            continue
        if any(token in normalized for token in drop_substrings):
            continue
        cleaned_lines.append(line.rstrip())

    cleaned = "\n".join(cleaned_lines).strip()
    if len(cleaned) > char_limit:
        cleaned = cleaned[:char_limit].rstrip() + "\n...[report truncated]"
    return cleaned


_REPORT_SECTIONS_BY_LAYER = {
    # Demographics come from the structured allow-list in ``_patient_context``.
    # Do not route the free-form patient section, where a renderer could add a
    # medication, history, or indication field that bypasses the sanitizer.
    "L0": ("SIGNAL QUALITY", "BEAT-LEVEL QUALITY SUMMARY"),
    "L1": ("GLOBAL MEASUREMENTS", "RHYTHM SUMMARY"),
    "L2": ("GLOBAL MEASUREMENTS", "PER-LEAD MEASUREMENTS"),
    "L3": ("GLOBAL MEASUREMENTS",),
    "L4": ("PER-LEAD MEASUREMENTS",),
}


def summarize_report_by_layer(
    sanitized_report: str, char_limit: int = REPORT_LAYER_CHAR_LIMIT
) -> dict[str, str]:
    """Route diagnosis-free report measurements to the matching stage.

    The readable report duplicates some JSON measurements through a separate
    formatting path.  It is useful as a cross-check, but diagnostic sections
    and the original patient Dx have already been removed by
    :func:`sanitize_report_text`.
    """
    if not sanitized_report.strip():
        return {key: "" for key in LAYER_KEYS}

    sections: dict[str, list[str]] = {}
    current = "PREAMBLE"
    for line in sanitized_report.splitlines():
        if _SECTION_HEADER_RE.match(line):
            current = line.strip(" ─").strip()
            sections.setdefault(current, []).append(line.rstrip())
        else:
            sections.setdefault(current, []).append(line.rstrip())

    routed: dict[str, str] = {}
    for layer_key, wanted_markers in _REPORT_SECTIONS_BY_LAYER.items():
        chunks = []
        for title, lines in sections.items():
            if any(marker in title for marker in wanted_markers):
                chunks.append("\n".join(lines).strip())
        text = "\n\n".join(chunk for chunk in chunks if chunk).strip()
        if len(text) > char_limit:
            text = text[:char_limit].rstrip() + "\n...[stage report excerpt truncated]"
        routed[layer_key] = text
    return routed


def _summarize_groups(groups: dict[str, Any], language: str = "en") -> str:
    if not groups:
        return "—"

    lines = []
    for group_id in sorted(groups, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        group = groups[group_id]
        pieces = [
            f"G{group.get('group_id', group_id)}",
            f"{group.get('member_count', 'N/A')} beats",
            f"{_fmt_num(group.get('member_pct'), 0, '%')}",
            f"mean RR {_fmt_num(group.get('mean_rr_ms'), 0, ' ms')}",
            f"mean QRS {_fmt_num(group.get('mean_qrs_ms'), 0, ' ms')}",
        ]
        if group.get("flags", {}).get("dominant_group"):
            pieces.append("dominant")
        lines.append(" | ".join(pieces))
    return "\n".join(f"- {line}" for line in lines)


def _summarize_representative_leads(representative_leads: dict[str, Any], quality: dict[str, Any]) -> str:
    if not representative_leads:
        return "—"

    lines = []
    for lead in STANDARD_12_LEADS:
        lead_info = representative_leads.get(lead, {})
        params = lead_info.get("params", {})
        q = quality.get(lead, {})
        reliability = "OK" if q.get("reliable", True) else f"BAD ({_fmt_list(q.get('flags', []))})"
        lines.append(
            f"- {lead:<3} | {reliability:<18} | "
            f"PR {_fmt_num(params.get('pr_ms'), 0, ' ms'):>8} | "
            f"QRS {_fmt_num(params.get('qrs_ms'), 0, ' ms'):>8} | "
            f"QT {_fmt_num(params.get('qt_ms'), 0, ' ms'):>8} | "
            f"R {_fmt_num(params.get('r_amp_mv'), 3, ' mV'):>10} | "
            f"T {_fmt_num(params.get('t_amp_mv'), 3, ' mV'):>10} | "
            f"ST-J {_fmt_num(params.get('st_on_mv'), 3, ' mV'):>10}"
        )
    return "\n".join(lines)


def summarize_clinical_interpretation(
    clinical: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(clinical, dict) or not clinical:
        return []
    statements = [
        item.get("statement") or item.get("statement_code")
        for item in clinical.get("final_statements", [])
        if isinstance(item, dict)
    ]
    conflicts = [
        f"{item.get('reference')}: {item.get('reason')}"
        for item in clinical.get("conflicts", [])
        if isinstance(item, dict)
    ]
    return [
        f"- Unified clinical summary: {clinical.get('overall_status', 'unavailable')}",
        f"- Unified final statements: {_fmt_list(statements)}",
        f"- Unified unavailable domains: {_fmt_list(clinical.get('unavailable_domains'))}",
        f"- Reference conflicts: {_fmt_list(conflicts)}",
    ]


def summarize_current_features(data: dict[str, Any], language: str = "en") -> str:
    """Render the legacy display summary, including automated interpretations.

    This function is not an independent-evaluation input.  Evaluation callers
    must use :func:`summarize_independent_measurements`; the context builder
    enforces that separation.
    """
    gf = data.get("global_features", {})
    interp = data.get("interpretation", {})
    groups = data.get("groups", {})
    quality = data.get("quality", {})
    representative_leads = data.get("representative_leads", {})
    metadata = _metadata_mapping(data)
    patient = metadata.get("patient_meta", {})
    clinical_lines = summarize_clinical_interpretation(
        data.get("clinical_interpretation")
        or metadata.get("clinical_interpretation")
    )
    unreliable_leads = [
        f"{lead}({_fmt_list(info.get('flags', []))})"
        for lead, info in quality.items()
        if not info.get("reliable", True)
    ]

    summary_lines = [
        "[Structured ECG Feature Summary]",
        (
            f"- Patient: Age {patient.get('age', 'N/A')} | Sex {patient.get('sex', 'N/A')} | "
            f"input_fs {metadata.get('input_fs', 'N/A')} Hz | internal_fs {metadata.get('internal_fs', 'N/A')} Hz | "
            f"detected beats {metadata.get('n_beats', 'N/A')}"
        ),
        (
            f"- Global measurements: HR {_fmt_num(gf.get('heart_rate_bpm'), 1, ' bpm')}, "
            f"Atrial rate {_fmt_num(gf.get('atrial_rate_bpm'), 1, ' bpm')}, "
            f"PR {_fmt_num(gf.get('pr_ms'), 0, ' ms')}, "
            f"QRS {_fmt_num(gf.get('qrs_ms'), 0, ' ms')}, "
            f"QT {_fmt_num(gf.get('qt_ms'), 0, ' ms')}, "
            f"QTcB {_fmt_num(gf.get('qtc_bazett_ms'), 0, ' ms')}, "
            f"QTcF {_fmt_num(gf.get('qtc_fridericia_ms'), 0, ' ms')}, "
            f"QT dispersion {_fmt_num(gf.get('qt_dispersion_ms'), 0, ' ms')}"
        ),
        (
            f"- Axes: P {_fmt_num(gf.get('p_axis_deg'), 0, '°')}, "
            f"QRS {_fmt_num(gf.get('qrs_axis_deg'), 0, '°')}, "
            f"T {_fmt_num(gf.get('t_axis_deg'), 0, '°')}, "
            f"ST {_fmt_num(gf.get('st_axis_deg'), 0, '°')}"
        ),
        *clinical_lines,
        (
            f"- Rhythm/interval interpretation: RR {interp.get('rr_irregularity_class', 'N/A')} "
            f"(CV={_fmt_num(interp.get('rr_cv'), 4)}), probable AF={_fmt_bool(interp.get('probable_af'), language)}, "
            f"HR class={interp.get('heart_rate_class', 'N/A')}, PR class={interp.get('pr_class', 'N/A')}, "
            f"QRS width={interp.get('qrs_width_class', 'N/A')}, QTc={interp.get('qtc_class', 'N/A')}, "
            f"AV block={interp.get('avb_grade') or 'none'}, BBB={interp.get('bundle_branch_block') or 'none'}, "
            f"WPW={_fmt_bool(interp.get('wpw_pattern'), language)}"
        ),
        (
            f"- Morphology interpretation: P morphology={interp.get('p_morphology_class') or '—'}, "
            f"LAE suspected={_fmt_bool(interp.get('lae_suspected'), language)}, "
            f"LAE definite={_fmt_bool(interp.get('lae_definite'), language)}, "
            f"PTF-V1={interp.get('ptf_v1_class') or '—'}, "
            f"Q-wave territories={_fmt_list(interp.get('q_wave_territories'))}, "
            f"R progression={interp.get('r_progression_class') or '—'}, "
            f"RS transition={interp.get('r_s_transition_lead') or '—'}"
        ),
        (
            f"- ST/T: ST elevation={_fmt_lead_map(interp.get('st_elevation_leads'))}, "
            f"ST depression={_fmt_lead_map(interp.get('st_depression_leads'))}, "
            f"ST elevated territories={_fmt_list(interp.get('st_territories_elevated'))}, "
            f"ST depressed territories={_fmt_list(interp.get('st_territories_depressed'))}, "
            f"reciprocal change={_fmt_bool(interp.get('reciprocal_change_detected'), language)}, "
            f"tall T leads={_fmt_list(interp.get('tall_t_leads'))}"
        ),
        (
            f"- Other structural clues: LVH={interp.get('lvh_class') or '—'}, "
            f"LVH criteria={_fmt_list(interp.get('lvh_voltage_criteria'))}, "
            f"low voltage={interp.get('low_voltage_class') or '—'}, "
            f"RVH suspected={_fmt_bool(interp.get('rvh_suspected'), language)}, "
            f"limb reversal={_fmt_bool(interp.get('limb_reversal_suspected'), language)}, "
            f"precordial reversal={_fmt_bool(interp.get('precordial_reversal_suspected'), language)}, "
            f"paced rhythm={_fmt_bool(gf.get('paced_rhythm'), language)}"
        ),
        (
            f"- Signal quality: overall unreliable leads={_fmt_list(unreliable_leads)}, "
            f"P-unreliable={_fmt_flagged_leads(quality, 'reliable_for_p')}, "
            f"QRS-unreliable={_fmt_flagged_leads(quality, 'reliable_for_qrs')}, "
            f"T-unreliable={_fmt_flagged_leads(quality, 'reliable_for_t')}, "
            f"QT-unreliable={_fmt_flagged_leads(quality, 'reliable_for_qt')}"
        ),
        f"- Reliable QT leads used by extractor: {_fmt_list(metadata.get('reliable_qt_leads'))}",
        "- Beat groups:",
        _summarize_groups(groups, language),
        "- Representative lead measurements:",
        _summarize_representative_leads(representative_leads, quality),
    ]
    return "\n".join(summary_lines)


def build_context_summary_from_paths(
    features_path: Path | None,
    report_path: Path | None,
    clinician_notes: str,
    language: str = "en",
) -> str:
    parts = []

    if features_path is not None:
        data = _safe_read_json(features_path)
        if is_current_features_schema(data):
            # Independent evaluation must never receive the extractor's
            # ``interpretation`` or ``clinical_interpretation`` answers.
            parts.append(summarize_independent_measurements(data, language))
        elif is_legacy_input_schema(data):
            parts.append(summarize_legacy_input(data, language))
        else:
            header = "[Uploaded JSON was not recognized as the current features schema]"
            if language == "ja":
                header = "[アップロードされた JSON は現在の features schema として認識されませんでした]"
            # Dumping an unknown object into the prompt previously allowed
            # arbitrary ``Dx``/ground-truth fields to bypass the schema-aware
            # sanitizer.  Fail closed and expose no unknown values.
            parts.append(header + "\n[Content withheld: unsupported schema]")

    if report_path is not None:
        report_text = sanitize_report_text(_safe_read_text(report_path))
        if report_text:
            parts.append("[Supplementary Readable Report (Dx labels removed)]\n" + report_text)

    if clinician_notes.strip():
        parts.append("[User Notes]\n" + clinician_notes.strip())

    return "\n\n".join(parts).strip()


def build_prompt(context_summary: str, language: str = "en") -> str:
    if language == "ja":
        system_message = """あなたは、ECG 特徴抽出器の構造化出力をもとに二次診断サマリーを作成する熟練の心電図判読医です。

次の原則を守ってください:
1. 入力に含まれる ECG 計測値、導聯別結果、信号品質のみに基づいて回答すること。自動診断や元ラベルは入力に使用しないこと。
2. 低品質導聯を明確に考慮し、信頼性の低い導聯を強い根拠として扱わないこと。
3. まず最も可能性の高い診断を示し、その後に類似診断/鑑別診断を 3 件、類似度の高い順に挙げること。
4. 各類似診断について「共通点」と「重要な鑑別点」を必ず書き、病名だけを列挙しないこと。
5. 心拍数、RR 規則性、PR/QRS/QT、電気軸、Q 波、R 波進行、ST-T 変化、局在分布などの具体的根拠を引用すること。
6. 根拠が不足している場合、相互に矛盾する場合、または結論が不安定な場合は、不確実性と再確認すべき点を明示すること。
7. 元ラベルを作り込まないこと。補足レポートから Dx 欄は除去済みだが、あくまで ECG 特徴そのものを根拠にすること。
8. 出力は日本語で行うこと。"""
        user_message = f"""以下の ECG コンテキストに基づいて診断サマリーを作成してください。

{context_summary}

必ず次の形式で出力してください:
### 最も可能性の高い診断
- ...

### 診断根拠
- ...

### 類似診断（類似度順）
1. 診断名: ...
   共通点: ...
   鑑別点: ...
2. 診断名: ...
   共通点: ...
   鑑別点: ...
3. 診断名: ...
   共通点: ...
   鑑別点: ...

### 不確実性と再確認ポイント
- ..."""
    else:
        system_message = """You are a senior ECG interpretation specialist creating a second-pass diagnostic summary from structured outputs produced by an ECG feature extractor.

Follow these rules:
1. Base your answer only on ECG measurements, per-lead findings, and signal-quality information. Automated diagnostic interpretations and original labels must not be used.
2. Explicitly consider low-quality leads and do not treat unreliable leads as strong evidence.
3. Give the single most likely diagnosis first, then list 3 similar/differential diagnoses ranked from most similar to least similar.
4. For each similar diagnosis, explain both the shared features and the key differentiating feature(s); do not just list disease names.
5. Cite concrete evidence such as heart rate, RR regularity, PR/QRS/QT, axis, Q waves, R-wave progression, ST-T changes, and territorial lead distribution.
6. If the evidence is insufficient, conflicting, or unstable, explicitly describe the uncertainty and what should be reviewed.
7. Do not invent original labels. The supplementary report has had Dx fields removed, and ECG-derived evidence should remain the basis of your reasoning.
8. Output must be in English."""
        user_message = f"""Please produce a diagnostic summary from the ECG context below.

{context_summary}

Use exactly this format:
### Most Likely Diagnosis
- ...

### Diagnostic Rationale
- ...

### Similar Diagnoses (ranked by similarity)
1. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...
2. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...
3. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...

### Uncertainty and Review Priorities
- ..."""

    return f"<start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def canonical_label_for_dir(dir_label: str) -> str | None:
    return CANONICAL_LABELS.get(dir_label.lower())


def normalize_diagnosis(raw_text: str | None) -> str | None:
    if not raw_text:
        return None

    normalized = re.sub(r"[^a-z0-9\s\-_./]", " ", raw_text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in DIAGNOSIS_ALIASES:
        return DIAGNOSIS_ALIASES[normalized]

    for alias, canonical in sorted(DIAGNOSIS_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in normalized:
            return canonical
    return None


def parse_medgemma_output(text: str) -> dict[str, Any]:
    top1_match = re.search(r"### Most Likely Diagnosis\s*-\s*(.+)", text, flags=re.IGNORECASE)
    similar_matches = re.findall(r"\d+\.\s*Diagnosis:\s*(.+)", text)
    rationale_match = re.search(
        r"### Diagnostic Rationale\s*(.+?)\s*### Similar Diagnoses",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    uncertainty_match = re.search(
        r"### Uncertainty and Review Priorities\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    top1_raw = top1_match.group(1).strip() if top1_match else None
    similar_raw = [item.strip() for item in similar_matches[:3]]
    return {
        "top1_raw": top1_raw,
        "top1_normalized": normalize_diagnosis(top1_raw),
        "similar_raw": similar_raw,
        "similar_normalized": [normalize_diagnosis(item) for item in similar_raw],
        "diagnostic_rationale": rationale_match.group(1).strip() if rationale_match else "",
        "uncertainty": uncertainty_match.group(1).strip() if uncertainty_match else "",
    }


# ---------------------------------------------------------------------------
# Layered chain-of-thought diagnosis pipeline
#
# The single-shot prompt above hands the model a full feature dump and a
# fixed output template, but nothing stops the model from writing a
# conclusion first and rationalizing it afterward. This pipeline instead:
#   1. splits raw evidence into the same layers a cardiologist reads in
#      (signal quality -> rhythm -> conduction -> axis/chambers -> ST-T),
#      each layer only seeing the fields relevant to it;
#   2. forbids feeding the model any pre-computed diagnostic conclusion
#      (including unified clinical rule output) so it cannot
#      shortcut past its own reasoning;
#   3. runs deterministic guardrail checks against known extractor false-
#      positive patterns after generation, and asks the model to revise
#      once if a guardrail applies but the model's own text never engaged
#      with it;
#   4. compares the final answer against the withheld rule-engine output
#      only as an informational cross-check, never as reasoning input.
# ---------------------------------------------------------------------------

LAYER_KEYS = ["L0", "L1", "L2", "L3", "L4"]

_LAYER_TITLES_EN = {
    "L0": "Signal Quality Gate",
    "L1": "Rhythm",
    "L2": "Conduction & Intervals",
    "L3": "Axis & Chamber Morphology",
    "L4": "ST-T / Ischemia & Infarction",
}

_LAYER_TITLES_JA = {
    "L0": "信号品質ゲート",
    "L1": "調律",
    "L2": "伝導・間期",
    "L3": "電気軸・心腔形態",
    "L4": "ST-T・虚血/梗塞",
}


def _fmt_per_lead_qrs(representative_leads: dict[str, Any] | None) -> str:
    if not representative_leads:
        return "—"
    parts = []
    for lead in STANDARD_12_LEADS:
        params = (representative_leads.get(lead) or {}).get("params", {})
        qrs = params.get("qrs_ms")
        if isinstance(qrs, (int, float)):
            parts.append(f"{lead}:{qrs:.0f}ms")
    return _fmt_list(parts)


def _fmt_q_wave_detail(mi_evidence: dict[str, Any] | None) -> str:
    if not isinstance(mi_evidence, dict) or not mi_evidence.get("available"):
        return "—"
    q_by_lead = mi_evidence.get("q_by_lead") or {}
    parts = []
    for lead in STANDARD_12_LEADS:
        info = q_by_lead.get(lead)
        if not isinstance(info, dict):
            continue
        if not (info.get("significant_q") or info.get("q_wave_mi_ratio")):
            continue
        parts.append(
            f"{lead}(q_amp={_fmt_num(info.get('q_amplitude_mV'), 3)}mV, "
            f"r_amp={_fmt_num(info.get('r_amplitude_mV'), 3)}mV, "
            f"q_r_ratio={_fmt_num(info.get('q_r_ratio'), 3)}, "
            f"significant_q={info.get('significant_q')}, "
            f"q_wave_mi_ratio={info.get('q_wave_mi_ratio')})"
        )
    return _fmt_list(parts)


def _finite_nonnegative(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _metadata_mapping(data: dict[str, Any]) -> Mapping[str, Any]:
    metadata = data.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _patient_metadata(data: dict[str, Any]) -> Mapping[str, Any]:
    patient = _metadata_mapping(data).get("patient_meta")
    return patient if isinstance(patient, Mapping) else {}


def _resolved_patient_age(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve age once, with exact ``age_days`` taking precedence.

    An explicitly supplied but invalid ``age_days`` is not silently replaced
    by a possibly stale year-age.  In that case age-dependent conclusions are
    withheld.  This mirrors the extractor's exact-day precedence while making
    the safety behavior for corrupt demographics explicit.
    """
    patient = _patient_metadata(data)

    raw_age_days = patient.get("age_days")
    if raw_age_days is not None:
        age_days = _finite_nonnegative(raw_age_days)
        if age_days is None:
            return {
                "known": False,
                "age_days": None,
                "age_years": None,
                "source": "invalid_age_days",
                "pediatric_morphology": False,
                "pediatric_rate": False,
            }
        age_years = age_days / 365.25
        source = "age_days"
    else:
        age_years = _finite_nonnegative(patient.get("age"))
        if age_years is None:
            return {
                "known": False,
                "age_days": None,
                "age_years": None,
                "source": "missing",
                "pediatric_morphology": False,
                "pediatric_rate": False,
            }
        age_days = age_years * 365.25
        source = "age_years"

    return {
        "known": True,
        "age_days": age_days,
        "age_years": age_years,
        "source": source,
        "pediatric_morphology": age_years < _PEDIATRIC_MORPHOLOGY_MAX_AGE_YEARS,
        "pediatric_rate": age_days < _RATE_ADULT_AGE_DAYS,
    }


def _age_context_text(age: dict[str, Any]) -> str:
    if not age["known"]:
        reason = "invalid age_days" if age["source"] == "invalid_age_days" else "not provided"
        return f"Age unknown ({reason}; age-specific thresholds unavailable)"
    if age["source"] == "age_days":
        days = float(age["age_days"])
        if days < 365.25:
            return (
                f"Age {days:g} days ({float(age['age_years']):.3f} years; "
                "age_days authoritative)"
            )
        return (
            f"Age {float(age['age_years']):.2f} years "
            f"(resolved from age_days={days:g})"
        )
    return f"Age {float(age['age_years']):g} years"


def _patient_context(data: dict[str, Any]) -> str:
    metadata = _metadata_mapping(data)
    patient = _patient_metadata(data)
    return (
        f"{_age_context_text(_resolved_patient_age(data))} | Sex {patient.get('sex', 'N/A')} | "
        f"beats {metadata.get('n_beats', len(data.get('beats') or []))} | "
        f"input fs {metadata.get('input_fs', data.get('fs', 'N/A'))} Hz"
    )


_LAYER_GATE_DOMAINS = {
    "L1": {"rhythm", "ectopy"},
    "L2": {"conduction", "intervals", "av_conduction", "qrs_conduction"},
    "L3": {
        "axis",
        "axis_quantitative",
        "voltage",
        "chamber",
        "r_progression",
        "voltage_chamber_r_progression",
    },
    "L4": {"repolarization", "st_t", "ischemia", "infarction", "q_wave"},
}


def diagnostic_gate_policy(
    data: dict[str, Any], *, allow_legacy_missing: bool = False
) -> dict[str, Any]:
    """Normalize the extractor's diagnostic gate for downstream consumers.

    Current feature artifacts must carry the complete structured gate
    contract. Missing, string-valued, or incomplete gates fail closed. The
    sole compatibility exception is an explicitly acknowledged legacy input
    schema on the non-independent path; it never applies to current exports.
    """
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw = metadata.get("diagnostic_gate")
    if raw is None:
        if (
            allow_legacy_missing
            and is_legacy_input_schema(data)
            and not is_current_features_schema(data)
        ):
            return {
                "state": "pass",
                "stop_reasons": [],
                "partial_reasons": [],
                "allowed_domains": ["all"],
                "suppressed_domains": [],
                "source": "explicit_non_independent_legacy_override",
            }
        return {
            "state": "stop",
            "stop_reasons": ["missing_diagnostic_gate"],
            "partial_reasons": [],
            "allowed_domains": [],
            "suppressed_domains": [],
            "source": "metadata.diagnostic_gate",
        }

    required_fields = {
        "state",
        "stop_reasons",
        "partial_reasons",
        "allowed_domains",
        "suppressed_domains",
    }
    list_fields = (
        "stop_reasons",
        "partial_reasons",
        "allowed_domains",
        "suppressed_domains",
    )
    malformed = not isinstance(raw, Mapping) or not required_fields.issubset(raw)
    if malformed:
        return {
            "state": "stop",
            "stop_reasons": ["malformed_diagnostic_gate"],
            "partial_reasons": [],
            "allowed_domains": [],
            "suppressed_domains": [],
            "source": "metadata.diagnostic_gate",
        }

    state = str(raw.get("state") or "").strip().lower()
    if state not in {"pass", "partial", "stop"}:
        return {
            "state": "stop",
            "stop_reasons": ["malformed_diagnostic_gate_state"],
            "partial_reasons": [],
            "allowed_domains": [],
            "suppressed_domains": [],
            "source": "metadata.diagnostic_gate",
        }

    normalized_lists: dict[str, list[str]] = {}
    for field in list_fields:
        value = raw.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            return {
                "state": "stop",
                "stop_reasons": [f"malformed_diagnostic_gate_{field}"],
                "partial_reasons": [],
                "allowed_domains": [],
                "suppressed_domains": [],
                "source": "metadata.diagnostic_gate",
            }
        normalized_lists[field] = [item.strip() for item in value]

    stop_reasons = normalized_lists["stop_reasons"]
    partial_reasons = normalized_lists["partial_reasons"]
    allowed = normalized_lists["allowed_domains"]
    suppressed = normalized_lists["suppressed_domains"]
    invalid_contract = (
        state == "pass"
        and (stop_reasons or partial_reasons or allowed != ["all"] or suppressed)
    ) or (
        state == "partial"
        and (stop_reasons or not partial_reasons or not allowed)
    ) or (state == "stop" and (not stop_reasons or allowed))
    if invalid_contract:
        return {
            "state": "stop",
            "stop_reasons": [f"malformed_diagnostic_gate_{state}_contract"],
            "partial_reasons": [],
            "allowed_domains": [],
            "suppressed_domains": [],
            "source": "metadata.diagnostic_gate",
        }

    return {
        "state": state,
        "stop_reasons": stop_reasons,
        "partial_reasons": partial_reasons,
        "allowed_domains": [item.lower() for item in allowed],
        "suppressed_domains": [item.lower() for item in suppressed],
        "source": "metadata.diagnostic_gate",
    }


def diagnostic_gate_blocked_layers(gate: dict[str, Any]) -> set[str]:
    """Map domain-level quality restrictions onto the MedGemma layers."""
    if gate.get("state") == "stop":
        return set(LAYER_KEYS[1:])

    blocked: set[str] = set()
    suppressed = set(gate.get("suppressed_domains") or [])
    for layer_key, domains in _LAYER_GATE_DOMAINS.items():
        if domains & suppressed:
            blocked.add(layer_key)

    allowed = set(gate.get("allowed_domains") or [])
    if allowed and "all" not in allowed:
        for layer_key, domains in _LAYER_GATE_DOMAINS.items():
            if not domains & allowed:
                blocked.add(layer_key)
    return blocked


def _gate_withheld_text(layer_key: str, gate: dict[str, Any]) -> str:
    reasons = list(gate.get("stop_reasons") or []) + list(gate.get("partial_reasons") or [])
    suppressed = gate.get("suppressed_domains") or []
    return (
        f"- WITHHELD BY DIAGNOSTIC GATE: {layer_key} evidence and deterministic facts are not "
        f"available for diagnosis (state={gate.get('state')}; reasons={_fmt_list(reasons)}; "
        f"suppressed_domains={_fmt_list(suppressed)}). The only permitted conclusion for this "
        "layer is abstention."
    )


def _raw_rr_intervals(data: dict[str, Any]) -> list[float]:
    intervals: list[float] = []
    for beat in data.get("beats") or []:
        if not isinstance(beat, dict):
            continue
        value = beat.get("rr_next_ms")
        if isinstance(value, (int, float)) and value > 0:
            intervals.append(float(value))
    return intervals


def _fmt_raw_rr_evidence(data: dict[str, Any]) -> str:
    intervals = _raw_rr_intervals(data)
    if not intervals:
        return "- RR intervals: unavailable"

    rr_median = median(intervals)
    rr_mean = sum(intervals) / len(intervals)
    rr_cv = pstdev(intervals) / rr_mean if len(intervals) > 1 and rr_mean else 0.0
    lower = 0.75 * rr_median
    upper = 1.25 * rr_median
    central = [value for value in intervals if lower <= value <= upper]
    central_mean = sum(central) / len(central) if central else 0.0
    central_cv = (
        pstdev(central) / central_mean
        if len(central) > 1 and central_mean
        else 0.0
    )
    outliers = [value for value in intervals if value < lower or value > upper]
    series = ", ".join(f"{value:.0f}" for value in intervals[:40])
    if len(intervals) > 40:
        series += ", ..."
    return "\n".join(
        [
            f"- Raw RR series ({len(intervals)} intervals, ms): {series}",
            f"- RR summary calculated from the series: median {rr_median:.0f} ms, "
            f"range {min(intervals):.0f}-{max(intervals):.0f} ms, all-interval CV {rr_cv:.4f}",
            f"- Robust regularity check: central intervals within ±25% of median "
            f"n={len(central)}, CV={central_cv:.4f}; isolated outliers={_fmt_list([f'{value:.0f}' for value in outliers])} ms",
        ]
    )


def _fmt_raw_p_evidence(representative_leads: dict[str, Any]) -> str:
    parts = []
    for lead in ("I", "II", "III", "aVF", "V1", "V2"):
        params = (representative_leads.get(lead) or {}).get("params") or {}
        parts.append(
            f"- {lead}: P amp={_fmt_num(params.get('p_amp_mv'), 3, ' mV')}, "
            f"duration={_fmt_num(params.get('p_dur_consensus_ms'), 0, ' ms')}, "
            f"notched={params.get('p_notched')}, biphasic={params.get('p_biphasic')}, "
            f"template corr={_fmt_num(params.get('p_template_corr'), 3)}, "
            f"PR consistency={_fmt_num(params.get('p_pr_consistency_score'), 3)}, "
            f"PP consistency={_fmt_num(params.get('p_pp_consistency_score'), 3)}, "
            f"P reliable={params.get('reliable_for_p')}"
        )
    return "\n".join(parts)


def _fmt_raw_atrial_residual_evidence(data: dict[str, Any]) -> str:
    rhythm_inputs = data.get("rhythm_inputs") or {}
    af_afl = rhythm_inputs.get("af_afl") or {}
    residual = af_afl.get("qrst_subtraction_quality") or {}
    per_lead = residual.get("per_lead_atrial_activity") or {}
    validation = (af_afl.get("qrst_subtraction") or {}).get("validation_metrics") or {}
    lines = [
        "- QRST-residual measurements (measurement evidence only; no AF/AFL classifier labels): "
        f"template corr={_fmt_num(validation.get('template_correlation'), 3)}, "
        f"residual RMS ratio={_fmt_num(validation.get('template_residual_rms_ratio'), 3)}, "
        f"usable leads={_fmt_list(validation.get('usable_leads'))}"
    ]
    for lead in ("I", "II", "III", "V1", "V2"):
        info = per_lead.get(lead)
        if not isinstance(info, dict):
            continue
        lines.append(
            f"- residual {lead}: RMS={_fmt_num(info.get('residual_rms_mv'), 3, ' mV')}, "
            f"repetitiveness={_fmt_num(info.get('repetitiveness'), 3)}, "
            f"stability={_fmt_num(info.get('stability'), 3)}, "
            f"dominant cycle={_fmt_num(info.get('dominant_cycle_ms'), 0, ' ms')}, "
            f"spectral entropy={_fmt_num(info.get('spectral_entropy'), 3)}, "
            f"organized power ratio={_fmt_num(info.get('organized_spectral_power_ratio'), 3)}"
        )
    return "\n".join(lines)


def _fmt_candidate_av_evidence(data: dict[str, Any]) -> str:
    rhythm_inputs = data.get("rhythm_inputs") or {}
    candidate = ((rhythm_inputs.get("av_block") or {}).get("evidence") or {})
    pr_series = candidate.get("pr_series_ms") or []
    events_per_rr = candidate.get("atrial_events_per_rr") or []
    p_events = rhythm_inputs.get("p_events") or []
    associations: dict[str, int] = {}
    for event in p_events:
        if not isinstance(event, dict):
            continue
        key = str(event.get("association_type") or "unassigned")
        associations[key] = associations.get(key, 0) + 1
    return "\n".join(
        [
            "- Unadjudicated atrial-event detector output (may contain false P detections): "
            + _fmt_list([f"{key}={value}" for key, value in sorted(associations.items())]),
            "- Candidate PR series (ms): "
            + _fmt_list([f"{value:.0f}" if isinstance(value, (int, float)) else value for value in pr_series]),
            "- Candidate atrial-event counts per RR interval: "
            + _fmt_list(events_per_rr),
        ]
    )


def _fmt_raw_qrs_evidence(representative_leads: dict[str, Any]) -> str:
    lines = []
    for lead in STANDARD_12_LEADS:
        params = (representative_leads.get(lead) or {}).get("params") or {}
        lines.append(
            f"- {lead}: QRS={_fmt_num(params.get('qrs_ms'), 0, ' ms')}, "
            f"Q dur={_fmt_num(params.get('q_duration_ms'), 0, ' ms')}, "
            f"R={_fmt_num(params.get('r_amp_mv'), 3, ' mV')}, "
            f"R'={_fmt_num(params.get('r_prime_amp_mv'), 3, ' mV')}, "
            f"R' dur={_fmt_num(params.get('r_prime_duration_ms'), 0, ' ms')}, "
            f"S={_fmt_num(params.get('s_amp_mv'), 3, ' mV')}, "
            f"S dur={_fmt_num(params.get('s_duration_ms'), 0, ' ms')}, "
            f"QRS reliable={params.get('reliable_for_qrs')}"
        )
    return "\n".join(lines)


def _fmt_raw_voltage_evidence(representative_leads: dict[str, Any]) -> str:
    lines = []
    for lead in STANDARD_12_LEADS:
        params = (representative_leads.get(lead) or {}).get("params") or {}
        lines.append(
            f"- {lead}: P={_fmt_num(params.get('p_amp_mv'), 3, ' mV')}, "
            f"P dur={_fmt_num(params.get('p_dur_consensus_ms'), 0, ' ms')}, "
            f"P terminal amp={_fmt_num(params.get('p_terminal_amp_mv'), 3, ' mV')}, "
            f"P terminal area={_fmt_num(params.get('p_terminal_area_mv_ms'), 2, ' mV·ms')}, "
            f"R={_fmt_num(params.get('r_amp_mv'), 3, ' mV')}, "
            f"S={_fmt_num(params.get('s_amp_mv'), 3, ' mV')}, "
            f"QRS signed area={_fmt_num(params.get('qrs_signed_area'), 2)}, "
            f"P/QRS reliable={params.get('reliable_for_p')}/{params.get('reliable_for_qrs')}"
        )
    return "\n".join(lines)


def _fmt_raw_repolarization_evidence(representative_leads: dict[str, Any]) -> str:
    lines = []
    for lead in STANDARD_12_LEADS:
        params = (representative_leads.get(lead) or {}).get("params") or {}
        lines.append(
            f"- {lead}: Q dur={_fmt_num(params.get('q_duration_ms'), 0, ' ms')}, "
            f"Q={_fmt_num(params.get('q_amp_mv'), 3, ' mV')}, "
            f"R={_fmt_num(params.get('r_amp_mv'), 3, ' mV')}, "
            f"Q/R={_fmt_num(params.get('q_r_ratio'), 3)}, "
            f"ST-J={_fmt_num(params.get('st_on_mv'), 3, ' mV')}, "
            f"ST-mid={_fmt_num(params.get('st_mid_mv'), 3, ' mV')}, "
            f"ST-80={_fmt_num(params.get('st_80ms_mv'), 3, ' mV')}, "
            f"J reliable={params.get('st_j_reliable')}, "
            f"T={_fmt_num(params.get('t_amp_mv'), 3, ' mV')}, "
            f"T polarity expected/observed={params.get('t_polarity_expected')}/{params.get('t_polarity_observed')}, "
            f"T reliable={params.get('reliable_for_t')}"
        )
    return "\n".join(lines)


def summarize_layered_evidence(data: dict[str, Any], language: str = "en") -> dict[str, str]:
    """Split raw features into per-layer evidence bundles.

    This path deliberately excludes *all* ``interpretation`` fields and
    clinical conclusions.  Only measurements, quality
    flags, beat intervals, and explicitly-labelled unadjudicated detector
    evidence are exposed.  That prevents the model from simply repeating
    ``probable_af``, ``bundle_branch_block``, ``pathological_q_leads``, etc.
    """
    gf = data.get("global_features", {})
    quality = data.get("quality", {})
    representative_leads = data.get("representative_leads", {})
    groups = data.get("groups", {})
    metadata = _metadata_mapping(data)
    patient_line = _patient_context(data)
    gate = diagnostic_gate_policy(data)
    blocked_layers = diagnostic_gate_blocked_layers(gate)

    unreliable_leads = [
        f"{lead}({_fmt_list(info.get('flags', []))})"
        for lead, info in quality.items()
        if not info.get("reliable", True)
    ]

    l0 = "\n".join(
        [
            f"- Patient/record: {patient_line}",
            f"- Record quality grade: {(metadata.get('record_quality') or {}).get('record_grade', 'N/A')}; "
            f"reason codes={_fmt_list((metadata.get('record_quality') or {}).get('reason_codes'))}",
            f"- Diagnostic gate: state={gate['state']}; "
            f"stop reasons={_fmt_list(gate.get('stop_reasons'))}; "
            f"partial reasons={_fmt_list(gate.get('partial_reasons'))}; "
            f"allowed domains={_fmt_list(gate.get('allowed_domains'))}; "
            f"suppressed domains={_fmt_list(gate.get('suppressed_domains'))}",
            f"- Overall unreliable leads: {_fmt_list(unreliable_leads)}",
            f"- P-unreliable={_fmt_flagged_leads(quality, 'reliable_for_p')}, "
            f"QRS-unreliable={_fmt_flagged_leads(quality, 'reliable_for_qrs')}, "
            f"T-unreliable={_fmt_flagged_leads(quality, 'reliable_for_t')}, "
            f"QT-unreliable={_fmt_flagged_leads(quality, 'reliable_for_qt')}",
            f"- Reliable QT leads used by extractor: {_fmt_list(metadata.get('reliable_qt_leads'))}",
            "- Beat groups:",
            _summarize_groups(groups, language),
        ]
    )

    l1 = "\n".join(
        [
            f"- Patient/record: {patient_line}",
            f"- HR {_fmt_num(gf.get('heart_rate_bpm'), 1, ' bpm')}, "
            f"atrial rate {_fmt_num(gf.get('atrial_rate_bpm'), 1, ' bpm')}",
            f"- Measured P axis: {_fmt_num(gf.get('p_axis_deg'), 0, '°')}",
            _fmt_raw_rr_evidence(data),
            "- Raw representative P-wave measurements:",
            _fmt_raw_p_evidence(representative_leads),
            _fmt_raw_atrial_residual_evidence(data),
        ]
    )

    l2 = "\n".join(
        [
            f"- Patient/record: {patient_line}",
            f"- HR {_fmt_num(gf.get('heart_rate_bpm'), 1, ' bpm')}; "
            f"measured PR {_fmt_num(gf.get('pr_ms'), 0, ' ms')}",
            f"- Consensus QRS {_fmt_num(gf.get('qrs_ms'), 0, ' ms')}",
            f"- QT {_fmt_num(gf.get('qt_ms'), 0, ' ms')}, "
            f"QTcB {_fmt_num(gf.get('qtc_bazett_ms'), 0, ' ms')}, "
            f"QTcF {_fmt_num(gf.get('qtc_fridericia_ms'), 0, ' ms')}, "
            f"QT dispersion {_fmt_num(gf.get('qt_dispersion_ms'), 0, ' ms')}",
            f"- Pacing spikes measured={_fmt_list(gf.get('pacing_spikes'))}; "
            f"paced rhythm measurement={gf.get('paced_rhythm')}",
            "- Raw per-lead QRS morphology measurements:",
            _fmt_raw_qrs_evidence(representative_leads),
            _fmt_candidate_av_evidence(data),
        ]
    )

    l3 = "\n".join(
        [
            f"- Patient/record: {patient_line}",
            f"- Measured axes: P {_fmt_num(gf.get('p_axis_deg'), 0, '°')}, "
            f"QRS {_fmt_num(gf.get('qrs_axis_deg'), 0, '°')}, "
            f"T {_fmt_num(gf.get('t_axis_deg'), 0, '°')}, "
            f"ST {_fmt_num(gf.get('st_axis_deg'), 0, '°')}",
            f"- Measured P-terminal force V1: {_fmt_num(gf.get('ptf_v1_mv_ms'), 3, ' mV·ms')}",
            "- Raw chamber/voltage measurements (no LVH/RVH/LAE classifier labels):",
            _fmt_raw_voltage_evidence(representative_leads),
        ]
    )

    l4 = "\n".join(
        [
            f"- Patient/record: {patient_line}",
            f"- HR {_fmt_num(gf.get('heart_rate_bpm'), 1, ' bpm')}; "
            f"consensus QRS {_fmt_num(gf.get('qrs_ms'), 0, ' ms')}",
            "- Raw Q/ST/T measurements (no pathological-Q, territory, ischemia, or STEMI classifier labels):",
            _fmt_raw_repolarization_evidence(representative_leads),
        ]
    )

    layers = {"L0": l0, "L1": l1, "L2": l2, "L3": l3, "L4": l4}
    for layer_key in blocked_layers:
        layers[layer_key] = _gate_withheld_text(layer_key, gate)
    return layers


def _numeric_measurement(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _rate_class(
    heart_rate: Any, age: dict[str, Any]
) -> tuple[str, float | None, float | None]:
    value = _numeric_measurement(heart_rate)
    if value is None:
        return "unavailable", None, None
    if not age["known"]:
        return "unavailable because age is unknown; adult limits were not applied", None, None
    age_days = float(age["age_days"])
    brady_limit = bradycardia_limit(age_days, _MEDGEMMA_RATE_CONFIG)
    tachy_limit = tachycardia_limit(age_days, _MEDGEMMA_RATE_CONFIG)
    profile = "age-adjusted pediatric" if age["pediatric_rate"] else "adult"
    if value < brady_limit:
        return f"bradycardic by {profile} limits", brady_limit, tachy_limit
    if value > tachy_limit:
        return f"tachycardic by {profile} limits", brady_limit, tachy_limit
    return f"normal by {profile} limits", brady_limit, tachy_limit


def _pr_limits_for_age(age: dict[str, Any]) -> dict[str, float] | None:
    if not age["known"]:
        return None
    if float(age["age_days"]) < _RATE_ADULT_AGE_DAYS:
        context = SimpleNamespace(
            patient=SimpleNamespace(age_days=float(age["age_days"]))
        )
        return dict(_glasgow_pr_limits(context))
    return {"short": 120.0, "first_degree": 200.0, "borderline": 200.0}


def _pr_class(pr_ms: Any, age: dict[str, Any]) -> tuple[str, dict[str, float] | None]:
    value = _numeric_measurement(pr_ms)
    limits = _pr_limits_for_age(age)
    if value is None:
        return "unavailable", limits
    if limits is None:
        return "unavailable because age is unknown; adult limits were not applied", None
    profile = "age-adjusted pediatric" if age["pediatric_rate"] else "adult"
    if value < limits["short"]:
        return f"short by {profile} limits", limits
    if value >= limits["first_degree"]:
        return (
            f"prolonged by {profile} limits; can support first-degree AV delay only with 1:1 conduction",
            limits,
        )
    return (
        f"within {profile} limits; does not support first-degree AV delay",
        limits,
    )


def _qrs_class(
    qrs_ms: Any, age: dict[str, Any]
) -> tuple[str, dict[str, float] | None]:
    value = _numeric_measurement(qrs_ms)
    if value is None:
        return "unavailable", None
    if not age["known"]:
        return "unavailable because age is unknown; adult limits were not applied", None
    if age["pediatric_morphology"]:
        normal = _peds_lookup(
            _PEDS_QRS_NORMAL_MS, float(age["age_years"])
        )
        if normal is None:
            return "pediatric threshold unavailable", None
        limits = {
            "normal": float(normal),
            "borderline": float(normal) * 1.10,
            "nonspecific": float(normal) * 1.20,
        }
        category = _peds_qrs_width_class(value, float(normal))
        labels = {
            "normal": "normal by age-adjusted pediatric limit",
            "above_normal_below_ivcd": "above the pediatric normal limit but below IVCD range",
            "borderline_ivcd_range": "pediatric borderline IVCD range",
            "nonspecific_ivcd_range": "pediatric nonspecific IVCD/BBB duration range",
            "indeterminate": "pediatric threshold unavailable",
        }
        return labels.get(category, str(category)), limits
    limits = {"normal": 110.0, "borderline": 110.0, "nonspecific": 120.0}
    if value < 110:
        return "normal adult duration", limits
    if value < 120:
        return "mildly prolonged; BBB requires compatible morphology", limits
    return "wide", limits


def _qrs_axis_class(axis_deg: Any, age: dict[str, Any]) -> str:
    value = _numeric_measurement(axis_deg)
    if value is None:
        return "unavailable"
    if not age["known"]:
        return "unavailable because age is unknown; adult limits were not applied"
    if age["pediatric_morphology"]:
        category = _peds_classify_qrs_axis(value, float(age["age_years"]))
        labels = {
            "LAD": "left axis deviation by age-adjusted pediatric limits",
            "LAFB": "left axis deviation/LAFB range by age-adjusted pediatric limits",
            "borderline_LAD": "borderline left axis by age-adjusted pediatric limits",
            "RAD": "right axis deviation by age-adjusted pediatric limits",
            "borderline_RAD": "borderline right axis by age-adjusted pediatric limits",
            "normal": "normal age-adjusted pediatric axis",
            "indeterminate": "pediatric axis threshold unavailable",
        }
        return labels.get(category, str(category))
    if value < -30:
        return "left axis deviation"
    if value > 90:
        return "right axis deviation"
    return "normal adult axis"


def _qtc_class(
    qtc_ms: Any, sex: str | None, age: dict[str, Any]
) -> tuple[str, str]:
    value = _numeric_measurement(qtc_ms)
    if not age["known"]:
        return (
            "unavailable because age is unknown; adult limits were not applied",
            "age-specific limits unavailable",
        )
    if age["pediatric_morphology"]:
        limits = _peds_qtc_limits(float(age["age_years"]), sex)
        borderline = limits["borderline_prolonged_ms"]
        prolonged = limits["prolonged_ms"]
        if value is None or borderline is None or prolonged is None:
            return "unavailable", "pediatric limits unavailable"
        if value < float(limits["short_ms"]):
            result = "short by age-adjusted pediatric limits"
        elif value > float(limits["severe_ms"]):
            result = "significantly prolonged by age-adjusted pediatric limits"
        elif value > float(prolonged):
            result = "prolonged by age-adjusted pediatric limits"
        elif value > float(borderline):
            result = "borderline prolonged by age-adjusted pediatric limits"
        else:
            result = "normal by age-adjusted pediatric limits"
        return (
            result,
            f"pediatric borderline>{float(borderline):.0f} ms, prolonged>{float(prolonged):.0f} ms",
        )

    sex_lower = (sex or "").lower()
    upper = 450.0 if sex_lower.startswith("m") else 470.0 if sex_lower.startswith("f") else 460.0
    if value is None:
        return "unavailable", f"adult upper {upper:.0f} ms"
    if value > upper:
        return "prolonged", f"adult upper {upper:.0f} ms"
    if value < 350:
        return "short", f"adult upper {upper:.0f} ms"
    return "within the usual adult range", f"adult upper {upper:.0f} ms"


def _lead_param(data: dict[str, Any], lead: str, key: str) -> Any:
    return (((data.get("representative_leads") or {}).get(lead) or {}).get("params") or {}).get(key)


def _lvh_voltage_facts(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return computed LVH voltage criteria and their audit strings."""
    patient = _patient_metadata(data)
    sex = str(patient.get("sex") or "")
    age = _resolved_patient_age(data)
    if not age["known"]:
        return [], [
            "LVH voltage classification withheld because age is unknown; adult voltage thresholds were not applied"
        ]

    if age["pediatric_morphology"]:
        age_years = float(age["age_years"])
        axis_category = _peds_classify_qrs_axis(
            (data.get("global_features") or {}).get("qrs_axis_deg"),
            age_years,
        )
        evidence = build_pediatric_hypertrophy_evidence(
            representative_leads=data.get("representative_leads") or {},
            age_years=age_years,
            sex=sex or None,
            qrs_axis_class=axis_category,
            # Automated BBB interpretations are intentionally unavailable on
            # this independent path; the voltage facts therefore do not claim
            # that a morphology-based bypass was adjudicated.
            bundle_branch_block=None,
        )
        lvh = evidence.get("lvh") or {}
        criteria = [
            "pediatric LVH R-V6 >98th-percentile voltage"
            for item in (lvh.get("criteria") or [])
            if item == "lvh_r_v6_98p_mv"
        ]
        r_v6 = _lead_param(data, "V6", "r_amp_mv")
        threshold = pediatric_voltage_threshold(
            age_years, sex or None, "lvh_r_v6_98p_mv"
        )
        audit = [
            "Pediatric LVH voltage route selected "
            f"(age_days={float(age['age_days']):g}, age_bin={pediatric_age_bin(age_years) or 'unavailable'})"
        ]
        if isinstance(r_v6, (int, float)) and threshold is not None:
            audit.append(
                f"Pediatric RV6={float(r_v6):.3f} mV (>98th-percentile threshold "
                f"{float(threshold):.3f}; met={float(r_v6) >= float(threshold)})"
            )
        else:
            audit.append("Pediatric RV6 voltage criterion unavailable")
        return criteria, audit

    ra_vl = _lead_param(data, "aVL", "r_amp_mv")
    sv1 = _lead_param(data, "V1", "s_amp_mv")
    sv3 = _lead_param(data, "V3", "s_amp_mv")
    rv5 = _lead_param(data, "V5", "r_amp_mv")
    rv6 = _lead_param(data, "V6", "r_amp_mv")

    criteria: list[str] = []
    audit: list[str] = []
    if isinstance(sv1, (int, float)) and isinstance(rv5, (int, float)) and isinstance(rv6, (int, float)):
        sokolow = abs(min(sv1, 0.0)) + max(rv5, rv6)
        met = sokolow >= 3.5
        audit.append(f"Sokolow-Lyon |SV1|+max(RV5,RV6)={sokolow:.3f} mV (threshold 3.5; met={met})")
        if met:
            criteria.append("Sokolow-Lyon")
    if isinstance(ra_vl, (int, float)) and isinstance(sv3, (int, float)):
        cornell = ra_vl + abs(min(sv3, 0.0))
        sex_lower = sex.strip().lower()
        threshold = (
            2.8
            if sex_lower.startswith("m")
            else 2.0
            if sex_lower.startswith("f")
            else None
        )
        if threshold is None:
            audit.append(
                f"Cornell RaVL+|SV3|={cornell:.3f} mV (indeterminate: sex-specific threshold "
                "cannot be selected because sex is unknown)"
            )
        else:
            met = cornell > threshold
            audit.append(
                f"Cornell RaVL+|SV3|={cornell:.3f} mV "
                f"(sex threshold {threshold:.1f}; met={met})"
            )
            if met:
                criteria.append("Cornell voltage")
    if isinstance(ra_vl, (int, float)):
        met = ra_vl >= 1.1
        audit.append(f"RaVL={ra_vl:.3f} mV (threshold 1.1; met={met})")
        if met:
            criteria.append("RaVL")
    return criteria, audit


def _t_polarity_facts(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    negative_observed: list[str] = []
    conflicts: list[str] = []
    for lead in STANDARD_12_LEADS:
        observed = _lead_param(data, lead, "t_polarity_observed")
        amplitude = _lead_param(data, lead, "t_amp_mv")
        if observed == -1:
            negative_observed.append(lead)
        if isinstance(amplitude, (int, float)) and abs(amplitude) >= 0.02 and observed in (-1, 1):
            amplitude_sign = 1 if amplitude > 0 else -1
            if amplitude_sign != observed:
                conflicts.append(f"{lead}(T={amplitude:+.3f}mV, observed={observed})")
    return negative_observed, conflicts


def _age_resolved_q_wave_leads(
    data: dict[str, Any], age: dict[str, Any]
) -> tuple[list[str], str]:
    if not age["known"]:
        return [], "age unknown; adult Q-wave duration criteria were not applied"

    qualifying: list[str] = []
    if age["pediatric_morphology"]:
        for lead in STANDARD_12_LEADS:
            if lead == "aVR":
                continue
            amplitude = _numeric_measurement(_lead_param(data, lead, "q_amp_mv"))
            duration = _numeric_measurement(_lead_param(data, lead, "q_duration_ms"))
            ratio = _numeric_measurement(_lead_param(data, lead, "q_r_ratio"))
            if amplitude is None or amplitude >= 0:
                continue
            infarct_ratio = ratio is not None and ratio > 0.20 and (
                duration is None or duration >= 35.0
            )
            if abs(amplitude) >= 0.15 or infarct_ratio:
                qualifying.append(lead)
        return qualifying, "pediatric large-Q criterion (|Q|>=0.15 mV or age-route Q/R criterion)"

    for lead in STANDARD_12_LEADS:
        duration = _numeric_measurement(_lead_param(data, lead, "q_duration_ms"))
        amplitude = _numeric_measurement(_lead_param(data, lead, "q_amp_mv"))
        if lead != "aVR" and duration is not None and duration >= 40 and amplitude is not None and amplitude < 0:
            qualifying.append(lead)
    return qualifying, "adult measured Q duration >=40 ms"


def _organized_atrial_residual_facts(
    data: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    """Independently check raw QRST-residual periodicity for flutter-like activity.

    This deliberately ignores ``probable_flutter``, atrial classification,
    morphology-validation flags, rule statements, and Dx.  It recomputes a
    strict multilead pattern from direct cycle, stability, and organized-power
    measurements only.
    """
    residual = (
        ((data.get("metadata") or {}).get("rhythm_analysis") or {}).get(
            "atrial_residual"
        )
        or (((data.get("rhythm_inputs") or {}).get("af_afl") or {}).get(
            "qrst_subtraction_quality"
        ))
        or (((data.get("rhythm_inputs") or {}).get("af_afl") or {}).get(
            "atrial_residual_signal_summary"
        ))
        or {}
    )
    per_lead = residual.get("per_lead_atrial_activity") or {}
    qualifying: list[str] = []
    audit: list[str] = []
    cycles: list[float] = []
    for lead in ("I", "II", "III", "aVF", "V1", "V2"):
        info = per_lead.get(lead) or {}
        cycle = info.get("dominant_cycle_ms")
        stability = info.get("stability")
        power = info.get("organized_spectral_power_ratio")
        if not all(isinstance(value, (int, float)) for value in (cycle, stability, power)):
            continue
        met = 170 <= cycle <= 250 and stability >= 0.20 and power >= 0.40
        audit.append(
            f"{lead}(cycle={cycle:.0f} ms/{60000.0 / cycle:.0f} bpm, "
            f"stability={stability:.3f}, organized_power={power:.3f}, met={met})"
        )
        if met:
            qualifying.append(lead)
            cycles.append(float(cycle))
    concordant = bool(cycles) and max(cycles) - min(cycles) <= 30
    key_lead_count = len(set(qualifying) & {"II", "III", "aVF", "V1"})
    strong = len(qualifying) >= 3 and key_lead_count >= 3 and concordant
    return strong, qualifying, audit


def build_validated_measurement_facts(data: dict[str, Any]) -> dict[str, str]:
    """Build deterministic facts used to constrain, not replace, model judgment.

    These facts are calculated only from measurement fields in the JSON.  No
    ``interpretation``, unified-rule output, or original Dx is used.
    """
    gf = data.get("global_features") or {}
    patient = _patient_metadata(data)
    sex = str(patient.get("sex") or "")
    age = _resolved_patient_age(data)
    hr = gf.get("heart_rate_bpm")
    pr = gf.get("pr_ms")
    qrs = gf.get("qrs_ms")
    qtc_b = gf.get("qtc_bazett_ms")
    qtc_f = gf.get("qtc_fridericia_ms")
    qrs_axis = gf.get("qrs_axis_deg")
    ptf_v1 = gf.get("ptf_v1_mv_ms")
    rate_class, brady_limit, tachy_limit = _rate_class(hr, age)
    pr_class, pr_limits = _pr_class(pr, age)
    qrs_class, qrs_limits = _qrs_class(qrs, age)
    qtc_b_class, qtc_limits_text = _qtc_class(qtc_b, sex, age)
    qtc_f_class, _ = _qtc_class(qtc_f, sex, age)
    qrs_axis_class = _qrs_axis_class(qrs_axis, age)
    voltage_criteria, voltage_audit = _lvh_voltage_facts(data)
    negative_t, t_conflicts = _t_polarity_facts(data)
    organized_atrial, organized_atrial_leads, organized_atrial_audit = (
        _organized_atrial_residual_facts(data)
    )

    q_wave_leads, q_wave_criterion = _age_resolved_q_wave_leads(data, age)

    j_offsets = []
    for lead in STANDARD_12_LEADS:
        value = _lead_param(data, lead, "st_on_mv")
        if isinstance(value, (int, float)):
            j_offsets.append(f"{lead}:{value:+.3f}")

    facts = {
        "L0": (
            "- Source policy: JSON measurement fields plus diagnosis-free report measurements; "
            "original Dx codes/labels and automated diagnostic conclusions are excluded.\n"
            f"- Demographic threshold route: {_age_context_text(age)}."
        ),
        "L1": "\n".join(
            [
                f"- Deterministic age-resolved rate check: HR {_fmt_num(hr, 1, ' bpm')} => {rate_class}; "
                f"brady limit={_fmt_num(brady_limit, 1, ' bpm')}, "
                f"tachy limit={_fmt_num(tachy_limit, 1, ' bpm')}. A named rhythm must still be "
                "supported by atrial activity and RR behavior.",
                f"- Independent rapid organized atrial-residual check (no classifier/Dx fields): "
                f"qualifying leads={_fmt_list(organized_atrial_leads)}; strict multilead support="
                f"{organized_atrial}.",
                *[f"- Residual audit: {item}." for item in organized_atrial_audit],
                *(
                    [
                        "- Three concordant key leads show 240-350 bpm organized residual activity; "
                        "this strongly supports an atrial-flutter ECG pattern and must be explicitly "
                        "adjudicated against sinus tachycardia."
                    ]
                    if organized_atrial
                    else []
                ),
            ]
        ),
        "L2": "\n".join(
            [
                f"- Deterministic age-resolved PR check: {_fmt_num(pr, 0, ' ms')} => {pr_class}; "
                f"limits={pr_limits or 'unavailable'}.",
                f"- Deterministic age-resolved QRS-duration check: {_fmt_num(qrs, 0, ' ms')} => "
                f"{qrs_class}; limits={qrs_limits or 'unavailable'}.",
                f"- Deterministic age-resolved QTc check ({sex or 'sex unknown'}; {qtc_limits_text}): "
                f"QTcB {_fmt_num(qtc_b, 0, ' ms')} => {qtc_b_class}; "
                f"QTcF {_fmt_num(qtc_f, 0, ' ms')} => {qtc_f_class}.",
            ]
        ),
        "L3": "\n".join(
            [
                f"- Deterministic age-resolved QRS-axis check: {_fmt_num(qrs_axis, 0, '°')} => {qrs_axis_class}.",
                (
                    f"- Deterministic PTF-V1 check: {_fmt_num(ptf_v1, 3, ' mV·ms')}; the shared morphology "
                    "threshold is about 4.0 mV·ms, so smaller magnitudes alone do not support LA abnormality."
                    if age["known"]
                    else "- Deterministic PTF-V1 diagnostic classification withheld because age is unknown."
                ),
                f"- Computed LVH voltage criteria met: {_fmt_list(voltage_criteria)}.",
                *[f"- {item}." for item in voltage_audit],
            ]
        ),
        "L4": "\n".join(
            [
                f"- Observed negative T polarity leads: {_fmt_list(negative_t)}; aVR negativity is usually expected.",
                f"- T-amplitude/polarity field conflicts requiring downgrade to uncertain: {_fmt_list(t_conflicts)}.",
                f"- Age-resolved Q-wave screen ({q_wave_criterion}): {_fmt_list(q_wave_leads)}; "
                "a prior-MI pattern still requires at least two contiguous territorial leads.",
                f"- Raw signed ST-J offsets (mV; do not reverse signs or substitute ST-80 for J point): "
                f"{'; '.join(j_offsets) or '—'}.",
            ]
        ),
    }
    gate = diagnostic_gate_policy(data)
    for layer_key in diagnostic_gate_blocked_layers(gate):
        facts[layer_key] = _gate_withheld_text(layer_key, gate)
    return facts


def summarize_independent_measurements(
    data: dict[str, Any], language: str = "en"
) -> str:
    """Build the single-shot context from measurements and quality only.

    This is intentionally separate from :func:`summarize_current_features`,
    which is retained for non-evaluation display compatibility and includes
    automated interpretations.  Keeping the independent DTO explicit makes it
    difficult for newly-added rule outputs to enter an evaluation prompt by
    accident.
    """
    layers = summarize_layered_evidence(data, language)
    facts = build_validated_measurement_facts(data)
    blocks = ["[Independent Measurement-Only ECG Context]"]
    for layer_key in LAYER_KEYS:
        blocks.append(
            f"[{layer_key} measurement evidence]\n{layers[layer_key]}\n\n"
            f"[{layer_key} deterministic measurement checks]\n{facts[layer_key]}"
        )
    return "\n\n".join(blocks)


def summarize_reference_only(data: dict[str, Any]) -> str:
    """Rule-engine output withheld from reasoning, for post-hoc comparison only."""
    lines = summarize_clinical_interpretation(
        data.get("clinical_interpretation")
        or data.get("metadata", {}).get("clinical_interpretation")
    )
    return "\n".join(line for line in lines if line) or "—"


_LAYER_TASKS_EN = {
    "L0": "Assess technical quality only. Identify which measurements can and cannot support later interpretation. Do not diagnose rhythm or disease.",
    "L1": "Infer rate, rhythm origin, and regularity from the raw RR and atrial measurements. Distinguish persistent irregularity from isolated pauses/outliers.",
    "L2": "Infer AV conduction, intraventricular conduction, and PR/QRS/QT findings from measured intervals and morphology. Treat candidate atrial-event associations as unadjudicated detector output.",
    "L3": "Infer axis and chamber/voltage patterns from numeric axes and per-lead amplitudes. State the exact voltage or morphology criterion used.",
    "L4": "Infer Q-wave and ST-T patterns from per-lead measurements. Separate a measured waveform abnormality from a clinical ischemia/infarction diagnosis.",
}

_LAYER_TASKS_JA = {
    "L0": "技術的品質だけを評価し、後続判定に使用できる測定値と使用できない測定値を分けてください。調律や疾患は診断しないでください。",
    "L1": "生のRR系列と心房測定から心拍数、調律起源、規則性を推定し、持続的な不規則性と孤立した休止・外れ値を区別してください。",
    "L2": "測定された間期と形態から房室伝導、心室内伝導、PR/QRS/QTを推定してください。心房イベント候補は未判定の検出器出力として扱ってください。",
    "L3": "数値軸と導聯別振幅から電気軸・心腔・電位パターンを推定し、使用した基準を具体的に示してください。",
    "L4": "導聯別のQ波・ST-T測定から波形パターンを推定し、測定異常と臨床的な虚血・梗塞診断を分けてください。",
}

_LAYER_RULES_EN = {
    "L0": (
        "A global Q0/Q1 grade does not override lead-specific flags. Explicitly name any "
        "unreliable modality (P, QRS, T, or QT)."
    ),
    "L1": (
        "Do not diagnose atrial fibrillation from a high all-interval RR CV alone when the "
        "central RR intervals are regular and only a few pauses/outliers drive the CV. AF "
        "requires persistent irregularity plus absent organized atrial activity; flutter "
        "requires reproducible organized atrial activity."
    ),
    "L2": (
        "Do not call incomplete RBBB from duration alone: require compatible V1/V2 terminal "
        "R' morphology, a broad terminal S in I/V6, and the supplied age-resolved QRS range "
        "(110-119 ms applies only to adults). First-degree AV delay requires the supplied "
        "age-resolved PR threshold plus 1:1 conduction; never substitute an adult threshold "
        "when age is pediatric or unknown. Do not call second-degree AV block "
        "without a reproducible P:QRS pattern (progressive PR with a dropped QRS, or fixed "
        "PR with dropped QRS). Never call a QTc normal when it exceeds the supplied "
        "sex-specific deterministic upper limit."
    ),
    "L3": (
        "Do not convert a single large amplitude into a chamber diagnosis. Check age/sex, "
        "contiguous-lead consistency, axis, and competing explanations such as lead placement. "
        "For adults, QRS axis below -30 degrees is left axis deviation and above +90 degrees "
        "is right axis deviation. Use the supplied computed voltage formulas rather than "
        "adding signed S-wave amplitudes incorrectly."
    ),
    "L4": (
        "Use the supplied age route and require age-appropriate Q-wave criteria plus contiguous "
        "territorial support before calling a prior-MI pattern; if age is unknown, do not force "
        "adult Q-duration criteria. A large Q/R ratio caused by a tiny R wave is not sufficient. Check high "
        "QRS voltage and lead placement as confounders. ST-J baseline offsets alone are not "
        "proof of acute ischemia. Preserve the sign of every ST value and use ST-J, not ST-80, "
        "for a J-point elevation claim. If T amplitude and observed polarity conflict, mark "
        "that lead uncertain rather than inventing inversion. Without symptoms, serial ECG "
        "change, and troponin, do not diagnose acute MI/NSTEMI; describe an ECG pattern and "
        "its uncertainty instead."
    ),
}

_LAYER_RULES_JA = {
    "L0": "全体Q0/Q1だけで導聯別フラグを上書きせず、P/QRS/T/QTのどの測定が不可靠か明記してください。",
    "L1": "中央RRが規則的で少数の休止・外れ値だけがCVを上げている場合、全RR CVだけで心房細動と診断しないでください。",
    "L2": "QRS時間だけで不完全右脚ブロックとせず、V1/V2の終末R'、I/V6の幅広いS、提示された年齢別QRS基準を確認してください（110–119 msは成人にのみ適用）。年齢が小児または不明な場合に成人PR/QRS基準を代用しないでください。再現性のあるP:QRS関係なしに二度房室ブロックと診断しないでください。",
    "L3": "単一の高振幅だけで心腔診断を行わず、年齢・性別、連続導聯、電気軸、電極位置の可能性を確認してください。",
    "L4": "既往梗塞パターンには年齢別Q波基準と連続する領域導聯の支持を必要とし、年齢不明時に成人Q波幅基準を強制せず、小さいRによる高Q/R比だけを根拠にしないでください。症状、連続ECG変化、トロポニンがない場合、急性MI/NSTEMIと断定せずECGパターンとして記載してください。",
}


def build_layer_reasoning_prompt(
    layer_key: str,
    evidence: str,
    quality_conclusion: str = "",
    language: str = "en",
) -> str:
    """Build one isolated reasoning call for a single ECG layer."""
    if layer_key not in LAYER_KEYS:
        raise ValueError(f"Unknown layer key: {layer_key}")

    if language == "ja":
        task = _LAYER_TASKS_JA[layer_key]
        rule = _LAYER_RULES_JA[layer_key]
        system_message = f"""あなたは熟練の心電図判読医です。これは独立した {layer_key} 判定段階です。

{task}

必須ルール:
- 提示されたこの段階の測定値と、L0品質結論だけを使用してください。
- 測定器の出力は誤る可能性があります。支持所見と反証を比較してください。
- エビデンス中の決定的測定チェックは算術制約です。矛盾せず、別フィールドと競合する場合は不確実として記載してください。
- 他段階の診断、統合臨床ルール、データセットラベルは推測しないでください。
- {rule}
- 段階全体を簡潔にし、目安として700文字以内にしてください。
- 内部思考過程ではなく、監査可能な簡潔な所見・反証・結論を日本語で出力してください。"""
        quality_label = "[先行するL0品質結論]"
        evidence_label = f"[{layer_key} 測定エビデンス]"
    else:
        task = _LAYER_TASKS_EN[layer_key]
        rule = _LAYER_RULES_EN[layer_key]
        system_message = f"""You are a senior ECG interpreter performing one isolated {layer_key} adjudication stage.

{task}

Mandatory rules:
- Use only this stage's measurements and the preceding L0 quality conclusion.
- Measurement-extractor output can be wrong. Compare supporting evidence with counterevidence.
- Deterministic measurement checks in the evidence are arithmetic constraints. Do not contradict them; if another field conflicts, report the conflict as uncertain.
- Do not infer or invent other-layer diagnoses, unified clinical rules, or dataset labels.
- {rule}
- Keep the complete stage response under 350 words.
- Return concise, auditable findings rather than hidden chain-of-thought. Output in English."""
        quality_label = "[Preceding L0 quality conclusion]"
        evidence_label = f"[{layer_key} measurement evidence]"

    user_message = f"""{quality_label}
{quality_conclusion.strip() or 'Not applicable; this is L0.'}

{evidence_label}
{evidence}

Output exactly these subsections, without an outer L0-L5 heading:
#### Findings
- ...

#### Counterevidence
- ...

#### Conclusion
- ...

#### Confidence
- high / moderate / low, with one-sentence reason"""
    return f"<start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def build_layer_revision_prompt(
    original_prompt: str,
    previous_output: str,
    guardrail_notes: list[str],
    language: str = "en",
) -> str:
    notes_block = "\n".join(f"- {note}" for note in guardrail_notes)
    if language == "ja":
        instruction = f"""直前の段階判定を、以下の決定的な落とし穴に照らして再判定してください。単にキーワードに触れるだけでなく、該当する数値、反証、結論への影響を明記し、同じ4小節形式で段階全文を再出力してください。

再判定事項:
{notes_block}"""
    else:
        instruction = f"""Re-adjudicate the preceding stage against these deterministic pitfall checks. Do not merely mention the topic: cite the relevant numbers, state the counterevidence, and explain whether the stage conclusion changes. Re-output the full stage using the same four-subsection format.

Items requiring explicit adjudication:
{notes_block}"""

    conversation = original_prompt
    model_turn_start = "<start_of_turn>model\n"
    if conversation.endswith(model_turn_start):
        conversation = conversation[: -len(model_turn_start)]
        conversation += model_turn_start + previous_output.strip() + "<end_of_turn>\n"
    return conversation + f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def build_synthesis_prompt(
    layer_outputs: dict[str, str],
    clinician_notes: str,
    language: str = "en",
    validated_facts: str = "",
) -> str:
    layer_block = "\n\n".join(
        f"### {key} adjudicated conclusion\n{layer_outputs.get(key, '—')}"
        for key in LAYER_KEYS
    )
    if language == "ja":
        system_message = """あなたは最終ECG統合判定者です。L0〜L4は互いに隔離された測定段階で個別判定されています。各段階の結論と反証だけを統合し、新しい測定値や診断を作らないでください。

必須ルール:
- 「WITHHELD BY DIAGNOSTIC GATE」の段階は完全に利用不可です。その領域の所見・診断・正常判定を統合に追加せず、棄却理由だけを記載してください。
- 最有力診断の各要素は、該当段階の具体的な数値または形態で支持されなければなりません。
- 「決定的測定チェック」は数値から計算された制約であり、段階文章と矛盾する場合は測定チェックを優先してください。
- 段階間の矛盾は解消するか、未解決として明記してください。
- ECG所見と臨床疾患を区別してください。症状、連続変化、トロポニンがなければ急性MI/NSTEMIと断定しないでください。
- 不確実な二次所見を長い複合主診断に追加しないでください。
- 日本語で出力してください。"""
        notes_label = "[臨床メモ]\n"
    else:
        system_message = """You are the final ECG synthesis adjudicator. L0-L4 were assessed in separate, measurement-isolated calls. Integrate their auditable findings and counterevidence; do not invent new measurements or diagnoses.

Mandatory rules:
- A stage marked WITHHELD BY DIAGNOSTIC GATE is completely unavailable. Do not add a finding, diagnosis, or normality claim from that domain; report only the abstention reason.
- Every component of the most likely diagnosis must be supported by concrete numeric or morphologic evidence in the relevant stage.
- The deterministic measurement checks are arithmetic constraints. If stage prose conflicts with them, the deterministic checks take precedence.
- Resolve cross-layer conflicts or explicitly leave them unresolved.
- Separate an ECG pattern from a clinical disease event. Without symptoms, serial change, and troponin, do not diagnose acute MI/NSTEMI.
- Do not append uncertain secondary findings into a long compound primary diagnosis; move them to uncertainty.
- Output in English."""
        notes_label = "[Clinical notes]\n"

    user_message = f"""[Independently adjudicated stage outputs]
{layer_block}

[Deterministic measurement checks computed from JSON, without Dx]
{validated_facts.strip() or '—'}

{notes_label}{clinician_notes.strip() or '—'}

Output exactly this synthesis body, without an outer L5 heading:
#### Most Likely Diagnosis
- ...

#### Diagnostic Rationale (layer references)
- ...

#### Similar Diagnoses (ranked by similarity)
1. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...
2. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...
3. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...

#### Layer Conflicts
- ...

#### Uncertainty and Review Priorities
- ..."""
    return f"<start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def _strip_outer_stage_heading(text: str, layer_key: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        rf"^###\s*{re.escape(layer_key)}\.[^\n]*\n",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def compose_sequential_output(
    layer_outputs: dict[str, str], synthesis_output: str, language: str = "en"
) -> str:
    titles = _LAYER_TITLES_JA if language == "ja" else _LAYER_TITLES_EN
    sections = []
    for key in LAYER_KEYS:
        body = _strip_outer_stage_heading(layer_outputs.get(key, ""), key)
        sections.append(f"### {key}. {titles[key]}\n{body or '- unavailable'}")
    synthesis_body = _strip_outer_stage_heading(synthesis_output, "L5")
    synthesis_title = "統合" if language == "ja" else "Synthesis"
    sections.append(f"### L5. {synthesis_title}\n{synthesis_body or '- unavailable'}")
    return "\n\n".join(sections)


def build_layered_prompt(
    layer_evidence: dict[str, str],
    report_text: str,
    clinician_notes: str,
    language: str = "en",
) -> str:
    if language == "ja":
        system_message = """あなたは熟練の心電図判読医です。診断結論を先に決めてから理由をこじつけるのではなく、下から積み上げる形で必ず次の順序でレイヤーごとに推論してください。各レイヤーは、そのレイヤーに提示された生の計測値・導聯別所見のみを根拠にすること。

L0. 信号品質ゲート — どの導聯・測定値が信頼できないかを述べ、以降のレイヤーでの重み付けを下げること。
L1. 調律 — L1の証拠のみで調律の起源と規則性を判定すること。
L2. 伝導・間期 — L2の証拠のみでPR/QRS/QT/AVブロック/脚ブロックを判定すること。特に、いずれかの導聯の生QRS幅がコンセンサスのQRS幅クラスと矛盾していないか必ず確認すること。
L3. 電気軸・心腔形態 — L3の証拠のみで軸偏位・心房/心室拡大を判定すること。
L4. ST-T・虚血/梗塞 — L4の証拠のみでQ波・ST・T波所見を判定すること。特に、病的Q波のフラグが真の梗塞ではなく高いQRS電位（LVHなど）で説明できないか必ず確認すること。
L5. 統合 — L1〜L4の「結論」（生の数値ではなく）を統合し、最も可能性の高い診断、類似度順の鑑別診断3件（共通点・鑑別点付き）、レイヤー間の矛盾点、不確実性と再確認事項を述べること。

重要な制約:
- 提供されたレイヤー証拠に含まれる具体的な数値を必ず引用すること。
- 他の自動診断システムの結論は一切提供されていない。存在しないものを参照したり、ラベルを創作したりしないこと。
- 出力は日本語で行うこと。"""
        report_label = "[補足レポート（Dxラベルおよびルールエンジンの結論は除去済み）]"
        notes_label = "[ユーザーメモ]"
        output_instruction = "必ず次の形式で、レイヤーの順に出力してください:"
    else:
        system_message = """You are a senior ECG interpretation specialist. Reason bottom-up, strictly in the following layer order — do not decide the diagnosis first and rationalize afterward. Each layer must be justified only by the raw evidence given for that layer.

L0. Signal Quality Gate — state which leads/measurements are unreliable and must be down-weighted in later layers.
L1. Rhythm — determine rhythm origin and regularity using only L1 evidence.
L2. Conduction & Intervals — determine PR/QRS/QT/AV-block/BBB findings using only L2 evidence. Explicitly check whether any per-lead raw QRS duration disagrees with the consensus QRS width class.
L3. Axis & Chamber Morphology — determine axis deviation and atrial/ventricular enlargement using only L3 evidence.
L4. ST-T / Ischemia & Infarction — determine Q-wave, ST, and T-wave findings using only L4 evidence. Explicitly check whether any pathological Q-wave flag could be explained by high QRS voltage (e.g. LVH) rather than true infarction.
L5. Synthesis — combine the CONCLUSIONS of L1-L4 (not raw numbers) into the most likely diagnosis, 3 ranked differential diagnoses with shared features and key differentiator each, an explicit note on any conflicts between layers, and uncertainty/review priorities.

Hard constraints:
- Cite concrete numeric evidence from the layer evidence provided.
- No output from any other automated diagnostic system, including the unified clinical rule engine, has been provided to you. Do not reference or invent such labels — reason only from the measurements below.
- Output must be in English."""
        report_label = "[Supplementary Report (Dx labels and rule-engine conclusions removed)]"
        notes_label = "[User Notes]"
        output_instruction = "Now produce your layered analysis, in layer order, using exactly this format:"

    user_message = f"""### L0 Evidence (Signal Quality Gate)
{layer_evidence.get('L0', '—')}

### L1 Evidence (Rhythm)
{layer_evidence.get('L1', '—')}

### L2 Evidence (Conduction & Intervals)
{layer_evidence.get('L2', '—')}

### L3 Evidence (Axis & Chamber Morphology)
{layer_evidence.get('L3', '—')}

### L4 Evidence (ST-T / Ischemia & Infarction)
{layer_evidence.get('L4', '—')}

{report_label}
{report_text or '—'}

{notes_label}
{clinician_notes.strip() or '—'}

{output_instruction}
### L0. Signal Quality Gate
- ...

### L1. Rhythm
- ...

### L2. Conduction & Intervals
- ...

### L3. Axis & Chamber Morphology
- ...

### L4. ST-T / Ischemia & Infarction
- ...

### L5. Synthesis
#### Most Likely Diagnosis
- ...

#### Diagnostic Rationale (layer references)
- ...

#### Similar Diagnoses (ranked by similarity)
1. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...
2. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...
3. Diagnosis: ...
   Shared features: ...
   Key differentiator: ...

#### Layer Conflicts
- ...

#### Uncertainty and Review Priorities
- ..."""

    return f"<start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def build_revision_prompt(
    original_prompt: str,
    previous_output: str,
    guardrail_notes: list[str],
    language: str = "en",
) -> str:
    notes_block = "\n".join(f"- {note}" for note in guardrail_notes)
    if language == "ja":
        instruction = f"""以下はあなたが直前に出力した分層診断です。抽出パイプラインの既知の落とし穴チェッカーが、次の項目について再確認が必要だと検出しました。該当する場合は関連する証拠を再検討し、結論を修正してください。該当しない場合は、なぜ該当しないかを一文で明記してください。その上で、同じフォーマット（L0〜L5、Most Likely Diagnosisなど）で修正後の全文を出力してください。

再確認事項:
{notes_block}"""
    else:
        instruction = f"""Below is the layered diagnostic output you just produced. An automated pitfall checker flagged the following items on the extraction pipeline for re-verification. If an item applies, re-examine the relevant evidence and revise your conclusion; if it does not apply, state in one sentence why not. Then output the full corrected analysis again in the same format (L0-L5, including Most Likely Diagnosis etc.).

Items to re-verify:
{notes_block}"""

    conversation = original_prompt
    model_turn_start = "<start_of_turn>model\n"
    if conversation.endswith(model_turn_start):
        conversation = conversation[: -len(model_turn_start)]
        conversation += model_turn_start + previous_output.strip() + "<end_of_turn>\n"
    return conversation + f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def parse_layered_output(text: str) -> dict[str, Any]:
    section_pattern = re.compile(r"###\s*(L[0-4])\.[^\n]*\n(.*?)(?=###\s*L[0-5]\.|\Z)", re.DOTALL)
    layers = {match.group(1).upper(): match.group(2).strip() for match in section_pattern.finditer(text)}

    l5_match = re.search(r"###\s*L5\.[^\n]*\n(.*)\Z", text, flags=re.DOTALL)
    l5_text = l5_match.group(1).strip() if l5_match else ""

    top1_match = re.search(r"Most Likely Diagnosis\s*\n?\s*-\s*(.+)", l5_text, flags=re.IGNORECASE)
    similar_matches = re.findall(r"\d+\.\s*Diagnosis:\s*(.+)", l5_text)
    conflicts_match = re.search(
        r"Layer Conflicts\s*(.+?)(?=####|\Z)", l5_text, flags=re.IGNORECASE | re.DOTALL
    )
    uncertainty_match = re.search(
        r"Uncertainty and Review Priorities\s*(.+)", l5_text, flags=re.IGNORECASE | re.DOTALL
    )

    top1_raw = top1_match.group(1).strip() if top1_match else None
    similar_raw = [item.strip() for item in similar_matches[:3]]
    return {
        "layers": layers,
        "l5_synthesis": l5_text,
        "top1_raw": top1_raw,
        "top1_normalized": normalize_diagnosis(top1_raw),
        "similar_raw": similar_raw,
        "similar_normalized": [normalize_diagnosis(item) for item in similar_raw],
        "layer_conflicts": conflicts_match.group(1).strip() if conflicts_match else "",
        "uncertainty": uncertainty_match.group(1).strip() if uncertainty_match else "",
    }


def check_q_wave_lvh_false_positive(data: dict[str, Any]) -> list[str]:
    """Guardrail for a known extractor pitfall: high QRS amplitude (e.g. LVH)
    can trip the pathological-Q/old-MI flag without a true infarct Q wave.
    Cross-checks pathological_q_leads against mi_evidence.q_wave_mi_ratio."""
    interp = data.get("interpretation", {})
    pathological = interp.get("pathological_q_leads") or {}
    flagged_leads = [lead for lead, is_path in pathological.items() if is_path]
    if not flagged_leads:
        return []

    lvh_criteria = interp.get("lvh_voltage_criteria") or []
    lvh_class = interp.get("lvh_class")
    if not lvh_criteria and not lvh_class:
        return []

    q_by_lead = ((interp.get("mi_evidence") or {}).get("q_by_lead")) or {}
    suspect_leads = [
        lead
        for lead in flagged_leads
        if (q_by_lead.get(lead) or {}).get("q_wave_mi_ratio") is False
        and (q_by_lead.get(lead) or {}).get("significant_q")
    ]
    if not suspect_leads:
        return []

    return [
        f"pathological_q_leads flags {suspect_leads} while LVH voltage criteria "
        f"({_fmt_list(lvh_criteria)}, class={lvh_class or 'unclassed'}) are present and "
        f"q_wave_mi_ratio is False for these leads. This pattern has previously produced "
        f"false-positive old-MI calls driven by high QRS amplitude rather than a true "
        f"infarct Q wave — confirm q_r_ratio/q_wave_mi_ratio before citing prior MI on "
        f"these leads."
    ]


def check_qrs_consensus_bbb_gap(data: dict[str, Any]) -> list[str]:
    """Guardrail for a known extractor pitfall: the global consensus QRS
    duration can undercount versus per-lead raw values near 100-120ms,
    which has caused missed LBBB/IVCD calls."""
    gf = data.get("global_features", {})
    interp = data.get("interpretation", {})
    representative_leads = data.get("representative_leads", {})

    width_class = interp.get("qrs_width_class")
    bbb = interp.get("bundle_branch_block")
    if bbb or width_class not in (None, "normal"):
        return []

    wide_leads = []
    for lead, info in representative_leads.items():
        raw_qrs = (info or {}).get("params", {}).get("qrs_ms")
        if isinstance(raw_qrs, (int, float)) and 100 <= raw_qrs < 120:
            wide_leads.append((lead, raw_qrs))
    if not wide_leads:
        return []

    leads_str = ", ".join(f"{lead}={val:.0f}ms" for lead, val in sorted(wide_leads, key=lambda x: -x[1]))
    consensus_qrs = gf.get("qrs_ms")
    return [
        f"Consensus QRS is {_fmt_num(consensus_qrs, 0, 'ms')} (width class={width_class or 'normal'}, "
        f"BBB={bbb or 'none'}) but per-lead raw QRS reaches {leads_str}. Global consensus QRS has "
        f"previously undercounted versus per-lead raw near 100-120ms, causing missed LBBB/IVCD — "
        f"review per-lead morphology before ruling out conduction delay."
    ]


def check_rr_outlier_driven_irregularity(data: dict[str, Any]) -> list[str]:
    """Flag records whose global RR variability is driven by a few outliers.

    This is intentionally calculated from the raw RR series rather than the
    extractor's probable-AF field.  It does not rule out AF; it forces L1 to
    explain why the central rhythm is regular or irregular before naming it.
    """
    intervals = _raw_rr_intervals(data)
    if len(intervals) < 6:
        return []
    rr_median = median(intervals)
    rr_mean = sum(intervals) / len(intervals)
    all_cv = pstdev(intervals) / rr_mean if rr_mean else 0.0
    lower = 0.75 * rr_median
    upper = 1.25 * rr_median
    central = [value for value in intervals if lower <= value <= upper]
    outliers = [value for value in intervals if value < lower or value > upper]
    if len(central) < 4 or not outliers:
        return []
    central_mean = sum(central) / len(central)
    central_cv = pstdev(central) / central_mean if central_mean else 0.0
    outlier_fraction = len(outliers) / len(intervals)
    if all_cv < 0.12 or central_cv > 0.06 or outlier_fraction > 0.25:
        return []
    return [
        f"Raw RR all-interval CV is {all_cv:.4f}, but the central {len(central)}/{len(intervals)} "
        f"intervals have CV {central_cv:.4f}; only {len(outliers)} isolated intervals "
        f"({_fmt_list([f'{value:.0f}ms' for value in outliers])}) drive most irregularity. "
        "Do not call persistent atrial fibrillation from the global CV alone; adjudicate the "
        "organized P-wave evidence and whether the outliers are pauses, ectopy, or artifact."
    ]


def run_feature_guardrails(data: dict[str, Any]) -> list[str]:
    """Legacy guardrails, including comparisons to extractor interpretations.

    Kept for display/backward-compatible callers only.  Independent MedGemma
    evaluation uses :func:`run_independent_feature_guardrails` below so these
    rule-engine answers cannot enter a revision prompt.
    """
    return (
        check_q_wave_lvh_false_positive(data)
        + check_qrs_consensus_bbb_gap(data)
        + check_rr_outlier_driven_irregularity(data)
    )


def run_independent_feature_guardrails(data: dict[str, Any]) -> list[str]:
    """Guardrails derived solely from raw measurements, never interpretation."""
    notes = check_rr_outlier_driven_irregularity(data)
    age = _resolved_patient_age(data)

    consensus_qrs = (data.get("global_features") or {}).get("qrs_ms")
    wide_leads: list[tuple[str, float]] = []
    if isinstance(consensus_qrs, (int, float)) and consensus_qrs < 100:
        for lead, info in (data.get("representative_leads") or {}).items():
            raw_qrs = ((info or {}).get("params") or {}).get("qrs_ms")
            if isinstance(raw_qrs, (int, float)) and 100 <= raw_qrs < 120:
                wide_leads.append((str(lead), float(raw_qrs)))
    if wide_leads:
        leads = ", ".join(
            f"{lead}={value:.0f}ms"
            for lead, value in sorted(wide_leads, key=lambda item: -item[1])
        )
        notes.append(
            f"Measured consensus QRS is {consensus_qrs:.0f}ms, but raw per-lead QRS reaches "
            f"{leads}; the consensus may undercount the widest lead. Review per-lead conduction "
            "morphology before ruling out a borderline "
            "conduction delay."
        )

    voltage_criteria, voltage_audit = _lvh_voltage_facts(data)
    q_wave_leads, q_wave_criterion = _age_resolved_q_wave_leads(data, age)
    if voltage_criteria and q_wave_leads:
        notes.append(
            f"Age-resolved Q-wave findings ({q_wave_criterion}) are present in "
            f"{_fmt_list(q_wave_leads)} while raw voltages meet "
            f"{_fmt_list(voltage_criteria)} ({'; '.join(voltage_audit)}). Adjudicate high-voltage "
            "confounding before calling an infarct Q-wave pattern."
        )
    return notes


def clean_model_output(raw_text: str) -> str:
    """Remove model-control and hidden-reasoning artifacts from saved output."""
    cleaned = re.sub(
        r"<unused\d+>\s*thought\b.*?<unused\d+>",
        "",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?unused\d+>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _claimed_t_inversion_leads(raw_output: str) -> set[str]:
    claimed: set[str] = set()
    for line in raw_output.splitlines():
        lowered = line.lower()
        phrase_match = re.search(r"t[- ]wave inversions?", lowered)
        if not phrase_match:
            continue
        # A findings bullet may mention ST leads before the T-wave sentence.
        # Restrict extraction to the T-inversion clause so those leads are not
        # incorrectly attributed to the T-wave claim.
        clause = line[phrase_match.start() :]
        clause = clause.split(".", 1)[0]
        for start, end in re.findall(r"V([1-6])\s*[-–]\s*V([1-6])", clause, flags=re.IGNORECASE):
            lo, hi = sorted((int(start), int(end)))
            claimed.update(f"V{idx}" for idx in range(lo, hi + 1))
        for lead in STANDARD_12_LEADS:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(lead)}(?![A-Za-z0-9])", clause, flags=re.IGNORECASE):
                claimed.add(lead)
    return claimed


def _has_nonnegated_claim(raw_output: str, patterns: tuple[str, ...]) -> bool:
    """Return True only when a diagnosis is asserted, not explicitly rejected.

    Stage outputs deliberately include counterevidence, so a plain substring
    search would misread sentences such as "first-degree AV block is not
    supported" as a positive diagnosis.
    """
    negation = re.compile(
        r"\b(?:no|not|without|absent|unlikely|unsupported|unconfirmed|"
        r"cannot|can't|doesn't|does not|against|insufficient|exclude[ds]?|"
        r"rule[ds]? out)\b",
        flags=re.IGNORECASE,
    )
    for clause in re.split(r"(?<=[.!?])\s+|\n", raw_output):
        if not any(re.search(pattern, clause, flags=re.IGNORECASE) for pattern in patterns):
            continue
        if negation.search(clause):
            continue
        return True
    return False


_PROLONGED_QT_PATTERNS = (
    r"prolonged\s+qt(?:c[bf]?|(?:\s+interval))?(?!\s+dispersion)\b",
    r"qt(?:c[bf]?)?(?:\s+interval)?\s+(?:is\s+)?prolonged",
)


def run_stage_claim_guardrails(
    data: dict[str, Any], stage_outputs: dict[str, str]
) -> dict[str, list[str]]:
    """Detect direct contradictions between stage prose and measured values."""
    notes: dict[str, list[str]] = {key: [] for key in LAYER_KEYS}
    gf = data.get("global_features") or {}
    patient = _patient_metadata(data)
    sex = str(patient.get("sex") or "")
    age = _resolved_patient_age(data)

    hr = gf.get("heart_rate_bpm")
    l1 = stage_outputs.get("L1", "").lower()
    rate_class, brady_limit, tachy_limit = _rate_class(hr, age)
    if not age["known"] and _has_nonnegated_claim(
        stage_outputs.get("L1", ""),
        (r"\bbradycard(?:ia|ic)\b", r"\btachycard(?:ia|ic)\b"),
    ):
        notes["L1"].append(
            "Age is unknown, so an age-specific bradycardia/tachycardia threshold cannot be "
            "selected. Do not justify the rate class with adult cutoffs; obtain age or mark it uncertain."
        )
    if rate_class.startswith("bradycardic") and "brady" not in l1:
        notes["L1"].append(
            f"HR is {_fmt_num(hr, 1, ' bpm')}, below the resolved age-specific bradycardia "
            f"limit of {_fmt_num(brady_limit, 1, ' bpm')}. "
            "Name the rate as bradycardic rather than merely 'sinus rhythm'."
        )
    if rate_class.startswith("tachycardic") and "tachy" not in l1:
        notes["L1"].append(
            f"HR is {_fmt_num(hr, 1, ' bpm')}, above the resolved age-specific tachycardia "
            f"limit of {_fmt_num(tachy_limit, 1, ' bpm')}. "
            "Name the rate as tachycardic and adjudicate the rhythm mechanism."
        )
    organized_atrial, organized_atrial_leads, organized_atrial_audit = (
        _organized_atrial_residual_facts(data)
    )
    flutter_claim = _has_nonnegated_claim(
        stage_outputs.get("L1", ""), (r"atrial\s+flutter",)
    )
    if organized_atrial and not flutter_claim:
        notes["L1"].append(
            "Raw QRST-residual measurements independently show concordant rapid organized "
            f"atrial activity in {_fmt_list(organized_atrial_leads)} "
            f"({' ; '.join(organized_atrial_audit)}). Explicitly adjudicate an atrial-flutter "
            "ECG pattern; sinus tachycardia alone does not explain this multilead activity."
        )

    l2 = stage_outputs.get("L2", "").lower()
    pr = gf.get("pr_ms")
    first_degree_claim = _has_nonnegated_claim(
        stage_outputs.get("L2", ""),
        (r"first[- ]degree[^.\n]{0,40}(?:av|atrioventricular)",),
    )
    pr_value = _numeric_measurement(pr)
    _pr_result, pr_limits = _pr_class(pr, age)
    first_degree_limit = pr_limits.get("first_degree") if pr_limits else None
    if first_degree_claim and first_degree_limit is None:
        notes["L2"].append(
            "Age is unknown, so the first-degree AV-delay PR threshold cannot be selected. "
            "Do not substitute the adult threshold; downgrade the claim pending age."
        )
    if (
        pr_value is not None
        and first_degree_limit is not None
        and pr_value < first_degree_limit
        and first_degree_claim
    ):
        notes["L2"].append(
            f"Measured PR is {pr_value:.0f} ms, below the resolved age-specific first-degree "
            f"threshold of {first_degree_limit:.0f} ms, so first-degree AV block/delay is not "
            "supported. Remove that diagnosis."
        )
    if (
        pr_value is not None
        and first_degree_limit is not None
        and pr_value >= first_degree_limit
        and not (
        first_degree_claim or "prolonged pr" in l2 or "pr prolong" in l2
        )
    ):
        notes["L2"].append(
            f"Measured PR is {pr_value:.0f} ms (age-specific threshold "
            f"{first_degree_limit:.0f} ms). With confirmed 1:1 conduction this supports "
            "first-degree AV delay; explicitly adjudicate it."
        )

    qtc_b = gf.get("qtc_bazett_ms")
    qtc_f = gf.get("qtc_fridericia_ms")
    qtc_value = max(value for value in (qtc_b, qtc_f) if isinstance(value, (int, float))) if any(
        isinstance(value, (int, float)) for value in (qtc_b, qtc_f)
    ) else None
    qtc_class, qtc_limits_text = _qtc_class(qtc_value, sex, age)
    normal_qt_claim = any(term in l2 for term in ("normal qt", "qt interval is normal", "qtc is normal"))
    qtc_abnormal_high = "prolonged" in qtc_class
    if qtc_abnormal_high and (normal_qt_claim or "prolong" not in l2):
        notes["L2"].append(
            f"QTc reaches {_fmt_num(qtc_value, 0, ' ms')} ({qtc_limits_text}). It cannot be "
            "called normal; report the age-resolved prolonged/borderline-prolonged QTc and "
            "its measurement reliability."
        )
    prolonged_qt_claim = _has_nonnegated_claim(
        stage_outputs.get("L2", ""),
        _PROLONGED_QT_PATTERNS,
    )
    if not age["known"] and prolonged_qt_claim:
        notes["L2"].append(
            "Age is unknown, so an age-appropriate QTc threshold cannot be selected. Do not "
            "assert prolonged QTc from an adult cutoff; report the measurement and uncertainty."
        )
    if qtc_class in {"within the usual adult range", "normal by age-adjusted pediatric limits"} and prolonged_qt_claim:
        notes["L2"].append(
            f"QTc is {_fmt_num(qtc_value, 0, ' ms')}, within the supplied age-specific limits; a prolonged "
            "QT diagnosis is not supported by this measurement."
        )

    l3 = stage_outputs.get("L3", "").lower()
    qrs_axis = gf.get("qrs_axis_deg")
    axis_class = _qrs_axis_class(qrs_axis, age)
    left_axis_claim = _has_nonnegated_claim(
        stage_outputs.get("L3", ""), (r"left\s+axis\s+deviation",)
    )
    right_axis_claim = _has_nonnegated_claim(
        stage_outputs.get("L3", ""), (r"right\s+axis\s+deviation",)
    )
    if not age["known"] and (left_axis_claim or right_axis_claim):
        notes["L3"].append(
            "Age is unknown, so age-appropriate QRS-axis limits cannot be selected. Do not "
            "substitute the adult -30°/+90° limits; report the measured axis as indeterminate."
        )
    if axis_class.startswith("left axis deviation") and (
        "normal qrs axis" in l3 or "left axis" not in l3
    ):
        notes["L3"].append(
            f"QRS axis is {_fmt_num(qrs_axis, 0, '°')}, classified as {axis_class}; it must not "
            "be called normal."
        )
    if axis_class.startswith("right axis deviation") and (
        "normal qrs axis" in l3 or "right axis" not in l3
    ):
        notes["L3"].append(
            f"QRS axis is {_fmt_num(qrs_axis, 0, '°')}, classified as {axis_class}; it must not "
            "be called normal."
        )
    if axis_class in {"normal adult axis", "normal age-adjusted pediatric axis"}:
        if left_axis_claim or right_axis_claim:
            claimed = "left" if left_axis_claim else "right"
            notes["L3"].append(
                f"QRS axis is {_fmt_num(qrs_axis, 0, '°')}, classified as {axis_class}. Do not "
                f"diagnose {claimed} QRS-axis deviation from the P or T axis."
            )

    ptf_v1 = gf.get("ptf_v1_mv_ms")
    if (
        age["known"]
        and isinstance(ptf_v1, (int, float))
        and abs(ptf_v1) < 4.0
        and _has_nonnegated_claim(
            stage_outputs.get("L3", ""),
            (r"left\s+atrial\s+enlargement", r"(?:^|\s)lae\b"),
        )
    ):
        notes["L3"].append(
            f"PTF-V1 magnitude is only {abs(ptf_v1):.3f} mV·ms, below the common 4.0 mV·ms "
            "abnormal threshold. Do not infer LAE from this value alone."
        )

    lvh_criteria, lvh_audit = _lvh_voltage_facts(data)
    lvh_claim = _has_nonnegated_claim(
        stage_outputs.get("L3", ""),
        (r"left\s+ventricular\s+hypertrophy", r"(?:^|\s)lvh\b"),
    )
    if not age["known"] and lvh_claim:
        notes["L3"].append(
            "Age is unknown, so age-appropriate ventricular-voltage thresholds cannot be "
            "selected. Do not apply adult LVH voltage criteria; downgrade the claim."
        )
    if lvh_criteria and not any(term in l3 for term in ("lvh", "left ventricular hypertroph")):
        notes["L3"].append(
            "Raw voltages meet ECG LVH voltage criteria "
            f"({_fmt_list(lvh_criteria)}; {'; '.join(lvh_audit)}). State the ECG voltage pattern "
            "while distinguishing it from an anatomic LVH diagnosis."
        )
    if lvh_criteria and any(term in l3 for term in ("no criteria met for left ventricular", "no lvh")):
        notes["L3"].append(
            f"The statement that no LVH voltage criterion is met contradicts the computed raw "
            f"criteria ({_fmt_list(lvh_criteria)}). Recalculate using absolute S-wave depth."
        )

    irbbb_claim = _has_nonnegated_claim(
        stage_outputs.get("L2", ""),
        (r"incomplete\s+right\s+bundle\s+branch\s+block", r"(?:^|\s)irbbb\b"),
    )
    if not age["known"] and irbbb_claim:
        notes["L2"].append(
            "Age is unknown, so the incomplete-RBBB QRS-duration range cannot be selected. "
            "Do not substitute the adult 110-119 ms range; require age and compatible morphology."
        )

    l4_raw = stage_outputs.get("L4", "")
    l4 = l4_raw.lower()
    observed_negative, polarity_conflicts = _t_polarity_facts(data)
    claimed_t = _claimed_t_inversion_leads(l4_raw)
    unsupported_t = sorted(claimed_t - set(observed_negative), key=STANDARD_12_LEADS.index)
    if unsupported_t:
        notes["L4"].append(
            f"The prose claims T-wave inversion in {_fmt_list(unsupported_t)}, but the JSON "
            f"observed-polarity field is not negative there. Observed-negative leads are "
            f"{_fmt_list(observed_negative)}; field conflicts are {_fmt_list(polarity_conflicts)}. "
            "Remove unsupported inversion claims and mark conflicting leads uncertain."
        )
    if ("widespread t-wave inversion" in l4 or "widespread t wave inversion" in l4) and len(observed_negative) < 4:
        notes["L4"].append(
            f"'Widespread T-wave inversion' is unsupported: observed-negative polarity occurs in "
            f"only {_fmt_list(observed_negative)}."
        )
    avr_infarct_claim = any(
        "avr" in sentence.lower()
        and any(term in sentence.lower() for term in ("prior mi", "prior myocardial", "infarct"))
        and not any(term in sentence.lower() for term in ("no prior", "does not", "not support"))
        for sentence in re.split(r"(?<=[.!?])\s+|\n", l4_raw)
    )
    if avr_infarct_claim:
        notes["L4"].append(
            "A Q/QS pattern in aVR alone is not contiguous territorial evidence for prior MI. "
            "Do not use isolated aVR to localize anterior/septal infarction."
        )
    return notes


def _guardrail_target_layer(note: str) -> str:
    lowered = note.lower()
    if "raw rr" in lowered or "atrial fibrillation" in lowered:
        return "L1"
    if "consensus qrs" in lowered or "per-lead raw qrs" in lowered:
        return "L2"
    return "L4"


def _guardrail_required_layers(note: str) -> set[str]:
    """Return every evidence layer a revision note depends upon."""
    lowered = note.lower()
    if "raw rr" in lowered or "atrial fibrillation" in lowered:
        return {"L1"}
    if "consensus qrs" in lowered or "per-lead raw qrs" in lowered:
        return {"L2"}
    if "voltage" in lowered and ("q wave" in lowered or "q-wave" in lowered):
        return {"L3", "L4"}
    return {_guardrail_target_layer(note)}


def _synthesis_guardrail_target_layer(note: str) -> str | None:
    """Classify a final guardrail so gate-suppressed domains stay withheld."""
    lowered = note.lower()
    if any(token in lowered for token in ("heart rate", " hr ", "atrial flutter", "atrial fibrillation", "tachycard", "bradycard")):
        return "L1"
    if any(token in lowered for token in ("qrs axis", "axis deviation", "lvh", "ventricular hypertroph", "ptf-v1")):
        return "L3"
    if any(token in lowered for token in ("pr is", "measured pr", "qtc", "qt interval", "qrs", "bundle branch")):
        return "L2"
    if any(token in lowered for token in ("t-wave", "t wave", "infarct", "myocardial", "stemi", "nstemi")):
        return "L4"
    return None


def _guardrail_addressed(note: str, raw_output: str) -> bool:
    """Require evidence + counterevidence + adjudication, not a topic word."""
    output_lower = raw_output.lower()
    if "lvh" in note.lower() or "voltage" in note.lower():
        q_evidence = any(
            kw in output_lower
            for kw in ("q/r", "q_r", "mi ratio", "q wave", "q-wave", "infarct")
        )
        voltage_evidence = any(
            kw in output_lower for kw in ("lvh", "hypertroph", "voltage", "amplitude")
        )
        adjudication = any(
            kw in output_lower
            for kw in ("false positive", "confound", "does not support", "insufficient", "unlikely", "cannot")
        )
        return q_evidence and voltage_evidence and adjudication
    if "consensus qrs" in note.lower():
        duration_evidence = any(
            kw in output_lower for kw in ("consensus", "per-lead", "raw qrs", "duration")
        )
        morphology_evidence = any(
            kw in output_lower for kw in ("morphology", "r'", "terminal s", "v1", "v2", "i/v6")
        )
        adjudication = any(
            kw in output_lower
            for kw in ("support", "does not", "insufficient", "borderline", "review", "cannot")
        )
        return duration_evidence and morphology_evidence and adjudication
    if "raw rr" in note.lower():
        robust_evidence = any(
            kw in output_lower for kw in ("central", "median", "outlier", "isolated", "pause")
        )
        rhythm_evidence = any(
            kw in output_lower for kw in ("atrial fibrillation", "af", "sinus", "organized p", "irregular")
        )
        adjudication = any(
            kw in output_lower for kw in ("persistent", "does not", "insufficient", "unlikely", "artifact", "ectopy")
        )
        return robust_evidence and rhythm_evidence and adjudication
    return False


def _has_contiguous_pathological_q_support(
    data: dict[str, Any], age: dict[str, Any] | None = None
) -> bool | None:
    resolved_age = age or _resolved_patient_age(data)
    if not resolved_age["known"]:
        return None
    qualifying = set(_age_resolved_q_wave_leads(data, resolved_age)[0])
    territories = (
        {"II", "III", "aVF"},
        {"V1", "V2", "V3", "V4"},
        {"I", "aVL", "V5", "V6"},
    )
    return any(len(qualifying & territory) >= 2 for territory in territories)


def run_synthesis_guardrails(
    data: dict[str, Any], parsed: dict[str, Any], clinician_notes: str
) -> list[str]:
    """Check high-risk claims in the final synthesis against raw evidence."""
    top1 = (parsed.get("top1_raw") or "").lower()
    if not top1:
        return ["The synthesis did not provide a parseable Most Likely Diagnosis."]

    notes: list[str] = []
    synthesis_text_raw = parsed.get("l5_synthesis") or parsed.get("diagnostic_rationale") or ""
    synthesis_text = synthesis_text_raw.lower()
    primary_match = re.search(
        r"#{3,4}\s*Most Likely Diagnosis\s*(.*?)(?=\n#{3,4}\s|\Z)",
        synthesis_text_raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    primary_claims = primary_match.group(1).strip() if primary_match else (parsed.get("top1_raw") or "")
    similar_heading = re.search(
        r"\n#{3,4}\s*Similar Diagnoses", synthesis_text_raw, flags=re.IGNORECASE
    )
    adjudicated_claims = (
        synthesis_text_raw[: similar_heading.start()]
        if similar_heading
        else synthesis_text_raw
    )
    gf = data.get("global_features") or {}
    patient = _patient_metadata(data)
    sex = str(patient.get("sex") or "")
    age = _resolved_patient_age(data)

    pr = gf.get("pr_ms")
    pr_value = _numeric_measurement(pr)
    _pr_result, pr_limits = _pr_class(pr, age)
    first_degree_limit = pr_limits.get("first_degree") if pr_limits else None
    first_degree_claim = _has_nonnegated_claim(
        primary_claims, (r"first[- ]degree[^.\n]{0,40}(?:av|atrioventricular)",)
    )
    if first_degree_claim and first_degree_limit is None:
        notes.append(
            "The synthesis asserts first-degree AV block/delay, but age is unknown and the "
            "age-specific PR threshold cannot be selected. Do not substitute the adult cutoff; "
            "downgrade the claim pending age."
        )
    if (
        pr_value is not None
        and first_degree_limit is not None
        and pr_value < first_degree_limit
        and first_degree_claim
    ):
        notes.append(
            f"The synthesis asserts first-degree AV block/delay, but measured PR is "
            f"{pr_value:.0f} ms, below the resolved age-specific threshold of "
            f"{first_degree_limit:.0f} ms. Remove the unsupported AV-block diagnosis."
        )

    hr = gf.get("heart_rate_bpm")
    rate_class, brady_limit, tachy_limit = _rate_class(hr, age)
    if not age["known"] and _has_nonnegated_claim(
        primary_claims,
        (r"\bbradycard(?:ia|ic)\b", r"\btachycard(?:ia|ic)\b"),
    ):
        notes.append(
            "The synthesis assigns a bradycardic/tachycardic rate class while age is unknown. "
            "Do not apply adult rate cutoffs; obtain age or mark the rate class uncertain."
        )
    if (
        rate_class.startswith("bradycardic")
        and any(term in top1 for term in ("sinus", "rhythm"))
        and "brady" not in top1
    ):
        notes.append(
            f"The most likely diagnosis omits bradycardia despite HR {float(hr):.1f} bpm "
            f"(resolved age-specific limit {_fmt_num(brady_limit, 1, ' bpm')}). "
            "Name sinus bradycardia when sinus origin is supported."
        )
    if (
        rate_class.startswith("tachycardic")
        and "tachy" not in synthesis_text
    ):
        notes.append(
            f"The synthesis omits the tachycardic rate despite HR {float(hr):.1f} bpm "
            f"(resolved age-specific limit {_fmt_num(tachy_limit, 1, ' bpm')}). "
            "Include tachycardia/rate response in the final interpretation."
        )

    qrs_axis = gf.get("qrs_axis_deg")
    axis_class = _qrs_axis_class(qrs_axis, age)
    left_axis_claim = _has_nonnegated_claim(primary_claims, (r"left\s+axis\s+deviation",))
    right_axis_claim = _has_nonnegated_claim(primary_claims, (r"right\s+axis\s+deviation",))
    if not age["known"] and (left_axis_claim or right_axis_claim):
        notes.append(
            "The synthesis assigns QRS-axis deviation while age is unknown. Do not substitute "
            "adult -30°/+90° limits; report the measured axis as age-indeterminate."
        )
    if axis_class.startswith("left axis deviation") and (
        "normal qrs axis" in synthesis_text or "left axis" not in synthesis_text
    ):
        notes.append(
            f"QRS axis {_fmt_num(qrs_axis, 0, '°')} is {axis_class}. The synthesis must not "
            "call it normal or omit this measured abnormality."
        )
    if axis_class.startswith("right axis deviation") and (
        "normal qrs axis" in synthesis_text or "right axis" not in synthesis_text
    ):
        notes.append(
            f"QRS axis {_fmt_num(qrs_axis, 0, '°')} is {axis_class}. The synthesis must not "
            "call it normal or omit this measured abnormality."
        )
    if axis_class in {"normal adult axis", "normal age-adjusted pediatric axis"}:
        if left_axis_claim or right_axis_claim:
            claimed = "left" if left_axis_claim else "right"
            notes.append(
                f"QRS axis {_fmt_num(qrs_axis, 0, '°')} is classified as {axis_class}. Remove "
                f"the unsupported {claimed} QRS-axis deviation; P/T axes do not define the QRS axis."
            )

    organized_atrial, organized_atrial_leads, organized_atrial_audit = (
        _organized_atrial_residual_facts(data)
    )
    if organized_atrial and not _has_nonnegated_claim(primary_claims, (r"atrial\s+flutter",)):
        notes.append(
            "The most likely diagnosis omits atrial flutter despite strict independent "
            f"multilead residual support in {_fmt_list(organized_atrial_leads)} "
            f"({' ; '.join(organized_atrial_audit)}). Reconcile the rapid organized atrial "
            "activity with the ventricular response and include the atrial-flutter ECG pattern."
        )

    qtc_values = [
        value
        for value in (gf.get("qtc_bazett_ms"), gf.get("qtc_fridericia_ms"))
        if isinstance(value, (int, float))
    ]
    qtc_value = max(qtc_values) if qtc_values else None
    qtc_class, qtc_limits_text = _qtc_class(qtc_value, sex, age)
    prolonged_qt_claim = _has_nonnegated_claim(
        adjudicated_claims, _PROLONGED_QT_PATTERNS
    )
    if not age["known"] and prolonged_qt_claim:
        notes.append(
            "The synthesis asserts prolonged QTc while age is unknown, so an age-appropriate "
            "threshold cannot be selected. Do not substitute an adult cutoff; report the measured "
            "QTc and uncertainty."
        )
    if "prolonged" in qtc_class and (
        any(term in synthesis_text for term in ("normal qt", "qt interval is normal", "qtc is normal"))
        or "prolong" not in synthesis_text
    ):
        notes.append(
            f"QTc reaches {_fmt_num(qtc_value, 0, ' ms')} ({qtc_limits_text}). The synthesis "
            "must report the age-resolved prolonged/borderline-prolonged QTc, not normal QT."
        )
    if qtc_class in {"within the usual adult range", "normal by age-adjusted pediatric limits"} and prolonged_qt_claim:
        notes.append(
            f"QTc reaches {_fmt_num(qtc_value, 0, ' ms')}, within the resolved age-specific "
            "limits. Remove the unsupported prolonged-QTc statement "
            "while keeping QT dispersion separate."
        )

    lvh_criteria, _lvh_audit = _lvh_voltage_facts(data)
    lvh_claim = _has_nonnegated_claim(
        primary_claims, (r"left\s+ventricular\s+hypertrophy", r"(?:^|\s)lvh\b")
    )
    if not age["known"] and lvh_claim:
        notes.append(
            "The synthesis asserts an LVH pattern while age is unknown, so age-appropriate "
            "voltage thresholds cannot be selected. Do not substitute adult LVH criteria."
        )
    if lvh_criteria and not any(term in synthesis_text for term in ("lvh", "left ventricular hypertroph")):
        notes.append(
            f"Raw measurements meet ECG LVH voltage criteria ({_fmt_list(lvh_criteria)}). "
            "Include this as an ECG voltage pattern while avoiding an unsupported anatomic claim."
        )
    if lvh_criteria and lvh_claim and "voltage" not in primary_claims.lower():
        notes.append(
            "Only ECG voltage criteria are available for LVH. Rephrase the primary finding as "
            "'voltage criteria/pattern for LVH' rather than asserting anatomic left ventricular "
            "hypertrophy."
        )

    observed_negative, polarity_conflicts = _t_polarity_facts(data)
    claimed_t = _claimed_t_inversion_leads(synthesis_text_raw)
    unsupported_t = sorted(claimed_t - set(observed_negative), key=STANDARD_12_LEADS.index)
    if unsupported_t:
        notes.append(
            f"The synthesis claims T-wave inversion in {_fmt_list(unsupported_t)}, but observed "
            f"negative polarity is limited to {_fmt_list(observed_negative)}. Remove unsupported "
            f"leads and mark field conflicts ({_fmt_list(polarity_conflicts)}) uncertain."
        )

    clinical_context = clinician_notes.lower()
    if any(term in top1 for term in ("nstemi", "stemi", "acute myocardial infarction", "acute mi")):
        has_biomarker_context = any(
            term in clinical_context for term in ("troponin", "serial ecg", "acute chest", "acs")
        )
        if not has_biomarker_context:
            notes.append(
                "The top diagnosis asserts an acute MI/NSTEMI/STEMI event, but no symptoms, "
                "serial ECG evolution, or troponin evidence were supplied. Rephrase as an ECG "
                "pattern requiring urgent clinical correlation rather than a confirmed event."
            )

    prior_mi_claim = bool(
        re.search(
            r"\b(?:prior|old|previous|remote)\b[^.\n]{0,80}\b(?:myocardial infarction|mi)\b",
            top1,
        )
    )
    if prior_mi_claim:
        q_support = _has_contiguous_pathological_q_support(data, age)
        if q_support is None:
            notes.append(
                "The top diagnosis asserts prior MI, but age is unknown, so age-appropriate "
                "Q-wave criteria cannot be selected. Do not substitute the adult >=40 ms "
                "criterion; obtain age or downgrade the claim to uncertain."
            )
        elif not q_support:
            _q_leads, q_criterion = _age_resolved_q_wave_leads(data, age)
            notes.append(
                "The top diagnosis asserts prior MI, but raw per-lead measurements do not show "
                f"the required {q_criterion} in at least two leads of a contiguous territory. Do not use "
                "Q/R ratio or extractor flags alone to confirm prior MI."
            )

    if any(term in top1 for term in ("incomplete right bundle branch block", "irbbb")):
        qrs = (data.get("global_features") or {}).get("qrs_ms")
        qrs_value = _numeric_measurement(qrs)
        _qrs_result, qrs_limits = _qrs_class(qrs, age)
        if not age["known"]:
            notes.append(
                "The top diagnosis asserts incomplete RBBB, but age is unknown, so the "
                "age-specific QRS-duration criterion cannot be selected. Do not substitute the "
                "adult 110-119 ms range; obtain age and require compatible morphology."
            )
        elif age["pediatric_morphology"]:
            incomplete_lower = (qrs_limits or {}).get("borderline")
            bbb_limit = (qrs_limits or {}).get("nonspecific")
            if (
                qrs_value is None
                or incomplete_lower is None
                or bbb_limit is None
                or not incomplete_lower < qrs_value <= bbb_limit
            ):
                notes.append(
                    f"The top diagnosis asserts incomplete RBBB, but QRS is "
                    f"{_fmt_num(qrs_value, 0, ' ms')}, outside the resolved pediatric "
                    f"borderline-IVCD to BBB range ({_fmt_num(incomplete_lower, 0, ' ms')} to "
                    f"{_fmt_num(bbb_limit, 0, ' ms')}). Require the pediatric V1 R' amplitude/"
                    "duration and lateral terminal-S morphology as well."
                )
        elif qrs_value is None or not 110 <= qrs_value < 120:
            notes.append(
                f"The top diagnosis asserts incomplete RBBB, but consensus QRS is "
                f"{_fmt_num(qrs_value, 0, ' ms')}, outside the adult 110-119 ms range; require both "
                "duration and compatible V1/V2 plus I/V6 terminal morphology."
            )

    if "atrial fibrillation" in top1:
        rr_note = check_rr_outlier_driven_irregularity(data)
        if rr_note:
            notes.append(rr_note[0])
    return notes


def build_synthesis_revision_prompt(
    original_prompt: str,
    previous_output: str,
    guardrail_notes: list[str],
    language: str = "en",
) -> str:
    notes_block = "\n".join(f"- {note}" for note in guardrail_notes)
    if language == "ja":
        instruction = f"""最終統合に高リスクな未解決事項があります。各項目を測定段階の証拠と照合し、支持されない診断を削除またはECGパターンへ格下げして、同じL5小節形式で全文を再出力してください。

必須修正事項:
{notes_block}"""
    else:
        instruction = f"""The final synthesis contains high-risk claims that failed deterministic evidence checks. Reconcile every item with the adjudicated stage evidence, remove or downgrade unsupported diagnoses to ECG patterns, and re-output the full synthesis in the same L5 subsection format.

Required corrections:
{notes_block}"""
    conversation = original_prompt
    model_turn_start = "<start_of_turn>model\n"
    if conversation.endswith(model_turn_start):
        conversation = conversation[: -len(model_turn_start)]
        conversation += model_turn_start + previous_output.strip() + "<end_of_turn>\n"
    return conversation + f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def run_layered_diagnosis(
    features_path: Path,
    report_path: Path | None,
    clinician_notes: str,
    language: str,
    generate_text_fn: Callable[[str], str],
    max_revision_rounds: int = 1,
) -> dict[str, Any]:
    """Run no-Dx JSON+report sequential ECG reasoning.

    The model is called once for each L0-L4 stage and once for L5 synthesis.
    Later diagnostic stages receive only the L0 quality conclusion plus their
    own JSON measurements, diagnosis-free report excerpt, and deterministic
    arithmetic checks.  Pitfall checks can revise the affected stage before
    L5 sees it. Unified clinical conclusions and original Dx remain
    withheld until the post-hoc audit comparison.
    """
    data = _safe_read_json(features_path)
    if not is_current_features_schema(data):
        raise ValueError("Sequential reasoning requires the current ECG features schema.")
    diagnostic_gate = diagnostic_gate_policy(data)
    blocked_layers = diagnostic_gate_blocked_layers(diagnostic_gate)
    layer_evidence = summarize_layered_evidence(data, language)
    report_text = sanitize_report_text(_safe_read_text(report_path)) if report_path is not None else ""
    report_by_layer = summarize_report_by_layer(report_text)
    if blocked_layers:
        # The readable report is a redundant, presentation-oriented view whose
        # broad sections can mix domains (for example axes and intervals in one
        # GLOBAL MEASUREMENTS table).  Once any domain is suppressed, using a
        # report excerpt in another stage could reintroduce the withheld values.
        # Keep the sanitized report for audit only and reason from the gated
        # structured JSON bundles.
        report_by_layer = {key: "" for key in LAYER_KEYS}
    validated_facts = build_validated_measurement_facts(data)
    reasoning_evidence: dict[str, str] = {}
    for layer_key in LAYER_KEYS:
        blocks = [
            layer_evidence[layer_key],
            "[Deterministic measurement checks; computed from JSON without Dx]\n"
            + validated_facts[layer_key],
        ]
        if layer_key not in blocked_layers and report_by_layer.get(layer_key):
            blocks.append(
                "[Diagnosis-free report measurement cross-check]\n"
                + report_by_layer[layer_key]
            )
        reasoning_evidence[layer_key] = "\n\n".join(blocks)

    if diagnostic_gate["state"] == "stop":
        stop_reasons = diagnostic_gate.get("stop_reasons") or ["unspecified_quality_stop"]
        stage_outputs = {
            "L0": (
                "- Diagnostic quality gate STOP. No model inference was performed. "
                f"Reasons: {_fmt_list(stop_reasons)}."
            ),
            **{
                layer_key: _gate_withheld_text(layer_key, diagnostic_gate)
                for layer_key in LAYER_KEYS[1:]
            },
        }
        synthesis_output = (
            "#### Most Likely Diagnosis\n"
            "- ABSTAIN — diagnostic quality gate stopped interpretation\n\n"
            "#### Diagnostic Rationale (layer references)\n"
            f"- No diagnosis is permitted. Gate reasons: {_fmt_list(stop_reasons)}.\n\n"
            "#### Similar Diagnoses (ranked by similarity)\n"
            "- Not generated.\n\n"
            "#### Layer Conflicts\n"
            "- Not assessed.\n\n"
            "#### Uncertainty and Review Priorities\n"
            "- Reacquire or repair the ECG before diagnostic interpretation."
        )
        raw_output = compose_sequential_output(stage_outputs, synthesis_output, language)
        parsed = parse_layered_output(raw_output)
        parsed.update(
            {
                "top1_raw": None,
                "top1_normalized": None,
                "similar_raw": [],
                "similar_normalized": [],
                "abstained": True,
                "abstention_reason": "diagnostic_gate_stop",
            }
        )
        reference_only = summarize_reference_only(data)
        return {
            "reasoning_mode": "diagnostic_gate_stop_abstention_v2",
            "gate_enforcement_version": DIAGNOSTIC_GATE_ENFORCEMENT_VERSION,
            "reasoning_sources": ["metadata.diagnostic_gate"],
            "dx_code_used_in_reasoning": False,
            "diagnostic_gate": diagnostic_gate,
            "blocked_layers": sorted(blocked_layers),
            "abstained": True,
            "abstention_reason": "diagnostic_gate_stop",
            "layer_evidence": layer_evidence,
            "reasoning_evidence": reasoning_evidence,
            "validated_measurement_facts": validated_facts,
            "report_layer_context": report_by_layer,
            "stage_prompts": {},
            "stage_outputs": stage_outputs,
            "stage_revision_prompts": {key: [] for key in LAYER_KEYS},
            "synthesis_prompt": "",
            "synthesis_output": synthesis_output,
            "synthesis_revision_prompts": [],
            "prompt": "",
            "raw_model_output": raw_output,
            "parsed": parsed,
            "guardrail_notes": [],
            "stage_claim_guardrail_notes": {key: [] for key in LAYER_KEYS},
            "unaddressed_guardrail_notes": [],
            "synthesis_guardrail_notes": [],
            "revised": False,
            "supplementary_report_sanitized": report_text,
            "supplementary_report_used_in_reasoning": False,
            "reference_only": reference_only,
            "reference_agreement": "not_compared_abstained",
        }

    stage_prompts: dict[str, str] = {}
    stage_outputs: dict[str, str] = {}
    revision_prompts: dict[str, list[str]] = {key: [] for key in LAYER_KEYS}

    l0_prompt = build_layer_reasoning_prompt("L0", reasoning_evidence["L0"], "", language)
    stage_prompts["L0"] = l0_prompt
    stage_outputs["L0"] = clean_model_output(generate_text_fn(l0_prompt))

    quality_conclusion = stage_outputs["L0"]
    for layer_key in LAYER_KEYS[1:]:
        if layer_key in blocked_layers:
            stage_outputs[layer_key] = _gate_withheld_text(
                layer_key, diagnostic_gate
            )
            continue
        stage_prompt = build_layer_reasoning_prompt(
            layer_key,
            reasoning_evidence[layer_key],
            quality_conclusion,
            language,
        )
        stage_prompts[layer_key] = stage_prompt
        stage_outputs[layer_key] = clean_model_output(generate_text_fn(stage_prompt))

    guardrail_notes = [
        note
        for note in run_independent_feature_guardrails(data)
        if not (_guardrail_required_layers(note) & blocked_layers)
    ]
    revised = False
    if max_revision_rounds > 0:
        for _round in range(max_revision_rounds):
            notes_by_layer: dict[str, list[str]] = {key: [] for key in LAYER_KEYS}
            for note in guardrail_notes:
                target = _guardrail_target_layer(note)
                if not _guardrail_addressed(note, stage_outputs.get(target, "")):
                    notes_by_layer[target].append(note)
            claim_notes = run_stage_claim_guardrails(data, stage_outputs)
            for layer_key in blocked_layers:
                claim_notes[layer_key] = []
            for layer_key, notes in claim_notes.items():
                notes_by_layer[layer_key].extend(notes)
            if not any(notes_by_layer.values()):
                break
            for layer_key, notes in notes_by_layer.items():
                if not notes or layer_key in blocked_layers:
                    continue
                notes = list(dict.fromkeys(notes))
                revision_prompt = build_layer_revision_prompt(
                    stage_prompts[layer_key],
                    stage_outputs[layer_key],
                    notes,
                    language,
                )
                revision_prompts[layer_key].append(revision_prompt)
                stage_outputs[layer_key] = clean_model_output(generate_text_fn(revision_prompt))
                stage_prompts[layer_key] = revision_prompt
                revised = True

    unaddressed_feature_notes = [
        note
        for note in guardrail_notes
        if not _guardrail_addressed(
            note, stage_outputs.get(_guardrail_target_layer(note), "")
        )
    ]
    final_stage_claim_notes = run_stage_claim_guardrails(data, stage_outputs)
    for layer_key in blocked_layers:
        final_stage_claim_notes[layer_key] = []
    unaddressed = unaddressed_feature_notes + [
        note for notes in final_stage_claim_notes.values() for note in notes
    ]

    synthesis_facts = "\n\n".join(
        f"### {layer_key}\n{validated_facts[layer_key]}" for layer_key in LAYER_KEYS
    )
    synthesis_prompt = build_synthesis_prompt(
        stage_outputs,
        clinician_notes,
        language,
        validated_facts=synthesis_facts,
    )
    synthesis_output = clean_model_output(generate_text_fn(synthesis_prompt))
    raw_output = compose_sequential_output(stage_outputs, synthesis_output, language)
    parsed = parse_layered_output(raw_output)

    synthesis_guardrail_notes = [
        note
        for note in run_synthesis_guardrails(data, parsed, clinician_notes)
        if _synthesis_guardrail_target_layer(note) not in blocked_layers
    ]
    synthesis_revision_prompts: list[str] = []
    if synthesis_guardrail_notes and max_revision_rounds > 0:
        for _round in range(max_revision_rounds):
            revision_prompt = build_synthesis_revision_prompt(
                synthesis_prompt,
                synthesis_output,
                synthesis_guardrail_notes,
                language,
            )
            synthesis_revision_prompts.append(revision_prompt)
            synthesis_output = clean_model_output(generate_text_fn(revision_prompt))
            raw_output = compose_sequential_output(stage_outputs, synthesis_output, language)
            parsed = parse_layered_output(raw_output)
            revised = True
            synthesis_guardrail_notes = [
                note
                for note in run_synthesis_guardrails(data, parsed, clinician_notes)
                if _synthesis_guardrail_target_layer(note) not in blocked_layers
            ]
            if not synthesis_guardrail_notes:
                break

    reference_only = summarize_reference_only(data)
    return {
        "reasoning_mode": MEDGEMMA_REASONING_PROTOCOL_VERSION,
        "gate_enforcement_version": DIAGNOSTIC_GATE_ENFORCEMENT_VERSION,
        "reasoning_sources": [
            "features_json_measurements",
            *(
                ["sanitized_report_measurements"]
                if report_text and not blocked_layers
                else []
            ),
            "deterministic_measurement_checks",
        ],
        "dx_code_used_in_reasoning": False,
        "diagnostic_gate": diagnostic_gate,
        "blocked_layers": sorted(blocked_layers),
        "abstained": False,
        "abstention_reason": None,
        "layer_evidence": layer_evidence,
        "reasoning_evidence": reasoning_evidence,
        "validated_measurement_facts": validated_facts,
        "report_layer_context": report_by_layer,
        "stage_prompts": stage_prompts,
        "stage_outputs": stage_outputs,
        "stage_revision_prompts": revision_prompts,
        "synthesis_prompt": synthesis_prompt,
        "synthesis_output": synthesis_output,
        "synthesis_revision_prompts": synthesis_revision_prompts,
        "prompt": synthesis_prompt,
        "raw_model_output": raw_output,
        "parsed": parsed,
        "guardrail_notes": guardrail_notes,
        "stage_claim_guardrail_notes": final_stage_claim_notes,
        "unaddressed_guardrail_notes": unaddressed,
        "synthesis_guardrail_notes": synthesis_guardrail_notes,
        "revised": revised,
        "supplementary_report_sanitized": report_text,
        "supplementary_report_used_in_reasoning": bool(report_text) and not blocked_layers,
        "reference_only": reference_only,
        "reference_agreement": compare_with_reference(parsed, reference_only),
    }


def compare_with_reference(parsed: dict[str, Any], reference_text: str) -> str:
    """Informational only — never fed back into the reasoning prompt. Flags
    whether the model's independently-derived diagnosis has any lexical
    overlap with the withheld rule-engine output, for human/audit review."""
    top1 = (parsed.get("top1_raw") or "").lower()
    if not top1 or reference_text == "—":
        return "no_reference_available"
    tokens = [tok for tok in re.split(r"[\s/,()-]+", top1) if len(tok) > 3]
    reference_lower = reference_text.lower()
    overlap = [tok for tok in tokens if tok in reference_lower]
    return "lexical_overlap_found" if overlap else "no_lexical_overlap_review_recommended"

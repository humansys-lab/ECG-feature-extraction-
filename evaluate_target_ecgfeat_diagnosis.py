#!/usr/bin/env python3
"""Run ecgfeat on the requested PTB-XL and local SNOMED datasets.

The script deliberately evaluates only authoritative matched statements from
``clinical_interpretation.final_statements``.  Reference labels are never
passed into the extractor, so the diagnostic comparison cannot leak labels
into predictions.

The available references are record-level labels, not beat-, lead-, or
fiducial-level truth.  Results are therefore label-agreement measurements and
an engineering failure analysis, not clinical validation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.models import PatientMeta, STANDARD_12_LEADS


DATASET_PTBXL = "00000"
DATASET_LOCAL = "010"
TARGET_EXTRACTION_ARTIFACT_VERSION = "ecgfeat.target-evaluation-artifact.v1"

PTBXL_T_WAVE_CODES = frozenset({"NDT", "LOWT", "NT_", "INVT", "TAB_"})
PTBXL_ISCHEMIA_ST_CODES = frozenset(
    {
        "NST_",
        "STD_",
        "ISC_",
        "ISCAL",
        "ISCIN",
        "ISCIL",
        "ISCAS",
        "ISCLA",
        "ISCAN",
        "ANEUR",
        "INJAS",
        "INJAL",
        "INJIN",
        "INJLA",
        "INJIL",
    }
)
PTBXL_MI_CODES = frozenset(
    {
        "AMI",
        "ASMI",
        "ALMI",
        "IMI",
        "ILMI",
        "LMI",
        "IPMI",
        "IPLMI",
        "INJAS",
        "INJAL",
        "INJIN",
        "INJLA",
        "INJIL",
    }
)


@dataclass(frozen=True)
class CategorySpec:
    key: str
    display_name_zh: str
    ptbxl_refs: frozenset[str]
    local_refs: frozenset[str]
    prediction_codes: frozenset[str]
    evaluates_codes: frozenset[str] = frozenset()
    rule_ids: frozenset[str] = frozenset()
    semantic_scope: str = "direct"

    def refs(self, dataset: str) -> frozenset[str]:
        return self.ptbxl_refs if dataset == DATASET_PTBXL else self.local_refs


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(
        "sinus_bradycardia",
        "心率过缓观察（参考窦缓标签）",
        frozenset({"SBRAD"}),
        frozenset({"426177001"}),
        frozenset({"sinus_bradycardia"}),
        evaluates_codes=frozenset({"adult_rate_abnormality"}),
        rule_ids=frozenset({"CLIN-RHYTHM-RATE-01"}),
    ),
    CategorySpec(
        "sinus_tachycardia",
        "心率过速观察（参考窦速标签）",
        frozenset({"STACH"}),
        frozenset({"427084000"}),
        frozenset({"sinus_tachycardia"}),
        evaluates_codes=frozenset({"adult_rate_abnormality"}),
        rule_ids=frozenset({"CLIN-RHYTHM-RATE-01"}),
    ),
    CategorySpec(
        "atrial_fibrillation",
        "心房颤动",
        frozenset({"AFIB"}),
        frozenset({"164889003"}),
        # The bare `atrial_fibrillation` code is a real entry in
        # DIAGNOSIS_CATALOG and is what ECGAgent actually emits; omitting it
        # here scored 8 correct AF calls across four repeats as nothing, and
        # charged the matching AFIB labels as false negatives. That made the
        # family look like a total gate failure when the agent was right.
        frozenset(
            {
                "atrial_fibrillation",
                "atrial_fibrillation_pattern",
                "probable_atrial_fibrillation_pattern",
            }
        ),
        evaluates_codes=frozenset({"atrial_fibrillation_pattern"}),
    ),
    CategorySpec(
        "atrial_flutter",
        "心房扑动",
        frozenset({"AFLT"}),
        frozenset({"164890007"}),
        frozenset(
            {"atrial_flutter_pattern", "probable_atrial_flutter_pattern"}
        ),
        evaluates_codes=frozenset({"atrial_flutter_pattern"}),
    ),
    CategorySpec(
        "premature_atrial_complexes",
        "房性早搏/房性二联律",
        frozenset({"PAC"}),
        frozenset({"284470004", "251173003"}),
        frozenset({"premature_atrial_complexes"}),
        evaluates_codes=frozenset({"premature_atrial_complexes"}),
    ),
    CategorySpec(
        "premature_ventricular_complexes",
        "室性早搏",
        frozenset({"PVC"}),
        frozenset({"17338001"}),
        frozenset(
            {
                "premature_ventricular_complexes",
                "probable_premature_ventricular_complexes",
            }
        ),
        evaluates_codes=frozenset({"premature_ventricular_complexes"}),
    ),
    CategorySpec(
        "unspecified_ectopy_pattern",
        "未定来源二/三联律",
        frozenset({"BIGU", "TRIGU"}),
        frozenset(),
        frozenset(
            {
                "atrial_bigeminy_pattern",
                "ventricular_bigeminy_pattern",
                "trigeminy_pattern",
            }
        ),
        evaluates_codes=frozenset(
            {
                "atrial_bigeminy_pattern",
                "ventricular_bigeminy_pattern",
                "trigeminy_pattern",
            }
        ),
        semantic_scope="broad",
    ),
    CategorySpec(
        "first_degree_av_block",
        "一度房室传导阻滞/PR 延长",
        frozenset({"1AVB", "LPR"}),
        frozenset({"270492004"}),
        # Same split as atrial_fibrillation: the rule engine emits
        # `first_degree_av_delay` while the agent's catalog offers only
        # `first_degree_av_block`, so 11 agent assertions scored as nothing.
        frozenset(
            {
                "first_degree_av_delay",
                "first_degree_av_block",
                "possible_first_degree_av_delay",
            }
        ),
        rule_ids=frozenset({"CLIN-INTERVAL-PR-01"}),
    ),
    CategorySpec(
        "complete_av_block",
        "三度/完全性房室传导阻滞",
        frozenset({"3AVB"}),
        frozenset(),
        frozenset({"complete_av_block_pattern"}),
        evaluates_codes=frozenset({"complete_av_block_pattern"}),
    ),
    CategorySpec(
        "right_bundle_branch_block",
        "右束支传导阻滞",
        frozenset({"IRBBB", "CRBBB"}),
        frozenset({"59118001"}),
        frozenset({"rbbb_pattern", "probable_rbbb_pattern"}),
        evaluates_codes=frozenset({"rbbb_pattern"}),
    ),
    CategorySpec(
        "left_bundle_branch_block",
        "左束支传导阻滞",
        frozenset({"ILBBB", "CLBBB"}),
        frozenset(),
        frozenset({"lbbb_pattern", "probable_lbbb_pattern"}),
        evaluates_codes=frozenset({"lbbb_pattern"}),
    ),
    CategorySpec(
        "ambiguous_left_conduction_family",
        "左束支/左分支传导异常家族（本地代码歧义）",
        frozenset(),
        frozenset({"164909002"}),
        frozenset(
            {
                "lbbb_pattern",
                "probable_lbbb_pattern",
                "lafb_pattern",
                "lpfb_pattern",
            }
        ),
        evaluates_codes=frozenset(
            {"lbbb_pattern", "lafb_pattern", "lpfb_pattern"}
        ),
        semantic_scope="broad",
    ),
    CategorySpec(
        "left_anterior_fascicular_block",
        "左前分支阻滞",
        frozenset({"LAFB"}),
        frozenset(),
        frozenset({"lafb_pattern"}),
        evaluates_codes=frozenset({"lafb_pattern"}),
    ),
    CategorySpec(
        "left_posterior_fascicular_block",
        "左后分支阻滞",
        frozenset({"LPFB"}),
        frozenset(),
        frozenset({"lpfb_pattern"}),
        evaluates_codes=frozenset({"lpfb_pattern"}),
    ),
    CategorySpec(
        "nonspecific_ivcd",
        "非特异性室内传导延迟",
        frozenset({"IVCD"}),
        frozenset({"698252002"}),
        frozenset({"nonspecific_ivcd", "probable_nonspecific_ivcd"}),
        evaluates_codes=frozenset({"nonspecific_ivcd"}),
    ),
    CategorySpec(
        "ventricular_preexcitation",
        "心室预激/WPW",
        frozenset({"WPW"}),
        frozenset(),
        frozenset({"ventricular_preexcitation_pattern"}),
        evaluates_codes=frozenset({"ventricular_preexcitation_pattern"}),
    ),
    CategorySpec(
        "left_ventricular_hypertrophy",
        "左室肥厚/电压标准",
        frozenset({"LVH", "VCLVH"}),
        frozenset({"55827005"}),
        frozenset({"lvh_voltage_criteria"}),
        evaluates_codes=frozenset({"lvh_voltage_criteria"}),
    ),
    CategorySpec(
        "right_ventricular_hypertrophy",
        "右室肥厚",
        frozenset({"RVH"}),
        frozenset(),
        frozenset({"rvh_pattern"}),
        evaluates_codes=frozenset({"rvh_pattern"}),
    ),
    CategorySpec(
        "atrial_abnormality",
        "心房异常/增大",
        frozenset({"LAO/LAE", "RAO/RAE"}),
        frozenset({"164912004"}),
        frozenset({"left_atrial_abnormality", "right_atrial_abnormality"}),
        evaluates_codes=frozenset(
            {"left_atrial_abnormality", "right_atrial_abnormality"}
        ),
        semantic_scope="broad",
    ),
    CategorySpec(
        "low_qrs_voltage",
        "QRS 低电压",
        frozenset({"LVOLT"}),
        frozenset({"251146004"}),
        frozenset(
            {
                "low_qrs_voltage_limb_leads",
                "low_qrs_voltage_precordial_leads",
            }
        ),
        evaluates_codes=frozenset(
            {
                "low_qrs_voltage_limb_leads",
                "low_qrs_voltage_precordial_leads",
            }
        ),
        semantic_scope="broad",
    ),
    CategorySpec(
        "prolonged_qt",
        "QT 间期延长",
        frozenset({"LNGQT"}),
        frozenset(),
        frozenset({"prolonged_qt", "markedly_prolonged_qt"}),
        rule_ids=frozenset({"CLIN-INTERVAL-QT-01"}),
    ),
    CategorySpec(
        "t_wave_abnormality",
        "T 波异常",
        PTBXL_T_WAVE_CODES,
        frozenset({"164934002", "59931005"}),
        frozenset(
            {"primary_t_wave_abnormality", "secondary_t_wave_abnormality"}
        ),
        evaluates_codes=frozenset(
            {"primary_t_wave_abnormality", "secondary_t_wave_abnormality"}
        ),
        semantic_scope="broad",
    ),
    CategorySpec(
        "ischemia_or_st_abnormality",
        "缺血/ST 段异常筛查",
        PTBXL_ISCHEMIA_ST_CODES,
        frozenset({"428750005", "429622005"}),
        # `st_elevation` is a DIAGNOSIS_CATALOG code in this very family that no
        # category scored, so an agent that emitted it would have been credited
        # nothing -- the same defect that hid every atrial_fibrillation call.
        frozenset(
            {
                "acute_occlusion_pattern",
                "st_depression",
                "st_elevation",
                "posterior_ischemia_screen",
                "sgarbossa_positive",
                "wellens_pattern",
                "left_main_pattern",
                "de_winter_pattern",
            }
        ),
        evaluates_codes=frozenset(
            {
                "acute_occlusion_pattern",
                "st_depression",
                "st_elevation",
                "posterior_ischemia_screen",
                "sgarbossa_positive",
                "wellens_pattern",
                "left_main_pattern",
                "de_winter_pattern",
            }
        ),
        semantic_scope="screening",
    ),
    CategorySpec(
        "infarction_or_q_wave",
        "心肌梗死/病理性 Q 波筛查",
        PTBXL_MI_CODES | frozenset({"QWAVE"}),
        frozenset({"164865005", "164917005"}),
        # `pathological_q_waves` is the agent-facing catalog name for exactly
        # this finding; only the rule engine's `prior_infarct_q_wave_pattern`
        # was scored.
        frozenset({"prior_infarct_q_wave_pattern", "pathological_q_waves"}),
        evaluates_codes=frozenset({"prior_infarct_q_wave_pattern"}),
        semantic_scope="screening",
    ),
)


KNOWN_UNSUPPORTED_REFS: dict[str, dict[str, str]] = {
    DATASET_PTBXL: {
        "NORM": "正常 ECG（权威层不输出阳性“正常”语句）",
        "SR": "窦性心律（权威层不输出阳性“窦律正常”语句）",
        "SARRH": "窦性心律不齐",
        "SVTAC": "室上性心动过速的机制分型",
        "PACE": "起搏器节律（仅作为解释上下文，未投影为最终诊断）",
        "DIG": "洋地黄效应",
        "EL": "电解质/药物效应",
        "ABQRS": "笼统 QRS 异常",
        "HVOLT": "笼统高 QRS 电压",
        "SEHYP": "室间隔肥厚",
    },
    DATASET_LOCAL: {
        "426783006": "窦性心律（权威层不输出阳性“窦律正常”语句）",
        "233917008": (
            "未分级房室传导阻滞；不能与一度、二度和完全性房室阻滞重复计分，"
            "需更细粒度标签后再评价"
        ),
        "39732003": "电轴左偏（已有轴数值，但未投影为最终诊断）",
        "47665007": "电轴右偏（已有轴数值，但未投影为最终诊断）",
        "251199005": "逆钟向转位",
    },
}


PREDICTION_CODE_ZH = {
    "bradycardia": "心动过缓",
    "tachycardia": "心动过速",
    "atrial_fibrillation_pattern": "房颤模式",
    "probable_atrial_fibrillation_pattern": "可能房颤模式",
    "atrial_flutter_pattern": "房扑模式",
    "probable_atrial_flutter_pattern": "可能房扑模式",
    "possible_atrial_flutter_pattern": "房扑可能性观察（需节律条复核）",
    "atrial_fibrillation_flutter_indeterminate": "房颤/房扑未定",
    "premature_atrial_complexes": "房性早搏",
    "premature_ventricular_complexes": "室性早搏",
    "probable_premature_ventricular_complexes": "可能室性早搏",
    "first_degree_av_delay": "一度房室传导延迟",
    "possible_first_degree_av_delay": "可能一度房室传导延迟（需同周期确认）",
    "second_degree_av_block_pattern": "二度房室传导阻滞模式",
    "complete_av_block_pattern": "完全性房室传导阻滞模式",
    "rbbb_pattern": "右束支阻滞",
    "probable_rbbb_pattern": "可能右束支阻滞",
    "lbbb_pattern": "左束支阻滞",
    "probable_lbbb_pattern": "可能左束支阻滞",
    "lafb_pattern": "左前分支阻滞",
    "lpfb_pattern": "左后分支阻滞",
    "nonspecific_ivcd": "非特异性室内传导延迟",
    "probable_nonspecific_ivcd": "可能非特异性室内传导延迟",
    "ventricular_preexcitation_pattern": "心室预激",
    "lvh_voltage_criteria": "左室肥厚电压标准",
    "rvh_pattern": "右室肥厚模式",
    "left_atrial_abnormality": "左房异常",
    "right_atrial_abnormality": "右房异常",
    "low_qrs_voltage_limb_leads": "肢体导联低电压",
    "low_qrs_voltage_precordial_leads": "胸前导联低电压",
    "prolonged_qt": "QT 延长",
    "primary_t_wave_abnormality": "原发性 T 波异常",
    "secondary_t_wave_abnormality": "继发性 T 波异常",
    "acute_occlusion_pattern": "急性缺血/冠脉闭塞筛查模式",
    "posterior_ischemia_screen": "后壁缺血筛查模式",
    "sgarbossa_positive": "Sgarbossa 阳性筛查",
    "prior_infarct_q_wave_pattern": "陈旧/未定期梗死 Q 波模式",
    "technically_limited": "技术质量受限",
}

# Explicit low-confidence observations that are useful in the clinical output
# but deliberately excluded from direct label-agreement scoring.
NONCOMPARABLE_OBSERVATION_CODES = frozenset(
    {
        "possible_atrial_flutter_pattern",
        "possible_first_degree_av_delay",
    }
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sex_from_ptbxl(value: Any) -> str | None:
    parsed = _finite(value)
    if parsed is None:
        return None
    return "male" if int(round(parsed)) == 1 else "female"


def _load_ptbxl_manifest(
    dataset_dir: Path,
    metadata_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    statements = {
        row[""]: row
        for row in _read_csv(metadata_dir / "scp_statements.csv")
        if row.get("")
    }
    database = _read_csv(metadata_dir / "ptbxl_database.csv")
    by_filename: dict[str, dict[str, str]] = {}
    for row in database:
        # PTB-XL distributes parallel 100 Hz ``*_lr`` and 500 Hz ``*_hr``
        # records. Index both so an extracted shard can be evaluated without
        # renaming files or accidentally matching it to the wrong sampling
        # profile.
        for field in ("filename_lr", "filename_hr"):
            filename = row.get(field)
            if filename:
                by_filename[Path(filename).name] = row
    manifest: list[dict[str, Any]] = []
    missing: list[str] = []
    for header in sorted(dataset_dir.glob("*.hea")):
        source = by_filename.get(header.stem)
        if source is None:
            missing.append(header.stem)
            continue
        score_map = ast.literal_eval(source.get("scp_codes", "{}"))
        raw_codes = list(score_map)
        active_codes: list[str] = []
        for code, score in score_map.items():
            statement = statements.get(code, {})
            is_diagnostic = statement.get("diagnostic") == "1.0"
            if is_diagnostic and float(score) < 50.0:
                continue
            active_codes.append(code)
        names = {
            code: statements.get(code, {}).get("description", code)
            for code in raw_codes
        }
        manifest.append(
            {
                "dataset": DATASET_PTBXL,
                "record": header.stem,
                "record_path": str(header.with_suffix("")),
                "age": _finite(source.get("age")),
                "sex": _sex_from_ptbxl(source.get("sex")),
                "reference_codes_raw": raw_codes,
                "reference_codes_active": active_codes,
                "reference_scores": {
                    code: float(score) for code, score in score_map.items()
                },
                "reference_names": names,
                "reference_report": source.get("report", ""),
                "reference_policy": (
                    "diagnostic SCP codes require likelihood >=50; "
                    "form/rhythm codes are active by presence"
                ),
            }
        )
    if missing:
        raise ValueError(
            f"{len(missing)} PTB-XL records lack metadata: {', '.join(missing[:10])}"
        )
    return manifest, statements


def _parse_local_header(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"age": None, "sex": None, "codes": []}
    for raw_line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = raw_line.strip()
        if line.startswith("#Age:"):
            result["age"] = _finite(line.split(":", 1)[1].strip())
        elif line.startswith("#Sex:"):
            token = line.split(":", 1)[1].strip()
            result["sex"] = (
                None if not token or token.lower() == "unknown" else token.lower()
            )
        elif line.startswith("#Dx:"):
            result["codes"] = [
                code.strip()
                for code in line.split(":", 1)[1].split(",")
                if code.strip()
            ]
    return result


def _load_local_code_names(condition_csv: Path) -> dict[str, str]:
    names = {
        str(row.get("Snomed_CT", "")).strip(): str(
            row.get("Full Name", "")
        ).strip()
        for row in _read_csv(condition_csv)
        if row.get("Snomed_CT")
    }
    names.setdefault("55827005", "Left ventricular hypertrophy")
    return names


def _load_local_manifest(
    dataset_dir: Path,
    condition_csv: Path,
) -> list[dict[str, Any]]:
    names = _load_local_code_names(condition_csv)
    manifest: list[dict[str, Any]] = []
    for header in sorted(dataset_dir.glob("*.hea")):
        parsed = _parse_local_header(header)
        codes = list(parsed["codes"])
        manifest.append(
            {
                "dataset": DATASET_LOCAL,
                "record": header.stem,
                "record_path": str(header.with_suffix("")),
                "age": parsed["age"],
                "sex": parsed["sex"],
                "reference_codes_raw": codes,
                "reference_codes_active": codes,
                "reference_scores": {},
                "reference_names": {
                    code: names.get(code, f"unmapped SNOMED-CT {code}")
                    for code in codes
                },
                "reference_report": "",
                "reference_policy": "all #Dx SNOMED-CT codes are active by presence",
            }
        )
    return manifest


def _load_record(record_path: Path) -> tuple[np.ndarray, float]:
    import wfdb

    record = wfdb.rdrecord(str(record_path))
    indices = {
        str(name).lower(): index
        for index, name in enumerate(record.sig_name)
    }
    missing = [
        lead for lead in STANDARD_12_LEADS if lead.lower() not in indices
    ]
    if missing:
        raise ValueError(f"missing standard leads: {', '.join(missing)}")
    ecg = np.asarray(record.p_signal, dtype=float)[
        :,
        [indices[lead.lower()] for lead in STANDARD_12_LEADS],
    ].T
    return ecg, float(record.fs)


def _quality_summary(result: Any) -> dict[str, Any]:
    quality = getattr(result, "quality", {}) or {}
    reliable = [
        lead
        for lead, item in quality.items()
        if bool(getattr(item, "reliable", False))
    ]
    qrs_reliable = [
        lead
        for lead, item in quality.items()
        if bool(getattr(item, "reliable_for_qrs", False))
    ]
    p_reliable = [
        lead
        for lead, item in quality.items()
        if bool(getattr(item, "reliable_for_p", False))
    ]
    t_reliable = [
        lead
        for lead, item in quality.items()
        if bool(getattr(item, "reliable_for_t", False))
    ]
    flags = sorted(
        {
            str(flag)
            for item in quality.values()
            for flag in (getattr(item, "flags", []) or [])
        }
    )
    return {
        "reliable_leads": reliable,
        "qrs_reliable_leads": qrs_reliable,
        "p_reliable_leads": p_reliable,
        "t_reliable_leads": t_reliable,
        "quality_flags": flags,
    }


def _signal_summary(ecg: np.ndarray) -> dict[str, Any]:
    per_lead_range = np.nanmax(ecg, axis=1) - np.nanmin(ecg, axis=1)
    return {
        "shape": [int(value) for value in ecg.shape],
        "minimum_mv": _finite(np.nanmin(ecg)),
        "maximum_mv": _finite(np.nanmax(ecg)),
        "nonfinite_samples": int(np.size(ecg) - np.isfinite(ecg).sum()),
        "per_lead_range_mv": {
            lead: _finite(per_lead_range[index])
            for index, lead in enumerate(STANDARD_12_LEADS)
        },
        "near_flat_leads": [
            lead
            for index, lead in enumerate(STANDARD_12_LEADS)
            if _finite(per_lead_range[index]) is not None
            and float(per_lead_range[index]) < 0.05
        ],
    }


def _global_summary(result: Any) -> dict[str, Any]:
    gf = result.global_features
    fields = (
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
        "qt_reliability",
        "qt_source",
        "paced_rhythm",
    )
    return {field: getattr(gf, field, None) for field in fields}


def _compact_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    evidence = rule.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "rule_id": rule.get("rule_id"),
        "domain": rule.get("domain"),
        "status": rule.get("status"),
        "statement_code": rule.get("statement_code"),
        "statement": rule.get("statement"),
        "confidence": rule.get("confidence"),
        "coverage": rule.get("coverage"),
        "missing_inputs": list(rule.get("missing_inputs") or []),
        "suppressed_by": list(rule.get("suppressed_by") or []),
        "evaluates_code": evidence.get("evaluates_code"),
        "evidence": evidence,
        "thresholds": rule.get("thresholds") or {},
    }


def _compact_clinical(clinical: Mapping[str, Any]) -> dict[str, Any]:
    domains = clinical.get("domains")
    domains = domains if isinstance(domains, dict) else {}
    rules = [
        _compact_rule(rule)
        for rows in domains.values()
        for rule in (rows or [])
        if isinstance(rule, dict)
    ]
    final_statements = [
        _compact_rule(rule)
        for rule in (clinical.get("final_statements") or [])
        if isinstance(rule, dict)
    ]
    return {
        "schema_version": clinical.get("schema_version"),
        "ruleset_version": clinical.get("ruleset_version"),
        "overall_status": clinical.get("overall_status"),
        "summary": clinical.get("summary") or {},
        "unavailable_domains": list(clinical.get("unavailable_domains") or []),
        "final_statements": final_statements,
        "rules": rules,
    }


def _hash_file(path: Path, digest: Any) -> None:
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


@lru_cache(maxsize=1)
def _extractor_code_fingerprint() -> str:
    digest = hashlib.sha256()
    source_root = FEATURE_ROOT / "ecgfeat"
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(b"\0")
        _hash_file(path, digest)
    return "sha256:" + digest.hexdigest()


def _target_extraction_contract(
    manifest_row: Mapping[str, Any],
    *,
    fs_internal: int,
    enable_pacing: bool,
) -> dict[str, Any] | None:
    base = Path(str(manifest_row.get("record_path") or ""))
    source_files = sorted(
        path for path in base.parent.glob(base.name + ".*") if path.is_file()
    )
    if not source_files:
        return None
    digest = hashlib.sha256()
    for path in source_files:
        _hash_file(path, digest)
    digest.update(
        json.dumps(
            {
                "record": manifest_row.get("record"),
                "age": manifest_row.get("age"),
                "age_days": manifest_row.get("age_days"),
                "sex": manifest_row.get("sex"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "schema_version": TARGET_EXTRACTION_ARTIFACT_VERSION,
        "source_fingerprint": "sha256:" + digest.hexdigest(),
        "extractor_code_fingerprint": _extractor_code_fingerprint(),
        "config": {
            "fs_internal": int(fs_internal),
            "mains_freq": 50,
            "enable_pacing": bool(enable_pacing),
        },
    }


def _extract_one(
    task: tuple[dict[str, Any], int, bool],
) -> dict[str, Any]:
    manifest_row, fs_internal, enable_pacing = task
    started = time.perf_counter()
    base = {
        key: manifest_row[key]
        for key in (
            "dataset",
            "record",
            "record_path",
            "age",
            "sex",
            "reference_codes_raw",
            "reference_codes_active",
            "reference_scores",
            "reference_names",
            "reference_report",
            "reference_policy",
        )
    }
    base["age_days"] = manifest_row.get("age_days")
    extraction_contract = _target_extraction_contract(
        manifest_row,
        fs_internal=fs_internal,
        enable_pacing=enable_pacing,
    )
    try:
        ecg, fs = _load_record(Path(manifest_row["record_path"]))
        extractor = ECGFeatureExtractor(
            fs_internal=fs_internal,
            mains_freq=50,
            enable_pacing=enable_pacing,
        )
        result = extractor.extract(
            ecg,
            fs=fs,
            meta=PatientMeta(
                age=manifest_row.get("age"),
                age_days=manifest_row.get("age_days"),
                sex=manifest_row.get("sex"),
            ),
        )
        metadata = getattr(result, "metadata", {}) or {}
        clinical = metadata.get("clinical_interpretation") or {}
        if not clinical:
            raise RuntimeError(
                "extractor returned no clinical_interpretation metadata"
            )
        quality = _quality_summary(result)
        return {
            **base,
            "extraction_status": "ok",
            "error": None,
            "traceback": None,
            "runtime_seconds": time.perf_counter() - started,
            "source_fs_hz": fs,
            "internal_fs_hz": fs_internal,
            "signal": _signal_summary(ecg),
            "n_beats": len(getattr(result, "beats", []) or []),
            "quality": quality,
            "record_quality": metadata.get("record_quality") or {},
            "pacing_state": metadata.get("pacing_state"),
            "global": _global_summary(result),
            "clinical": _compact_clinical(clinical),
            "extraction_contract": extraction_contract,
        }
    except Exception as exc:
        return {
            **base,
            "extraction_status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
            "source_fs_hz": None,
            "internal_fs_hz": fs_internal,
            "signal": {},
            "n_beats": None,
            "quality": {},
            "record_quality": {},
            "pacing_state": None,
            "global": {},
            "clinical": {},
            "extraction_contract": extraction_contract,
        }


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _record_output_path(raw_dir: Path, record: Mapping[str, Any]) -> Path:
    return raw_dir / str(record["dataset"]) / f"{record['record']}.json"


def _extract_all(
    manifest: Sequence[dict[str, Any]],
    *,
    out_dir: Path,
    workers: int,
    fs_internal: int,
    enable_pacing: bool,
    reuse_existing: bool,
) -> list[dict[str, Any]]:
    raw_dir = out_dir / "raw"
    for dataset in {row["dataset"] for row in manifest}:
        (raw_dir / dataset).mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for row in manifest:
        output_path = _record_output_path(raw_dir, row)
        if reuse_existing and output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                selected.append(row)
            else:
                expected_contract = _target_extraction_contract(
                    row,
                    fs_internal=fs_internal,
                    enable_pacing=enable_pacing,
                )
                if (
                    isinstance(existing, Mapping)
                    and existing.get("extraction_status") == "ok"
                    and expected_contract is not None
                    and existing.get("extraction_contract") == expected_contract
                ):
                    results.append(existing)
                else:
                    selected.append(row)
        else:
            selected.append(row)

    tasks = [
        (row, int(fs_internal), bool(enable_pacing))
        for row in selected
    ]
    total = len(manifest)
    completed = len(results)
    if completed:
        print(f"[resume] loaded {completed}/{total} existing compact results")

    def persist(result: dict[str, Any]) -> None:
        nonlocal completed
        _json_dump(_record_output_path(raw_dir, result), result)
        results.append(result)
        completed += 1
        print(
            f"[{completed}/{total}] {result['dataset']}/{result['record']} "
            f"{result['extraction_status']} "
            f"{float(result['runtime_seconds']):.2f}s",
            flush=True,
        )

    if workers <= 1:
        for task in tasks:
            persist(_extract_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_task = {
                pool.submit(_extract_one, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                persist(future.result())
    results.sort(key=lambda row: (row["dataset"], row["record"]))
    return results


def _active_specs(dataset: str) -> list[CategorySpec]:
    return [spec for spec in CATEGORIES if spec.refs(dataset)]


def _reference_categories(
    dataset: str, active_codes: set[str]
) -> set[str]:
    return {
        spec.key
        for spec in _active_specs(dataset)
        if active_codes & spec.refs(dataset)
    }


def _prediction_categories(
    dataset: str, prediction_codes: set[str]
) -> set[str]:
    return {
        spec.key
        for spec in _active_specs(dataset)
        if prediction_codes & spec.prediction_codes
    }


def _category_lookup() -> dict[str, CategorySpec]:
    return {spec.key: spec for spec in CATEGORIES}


def _rules_for_category(
    spec: CategorySpec,
    rules: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected = []
    for rule in rules:
        if rule.get("rule_id") in spec.rule_ids:
            selected.append(rule)
            continue
        evaluates = rule.get("evaluates_code")
        statement_code = rule.get("statement_code")
        if evaluates in spec.evaluates_codes:
            selected.append(rule)
        elif statement_code in spec.prediction_codes:
            selected.append(rule)
    return selected


def _short_json(value: Any, limit: int = 500) -> str:
    text = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _important_evidence(rule: Mapping[str, Any]) -> dict[str, Any]:
    evidence = rule.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    preferred = (
        "heart_rate_bpm",
        "age_years",
        "pr_ms",
        "qrs_ms",
        "qrs_axis_deg",
        "selected_qtc_ms",
        "qt_ms",
        "r_v1_mv",
        "s_v1_mv",
        "cornell_voltage_mv",
        "assessed_criteria",
        "matched_criteria",
        "v1_r_prime_mv",
        "v1_r_prime_duration_ms",
        "lateral_s",
        "candidate_beats",
        "count",
        "analyzable_beats",
        "burden_percent",
        "inverted_leads",
        "inverted_contiguous_pairs",
        "tall_positive_t_leads",
        "qualifying_contiguous_pairs",
        "matched_groups",
        "territory_results",
        "qualifying_leads",
        "disproving_leads",
        "not_applicable_by",
        "manual_confirmation_required",
    )
    selected = {key: evidence[key] for key in preferred if key in evidence}
    if not selected:
        selected = {
            key: value
            for key, value in evidence.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
    return selected


def _miss_reason(
    spec: CategorySpec,
    rules: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
) -> tuple[str, list[str]]:
    relevant = _rules_for_category(spec, rules)
    tags: list[str] = []
    if spec.semantic_scope in {"broad", "screening"}:
        tags.append("标签与规则语义粒度不一致")
    if not relevant:
        return (
            f"{spec.display_name_zh}：当前规则集中未找到可追溯的对应规则。",
            tags + ["权威输出覆盖缺口"],
        )

    statuses = Counter(str(rule.get("status") or "unknown") for rule in relevant)
    details: list[str] = []
    for rule in relevant:
        status = str(rule.get("status") or "unknown")
        rule_id = str(rule.get("rule_id") or "?")
        missing = list(rule.get("missing_inputs") or [])
        suppressed = list(rule.get("suppressed_by") or [])
        evidence = _important_evidence(rule)
        fragment = f"{rule_id}={status}"
        if missing:
            fragment += f"，缺少 {','.join(map(str, missing))}"
        if suppressed:
            fragment += f"，被 {','.join(map(str, suppressed))} 抑制"
        if evidence:
            fragment += f"，关键证据 {_short_json(evidence)}"
        details.append(fragment)
    if statuses.get("unavailable") or statuses.get("indeterminate"):
        tags.append("规则输入缺失或质量门控")
    if statuses.get("not_applicable"):
        tags.append("规则不适用")
    if statuses.get("suppressed"):
        tags.append("规则结果被抑制")
    if statuses.get("not_matched"):
        tags.append("规则阈值/形态未满足")
    if not tags:
        tags.append("最终语句解析或类别映射差异")

    quality = record.get("quality") or {}
    quality_note = (
        f"可用导联 QRS/P/T="
        f"{len(quality.get('qrs_reliable_leads') or [])}/"
        f"{len(quality.get('p_reliable_leads') or [])}/"
        f"{len(quality.get('t_reliable_leads') or [])}"
    )
    return (
        f"{spec.display_name_zh} 漏检；" + "；".join(details) + f"；{quality_note}。",
        tags,
    )


def _unsupported_reference_details(
    dataset: str,
    codes: Iterable[str],
    names: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    mapped_refs = set().union(
        *(spec.refs(dataset) for spec in _active_specs(dataset))
    )
    unsupported = sorted(set(codes) - mapped_refs)
    details = []
    for code in unsupported:
        reason = KNOWN_UNSUPPORTED_REFS.get(dataset, {}).get(code)
        label = names.get(code, code)
        if reason:
            details.append(f"{code} {label}：{reason}")
        else:
            details.append(
                f"{code} {label}：当前权威规则层无可比的直接诊断家族映射"
            )
    return unsupported, details


def _root_cause_improvements(tags: Iterable[str]) -> list[str]:
    tag_set = set(tags)
    suggestions: list[str] = []
    if "规则输入缺失或质量门控" in tag_set:
        suggestions.append(
            "提高 P/QRS/T 波界与形态字段的多导联可用率，并把缺失字段追溯到具体导联和心拍。"
        )
    if "规则阈值/形态未满足" in tag_set:
        suggestions.append(
            "在独立验证集上校准阈值，采用连续风险分数与灰区，而不是只用单一硬阈值。"
        )
    if (
        "规则不适用" in tag_set
        or "规则结果被抑制" in tag_set
    ):
        suggestions.append(
            "复核 AF/AFL、起搏和传导异常的控制流；保留被抑制候选及抑制原因供人工复核。"
        )
    if "标签与规则语义粒度不一致" in tag_set:
        suggestions.append(
            "建立专家确认的标签本体映射，分开评价直接诊断、形态证据和宽泛筛查家族。"
        )
    if "权威输出覆盖缺口" in tag_set:
        suggestions.append(
            "把已有轴、节律、起搏和形态测量投影为可审计的最终语句，或显式声明不支持。"
        )
    if "标签集无对应可比项" in tag_set:
        suggestions.append(
            "为短 QT、起搏、预激及质量语句补充相应参考标签或独立专家复核，避免误算为真阴性。"
        )
    if "潜在误报或参考标签不完备" in tag_set:
        suggestions.append(
            "对额外输出做盲法人工复核；记录级标签未必穷尽所有 ECG 异常，不能自动等同临床误报。"
        )
    if "执行失败" in tag_set:
        suggestions.append(
            "对输入格式、幅值单位、导联顺序和异常堆栈增加前置校验与可恢复降级路径。"
        )
    return suggestions


def _fmt_num(value: Any, digits: int = 1) -> str:
    parsed = _finite(value)
    return "NA" if parsed is None else f"{parsed:.{digits}f}"


def _statement_text(final_statements: Sequence[Mapping[str, Any]]) -> str:
    chunks = []
    for statement in final_statements:
        code = str(statement.get("statement_code") or "")
        label = PREDICTION_CODE_ZH.get(
            code, str(statement.get("statement") or code)
        )
        chunks.append(f"{code}:{label}")
    return "；".join(chunks)


def _record_verdict(
    *,
    extraction_ok: bool,
    reference_categories: set[str],
    hits: set[str],
    misses: set[str],
    extras: set[str],
    unsupported_codes: Sequence[str],
    unaligned_prediction_codes: Sequence[str],
) -> str:
    if not extraction_ok:
        return "执行失败"
    if misses and hits:
        return "部分命中但有漏诊"
    if misses and not hits:
        return "未命中可比参考异常"
    if reference_categories and not misses and extras:
        return "参考异常命中但有额外输出"
    if reference_categories and not misses:
        return "支持范围内一致"
    if extras:
        return "无可比参考阳性但有额外输出"
    if unaligned_prediction_codes:
        return "存在参考标签无法验证的算法输出"
    if unsupported_codes:
        return "仅含当前不支持/不可比标签"
    return "无可比阳性且无额外输出"


def _analyze_record(record: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(record["dataset"])
    names = record.get("reference_names") or {}
    active_codes = set(record.get("reference_codes_active") or [])
    extraction_ok = record.get("extraction_status") == "ok"
    clinical = record.get("clinical") or {}
    final_statements = list(clinical.get("final_statements") or [])
    prediction_codes = {
        str(item.get("statement_code"))
        for item in final_statements
        if item.get("status") == "matched" and item.get("statement_code")
    }
    reference_categories = _reference_categories(dataset, active_codes)
    predicted_categories = _prediction_categories(dataset, prediction_codes)
    mapped_prediction_codes = set().union(
        *(spec.prediction_codes for spec in _active_specs(dataset))
    ) | set(NONCOMPARABLE_OBSERVATION_CODES)
    unaligned_prediction_codes = sorted(
        prediction_codes - mapped_prediction_codes
    )
    hits = reference_categories & predicted_categories
    misses = reference_categories - predicted_categories
    extras = predicted_categories - reference_categories
    unsupported_codes, unsupported_details = _unsupported_reference_details(
        dataset, active_codes, names
    )

    lookup = _category_lookup()
    rules = list(clinical.get("rules") or [])
    cause_tags: list[str] = []
    detail_parts: list[str] = []
    for key in sorted(misses):
        reason, tags = _miss_reason(lookup[key], rules, record)
        detail_parts.append(reason)
        cause_tags.extend(tags)
    if extras:
        cause_tags.append("潜在误报或参考标签不完备")
        detail_parts.append(
            "额外输出："
            + "、".join(lookup[key].display_name_zh for key in sorted(extras))
            + "。这些是相对记录级标签的额外项，不能在无逐项专家复核时直接认定为临床误报。"
        )
    if unsupported_details:
        cause_tags.append("权威输出覆盖缺口")
        detail_parts.append("不可比/未支持标签：" + "；".join(unsupported_details))
    if unaligned_prediction_codes:
        cause_tags.append("标签集无对应可比项")
        detail_parts.append(
            "参考标签无法直接验证的算法输出："
            + "、".join(
                f"{code}:{PREDICTION_CODE_ZH.get(code, code)}"
                for code in unaligned_prediction_codes
            )
            + "。"
        )
    if not extraction_ok:
        cause_tags.append("执行失败")
        detail_parts.append(f"ecgfeat 执行失败：{record.get('error')}")
    cause_tags = list(dict.fromkeys(cause_tags))
    improvements = _root_cause_improvements(cause_tags)

    global_values = record.get("global") or {}
    quality = record.get("quality") or {}
    record_quality = record.get("record_quality") or {}
    promoted_quality_flags = sorted(
        str(code) for code in (record_quality.get("reason_codes") or [])
    )
    advisory_quality_flags = sorted(
        set(str(code) for code in (quality.get("quality_flags") or []))
        - set(promoted_quality_flags)
    )
    reference_text = "；".join(
        f"{code}:{names.get(code, code)}"
        + (
            f"({record.get('reference_scores', {}).get(code):g})"
            if code in (record.get("reference_scores") or {})
            else ""
        )
        for code in record.get("reference_codes_active") or []
    )
    measurement_text = (
        f"HR {_fmt_num(global_values.get('heart_rate_bpm'))} bpm，"
        f"PR {_fmt_num(global_values.get('pr_ms'), 0)} ms，"
        f"QRS {_fmt_num(global_values.get('qrs_ms'), 0)} ms，"
        f"QT {_fmt_num(global_values.get('qt_ms'), 0)} ms，"
        f"QTcB {_fmt_num(global_values.get('qtc_bazett_ms'), 0)} ms，"
        f"QRS轴 {_fmt_num(global_values.get('qrs_axis_deg'))}°，"
        f"心拍 {record.get('n_beats') if record.get('n_beats') is not None else 'NA'}，"
        f"QRS/P/T可用导联 "
        f"{len(quality.get('qrs_reliable_leads') or [])}/"
        f"{len(quality.get('p_reliable_leads') or [])}/"
        f"{len(quality.get('t_reliable_leads') or [])}"
    )
    analysis_text = (
        f"参考={reference_text or '无'}；"
        f"算法={_statement_text(final_statements) or '无权威异常语句'}；"
        f"测量={measurement_text}。"
    )
    if detail_parts:
        analysis_text += " " + " ".join(detail_parts)

    return {
        "dataset": dataset,
        "record": record["record"],
        "extraction_status": record.get("extraction_status"),
        "runtime_seconds": record.get("runtime_seconds"),
        "reference_policy": record.get("reference_policy"),
        "reference_codes_active": ",".join(
            record.get("reference_codes_active") or []
        ),
        "reference_labels_active": reference_text,
        "reference_codes_raw": ",".join(
            record.get("reference_codes_raw") or []
        ),
        "reference_report": record.get("reference_report") or "",
        "clinical_overall_status": clinical.get("overall_status"),
        "prediction_codes": ",".join(sorted(prediction_codes)),
        "prediction_statements": _statement_text(final_statements),
        "verdict": _record_verdict(
            extraction_ok=extraction_ok,
            reference_categories=reference_categories,
            hits=hits,
            misses=misses,
            extras=extras,
            unsupported_codes=unsupported_codes,
            unaligned_prediction_codes=unaligned_prediction_codes,
        ),
        "hit_categories": "、".join(
            lookup[key].display_name_zh for key in sorted(hits)
        ),
        "missed_categories": "、".join(
            lookup[key].display_name_zh for key in sorted(misses)
        ),
        "extra_categories": "、".join(
            lookup[key].display_name_zh for key in sorted(extras)
        ),
        "unsupported_reference_codes": ",".join(unsupported_codes),
        "unverifiable_prediction_codes": ",".join(
            unaligned_prediction_codes
        ),
        "root_cause_tags": " | ".join(cause_tags),
        "heart_rate_bpm": global_values.get("heart_rate_bpm"),
        "atrial_rate_bpm": global_values.get("atrial_rate_bpm"),
        "pr_ms": global_values.get("pr_ms"),
        "qrs_ms": global_values.get("qrs_ms"),
        "qt_ms": global_values.get("qt_ms"),
        "qtc_bazett_ms": global_values.get("qtc_bazett_ms"),
        "qtc_fridericia_ms": global_values.get("qtc_fridericia_ms"),
        "p_axis_deg": global_values.get("p_axis_deg"),
        "qrs_axis_deg": global_values.get("qrs_axis_deg"),
        "t_axis_deg": global_values.get("t_axis_deg"),
        "qt_reliability": global_values.get("qt_reliability"),
        "n_beats": record.get("n_beats"),
        "reliable_leads": len(quality.get("reliable_leads") or []),
        "qrs_reliable_leads": len(
            quality.get("qrs_reliable_leads") or []
        ),
        "p_reliable_leads": len(quality.get("p_reliable_leads") or []),
        "t_reliable_leads": len(quality.get("t_reliable_leads") or []),
        "quality_flags": ",".join(promoted_quality_flags),
        "quality_advisories": ",".join(advisory_quality_flags),
        "near_flat_leads": ",".join(
            (record.get("signal") or {}).get("near_flat_leads") or []
        ),
        "error": record.get("error") or "",
        "detailed_analysis_zh": analysis_text,
        "algorithm_improvements_zh": "；".join(improvements),
    }


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if recall is None:
        return None
    if recall == 0.0:
        return 0.0
    if precision is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _category_metrics(
    raw_records: Sequence[Mapping[str, Any]],
    dataset: str,
) -> list[dict[str, Any]]:
    records = [
        row
        for row in raw_records
        if row["dataset"] == dataset
        and row.get("extraction_status") == "ok"
    ]
    metrics: list[dict[str, Any]] = []
    for spec in _active_specs(dataset):
        tp = fp = fn = tn = 0
        for record in records:
            refs = set(record.get("reference_codes_active") or [])
            final = (record.get("clinical") or {}).get(
                "final_statements"
            ) or []
            predictions = {
                str(item.get("statement_code"))
                for item in final
                if item.get("status") == "matched"
                and item.get("statement_code")
            }
            truth = bool(refs & spec.refs(dataset))
            predicted = bool(predictions & spec.prediction_codes)
            if truth and predicted:
                tp += 1
            elif truth:
                fn += 1
            elif predicted:
                fp += 1
            else:
                tn += 1
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        metrics.append(
            {
                "dataset": dataset,
                "category": spec.key,
                "category_zh": spec.display_name_zh,
                "semantic_scope": spec.semantic_scope,
                "reference_codes": ",".join(sorted(spec.refs(dataset))),
                "prediction_codes": ",".join(
                    sorted(spec.prediction_codes)
                ),
                "ground_truth_positive": tp + fn,
                "predicted_positive": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "f1": _f1(precision, recall),
            }
        )
    return metrics


def _dataset_summary(
    raw_records: Sequence[Mapping[str, Any]],
    analyzed_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    raw = [row for row in raw_records if row["dataset"] == dataset]
    analyzed = [row for row in analyzed_rows if row["dataset"] == dataset]
    metrics = [row for row in metric_rows if row["dataset"] == dataset]
    ok = [row for row in raw if row.get("extraction_status") == "ok"]
    direct_metrics = [
        row for row in metrics if row.get("semantic_scope") == "direct"
    ]
    tp = sum(int(row["tp"]) for row in metrics)
    fp = sum(int(row["fp"]) for row in metrics)
    fn = sum(int(row["fn"]) for row in metrics)
    direct_tp = sum(int(row["tp"]) for row in direct_metrics)
    direct_fp = sum(int(row["fp"]) for row in direct_metrics)
    direct_fn = sum(int(row["fn"]) for row in direct_metrics)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    direct_precision = _safe_div(direct_tp, direct_tp + direct_fp)
    direct_recall = _safe_div(direct_tp, direct_tp + direct_fn)
    return {
        "dataset": dataset,
        "record_count": len(raw),
        "extraction_succeeded": len(ok),
        "extraction_failed": len(raw) - len(ok),
        "runtime_seconds_total": sum(
            float(row.get("runtime_seconds") or 0.0) for row in raw
        ),
        "verdict_counts": dict(
            Counter(str(row["verdict"]) for row in analyzed)
        ),
        "root_cause_record_counts": dict(
            Counter(
                tag
                for row in analyzed
                for tag in str(row["root_cause_tags"]).split(" | ")
                if tag
            )
        ),
        "supported_category_count": len(metrics),
        "direct_category_count": len(direct_metrics),
        "micro_tp": tp,
        "micro_fp": fp,
        "micro_fn": fn,
        "micro_precision": precision,
        "micro_recall_sensitivity": recall,
        "micro_f1": _f1(precision, recall),
        "direct_micro_tp": direct_tp,
        "direct_micro_fp": direct_fp,
        "direct_micro_fn": direct_fn,
        "direct_micro_precision": direct_precision,
        "direct_micro_recall_sensitivity": direct_recall,
        "direct_micro_f1": _f1(direct_precision, direct_recall),
    }


def _fmt_metric(value: Any) -> str:
    parsed = _finite(value)
    return "N/A" if parsed is None else f"{parsed:.3f}"


def _write_markdown_summary(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# ecgfeat 目标数据诊断评估",
        "",
        "> 研究与工程用途。参考标签是记录级标签，未提供逐导联、逐心拍或波界真值；",
        "> 下列指标是标签一致性，不是临床准确率，也不能替代医生判读。",
        "",
        "## 执行与总体结果",
        "",
        "| 数据集 | 记录 | 成功 | 失败 | 直接诊断 P/R/F1 | 全部映射 P/R/F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['dataset']}` | {summary['record_count']} | "
            f"{summary['extraction_succeeded']} | "
            f"{summary['extraction_failed']} | "
            f"{_fmt_metric(summary['direct_micro_precision'])}/"
            f"{_fmt_metric(summary['direct_micro_recall_sensitivity'])}/"
            f"{_fmt_metric(summary['direct_micro_f1'])} | "
            f"{_fmt_metric(summary['micro_precision'])}/"
            f"{_fmt_metric(summary['micro_recall_sensitivity'])}/"
            f"{_fmt_metric(summary['micro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "“直接诊断”只汇总语义可直接对应的家族；“全部映射”还包含 broad/screening，",
            "用于工程覆盖概览，不应与直接诊断 F1 混为同一个临床性能分数。",
            "",
        ]
    )
    for summary in summaries:
        dataset = summary["dataset"]
        lines.extend(
            [
                f"## 数据集 `{dataset}`",
                "",
                "### 逐记录结论计数",
                "",
            ]
        )
        for key, value in sorted(
            summary["verdict_counts"].items(),
            key=lambda item: (-int(item[1]), item[0]),
        ):
            lines.append(f"- {key}：{value}")
        lines.extend(["", "### 主要失败原因", ""])
        for key, value in sorted(
            summary["root_cause_record_counts"].items(),
            key=lambda item: (-int(item[1]), item[0]),
        ):
            lines.append(f"- {key}：{value} 条")
        lines.extend(
            [
                "",
                "### 分诊断家族结果",
                "",
                "| 诊断家族 | 参考阳性 | 预测阳性 | TP | FP | FN | 精确率 | 召回率 | F1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metric_rows:
            if row["dataset"] != dataset:
                continue
            lines.append(
                f"| {row['category_zh']} | {row['ground_truth_positive']} | "
                f"{row['predicted_positive']} | {row['tp']} | {row['fp']} | "
                f"{row['fn']} | {_fmt_metric(row['precision'])} | "
                f"{_fmt_metric(row['recall_sensitivity'])} | "
                f"{_fmt_metric(row['f1'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 输出说明",
            "",
            "- `逐条诊断结果_<dataset>.csv`：逐记录诊断、测量、原因和改进建议。",
            "- `诊断家族指标.csv`：逐诊断家族混淆矩阵和指标。",
            "- `summary.json`：机器可读总体统计。",
            "- `raw/<dataset>/<record>.json`：当前 ecgfeat 规则状态与关键证据，供逐例审计。",
            "",
            "## 解释边界",
            "",
            "- PTB-XL：诊断 SCP 代码仅在 likelihood ≥50 时作为严格阳性；form/rhythm 描述码按存在处理。",
            "- `010`：按头文件 `#Dx` 全部 SNOMED-CT 代码处理；标签不保证穷尽该 ECG 的所有异常。",
            "- “额外输出”只表示标签集中没有同家族阳性，不自动等价于临床误报。",
            "- 宽泛的 ST/缺血/梗死标签与 ecgfeat 的形态筛查语句语义不同，已明确标为 screening/broad。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_analysis(
    raw_records: Sequence[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    analyzed = [_analyze_record(record) for record in raw_records]
    analyzed.sort(key=lambda row: (row["dataset"], row["record"]))
    active_datasets = [
        dataset
        for dataset in (DATASET_PTBXL, DATASET_LOCAL)
        if any(record.get("dataset") == dataset for record in raw_records)
    ]
    for dataset in active_datasets:
        rows = [row for row in analyzed if row["dataset"] == dataset]
        _write_csv(out_dir / f"逐条诊断结果_{dataset}.csv", rows)
    metric_rows = [
        row
        for dataset in active_datasets
        for row in _category_metrics(raw_records, dataset)
    ]
    _write_csv(out_dir / "诊断家族指标.csv", metric_rows)
    summaries = [
        _dataset_summary(raw_records, analyzed, metric_rows, dataset)
        for dataset in active_datasets
    ]
    payload = {
        "schema_version": "ecgfeat_target_diagnosis_evaluation.v1",
        "prediction_source": (
            "clinical_interpretation.final_statements, matched only"
        ),
        "reference_level": "record-level labels",
        "datasets": summaries,
        "release_gate": {
            "operational_coverage_complete": all(
                int(summary["extraction_failed"]) == 0 for summary in summaries
            ),
            "failed_records": sum(
                int(summary["extraction_failed"]) for summary in summaries
            ),
        },
        "category_mapping": [
            {
                "category": spec.key,
                "category_zh": spec.display_name_zh,
                "ptbxl_refs": sorted(spec.ptbxl_refs),
                "local_refs": sorted(spec.local_refs),
                "prediction_codes": sorted(spec.prediction_codes),
                "semantic_scope": spec.semantic_scope,
            }
            for spec in CATEGORIES
        ],
    }
    _json_dump(out_dir / "summary.json", payload)
    _write_markdown_summary(
        out_dir / "RESULTS.md", summaries, metric_rows
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ptbxl-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "00000",
    )
    parser.add_argument(
        "--ptbxl-metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ptb-xl-metadata-1.0.1",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "010",
    )
    parser.add_argument(
        "--condition-csv",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
            / "ConditionNames_SNOMED-CT.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "ecgfeat_target_diagnosis_20260726",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--fs-internal", type=int, default=500)
    parser.add_argument(
        "--disable-pacing",
        action="store_true",
        help="Disable pacing detector (normally leave enabled at 500 Hz).",
    )
    parser.add_argument(
        "--datasets",
        choices=("both", DATASET_PTBXL, DATASET_LOCAL),
        default="both",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test limit applied independently to each selected dataset.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse compact per-record raw results already present in out-dir.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.disable_pacing and args.fs_internal <= 200:
        raise SystemExit(
            "pacing detection requires a higher internal rate; use "
            "--fs-internal 500 or --disable-pacing"
        )
    selected: list[dict[str, Any]] = []
    if args.datasets in {"both", DATASET_PTBXL}:
        ptbxl, _ = _load_ptbxl_manifest(
            args.ptbxl_dir.resolve(),
            args.ptbxl_metadata_dir.resolve(),
        )
        if args.limit is not None:
            ptbxl = ptbxl[: max(0, args.limit)]
        selected.extend(ptbxl)
    if args.datasets in {"both", DATASET_LOCAL}:
        local = _load_local_manifest(
            args.local_dir.resolve(), args.condition_csv.resolve()
        )
        if args.limit is not None:
            local = local[: max(0, args.limit)]
        selected.extend(local)
    if not selected:
        raise SystemExit("no records selected")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_records = _extract_all(
        selected,
        out_dir=out_dir,
        workers=max(1, int(args.workers)),
        fs_internal=int(args.fs_internal),
        enable_pacing=not args.disable_pacing,
        reuse_existing=bool(args.reuse_existing),
    )
    payload = _run_analysis(raw_records, out_dir)
    payload["wall_runtime_seconds"] = time.perf_counter() - started
    _json_dump(out_dir / "summary.json", payload)
    print(
        f"completed {len(raw_records)} records in "
        f"{payload['wall_runtime_seconds']:.1f}s; output={out_dir}",
        flush=True,
    )
    return (
        0
        if payload["release_gate"]["operational_coverage_complete"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate ecgfeat diagnostic statements against LUDB record diagnoses.

This evaluator deliberately reads predictions only from
``clinical_interpretation.final_statements`` in ecgfeat feature JSON files.
The diagnosis text embedded in the generated human-readable reports is not
used, because it is copied from the LUDB header and would leak the reference
labels into the prediction.

The LUDB vocabulary and the ecgfeat statement vocabulary are different.
Consequently, the primary benchmark uses explicitly documented diagnosis
families rather than fuzzy text matching.  A second set of LUDB families with
no corresponding authoritative ecgfeat statement is reported as unsupported
coverage rather than silently discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LUDB_DIR = (
    PROJECT_ROOT
    / "data"
    / "lobachevsky-university-electrocardiography-database-1.0.1"
    / "data"
)
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "ludb_full_compare_20260726_qrs_tail_rescue"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ludb_diagnosis_evaluation"


@dataclass(frozen=True)
class CategorySpec:
    key: str
    display_name: str
    gt_match: Callable[[Sequence[str]], bool]
    prediction_codes: frozenset[str]
    mapping_note: str

    @property
    def supported(self) -> bool:
        return bool(self.prediction_codes)


def _exact(*targets: str) -> Callable[[Sequence[str]], bool]:
    target_set = frozenset(targets)
    return lambda labels: bool(target_set.intersection(labels))


def _prefix(*prefixes: str) -> Callable[[Sequence[str]], bool]:
    return lambda labels: any(
        label.startswith(prefix) for label in labels for prefix in prefixes
    )


def _contains(*needles: str) -> Callable[[Sequence[str]], bool]:
    return lambda labels: any(
        needle in label for label in labels for needle in needles
    )


SUPPORTED_CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(
        "sinus_bradycardia",
        "Sinus bradycardia / bradycardia",
        _exact("Rhythm: Sinus bradycardia"),
        frozenset({"bradycardia"}),
        "LUDB sinus bradycardia is compared with the ecgfeat rate statement.",
    ),
    CategorySpec(
        "sinus_tachycardia",
        "Sinus tachycardia / tachycardia",
        _exact("Rhythm: Sinus tachycardia"),
        frozenset({"tachycardia"}),
        "LUDB sinus tachycardia is compared with the ecgfeat rate statement.",
    ),
    CategorySpec(
        "atrial_fibrillation",
        "Atrial fibrillation",
        _exact("Rhythm: Atrial fibrillation"),
        frozenset({"atrial_fibrillation_pattern"}),
        "Indeterminate AF/AFL statements are not counted as definite AF.",
    ),
    CategorySpec(
        "atrial_flutter",
        "Atrial flutter",
        _exact("Rhythm: Atrial flutter, typical"),
        frozenset({"atrial_flutter_pattern"}),
        "Indeterminate AF/AFL statements are not counted as definite flutter.",
    ),
    CategorySpec(
        "premature_atrial_complexes",
        "Premature atrial complexes",
        _prefix("Atrial extrasystole"),
        frozenset({"premature_atrial_complexes"}),
        "All LUDB atrial-extrasystole subtypes are collapsed to PAC.",
    ),
    CategorySpec(
        "premature_ventricular_complexes",
        "Premature ventricular complexes",
        _prefix("Ventricular extrasystole"),
        frozenset({"premature_ventricular_complexes"}),
        "All LUDB ventricular-extrasystole subtypes are collapsed to PVC.",
    ),
    CategorySpec(
        "first_degree_av_block",
        "First-degree AV block/delay",
        _exact("I degree AV block"),
        frozenset({"first_degree_av_delay"}),
        "The ecgfeat wording is delay; LUDB uses block.",
    ),
    CategorySpec(
        "complete_av_block",
        "Complete/third-degree AV block",
        _exact("III degree AV-block"),
        frozenset({"complete_av_block_pattern"}),
        "Only a matched complete-AV-block statement counts as positive.",
    ),
    CategorySpec(
        "right_bundle_branch_block",
        "Right bundle branch block",
        _exact(
            "Incomplete right bundle branch block",
            "Complete right bundle branch block",
        ),
        frozenset({"rbbb_pattern", "probable_rbbb_pattern"}),
        "Complete and incomplete LUDB RBBB are pooled because ecgfeat often reports probable RBBB.",
    ),
    CategorySpec(
        "left_bundle_branch_block",
        "Left bundle branch block",
        _exact(
            "Incomplete left bundle branch block",
            "Complete left bundle branch block",
        ),
        frozenset({"lbbb_pattern", "probable_lbbb_pattern"}),
        "Complete and incomplete LUDB LBBB are pooled.",
    ),
    CategorySpec(
        "left_anterior_fascicular_block",
        "Left anterior fascicular block",
        _exact("Left anterior hemiblock"),
        frozenset({"lafb_pattern"}),
        "LUDB hemiblock is mapped to the ecgfeat fascicular-block statement.",
    ),
    CategorySpec(
        "nonspecific_ivcd",
        "Nonspecific intraventricular conduction delay",
        _contains("intravintricular conduction delay"),
        frozenset({"nonspecific_ivcd"}),
        "The matcher preserves LUDB's source spelling 'intravintricular'.",
    ),
    CategorySpec(
        "left_ventricular_hypertrophy",
        "Left ventricular hypertrophy/overload",
        _exact("Left ventricular hypertrophy", "Left ventricular overload"),
        frozenset({"lvh_voltage_criteria"}),
        "LUDB LVH/overload is compared with the authoritative LVH voltage statement.",
    ),
    CategorySpec(
        "nonspecific_repolarization_abnormality",
        "Nonspecific repolarization abnormality",
        _prefix(
            "Non-specific repolarization abnormalities",
            "Non- specific repolarization abnormalities",
        ),
        frozenset(
            {"primary_t_wave_abnormality", "secondary_t_wave_abnormality"}
        ),
        "Territories are pooled; either primary or secondary T-wave abnormality counts.",
    ),
    CategorySpec(
        "ischemia_or_stemi",
        "Ischemia or STEMI",
        _prefix("Ischemia:", "STEMI:"),
        frozenset(
            {
                "acute_occlusion_pattern",
                "sgarbossa_positive",
                "posterior_ischemia_screen",
            }
        ),
        "Broad screening family; it does not validate acuity or territory.",
    ),
    CategorySpec(
        "scar_or_indeterminate_infarction",
        "Scar or age-indeterminate infarction",
        _prefix("Scar formation:", "Undefined ischemia/scar/supp.NSTEMI:"),
        frozenset({"prior_infarct_q_wave_pattern"}),
        "Broad screening family; it does not validate territory.",
    ),
)


UNSUPPORTED_CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(
        "sinus_rhythm",
        "Sinus rhythm",
        _exact("Rhythm: Sinus rhythm"),
        frozenset(),
        "No positive normal-sinus statement is emitted in authoritative final statements.",
    ),
    CategorySpec(
        "other_sinus_dysrhythmia",
        "Sinus arrhythmia / irregular sinus rhythm",
        _exact("Rhythm: Sinus arrhythmia", "Rhythm: Irregular sinus rhythm"),
        frozenset(),
        "No directly corresponding authoritative final statement.",
    ),
    CategorySpec(
        "normal_or_positional_axis",
        "Normal/vertical/horizontal QRS axis",
        _exact(
            "Electric axis of the heart: normal",
            "Electric axis of the heart: vertical",
            "Electric axis of the heart: horizontal",
        ),
        frozenset(),
        "Axis is measured elsewhere in ecgfeat but not emitted as an authoritative final diagnosis.",
    ),
    CategorySpec(
        "left_axis_deviation",
        "Left axis deviation",
        _exact("Electric axis of the heart: left axis deviation"),
        frozenset(),
        "Axis is measured elsewhere in ecgfeat but not emitted as an authoritative final diagnosis.",
    ),
    CategorySpec(
        "right_axis_deviation",
        "Right axis deviation",
        _exact("Electric axis of the heart: right axis deviation"),
        frozenset(),
        "Axis is measured elsewhere in ecgfeat but not emitted as an authoritative final diagnosis.",
    ),
    CategorySpec(
        "atrial_enlargement_or_overload",
        "Atrial hypertrophy/overload",
        _contains(
            "atrial hypertrophy",
            "atrial overload",
        ),
        frozenset(),
        "No corresponding authoritative final statement was present in this run.",
    ),
    CategorySpec(
        "right_ventricular_hypertrophy",
        "Right ventricular hypertrophy",
        _exact("Right ventricular hypertrophy"),
        frozenset(),
        "No corresponding authoritative final statement was present in this run.",
    ),
    CategorySpec(
        "ventricular_pacing",
        "Ventricular pacing",
        _contains("ventricular pacing"),
        frozenset(),
        "Pacing affects interpretation policy but is not emitted as an authoritative final diagnosis.",
    ),
    CategorySpec(
        "early_repolarization",
        "Early repolarization",
        _exact("Early repolarization syndrome"),
        frozenset(),
        "No corresponding authoritative final statement was present in this run.",
    ),
)


ALL_CATEGORIES = SUPPORTED_CATEGORIES + UNSUPPORTED_CATEGORIES


BROAD_SCREENING_CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(
        "atrial_fibrillation_or_flutter",
        "AF or atrial flutter",
        lambda labels: any(
            label in {"Rhythm: Atrial fibrillation", "Rhythm: Atrial flutter, typical"}
            for label in labels
        ),
        frozenset(
            {
                "atrial_fibrillation_pattern",
                "atrial_flutter_pattern",
                "atrial_fibrillation_flutter_indeterminate",
            }
        ),
        "Definite and indeterminate AF/AFL outputs are pooled.",
    ),
    CategorySpec(
        "any_ectopy",
        "Any PAC or PVC",
        lambda labels: any(
            label.startswith(("Atrial extrasystole", "Ventricular extrasystole"))
            for label in labels
        ),
        frozenset(
            {"premature_atrial_complexes", "premature_ventricular_complexes"}
        ),
        "Atrial and ventricular ectopy are pooled without subtype or localization.",
    ),
    CategorySpec(
        "any_bbb_or_ivcd",
        "Any bundle-branch block or IVCD",
        lambda labels: any(
            label
            in {
                "Incomplete right bundle branch block",
                "Complete right bundle branch block",
                "Incomplete left bundle branch block",
                "Complete left bundle branch block",
                "Non-specific intravintricular conduction delay",
            }
            for label in labels
        ),
        frozenset(
            {
                "rbbb_pattern",
                "probable_rbbb_pattern",
                "lbbb_pattern",
                "probable_lbbb_pattern",
                "nonspecific_ivcd",
            }
        ),
        "Laterality and complete/incomplete subtype are ignored.",
    ),
    CategorySpec(
        "any_av_block",
        "Any first-, second-, or third-degree AV block",
        lambda labels: any(
            label in {"I degree AV block", "III degree AV-block"} for label in labels
        ),
        frozenset(
            {
                "first_degree_av_delay",
                "second_degree_av_block_pattern",
                "complete_av_block_pattern",
            }
        ),
        "AV-block grade is ignored; LUDB has no second-degree label in this set.",
    ),
    CategorySpec(
        "any_repolarization_ischemia_or_scar",
        "Any repolarization, ischemia, STEMI, or scar finding",
        lambda labels: any(
            label.startswith(
                (
                    "Non-specific repolarization",
                    "Non- specific repolarization",
                    "Ischemia:",
                    "STEMI:",
                    "Scar formation:",
                    "Undefined ischemia/scar/supp.NSTEMI:",
                )
            )
            for label in labels
        ),
        frozenset(
            {
                "primary_t_wave_abnormality",
                "secondary_t_wave_abnormality",
                "acute_occlusion_pattern",
                "sgarbossa_positive",
                "posterior_ischemia_screen",
                "prior_infarct_q_wave_pattern",
            }
        ),
        "A deliberately permissive screen that ignores acuity, mechanism, and territory.",
    ),
)


BENIGN_REFERENCE_LABELS = frozenset(
    {
        "Rhythm: Sinus rhythm",
        "Electric axis of the heart: normal",
        "Electric axis of the heart: vertical",
        "Electric axis of the heart: horizontal",
    }
)


def _parse_ludb_labels(header_path: Path) -> list[str]:
    labels: list[str] = []
    in_diagnosis_block = False
    for raw_line in header_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("#<diagnoses>:"):
            in_diagnosis_block = True
            continue
        if in_diagnosis_block and line.startswith("#"):
            label = line[1:].strip().rstrip(".")
            if label:
                labels.append(label)
        elif in_diagnosis_block:
            in_diagnosis_block = False
    return labels


def _load_prediction(feature_path: Path) -> tuple[set[str], str]:
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    clinical = payload.get("clinical_interpretation") or {}
    codes = {
        str(statement["statement_code"])
        for statement in clinical.get("final_statements") or []
        if statement.get("status") == "matched" and statement.get("statement_code")
    }
    return codes, str(clinical.get("overall_status") or "unknown")


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if recall is None:
        return None
    if precision is None:
        return 0.0 if recall == 0.0 else None
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator <= 0:
        return None
    return (tp * tn - fp * fn) / math.sqrt(denominator)


def _confusion_metrics(
    *,
    category: str,
    display_name: str,
    scope: str,
    mapping_note: str,
    prediction_codes: Iterable[str],
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, object]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "category": category,
        "display_name": display_name,
        "scope": scope,
        "ground_truth_positive": tp + fn,
        "predicted_positive": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": _f1(precision, recall),
        "accuracy": accuracy,
        "balanced_accuracy": (
            None if recall is None or specificity is None else (recall + specificity) / 2.0
        ),
        "mcc": _mcc(tp, fp, fn, tn),
        "prediction_codes": " | ".join(sorted(prediction_codes)),
        "mapping_note": mapping_note,
    }


def _aggregate_metric_rows(
    rows: Sequence[dict[str, object]],
    record_count: int,
) -> dict[str, object]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _f1(precision, recall)

    def macro(field: str) -> float | None:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return mean(values) if values else None

    return {
        "record_count": record_count,
        "category_count": len(rows),
        "micro_tp": tp,
        "micro_fp": fp,
        "micro_fn": fn,
        "micro_tn": tn,
        "micro_precision": precision,
        "micro_recall_sensitivity": recall,
        "micro_specificity": specificity,
        "micro_f1": f1,
        "macro_precision": macro("precision"),
        "macro_recall_sensitivity": macro("recall_sensitivity"),
        "macro_specificity": macro("specificity"),
        "macro_f1": macro("f1"),
        "macro_balanced_accuracy": macro("balanced_accuracy"),
    }


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_metric(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _measurement_summary(summary_path: Path) -> list[dict[str, object]]:
    if not summary_path.exists():
        return []
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    specs = (
        ("HR", "diff_hr", None),
        ("PR", "diff_pr", None),
        ("QRS", "diff_qrs", None),
        ("QT", "diff_qt", None),
        ("QTcB", "diff_qtcb", None),
        ("QTcF", "diff_qtcf", None),
        ("P axis", None, ("algorithm_p_axis", "ground_truth_p_axis")),
        ("QRS axis", None, ("algorithm_qrs_axis", "ground_truth_qrs_axis")),
        ("T axis", None, ("algorithm_t_axis", "ground_truth_t_axis")),
        ("QT dispersion", "diff_qt_dispersion", None),
    )
    results: list[dict[str, object]] = []
    for name, field, circular_fields in specs:
        values: list[float] = []
        for row in rows:
            if circular_fields is not None:
                algorithm_token = row.get(circular_fields[0], "")
                reference_token = row.get(circular_fields[1], "")
                if algorithm_token not in (None, "") and reference_token not in (
                    None,
                    "",
                ):
                    raw_difference = float(algorithm_token) - float(reference_token)
                    values.append((raw_difference + 180.0) % 360.0 - 180.0)
            else:
                token = row.get(str(field), "")
                if token not in (None, ""):
                    values.append(float(token))
        absolute_values = [abs(value) for value in values]
        results.append(
            {
                "measurement": name,
                "comparable": len(values),
                "bias": None if not values else mean(values),
                "mae": None if not absolute_values else mean(absolute_values),
            }
        )
    return results


def evaluate(
    ludb_dir: Path,
    feature_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    headers = sorted(
        ludb_dir.glob("*.hea"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if not headers:
        raise FileNotFoundError(f"No LUDB .hea files found under {ludb_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    gt_label_counts: Counter[str] = Counter()
    prediction_code_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for header_path in headers:
        record_id = header_path.stem
        feature_path = feature_dir / record_id / f"{record_id}_features.json"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature prediction: {feature_path}")
        labels = _parse_ludb_labels(header_path)
        codes, overall_status = _load_prediction(feature_path)
        gt_label_counts.update(labels)
        prediction_code_counts.update(codes)
        status_counts[overall_status] += 1
        records.append(
            {
                "record_id": record_id,
                "labels": labels,
                "codes": codes,
                "overall_status": overall_status,
            }
        )

    metric_rows: list[dict[str, object]] = []
    for spec in ALL_CATEGORIES:
        tp = fp = fn = tn = 0
        for record in records:
            gt_positive = spec.gt_match(record["labels"])
            predicted_positive = bool(spec.prediction_codes.intersection(record["codes"]))
            if gt_positive and predicted_positive:
                tp += 1
            elif predicted_positive:
                fp += 1
            elif gt_positive:
                fn += 1
            else:
                tn += 1
        metric_rows.append(
            _confusion_metrics(
                category=spec.key,
                display_name=spec.display_name,
                scope="supported" if spec.supported else "unsupported_output_family",
                mapping_note=spec.mapping_note,
                prediction_codes=spec.prediction_codes,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
            )
        )

    supported_rows = [row for row in metric_rows if row["scope"] == "supported"]
    broad_rows: list[dict[str, object]] = []
    for spec in BROAD_SCREENING_CATEGORIES:
        tp = fp = fn = tn = 0
        for record in records:
            gt_positive = spec.gt_match(record["labels"])
            predicted_positive = bool(spec.prediction_codes.intersection(record["codes"]))
            if gt_positive and predicted_positive:
                tp += 1
            elif predicted_positive:
                fp += 1
            elif gt_positive:
                fn += 1
            else:
                tn += 1
        broad_rows.append(
            _confusion_metrics(
                category=spec.key,
                display_name=spec.display_name,
                scope="broad_screening",
                mapping_note=spec.mapping_note,
                prediction_codes=spec.prediction_codes,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
            )
        )
    record_rows: list[dict[str, object]] = []
    record_jaccards: list[float] = []
    reference_positive_jaccards: list[float] = []
    record_exact_matches = 0
    reference_positive_exact_matches = 0
    reference_positive_record_count = 0
    for record in records:
        gt_categories = {
            spec.key for spec in SUPPORTED_CATEGORIES if spec.gt_match(record["labels"])
        }
        predicted_categories = {
            spec.key
            for spec in SUPPORTED_CATEGORIES
            if spec.prediction_codes.intersection(record["codes"])
        }
        true_categories = gt_categories.intersection(predicted_categories)
        false_positive_categories = predicted_categories - gt_categories
        false_negative_categories = gt_categories - predicted_categories
        union = gt_categories.union(predicted_categories)
        jaccard = 1.0 if not union else len(true_categories) / len(union)
        exact_match = gt_categories == predicted_categories
        record_jaccards.append(jaccard)
        record_exact_matches += int(exact_match)
        if gt_categories:
            reference_positive_record_count += 1
            reference_positive_jaccards.append(jaccard)
            reference_positive_exact_matches += int(exact_match)
        record_rows.append(
            {
                "record_id": record["record_id"],
                "ground_truth_labels": " | ".join(record["labels"]),
                "authoritative_statement_codes": " | ".join(sorted(record["codes"])),
                "ground_truth_supported_categories": " | ".join(sorted(gt_categories)),
                "predicted_supported_categories": " | ".join(sorted(predicted_categories)),
                "true_positive_categories": " | ".join(sorted(true_categories)),
                "false_positive_categories": " | ".join(
                    sorted(false_positive_categories)
                ),
                "false_negative_categories": " | ".join(
                    sorted(false_negative_categories)
                ),
                "jaccard": jaccard,
                "exact_category_set_match": exact_match,
                "overall_status": record["overall_status"],
            }
        )

    all_gt_label_instances = sum(gt_label_counts.values())
    supported_mapped_gt_label_instances = 0
    mapped_gt_label_instances = 0
    unmapped_label_counts: Counter[str] = Counter()
    for label, count in gt_label_counts.items():
        if any(spec.gt_match([label]) for spec in SUPPORTED_CATEGORIES):
            supported_mapped_gt_label_instances += count
        if any(spec.gt_match([label]) for spec in ALL_CATEGORIES):
            mapped_gt_label_instances += count
        else:
            unmapped_label_counts[label] += count

    low_complexity_normal_like = [
        record
        for record in records
        if set(record["labels"]).issubset(BENIGN_REFERENCE_LABELS)
    ]
    predicted_abnormal = [
        record
        for record in records
        if record["overall_status"]
        in {
            "abnormal",
            "abnormal_with_limited_coverage",
            "technically_limited_with_findings",
        }
    ]
    triage_tp = sum(
        1
        for record in records
        if record not in low_complexity_normal_like and record in predicted_abnormal
    )
    triage_fp = sum(
        1
        for record in records
        if record in low_complexity_normal_like and record in predicted_abnormal
    )
    triage_fn = sum(
        1
        for record in records
        if record not in low_complexity_normal_like and record not in predicted_abnormal
    )
    triage_tn = sum(
        1
        for record in records
        if record in low_complexity_normal_like and record not in predicted_abnormal
    )
    triage = _confusion_metrics(
        category="abnormal_triage_surrogate",
        display_name="Any abnormal finding (surrogate)",
        scope="surrogate",
        mapping_note=(
            "Reference normal-like means the LUDB labels contain only sinus rhythm "
            "plus normal/vertical/horizontal axis. This is not an official LUDB normal label."
        ),
        prediction_codes=(),
        tp=triage_tp,
        fp=triage_fp,
        fn=triage_fn,
        tn=triage_tn,
    )

    measurement_rows = _measurement_summary(feature_dir / "summary.csv")
    summary = {
        "schema_version": "ludb_ecgfeat_diagnosis_evaluation.v1",
        "prediction_source": "clinical_interpretation.final_statements matched codes only",
        "reference_source": "LUDB header <diagnoses> blocks",
        "record_count": len(records),
        "feature_file_count": len(records),
        "ground_truth_label_instances": all_gt_label_instances,
        "ground_truth_unique_labels": len(gt_label_counts),
        "mapped_ground_truth_label_instances": mapped_gt_label_instances,
        "mapped_ground_truth_label_fraction": _safe_div(
            mapped_gt_label_instances, all_gt_label_instances
        ),
        "supported_mapped_ground_truth_label_instances": supported_mapped_gt_label_instances,
        "supported_mapped_ground_truth_label_fraction": _safe_div(
            supported_mapped_gt_label_instances, all_gt_label_instances
        ),
        "supported_category_metrics": _aggregate_metric_rows(
            supported_rows, len(records)
        ),
        "broad_screening_metrics": _aggregate_metric_rows(broad_rows, len(records)),
        "record_level_supported_categories": {
            "exact_set_matches": record_exact_matches,
            "exact_set_match_fraction": _safe_div(record_exact_matches, len(records)),
            "mean_jaccard": mean(record_jaccards),
            "reference_positive_record_count": reference_positive_record_count,
            "reference_positive_exact_set_matches": reference_positive_exact_matches,
            "reference_positive_exact_set_match_fraction": _safe_div(
                reference_positive_exact_matches, reference_positive_record_count
            ),
            "reference_positive_mean_jaccard": mean(reference_positive_jaccards),
        },
        "overall_status_counts": dict(sorted(status_counts.items())),
        "normal_like_surrogate_record_count": len(low_complexity_normal_like),
        "abnormal_triage_surrogate": triage,
        "measurement_summary": measurement_rows,
        "important_audits": {
            "low_qrs_voltage_precordial_leads_predictions": prediction_code_counts.get(
                "low_qrs_voltage_precordial_leads", 0
            ),
            "indeterminate_af_flutter_predictions": prediction_code_counts.get(
                "atrial_fibrillation_flutter_indeterminate", 0
            ),
            "second_degree_av_block_predictions": prediction_code_counts.get(
                "second_degree_av_block_pattern", 0
            ),
        },
    }

    _write_csv(metric_rows, output_dir / "diagnostic_metrics.csv")
    _write_csv(broad_rows, output_dir / "broad_screening_metrics.csv")
    _write_csv(record_rows, output_dir / "record_diagnostic_results.csv")
    _write_csv(
        [
            {"statement_code": code, "count": count}
            for code, count in prediction_code_counts.most_common()
        ],
        output_dir / "predicted_statement_counts.csv",
    )
    _write_csv(
        [
            {"ground_truth_label": label, "count": count}
            for label, count in unmapped_label_counts.most_common()
        ],
        output_dir / "unmapped_ground_truth_labels.csv",
    )
    _write_csv(measurement_rows, output_dir / "measurement_metrics.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    supported = summary["supported_category_metrics"]
    broad = summary["broad_screening_metrics"]
    record_level = summary["record_level_supported_categories"]
    report_lines = [
        "# ecgfeat on LUDB: diagnosis evaluation",
        "",
        "## Conclusion",
        "",
        (
            "**The current ecgfeat run cannot be considered a correct or complete LUDB "
            "diagnostic system.** It is substantially stronger as a measurement and "
            "wave-delineation pipeline than as a record-level disease classifier."
        ),
        "",
        "## Evaluation guardrails",
        "",
        f"- Records: {len(records)} / {len(headers)} completed.",
        "- Predictions: only matched codes in `clinical_interpretation.final_statements`.",
        "- References: only LUDB `<diagnoses>` header labels.",
        "- Generated report `Dx Codes`, `Dx Detail`, and `Hx` fields were excluded because they copy LUDB labels.",
        "- Category mappings are explicit in `diagnostic_metrics.csv`; no fuzzy text matching was used.",
        "- LUDB is primarily a 200-record ECG delineation resource; this diagnosis-label benchmark is exploratory, not external clinical validation.",
        "- Dataset reference: https://physionet.org/content/ludb/1.0.1/",
        "",
        "## Supported diagnostic-family results",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Micro precision | {_fmt_metric(supported['micro_precision'])} |",
        f"| Micro recall/sensitivity | {_fmt_metric(supported['micro_recall_sensitivity'])} |",
        f"| Micro F1 | {_fmt_metric(supported['micro_f1'])} |",
        f"| Micro specificity | {_fmt_metric(supported['micro_specificity'])} |",
        f"| Macro F1 | {_fmt_metric(supported['macro_f1'])} |",
        (
            "| Record exact category-set match (reference-positive subset) | "
            f"{record_level['reference_positive_exact_set_matches']}/"
            f"{record_level['reference_positive_record_count']} "
            f"({_fmt_metric(record_level['reference_positive_exact_set_match_fraction'])}) |"
        ),
        f"| Mean record Jaccard (reference-positive subset) | {_fmt_metric(record_level['reference_positive_mean_jaccard'])} |",
        "",
        "| Diagnosis family | GT+ | Pred+ | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in supported_rows:
        report_lines.append(
            f"| {row['display_name']} | {row['ground_truth_positive']} | "
            f"{row['predicted_positive']} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {_fmt_metric(row['precision'])} | "
            f"{_fmt_metric(row['recall_sensitivity'])} | {_fmt_metric(row['f1'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Permissive broad-screening results",
            "",
            (
                "These mappings intentionally ignore subtype, territory, and sometimes "
                "acuity. They are an upper-bound screening view, not exact diagnosis."
            ),
            "",
            "| Metric | Result |",
            "| --- | ---: |",
            f"| Micro precision | {_fmt_metric(broad['micro_precision'])} |",
            f"| Micro recall/sensitivity | {_fmt_metric(broad['micro_recall_sensitivity'])} |",
            f"| Micro F1 | {_fmt_metric(broad['micro_f1'])} |",
            "",
            "| Broad family | GT+ | Pred+ | TP | FP | FN | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in broad_rows:
        report_lines.append(
            f"| {row['display_name']} | {row['ground_truth_positive']} | "
            f"{row['predicted_positive']} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {_fmt_metric(row['precision'])} | "
            f"{_fmt_metric(row['recall_sensitivity'])} | {_fmt_metric(row['f1'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Coverage and failure signals",
            "",
            (
                f"- Families with a corresponding authoritative ecgfeat output cover "
                f"{supported_mapped_gt_label_instances}/{all_gt_label_instances} "
                f"({_fmt_metric(summary['supported_mapped_ground_truth_label_fraction'])}) "
                "LUDB label instances. The extended audit taxonomy maps "
                f"{mapped_gt_label_instances}/{all_gt_label_instances} "
                f"({_fmt_metric(summary['mapped_ground_truth_label_fraction'])}); "
                "the difference is unsupported output coverage."
            ),
            (
                f"- All {len(predicted_abnormal)}/{len(records)} records received an "
                "overall status with findings. On the 23-record normal-like surrogate, "
                f"specificity was {_fmt_metric(triage['specificity'])}."
            ),
            (
                "- `low_qrs_voltage_precordial_leads` was emitted for "
                f"{prediction_code_counts.get('low_qrs_voltage_precordial_leads', 0)}/"
                f"{len(records)} records. This systematic output dominates abnormal triage "
                "and requires calibration/audit."
            ),
            (
                "- Important unsupported LUDB families include sinus-rhythm confirmation, "
                "axis labels, atrial enlargement/overload, ventricular pacing, RVH, and "
                "early repolarization in authoritative final statements."
            ),
            "",
            "## Measurement context",
            "",
            "These figures compare ecgfeat measurements with values reconstructed from LUDB expert wave annotations. They do not prove diagnostic accuracy.",
            "",
            "| Measurement | Comparable | Bias | MAE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in measurement_rows:
        report_lines.append(
            f"| {row['measurement']} | {row['comparable']} | "
            f"{_fmt_metric(row['bias'], 2)} | {_fmt_metric(row['mae'], 2)} |"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- AF is the strongest major rhythm result in this snapshot, but still has both false positives and false negatives.",
            "- Bradycardia sensitivity is high, but precision is low because the rule is rate-based rather than specific to sinus bradycardia.",
            "- PVC precision is high in this small sample, but sensitivity is limited.",
            "- Bundle-branch block, IVCD, LVH, repolarization, ischemia, and scar families have insufficient sensitivity and/or precision for dependable diagnosis.",
            "- High measurement accuracy for HR/intervals cannot be promoted to clinical diagnostic validity without label-level performance and external validation.",
            "",
            "## Files",
            "",
            "- `diagnostic_metrics.csv`: per-family confusion matrices and mapping notes.",
            "- `broad_screening_metrics.csv`: permissive family-level screening results.",
            "- `record_diagnostic_results.csv`: per-record TP/FP/FN categories.",
            "- `predicted_statement_counts.csv`: all authoritative matched statement counts.",
            "- `unmapped_ground_truth_labels.csv`: LUDB labels outside the explicit family map.",
            "- `measurement_metrics.csv`: measurement MAE context.",
            "- `summary.json`: machine-readable aggregate results.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ecgfeat authoritative diagnoses against LUDB labels."
    )
    parser.add_argument("--ludb-dir", type=Path, default=DEFAULT_LUDB_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = evaluate(
        ludb_dir=args.ludb_dir.resolve(),
        feature_dir=args.feature_dir.resolve(),
        output_dir=args.out_dir.resolve(),
    )
    metrics = summary["supported_category_metrics"]
    print(f"Evaluated {summary['record_count']} LUDB records.")
    print(
        "Supported families: "
        f"micro precision={_fmt_metric(metrics['micro_precision'])}, "
        f"recall={_fmt_metric(metrics['micro_recall_sensitivity'])}, "
        f"F1={_fmt_metric(metrics['micro_f1'])}."
    )
    print(f"Saved: {args.out_dir.resolve() / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

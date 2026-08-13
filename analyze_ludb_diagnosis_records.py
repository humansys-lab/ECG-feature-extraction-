#!/usr/bin/env python3
"""Create a Chinese record-by-record analysis of ecgfeat diagnoses on LUDB."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from evaluate_ludb_diagnosis import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_LUDB_DIR,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_CATEGORIES,
    UNSUPPORTED_CATEGORIES,
    _load_prediction,
    _parse_ludb_labels,
)


PREDICTION_CODE_ZH = {
    "acute_occlusion_pattern": "急性缺血/冠脉闭塞提示",
    "atrial_fibrillation_flutter_indeterminate": "房颤/房扑未定",
    "atrial_fibrillation_pattern": "房颤模式",
    "atrial_flutter_pattern": "房扑模式",
    "borderline_short_qt": "临界短 QT",
    "bradycardia": "心动过缓",
    "first_degree_av_delay": "一度房室传导延迟",
    "lafb_pattern": "左前分支阻滞模式",
    "lbbb_pattern": "左束支阻滞模式",
    "low_qrs_voltage_limb_leads": "肢体导联 QRS 低电压",
    "low_qrs_voltage_precordial_leads": "胸前导联 QRS 低电压*",
    "lpfb_pattern": "左后分支阻滞模式",
    "lvh_voltage_criteria": "左室肥厚电压标准",
    "nonspecific_ivcd": "非特异性室内传导延迟",
    "possible_precordial_lead_reversal": "可能胸前导联错接",
    "possible_short_qt_pattern": "可能短 QT 模式",
    "posterior_ischemia_screen": "后壁缺血筛查阳性",
    "premature_atrial_complexes": "房性早搏",
    "premature_ventricular_complexes": "室性早搏",
    "primary_t_wave_abnormality": "原发性 T 波异常",
    "prior_infarct_q_wave_pattern": "既往/年龄不确定性梗死 Q 波模式",
    "probable_lbbb_pattern": "可能左束支阻滞",
    "probable_rbbb_pattern": "可能右束支阻滞",
    "prolonged_qt": "QT 延长",
    "rbbb_pattern": "右束支阻滞模式",
    "second_degree_av_block_pattern": "二度房室传导阻滞模式",
    "secondary_t_wave_abnormality": "继发性 T 波异常",
    "sgarbossa_positive": "Sgarbossa 标准阳性",
    "tachycardia": "心动过速",
    "technically_limited": "技术质量受限",
    "ventricular_preexcitation_pattern": "心室预激模式",
    "wide_qrs_repolarization_review": "宽 QRS 下复极间期需复核",
}


CATEGORY_ZH = {
    "sinus_bradycardia": "窦性心动过缓",
    "sinus_tachycardia": "窦性心动过速",
    "atrial_fibrillation": "心房颤动",
    "atrial_flutter": "心房扑动",
    "premature_atrial_complexes": "房性早搏",
    "premature_ventricular_complexes": "室性早搏",
    "first_degree_av_block": "一度房室传导阻滞",
    "complete_av_block": "三度/完全性房室传导阻滞",
    "right_bundle_branch_block": "右束支阻滞",
    "left_bundle_branch_block": "左束支阻滞",
    "left_anterior_fascicular_block": "左前分支阻滞",
    "nonspecific_ivcd": "非特异性室内传导延迟",
    "left_ventricular_hypertrophy": "左室肥厚/负荷",
    "nonspecific_repolarization_abnormality": "非特异性复极异常",
    "ischemia_or_stemi": "缺血/STEMI",
    "scar_or_indeterminate_infarction": "瘢痕/未定性梗死",
    "sinus_rhythm": "窦性心律",
    "other_sinus_dysrhythmia": "窦性心律不齐/不规则窦律",
    "normal_or_positional_axis": "正常/垂直/水平电轴",
    "left_axis_deviation": "电轴左偏",
    "right_axis_deviation": "电轴右偏",
    "atrial_enlargement_or_overload": "心房肥厚/负荷",
    "right_ventricular_hypertrophy": "右室肥厚",
    "ventricular_pacing": "心室起搏",
    "early_repolarization": "早期复极",
}


EXACT_LABEL_ZH = {
    "Rhythm: Sinus rhythm": "窦性心律",
    "Rhythm: Sinus bradycardia": "窦性心动过缓",
    "Rhythm: Sinus tachycardia": "窦性心动过速",
    "Rhythm: Sinus arrhythmia": "窦性心律不齐",
    "Rhythm: Irregular sinus rhythm": "不规则窦性心律",
    "Rhythm: Atrial fibrillation": "心房颤动",
    "Rhythm: Atrial flutter, typical": "典型心房扑动",
    "Electric axis of the heart: normal": "心电轴正常",
    "Electric axis of the heart: left axis deviation": "心电轴左偏",
    "Electric axis of the heart: right axis deviation": "心电轴右偏",
    "Electric axis of the heart: vertical": "垂直电轴",
    "Electric axis of the heart: horizontal": "水平电轴",
    "Incomplete right bundle branch block": "不完全性右束支阻滞",
    "Complete right bundle branch block": "完全性右束支阻滞",
    "Incomplete left bundle branch block": "不完全性左束支阻滞",
    "Complete left bundle branch block": "完全性左束支阻滞",
    "Left anterior hemiblock": "左前分支阻滞",
    "Non-specific intravintricular conduction delay": "非特异性室内传导延迟",
    "Aberrant conduction": "差异性传导",
    "I degree AV block": "一度房室传导阻滞",
    "III degree AV-block": "三度房室传导阻滞",
    "Sinoatrial blockade, undefined": "未定型窦房阻滞",
    "Left ventricular hypertrophy": "左室肥厚",
    "Right ventricular hypertrophy": "右室肥厚",
    "Left ventricular overload": "左室负荷",
    "Left atrial hypertrophy": "左房肥厚",
    "Right atrial hypertrophy": "右房肥厚",
    "Left atrial overload": "左房负荷",
    "Right atrial overload": "右房负荷",
    "Early repolarization syndrome": "早期复极综合征",
    "Wandering atrial pacemaker": "游走性房性起搏点",
    "P-synchrony": "P 波同步",
    "UNIpolar ventricular pacing": "单极心室起搏",
    "BIpolar ventricular pacing": "双极心室起搏",
    "Biventricular pacing": "双心室起搏",
    "UNIpolar atrial pacing": "单极心房起搏",
}


TERRITORY_ZH = {
    "anterior wall": "前壁",
    "lateral wall": "侧壁",
    "inferior wall": "下壁",
    "posterior wall": "后壁",
    "septal": "间隔",
    "apical": "心尖",
}


HIGH_IMPACT_CATEGORIES = {
    "atrial_fibrillation",
    "atrial_flutter",
    "complete_av_block",
    "right_bundle_branch_block",
    "left_bundle_branch_block",
    "nonspecific_ivcd",
    "ischemia_or_stemi",
    "scar_or_indeterminate_infarction",
}


SYSTEMATIC_UNVERIFIED_CODES = {"low_qrs_voltage_precordial_leads"}


def _translate_label(label: str) -> str:
    if label in EXACT_LABEL_ZH:
        return EXACT_LABEL_ZH[label]

    territory = None
    if ":" in label:
        territory = TERRITORY_ZH.get(label.rsplit(":", 1)[1].strip())

    if label.startswith(("Non-specific repolarization", "Non- specific repolarization")):
        return f"非特异性复极异常（{territory or label.rsplit(':', 1)[-1]}）"
    if label.startswith("Undefined ischemia/scar/supp.NSTEMI:"):
        return f"未定性缺血/瘢痕/疑似 NSTEMI（{territory or label.rsplit(':', 1)[-1]}）"
    if label.startswith("Scar formation:"):
        return f"瘢痕形成（{territory or label.rsplit(':', 1)[-1]}）"
    if label.startswith("Ischemia:"):
        return f"缺血（{territory or label.rsplit(':', 1)[-1]}）"
    if label.startswith("STEMI:"):
        return f"STEMI（{territory or label.rsplit(':', 1)[-1]}）"
    if label.startswith("Atrial extrasystole"):
        subtype = label.split(":", 1)[1].strip() if ":" in label else label
        return f"房性早搏（{subtype}）"
    if label.startswith("Ventricular extrasystole"):
        subtype = label.split(":", 1)[1].strip() if ":" in label else label
        return f"室性早搏（{subtype}）"
    return label


def _join_zh(keys: Iterable[str]) -> str:
    values = [CATEGORY_ZH.get(key, key) for key in sorted(keys)]
    return "、".join(values) if values else "无"


def _join_predictions(codes: Iterable[str]) -> str:
    values = [
        PREDICTION_CODE_ZH.get(code, code)
        for code in sorted(codes, key=lambda code: PREDICTION_CODE_ZH.get(code, code))
    ]
    return "、".join(values) if values else "无权威阳性语句"


def _format_value(token: object, digits: int = 1) -> str:
    if token in (None, ""):
        return "N/A"
    return f"{float(token):.{digits}f}"


def _metric_pair(
    row: dict[str, str],
    algorithm_field: str,
    reference_field: str,
    diff_field: str,
    unit: str,
) -> str:
    algorithm = _format_value(row.get(algorithm_field))
    reference = _format_value(row.get(reference_field))
    difference = row.get(diff_field)
    diff_text = "N/A" if difference in (None, "") else f"{float(difference):+.1f}"
    return f"{algorithm}/{reference} {unit}（Δ{diff_text}）"


def _measurement_text(row: dict[str, str]) -> str:
    return "；".join(
        (
            "HR "
            + _metric_pair(
                row, "algorithm_hr", "ground_truth_hr", "diff_hr", "bpm"
            ),
            "PR "
            + _metric_pair(
                row, "algorithm_pr", "ground_truth_pr", "diff_pr", "ms"
            ),
            "QRS "
            + _metric_pair(
                row, "algorithm_qrs", "ground_truth_qrs", "diff_qrs", "ms"
            ),
            "QT "
            + _metric_pair(
                row, "algorithm_qt", "ground_truth_qt", "diff_qt", "ms"
            ),
            "QRS轴 "
            + _metric_pair(
                row,
                "algorithm_qrs_axis",
                "ground_truth_qrs_axis",
                "diff_qrs_axis",
                "°",
            ),
        )
    )


def _large_measurement_errors(row: dict[str, str]) -> list[str]:
    checks = (
        ("HR", "diff_hr", 5.0, "bpm"),
        ("PR", "diff_pr", 20.0, "ms"),
        ("QRS", "diff_qrs", 15.0, "ms"),
        ("QT", "diff_qt", 20.0, "ms"),
        ("QRS 电轴", "abs_qrs_axis_circular_diff", 20.0, "°"),
    )
    errors: list[str] = []
    for name, field, threshold, unit in checks:
        token = row.get(field)
        if token in (None, ""):
            continue
        value = float(token)
        if abs(value) > threshold:
            errors.append(f"{name} 偏差较大（{value:+.1f}{unit}）")
    return errors


def _verdict(
    gt_categories: set[str],
    predicted_categories: set[str],
    true_positive: set[str],
    false_positive: set[str],
    false_negative: set[str],
) -> str:
    if gt_categories:
        if not false_negative and not false_positive:
            return "支持范围内一致"
        if not true_positive:
            return "未识别参考异常"
        if false_negative.intersection(HIGH_IMPACT_CATEGORIES):
            return "部分正确，但有重要漏诊"
        if len(true_positive) >= len(false_negative):
            return "部分正确"
        return "部分正确，但漏诊较多"
    if predicted_categories:
        return "无参考阳性但存在类别误报"
    return "支持范围内无可比阳性"


def _analysis_clues(
    *,
    gt_categories: set[str],
    predicted_categories: set[str],
    true_positive: set[str],
    false_positive: set[str],
    false_negative: set[str],
    unsupported_gt: set[str],
    codes: set[str],
    measurement_row: dict[str, str],
    unavailable_domains: Sequence[str],
) -> list[str]:
    clues: list[str] = []
    if true_positive:
        clues.append(f"正确识别 {_join_zh(true_positive)}")

    if false_negative:
        clues.append(f"漏掉 {_join_zh(false_negative)}")
    if false_positive:
        clues.append(f"额外报出 {_join_zh(false_positive)}")

    if "left_ventricular_hypertrophy" in false_negative:
        clues.append("参考含左室肥厚/负荷，但 LVH 电压规则未触发")
    if "complete_av_block" in false_negative and "second_degree_av_block_pattern" in codes:
        clues.append("参考为三度房室阻滞，而算法报二度模式，存在分级错误")
    if "atrial_fibrillation" in false_negative:
        if "atrial_fibrillation_flutter_indeterminate" in codes:
            clues.append("房颤仅被判为房颤/房扑未定")
        else:
            clues.append("房颤规则未触发")
    if "atrial_flutter" in false_negative:
        if "atrial_fibrillation_pattern" in codes:
            clues.append("房扑被判成房颤")
        elif "atrial_fibrillation_flutter_indeterminate" in codes:
            clues.append("房扑仅被判为房颤/房扑未定")
        else:
            clues.append("房扑规则未触发")
    if false_negative.intersection(
        {"right_bundle_branch_block", "left_bundle_branch_block", "nonspecific_ivcd"}
    ):
        if predicted_categories.intersection(
            {"right_bundle_branch_block", "left_bundle_branch_block", "nonspecific_ivcd"}
        ):
            clues.append("存在传导异常输出，但束支/IVCD 亚型与参考不一致")
        else:
            clues.append("参考传导异常未触发相应权威语句")
    if false_negative.intersection(
        {
            "nonspecific_repolarization_abnormality",
            "ischemia_or_stemi",
            "scar_or_indeterminate_infarction",
        }
    ):
        if codes.intersection(
            {
                "primary_t_wave_abnormality",
                "secondary_t_wave_abnormality",
                "acute_occlusion_pattern",
                "sgarbossa_positive",
                "posterior_ischemia_screen",
                "prior_infarct_q_wave_pattern",
            }
        ):
            clues.append("有广义 ST-T/Q 波异常输出，但临床家族或参考亚型未对齐")
        else:
            clues.append("参考 ST-T/缺血/瘢痕异常未触发相应权威语句")

    if "sinus_bradycardia" in false_positive:
        hr_token = measurement_row.get("algorithm_hr")
        if hr_token not in (None, ""):
            clues.append(
                f"算法 HR={float(hr_token):.1f} bpm 触发心动过缓阈值，但 LUDB 未标窦缓"
            )

    if unsupported_gt:
        clues.append(
            f"另有权威最终语句未直接覆盖的参考家族：{_join_zh(unsupported_gt)}"
        )
    if unavailable_domains:
        clues.append(
            "规则覆盖受限领域：" + "、".join(sorted(str(item) for item in unavailable_domains))
        )

    clues.extend(_large_measurement_errors(measurement_row))
    if SYSTEMATIC_UNVERIFIED_CODES.intersection(codes):
        clues.append("胸前低电压为本批次 200/200 条共有输出，不能视为本条已独立验证命中")
    return clues or ["支持范围内未见可比较的阳性家族"]


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_records(
    ludb_dir: Path,
    feature_dir: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = feature_dir / "summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        measurement_by_id = {
            row["record_id"]: row for row in csv.DictReader(handle)
        }

    headers = sorted(
        ludb_dir.glob("*.hea"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    supported_code_union = frozenset(
        code for spec in SUPPORTED_CATEGORIES for code in spec.prediction_codes
    )
    for header_path in headers:
        record_id = header_path.stem
        labels = _parse_ludb_labels(header_path)
        feature_path = feature_dir / record_id / f"{record_id}_features.json"
        codes, overall_status = _load_prediction(feature_path)
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
        clinical = payload.get("clinical_interpretation") or {}
        unavailable_domains = clinical.get("unavailable_domains") or []

        gt_supported = {
            spec.key for spec in SUPPORTED_CATEGORIES if spec.gt_match(labels)
        }
        gt_unsupported = {
            spec.key for spec in UNSUPPORTED_CATEGORIES if spec.gt_match(labels)
        }
        predicted_supported = {
            spec.key
            for spec in SUPPORTED_CATEGORIES
            if spec.prediction_codes.intersection(codes)
        }
        true_positive = gt_supported.intersection(predicted_supported)
        false_positive = predicted_supported - gt_supported
        false_negative = gt_supported - predicted_supported
        other_codes = codes - supported_code_union
        measurement_row = measurement_by_id.get(record_id, {})
        verdict = _verdict(
            gt_supported,
            predicted_supported,
            true_positive,
            false_positive,
            false_negative,
        )
        clues = _analysis_clues(
            gt_categories=gt_supported,
            predicted_categories=predicted_supported,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            unsupported_gt=gt_unsupported,
            codes=codes,
            measurement_row=measurement_row,
            unavailable_domains=unavailable_domains,
        )

        rows.append(
            {
                "record_id": record_id,
                "verdict": verdict,
                "reference_diagnoses_zh": "；".join(
                    _translate_label(label) for label in labels
                ),
                "ecgfeat_statements_zh": _join_predictions(codes),
                "matched_categories": _join_zh(true_positive),
                "missed_categories": _join_zh(false_negative),
                "extra_categories": _join_zh(false_positive),
                "unsupported_reference_categories": _join_zh(gt_unsupported),
                "other_unaligned_statement_codes": " | ".join(sorted(other_codes)),
                "overall_status": overall_status,
                "record_grade": measurement_row.get("record_grade", ""),
                "pacing_state": measurement_row.get("pacing_state", ""),
                "analysis": "；".join(clues),
            }
        )
        details.append(
            {
                "record_id": record_id,
                "verdict": verdict,
                "labels": labels,
                "codes": codes,
                "gt_supported": gt_supported,
                "gt_unsupported": gt_unsupported,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "other_codes": other_codes,
                "overall_status": overall_status,
                "record_grade": measurement_row.get("record_grade", ""),
                "pacing_state": measurement_row.get("pacing_state", ""),
                "measurement_text": _measurement_text(measurement_row),
                "clues": clues,
            }
        )

    _write_csv(rows, output_dir / "record_by_record_analysis.csv")

    verdict_counts = Counter(str(detail["verdict"]) for detail in details)
    lines = [
        "# LUDB 200 条记录：ecgfeat 逐条诊断分析",
        "",
        "## 阅读说明",
        "",
        "- 参考诊断来自 LUDB `.hea` 文件中的 `<diagnoses>`。",
        "- 算法诊断只取 `clinical_interpretation.final_statements` 中状态为 `matched` 的权威语句。",
        "- 报告原有的 `Dx Codes / Dx Detail / Hx` 直接包含 LUDB 标签，未用于比较。",
        "- “支持范围”是指已建立明确 LUDB↔ecgfeat 映射的 16 个诊断家族。",
        "- `胸前导联 QRS 低电压*` 在 200/200 条记录中均出现，是系统性待审计输出，不能当作逐条正确命中。",
        "- 测量格式为 `算法值/专家标注重建值（差值）`。",
        "",
        "## 逐条结论分布",
        "",
        "| 结论 | 记录数 |",
        "| --- | ---: |",
    ]
    for verdict, count in verdict_counts.most_common():
        lines.append(f"| {verdict} | {count} |")

    lines.extend(
        [
            "",
            "## 记录索引",
            "",
            "| 记录 | 结论 | 命中 | 漏诊 | 误报 |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for detail in details:
        lines.append(
            f"| [{detail['record_id']}](#record-{detail['record_id']}) | "
            f"{detail['verdict']} | {_join_zh(detail['true_positive'])} | "
            f"{_join_zh(detail['false_negative'])} | "
            f"{_join_zh(detail['false_positive'])} |"
        )

    lines.extend(["", "## 逐记录分析", ""])
    for detail in details:
        record_id = str(detail["record_id"])
        lines.extend(
            [
                f'<a id="record-{record_id}"></a>',
                f"### Record {record_id} — {detail['verdict']}",
                "",
                "- LUDB 参考："
                + "；".join(_translate_label(label) for label in detail["labels"]),
                "- ecgfeat 输出：" + _join_predictions(detail["codes"]),
                (
                    "- 对齐结果：命中 "
                    + _join_zh(detail["true_positive"])
                    + "；漏诊 "
                    + _join_zh(detail["false_negative"])
                    + "；误报 "
                    + _join_zh(detail["false_positive"])
                ),
                "- 未直接覆盖的参考家族：" + _join_zh(detail["gt_unsupported"]),
                (
                    "- 测量："
                    + detail["measurement_text"]
                    + f"；质量={detail['record_grade'] or 'N/A'}"
                    + f"；起搏状态={detail['pacing_state'] or 'N/A'}"
                ),
                "- 分析：" + "；".join(detail["clues"]) + "。",
                "",
            ]
        )

    (output_dir / "RECORD_BY_RECORD.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (output_dir / "record_verdict_counts.json").write_text(
        json.dumps(
            {
                "record_count": len(details),
                "verdict_counts": dict(verdict_counts),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chinese record-by-record LUDB diagnosis analysis."
    )
    parser.add_argument("--ludb-dir", type=Path, default=DEFAULT_LUDB_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows = analyze_records(
        ludb_dir=args.ludb_dir.resolve(),
        feature_dir=args.feature_dir.resolve(),
        output_dir=args.out_dir.resolve(),
    )
    print(f"Analyzed {len(rows)} records.")
    print(f"Saved: {args.out_dir.resolve() / 'RECORD_BY_RECORD.md'}")


if __name__ == "__main__":
    main()

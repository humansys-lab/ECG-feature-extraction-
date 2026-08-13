#!/usr/bin/env python3
"""Explain, record by record, why ecgfeat disagrees with LUDB diagnoses.

The output separates:

1. dataset-wide limitations (especially LUDB's per-lead amplitude scaling);
2. direct rule-engine states (not matched, unavailable, not applicable,
   suppressed);
3. upstream measurement disagreements;
4. diagnosis-vocabulary and authoritative-output coverage gaps.

The explanations are evidence-backed descriptions of the current rule
execution.  They are not retrospective clinical adjudications.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import wfdb

from analyze_ludb_diagnosis_records import (
    CATEGORY_ZH,
    PREDICTION_CODE_ZH,
    SYSTEMATIC_UNVERIFIED_CODES,
    _join_predictions,
    _join_zh,
    _large_measurement_errors,
    _translate_label,
    _verdict,
)
from evaluate_ludb_diagnosis import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_LUDB_DIR,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_CATEGORIES,
    UNSUPPORTED_CATEGORIES,
    _load_prediction,
    _parse_ludb_labels,
)


CATEGORY_RULE_CODES = {
    "atrial_fibrillation": {"atrial_fibrillation_pattern"},
    "atrial_flutter": {"atrial_flutter_pattern"},
    "premature_atrial_complexes": {"premature_atrial_complexes"},
    "premature_ventricular_complexes": {"premature_ventricular_complexes"},
    "complete_av_block": {"complete_av_block_pattern"},
    "right_bundle_branch_block": {"rbbb_pattern"},
    "left_bundle_branch_block": {"lbbb_pattern"},
    "left_anterior_fascicular_block": {"lafb_pattern"},
    "nonspecific_ivcd": {"nonspecific_ivcd"},
    "left_ventricular_hypertrophy": {"lvh_voltage_criteria"},
    "nonspecific_repolarization_abnormality": {
        "primary_t_wave_abnormality",
        "secondary_t_wave_abnormality",
    },
    "ischemia_or_stemi": {
        "acute_occlusion_pattern",
        "posterior_ischemia_screen",
        "sgarbossa_positive",
    },
    "scar_or_indeterminate_infarction": {"prior_infarct_q_wave_pattern"},
}


AMPLITUDE_DEPENDENT_CATEGORIES = {
    "right_bundle_branch_block",
    "left_bundle_branch_block",
    "left_anterior_fascicular_block",
    "nonspecific_ivcd",
    "left_ventricular_hypertrophy",
    "nonspecific_repolarization_abnormality",
    "ischemia_or_stemi",
    "scar_or_indeterminate_infarction",
}


CAUSE_TAG_ZH = {
    "amplitude_scaling": "LUDB 幅值归一化限制",
    "threshold_not_met": "规则阈值/形态未满足",
    "missing_input": "规则输入缺失",
    "not_applicable": "规则被判定不适用",
    "suppressed": "规则结果被抑制",
    "rhythm_confusion": "节律/异位搏动混淆",
    "subtype_confusion": "诊断亚型或分级错误",
    "measurement_error": "上游测量偏差",
    "output_gap": "权威输出覆盖缺口",
    "ontology_mismatch": "标签与规则语义不完全一致",
    "systematic_false_output": "系统性可疑输出",
}


STATUS_ZH = {
    "matched": "已触发",
    "not_matched": "未满足",
    "unavailable": "不可用",
    "indeterminate": "不确定",
    "not_applicable": "不适用",
    "suppressed": "被抑制",
}


MISSING_INPUT_ZH = {
    "global.pr_ms": "全局 PR 间期",
    "global.qrs_ms": "全局 QRS 时限",
    "global.qrs_axis_deg": "QRS 电轴",
    "rhythm.multilead_F_wave_morphology": "多导联 F 波形态",
    "rhythm.validated_qrst_subtraction": "通过验证的 QRST 消减",
    "rhythm.af_afl_indeterminate": "房颤/房扑判别确定性",
    "stable_p_qrs_association_for_av_block_classification": "稳定 P-QRS 关系",
    "nonpaced_rhythm_interpretation_available": "非起搏节律可解释性",
    "complete_bbb_morphology_exclusion": "完整束支阻滞形态排除证据",
    "two_reliable_contiguous_st_leads": "至少两个可靠相邻 ST 导联",
    "reliable_t_measurements_across_anterior_inferior_lateral_territories": "前/下/侧壁可靠 T 波测量",
    "pediatric_t_wave_reference_table": "儿童 T 波参考表",
    "pediatric_lvh_percentile_table": "儿童 LVH 百分位参考表",
    "pediatric_rvh_percentile_table": "儿童 RVH 百分位参考表",
    "pediatric_atrial_percentile_table": "儿童心房参考表",
}


FIX_BY_TAG = {
    "amplitude_scaling": "不要在当前 LUDB 物理幅值上验证电压诊断；恢复原始校准幅值，或在 LUDB 评估时关闭电压/ST/Q/T 幅值规则。",
    "threshold_not_met": "逐规则核对阈值与 LUDB 标签定义，并用专家标注病例校准形态阈值。",
    "missing_input": "先提高相应 P/QRS/T 形态字段的可用率，再开放该诊断规则。",
    "not_applicable": "调整 AF/起搏等互斥门控，保留可并存诊断的独立证据通道。",
    "suppressed": "审计抑制优先级，确认抑制项是否真的足以否定目标诊断。",
    "rhythm_confusion": "强化有序 P 波、PAC/PVC 与 AF/AFL 的联合时序及形态判别。",
    "subtype_confusion": "先识别广义异常，再用独立证据判定束支侧别或房室阻滞分级。",
    "measurement_error": "回查该记录的波界、代表心搏和多导联共识，修正上游测量。",
    "output_gap": "将已有测量/形态结果投影为可审计的权威诊断语句，或明确标记为不支持。",
    "ontology_mismatch": "建立经专家确认的 LUDB 标签到规则语义映射，分开评估筛查、形态和临床诊断。",
    "systematic_false_output": "增加数据集级校准审计和异常分布门控，禁止单一规则在 100% 记录上无告警触发。",
}


def _iter_rules(payload: dict) -> list[dict]:
    clinical = payload.get("clinical_interpretation") or {}
    rows: list[dict] = []
    for domain, domain_rows in (clinical.get("domains") or {}).items():
        for rule in domain_rows or []:
            copied = dict(rule)
            copied["_domain_key"] = domain
            rows.append(copied)
    return rows


def _evaluates_code(rule: dict) -> str | None:
    evidence = rule.get("evidence") or {}
    return evidence.get("evaluates_code") or rule.get("statement_code")


def _rules_for_category(category: str, rules: Sequence[dict]) -> list[dict]:
    if category in {"sinus_bradycardia", "sinus_tachycardia"}:
        return [rule for rule in rules if rule.get("rule_id") == "CLIN-RHYTHM-RATE-01"]
    if category == "first_degree_av_block":
        return [rule for rule in rules if rule.get("rule_id") == "CLIN-INTERVAL-PR-01"]
    codes = CATEGORY_RULE_CODES.get(category, set())
    return [rule for rule in rules if _evaluates_code(rule) in codes]


def _format_missing(inputs: Iterable[str]) -> str:
    values = []
    for item in inputs:
        text = str(item)
        values.append(MISSING_INPUT_ZH.get(text, text))
    return "、".join(values) if values else "未记录具体缺失项"


def _rule_state_summary(rules: Sequence[dict]) -> str:
    if not rules:
        return "未找到对应权威规则"
    parts = []
    for rule in rules:
        state = STATUS_ZH.get(str(rule.get("status")), str(rule.get("status")))
        missing = rule.get("missing_inputs") or []
        suppressors = rule.get("suppressed_by") or []
        suffix = ""
        if missing:
            suffix += f"，缺少 {_format_missing(missing)}"
        if suppressors:
            suffix += "，被 " + "、".join(str(item) for item in suppressors) + " 抑制"
        parts.append(f"{rule.get('rule_id')}={state}{suffix}")
    return "；".join(parts)


def _float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _compact_pairs(rows: Sequence[Sequence[object]]) -> str:
    return "、".join("-".join(str(item) for item in row) for row in rows) or "无"


def _specific_miss_reason(
    category: str,
    rules: Sequence[dict],
    measurement: dict[str, str],
    labels: Sequence[str],
) -> tuple[str, set[str]]:
    tags: set[str] = set()
    statuses = {str(rule.get("status")) for rule in rules}
    if statuses.intersection({"unavailable", "indeterminate"}):
        tags.add("missing_input")
    if "not_applicable" in statuses:
        tags.add("not_applicable")
    if "suppressed" in statuses:
        tags.add("suppressed")
    if statuses and statuses.issubset({"not_matched"}):
        tags.add("threshold_not_met")

    if category in AMPLITUDE_DEPENDENT_CATEGORIES:
        tags.add("amplitude_scaling")

    rule = rules[0] if rules else {}
    evidence = rule.get("evidence") or {}

    if category == "sinus_bradycardia":
        heart_rate = _float(evidence.get("heart_rate_bpm"))
        tags.add("threshold_not_met")
        return (
            f"心率规则要求 HR<60 bpm；当前算法 HR={heart_rate:.1f} bpm，未达到阈值。"
            if heart_rate is not None
            else "心率输入不可用，窦缓规则无法触发。"
        ), tags

    if category == "sinus_tachycardia":
        heart_rate = _float(evidence.get("heart_rate_bpm"))
        tags.add("threshold_not_met")
        return (
            f"心率规则要求 HR>100 bpm；当前算法 HR={heart_rate:.1f} bpm，未达到阈值。"
            if heart_rate is not None
            else "心率输入不可用，心动过速规则无法触发。"
        ), tags

    if category == "atrial_fibrillation":
        rr_cv = _float(evidence.get("rr_cv"))
        organized = _float(evidence.get("organized_p_ratio"))
        tags.add("rhythm_confusion")
        values = []
        if rr_cv is not None:
            values.append(f"RR-CV={rr_cv:.3f}")
        if organized is not None:
            values.append(f"有序 P 支持={organized:.3f}")
        return (
            "房颤规则未形成“明显 RR 不规则且缺少有序心房活动”的组合证据"
            + (f"（{', '.join(values)}）" if values else "")
            + f"；规则状态：{_rule_state_summary(rules)}。"
        ), tags

    if category == "atrial_flutter":
        tags.update({"rhythm_confusion", "missing_input"})
        missing = sorted(
            {
                str(item)
                for flutter_rule in rules
                for item in (flutter_rule.get("missing_inputs") or [])
            }
        )
        return (
            "房扑需要通过验证的 QRST 消减及多导联 F 波形态；当前"
            + (_format_missing(missing) if missing else "F 波证据未达到规则要求")
            + "，因此未能确认房扑。"
        ), tags

    if category in {
        "premature_atrial_complexes",
        "premature_ventricular_complexes",
    }:
        tags.add("rhythm_confusion")
        candidates = evidence.get("candidate_beats") or []
        analyzable = evidence.get("analyzable_beats")
        not_applicable_by = evidence.get("not_applicable_by")
        if not_applicable_by:
            return (
                f"异位搏动规则因 {not_applicable_by} 被判不适用；"
                f"可分析心搏={analyzable}，候选={len(candidates)}。"
            ), tags
        return (
            f"异位搏动规则在 {analyzable if analyzable is not None else 'N/A'} 个可分析心搏中"
            f"仅找到 {len(candidates)} 个满足 RR 提前、QRS 宽度/形态及 P 波置信度条件的候选。"
        ), tags

    if category == "first_degree_av_block":
        pr_ms = _float(evidence.get("pr_ms"))
        if pr_ms is None:
            tags.add("missing_input")
            return "全局 PR 不可用，无法执行 PR>200 ms 的一度房室传导延迟规则。", tags
        tags.add("threshold_not_met")
        return f"算法 PR={pr_ms:.1f} ms，未超过规则的 200 ms 阈值。", tags

    if category == "complete_av_block":
        tags.update({"rhythm_confusion", "subtype_confusion"})
        not_applicable = evidence.get("not_applicable_by")
        if not_applicable:
            return f"完全性房室阻滞规则因 {not_applicable} 被判不适用。", tags
        return (
            "规则未同时得到 complete_av_block 与房室分离/心房率快于心室率证据"
            f"（complete={evidence.get('complete_av_block')}，"
            f"AV dissociation={evidence.get('av_dissociation')}，"
            f"atrial faster={evidence.get('atrial_faster_than_ventricular')}）。"
        ), tags

    if category == "right_bundle_branch_block":
        qrs_ms = _float(evidence.get("qrs_ms"))
        if qrs_ms is not None and qrs_ms < 110:
            return f"算法 QRS={qrs_ms:.1f} ms，低于 RBBB 规则 110 ms 的入口阈值。", tags
        return (
            f"QRS={qrs_ms:.1f} ms；V1 R′及 I/V6 终末 S 的幅度/时限组合未满足，"
            f"或输入缺失。规则状态：{_rule_state_summary(rules)}。"
            if qrs_ms is not None
            else f"RBBB 形态输入不足；{_rule_state_summary(rules)}。"
        ), tags

    if category == "left_bundle_branch_block":
        qrs_ms = _float(evidence.get("qrs_ms"))
        if qrs_ms is not None and qrs_ms < 120:
            return f"算法 QRS={qrs_ms:.1f} ms，低于 LBBB 规则 120 ms 阈值。", tags
        return (
            f"QRS={qrs_ms:.1f} ms，但 V1 小 R 与侧壁无 Q/宽 R 组合未满足，"
            f"或侧壁 R 时限缺失。规则状态：{_rule_state_summary(rules)}。"
            if qrs_ms is not None
            else f"LBBB 时限/形态输入不足；{_rule_state_summary(rules)}。"
        ), tags

    if category == "left_anterior_fascicular_block":
        axis = _float(evidence.get("qrs_axis_deg"))
        if axis is not None and not -90 <= axis <= -45:
            return f"算法 QRS 轴={axis:.1f}°，不在 LAFB 入口范围 -90°至-45°。", tags
        return (
            f"QRS 轴={axis:.1f}°，但 aVL qR 与 II/III/aVF rS 组合未完整满足。"
            if axis is not None
            else f"QRS 电轴或分支形态输入不足；{_rule_state_summary(rules)}。"
        ), tags

    if category == "nonspecific_ivcd":
        qrs_ms = _float(evidence.get("qrs_ms"))
        if qrs_ms is not None and qrs_ms <= 110:
            return f"算法 QRS={qrs_ms:.1f} ms，不满足 IVCD 的 >110 ms 条件。", tags
        return (
            f"QRS={qrs_ms:.1f} ms，但完整 BBB 排除证据缺失或规则转而匹配了束支阻滞。"
            if qrs_ms is not None
            else f"QRS/BBB 排除证据不足；{_rule_state_summary(rules)}。"
        ), tags

    if category == "left_ventricular_hypertrophy":
        matched = evidence.get("matched_criteria") or []
        assessed = evidence.get("assessed_criteria") or []
        cornell = _float(evidence.get("cornell_voltage_mv"))
        return (
            "LVH 规则虽评估了 "
            + ("、".join(str(item) for item in assessed) if assessed else "有限电压标准")
            + f"，但命中标准={matched or '无'}"
            + (f"，Cornell={cornell:.3f} mV" if cornell is not None else "")
            + "。LUDB 每导联独立幅值归一化使 Cornell、Sokolow-Lyon 和 R-aVL 失去有效绝对量纲。"
        ), tags

    if category == "nonspecific_repolarization_abnormality":
        inverted = evidence.get("inverted_contiguous_pairs") or []
        tall = evidence.get("tall_contiguous_pairs") or []
        return (
            f"T 波规则没有找到满足阈值的连续倒置或高尖组合"
            f"（倒置对={_compact_pairs(inverted)}，高尖对={_compact_pairs(tall)}）；"
            "逐导联幅值归一化还会改变 T/QRS 相对幅值判据。"
        ), tags

    if category == "ischemia_or_stemi":
        acute = next(
            (item for item in rules if _evaluates_code(item) == "acute_occlusion_pattern"),
            {},
        )
        acute_ev = acute.get("evidence") or {}
        pairs = acute_ev.get("qualifying_contiguous_pairs") or []
        posterior = next(
            (item for item in rules if _evaluates_code(item) == "posterior_ischemia_screen"),
            {},
        )
        posterior_ev = posterior.get("evidence") or {}
        tags.add("ontology_mismatch")
        return (
            f"ecgfeat 的该家族是急性 ST 抬高/后壁筛查，而 LUDB 标签还包含更宽泛的“Ischemia”。"
            f"当前相邻 ST 抬高对={_compact_pairs(pairs)}，"
            f"后壁支持导联={posterior_ev.get('supported_leads') or '无'}；"
            "逐导联幅值归一化使 0.10/0.20 mV 等绝对 ST 阈值不可直接解释。"
        ), tags

    if category == "scar_or_indeterminate_infarction":
        territory_results = evidence.get("territory_results") or {}
        matched = evidence.get("matched_groups") or {}
        unresolved = {
            territory: row.get("unresolved_candidate_leads")
            for territory, row in territory_results.items()
            if row.get("unresolved_candidate_leads")
        }
        tags.add("ontology_mismatch")
        return (
            f"病理 Q 波规则要求同一壁至少 2 个导联 Q≥30 ms 且幅度/比例达标；"
            f"命中壁={matched or '无'}，未决候选={unresolved or '无'}。"
            "LUDB 的 scar/undefined ischemia 标签比单一病理 Q 波规则更宽。"
        ), tags

    return f"对应规则未触发；{_rule_state_summary(rules)}。", tags


def _specific_false_positive_reason(
    category: str,
    rules: Sequence[dict],
    labels: Sequence[str],
) -> tuple[str, set[str]]:
    tags: set[str] = set()
    matched_rule = next(
        (rule for rule in rules if str(rule.get("status")) == "matched"),
        rules[0] if rules else {},
    )
    evidence = matched_rule.get("evidence") or {}
    if category in AMPLITUDE_DEPENDENT_CATEGORIES:
        tags.add("amplitude_scaling")

    if category == "sinus_bradycardia":
        tags.add("ontology_mismatch")
        return (
            f"算法 HR={_float(evidence.get('heart_rate_bpm')):.1f} bpm 低于 60，"
            "所以输出一般性心动过缓；LUDB 没有标“窦性心动过缓”，两者节律语义不完全相同。"
        ), tags
    if category == "sinus_tachycardia":
        tags.add("ontology_mismatch")
        return (
            f"算法 HR={_float(evidence.get('heart_rate_bpm')):.1f} bpm 高于 100，"
            "所以输出一般性心动过速；LUDB 没有标“窦性心动过速”。"
        ), tags
    if category == "atrial_fibrillation":
        tags.add("rhythm_confusion")
        confounder = ""
        if any(label.startswith("Atrial extrasystole") for label in labels):
            confounder = "；LUDB 同时存在 PAC/不规则窦律，提示异位搏动造成的 RR 不齐被误作 AF"
        return (
            f"AF 规则由 RR-CV={_float(evidence.get('rr_cv')):.3f}、"
            f"有序 P 支持={_float(evidence.get('organized_p_ratio')):.3f} 触发{confounder}。"
        ), tags
    if category == "atrial_flutter":
        tags.add("rhythm_confusion")
        return "多导联 F 波规则触发，但 LUDB 未标房扑；需人工复核 QRST 消减残差。", tags
    if category in {
        "premature_atrial_complexes",
        "premature_ventricular_complexes",
    }:
        tags.add("rhythm_confusion")
        return (
            f"规则发现 {evidence.get('count', 0)}/{evidence.get('analyzable_beats', 'N/A')} "
            "个满足提前 RR 与形态条件的候选，但 LUDB 未标对应异位搏动。"
        ), tags
    if category == "first_degree_av_block":
        tags.add("subtype_confusion")
        return (
            f"算法 PR={_float(evidence.get('pr_ms')):.1f} ms 超过 200 ms，"
            "触发一度房室传导延迟；可能是 P 起点选择或代表心搏 PR 偏长。"
        ), tags
    if category == "complete_av_block":
        tags.add("subtype_confusion")
        return "算法房室分离证据触发完全性阻滞，但 LUDB 未标该分级。", tags
    if category == "right_bundle_branch_block":
        tags.add("subtype_confusion")
        return (
            f"QRS={_float(evidence.get('qrs_ms')):.1f} ms 且 V1 R′/侧壁 S 组合触发 RBBB，"
            "但 LUDB 未给出该束支阻滞标签。"
        ), tags
    if category == "left_bundle_branch_block":
        tags.add("subtype_confusion")
        return (
            f"QRS={_float(evidence.get('qrs_ms')):.1f} ms 且侧壁形态触发 LBBB，"
            "但 LUDB 未标该亚型。"
        ), tags
    if category == "left_anterior_fascicular_block":
        tags.add("subtype_confusion")
        return (
            f"算法 QRS 轴={_float(evidence.get('qrs_axis_deg')):.1f}° 并满足分支形态，"
            "但 LUDB 未标 LAFB。"
        ), tags
    if category == "nonspecific_ivcd":
        tags.add("subtype_confusion")
        return (
            f"算法 QRS={_float(evidence.get('qrs_ms')):.1f} ms 且未匹配 BBB，"
            "因此回退为 IVCD；LUDB 可能标了不同传导亚型或未标。"
        ), tags
    if category == "left_ventricular_hypertrophy":
        tags.add("amplitude_scaling")
        return (
            f"LVH 电压规则命中 {evidence.get('matched_criteria') or []}，"
            "但 LUDB 未标 LVH；在逐导联幅值归一化数据上该命中不可验证。"
        ), tags
    if category == "nonspecific_repolarization_abnormality":
        tags.add("ontology_mismatch")
        return (
            f"T 波规则由连续倒置对={_compact_pairs(evidence.get('inverted_contiguous_pairs') or [])}、"
            f"高尖对={_compact_pairs(evidence.get('tall_contiguous_pairs') or [])} 触发；"
            "若 LUDB 标的是缺血/瘢痕，这更像家族归类不一致，而非纯粹无异常。"
        ), tags
    if category == "ischemia_or_stemi":
        tags.update({"ontology_mismatch", "amplitude_scaling"})
        pairs = evidence.get("qualifying_contiguous_pairs") or []
        return (
            f"绝对 ST 阈值在相邻导联 {pairs or '未知'} 上触发急性缺血筛查；"
            "LUDB 未标 Ischemia/STEMI，且幅值归一化使该触发可信度不足。"
        ), tags
    if category == "scar_or_indeterminate_infarction":
        tags.update({"ontology_mismatch", "amplitude_scaling"})
        return (
            f"病理 Q 波规则命中壁={evidence.get('matched_groups') or '未知'}，"
            "但 LUDB 未标 scar/undefined infarction；需在校准幅值及专家图形上复核。"
        ), tags
    return f"规则已触发但 LUDB 无对应标签；{_rule_state_summary(rules)}。", tags


def _unsupported_reason(
    category: str,
    labels: Sequence[str],
    rules: Sequence[dict],
    measurement: dict[str, str],
) -> tuple[str, set[str]]:
    tags = {"output_gap"}
    if category == "sinus_rhythm":
        return "权威规则层只输出心率异常、AF/AFL 等异常语句，不输出“窦性心律正常”确认。", tags
    if category == "other_sinus_dysrhythmia":
        tags.add("rhythm_confusion")
        return "权威规则层没有窦性心律不齐/不规则窦律的最终语句，且容易与 PAC 或 AF 的 RR 不齐混淆。", tags
    if category in {
        "normal_or_positional_axis",
        "left_axis_deviation",
        "right_axis_deviation",
    }:
        tags.add("amplitude_scaling")
        algorithm_axis = measurement.get("algorithm_qrs_axis")
        reference_axis = measurement.get("ground_truth_qrs_axis")
        return (
            "QRS 轴虽有数值测量"
            f"（算法={algorithm_axis or 'N/A'}°，参考={reference_axis or 'N/A'}°），"
            "但没有被投影为权威最终轴诊断；逐导联独立缩放也会扭曲跨导联电轴计算。"
        ), tags
    if category == "atrial_enlargement_or_overload":
        tags.add("amplitude_scaling")
        target_rules = [
            rule
            for rule in rules
            if _evaluates_code(rule)
            in {"left_atrial_abnormality", "right_atrial_abnormality"}
        ]
        if not target_rules:
            return "未找到心房异常权威规则结果。", tags
        return (
            "心房肥厚/负荷需要 II 导联 P 波及 V1 终末 P 波幅度/时限；"
            f"当前规则状态：{_rule_state_summary(target_rules)}。"
        ), tags
    if category == "right_ventricular_hypertrophy":
        tags.add("amplitude_scaling")
        target = [
            rule for rule in rules if _evaluates_code(rule) == "rvh_pattern"
        ]
        return (
            "RVH 规则需要 V1 R/S 幅度及右轴偏；逐导联幅值归一化破坏电压关系。"
            f"规则状态：{_rule_state_summary(target)}。"
        ), tags
    if category == "ventricular_pacing":
        return "起搏信息用于测量门控和抑制策略，但当前权威最终语句没有稳定输出“心室起搏”诊断。", tags
    if category == "early_repolarization":
        tags.update({"ontology_mismatch", "amplitude_scaling"})
        return "当前权威临床规则层没有独立的早期复极最终语句；绝对 ST/J 幅值又受 LUDB 缩放限制。", tags
    return "该 LUDB 参考家族没有对应的权威最终输出映射。", tags


def _unaligned_prediction_reason(code: str, rules: Sequence[dict]) -> tuple[str, set[str]]:
    translated = PREDICTION_CODE_ZH.get(code, code)
    matched = [
        rule
        for rule in rules
        if rule.get("statement_code") == code and rule.get("status") == "matched"
    ]
    evidence = (matched[0].get("evidence") or {}) if matched else {}
    if code == "low_qrs_voltage_precordial_leads":
        values = evidence.get("qrs_peak_to_peak_mv") or {}
        value_text = "、".join(
            f"{lead}={float(value):.3f}" for lead, value in values.items()
        )
        return (
            f"{translated}：规则看到六个胸前导联峰峰值均 <1.0 mV（{value_text}）。"
            "但本地 LUDB 的每个导联整段范围都被缩放为恰好 1.000 mV，因此这是数据标度与规则阈值耦合造成的系统性输出。"
        ), {"systematic_false_output", "amplitude_scaling"}
    if code in {
        "borderline_short_qt",
        "possible_short_qt_pattern",
        "prolonged_qt",
        "wide_qrs_repolarization_review",
    }:
        return f"{translated}：LUDB 诊断词表没有 QT/JT 类标签，因此只能标记为未验证输出，不能据此判真伪。", {"ontology_mismatch"}
    if code == "second_degree_av_block_pattern":
        return f"{translated}：LUDB 本批没有二度房室阻滞标签；若参考为三度阻滞则属于分级错误，否则是未验证/潜在误报。", {"subtype_confusion"}
    if code == "lpfb_pattern":
        return f"{translated}：LUDB 诊断词表没有 LPFB 标签，且右轴偏可受跨导联幅值缩放影响。", {"ontology_mismatch", "amplitude_scaling"}
    if code == "ventricular_preexcitation_pattern":
        return f"{translated}：LUDB 本批没有预激标签，当前输出属于未验证筛查结果。", {"ontology_mismatch"}
    if code == "possible_precordial_lead_reversal":
        return f"{translated}：LUDB 没有导联错接参考标签，需从原始波形人工复核。", {"ontology_mismatch"}
    if code == "technically_limited":
        return f"{translated}：这是质量限制而非诊断，但会使部分临床规则不可用。", {"missing_input"}
    if code == "low_qrs_voltage_limb_leads":
        return f"{translated}：绝对幅值判据在当前逐导联归一化 LUDB 上不可验证。", {"amplitude_scaling"}
    return f"{translated}：LUDB 没有直接可比标签或当前映射未覆盖。", {"ontology_mismatch"}


def _audit_ludb_scaling(ludb_dir: Path) -> dict[str, object]:
    ranges: list[float] = []
    record_count = 0
    for header_path in sorted(
        ludb_dir.glob("*.hea"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    ):
        record = wfdb.rdrecord(str(header_path.with_suffix("")))
        signal = np.asarray(record.p_signal, dtype=float)
        ranges.extend(
            (np.nanmax(signal, axis=0) - np.nanmin(signal, axis=0)).tolist()
        )
        record_count += 1
    values = np.asarray(ranges, dtype=float)
    return {
        "record_count": record_count,
        "lead_trace_count": int(values.size),
        "minimum_full_range_mv": float(np.min(values)),
        "median_full_range_mv": float(np.median(values)),
        "maximum_full_range_mv": float(np.max(values)),
        "exactly_one_mv_count": int(np.sum(np.isclose(values, 1.0, atol=1e-9))),
    }


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_root_causes(
    ludb_dir: Path,
    feature_dir: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaling_audit = _audit_ludb_scaling(ludb_dir)

    with (feature_dir / "summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        measurements = {row["record_id"]: row for row in csv.DictReader(handle)}

    headers = sorted(
        ludb_dir.glob("*.hea"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    tag_record_counts: Counter[str] = Counter()
    missed_category_counts: Counter[str] = Counter()
    extra_category_counts: Counter[str] = Counter()

    supported_code_union = frozenset(
        code for spec in SUPPORTED_CATEGORIES for code in spec.prediction_codes
    )

    for header_path in headers:
        record_id = header_path.stem
        labels = _parse_ludb_labels(header_path)
        feature_path = feature_dir / record_id / f"{record_id}_features.json"
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
        codes, overall_status = _load_prediction(feature_path)
        rules = _iter_rules(payload)
        measurement = measurements.get(record_id, {})

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
        verdict = _verdict(
            gt_supported,
            predicted_supported,
            true_positive,
            false_positive,
            false_negative,
        )

        causes: list[str] = []
        tags: set[str] = set()
        for category in sorted(false_negative):
            category_rules = _rules_for_category(category, rules)
            reason, reason_tags = _specific_miss_reason(
                category, category_rules, measurement, labels
            )
            causes.append(f"漏诊[{CATEGORY_ZH[category]}]：{reason}")
            tags.update(reason_tags)
            missed_category_counts[category] += 1

        for category in sorted(false_positive):
            category_rules = _rules_for_category(category, rules)
            reason, reason_tags = _specific_false_positive_reason(
                category, category_rules, labels
            )
            causes.append(f"误报[{CATEGORY_ZH[category]}]：{reason}")
            tags.update(reason_tags)
            extra_category_counts[category] += 1

        for category in sorted(gt_unsupported):
            reason, reason_tags = _unsupported_reason(
                category, labels, rules, measurement
            )
            causes.append(f"未覆盖[{CATEGORY_ZH[category]}]：{reason}")
            tags.update(reason_tags)

        for code in sorted(other_codes):
            reason, reason_tags = _unaligned_prediction_reason(code, rules)
            causes.append(f"额外输出[{code}]：{reason}")
            tags.update(reason_tags)

        measurement_errors = _large_measurement_errors(measurement)
        if measurement_errors:
            tags.add("measurement_error")
            causes.append("测量链问题：" + "；".join(measurement_errors))

        if SYSTEMATIC_UNVERIFIED_CODES.intersection(codes):
            tags.update({"systematic_false_output", "amplitude_scaling"})

        if not causes:
            causes.append("支持范围内诊断家族一致，未发现可解释的类别级错误。")

        for tag in tags:
            tag_record_counts[tag] += 1
        fixes = [FIX_BY_TAG[tag] for tag in sorted(tags)]

        row = {
            "record_id": record_id,
            "verdict": verdict,
            "matched_categories": _join_zh(true_positive),
            "missed_categories": _join_zh(false_negative),
            "extra_categories": _join_zh(false_positive),
            "uncovered_reference_categories": _join_zh(gt_unsupported),
            "cause_tags": " | ".join(CAUSE_TAG_ZH[tag] for tag in sorted(tags)),
            "root_cause_analysis": " || ".join(causes),
            "recommended_actions": " || ".join(fixes),
            "overall_status": overall_status,
        }
        rows.append(row)
        details.append(
            {
                **row,
                "labels": labels,
                "codes": codes,
                "causes": causes,
                "fixes": fixes,
            }
        )

    _write_csv(rows, output_dir / "record_root_cause_analysis.csv")
    summary = {
        "schema_version": "ludb_ecgfeat_root_cause_analysis.v1",
        "record_count": len(rows),
        "ludb_scaling_audit": scaling_audit,
        "root_cause_record_counts": {
            CAUSE_TAG_ZH[tag]: count for tag, count in tag_record_counts.most_common()
        },
        "missed_category_counts": {
            CATEGORY_ZH[key]: count for key, count in missed_category_counts.most_common()
        },
        "extra_category_counts": {
            CATEGORY_ZH[key]: count for key, count in extra_category_counts.most_common()
        },
    }
    (output_dir / "root_cause_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# LUDB 200 条记录：ecgfeat 无法正确诊断的逐条根因分析",
        "",
        "## 最重要的全局根因",
        "",
        (
            f"- 本地 LUDB 共 {scaling_audit['record_count']} 条记录、"
            f"{scaling_audit['lead_trace_count']} 条导联轨迹。经 WFDB 物理单位转换后，"
            f"{scaling_audit['exactly_one_mv_count']}/{scaling_audit['lead_trace_count']} "
            "条轨迹的整段最大值减最小值都恰好为 1.000 mV。"
        ),
        "- 这表明每个导联被独立幅值归一化；绝对电压和跨导联幅值比例不再具有原始校准意义。",
        "- 因而 LVH、低电压、ST/T 幅度、病理 Q 波幅度、电轴和部分 BBB 形态规则无法在该信号标度上被公平验证。",
        "- 该限制直接对应 `胸前导联低电压` 200/200，以及 LUDB 的 LVH/负荷 108 条中 0 条命中。",
        "- 以下“原因”描述当前规则为什么未触发或误触发，不等同于重新做一次心脏科专家判读。",
        "",
        "## 根因覆盖统计",
        "",
        "| 根因 | 涉及记录数 |",
        "| --- | ---: |",
    ]
    for tag, count in tag_record_counts.most_common():
        lines.append(f"| {CAUSE_TAG_ZH[tag]} | {count} |")

    lines.extend(
        [
            "",
            "## 逐记录索引",
            "",
            "| 记录 | 原结论 | 漏诊 | 误报 | 主要根因 |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for detail in details:
        lines.append(
            f"| [{detail['record_id']}](#root-record-{detail['record_id']}) | "
            f"{detail['verdict']} | {detail['missed_categories']} | "
            f"{detail['extra_categories']} | "
            f"{detail['cause_tags'].replace(' | ', '；')} |"
        )

    lines.extend(["", "## 逐条根因", ""])
    for detail in details:
        record_id = str(detail["record_id"])
        lines.extend(
            [
                f'<a id="root-record-{record_id}"></a>',
                f"### Record {record_id} — {detail['verdict']}",
                "",
                "- LUDB 参考：" + "；".join(
                    _translate_label(label) for label in detail["labels"]
                ),
                "- ecgfeat 输出：" + _join_predictions(detail["codes"]),
                "- 失败原因：",
                "",
            ]
        )
        for index, cause in enumerate(detail["causes"], start=1):
            lines.append(f"  {index}. {cause}")
        lines.extend(["", "- 建议：", ""])
        if detail["fixes"]:
            for index, fix in enumerate(detail["fixes"], start=1):
                lines.append(f"  {index}. {fix}")
        else:
            lines.append("  1. 当前可比家族一致；仍需在校准幅值和外部数据上复核。")
        lines.append("")

    (output_dir / "ROOT_CAUSE_BY_RECORD.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain per-record ecgfeat diagnostic failures on LUDB."
    )
    parser.add_argument("--ludb-dir", type=Path, default=DEFAULT_LUDB_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows = analyze_root_causes(
        ludb_dir=args.ludb_dir.resolve(),
        feature_dir=args.feature_dir.resolve(),
        output_dir=args.out_dir.resolve(),
    )
    print(f"Explained {len(rows)} records.")
    print(f"Saved: {args.out_dir.resolve() / 'ROOT_CAUSE_BY_RECORD.md'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import Any, Callable

from medgemma_ecg_core import (
    DEFAULT_MODEL_MAX_LEN,
    DEFAULT_MAX_OUTPUT_TOKENS,
    build_context_summary_from_paths,
    build_prompt as shared_build_prompt,
    diagnostic_gate_policy,
    sanitize_report_text as shared_sanitize_report_text,
    summarize_clinical_interpretation,
)
from medgemma_runtime import infer_runtime_config

MODEL_PATH = "./medgemma-27b"
MODEL_MAX_LEN = DEFAULT_MODEL_MAX_LEN
MODEL_MAX_OUTPUT_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS
REPORT_CHAR_LIMIT = 6000
STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LANGUAGE_CHOICES = [("English", "en"), ("日本語", "ja")]


def _is_loopback_host(host: str) -> bool:
    value = host.strip().strip("[]").lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False

I18N = {
    "en": {
        "title_markdown": "# ECG Gemma — ECG Diagnostic Assistant",
        "description_markdown": (
            "Upload a current `*_features.json` and an optional `*_report.txt`. "
            "The app prioritizes structured features, removes `Dx` labels from the "
            "readable report before using it as supplemental context, and returns "
            "the most likely diagnosis plus ranked similar diagnoses."
        ),
        "language_label": "Language",
        "runtime_header": "### Runtime Configuration",
        "runtime_env_line": (
            "- Available environment variables: "
            "`ECG_GEMMA_QUANTIZATION=auto|none|fp8`, `ECG_GEMMA_TP_SIZE=<n>`"
        ),
        "input_header": "### Inputs",
        "output_header": "### Diagnosis Output",
        "features_label": "Upload *_features.json or legacy ecg_input.json",
        "report_label": "Upload *_report.txt (optional)",
        "notes_label": "Additional notes (optional)",
        "notes_placeholder": (
            "You can enter symptoms, the chief complaint, or what you want the model to focus on."
        ),
        "run_button": "Run Diagnosis",
        "context_accordion": "Model Input Summary",
        "prompt_accordion": "Prompt Preview",
        "preview_error_prefix": "Context build failed",
        "empty_context_message": "Please upload `*_features.json` or `*_report.txt`, or add brief notes.",
        "missing_input_message": "Please upload a valid ECG feature file or report file first.",
        "quality_stop_message": (
            "ABSTAIN — the ECG diagnostic quality gate is {state}. This single-shot UI "
            "cannot safely enforce domain-level restrictions, so no model inference was "
            "performed. Reacquire/repair the ECG or use the governed layered workflow. "
            "Reasons: {reasons}."
        ),
        "regenerate_notice": "Language updated. Click `Run Diagnosis` to regenerate the diagnostic answer in the selected language.",
        "unknown_json_header": "[Uploaded JSON was not recognized as the current features schema]",
        "supplementary_report_header": "[Supplementary Readable Report (Dx labels removed)]",
        "user_notes_header": "[User Notes]",
        "legacy_header": "[Legacy ECG Input]",
        "legacy_hr": "Heart rate",
        "legacy_rri": "R-R interval",
        "legacy_p_wave": "P-wave morphology/relationship",
        "legacy_pr": "PR interval",
        "legacy_qrs": "QRS complex",
        "summary_header": "[Structured ECG Feature Summary]",
        "patient": "Patient",
        "age": "Age",
        "sex": "Sex",
        "input_fs": "input_fs",
        "internal_fs": "internal_fs",
        "detected_beats": "detected beats",
        "global_measurements": "Global measurements",
        "atrial_rate": "Atrial rate",
        "qt_dispersion": "QT dispersion",
        "axes": "Axes",
        "rhythm_interpretation": "Rhythm/interval interpretation",
        "probable_af": "probable AF",
        "hr_class": "HR class",
        "pr_class": "PR class",
        "qrs_width": "QRS width",
        "qtc_class": "QTc",
        "av_block": "AV block",
        "bbb": "BBB",
        "wpw": "WPW",
        "morphology_interpretation": "Morphology interpretation",
        "p_morphology": "P morphology",
        "lae_suspected": "LAE suspected",
        "lae_definite": "LAE definite",
        "ptf_v1": "PTF-V1",
        "q_wave_territories": "Q-wave territories",
        "r_progression": "R progression",
        "rs_transition": "RS transition",
        "st_t": "ST/T",
        "st_elevation": "ST elevation",
        "st_depression": "ST depression",
        "st_elevated_territories": "ST elevated territories",
        "st_depressed_territories": "ST depressed territories",
        "reciprocal_change": "reciprocal change",
        "tall_t_leads": "tall T leads",
        "other_clues": "Other structural clues",
        "lvh": "LVH",
        "lvh_criteria": "LVH criteria",
        "low_voltage": "low voltage",
        "rvh_suspected": "RVH suspected",
        "limb_reversal": "limb reversal",
        "precordial_reversal": "precordial reversal",
        "paced_rhythm": "paced rhythm",
        "signal_quality": "Signal quality",
        "overall_unreliable": "overall unreliable leads",
        "p_unreliable": "P-unreliable",
        "qrs_unreliable": "QRS-unreliable",
        "t_unreliable": "T-unreliable",
        "qt_unreliable": "QT-unreliable",
        "reliable_qt": "Reliable QT leads used by extractor",
        "beat_groups": "Beat groups",
        "representative_measurements": "Representative lead measurements",
        "beats": "beats",
        "mean_rr": "mean RR",
        "mean_qrs": "mean QRS",
        "dominant": "dominant",
        "quality_ok": "OK",
        "quality_bad": "BAD",
        "runtime_model": "Model",
        "runtime_gpu": "GPU",
        "runtime_gpu_count": "GPU count",
        "runtime_cc": "Compute capability",
        "runtime_quantization": "Quantization",
        "runtime_tp": "Tensor parallel size",
        "yes": "Yes",
        "no": "No",
        "unknown": "Unknown",
    },
    "ja": {
        "title_markdown": "# ECG Gemma — ECG診断アシスタント",
        "description_markdown": (
            "最新の `*_features.json` と、必要に応じて `*_report.txt` をアップロードしてください。"
            "このアプリは構造化特徴を優先して使用し、可読レポート中の `Dx` ラベルを除外したうえで"
            "補助コンテキストとして利用し、最も可能性の高い診断と類似診断を順位付きで返します。"
        ),
        "language_label": "言語",
        "runtime_header": "### 実行時設定",
        "runtime_env_line": (
            "- 利用可能な環境変数: "
            "`ECG_GEMMA_QUANTIZATION=auto|none|fp8`, `ECG_GEMMA_TP_SIZE=<n>`"
        ),
        "input_header": "### 入力",
        "output_header": "### 診断出力",
        "features_label": "*_features.json または旧式 ecg_input.json をアップロード",
        "report_label": "*_report.txt をアップロード（任意）",
        "notes_label": "補足情報（任意）",
        "notes_placeholder": "症状、主訴、またはモデルに特に注目してほしい点を入力できます。",
        "run_button": "診断を実行",
        "context_accordion": "モデル入力サマリー",
        "prompt_accordion": "Prompt プレビュー",
        "preview_error_prefix": "コンテキスト構築に失敗しました",
        "empty_context_message": "`*_features.json` または `*_report.txt` をアップロードするか、短い補足情報を入力してください。",
        "missing_input_message": "先に有効な ECG 特徴ファイルまたはレポートファイルをアップロードしてください。",
        "quality_stop_message": (
            "判定保留 — ECG 診断品質ゲートは {state} です。この単発 UI では領域別制限を"
            "安全に強制できないため、モデル推論は実行されませんでした。ECG を再取得・修復するか、"
            "管理された分層ワークフローを使用してください。理由: {reasons}。"
        ),
        "regenerate_notice": "言語を更新しました。選択した言語で診断結果を再生成するには `診断を実行` を押してください。",
        "unknown_json_header": "[アップロードされた JSON は現在の features schema として認識されませんでした]",
        "supplementary_report_header": "[補足可読レポート（Dx ラベル削除済み）]",
        "user_notes_header": "[ユーザー補足情報]",
        "legacy_header": "[旧形式 ECG 入力]",
        "legacy_hr": "心拍数",
        "legacy_rri": "R-R 間隔",
        "legacy_p_wave": "P波形態・関係",
        "legacy_pr": "PR 間隔",
        "legacy_qrs": "QRS 波形",
        "summary_header": "[構造化 ECG 特徴サマリー]",
        "patient": "患者",
        "age": "年齢",
        "sex": "性別",
        "input_fs": "入力 fs",
        "internal_fs": "内部 fs",
        "detected_beats": "検出拍数",
        "global_measurements": "全体計測",
        "atrial_rate": "心房レート",
        "qt_dispersion": "QT dispersion",
        "axes": "電気軸",
        "rhythm_interpretation": "リズム・間隔の解釈",
        "probable_af": "AF 疑い",
        "hr_class": "心拍数分類",
        "pr_class": "PR 分類",
        "qrs_width": "QRS 幅分類",
        "qtc_class": "QTc 分類",
        "av_block": "房室ブロック",
        "bbb": "脚ブロック",
        "wpw": "WPW",
        "morphology_interpretation": "波形の解釈",
        "p_morphology": "P 波形態",
        "lae_suspected": "LAE 疑い",
        "lae_definite": "LAE 確実",
        "ptf_v1": "PTF-V1",
        "q_wave_territories": "Q 波領域",
        "r_progression": "R 波進行",
        "rs_transition": "RS transition",
        "st_t": "ST/T",
        "st_elevation": "ST 上昇",
        "st_depression": "ST 低下",
        "st_elevated_territories": "ST 上昇領域",
        "st_depressed_territories": "ST 低下領域",
        "reciprocal_change": "鏡像変化",
        "tall_t_leads": "高い T 波導聯",
        "other_clues": "その他の構造的手掛かり",
        "lvh": "LVH",
        "lvh_criteria": "LVH 基準",
        "low_voltage": "低電位",
        "rvh_suspected": "RVH 疑い",
        "limb_reversal": "四肢導聯逆転",
        "precordial_reversal": "胸部導聯逆転",
        "paced_rhythm": "ペーシングリズム",
        "signal_quality": "信号品質",
        "overall_unreliable": "全体的に不良な導聯",
        "p_unreliable": "P 波不良",
        "qrs_unreliable": "QRS 不良",
        "t_unreliable": "T 波不良",
        "qt_unreliable": "QT 不良",
        "reliable_qt": "抽出器が使用した reliable QT leads",
        "beat_groups": "拍群",
        "representative_measurements": "代表導聯計測値",
        "beats": "拍",
        "mean_rr": "平均 RR",
        "mean_qrs": "平均 QRS",
        "dominant": "主要群",
        "quality_ok": "良好",
        "quality_bad": "不良",
        "runtime_model": "モデル",
        "runtime_gpu": "GPU",
        "runtime_gpu_count": "GPU 数",
        "runtime_cc": "Compute capability",
        "runtime_quantization": "量子化",
        "runtime_tp": "Tensor parallel size",
        "yes": "はい",
        "no": "いいえ",
        "unknown": "不明",
    },
}


def get_text(language: str) -> dict[str, str]:
    return I18N.get(language, I18N["en"])


def _fmt_num(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return ("Yes" if value else "No") + unit
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _fmt_bool(value: Any, language: str = "en") -> str:
    t = get_text(language)
    if value is None:
        return t["unknown"]
    return t["yes"] if value else t["no"]


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


def _safe_read_json(file_obj) -> dict[str, Any]:
    with open(file_obj.name, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_text(file_obj) -> str:
    with open(file_obj.name, "r", encoding="utf-8") as f:
        return f.read()


def build_startup_banner(runtime_config: dict[str, Any], language: str = "en") -> str:
    t = get_text(language)
    quant_text = runtime_config["quantization"] or "none"
    return (
        f"{t['runtime_header']}\n"
        f"- {t['runtime_model']}: `{MODEL_PATH}`\n"
        f"- {t['runtime_gpu']}: `{runtime_config['device_name']}`\n"
        f"- {t['runtime_gpu_count']}: `{runtime_config['device_count']}`\n"
        f"- {t['runtime_cc']}: `{runtime_config['compute_capability']}`\n"
        f"- {t['runtime_quantization']}: `{quant_text}`\n"
        f"- {t['runtime_tp']}: `{runtime_config['tensor_parallel_size']}`\n"
        f"{t['runtime_env_line']}"
    )


def _is_current_features_schema(data: dict[str, Any]) -> bool:
    return isinstance(data, dict) and "global_features" in data and "interpretation" in data


def _is_legacy_input_schema(data: dict[str, Any]) -> bool:
    legacy_keys = {"心率 (HR)", "R-R 间期 (RRI)", "P波形态与关系", "PR 间期", "QRS 波群"}
    return isinstance(data, dict) and any(key in data for key in legacy_keys)


def _summarize_legacy_input(data: dict[str, Any], language: str = "en") -> str:
    t = get_text(language)
    lines = [
        t["legacy_header"],
        f"- {t['legacy_hr']}: {data.get('心率 (HR)', 'N/A')}",
        f"- {t['legacy_rri']}: {data.get('R-R 间期 (RRI)', 'N/A')}",
        f"- {t['legacy_p_wave']}: {data.get('P波形态与关系', 'N/A')}",
        f"- {t['legacy_pr']}: {data.get('PR 间期', 'N/A')}",
        f"- {t['legacy_qrs']}: {data.get('QRS 波群', 'N/A')}",
    ]
    return "\n".join(lines)


def _summarize_groups(groups: dict[str, Any], language: str = "en") -> str:
    t = get_text(language)
    if not groups:
        return "—"
    lines = []
    for group_id in sorted(groups, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        group = groups[group_id]
        dominant = group.get("flags", {}).get("dominant_group", False)
        pieces = [
            f"G{group.get('group_id', group_id)}",
            f"{group.get('member_count', 'N/A')} {t['beats']}",
            f"{_fmt_num(group.get('member_pct'), 0, '%')}",
            f"{t['mean_rr']} {_fmt_num(group.get('mean_rr_ms'), 0, ' ms')}",
            f"{t['mean_qrs']} {_fmt_num(group.get('mean_qrs_ms'), 0, ' ms')}",
        ]
        if dominant:
            pieces.append(t["dominant"])
        lines.append(" | ".join(pieces))
    return "\n".join(f"- {line}" for line in lines)


def _summarize_representative_leads(
    representative_leads: dict[str, Any],
    quality: dict[str, Any],
    language: str = "en",
) -> str:
    t = get_text(language)
    if not representative_leads:
        return "—"

    lines = []
    for lead in STANDARD_12_LEADS:
        lead_info = representative_leads.get(lead, {})
        params = lead_info.get("params", {})
        q = quality.get(lead, {})
        reliability = (
            t["quality_ok"]
            if q.get("reliable", True)
            else f"{t['quality_bad']} ({_fmt_list(q.get('flags', []))})"
        )
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


def _summarize_current_features(data: dict[str, Any], language: str = "en") -> str:
    t = get_text(language)
    gf = data.get("global_features", {})
    interp = data.get("interpretation", {})
    groups = data.get("groups", {})
    quality = data.get("quality", {})
    representative_leads = data.get("representative_leads", {})
    metadata = data.get("metadata", {})
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
        t["summary_header"],
        f"- {t['patient']}: {t['age']} {patient.get('age', 'N/A')} | {t['sex']} {patient.get('sex', 'N/A')} | "
        f"{t['input_fs']} {metadata.get('input_fs', 'N/A')} Hz | {t['internal_fs']} {metadata.get('internal_fs', 'N/A')} Hz | "
        f"{t['detected_beats']} {metadata.get('n_beats', 'N/A')}",
        f"- {t['global_measurements']}: HR {_fmt_num(gf.get('heart_rate_bpm'), 1, ' bpm')}, "
        f"{t['atrial_rate']} {_fmt_num(gf.get('atrial_rate_bpm'), 1, ' bpm')}, "
        f"PR {_fmt_num(gf.get('pr_ms'), 0, ' ms')}, "
        f"QRS {_fmt_num(gf.get('qrs_ms'), 0, ' ms')}, "
        f"QT {_fmt_num(gf.get('qt_ms'), 0, ' ms')}, "
        f"QTcB {_fmt_num(gf.get('qtc_bazett_ms'), 0, ' ms')}, "
        f"QTcF {_fmt_num(gf.get('qtc_fridericia_ms'), 0, ' ms')}, "
        f"{t['qt_dispersion']} {_fmt_num(gf.get('qt_dispersion_ms'), 0, ' ms')}",
        f"- {t['axes']}: P {_fmt_num(gf.get('p_axis_deg'), 0, '°')}, "
        f"QRS {_fmt_num(gf.get('qrs_axis_deg'), 0, '°')}, "
        f"T {_fmt_num(gf.get('t_axis_deg'), 0, '°')}, "
        f"ST {_fmt_num(gf.get('st_axis_deg'), 0, '°')}",
        *clinical_lines,
        f"- {t['rhythm_interpretation']}: RR {interp.get('rr_irregularity_class', 'N/A')} (CV={_fmt_num(interp.get('rr_cv'), 4)}), "
        f"{t['probable_af']}={_fmt_bool(interp.get('probable_af'), language)}, "
        f"{t['hr_class']}={interp.get('heart_rate_class', 'N/A')}, "
        f"{t['pr_class']}={interp.get('pr_class', 'N/A')}, "
        f"{t['qrs_width']}={interp.get('qrs_width_class', 'N/A')}, "
        f"{t['qtc_class']}={interp.get('qtc_class', 'N/A')}, "
        f"{t['av_block']}={interp.get('avb_grade') or 'none'}, "
        f"{t['bbb']}={interp.get('bundle_branch_block') or 'none'}, "
        f"{t['wpw']}={_fmt_bool(interp.get('wpw_pattern'), language)}",
        f"- {t['morphology_interpretation']}: {t['p_morphology']}={interp.get('p_morphology_class') or '—'}, "
        f"{t['lae_suspected']}={_fmt_bool(interp.get('lae_suspected'), language)}, "
        f"{t['lae_definite']}={_fmt_bool(interp.get('lae_definite'), language)}, "
        f"{t['ptf_v1']}={interp.get('ptf_v1_class') or '—'}, "
        f"{t['q_wave_territories']}={_fmt_list(interp.get('q_wave_territories'))}, "
        f"{t['r_progression']}={interp.get('r_progression_class') or '—'}, "
        f"{t['rs_transition']}={interp.get('r_s_transition_lead') or '—'}",
        f"- {t['st_t']}: {t['st_elevation']}={_fmt_lead_map(interp.get('st_elevation_leads'))}, "
        f"{t['st_depression']}={_fmt_lead_map(interp.get('st_depression_leads'))}, "
        f"{t['st_elevated_territories']}={_fmt_list(interp.get('st_territories_elevated'))}, "
        f"{t['st_depressed_territories']}={_fmt_list(interp.get('st_territories_depressed'))}, "
        f"{t['reciprocal_change']}={_fmt_bool(interp.get('reciprocal_change_detected'), language)}, "
        f"{t['tall_t_leads']}={_fmt_list(interp.get('tall_t_leads'))}",
        f"- {t['other_clues']}: {t['lvh']}={interp.get('lvh_class') or '—'}, "
        f"{t['lvh_criteria']}={_fmt_list(interp.get('lvh_voltage_criteria'))}, "
        f"{t['low_voltage']}={interp.get('low_voltage_class') or '—'}, "
        f"{t['rvh_suspected']}={_fmt_bool(interp.get('rvh_suspected'), language)}, "
        f"{t['limb_reversal']}={_fmt_bool(interp.get('limb_reversal_suspected'), language)}, "
        f"{t['precordial_reversal']}={_fmt_bool(interp.get('precordial_reversal_suspected'), language)}, "
        f"{t['paced_rhythm']}={_fmt_bool(gf.get('paced_rhythm'), language)}",
        f"- {t['signal_quality']}: {t['overall_unreliable']}={_fmt_list(unreliable_leads)}, "
        f"{t['p_unreliable']}={_fmt_flagged_leads(quality, 'reliable_for_p')}, "
        f"{t['qrs_unreliable']}={_fmt_flagged_leads(quality, 'reliable_for_qrs')}, "
        f"{t['t_unreliable']}={_fmt_flagged_leads(quality, 'reliable_for_t')}, "
        f"{t['qt_unreliable']}={_fmt_flagged_leads(quality, 'reliable_for_qt')}",
        f"- {t['reliable_qt']}: {_fmt_list(metadata.get('reliable_qt_leads'))}",
        f"- {t['beat_groups']}:",
        _summarize_groups(groups, language),
        f"- {t['representative_measurements']}:",
        _summarize_representative_leads(representative_leads, quality, language),
    ]
    return "\n".join(summary_lines)


def sanitize_report_text(report_text: str, char_limit: int = REPORT_CHAR_LIMIT) -> str:
    return shared_sanitize_report_text(report_text, char_limit)


def build_context_summary(features_file, report_file, clinician_notes: str, language: str = "en") -> str:
    features_path = Path(features_file.name) if features_file is not None else None
    report_path = Path(report_file.name) if report_file is not None else None

    if features_path is None and report_path is None and not clinician_notes.strip():
        return ""

    return build_context_summary_from_paths(features_path, report_path, clinician_notes, language)


def build_prompt(context_summary: str, language: str = "en") -> str:
    return shared_build_prompt(context_summary, language)


def features_diagnostic_gate(features_file) -> dict[str, Any] | None:
    """Return a strict gate for every uploaded feature artifact."""
    if features_file is None:
        return None
    path = Path(features_file.name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A file supplied through the feature input is never allowed to become
        # model context merely because its safety contract cannot be parsed.
        data = {}
    if not isinstance(data, dict):
        data = {}
    return diagnostic_gate_policy(data)


def guarded_diagnosis(
    features_file,
    report_file,
    clinician_notes: str,
    language: str,
    generate_text_fn: Callable[[str], str],
) -> tuple[str, str, str]:
    """Build the prompt and fail closed before inference when quality says STOP."""
    context_summary, prompt = preview_prompt(
        features_file,
        report_file,
        clinician_notes,
        language,
    )
    gate = features_diagnostic_gate(features_file)
    if gate is not None and gate.get("state") != "pass":
        state = str(gate.get("state") or "stop").upper()
        reasons = ", ".join(
            [
                *list(gate.get("stop_reasons") or []),
                *list(gate.get("partial_reasons") or []),
            ]
            or ["unspecified_quality_limitation"]
        )
        message = get_text(language)["quality_stop_message"].format(
            state=state,
            reasons=reasons,
        )
        return context_summary, "", message
    if not prompt:
        return context_summary, prompt, get_text(language)["missing_input_message"]
    return context_summary, prompt, generate_text_fn(prompt)


def preview_prompt(features_file, report_file, clinician_notes, language: str = "en"):
    t = get_text(language)
    try:
        context_summary = build_context_summary(features_file, report_file, clinician_notes or "", language)
    except Exception as exc:  # pragma: no cover - UI error surface
        return f"{t['preview_error_prefix']}: {exc}", ""

    if not context_summary:
        return t["empty_context_message"], ""

    return context_summary, build_prompt(context_summary, language)


if __name__ == "__main__":
    import gradio as gr
    from vllm import LLM, SamplingParams

    runtime_config = infer_runtime_config()
    startup_banner = build_startup_banner(runtime_config, "en")

    llm_kwargs = {
        "model": MODEL_PATH,
        "dtype": "bfloat16",
        "max_model_len": MODEL_MAX_LEN,
        "tensor_parallel_size": runtime_config["tensor_parallel_size"],
    }
    if runtime_config["quantization"]:
        llm_kwargs["quantization"] = runtime_config["quantization"]

    try:
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "vLLM initialization failed."
            f" Current configuration: quantization={runtime_config['quantization'] or 'none'},"
            f" tensor_parallel_size={runtime_config['tensor_parallel_size']},"
            f" GPU={runtime_config['device_name']} (cc {runtime_config['compute_capability']})."
            " If your GPU does not support FP8, try setting `ECG_GEMMA_QUANTIZATION=none`."
            " If a single GPU does not have enough memory, try `ECG_GEMMA_TP_SIZE=2` or higher on a multi-GPU setup."
            f" Original error: {exc}"
        ) from exc

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=MODEL_MAX_OUTPUT_TOKENS,
    )

    def diagnose(features_file, report_file, clinician_notes, language):
        def generate(prompt: str) -> str:
            outputs = llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text

        return guarded_diagnosis(
            features_file,
            report_file,
            clinician_notes or "",
            language,
            generate,
        )

    def refresh_ui(language, features_file, report_file, clinician_notes):
        t = get_text(language)
        context_summary, prompt = preview_prompt(features_file, report_file, clinician_notes, language)
        return (
            gr.update(value=t["title_markdown"]),
            gr.update(value=t["description_markdown"]),
            gr.update(value=build_startup_banner(runtime_config, language)),
            gr.update(value=t["input_header"]),
            gr.update(value=t["output_header"]),
            gr.update(label=t["features_label"]),
            gr.update(label=t["report_label"]),
            gr.update(label=t["notes_label"], placeholder=t["notes_placeholder"]),
            gr.update(value=t["run_button"]),
            gr.update(label=t["context_accordion"]),
            gr.update(label=t["prompt_accordion"]),
            context_summary,
            prompt,
            t["regenerate_notice"],
        )

    with gr.Blocks(title="ECG Gemma") as demo:
        title_md = gr.Markdown(get_text("en")["title_markdown"])
        description_md = gr.Markdown(get_text("en")["description_markdown"])
        startup_md = gr.Markdown(startup_banner)
        language = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            value="en",
            label="Language / 言語",
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_md = gr.Markdown(get_text("en")["input_header"])
                features_file = gr.File(
                    label=get_text("en")["features_label"],
                    file_types=[".json"],
                )
                report_file = gr.File(
                    label=get_text("en")["report_label"],
                    file_types=[".txt"],
                )
                clinician_notes = gr.Textbox(
                    label=get_text("en")["notes_label"],
                    lines=6,
                    placeholder=get_text("en")["notes_placeholder"],
                )
                btn = gr.Button(get_text("en")["run_button"], variant="primary")

            with gr.Column(scale=1):
                output_md = gr.Markdown(get_text("en")["output_header"])
                output = gr.Markdown()

        with gr.Accordion(get_text("en")["context_accordion"], open=False) as context_acc:
            context_preview = gr.Textbox(lines=20, buttons=["copy"])
        with gr.Accordion(get_text("en")["prompt_accordion"], open=False) as prompt_acc:
            prompt_preview = gr.Textbox(lines=24, buttons=["copy"])

        language.change(
            refresh_ui,
            inputs=[language, features_file, report_file, clinician_notes],
            outputs=[
                title_md,
                description_md,
                startup_md,
                input_md,
                output_md,
                features_file,
                report_file,
                clinician_notes,
                btn,
                context_acc,
                prompt_acc,
                context_preview,
                prompt_preview,
                output,
            ],
        )

        features_file.change(
            preview_prompt,
            inputs=[features_file, report_file, clinician_notes, language],
            outputs=[context_preview, prompt_preview],
        )
        report_file.change(
            preview_prompt,
            inputs=[features_file, report_file, clinician_notes, language],
            outputs=[context_preview, prompt_preview],
        )
        clinician_notes.change(
            preview_prompt,
            inputs=[features_file, report_file, clinician_notes, language],
            outputs=[context_preview, prompt_preview],
        )
        btn.click(
            diagnose,
            inputs=[features_file, report_file, clinician_notes, language],
            outputs=[context_preview, prompt_preview, output],
        )

    server_name = os.getenv("ECG_GEMMA_UI_HOST", "127.0.0.1").strip() or "127.0.0.1"
    server_port = int(os.getenv("ECG_GEMMA_UI_PORT", "7860"))
    auth = None
    if not _is_loopback_host(server_name):
        username = os.getenv("ECG_GEMMA_UI_USERNAME", "").strip()
        password = os.getenv("ECG_GEMMA_UI_PASSWORD", "")
        if not username or len(password) < 16:
            raise RuntimeError(
                "ECG_GEMMA_UI_HOST 指向非回环地址时，必须设置用户名和至少 16 字符的密码"
            )
        auth = (username, password)
    demo.launch(server_name=server_name, server_port=server_port, auth=auth)

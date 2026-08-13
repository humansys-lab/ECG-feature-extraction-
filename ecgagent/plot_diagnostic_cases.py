"""Plot raw PTB-XL waveforms with ecgfeat/ECGAgent evidence overlays.

This is a scientific visualization, not a generated illustration.  Raw WFDB
samples provide the traces; ecgfeat's exported beat locations, morphology
groups, quality flags, J points and ST measurements provide the annotations.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wfdb

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
GROUP_COLORS = ("#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#D55E00")

CASE_CONFIG: dict[str, dict[str, Any]] = {
    "05135_hr": {
        "focus_beat": 9,
        "focus_leads": ("aVL", "II", "III", "aVF"),
        "focus_title": "High-lateral / reciprocal ST evidence (Agent hypothesis; unconfirmed)",
        "reference": "PTB-XL label: PACE",
    },
    "05540_hr": {
        "focus_beat": 7,
        "focus_leads": ("V2", "V3", "V4", "III"),
        "focus_title": "Single-beat anterior ST evidence (Agent hypothesis; unconfirmed)",
        "reference": "PTB-XL label: PACE",
    },
}


def _load(
    record_path: Path,
    feature_path: Path,
    result_path: Path,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    record = wfdb.rdrecord(str(record_path))
    by_name = {str(name).lower(): index for index, name in enumerate(record.sig_name)}
    signal = np.column_stack(
        [record.p_signal[:, by_name[lead.lower()]] for lead in LEADS]
    )
    return (
        np.asarray(signal, dtype=float),
        float(record.fs),
        json.loads(feature_path.read_text(encoding="utf-8")),
        json.loads(result_path.read_text(encoding="utf-8")),
    )


def _grid(axis: Any, duration: float, ylim: float) -> None:
    axis.set_xlim(0, duration)
    axis.set_ylim(-ylim, ylim)
    axis.set_xticks(np.arange(0, duration + 0.001, 1.0))
    axis.set_xticks(np.arange(0, duration + 0.001, 0.2), minor=True)
    axis.set_yticks(np.arange(-ylim, ylim + 0.001, 0.5))
    axis.grid(which="major", color="#e9a8a8", linewidth=0.42, alpha=0.55)
    axis.grid(which="minor", axis="x", color="#f4cccc", linewidth=0.25, alpha=0.5)
    axis.tick_params(axis="both", labelsize=7, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _quality_suffix(features: dict[str, Any], lead: str) -> str:
    quality = (features.get("quality") or {}).get(lead) or {}
    cautions = [
        wave
        for wave, key in (
            ("P", "reliable_for_p"),
            ("QRS", "reliable_for_qrs"),
            ("T", "reliable_for_t"),
            ("QT", "reliable_for_qt"),
        )
        if quality.get(key) is False
    ]
    return f"  caution:{'/'.join(cautions)}" if cautions else ""


def _diagnosis_text(result: dict[str, Any], reference: str) -> str:
    verdict = result.get("verdict") or {}
    lines = [
        "ECGAgent diagnoses",
        "------------------",
    ]
    for diagnosis in verdict.get("diagnoses") or []:
        lines.append(
            f"[{diagnosis.get('confidence')}/{diagnosis.get('urgency')}] "
            f"{diagnosis.get('statement')}"
        )
    lines.extend(
        [
            "",
            "Differential",
            "------------",
        ]
    )
    for diagnosis in verdict.get("differential_diagnoses") or []:
        lines.append(
            f"[{diagnosis.get('confidence')}] {diagnosis.get('statement')}"
        )
    lines.extend(
        [
            "",
            reference,
            "",
            "Evidence verification",
            "---------------------",
            (
                f"passed={result.get('verified')}  "
                f"claims={(result.get('verification') or {}).get('n_claims')}  "
                f"citations={(result.get('verification') or {}).get('n_citations')}"
            ),
            "",
            "CAUTION",
            "-------",
            "Evidence-verified does not mean clinically confirmed.",
            "Quality-flagged values are shown as lower-weight evidence.",
            "Acute ischemia calls require raw-trace/serial-ECG review.",
        ]
    )
    return "\n".join(textwrap.fill(line, width=52) if line else "" for line in lines)


def plot_overview(
    record_id: str,
    signal: np.ndarray,
    fs: float,
    features: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> None:
    duration = signal.shape[0] / fs
    time = np.arange(signal.shape[0]) / fs
    finite = np.abs(signal[np.isfinite(signal)])
    ylim = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.5
    ylim = max(1.0, min(5.0, np.ceil(ylim * 2.0) / 2.0))

    figure = plt.figure(figsize=(22, 17), constrained_layout=True)
    grid = figure.add_gridspec(12, 2, width_ratios=(3.4, 1.25), wspace=0.08)
    beats = features.get("beats") or []
    focus_beat = int(config["focus_beat"])

    for lead_index, lead in enumerate(LEADS):
        axis = figure.add_subplot(grid[lead_index, 0])
        _grid(axis, duration, ylim)
        axis.plot(time, signal[:, lead_index], color="#111111", linewidth=0.65)
        axis.set_ylabel(
            lead + _quality_suffix(features, lead),
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        if lead_index != len(LEADS) - 1:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel("seconds")

        for beat in beats:
            sample = int(beat.get("r_index") or 0)
            if not 0 <= sample < signal.shape[0]:
                continue
            beat_id = int(beat.get("beat_id") or 0)
            group = int(beat.get("group_id") or 0)
            color = GROUP_COLORS[(max(group, 1) - 1) % len(GROUP_COLORS)]
            marker = "*" if beat.get("paced") else "o"
            size = 38 if beat.get("paced") else 12
            axis.scatter(
                sample / fs,
                signal[sample, lead_index],
                marker=marker,
                s=size,
                color="#D62728" if beat.get("paced") else color,
                zorder=5,
            )
            if beat_id == focus_beat:
                axis.axvline(sample / fs, color="#F0A202", linewidth=1.0, alpha=0.9)

    text_axis = figure.add_subplot(grid[:, 1])
    text_axis.axis("off")
    text_axis.text(
        0,
        1,
        _diagnosis_text(result, str(config["reference"])),
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.22,
        family="DejaVu Sans",
    )
    figure.suptitle(
        f"{record_id} — raw 12-lead ECG with ecgfeat beat/morphology overlay\n"
        "red star = paced; colored dot = QRS morphology group; gold line = focus beat",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _feature_index(features: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row.get("lead")), int(row.get("beat_id"))): row
        for row in (features.get("beat_features") or [])
        if row.get("lead") is not None and row.get("beat_id") is not None
    }


def plot_focus(
    record_id: str,
    signal: np.ndarray,
    fs: float,
    features: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> None:
    beat_id = int(config["focus_beat"])
    beat = next(row for row in (features.get("beats") or []) if int(row["beat_id"]) == beat_id)
    r_index = int(beat["r_index"])
    start = max(0, r_index - int(0.65 * fs))
    stop = min(signal.shape[0], r_index + int(0.95 * fs))
    time = (np.arange(start, stop) - r_index) / fs
    by_feature = _feature_index(features)

    figure, axes = plt.subplots(
        len(config["focus_leads"]),
        1,
        figsize=(15, 10),
        sharex=True,
        constrained_layout=True,
    )
    for axis, lead in zip(axes, config["focus_leads"]):
        lead_index = LEADS.index(lead)
        values = signal[start:stop, lead_index]
        local_limit = max(0.75, float(np.nanpercentile(np.abs(values), 99.5)) * 1.15)
        axis.plot(time, values, color="#111111", linewidth=1.15)
        axis.axvline(0, color="#0072B2", linewidth=0.8, linestyle="--", label="R")
        axis.axhline(0, color="#777777", linewidth=0.55)
        axis.set_ylim(-local_limit, local_limit)
        axis.set_ylabel(lead, rotation=0, ha="right", fontsize=11, fontweight="bold")
        axis.grid(color="#efb5b5", linewidth=0.45, alpha=0.65)
        row = by_feature.get((lead, beat_id)) or {}
        j_index = row.get("j_index") or row.get("st_hybrid_j_index")
        if isinstance(j_index, int) and start <= j_index < stop:
            axis.scatter(
                [(j_index - r_index) / fs],
                [signal[j_index, lead_index]],
                color="#D62728",
                s=32,
                zorder=6,
                label="ecgfeat J",
            )
        st_value = row.get("st_on_mv")
        qrs_value = row.get("qrs_ms")
        annotation = (
            f"ecgfeat: ST_on={st_value:.3f} mV"
            if isinstance(st_value, (int, float))
            else "ecgfeat: ST_on unavailable"
        )
        if isinstance(qrs_value, (int, float)):
            annotation += f" | QRS={qrs_value:.0f} ms"
        axis.text(
            0.995,
            0.92,
            annotation,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )
    axes[-1].set_xlabel("seconds relative to R peak")
    figure.suptitle(
        f"{record_id}, beat {beat_id} — {config['focus_title']}\n"
        f"group={beat.get('group_id')}, paced={beat.get('paced')}; "
        "red point is ecgfeat's J location",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_id", choices=sorted(CASE_CONFIG))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    record_id = args.record_id
    signal, fs, features, result = _load(
        root / "data" / "ptb-xl" / "05000" / record_id,
        root / "ptbxl_05000_ecgagent" / "features" / f"{record_id}_features.json",
        root
        / "ptbxl_05000_ecgagent"
        / "case_studies"
        / "diagnostic_v1"
        / f"{record_id}.json",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(
        record_id,
        signal,
        fs,
        features,
        result,
        CASE_CONFIG[record_id],
        args.output_dir / f"{record_id}_diagnostic_overlay.png",
    )
    plot_focus(
        record_id,
        signal,
        fs,
        features,
        result,
        CASE_CONFIG[record_id],
        args.output_dir / f"{record_id}_focus_beat.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

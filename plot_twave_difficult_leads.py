from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_annotations import get_gt_for_lead, load_record
from feature_extraction.ecgfeat.api import ECGFeatureExtractor
from feature_extraction.ecgfeat.models import LeadBeatFeatures, STANDARD_12_LEADS
from feature_extraction.ecgfeat.preprocess import analysis_signal, resample_ecg


DEFAULT_RECORDS = ("24", "57", "73", "81", "125", "133")
DEFAULT_LEADS = ("II", "V2", "V3", "V4", "V5")
FS_INTERNAL = 500


@dataclass(frozen=True)
class DifficultBeat:
    record: str
    lead: str
    feature: LeadBeatFeatures
    ground_truth: dict[str, int | None]
    r_index: int
    offset_error_ms: float


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _finite_index(value: object) -> int | None:
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return index if index >= 0 else None


def _matched_gt(
    feature: LeadBeatFeatures,
    gt_beats: Iterable[dict[str, int | None]],
    *,
    r_index: int,
    fs: int,
) -> dict[str, int | None] | None:
    beats = list(gt_beats)
    if not beats:
        return None
    distances = [
        abs(int(beat["r_sample"]) - int(r_index))
        if beat.get("r_sample") is not None
        else np.inf
        for beat in beats
    ]
    best_index = int(np.argmin(distances))
    return beats[best_index] if distances[best_index] <= int(round(0.075 * fs)) else None


def _worst_t_offset_beat(
    record: str,
    lead: str,
    beat_features: list[LeadBeatFeatures],
    r_locs: np.ndarray,
    *,
    fs: int,
) -> DifficultBeat | None:
    gt_beats = get_gt_for_lead(record, lead)
    matches: list[DifficultBeat] = []
    for feature in beat_features:
        if feature.lead != lead or not (0 <= int(feature.beat_id) < len(r_locs)):
            continue
        detected_offset = _finite_index(feature.t.offset)
        if detected_offset is None:
            continue
        r_index = int(r_locs[int(feature.beat_id)])
        ground_truth = _matched_gt(
            feature,
            gt_beats,
            r_index=r_index,
            fs=fs,
        )
        if ground_truth is None:
            continue
        gt_offset = _finite_index(ground_truth.get("t_off"))
        if gt_offset is None:
            continue
        matches.append(
            DifficultBeat(
                record=record,
                lead=lead,
                feature=feature,
                ground_truth=ground_truth,
                r_index=r_index,
                offset_error_ms=(detected_offset - gt_offset) * 1000.0 / fs,
            )
        )
    return max(matches, key=lambda item: abs(item.offset_error_ms), default=None)


def _relative_ms(index: int | None, *, r_index: int, fs: int) -> float | None:
    return None if index is None else (int(index) - int(r_index)) * 1000.0 / fs


def _draw_boundary(
    axis: plt.Axes,
    index: int | None,
    *,
    r_index: int,
    fs: int,
    color: str,
    label: str,
    linestyle: str,
    linewidth: float = 1.3,
    alpha: float = 0.95,
) -> None:
    x = _relative_ms(index, r_index=r_index, fs=fs)
    if x is not None:
        axis.axvline(
            x,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )


def plot_difficult_beat(
    axis: plt.Axes,
    signal: np.ndarray,
    difficult: DifficultBeat,
    *,
    fs: int,
) -> None:
    feature = difficult.feature
    ground_truth = difficult.ground_truth
    r_index = difficult.r_index

    indices = [
        _finite_index(ground_truth.get(key))
        for key in ("qrs_on", "qrs_off", "t_on", "t_peak", "t_off")
    ] + [
        _finite_index(value)
        for value in (
            feature.qrs.onset,
            feature.qrs.offset,
            feature.t.onset,
            feature.t.peak,
            feature.t.offset,
            feature.t_wavelet_onset_index,
            feature.t_wavelet_offset_index,
            feature.t_trapezium_offset_index,
            feature.t_offset_robust_center_index,
            feature.t_offset_latest_p85_index,
            feature.t_rms_offset_index,
            feature.t_pc1_offset_index,
        )
    ]
    finite = [index for index in indices if index is not None]
    lo = max(0, min(finite + [r_index]) - int(round(0.100 * fs)))
    hi = min(len(signal) - 1, max(finite + [r_index]) + int(round(0.120 * fs)))
    if hi - lo > int(round(1.050 * fs)):
        lo = max(0, r_index - int(round(0.180 * fs)))
        hi = min(len(signal) - 1, r_index + int(round(0.750 * fs)))

    samples = np.arange(lo, hi + 1)
    times_ms = (samples - r_index) * 1000.0 / fs
    axis.plot(times_ms, signal[lo : hi + 1], color="#202124", linewidth=1.05)
    axis.axvline(0.0, color="#9aa0a6", linestyle=":", linewidth=1.0, label="R")

    _draw_boundary(
        axis,
        _finite_index(ground_truth.get("t_on")),
        r_index=r_index,
        fs=fs,
        color="#188038",
        label="GT Ton",
        linestyle="-",
        linewidth=1.6,
    )
    _draw_boundary(
        axis,
        _finite_index(ground_truth.get("t_off")),
        r_index=r_index,
        fs=fs,
        color="#0d652d",
        label="GT Toff",
        linestyle="-",
        linewidth=2.0,
    )
    _draw_boundary(
        axis,
        _finite_index(feature.t.onset),
        r_index=r_index,
        fs=fs,
        color="#1a73e8",
        label="ecgfeat Ton",
        linestyle="--",
        linewidth=1.6,
    )
    _draw_boundary(
        axis,
        _finite_index(feature.t.offset),
        r_index=r_index,
        fs=fs,
        color="#174ea6",
        label="ecgfeat Toff",
        linestyle="--",
        linewidth=2.0,
    )

    peak_gt = _finite_index(ground_truth.get("t_peak"))
    peak_det = _finite_index(feature.t.peak)
    for index, color, marker, label in (
        (peak_gt, "#188038", "o", "GT T peak"),
        (peak_det, "#1a73e8", "x", "ecgfeat T peak"),
    ):
        if index is not None and 0 <= index < len(signal):
            axis.scatter(
                [_relative_ms(index, r_index=r_index, fs=fs)],
                [signal[index]],
                color=color,
                marker=marker,
                s=34,
                zorder=5,
                label=label,
            )

    candidate_specs = (
        (feature.t_wavelet_offset_index, "#f9ab00", "-.", "Mallat Toff"),
        (feature.t_trapezium_offset_index, "#a142f4", "-.", "Trapezium Toff"),
        (feature.t_offset_robust_center_index, "#d93025", ":", "Fusion center"),
        (feature.t_offset_latest_p85_index, "#b31412", "--", "Fusion P85"),
        (feature.t_rms_offset_index, "#00acc1", ":", "RMS Toff"),
        (feature.t_pc1_offset_index, "#e8710a", ":", "PC1 Toff"),
    )
    for index, color, linestyle, label in candidate_specs:
        _draw_boundary(
            axis,
            _finite_index(index),
            r_index=r_index,
            fs=fs,
            color=color,
            label=label,
            linestyle=linestyle,
            linewidth=1.05,
            alpha=0.85,
        )

    sqi = feature.t_sqi_score
    mad = feature.t_offset_fusion_mad_ms
    sqi_text = "N/A" if sqi is None else f"{float(sqi):.2f}"
    mad_text = "N/A" if mad is None else f"{float(mad):.1f} ms"
    axis.set_title(
        f"Record {difficult.record} · {difficult.lead} · "
        f"worst |Toff error|={abs(difficult.offset_error_ms):.1f} ms "
        f"(signed {difficult.offset_error_ms:+.1f}) · SQI={sqi_text} · MAD={mad_text}",
        fontsize=9,
        loc="left",
    )
    axis.set_ylabel("mV")
    axis.grid(alpha=0.18)


def _deduplicated_legend(axis: plt.Axes, *, columns: int = 4) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axis.legend(
        unique.values(),
        unique.keys(),
        loc="upper right",
        fontsize=6.5,
        ncol=columns,
        framealpha=0.92,
    )


def generate_plots(
    records: tuple[str, ...],
    leads: tuple[str, ...],
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = ECGFeatureExtractor(fs_internal=FS_INTERNAL, mains_freq=50)
    rows: list[dict[str, object]] = []

    for record in records:
        ecg, fs_input = load_record(record)
        features = extractor.extract(ecg, float(fs_input))
        ecg_run = resample_ecg(ecg, fs_input, FS_INTERNAL)
        ecg_measure = analysis_signal(ecg_run, FS_INTERNAL, mains_hz=50)
        r_locs = np.asarray([beat.r_index for beat in features.beats], dtype=int)
        difficult_by_lead: dict[str, DifficultBeat] = {}

        for lead in leads:
            difficult = _worst_t_offset_beat(
                record,
                lead,
                features.beat_features,
                r_locs,
                fs=FS_INTERNAL,
            )
            if difficult is None:
                continue
            difficult_by_lead[lead] = difficult
            lead_index = STANDARD_12_LEADS.index(lead)
            fig, axis = plt.subplots(figsize=(13, 4.2))
            plot_difficult_beat(
                axis,
                ecg_measure[lead_index],
                difficult,
                fs=FS_INTERNAL,
            )
            axis.set_xlabel("Time relative to global R (ms)")
            _deduplicated_legend(axis)
            fig.tight_layout()
            image_path = output_dir / f"record_{record}_{lead}_worst_t_offset.png"
            fig.savefig(image_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            feature = difficult.feature
            rows.append(
                {
                    "record": record,
                    "lead": lead,
                    "beat_id": feature.beat_id,
                    "offset_error_ms": difficult.offset_error_ms,
                    "t_sqi_score": feature.t_sqi_score,
                    "fusion_mad_ms": feature.t_offset_fusion_mad_ms,
                    "fusion_support": feature.t_offset_fusion_support,
                    "image": image_path.name,
                }
            )

        if difficult_by_lead:
            fig, axes = plt.subplots(
                len(difficult_by_lead),
                1,
                figsize=(16, 3.7 * len(difficult_by_lead)),
                squeeze=False,
            )
            for axis, lead in zip(axes[:, 0], difficult_by_lead):
                lead_index = STANDARD_12_LEADS.index(lead)
                plot_difficult_beat(
                    axis,
                    ecg_measure[lead_index],
                    difficult_by_lead[lead],
                    fs=FS_INTERNAL,
                )
                axis.set_xlabel("Time relative to global R (ms)")
            _deduplicated_legend(axes[0, 0], columns=5)
            fig.suptitle(
                f"LUDB record {record}: difficult single-lead T offsets",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            fig.savefig(
                output_dir / f"record_{record}_five_lead_summary.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    with (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "record",
            "lead",
            "beat_id",
            "offset_error_ms",
            "t_sqi_score",
            "fusion_mad_ms",
            "fusion_support",
            "image",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the worst matched T-offset beat for difficult LUDB leads."
    )
    parser.add_argument("--records", default=",".join(DEFAULT_RECORDS))
    parser.add_argument("--leads", default=",".join(DEFAULT_LEADS))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("twave_difficult_lead_plots"),
    )
    args = parser.parse_args()
    records = _parse_csv(args.records)
    leads = _parse_csv(args.leads)
    invalid_leads = [lead for lead in leads if lead not in STANDARD_12_LEADS]
    if invalid_leads:
        parser.error(f"invalid leads: {', '.join(invalid_leads)}")
    rows = generate_plots(records, leads, args.out_dir)
    print(
        f"Generated {len(rows)} single-lead plots and "
        f"{len(set(row['record'] for row in rows))} record summaries in "
        f"{args.out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()

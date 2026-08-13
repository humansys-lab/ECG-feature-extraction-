#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from medgemma_ecg_core import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_MAX_LEN,
    build_context_summary_from_paths,
    build_prompt,
    canonical_label_for_dir,
    parse_medgemma_output,
    run_layered_diagnosis,
)
from medgemma_runtime import build_generate_text_fn as _build_generate_text_fn
from medgemma_runtime import infer_runtime_config


@dataclass(frozen=True)
class SampleJob:
    disease_dir_label: str
    split_name: str
    sample_id: str
    features_path: Path
    report_path: Path | None
    local_output_dir: Path
    mirror_output_dir: Path


MODEL_PATH = "./medgemma-27b"
MODEL_MAX_LEN = DEFAULT_MODEL_MAX_LEN
MODEL_MAX_OUTPUT_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS


def build_generate_text_fn(model_path: str = MODEL_PATH) -> Callable[[str], str]:
    """Compatibility wrapper around the shared MedGemma runtime factory."""
    return _build_generate_text_fn(
        model_path,
        model_max_len=MODEL_MAX_LEN,
        max_output_tokens=MODEL_MAX_OUTPUT_TOKENS,
    )


def iter_samples(input_root: Path, splits: Sequence[str]) -> Iterable[SampleJob]:
    for disease_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for split_name in splits:
            split_dir = disease_dir / split_name
            if not split_dir.exists():
                continue
            for features_path in sorted(split_dir.glob("*_features.json")):
                sample_id = features_path.stem.replace("_features", "")
                report_path = split_dir / f"{sample_id}_report.txt"
                yield SampleJob(
                    disease_dir_label=disease_dir.name,
                    split_name=split_name,
                    sample_id=sample_id,
                    features_path=features_path,
                    report_path=report_path if report_path.exists() else None,
                    local_output_dir=split_dir,
                    mirror_output_dir=input_root.parent / "all_diseases_medgemma" / disease_dir.name / split_name,
                )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def process_sample(
    sample: SampleJob,
    language: str,
    clinician_notes: str,
    generate_text_fn: Callable[[str], str],
) -> dict[str, Any]:
    sanitized_context = build_context_summary_from_paths(
        sample.features_path,
        sample.report_path,
        clinician_notes,
        language,
    )
    prompt = build_prompt(sanitized_context, language)
    raw_model_output = generate_text_fn(prompt)
    parsed = parse_medgemma_output(raw_model_output)
    result = {
        "sample_id": sample.sample_id,
        "split": sample.split_name,
        "ground_truth_dir_label": sample.disease_dir_label,
        "ground_truth_normalized_label": canonical_label_for_dir(sample.disease_dir_label),
        "relative_features_path": str(sample.features_path),
        "relative_report_path": str(sample.report_path) if sample.report_path is not None else None,
        "sanitized_context": sanitized_context,
        "prompt": prompt,
        "raw_model_output": raw_model_output,
        "parsed": parsed,
        "parse_status": "ok" if parsed.get("top1_raw") else "missing_top1",
        "errors": [],
    }

    local_txt = sample.local_output_dir / f"{sample.sample_id}_medgemma.txt"
    local_json = sample.local_output_dir / f"{sample.sample_id}_medgemma.json"
    mirror_txt = sample.mirror_output_dir / f"{sample.sample_id}_medgemma.txt"
    mirror_json = sample.mirror_output_dir / f"{sample.sample_id}_medgemma.json"

    _write_text(local_txt, raw_model_output)
    _write_json(local_json, result)
    _write_text(mirror_txt, raw_model_output)
    _write_json(mirror_json, result)
    return result


def process_sample_layered(
    sample: SampleJob,
    language: str,
    clinician_notes: str,
    generate_text_fn: Callable[[str], str],
) -> dict[str, Any]:
    layered = run_layered_diagnosis(
        sample.features_path,
        sample.report_path,
        clinician_notes,
        language,
        generate_text_fn,
    )
    parsed = layered["parsed"]
    result = {
        "sample_id": sample.sample_id,
        "split": sample.split_name,
        "ground_truth_dir_label": sample.disease_dir_label,
        "ground_truth_normalized_label": canonical_label_for_dir(sample.disease_dir_label),
        "relative_features_path": str(sample.features_path),
        "relative_report_path": str(sample.report_path) if sample.report_path is not None else None,
        "prompt": layered["prompt"],
        "raw_model_output": layered["raw_model_output"],
        "parsed": parsed,
        "guardrail_notes": layered["guardrail_notes"],
        "unaddressed_guardrail_notes": layered["unaddressed_guardrail_notes"],
        "revised": layered["revised"],
        "reference_only": layered["reference_only"],
        "reference_agreement": layered["reference_agreement"],
        "parse_status": "ok" if parsed.get("top1_raw") else "missing_top1",
        "errors": [],
    }

    local_txt = sample.local_output_dir / f"{sample.sample_id}_medgemma_layered.txt"
    local_json = sample.local_output_dir / f"{sample.sample_id}_medgemma_layered.json"
    mirror_txt = sample.mirror_output_dir / f"{sample.sample_id}_medgemma_layered.txt"
    mirror_json = sample.mirror_output_dir / f"{sample.sample_id}_medgemma_layered.json"

    _write_text(local_txt, layered["raw_model_output"])
    _write_json(local_json, result)
    _write_text(mirror_txt, layered["raw_model_output"])
    _write_json(mirror_json, result)
    return result


def _top3_hit(row: dict[str, Any]) -> bool:
    truth = row["ground_truth_normalized_label"]
    candidates = [row.get("top1_normalized")] + [item for item in row.get("similar_normalized", []) if item]
    return truth in candidates


def write_summary_outputs(rows: list[dict[str, Any]], summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "ground_truth_dir_label",
                "ground_truth_normalized_label",
                "top1_raw",
                "top1_normalized",
                "similar_raw",
                "similar_normalized",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "ground_truth_dir_label": row["ground_truth_dir_label"],
                    "ground_truth_normalized_label": row["ground_truth_normalized_label"],
                    "top1_raw": row["top1_raw"],
                    "top1_normalized": row["top1_normalized"],
                    "similar_raw": json.dumps(row["similar_raw"], ensure_ascii=False),
                    "similar_normalized": json.dumps(row["similar_normalized"], ensure_ascii=False),
                }
            )

    raw_counts: dict[str, Any] = {"top1": {}, "similar": {}, "by_truth": {}}
    combined_top1 = Counter(row["top1_raw"] for row in rows if row.get("top1_raw"))
    combined_similar = Counter(item for row in rows for item in row.get("similar_raw", []) if item)
    raw_counts["top1"]["combined"] = dict(combined_top1)
    raw_counts["similar"]["combined"] = dict(combined_similar)
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        raw_counts["top1"][split] = dict(Counter(row["top1_raw"] for row in split_rows if row.get("top1_raw")))
        raw_counts["similar"][split] = dict(
            Counter(item for row in split_rows for item in row.get("similar_raw", []) if item)
        )
        truth_counts: dict[str, Any] = {}
        for truth in sorted({row["ground_truth_dir_label"] for row in split_rows}):
            truth_rows = [row for row in split_rows if row["ground_truth_dir_label"] == truth]
            truth_counts[truth] = {
                "top1": dict(Counter(row["top1_raw"] for row in truth_rows if row.get("top1_raw"))),
                "similar": dict(
                    Counter(item for row in truth_rows for item in row.get("similar_raw", []) if item)
                ),
            }
        raw_counts["by_truth"][split] = truth_counts
    _write_json(summary_dir / "raw_counts.json", raw_counts)

    by_split = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        total = len(split_rows)
        top1_correct = sum(
            1 for row in split_rows if row["ground_truth_normalized_label"] == row.get("top1_normalized")
        )
        top3_correct = sum(1 for row in split_rows if _top3_hit(row))
        by_split[split] = {
            "count": total,
            "top1_accuracy": (top1_correct / total) if total else 0.0,
            "top3_hit_rate": (top3_correct / total) if total else 0.0,
            "unknown_top1_count": sum(1 for row in split_rows if row.get("top1_normalized") is None),
        }
    total = len(rows)
    total_top1_correct = sum(1 for row in rows if row["ground_truth_normalized_label"] == row.get("top1_normalized"))
    total_top3_correct = sum(1 for row in rows if _top3_hit(row))
    by_truth = {}
    for truth in sorted({row["ground_truth_normalized_label"] for row in rows if row.get("ground_truth_normalized_label")}):
        truth_rows = [row for row in rows if row.get("ground_truth_normalized_label") == truth]
        truth_total = len(truth_rows)
        by_truth[truth] = {
            "count": truth_total,
            "top1_accuracy": (
                sum(1 for row in truth_rows if row.get("top1_normalized") == truth) / truth_total
            )
            if truth_total
            else 0.0,
            "top3_hit_rate": (sum(1 for row in truth_rows if _top3_hit(row)) / truth_total) if truth_total else 0.0,
            "unknown_top1_count": sum(1 for row in truth_rows if row.get("top1_normalized") is None),
        }
    _write_json(
        summary_dir / "normalized_metrics.json",
        {
            "combined": {
                "count": total,
                "top1_accuracy": (total_top1_correct / total) if total else 0.0,
                "top3_hit_rate": (total_top3_correct / total) if total else 0.0,
                "unknown_top1_count": sum(1 for row in rows if row.get("top1_normalized") is None),
            },
            "by_split": by_split,
            "by_truth": by_truth,
        },
    )

    labels = sorted(
        {row["ground_truth_normalized_label"] for row in rows if row.get("ground_truth_normalized_label")}
        | {row["top1_normalized"] for row in rows if row.get("top1_normalized")}
    )
    with (summary_dir / "confusion_top1.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["truth/pred"] + labels)
        for truth in labels:
            writer.writerow(
                [truth]
                + [
                    sum(
                        1
                        for row in rows
                        if row.get("ground_truth_normalized_label") == truth and row.get("top1_normalized") == pred
                    )
                    for pred in labels
                ]
            )

    with (summary_dir / "confusion_top3.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["truth/pred"] + labels)
        for truth in labels:
            writer.writerow(
                [truth]
                + [
                    sum(
                        1
                        for row in rows
                        if row.get("ground_truth_normalized_label") == truth
                        and pred in ([row.get("top1_normalized")] + [item for item in row.get("similar_normalized", []) if item])
                    )
                    for pred in labels
                ]
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch MedGemma diagnostics over all_diseases_ecgfeat.")
    parser.add_argument("--input-dir", type=Path, default=Path("all_diseases_ecgfeat"))
    parser.add_argument("--mirror-output-dir", type=Path, default=Path("all_diseases_medgemma"))
    parser.add_argument("--splits", type=str, default="generated,real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--method",
        type=str,
        default="legacy",
        choices=["legacy", "layered"],
        help=(
            "legacy: single-shot prompt (build_prompt). "
            "layered: bottom-up L0-L5 chain-of-thought with guardrail-triggered revision "
            "(run_layered_diagnosis); withholds unified-clinical-rule conclusions "
            "from the reasoning prompt entirely."
        ),
    )
    return parser


def parse_splits(raw: str) -> tuple[str, ...]:
    parts = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not parts:
        raise ValueError("at least one split must be provided")
    return parts


def run_batch(args: argparse.Namespace) -> int:
    splits = parse_splits(args.splits)
    generate_text_fn = build_generate_text_fn(args.model_path)
    rows: list[dict[str, Any]] = []
    sample_count = 0

    output_suffix = "_medgemma_layered" if args.method == "layered" else "_medgemma"
    process_fn = process_sample_layered if args.method == "layered" else process_sample

    for sample in iter_samples(args.input_dir, splits):
        if args.limit is not None and sample_count >= args.limit:
            break
        if args.skip_existing and (sample.local_output_dir / f"{sample.sample_id}{output_suffix}.json").exists():
            continue

        concrete_sample = SampleJob(
            disease_dir_label=sample.disease_dir_label,
            split_name=sample.split_name,
            sample_id=sample.sample_id,
            features_path=sample.features_path,
            report_path=sample.report_path,
            local_output_dir=sample.local_output_dir,
            mirror_output_dir=args.mirror_output_dir / sample.disease_dir_label / sample.split_name,
        )
        try:
            result = process_fn(concrete_sample, args.language, "", generate_text_fn)
        except Exception as exc:
            error_payload = {
                "sample_id": concrete_sample.sample_id,
                "split": concrete_sample.split_name,
                "ground_truth_dir_label": concrete_sample.disease_dir_label,
                "ground_truth_normalized_label": canonical_label_for_dir(concrete_sample.disease_dir_label),
                "top1_raw": None,
                "top1_normalized": None,
                "similar_raw": [],
                "similar_normalized": [],
                "error": str(exc),
            }
            error_json_payload = {
                "sample_id": concrete_sample.sample_id,
                "split": concrete_sample.split_name,
                "ground_truth_dir_label": concrete_sample.disease_dir_label,
                "ground_truth_normalized_label": canonical_label_for_dir(concrete_sample.disease_dir_label),
                "sanitized_context": "",
                "prompt": "",
                "raw_model_output": "",
                "parsed": {},
                "parse_status": "error",
                "errors": [str(exc)],
            }
            _write_json(
                concrete_sample.local_output_dir / f"{concrete_sample.sample_id}{output_suffix}.json",
                error_json_payload,
            )
            _write_json(
                args.mirror_output_dir
                / concrete_sample.disease_dir_label
                / concrete_sample.split_name
                / f"{concrete_sample.sample_id}{output_suffix}.json",
                error_json_payload,
            )
            rows.append(error_payload)
            sample_count += 1
            continue

        rows.append(
            {
                "sample_id": result["sample_id"],
                "split": result["split"],
                "ground_truth_dir_label": result["ground_truth_dir_label"],
                "ground_truth_normalized_label": result["ground_truth_normalized_label"],
                "top1_raw": result["parsed"]["top1_raw"],
                "top1_normalized": result["parsed"]["top1_normalized"],
                "similar_raw": result["parsed"]["similar_raw"],
                "similar_normalized": result["parsed"]["similar_normalized"],
            }
        )
        sample_count += 1

    write_summary_outputs(rows, args.mirror_output_dir / "summary")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

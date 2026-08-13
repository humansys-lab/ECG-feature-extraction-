#!/usr/bin/env python3
"""Run diagnosis-free JSON+report sequential ECG reasoning over a flat directory
of `<id>_features.json` / `<id>_report.txt` pairs (no disease/split
subfolders) — the layout used by data/WFDBRecords_03_030_ecgfeat_out.

Example:
    python3 run_layered_batch_flat.py \
        --input-dir data/WFDBRecords_03_030_ecgfeat_out \
        --output-dir data/WFDBRecords_03_030_medgemma_layered \
        --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from medgemma_ecg_core import json_dumps, run_layered_diagnosis
from medgemma_runtime import build_generate_text_fn, infer_runtime_config

MODEL_PATH = "./medgemma-27b"


def iter_pairs(input_dir: Path) -> list[tuple[str, Path, Path | None]]:
    pairs = []
    for features_path in sorted(input_dir.glob("*_features.json")):
        sample_id = features_path.stem.replace("_features", "")
        report_path = input_dir / f"{sample_id}_report.txt"
        pairs.append((sample_id, features_path, report_path if report_path.exists() else None))
    return pairs


def _summary_row(sample_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "reasoning_mode": result["reasoning_mode"],
        "dx_code_used_in_reasoning": result["dx_code_used_in_reasoning"],
        "report_used_in_reasoning": result["supplementary_report_used_in_reasoning"],
        "top1_raw": result["parsed"].get("top1_raw"),
        "top1_normalized": result["parsed"].get("top1_normalized"),
        "revised": result["revised"],
        "guardrail_notes_count": len(result["guardrail_notes"]),
        "unaddressed_guardrail_notes_count": len(result["unaddressed_guardrail_notes"]),
        "synthesis_guardrail_notes_count": len(result["synthesis_guardrail_notes"]),
        "reference_agreement": result["reference_agreement"],
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", type=str, default="en", choices=["en", "ja"])
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Process only this sample ID; repeat the option for multiple IDs.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_text_fn = build_generate_text_fn(args.model_path)

    summary_rows: list[dict[str, Any]] = []
    processed = 0
    selected_ids = set(args.sample_id)
    pairs = iter_pairs(args.input_dir)
    if selected_ids:
        pairs = [pair for pair in pairs if pair[0] in selected_ids]
        missing_ids = selected_ids - {pair[0] for pair in pairs}
        if missing_ids:
            parser.error(f"sample IDs not found: {', '.join(sorted(missing_ids))}")

    for sample_id, features_path, report_path in pairs:
        if args.limit is not None and processed >= args.limit:
            break
        out_txt = args.output_dir / f"{sample_id}_medgemma_layered.txt"
        out_json = args.output_dir / f"{sample_id}_medgemma_layered.json"
        if args.skip_existing and out_json.exists():
            continue

        print(f"[{processed + 1}] {sample_id} ...", flush=True)
        try:
            result = run_layered_diagnosis(
                features_path, report_path, "", args.language, generate_text_fn
            )
            out_txt.write_text(result["raw_model_output"], encoding="utf-8")
            out_json.write_text(json_dumps(result), encoding="utf-8")
            summary_rows.append(_summary_row(sample_id, result))
        except Exception as exc:  # keep the batch going on a single bad sample
            print(f"  [error] {sample_id}: {exc}")
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "reasoning_mode": None,
                    "dx_code_used_in_reasoning": None,
                    "report_used_in_reasoning": None,
                    "top1_raw": None,
                    "top1_normalized": None,
                    "revised": None,
                    "guardrail_notes_count": None,
                    "unaddressed_guardrail_notes_count": None,
                    "synthesis_guardrail_notes_count": None,
                    "reference_agreement": None,
                    "error": str(exc),
                }
            )
        processed += 1

    # Preserve a complete directory-level summary after a selected-ID rerun.
    current_ids = {row["sample_id"] for row in summary_rows}
    for out_json in sorted(args.output_dir.glob("*_medgemma_layered.json")):
        sample_id = out_json.name.removesuffix("_medgemma_layered.json")
        if sample_id in current_ids:
            continue
        try:
            result = json.loads(out_json.read_text(encoding="utf-8"))
            summary_rows.append(_summary_row(sample_id, result))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    summary_rows.sort(key=lambda row: row["sample_id"])

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "reasoning_mode",
                "dx_code_used_in_reasoning",
                "report_used_in_reasoning",
                "top1_raw",
                "top1_normalized",
                "revised",
                "guardrail_notes_count",
                "unaddressed_guardrail_notes_count",
                "synthesis_guardrail_notes_count",
                "reference_agreement",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. {processed} samples processed. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the layered chain-of-thought diagnosis pipeline on one features/report pair.

Example:
    python3 run_layered_single.py \
        --features medgemma_single_input/js00010/real/JS00010_features.json \
        --report medgemma_single_input/js00010/real/JS00010_report.txt \
        --output-dir medgemma_single_output/js00010/layered
"""
from __future__ import annotations

import argparse
from pathlib import Path

from medgemma_ecg_core import json_dumps, run_layered_diagnosis
from medgemma_runtime import build_generate_text_fn, infer_runtime_config

MODEL_PATH = "./medgemma-27b"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--language", type=str, default="en", choices=["en", "ja"])
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    generate_text_fn = build_generate_text_fn(args.model_path)
    result = run_layered_diagnosis(
        args.features,
        args.report,
        args.notes,
        args.language,
        generate_text_fn,
    )

    print(result["raw_model_output"])
    print("\n--- guardrail notes ---")
    print(result["guardrail_notes"] or "(none triggered)")
    print("revised:", result["revised"])
    print("reference_agreement:", result["reference_agreement"])

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.features.stem.replace("_features", "")
        (args.output_dir / f"{stem}_medgemma_layered.txt").write_text(
            result["raw_model_output"], encoding="utf-8"
        )
        (args.output_dir / f"{stem}_medgemma_layered.json").write_text(
            json_dumps(result), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_EXTRACTION_ROOT = PROJECT_ROOT / "feature_extraction"


@dataclass(frozen=True)
class BatchJob:
    disease_label: str
    source_name: str
    input_path: Path
    output_dir: Path
    metadata_path: Path | None = None


@dataclass
class PatientMetaStub:
    age: int | None = None
    sex: str | None = None


def infer_shape(data: Any) -> tuple[int, ...]:
    """Infer the shape of nested sequences, numpy arrays, or tensors."""
    if hasattr(data, "shape"):
        shape = getattr(data, "shape")
        try:
            return tuple(int(dim) for dim in shape)
        except TypeError:
            pass

    if isinstance(data, (list, tuple)):
        if not data:
            return (0,)
        return (len(data),) + infer_shape(data[0])

    return ()


def _ensure_feature_extraction_path() -> None:
    feature_path = str(FEATURE_EXTRACTION_ROOT)
    if feature_path not in sys.path:
        sys.path.insert(0, feature_path)


def _import_runtime_components():
    try:
        _ensure_feature_extraction_path()
        from ecgfeat.api import ECGFeatureExtractor
        from ecgfeat.export import prepare_json_export, to_dict
        from ecgfeat.models import PatientMeta
        from demo_feature_extraction import generate_ecg_annotated_plot, generate_report
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise RuntimeError(
            "batch extraction runtime dependencies are incomplete; "
            f"missing module: {missing}. Install the local `feature_extraction` dependencies "
            "and PyTorch in the target environment, then rerun the script."
        ) from exc

    return ECGFeatureExtractor, to_dict, PatientMeta, generate_report, generate_ecg_annotated_plot, prepare_json_export


def load_pt(path: Path) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required to read input .pt files. "
            "Install torch in the runtime environment before running this script."
        ) from exc

    return torch.load(path, map_location="cpu")


def save_pt(path: Path, payload: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required to write .pt feature outputs. "
            "Install torch in the runtime environment before running this script."
        ) from exc

    torch.save(payload, path)


def _maybe_to_numpy(data: Any) -> Any:
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        try:
            return data.numpy()
        except Exception:
            pass

    try:
        import numpy as np
    except ModuleNotFoundError:
        return data

    try:
        return np.asarray(data, dtype=float)
    except Exception:
        return data


def _transpose_2d(data: Any) -> Any:
    if hasattr(data, "transpose"):
        try:
            return data.transpose(1, 0)
        except TypeError:
            pass
    return [list(row) for row in zip(*data)]


def _transpose_batch_last_two(data: Any) -> Any:
    if hasattr(data, "transpose"):
        try:
            return data.transpose(0, 2, 1)
        except TypeError:
            pass
    return [[list(row) for row in zip(*sample)] for sample in data]


def normalize_batch_ecg(data: Any, source_path: Path) -> Any:
    """Normalize input ECG data to [n_samples, 12, n_points]."""
    normalized = _maybe_to_numpy(data)
    shape = infer_shape(normalized)

    if len(shape) == 2:
        if shape[0] == 12:
            batch = [normalized]
        elif shape[1] == 12:
            batch = [_transpose_2d(normalized)]
        else:
            raise ValueError(f"{source_path} has unsupported 2D shape {shape}; expected [12, n_points] or [n_points, 12].")
    elif len(shape) == 3:
        if shape[1] == 12:
            batch = normalized
        elif shape[2] == 12:
            batch = _transpose_batch_last_two(normalized)
        else:
            raise ValueError(
                f"{source_path} has unsupported 3D shape {shape}; expected [n_samples, 12, n_points] "
                f"or [n_samples, n_points, 12]."
            )
    else:
        raise ValueError(f"{source_path} has unsupported shape {shape}; expected a 2D or 3D ECG tensor.")

    return _maybe_to_numpy(batch)


def iter_jobs(input_dir: Path, output_dir: Path, sources: Sequence[str]) -> Iterable[BatchJob]:
    for disease_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        metadata_path = disease_dir / "metadata.pt"
        metadata_ref = metadata_path if metadata_path.exists() else None
        for source_name in sources:
            input_path = disease_dir / f"{source_name}.pt"
            if not input_path.exists():
                continue
            yield BatchJob(
                disease_label=disease_dir.name,
                source_name=source_name,
                input_path=input_path,
                output_dir=output_dir / disease_dir.name / source_name,
                metadata_path=metadata_ref,
            )


def build_sample_header(
    disease_label: str,
    record_id: str,
    source_name: str,
    sampling_rate: int,
    n_points: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = metadata or {}
    label = metadata.get("label") or disease_label.upper()
    dataset = metadata.get("dataset", "unknown")
    return {
        "record": record_id,
        "fs": int(sampling_rate),
        "n_samples": int(n_points),
        "age": metadata.get("age"),
        "sex": metadata.get("sex"),
        "dx": [str(label)],
        "rx": "Unknown",
        "hx": f"dataset={dataset}; source={source_name}; disease={disease_label}",
        "sx": f"batch_id={record_id}",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def process_job(
    job: BatchJob,
    extractor: Any,
    to_dict_fn: Callable[[Any], Any],
    report_writer: Callable[[str, dict[str, Any], Any, Any, Path], None],
    sampling_rate: int,
    annotated_plot_writer: Callable[[str, dict[str, Any], Any, Any, Path], None] | None = None,
    load_pt_fn: Callable[[Path], Any] = load_pt,
    save_pt_fn: Callable[[Path, Any], None] = save_pt,
    patient_meta_factory: Callable[..., Any] = PatientMetaStub,
    limit: int | None = None,
    skip_existing: bool = False,
    prepare_json_export_fn: Callable[..., Any] | None = None,
    include_beat_features: bool = False,
) -> dict[str, Any]:
    raw_batch = load_pt_fn(job.input_path)
    batch = normalize_batch_ecg(raw_batch, job.input_path)
    metadata = {}
    if job.metadata_path is not None and job.metadata_path.exists():
        loaded_metadata = load_pt_fn(job.metadata_path)
        if isinstance(loaded_metadata, dict):
            metadata = loaded_metadata

    job.output_dir.mkdir(parents=True, exist_ok=True)
    batch_shape = infer_shape(batch)
    manifest = {
        "disease_label": job.disease_label,
        "source_name": job.source_name,
        "input_path": str(job.input_path),
        "metadata_path": str(job.metadata_path) if job.metadata_path is not None else None,
        "sampling_rate": sampling_rate,
        "input_shape": list(batch_shape),
        "dataset": metadata.get("dataset"),
        "label": metadata.get("label", job.disease_label.upper()),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "annotated_plots": 0,
        "errors": [],
    }

    for sample_index, ecg in enumerate(batch):
        if limit is not None and sample_index >= limit:
            break

        sample_id = f"{sample_index:03d}"
        feature_json_path = job.output_dir / f"{sample_id}_features.json"
        feature_pt_path = job.output_dir / f"{sample_id}_features.pt"
        report_path = job.output_dir / f"{sample_id}_report.txt"
        annotated_plot_path = job.output_dir / f"{sample_id}_ecg_annotated.png"

        expected_paths = [feature_json_path, feature_pt_path, report_path]
        if annotated_plot_writer is not None:
            expected_paths.append(annotated_plot_path)
        if skip_existing and all(path.exists() for path in expected_paths):
            manifest["skipped"] += 1
            continue

        sample_shape = infer_shape(ecg)
        if len(sample_shape) != 2 or sample_shape[0] != 12:
            manifest["failed"] += 1
            manifest["errors"].append(f"{sample_id}: normalized sample shape {sample_shape} is not [12, n_points]")
            continue

        try:
            record_id = f"{job.disease_label}_{job.source_name}_{sample_id}"
            patient_meta = patient_meta_factory(age=metadata.get("age"), sex=metadata.get("sex"))
            header = build_sample_header(
                disease_label=job.disease_label,
                record_id=record_id,
                source_name=job.source_name,
                sampling_rate=sampling_rate,
                n_points=sample_shape[1],
                metadata=metadata,
            )
            result = extractor.extract(ecg, fs=sampling_rate, meta=patient_meta)
            feature_payload = to_dict_fn(result)
            clinical = feature_payload.get("clinical_interpretation", {})
            if isinstance(clinical, dict) and clinical.get("ruleset_version"):
                manifest["clinical_ruleset_version"] = clinical["ruleset_version"]

            json_payload = (
                prepare_json_export_fn(
                    feature_payload, include_beat_features=include_beat_features
                )
                if prepare_json_export_fn is not None
                else feature_payload
            )
            _write_json(feature_json_path, json_payload)
            save_pt_fn(feature_pt_path, feature_payload)
            report_writer(record_id, header, ecg, result, report_path)
            if annotated_plot_writer is not None:
                annotated_plot_writer(record_id, header, ecg, result, annotated_plot_path)
                manifest["annotated_plots"] += 1
            manifest["processed"] += 1
        except Exception as exc:
            manifest["failed"] += 1
            manifest["errors"].append(f"{sample_id}: {exc}")

    _write_json(job.output_dir / "manifest.json", manifest)
    return manifest


def parse_sources(raw_sources: str) -> tuple[str, ...]:
    sources = tuple(part.strip() for part in raw_sources.split(",") if part.strip())
    if not sources:
        raise ValueError("at least one source must be provided")
    return sources


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ecgfeat features from every sample inside disease-level generated.pt and real.pt batches, "
            "and save per-sample features/report artifacts into a mirrored output tree."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "all_diseases_pt100",
        help="Directory containing disease subfolders with generated.pt and real.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "all_diseases_ecgfeat",
        help="Destination directory for extracted features and reports.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=100,
        help="Sampling rate of the stored ECG tensors.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="generated,real",
        help="Comma-separated split names to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples to process per split, useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose feature JSON, feature PT, and report TXT already exist.",
    )
    parser.add_argument(
        "--annotated-plots",
        action="store_true",
        help="Also write per-sample *_ecg_annotated.png waveform/abnormality visualizations.",
    )
    parser.add_argument(
        "--include-beat-features",
        action="store_true",
        help=(
            "Keep the full per-beat/per-lead measurement detail (beat_features) in the "
            "exported feature JSON. Omitted by default to shrink the JSON artifact; the "
            ".pt feature file always keeps the full detail regardless of this flag."
        ),
    )
    return parser


def run_batch(args: argparse.Namespace) -> int:
    sources = parse_sources(args.sources)
    (
        ECGFeatureExtractor,
        to_dict_fn,
        patient_meta_factory,
        report_writer,
        annotated_plot_writer,
        prepare_json_export_fn,
    ) = _import_runtime_components()
    extractor = ECGFeatureExtractor(mains_freq=50)

    jobs = list(iter_jobs(args.input_dir, args.output_dir, sources))
    if not jobs:
        print(f"[warning] no matching source files found under {args.input_dir}")
        return 1

    total_processed = 0
    total_failed = 0
    total_skipped = 0

    for job in jobs:
        print(f"[process] {job.disease_label}/{job.source_name} -> {job.output_dir}")
        summary = process_job(
            job=job,
            extractor=extractor,
            to_dict_fn=to_dict_fn,
            report_writer=report_writer,
            sampling_rate=args.sampling_rate,
            annotated_plot_writer=annotated_plot_writer if args.annotated_plots else None,
            load_pt_fn=load_pt,
            save_pt_fn=save_pt,
            patient_meta_factory=patient_meta_factory,
            limit=args.limit,
            skip_existing=args.skip_existing,
            prepare_json_export_fn=prepare_json_export_fn,
            include_beat_features=args.include_beat_features,
        )
        total_processed += int(summary["processed"])
        total_failed += int(summary["failed"])
        total_skipped += int(summary["skipped"])
        print(
            f"[done] {job.disease_label}/{job.source_name}: "
            f"processed={summary['processed']} skipped={summary['skipped']} failed={summary['failed']}"
        )

    print(
        f"[summary] jobs={len(jobs)} processed={total_processed} "
        f"skipped={total_skipped} failed={total_failed}"
    )
    return 0 if total_failed == 0 else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

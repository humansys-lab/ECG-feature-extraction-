#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Mapping
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
    age: float | None = None
    age_days: float | None = None
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

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "the installed PyTorch does not support safe weights-only loading; "
            "upgrade PyTorch or convert the input to a non-pickle format"
        ) from exc

    def safe_value(value: Any) -> bool:
        if torch.is_tensor(value) or value is None or isinstance(
            value, (bool, int, float, str, bytes)
        ):
            return True
        if isinstance(value, (list, tuple)):
            return all(safe_value(item) for item in value)
        if isinstance(value, Mapping):
            return all(
                isinstance(key, (bool, int, float, str, bytes))
                and safe_value(item)
                for key, item in value.items()
            )
        return False

    if not safe_value(payload):
        raise ValueError(
            f"{path} contains unsupported object types; expected tensors and basic containers"
        )
    return payload


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


def _resolve_age_metadata(
    metadata: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    """Return a consistent year/day pair with fail-closed day precedence."""

    raw_age_days = metadata.get("age_days")
    if raw_age_days is None:
        raw_age = metadata.get("age")
        if raw_age is None or isinstance(raw_age, bool):
            return None, None
        try:
            age_years = float(raw_age)
        except (TypeError, ValueError, OverflowError):
            return None, None
        if not math.isfinite(age_years) or age_years < 0.0:
            return None, None
        return age_years, None
    try:
        age_days = float(raw_age_days)
    except (TypeError, ValueError, OverflowError):
        return None, None
    if not math.isfinite(age_days) or age_days < 0.0:
        return None, None
    return age_days / 365.25, age_days


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
    age, resolved_age_days = _resolve_age_metadata(metadata)
    return {
        "record": record_id,
        "fs": int(sampling_rate),
        "n_samples": int(n_points),
        "age": age,
        "age_days": resolved_age_days,
        "sex": metadata.get("sex"),
        "dx": [str(label)],
        "rx": "Unknown",
        "hx": f"dataset={dataset}; source={source_name}; disease={disease_label}",
        "sx": f"batch_id={record_id}",
    }


def opaque_record_id(job: BatchJob, sample_index: int) -> str:
    """Build a stable identifier that does not expose the ground-truth label."""
    identity = "\0".join(
        (
            str(job.input_path.resolve()),
            job.disease_label,
            job.source_name,
            str(int(sample_index)),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"ecg_{digest}"


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
        return
    path.write_text(text, encoding="utf-8")


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
    export_profile: str | None = None,
    save_pt_output: bool = True,
    gzip_json: bool = False,
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
        "export_profile": export_profile,
        "pt_output_enabled": bool(save_pt_output),
        "json_gzip_enabled": bool(gzip_json),
        "errors": [],
    }

    for sample_index, ecg in enumerate(batch):
        if limit is not None and sample_index >= limit:
            break

        sample_id = f"{sample_index:03d}"
        json_suffix = ".json.gz" if gzip_json else ".json"
        feature_json_path = job.output_dir / f"{sample_id}_features{json_suffix}"
        feature_pt_path = job.output_dir / f"{sample_id}_features.pt"
        report_path = job.output_dir / f"{sample_id}_report.txt"
        annotated_plot_path = job.output_dir / f"{sample_id}_ecg_annotated.png"

        expected_paths = [feature_json_path, report_path]
        if save_pt_output:
            expected_paths.append(feature_pt_path)
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
            record_id = opaque_record_id(job, sample_index)
            age, age_days = _resolve_age_metadata(metadata)
            patient_meta = patient_meta_factory(
                age=age,
                age_days=age_days,
                sex=metadata.get("sex"),
            )
            header = build_sample_header(
                disease_label=job.disease_label,
                record_id=record_id,
                source_name=job.source_name,
                sampling_rate=sampling_rate,
                n_points=sample_shape[1],
                metadata=metadata,
            )
            result = extractor.extract(ecg, fs=sampling_rate, meta=patient_meta)
            # Keep the full payload when writing PT. JSON-only runs can skip
            # copying data the selected export profile will discard. Inspect
            # the injected serializer so existing one-argument callers work.
            import inspect
            serializer_options = {}
            if not save_pt_output and export_profile is not None and prepare_json_export_fn is not None:
                try:
                    if "profile" in inspect.signature(to_dict_fn).parameters:
                        serializer_options["profile"] = export_profile
                except (TypeError, ValueError):
                    pass
            feature_payload = to_dict_fn(result, **serializer_options)
            clinical = feature_payload.get("clinical_interpretation", {})
            if isinstance(clinical, dict) and clinical.get("ruleset_version"):
                manifest["clinical_ruleset_version"] = clinical["ruleset_version"]

            if prepare_json_export_fn is not None:
                export_options: dict[str, Any] = {
                    "include_beat_features": include_beat_features,
                }
                if export_profile is not None:
                    export_options["profile"] = export_profile
                json_payload = prepare_json_export_fn(
                    feature_payload,
                    **export_options,
                )
            else:
                json_payload = feature_payload
            _write_json(feature_json_path, json_payload)
            if save_pt_output:
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
        help=(
            "Skip samples whose artifacts enabled by the current options already "
            "exist (JSON, report, optional PT, and optional annotated plot)."
        ),
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
            ".pt feature file, when enabled, keeps the full detail regardless of this flag."
        ),
    )
    parser.add_argument(
        "--export-profile",
        choices=("summary", "audit", "debug"),
        default="summary",
        help=(
            "JSON detail level: summary removes repeated audit arrays, audit keeps "
            "provenance, and debug keeps full per-beat features (default: summary)."
        ),
    )
    parser.add_argument(
        "--no-pt",
        action="store_true",
        help="Do not write the duplicate full-detail .pt feature artifact.",
    )
    parser.add_argument(
        "--gzip-json",
        action="store_true",
        help=(
            "Write feature JSON as .json.gz to reduce storage and I/O; the bundled "
            "MedGemma batch diagnostics discover and read this form directly."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of disease/source jobs to process concurrently (default: 1).",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="Numerical-library threads per worker (default: 1).",
    )
    return parser


def _configure_numerical_threads(thread_count: int) -> None:
    value = str(max(1, int(thread_count)))
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = value


def _process_job_runtime(job: BatchJob, config: dict[str, Any]) -> dict[str, Any]:
    (
        ECGFeatureExtractor,
        to_dict_fn,
        patient_meta_factory,
        report_writer,
        annotated_plot_writer,
        prepare_json_export_fn,
    ) = _import_runtime_components()
    extractor = ECGFeatureExtractor(mains_freq=50)
    return process_job(
        job=job,
        extractor=extractor,
        to_dict_fn=to_dict_fn,
        report_writer=report_writer,
        sampling_rate=int(config["sampling_rate"]),
        annotated_plot_writer=(
            annotated_plot_writer if config["annotated_plots"] else None
        ),
        load_pt_fn=load_pt,
        save_pt_fn=save_pt,
        patient_meta_factory=patient_meta_factory,
        limit=config["limit"],
        skip_existing=bool(config["skip_existing"]),
        prepare_json_export_fn=prepare_json_export_fn,
        include_beat_features=bool(config["include_beat_features"]),
        export_profile=str(config["export_profile"]),
        save_pt_output=bool(config["save_pt_output"]),
        gzip_json=bool(config["gzip_json"]),
    )


def run_batch(args: argparse.Namespace) -> int:
    sources = parse_sources(args.sources)
    workers = int(getattr(args, "workers", 1))
    blas_threads = int(getattr(args, "blas_threads", 1))
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if blas_threads < 1:
        raise ValueError("--blas-threads must be at least 1")
    serial_runtime = None
    if workers == 1:
        _configure_numerical_threads(blas_threads)
        serial_runtime = _import_runtime_components()

    jobs = list(iter_jobs(args.input_dir, args.output_dir, sources))
    if not jobs:
        print(f"[warning] no matching source files found under {args.input_dir}")
        return 1

    total_processed = 0
    total_failed = 0
    total_skipped = 0
    include_beat_features = bool(getattr(args, "include_beat_features", False))
    export_profile = (
        "debug"
        if include_beat_features
        else str(getattr(args, "export_profile", "summary"))
    )
    config = {
        "sampling_rate": int(args.sampling_rate),
        "annotated_plots": bool(args.annotated_plots),
        "limit": args.limit,
        "skip_existing": bool(args.skip_existing),
        "include_beat_features": include_beat_features,
        "export_profile": export_profile,
        "save_pt_output": not bool(getattr(args, "no_pt", False)),
        "gzip_json": bool(getattr(args, "gzip_json", False)),
    }

    def record_summary(job: BatchJob, summary: dict[str, Any]) -> None:
        nonlocal total_processed, total_failed, total_skipped
        total_processed += int(summary["processed"])
        total_failed += int(summary["failed"])
        total_skipped += int(summary["skipped"])
        print(
            f"[done] {job.disease_label}/{job.source_name}: "
            f"processed={summary['processed']} skipped={summary['skipped']} failed={summary['failed']}"
        )

    if workers == 1:
        assert serial_runtime is not None
        (
            ECGFeatureExtractor,
            to_dict_fn,
            patient_meta_factory,
            report_writer,
            annotated_plot_writer,
            prepare_json_export_fn,
        ) = serial_runtime
        extractor = ECGFeatureExtractor(mains_freq=50)
        for job in jobs:
            print(f"[process] {job.disease_label}/{job.source_name} -> {job.output_dir}")
            summary = process_job(
                job=job,
                extractor=extractor,
                to_dict_fn=to_dict_fn,
                report_writer=report_writer,
                sampling_rate=config["sampling_rate"],
                annotated_plot_writer=(
                    annotated_plot_writer if config["annotated_plots"] else None
                ),
                load_pt_fn=load_pt,
                save_pt_fn=save_pt,
                patient_meta_factory=patient_meta_factory,
                limit=config["limit"],
                skip_existing=config["skip_existing"],
                prepare_json_export_fn=prepare_json_export_fn,
                include_beat_features=config["include_beat_features"],
                export_profile=config["export_profile"],
                save_pt_output=config["save_pt_output"],
                gzip_json=config["gzip_json"],
            )
            record_summary(job, summary)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_configure_numerical_threads,
            initargs=(blas_threads,),
        ) as executor:
            futures = {}
            for job in jobs:
                print(f"[process] {job.disease_label}/{job.source_name} -> {job.output_dir}")
                futures[executor.submit(_process_job_runtime, job, config)] = job
            for future in as_completed(futures):
                job = futures[future]
                try:
                    summary = future.result()
                except Exception as exc:
                    total_failed += 1
                    print(
                        f"[failed] {job.disease_label}/{job.source_name}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                record_summary(job, summary)

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
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

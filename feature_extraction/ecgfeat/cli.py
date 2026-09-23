"""Small CLI facade for the public ECG Record API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from .config import ECGConfig, PWaveConfig, RefinementConfig
from .errors import ECGInputError, ConfigurationError, ComputationInvariantError
from .record import MeasurementQuery, dump_measurements, load_record, query_measurement, resolve_address, select_measurements


def _load_signal(path: str | Path) -> Any:
    import numpy as np

    if str(path) == "-":
        path = sys.stdin.buffer
    source = np.load(path, allow_pickle=False)
    if isinstance(source, np.ndarray):
        return source
    with source:
        if "signal" not in source:
            raise ValueError("NPZ signal input must contain a 'signal' array")
        return source["signal"]


def _write_text(value: str, destination: str | Path) -> None:
    if str(destination) == "-":
        sys.stdout.write(value)
    else:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, prefix=".ecg-record-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _config_from_file(path: str | Path | None) -> ECGConfig | None:
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        if "refinement" in data:
            data["refinement"] = RefinementConfig(**data["refinement"])
        if "p_wave" in data:
            data["p_wave"] = PWaveConfig(**data["p_wave"])
        return ECGConfig(**data)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


def cli_measure(signal: str | Path, *, sampling_rate: float, lead_names: Sequence[str], amplitude_unit: str = "mV", input_mode: str = "standard_12", method: str = "default", profile: str = "summary", config: str | Path | None = None, sidecar: str | Path | None = None, output: str | Path = "-") -> int:
    from .pipeline import ecg_record
    if sidecar is not None:
        raise ConfigurationError("sidecar-backed extraction is not implemented; omit --sidecar for inline JSON")
    record = ecg_record(_load_signal(signal), sampling_rate=sampling_rate, lead_names=lead_names, amplitude_unit=amplitude_unit, input_mode=input_mode, method=method, profile=profile, config=_config_from_file(config))
    from .record.serialize import serialize_record
    encoded = serialize_record(record, profile=profile)
    _write_text(encoded.json_bytes.decode("utf-8") + "\n", output)
    return 0


def _record_source(path: str | Path) -> Any:
    return sys.stdin if str(path) == "-" else path


def cli_validate(record: str | Path, *, level: str = "strict") -> int:
    loaded = load_record(_record_source(record), validate=level)  # type: ignore[arg-type]
    sys.stdout.write(json.dumps({"valid": True, "record_id": loaded.record_id, "schema_version": loaded.schema_version}, sort_keys=True) + "\n")
    return 0


def cli_query(record: str | Path, name: str, *, lead: str | None = None, beat: int | None = None) -> int:
    result = query_measurement(load_record(_record_source(record)), name, lead=lead, beat=beat)
    payload = asdict(result)
    payload["address"] = str(result.address)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def cli_resolve(record: str | Path, address: str) -> int:
    result = resolve_address(load_record(_record_source(record)), address)
    sys.stdout.write(json.dumps({"address": str(result.address), "value": result.value, "node_kind": result.node_kind}, sort_keys=True) + "\n")
    return 0


def cli_select(record: str | Path, *, queries: str | Path, format: str = "json", output: str | Path = "-") -> int:
    raw = json.loads(Path(queries).read_text(encoding="utf-8"))
    values = raw.get("queries", raw) if isinstance(raw, dict) else raw
    if not isinstance(values, list) or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) or set(item) - {"name", "lead", "beat"} for item in values):
        raise ConfigurationError("queries must be an array of name/lead/beat objects")
    selection = select_measurements(load_record(_record_source(record)), tuple(MeasurementQuery(item["name"], item.get("lead"), item.get("beat")) for item in values))
    import io
    stream = io.StringIO()
    dump_measurements(selection, stream, format=format)  # type: ignore[arg-type]
    _write_text(stream.getvalue(), output)
    return 0


def cli_batch(manifest: str | Path, *, output_dir: str | Path, jobs: int = 1, fail_fast: bool = False) -> int:
    """Bounded parallel processing with ordered JSONL statuses and atomic files."""
    if type(jobs) is not int or jobs < 1:
        raise ConfigurationError("jobs must be a positive integer")
    manifest_path = Path(manifest)
    base = manifest_path.parent if str(manifest) != "-" else Path.cwd()
    lines = sys.stdin if str(manifest) == "-" else manifest_path.open(encoding="utf-8")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    failures = 0
    targets: set[Path] = set()

    def prepare(raw: str) -> dict:
        item = json.loads(raw)
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("manifest item requires a nonempty string id")
        allowed = {"id", "output", "signal", "sampling_rate", "lead_names", "amplitude_unit", "input_mode", "method", "profile", "config"}
        if set(item) - allowed:
            raise ConfigurationError(f"unknown manifest members: {sorted(set(item) - allowed)}")
        target = (root / item.get("output", f"{item['id']}.json")).resolve()
        if target == root or not target.is_relative_to(root):
            raise ValueError("output must be a file under output-dir")
        if target in targets:
            raise ValueError("duplicate manifest output path")
        targets.add(target)
        for key in ("signal", "config"):
            if key in item:
                if item[key] == "-":
                    raise ValueError("batch items cannot consume stdin")
                path = Path(item[key])
                item[key] = path if path.is_absolute() else base / path
        item["output"] = target
        return item

    def execute(item: dict) -> dict:
        try:
            cli_measure(item["signal"], sampling_rate=item["sampling_rate"], lead_names=item["lead_names"], amplitude_unit=item.get("amplitude_unit", "mV"), input_mode=item.get("input_mode", "standard_12"), method=item.get("method", "default"), profile=item.get("profile", "summary"), config=item.get("config"), output=item["output"])
            return {"id": item["id"], "output": str(item["output"])}
        except Exception as exc:
            return {"id": item["id"], "error": type(exc).__name__, "message": str(exc)}

    def work(executor):
        for raw in lines:
            if not raw.strip():
                continue
            try:
                item = prepare(raw)
            except Exception as exc:
                try:
                    parsed = json.loads(raw)
                    item_id = parsed.get("id") if isinstance(parsed, dict) else None
                except ValueError:
                    item_id = None
                yield {"id": item_id, "error": type(exc).__name__, "message": str(exc)}
            else:
                yield executor.submit(execute, item)

    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            pending = deque()
            tasks = iter(work(executor))
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < (1 if fail_fast else jobs):
                    task = next(tasks, None)
                    if task is None:
                        exhausted = True
                    else:
                        pending.append(task)
                if not pending:
                    break
                task = pending.popleft()
                status = task if isinstance(task, dict) else task.result()
                sys.stdout.write(json.dumps(status, sort_keys=True) + "\n")
                failures += "error" in status
                if fail_fast and failures:
                    break
    finally:
        if hasattr(lines, "close") and lines is not sys.stdin:
            lines.close()
    return 4 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecg-record")
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("signal")
    measure.add_argument("--sampling-rate", type=float, required=True)
    measure.add_argument("--lead-names", nargs="+", required=True)
    measure.add_argument("--amplitude-unit", choices=["mV", "uV"], default="mV")
    measure.add_argument("--input-mode", choices=["standard_12", "limited"], default="standard_12")
    measure.add_argument("--method", default="default")
    measure.add_argument("--profile", choices=["summary", "all", "debug"], default="summary")
    measure.add_argument("--config")
    measure.add_argument("--sidecar")
    measure.add_argument("-o", "--output", default="-")
    validate = sub.add_parser("validate")
    validate.add_argument("record")
    validate.add_argument("--level", choices=["strict", "schema"], default="strict")
    query = sub.add_parser("query")
    query.add_argument("record")
    query.add_argument("name")
    query.add_argument("--lead")
    query.add_argument("--beat", type=int)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("record")
    resolve.add_argument("address")
    select = sub.add_parser("select")
    select.add_argument("record")
    select.add_argument("--queries", required=True)
    select.add_argument("--format", choices=["json", "jsonl"], default="json")
    select.add_argument("-o", "--output", default="-")
    batch = sub.add_parser("batch")
    batch.add_argument("manifest")
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--jobs", type=int, default=1)
    batch.add_argument("--fail-fast", action="store_true")
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "measure":
        return cli_measure(args.signal, sampling_rate=args.sampling_rate, lead_names=args.lead_names, amplitude_unit=args.amplitude_unit, input_mode=args.input_mode, method=args.method, profile=args.profile, config=args.config, sidecar=args.sidecar, output=args.output)
    if args.command == "validate":
        return cli_validate(args.record, level=args.level)
    if args.command == "query":
        return cli_query(args.record, args.name, lead=args.lead, beat=args.beat)
    if args.command == "resolve":
        return cli_resolve(args.record, args.address)
    if args.command == "select":
        return cli_select(args.record, queries=args.queries, format=args.format, output=args.output)
    if args.command == "batch":
        return cli_batch(args.manifest, output_dir=args.output_dir, jobs=args.jobs, fail_fast=args.fail_fast)
    raise AssertionError(args.command)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            code = 2
        elif isinstance(exc, ComputationInvariantError):
            code = 70
        elif isinstance(exc, (ECGInputError, ValueError)):
            code = 3
        elif isinstance(exc, OSError):
            code = 5
        else:
            code = 70
        sys.stderr.write(json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True) + "\n")
        return code


if __name__ == "__main__":
    raise SystemExit(main())

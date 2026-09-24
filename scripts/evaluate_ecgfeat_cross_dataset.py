#!/usr/bin/env python3
"""Frozen external evaluation: full QRS streams plus end-to-end ECG extraction.

INCART/SVDB: all records, full-duration detector and fixed first/middle/last 30s
pipeline windows. PWAVE: all records, full-duration pipeline. No annotation-guided
channel choice, time correction, window selection, or algorithm fitting.
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ecgfeat_external import (
    windows, score, sparse_match, WaveEvent, aggregate_metrics, write_csv, BEAT_SYMBOLS,
)
import numpy as np
import wfdb

DATASETS = {"incartdb": 75, "svdb": 78, "pwave": 12}
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PROTOCOL = dict(
    version=2, fs_internal=500, core_seconds=30, context_seconds=2,
    full_qrs_datasets=["incartdb", "svdb"], full_pipeline_datasets=["pwave"],
    sampled_pipeline="each record: first, centered middle, last 30s; selection uses length only",
    variants=["default", "enhanced"], qrs_tolerance_ms=[75, 50, 150], p_tolerance_ms=[150, 50],
    mains_hz=dict(incartdb=50, svdb=60, pwave=60), units="WFDB physical mV",
    input_mode="INCART standard 12 leads; SVDB/PWAVE limited with original header channel names",
    canonicalization="AVR/AVL/AVF mapped to aVR/aVL/aVF; other names preserved",
    default="all refinement flags false",
    enhanced="qrs_adaptive_consensus+p_pathology_candidates+atrial_event_validation+t_sequence_offset_only",
    detector_scope="production QRS preprocessing and detector; omits subsequent pacing/rescue/delineation",
    qrs_annotation_symbols="".join(sorted(BEAT_SYMBOLS)), p_annotation_symbols=["p"],
    annotation_time_shift=False, matching="one-to-one maximum count then minimum absolute timing error",
    duplicate_reference_policy="coalesce identical symbol/sample annotations and audit every duplicate; conflicting symbols fail",
    failure_policy="no predictions for failed windows; keep all corresponding reference beats",
    overlap="exclude whole source record if previously present in BUT, QTDB or NSTDB",
    pwave_caveat="upstream expert P annotations not guaranteed exhaustive; FP is unmatched prediction",
    incart_caveat="QRS center annotations, not manually corrected R apex; 150ms secondary sensitivity analysis",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate_source(source):
    path = str(Path(source).resolve() / "feature_extraction")
    sys.path.insert(0, path)
    import ecgfeat
    if not Path(ecgfeat.__file__).resolve().is_relative_to(Path(path)):
        raise RuntimeError(f"wrong ecgfeat source loaded: {ecgfeat.__file__}")


def select_windows(length, fs, scope):
    if scope in {"detector_full", "pipeline_full"}:
        return list(windows(length, fs))
    if scope != "pipeline_sampled":
        raise ValueError(scope)
    core, context = round(30 * fs), round(2 * fs)
    if length < 3 * core:
        raise ValueError("sampled protocol requires at least 90 seconds")
    return [(start, start + core, max(0, start-context), min(length, start+core+context))
            for start in (0, (length-core)//2, length-core)]


def in_cores(sample, bounds):
    return any(a <= sample < b for a, b, _, _ in bounds)


def source_overlaps(dataset, record, root=ROOT):
    reasons = []
    if dataset == "incartdb":
        return reasons
    text = (root / "data/external_ecg/but-pdb/README.txt").read_text()
    database = "Supraventricular Arrhythmia" if dataset == "svdb" else "Arrhythmia"
    pattern = r"MIT-BIH " + database + r" Database\s+(\d+)\s+\["
    if record in set(re.findall(pattern, text)):
        reasons.append("BUT source record")
    if (root / "data/qtdb" / f"sel{record}.hea").exists():
        reasons.append("QTDB source record")
    if dataset == "pwave" and record in {"118", "119"}:
        reasons.append("NSTDB source record")
    return reasons


def annotations(path, extension, length, wave):
    ann = wfdb.rdann(str(path), extension)
    allowed = BEAT_SYMBOLS if wave == "QRS" else {"p"}
    truth, symbols, excluded = [], {}, []
    for sample, symbol in zip(ann.sample, ann.symbol):
        sample = int(sample)
        if not 0 <= sample < length or symbol not in allowed:
            excluded.append(dict(sample=sample, symbol=symbol))
        else:
            if sample in symbols:
                if symbols[sample] != symbol:
                    raise ValueError(f"conflicting reference {wave} annotation at {sample}")
                excluded.append(dict(sample=sample, symbol=symbol, action="duplicate_identical_reference"))
                continue
            truth.append(WaveEvent(None, sample, None))
            symbols[sample] = symbol
    if not truth:
        raise ValueError(f"no scorable {wave} annotations in {path}.{extension}")
    return truth, symbols, excluded


def adapt_record(record, dataset):
    expected = 12 if dataset == "incartdb" else 2
    if record.n_sig != expected or any(unit != "mV" for unit in record.units):
        raise ValueError(f"unexpected input: {record.n_sig} leads, units={record.units}")
    signal = record.p_signal.T
    if not np.isfinite(signal).all():
        raise ValueError("nonfinite signal; no silent interpolation")
    canonical = {name.lower(): name for name in LEADS}
    names = [canonical.get(name.lower(), name) for name in record.sig_name]
    if len(set(names)) != len(names):
        raise ValueError("duplicate channel names")
    return signal, names


def pipeline_events(features, first_slot):
    accepted = {a.beat_id for a in features.p_wave_assessments if a.accepted}
    items = [b for b in features.beat_features if b.lead == first_slot and b.p.peak is not None]
    return {
        "QRS": [b.r_index for b in features.beats],
        "P_native": [b.p.peak for b in items],
        "P_accepted": [b.p.peak for b in items if b.beat_id in accepted],
        "P_atrial": [int(e["sample"]) for e in features.metadata.get("rhythm_analysis", {}).get("atrial_events", [])],
    }


def evaluate(task):
    activate_source(task["source"])
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor
    from ecgfeat.refinement import RefinementConfig
    from ecgfeat.validation import validate_ecg_input
    from ecgfeat.preprocess import resample_ecg, analysis_signal, lowpass_filter
    from ecgfeat.quality import _bsqi
    from ecgfeat.qrs import detect_qrs_multilead_with_meta, DEFAULT_QRS_LEADS

    started = time.perf_counter()
    dataset, rid, scope, variant = (task[k] for k in ("dataset", "record", "scope", "variant"))
    folder = Path(task["data_root"]) / dataset
    path = folder / rid
    record = wfdb.rdrecord(str(path))
    fs, length = float(record.fs), record.sig_len
    bounds = select_windows(length, fs, scope)
    reference = folder / "mitdb_reference" / rid if dataset == "pwave" else path
    truth, symbols, excluded = annotations(reference, "atr", length, "QRS")
    truth = {"QRS": [e for e in truth if in_cores(e.peak, bounds)]}
    if dataset == "pwave":
        p_truth, _, p_excluded = annotations(path, "pwave", length, "P")
        truth["P"] = [e for e in p_truth if in_cores(e.peak, bounds)]
        excluded += [dict(wave="P", **e) for e in p_excluded]
    predictions = {name: [] for name in ("QRS", "P_native", "P_accepted", "P_atrial")}
    failures, timings, output_audit = [], [], []
    lead_contract = None
    try:
        signal, names = adapt_record(record, dataset)
        limited = dataset != "incartdb"
        enhanced = variant == "enhanced"
        refinement = RefinementConfig(qrs_adaptive_consensus=enhanced, p_pathology_candidates=enhanced,
                                      atrial_event_validation=enhanced, t_sequence_offset_only=enhanced)
        extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=PROTOCOL["mains_hz"][dataset],
                                        input_mode="limited" if limited else "standard", refinement=refinement)
        for a, b, c, d in bounds:
            before = time.perf_counter()
            try:
                raw = signal[:, c:d]
                normalized, _, contract = validate_ecg_input(raw, fs, 500, lead_names=names,
                                                             amplitude_unit="mV", allow_limited_leads=limited)
                lead_contract = contract.get("lead_slot_map")
                if scope == "detector_full":
                    measured = analysis_signal(resample_ecg(normalized, fs, 500), 500,
                                               mains_hz=PROTOCOL["mains_hz"][dataset])
                    detect = lowpass_filter(measured, 500, cutoff_hz=40, order=4)
                    slots = tuple(LEADS.index(name) for name in contract["available_leads"]) if limited else DEFAULT_QRS_LEADS
                    quality = {LEADS[i]: SimpleNamespace(b_sqi=_bsqi(measured[i], 500)) for i in slots}
                    result = detect_qrs_multilead_with_meta(detect, 500, leads=slots, quality=quality,
                                                            adaptive_consensus=enhanced)
                    events = {"QRS": result.r_locs}
                else:
                    features = extractor.extract(raw, fs=fs, amplitude_unit="mV", lead_names=names)
                    first_slot = contract["lead_slot_map"][names[0]] if limited else "II"
                    events = pipeline_events(features, first_slot)
                    output_audit.append(dict(core_start=a, beats=len(features.beats),
                                             native_p=len(events["P_native"]), accepted_p=len(events["P_accepted"])))
                for kind, points in events.items():
                    predictions[kind].extend(int(round(p*fs/500))+c for p in points
                                             if a <= int(round(p*fs/500))+c < b)
            except Exception as exc:
                failures.append(dict(core_start=a, core_end=b, error=f"{type(exc).__name__}: {exc}"))
            timings.append(dict(core_start=a, core_seconds=(b-a)/fs,
                                input_seconds=(d-c)/fs, elapsed_seconds=time.perf_counter()-before))
    except Exception as exc:
        failures.append(dict(core_start=0, core_end=length, error=f"{type(exc).__name__}: {exc}"))
    predictions = {name: sorted(set(values)) for name, values in predictions.items()}
    rows, matches, by_symbol = [], [], []
    status = "partial_failure" if failures else "ok"
    overlap = source_overlaps(dataset, rid)
    patient = rid
    for comment in record.comments:
        found = re.search(r"patient\s+(\d+)", comment, re.I)
        if found:
            patient = found.group(1)
    for kind, points in predictions.items():
        wave = "QRS" if kind == "QRS" else "P"
        if wave not in truth:
            continue
        detected = [WaveEvent(None, p, None) for p in points]
        tolerances = PROTOCOL["qrs_tolerance_ms" if wave == "QRS" else "p_tolerance_ms"]
        for tolerance in tolerances:
            extra = dict(dataset=dataset, scope=scope, protocol=f"peak_{tolerance}ms", patient=patient,
                         prior_source_overlap=bool(overlap))
            row, matched = score(rid, variant, wave, kind, truth[wave], detected, fs, status,
                                 tolerance=tolerance, extra=extra)
            rows.append(row)
            matches.extend(matched)
            if wave == "QRS":
                found = {m["gt_peak_sample"] for m in matched}
                counts = Counter(symbols[e.peak] for e in truth[wave])
                for symbol, count in sorted(counts.items()):
                    tp = sum(e.peak in found for e in truth[wave] if symbols[e.peak] == symbol)
                    by_symbol.append(dict(record=rid, method=variant, symbol=symbol,
                                          gt_count=count, tp=tp, fn=count-tp, **extra))
    item = dict(dataset=dataset, record=rid, scope=scope, variant=variant, fingerprint=task["fingerprint"],
                fs=fs, samples=length, duration_seconds=length/fs, evaluated_seconds=sum((b-a)/fs for a,b,_,_ in bounds),
                original_leads=record.sig_name, original_units=record.units, lead_slot_map=lead_contract,
                patient=patient, overlap_reasons=overlap, failures=failures, annotations_excluded=excluded,
                timings=timings, seconds=time.perf_counter()-started, output_audit=output_audit,
                rows=rows, matches=matches, by_symbol=by_symbol, predictions=predictions)
    target = Path(task["checkpoint"])
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(item, allow_nan=False))
    temporary.replace(target)
    return {key: item[key] for key in ("dataset", "record", "scope", "variant", "seconds", "failures")}


def summarize(out, checkpoint_dirs=None):
    rows, matches, symbols, records = [], [], [], []
    paths = [path for folder in (checkpoint_dirs or [out / "checkpoints"]) for path in folder.glob("*.json")]
    seen = set()
    for path in sorted(paths):
        item = json.loads(path.read_text())
        key = tuple(item[k] for k in ("dataset", "record", "scope", "variant"))
        if key in seen:
            raise ValueError(f"duplicate checkpoint task: {key}")
        seen.add(key)
        rows.extend(item.pop("rows")); matches.extend(item.pop("matches")); symbols.extend(item.pop("by_symbol"))
        item.pop("predictions"); records.append(item)
    fields = ("dataset", "scope", "method", "wave", "lead", "protocol")
    metrics = aggregate_metrics(rows, matches, fields)
    novel = aggregate_metrics([r for r in rows if not r["prior_source_overlap"]],
                              [m for m in matches if not m["prior_source_overlap"]], fields)
    write_csv(out / "detection_by_record.csv", rows)
    write_csv(out / "matched_events.csv", matches)
    write_csv(out / "qrs_by_symbol.csv", symbols)
    (out / "records.json").write_text(json.dumps(records, indent=2))
    result = dict(protocol=PROTOCOL, completed_tasks=len(records), metrics=metrics,
                  excluding_prior_source_records=novel,
                  failures=[{key: r[key] for key in ("dataset", "record", "scope", "variant", "failures")}
                            for r in records if r["failures"]])
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external_ecg")
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--records", nargs="+", help="explicit smoke-test cohort; never silently called full evaluation")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(exist_ok=True)
    specs = []
    for dataset in args.datasets:
        records = sorted(p.stem for p in (args.data_root / dataset).glob("*.hea"))
        if len(records) != DATASETS[dataset]:
            parser.error(f"incomplete {dataset}: {len(records)}/{DATASETS[dataset]}")
        for rid in records:
            if args.records and rid not in args.records:
                continue
            for scope in (["pipeline_full"] if dataset == "pwave" else ["detector_full", "pipeline_sampled"]):
                for variant in PROTOCOL["variants"]:
                    specs.append(dict(dataset=dataset, record=rid, scope=scope, variant=variant))
    if not specs:
        parser.error("empty cohort")
    code = {str(p.relative_to(args.source)): sha256(p) for p in sorted((args.source / "feature_extraction/ecgfeat").rglob("*.py"))}
    harness = {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / "scripts/evaluate_ecgfeat_external.py",
               ROOT / "scripts/evaluate_ecgfeat_candidates.py", ROOT / "compare_ludb_detectors.py"]}
    data = {str(p.relative_to(args.data_root)): sha256(p) for dataset in args.datasets
            for p in sorted((args.data_root / dataset).rglob("*")) if p.is_file() and p.suffix != ".part"}
    overlap_files = [ROOT / "data/external_ecg/but-pdb/README.txt", *sorted((ROOT / "data/qtdb").glob("*.hea"))]
    manifest = dict(protocol=PROTOCOL, tasks=specs, source_sha256=code, harness_sha256=harness, data_sha256=data,
                    overlap_metadata_sha256={str(p.relative_to(ROOT)): sha256(p) for p in overlap_files},
                    source_path=str(args.source.resolve()), data_root=str(args.data_root.resolve()))
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    target = args.out / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        parser.error("source/data/protocol differs from existing results; use a new output directory")
    target.write_text(json.dumps(manifest, indent=2))
    tasks = []
    for spec in specs:
        checkpoint = args.out / "checkpoints" / ("__".join(spec.values()) + ".json")
        if checkpoint.exists():
            if json.loads(checkpoint.read_text())["fingerprint"] != fingerprint:
                parser.error(f"stale checkpoint: {checkpoint}")
            continue
        tasks.append(dict(**spec, checkpoint=str(checkpoint), fingerprint=fingerprint,
                          source=str(args.source.resolve()), data_root=str(args.data_root.resolve())))
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(evaluate, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            print(i, len(tasks), json.dumps(future.result()), flush=True)
    summary = summarize(args.out)
    (args.out / "execution.json").write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,
            workers=args.workers, expected_tasks=len(specs), newly_executed_tasks=len(tasks)), indent=2))
    return int(bool(summary["failures"]) or summary["completed_tasks"] != len(specs))


if __name__ == "__main__":
    raise SystemExit(main())

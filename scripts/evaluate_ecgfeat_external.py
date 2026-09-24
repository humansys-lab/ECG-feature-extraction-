#!/usr/bin/env python3
"""Frozen traditional-algorithm evaluation on ISP v2, BUT PDB and NSTDB.

No learning, annotation-guided extraction, or parameter fitting. Checkpoints
include source/data hashes; incomplete/failed windows remain in denominators.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feature_extraction"))
sys.path.insert(1, str(ROOT))

import numpy as np
import wfdb
from scipy.optimize import linear_sum_assignment
from compare_ludb_detectors import WaveEvent, match_events, aggregate_metrics
from scripts.evaluate_ecgfeat_candidates import variant_options, write_csv

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")
PROTOCOL = {
    "version": 1, "internal_fs": 500, "core_seconds": 30, "context_seconds": 2,
    "qrs_tolerance_ms": 75, "p_tolerance_ms": [150, 50],
    "isp_iou_threshold": 0.5, "isp_interval_convention": "[onset, offset)",
    "isp_primary_output": "median boundaries across available native lead waves per beat",
    "isp_secondary_output": "native lead II against integrated labels (not lead-specific truth)",
    "isp_units": "mkv interpreted as microvolts, divided by 1000",
    "isp_invalid_annotations": "zero-length segment excluded; out-of-range right endpoint clipped for visible IoU/Dice and censored for offset MAE",
    "sparse_adapter": "channel 0 -> II, channel 1 -> V2; remaining channels zero; no anatomical claim",
    "but_units": "WFDB physical values in parsed mV; malformed literal \\muV header token retained in audit",
    "nstdb_noise": "prebuilt electrode-motion ECGs only; t>=300 and floor((t-300)/120)%2==0",
    "mains_hz": {"isp": 50, "but-pdb": 60, "nstdb": 60},
    "failures": "failed window yields no predictions; complete ground truth still scored",
    "no_annotation_span_restriction": True,
}


def windows(length, fs, core_seconds=30, context_seconds=2):
    core, context = round(core_seconds * fs), round(context_seconds * fs)
    start = 0
    while start < length:
        end = min(length, start + core)
        if 0 < length - end < 5 * fs:
            end = length
        yield start, end, max(0, start - context), min(length, end + context)
        start = end


def noise_phase(sample, fs):
    seconds = sample / fs
    return "noisy" if seconds >= 300 and int((seconds - 300) // 120) % 2 == 0 else "clean"


def sparse_match(gt, det, tolerance):
    """Exact existing DP on independent components; avoids quadratic long-record RAM.

    A gap larger than tolerance in the merged sorted coordinates forbids every
    cross-gap edge, so decomposition preserves the original DP objective.
    """
    merged = sorted([(e.peak, 0, i, e) for i, e in enumerate(gt)] +
                    [(e.peak, 1, i, e) for i, e in enumerate(det)])
    pairs, group = [], [[], []]
    previous = None
    for peak, kind, _, event in merged:
        if previous is not None and peak - previous > tolerance:
            pairs.extend(match_events(*group, tolerance)[0])
            group = [[], []]
        group[kind].append(event)
        previous = peak
    pairs.extend(match_events(*group, tolerance)[0])
    return pairs


def interval_match(gt, det, threshold=0.5):
    """Maximum-cardinality one-to-one IoU matching, then maximum total IoU."""
    if not gt or not det:
        return []
    ious = np.zeros((len(gt), len(det)))
    for i, a in enumerate(gt):
        for j, b in enumerate(det):
            if b.onset is None or b.offset is None or b.offset <= b.onset:
                continue
            intersection = max(0, min(a.offset, b.offset) - max(a.onset, b.onset))
            union = max(a.offset, b.offset) - min(a.onset, b.onset)
            ious[i, j] = intersection / union if union else 0
    eligible = ious >= threshold
    reward = eligible * (min(len(gt), len(det)) + 1 + ious)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    return [(gt[i], det[j]) for i, j in zip(rows, cols) if eligible[i, j]]


def interval_samples(events, length):
    mask = np.zeros(length, dtype=bool)
    for e in events:
        if e.onset is not None and e.offset is not None:
            mask[max(0, e.onset):min(length, e.offset)] = True
    return mask


def score(record, variant, wave, lead, gt, det, fs, status, tolerance=150,
          intervals=False, length=0, extra=None):
    pairs = interval_match(gt, det) if intervals else sparse_match(gt, det, max(1, round(tolerance * fs / 1000)))
    extra = extra or {}
    common = dict(record=record, method=variant, wave=wave, lead=lead, **extra)
    row = dict(common, status=status, gt_count=len(gt), det_count=len(det),
               tp=len(pairs), fn=len(gt)-len(pairs), fp=len(det)-len(pairs),
               tolerance_ms=None if intervals else tolerance)
    matches = []
    for a, b in pairs:
        item = dict(common)
        for boundary in ("onset", "peak", "offset"):
            x, y = getattr(a, boundary), getattr(b, boundary)
            item[f"gt_{boundary}_sample"] = x
            item[f"det_{boundary}_sample"] = y
            item[f"{boundary}_error_ms"] = (y-x)*1000/fs if x is not None and y is not None else None
        matches.append(item)
    if intervals:
        g, d = interval_samples(gt, length), interval_samples(det, length)
        row.update(sample_gt=int(g.sum()), sample_det=int(d.sum()), sample_intersection=int((g & d).sum()))
    return row, matches


def discover(dataset, root):
    if dataset == "isp":
        directory = root / "isp/isp_delineation_dataset"
        records = []
        for split in ("train", "test"):
            with (directory / f"{split}_isp_delineation_data.csv").open() as handle:
                for row in csv.DictReader(handle):
                    records.append(dict(record=f"{split}/{row['file_name']}", split=split,
                                        path=str(directory / f"{split}_data" / row['file_name']),
                                        target=ast.literal_eval(row["target"])))
        return records
    directory = root / dataset
    if dataset == "but-pdb":
        diagnoses = {f"{int(n):02d}": value.strip() for n, value in
                     re.findall(r"(?m)^(\d+)\s+'([^']*)'", (directory / "README.txt").read_text())}
        return [dict(record=f"{n:02d}", path=str(directory / f"{n:02d}"),
                     diagnosis=diagnoses[f"{n:02d}"]) for n in range(1, 51)]
    return [dict(record=p.stem, path=str(p.with_suffix("")),
                 snr=-6 if p.stem.endswith("_6") else int(p.stem[-2:]), source_record=p.stem[:3])
            for p in sorted(directory.glob("*.hea")) if re.fullmatch(r"11[89]e(?:24|18|12|06|00|_6)", p.stem)]


def load_annotations(dataset, spec, fs, length):
    excluded = []
    if dataset == "isp":
        result = {w: [] for w in ("P", "QRS", "T")}
        for label, onset, offset in spec["target"]:
            if label not in (0, 1, 2) or onset < 0 or onset >= length or offset < onset:
                raise ValueError(f"invalid ISP annotation: {(label, onset, offset)} / {length}")
            if onset == offset:
                excluded.append(dict(wave=("P", "QRS", "T")[label], onset=onset, offset=offset, action="exclude_zero_length"))
                continue
            if offset > length:
                excluded.append(dict(wave=("P", "QRS", "T")[label], onset=onset, offset=offset, action="clip_right_censored"))
                offset = length
            result[("P", "QRS", "T")[label]].append(WaveEvent(onset, None, offset))
        return result, excluded
    result = {}
    for wave, extension in (("P", "pwave"), ("QRS", "qrs")) if dataset == "but-pdb" else (("QRS", "atr"),):
        ann = wfdb.rdann(spec["path"], extension)
        result[wave] = []
        for sample, symbol in zip(ann.sample, ann.symbol):
            if not 0 <= sample < length or (wave == "QRS" and symbol not in BEAT_SYMBOLS):
                excluded.append(dict(extension=extension, sample=int(sample), symbol=symbol))
            else:
                result[wave].append(WaveEvent(None, int(sample), None))
    return result, excluded


def adapt_signal(record, dataset):
    values = record.p_signal.T.copy()
    if not np.isfinite(values).all():
        raise ValueError("nonfinite input (no silent interpolation)")
    if dataset == "isp":
        if any(u != "mkv" for u in record.units):
            raise ValueError(f"unexpected ISP units: {record.units}")
        lookup = {name.lower(): i for i, name in enumerate(record.sig_name)}
        return values[[lookup[name.lower()] for name in LEADS]] / 1000
    if record.n_sig != 2 or record.units != ["mV", "mV"]:
        raise ValueError(f"unexpected sparse input: {record.n_sig}, {record.units}")
    output = np.zeros((12, record.sig_len))
    output[1], output[7] = values
    return output


def result_events(features, dataset):
    """Return native indices at features.fs, without consulting annotations."""
    outputs = {"II": {w: [] for w in ("P", "QRS", "T")}}
    if dataset == "isp":
        outputs["consensus_median"] = {w: [] for w in ("P", "QRS", "T")}
    by_beat = {}
    for item in features.beat_features:
        by_beat.setdefault(item.beat_id, []).append(item)
        if item.lead == "II":
            for wave in ("P", "QRS", "T"):
                bounds = getattr(item, wave.lower())
                if bounds.peak is not None:
                    outputs["II"][wave].append(WaveEvent(bounds.onset, bounds.peak, bounds.offset))
    if dataset != "isp":
        outputs["II"]["QRS"] = [WaveEvent(None, b.r_index, None) for b in features.beats]
    else:
        for items in by_beat.values():
            for wave in ("P", "QRS", "T"):
                bounds = [getattr(item, wave.lower()) for item in items]
                valid = [b for b in bounds if b.onset is not None and b.offset is not None and b.onset < b.offset]
                if valid:
                    onset, offset = (int(round(np.median([getattr(b, key) for b in valid]))) for key in ("onset", "offset"))
                    outputs["consensus_median"][wave].append(WaveEvent(onset, (onset+offset)//2, offset))
    if dataset == "but-pdb":
        accepted = {p.beat_id for p in features.p_wave_assessments if p.accepted}
        outputs["accepted_native"] = {"P": [WaveEvent(b.p.onset, b.p.peak, b.p.offset)
                                            for b in features.beat_features if b.lead == "II" and b.beat_id in accepted and b.p.peak is not None]}
        outputs["atrial_events"] = {"P": [WaveEvent(None, int(e["sample"]), None)
                                           for e in features.metadata.get("rhythm_analysis", {}).get("atrial_events", [])]}
    return outputs


def evaluate_task(payload):
    dataset, spec, variant, checkpoint, fingerprint = payload
    start = time.perf_counter()
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor
    record = wfdb.rdrecord(spec["path"])
    fs, length = float(record.fs), record.sig_len
    gt, excluded = load_annotations(dataset, spec, fs, length)
    failures, predictions, timings = [], {}, []
    try:
        signal = adapt_signal(record, dataset)
        extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=PROTOCOL["mains_hz"][dataset], **variant_options(variant))
        chunks = [(0, length, 0, length)] if dataset == "isp" else windows(length, fs)
        for core_start, core_end, start_sample, end_sample in chunks:
            before = time.perf_counter()
            try:
                input_signal = signal[:, start_sample:end_sample]
                input_kwargs = {}
                if extractor.input_mode == "limited":
                    if dataset == "isp":
                        raise ValueError("limited mode is intended for the two-channel external datasets")
                    input_signal = input_signal[[1, 7]]
                    input_kwargs["lead_names"] = ["channel_0", "channel_1"]
                features = extractor.extract(input_signal, fs=fs, amplitude_unit="mV", **input_kwargs)
                for lead, waves in result_events(features, dataset).items():
                    target = predictions.setdefault(lead, {w: [] for w in ("P", "QRS", "T")})
                    for wave, events in waves.items():
                        for e in events:
                            coordinates = [None if x is None else int(round(x*fs/features.fs))+start_sample
                                           for x in (e.onset, e.peak, e.offset)]
                            if coordinates[1] is not None and core_start <= coordinates[1] < core_end:
                                target[wave].append(WaveEvent(*coordinates))
            except Exception as exc:
                failures.append(dict(core_start=core_start, core_end=core_end, error=f"{type(exc).__name__}: {exc}"))
            timings.append(time.perf_counter()-before)
    except Exception as exc:
        failures.append(dict(core_start=0, core_end=length, error=f"{type(exc).__name__}: {exc}"))
    for waves in predictions.values():
        for wave, events in waves.items():
            waves[wave] = sorted(set(events), key=lambda e: e.peak)
    rows, matches = [], []
    status = "partial_failure" if failures else "ok"
    if dataset == "nstdb":
        detected = predictions.get("II", {}).get("QRS", [])
        # Match once across the entire recording, then allocate TP/FN by the
        # reference time and FP by prediction time, even across phase boundaries.
        paired = sparse_match(gt["QRS"], detected, round(.075*fs))
        used_gt, used_det = {a for a, _ in paired}, {b for _, b in paired}
        for phase in ("noisy", "clean"):
            pg = [g for g in gt["QRS"] if noise_phase(g.peak, fs) == phase]
            pp = [(g, d) for g, d in paired if noise_phase(g.peak, fs) == phase]
            fp = [d for d in detected if d not in used_det and noise_phase(d.peak, fs) == phase]
            extra = dict(snr=spec["snr"], phase=phase, source_record=spec["source_record"])
            row, matched = score(spec["record"], variant, "QRS", "II", [], [], fs, status, 75, extra=extra)
            row.update(gt_count=len(pg), det_count=len(pp)+len(fp), tp=len(pp), fp=len(fp), fn=sum(g not in used_gt for g in pg))
            for a, b in pp:
                matched.append(dict(record=spec["record"], method=variant, wave="QRS", lead="II", **extra,
                                    gt_peak_sample=a.peak, det_peak_sample=b.peak, peak_error_ms=(b.peak-a.peak)*1000/fs))
            rows.append(row); matches.extend(matched)
    else:
        for lead in (["consensus_median", "II"] if dataset == "isp" else ["II", "atrial_events", "accepted_native"]):
            for wave, truth in gt.items():
                if lead in {"atrial_events", "accepted_native"} and wave != "P":
                    continue
                detected = predictions.get(lead, {}).get(wave, [])
                for tolerance in ([150, 50] if dataset == "but-pdb" and wave == "P" else [75 if wave == "QRS" else 150]):
                    extra = dict(split=spec.get("split", "all"), protocol="iou_0.5" if dataset == "isp" else f"peak_{tolerance}ms")
                    row, matched = score(spec["record"], variant, wave, lead, truth, detected, fs, status,
                                         tolerance, dataset == "isp", length, extra)
                    if dataset == "isp":
                        censored = {e["onset"] for e in excluded if e["action"] == "clip_right_censored" and e["wave"] == wave}
                        row.update(gt_onset_count=len(truth), gt_offset_count=len(truth)-len(censored))
                        for m in matched:
                            if m["gt_onset_sample"] in censored:
                                m["gt_offset_sample"] = None
                                m["offset_error_ms"] = None
                    rows.append(row); matches.extend(matched)
    item = dict(dataset=dataset, record=spec["record"], variant=variant, fingerprint=fingerprint,
                fs=fs, samples=length, duration_seconds=length/fs, original_leads=record.sig_name,
                original_units=record.units, diagnosis=spec.get("diagnosis"), annotation_audit=excluded,
                failures=failures, window_seconds=timings, seconds=time.perf_counter()-start,
                rows=rows, matches=matches,
                predictions={lead: {w: [asdict(e) for e in es] for w, es in waves.items()} for lead, waves in predictions.items()})
    path = Path(checkpoint)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(item, allow_nan=False))
    temporary.replace(path)
    return dict(record=spec["record"], variant=variant, failures=failures, seconds=item["seconds"])


def summarize(directory, dataset, variants):
    rows, matches, records = [], [], []
    for path in sorted((directory / "checkpoints").glob("*.json")):
        item = json.loads(path.read_text())
        rows.extend(item.pop("rows")); matches.extend(item.pop("matches")); item.pop("predictions")
        records.append(item)
    write_csv(directory / "detection_by_record.csv", rows)
    write_csv(directory / "matched_events.csv", matches)
    (directory / "records.json").write_text(json.dumps(records, indent=2))
    fields = ("method", "wave", "lead", "snr", "phase") if dataset == "nstdb" else ("method", "wave", "lead", "protocol", "split")
    metrics = aggregate_metrics(rows, matches, fields)
    for metric in metrics:
        matching = [r for r in rows if all(r[k] == metric[k] for k in fields)]
        if dataset == "isp":
            totals = {k: sum(r[k] for r in matching) for k in ("sample_gt", "sample_det", "sample_intersection")}
            metric.update(totals)
            denom = totals["sample_gt"]+totals["sample_det"]
            metric["sample_dice"] = 2*totals["sample_intersection"]/denom if denom else None
            metric["onset_coverage"] = metric["onset_n"]/metric["gt_count"] if metric["gt_count"] else None
            metric["gt_offset_count"] = sum(r["gt_offset_count"] for r in matching)
            metric["offset_coverage"] = metric["offset_n"]/metric["gt_offset_count"] if metric["gt_offset_count"] else None
    result = dict(protocol=PROTOCOL, completed_tasks=len(records), variants=variants, metrics=metrics,
                  failures=[dict(record=r["record"], variant=r["variant"], failures=r["failures"]) for r in records if r["failures"]])
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("isp", "but-pdb", "nstdb"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external_ecg")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["default"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--records", nargs="+")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    for variant in args.variants:
        variant_options(variant)
    specs = discover(args.dataset, args.data_root)
    if args.records:
        unknown = set(args.records)-{s["record"] for s in specs}
        if unknown:
            parser.error(f"unknown records: {unknown}")
        specs = [s for s in specs if s["record"] in args.records]
    if not specs:
        parser.error("empty cohort")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(exist_ok=True)
    paths = list((ROOT / "feature_extraction/ecgfeat").rglob("*.py")) + [Path(__file__), ROOT / "compare_ludb_detectors.py", ROOT / "scripts/evaluate_ecgfeat_candidates.py"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    data_hashes = {str(p.relative_to(args.data_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((args.data_root / args.dataset).rglob("*")) if p.is_file()}
    manifest = dict(dataset=args.dataset, records=specs, variants=args.variants, protocol=PROTOCOL,
                    source_sha256=source_hashes, data_sha256=data_hashes)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    target = args.out / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        parser.error("existing experiment has different source, data or protocol; choose a new output")
    target.write_text(json.dumps(manifest, indent=2))
    tasks = []
    for spec in specs:
        for variant in args.variants:
            checkpoint = args.out / "checkpoints" / f"{variant}__{spec['record'].replace('/', '_')}.json"
            if checkpoint.exists():
                if json.loads(checkpoint.read_text())["fingerprint"] != fingerprint:
                    parser.error(f"stale checkpoint: {checkpoint}")
                continue
            tasks.append((args.dataset, spec, variant, str(checkpoint), fingerprint))
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_task, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            item = future.result()
            print(f"{n}/{len(tasks)} {item}", flush=True)
    summary = summarize(args.out, args.dataset, args.variants)
    (args.out / "execution.json").write_text(json.dumps(dict(wall_seconds=time.perf_counter()-start, workers=args.workers,
                                                            newly_executed_tasks=len(tasks), expected_tasks=len(specs)*len(args.variants))))
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())

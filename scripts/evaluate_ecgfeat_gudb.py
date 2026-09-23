#!/usr/bin/env python3
"""Independent GUDB QRS-detector evaluation, without diagnostic/pacing rescue.

Use raw volts converted to mV, fixed 30s+2s windows, and all annotated samples.
Missing annotation streams are unscorable (reported), never interpreted as no
beats. Chest strap and cables have separate time axes and are scored separately.
No accelerometry, annotation-selected channel, time shift, or rhythm fitting.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ecgfeat_external import windows, score, WaveEvent, aggregate_metrics, write_csv
import numpy as np
from ecgfeat.preprocess import resample_ecg, analysis_signal, lowpass_filter
from ecgfeat.quality import _bsqi
from ecgfeat.qrs import detect_qrs_multilead_with_meta
from ecgfeat.models import STANDARD_12_LEADS

PROTOCOL = dict(fs_original=250, fs_internal=500, core_seconds=30, context_seconds=2,
                mains_hz=50, tolerance_ms=[75, 50], input_unit="V", conversion_to_mv=1000,
                scope="QRS detector with production preprocessing; no later pipeline rescue",
                cables=dict(columns=[1,2], slots=[1,7], reference="annotation_cables.tsv"),
                chest_strap=dict(columns=[0], slots=[1], reference="annotation_cs.tsv"),
                cross_device_fusion="disabled; device streams not aligned at sample level",
                annotation_origin="zero-based samples, as used by upstream Python API",
                missing_annotations="unscorable streams explicitly reported", no_annotation_time_shift=True)


def load_stream(path, stream):
    data = np.loadtxt(path / "ECG.tsv")
    if data.ndim != 2 or data.shape[1] != 6 or not np.isfinite(data).all():
        raise ValueError("GUDB expects six finite raw columns")
    config = PROTOCOL[stream]
    ecg = np.zeros((12, len(data)))
    ecg[config["slots"]] = data[:, config["columns"]].T*1000
    truth = np.atleast_1d(np.loadtxt(path / config["reference"]))
    if (not np.isfinite(truth).all() or np.any(truth != np.floor(truth))
            or np.any(truth < 0) or np.any(truth >= len(data)) or np.any(np.diff(truth) <= 0)):
        raise ValueError("invalid R annotations")
    return ecg, truth.astype(int)


def evaluate(task):
    path, stream, variant, out, fingerprint = task
    path = Path(path)
    rid = path.parent.name + "/" + path.name
    started = time.perf_counter()
    signal, truth = load_stream(path, stream)
    detections, failures = [], []
    slots = PROTOCOL[stream]["slots"]
    for a,b,c,d in windows(signal.shape[1], 250):
        try:
            measured = analysis_signal(resample_ecg(signal[:,c:d],250,500),500,mains_hz=50)
            detection = lowpass_filter(measured,500,40,order=4)
            quality = {STANDARD_12_LEADS[i]: SimpleNamespace(b_sqi=_bsqi(measured[i],500)) for i in slots}
            result = detect_qrs_multilead_with_meta(detection,500,leads=slots,quality=quality,
                                                   adaptive_consensus=variant=="adaptive")
            detections.extend(round(p/2)+c for p in result.r_locs if a<=round(p/2)+c<b)
        except Exception as exc:
            failures.append(dict(start=a,end=b,error=f"{type(exc).__name__}: {exc}"))
    rows, matches = [], []
    for tolerance in (75,50):
        row, paired = score(rid,variant,"QRS",stream,[WaveEvent(None,int(p),None) for p in truth],
                            [WaveEvent(None,p,None) for p in sorted(set(detections))],250,
                            "partial_failure" if failures else "ok", tolerance,
                            extra=dict(activity=path.name,subject=path.parent.name,protocol=f"peak_{tolerance}ms"))
        rows.append(row); matches.extend(paired)
    result = dict(record=rid,stream=stream,variant=variant,rows=rows,matches=matches,
                  failures=failures,seconds=time.perf_counter()-started,fingerprint=fingerprint)
    target=Path(out)/"checkpoints"/f"{rid.replace('/', '_')}__{stream}__{variant}.json"
    temporary=target.with_suffix(".tmp"); temporary.write_text(json.dumps(result)); temporary.replace(target)
    return rid,stream,variant,len(failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"data/external_ecg/gudb")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    directories=sorted(args.root.glob("*/docs/experiment_data/subject_*/*/ECG.tsv"))
    if len(directories)!=125 or args.workers<1:
        parser.error("expected 125 GUDB recordings and positive workers")
    args.out.mkdir(parents=True,exist_ok=True); (args.out/"checkpoints").mkdir(exist_ok=True)
    files=[*(ROOT/"feature_extraction/ecgfeat").rglob("*.py"),Path(__file__),
           ROOT/"scripts/evaluate_ecgfeat_external.py",ROOT/"compare_ludb_detectors.py"]
    manifest=dict(protocol=PROTOCOL,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  data_sha256={str(p.relative_to(args.root)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in args.root.rglob("*.tsv")})
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest["fingerprint"]=fingerprint
    target=args.out/"manifest.json"
    if target.exists() and json.loads(target.read_text())!=manifest:
        parser.error("source/data/protocol changed; choose a new output")
    target.write_text(json.dumps(manifest,indent=2))
    tasks=[]; missing=[]
    for file in directories:
        for stream in ("cables","chest_strap"):
            if not (file.parent/PROTOCOL[stream]["reference"]).exists():
                missing.append(dict(record=str(file.parent.relative_to(file.parents[2])),stream=stream))
                continue
            for variant in ("default","adaptive"):
                rid=file.parent.parent.name+"_"+file.parent.name
                checkpoint=args.out/"checkpoints"/f"{rid}__{stream}__{variant}.json"
                if checkpoint.exists():
                    if json.loads(checkpoint.read_text())["fingerprint"]!=fingerprint:
                        parser.error("stale checkpoint")
                    continue
                tasks.append((str(file.parent),stream,variant,str(args.out),fingerprint))
    with ProcessPoolExecutor(args.workers) as pool:
        for n,result in enumerate(pool.map(evaluate,tasks),1):
            print(n,len(tasks),result,flush=True)
    rows=[]; matches=[]; failures=[]; timings=[]
    for file in sorted((args.out/"checkpoints").glob("*.json")):
        item=json.loads(file.read_text());rows.extend(item["rows"]);matches.extend(item["matches"])
        if item["failures"]:failures.append(item)
        timings.append({k:item[k] for k in ("record","stream","variant","seconds")})
    write_csv(args.out/"detection_by_record.csv",rows);write_csv(args.out/"matched_events.csv",matches)
    summary=dict(protocol=PROTOCOL,metrics=aggregate_metrics(rows,matches,("method","wave","lead","protocol","activity")),
                 overall=aggregate_metrics(rows,matches,("method","wave","lead","protocol")),
                 missing_annotation_streams=missing,failures=failures,timings=timings,completed_tasks=len(timings))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    return int(bool(failures))

if __name__=="__main__":
    raise SystemExit(main())

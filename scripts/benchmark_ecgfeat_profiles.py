#!/usr/bin/env python3
"""Sequential warmed profile costs on fixed signals (run without other jobs)."""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_ecgfeat_external import adapt_signal
from scripts.evaluate_ecgfeat_candidates import variant_options
from ecgfeat import ECGFeatureExtractor
import wfdb


def main():
    cases=[("but-pdb","01",0,["limited","limited_p","limited_qrs","limited_robust"]),
           ("but-pdb","13",0,["limited","limited_robust"]),
           ("nstdb","118e00",300,["default","qrs_adaptive_consensus"]),
           ("isp","test_data/1",0,["default","t_sequence_offset_only","classical_offsets"])]
    rows=[]
    for dataset,record,start,variants in cases:
        folder=ROOT/"data/external_ecg"/dataset
        if dataset=="isp":folder=folder/"isp_delineation_dataset"
        raw=wfdb.rdrecord(str(folder/record))
        signal=adapt_signal(raw,dataset)
        signal=signal[:,round(start*raw.fs):min(signal.shape[1],round((start+32)*raw.fs))]
        for variant in variants:
            extractor=ECGFeatureExtractor(fs_internal=500,mains_freq=50 if dataset=="isp" else 60,**variant_options(variant))
            limited=extractor.input_mode=="limited"
            values=signal[[1,7]] if limited else signal
            kwargs={"lead_names":["channel_0","channel_1"]} if limited else {}
            extractor.extract(values,fs=raw.fs,**kwargs)
            times=[];cpus=[]
            for _ in range(3):
                wall=time.perf_counter();cpu=time.process_time()
                result=extractor.extract(values,fs=raw.fs,**kwargs)
                times.append(time.perf_counter()-wall);cpus.append(time.process_time()-cpu)
            row=dict(dataset=dataset,record=record,variant=variant,signal_seconds=signal.shape[1]/raw.fs,
                     seconds=times,cpu_seconds=cpus,median_seconds=statistics.median(times),beats=len(result.beats))
            rows.append(row);print(dataset,record,variant,row["median_seconds"],flush=True)
    (ROOT/"ecgfeat_robust_20260922/profile_runtime.json").write_text(json.dumps(
        dict(warmup=1,repeats=3,threads=1,execution="sequential; no concurrent benchmark jobs",records=rows),indent=2))

if __name__=="__main__":main()

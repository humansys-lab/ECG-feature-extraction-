#!/usr/bin/env python3
"""Cache annotation-free development extraction for atrial-filter ablations.

Only the eight previously inspected difficult BUT records are used here.
Annotations are read later by scoring, never by extraction or validation.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ecgfeat_external import windows, adapt_signal
from ecgfeat.compat.api_v0 import ECGFeatureExtractor
from ecgfeat.preprocess import resample_ecg, analysis_signal
import wfdb

def run(record):
    data = wfdb.rdrecord(str(ROOT / "data/external_ecg/but-pdb" / record))
    signal = adapt_signal(data, "but-pdb")
    rows = []
    for a,b,c,d in windows(data.sig_len, data.fs):
        result = ECGFeatureExtractor(fs_internal=500, mains_freq=60).extract(signal[:, c:d], fs=data.fs)
        measured = analysis_signal(resample_ecg(signal[:,c:d], data.fs,500),500,mains_hz=60)
        rows.append(dict(core=(a,b), context_start=c, fs_original=data.fs, ecg=measured,
                         events=result.metadata["rhythm_analysis"]["atrial_events"],
                         features=result.beat_features, quality=result.quality))
    folder = ROOT / "ecgfeat_robust_20260922/probes/cache"
    folder.mkdir(exist_ok=True)
    (folder / (record + ".pkl")).write_bytes(pickle.dumps(rows))
    return record

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as pool:
        for name in pool.map(run, ["01", "08", "10", "13", "22", "30", "42", "48"]):
            print(name, flush=True)

#!/usr/bin/env python3
"""Serial warm end-to-end latency, on fixed first/last records of each new set."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import sys
import time
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ecgfeat_cross_dataset import activate_source, adapt_record, pipeline_events
import numpy as np
import scipy
import wfdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/'data/external_ecg')
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    activate_source(args.source)
    from ecgfeat import RefinementConfig
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor
    from ecgfeat.validation import validate_ecg_input
    results, profiles = [], []
    for dataset in ('incartdb','svdb','pwave'):
        records = sorted((args.root/dataset).glob('*.hea'))
        for header in (records[0],records[-1]):
            fs = wfdb.rdheader(str(header.with_suffix(''))).fs
            record = wfdb.rdrecord(str(header.with_suffix('')), sampto=round(32*fs))
            signal, names = adapt_record(record,dataset)
            limited = dataset != 'incartdb'
            _,_,contract = validate_ecg_input(signal,fs,500,lead_names=names,allow_limited_leads=limited)
            slot = contract['lead_slot_map'][names[0]] if limited else 'II'
            extractors = {variant: ECGFeatureExtractor(fs_internal=500, mains_freq=50 if dataset=='incartdb' else 60,
                input_mode='limited' if limited else 'standard', refinement=RefinementConfig(
                    qrs_adaptive_consensus=variant=='enhanced', p_pathology_candidates=variant=='enhanced',
                    atrial_event_validation=variant=='enhanced', t_sequence_offset_only=variant=='enhanced'))
                for variant in ('default','enhanced')}
            timings, fingerprints = {v:[] for v in extractors}, {v:[] for v in extractors}
            for repeat in range(4):
                for variant in (('default','enhanced') if repeat%2==0 else ('enhanced','default')):
                    started=time.perf_counter()
                    result=extractors[variant].extract(signal,fs=fs,lead_names=names,amplitude_unit='mV')
                    elapsed=time.perf_counter()-started
                    if repeat:
                        timings[variant].append(elapsed)
                    events=pipeline_events(result,slot)
                    fingerprints[variant].append(hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest())
            for variant,values in timings.items():
                if len(set(fingerprints[variant])) != 1:
                    raise AssertionError(f'nondeterministic detections: {dataset}/{header.stem}/{variant}')
                item=dict(dataset=dataset, record=header.stem, variant=variant, input_seconds=32,
                          repetitions_seconds=values, median_seconds=float(np.median(values)),
                          real_time_factor=float(np.median(values))/32,
                          stable_prediction_sha256=fingerprints[variant][0])
                results.append(item);print(json.dumps(item),flush=True)
            if header == records[0]:
                profiler=cProfile.Profile()
                profiler.enable()
                extractors['enhanced'].extract(signal,fs=fs,lead_names=names,amplitude_unit='mV')
                profiler.disable()
                profiler.dump_stats(str(args.out.with_name(f'profile_{dataset}_{header.stem}.prof')))
                stats=pstats.Stats(profiler)
                functions=[]
                for (filename,line,function),(primitive,calls,self_time,cumulative,_) in stats.stats.items():
                    if 'ecgfeat' in filename:
                        functions.append(dict(file=Path(filename).name,line=line,function=function,
                                              calls=calls,self_seconds=self_time,cumulative_seconds=cumulative))
                profiles.append(dict(dataset=dataset,record=header.stem,variant='enhanced',
                                     note='cProfile instrumentation adds overhead; cumulative times overlap',
                                     functions=sorted(functions,key=lambda f:-f['cumulative_seconds'])[:30]))
    output=dict(protocol='one process, BLAS/OMP one thread, one warmup then three timed repetitions per config; alternating order',
                excludes='WFDB file loading and scoring; includes validation/resampling/all extraction stages',
                scope='first 32 seconds of lexicographically first and last records, all three datasets',
                environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                 wfdb=wfdb.__version__, platform=platform.platform(),cpu_count=os.cpu_count(),
                                 available_cpu_count=len(os.sched_getaffinity(0))),
                source_sha256={str(p.relative_to(args.source)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((args.source/'feature_extraction/ecgfeat').rglob('*.py'))},
                results=results, profiles=profiles)
    args.out.write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Resume missing tasks using an archived harness with the original fingerprint."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--harness',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    manifest=json.loads((args.out/'manifest.json').read_text())
    source=Path(manifest['source_path']);data=Path(manifest['data_root'])
    for key,digest in manifest['source_sha256'].items():
        assert sha(source/key)==digest, f'source changed: {key}'
    for key,digest in manifest['data_sha256'].items():
        assert sha(data/key)==digest, f'data changed: {key}'
    for key,digest in manifest['harness_sha256'].items():
        path=args.harness if key=='scripts/evaluate_ecgfeat_cross_dataset.py' else ROOT/key
        assert sha(path)==digest, f'harness changed: {key}'
    for key,digest in manifest['overlap_metadata_sha256'].items():
        assert sha(ROOT/key)==digest, f'overlap metadata changed: {key}'
    spec=importlib.util.spec_from_file_location('frozen_external_evaluator',args.harness)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    tasks=[]
    for item in manifest['tasks']:
        path=args.out/'checkpoints'/('__'.join(item[k] for k in ('dataset','record','scope','variant'))+'.json')
        if path.exists():
            assert json.loads(path.read_text())['fingerprint']==manifest['fingerprint']
            continue
        tasks.append(dict(**item,checkpoint=str(path),fingerprint=manifest['fingerprint'],
                          source=str(source),data_root=str(data)))
    print('resuming',len(tasks),'tasks',flush=True)
    started=time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(module.evaluate,task) for task in tasks]
        for future in as_completed(futures):
            print(json.dumps(future.result()),flush=True)
    summary=module.summarize(args.out)
    execution=dict(wall_seconds=time.time()-(args.out/'manifest.json').stat().st_mtime,
                   expected_tasks=len(manifest['tasks']),newly_executed_tasks=len(manifest['tasks']),
                   workers_initial=8,workers_resumption=args.workers,resumed_tasks=len(tasks),
                   resumption_seconds=time.perf_counter()-started,initial_process_exit_code=143,
                   note='initial process received SIGTERM; only missing tasks rerun with hash-verified original harness, source and data')
    (args.out/'execution.json').write_text(json.dumps(execution,indent=2))
    assert summary['completed_tasks']==len(manifest['tasks']) and not summary['failures']


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Merge valid initial PWAVE tasks with the two duplicate-annotation repairs.

Preserve original checkpoints/fingerprints. Re-score EVERY saved prediction with
the audited parser and require exactly unchanged rows and event matches. This
permits reuse without rerunning identical extraction on the ten unaffected
records, and makes the scoring-only protocol amendment explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_ecgfeat_cross_dataset import annotations, score, WaveEvent, summarize, PROTOCOL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'ecgfeat_cross_dataset_20260922')
    args=parser.parse_args(); out=args.out
    initial=out/'pwave'; repair=out/'pwave_annotation_fix'
    manifests=[json.loads((folder/'manifest.json').read_text()) for folder in (initial,repair)]
    frozen=json.loads((out/'production_source_sha256.json').read_text())
    assert all(m['source_sha256']==frozen for m in manifests)
    assert manifests[0]['data_sha256']==manifests[1]['data_sha256']
    a,b=[dict(m['protocol']) for m in manifests]
    for p in (a,b):
        p.pop('version');p.pop('duplicate_reference_policy',None)
    assert a==b, 'changes beyond duplicate reference handling'
    expected={tuple(task[k] for k in ('dataset','record','scope','variant')) for task in manifests[0]['tasks']}
    seen=set(); components=[]; checked_rows=0; checked_matches=0
    for folder,manifest in zip((initial,repair),manifests):
        paths=sorted((folder/'checkpoints').glob('*.json'))
        components.append(dict(manifest=str((folder/'manifest.json').relative_to(out)),
                               fingerprint=manifest['fingerprint'],completed_tasks=len(paths),
                               checkpoints={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
        for path in paths:
            item=json.loads(path.read_text())
            key=tuple(item[k] for k in ('dataset','record','scope','variant'))
            assert key not in seen and key in expected
            seen.add(key)
            assert item['fingerprint']==manifest['fingerprint']
            assert not item['failures']
            root=Path(manifest['data_root'])/'pwave'
            truth={}
            for wave,extension,base in [('P','pwave',root/item['record']),('QRS','atr',root/'mitdb_reference'/item['record'])]:
                truth[wave]=annotations(base,extension,item['samples'],wave)[0]
            rebuilt_matches=[]
            for old in item['rows']:
                extra={k:old[k] for k in ('dataset','scope','protocol','patient','prior_source_overlap')}
                predictions=[WaveEvent(None,p,None) for p in item['predictions'][old['lead']]]
                row,matches=score(item['record'],item['variant'],old['wave'],old['lead'],truth[old['wave']],
                                  predictions,item['fs'],old['status'],old['tolerance_ms'],extra=extra)
                assert row==old, f'row changed: {path.name}/{old["lead"]}'
                rebuilt_matches.extend(matches);checked_rows+=1;checked_matches+=len(matches)
            assert rebuilt_matches==item['matches'], f'matches changed: {path.name}'
    assert seen==expected and len(seen)==24
    assert [c['completed_tasks'] for c in components]==[20,4]
    summary=summarize(initial,[initial/'checkpoints',repair/'checkpoints'])
    audit=dict(protocol=PROTOCOL, components=components, unchanged_scoring_rows=checked_rows,
               unchanged_scoring_matches=checked_matches, unique_complete_tasks=len(seen),
               production_source_unchanged=True, amendment='119:77 and 214:1 identical P-reference duplicates coalesced; no extraction change')
    (initial/'composition_manifest.json').write_text(json.dumps(audit,indent=2))
    (initial/'execution.json').write_text(json.dumps(dict(
        wall_seconds=time.time()-(initial/'manifest.json').stat().st_mtime,
        expected_tasks=24,newly_executed_tasks=24,workers_initial=8,workers_annotation_repair=4,
        initial_reference_validation_aborts=4,final_failed_tasks=len(summary['failures']),
        note='wall time includes initial run, overlap with repair run, and validation/merge; inspect individual extraction timings for latency'),indent=2))
    print(json.dumps({key:audit[key] for key in ('unique_complete_tasks','unchanged_scoring_rows','unchanged_scoring_matches')}))


if __name__=='__main__':
    main()

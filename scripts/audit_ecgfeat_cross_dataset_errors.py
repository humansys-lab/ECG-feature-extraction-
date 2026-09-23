#!/usr/bin/env python3
"""Post-hoc inspection of a timing regression; preserves frozen extraction."""
import json
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_ecgfeat_cross_dataset import activate_source, adapt_record, LEADS
import numpy as np
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=ROOT/'ecgfeat_cross_dataset_20260922'
    activate_source(out/'frozen_source')
    from ecgfeat.validation import validate_ecg_input
    from ecgfeat.preprocess import resample_ecg, analysis_signal, lowpass_filter
    from ecgfeat.quality import _bsqi
    from ecgfeat.qrs import detect_qrs_multilead_with_meta
    dataset,rid='svdb','862'
    checkpoints={variant:json.loads((out/dataset/'checkpoints'/f'{dataset}__{rid}__detector_full__{variant}.json').read_text())
                 for variant in ('default','enhanced')}
    late=[m for m in checkpoints['enhanced']['matches'] if m['protocol']=='peak_150ms' and abs(m['peak_error_ms'])>75]
    first=late[0]['gt_peak_sample'];fs=128
    a=(first//(30*fs))*30*fs;b=a+30*fs;c=max(0,a-2*fs);d=b+2*fs
    record=wfdb.rdrecord(str(ROOT/'data/external_ecg'/dataset/rid),sampfrom=c,sampto=d)
    signal,names=adapt_record(record,dataset)
    normalized,_,contract=validate_ecg_input(signal,fs,500,lead_names=names,allow_limited_leads=True)
    measured=analysis_signal(resample_ecg(normalized,fs,500),500,mains_hz=60)
    detect=lowpass_filter(measured,500,cutoff_hz=40,order=4)
    slots=tuple(LEADS.index(name) for name in contract['available_leads'])
    qualities={LEADS[i]:SimpleNamespace(b_sqi=_bsqi(measured[i],500)) for i in slots}
    results={}
    for variant in checkpoints:
        result=detect_qrs_multilead_with_meta(detect,500,leads=slots,quality=qualities,
                                            adaptive_consensus=variant=='enhanced')
        predictions=sorted(set(round(p*fs/500)+c for p in result.r_locs if a<=round(p*fs/500)+c<b))
        saved=[p for p in checkpoints[variant]['predictions']['QRS'] if a<=p<b]
        assert predictions==saved, 'inspection must exactly reproduce original predictions'
        candidate=min(result.qrs_candidate_windows,key=lambda candidate:abs(round(candidate.refined_r_index*fs/500)+c-first))
        candidate_audit=asdict(candidate)
        candidate_audit['absolute_original_samples']={name:round(getattr(candidate,name)*fs/500)+c
            for name in ('detector_peak_index','search_start_index','search_end_index','refined_r_index')}
        results[variant]=dict(predictions=predictions,detection_leads=result.detection_leads,
                              fallback_reason=result.fallback_reason,energy_threshold=result.energy_threshold,
                              first_shifted_reference_candidate=candidate_audit)
    audit=dict(dataset=dataset,record=rid,selection='first enhanced error >75ms that still matches within 150ms; post hoc',
               core_seconds=[a/fs,b/fs],input_seconds=[c/fs,d/fs],lead_slot_map=contract['lead_slot_map'],
               b_sqi={name:q.b_sqi for name,q in qualities.items()},exact_saved_predictions_reproduced=True,
               results=results,full_record_shifted_matches_at_150ms=len(late),
               first_shifted_matches=[{k:m[k] for k in ('gt_peak_sample','det_peak_sample','peak_error_ms')} for m in late[:6]])
    (out/'svdb/862_timing_audit.json').write_text(json.dumps(audit,indent=2))
    annotation=wfdb.rdann(str(ROOT/'data/external_ecg'/dataset/rid),'atr')
    reference=[int(p) for p,s in zip(annotation.sample,annotation.symbol) if s in set('NLRBAaJSVrFejnE/fQ?')]
    lo,hi=first-2*fs,first+4*fs
    fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True,layout='constrained')
    for channel,ax in enumerate(axes):
        t=(np.arange(record.sig_len)+c)/fs
        ax.plot(t,signal[channel],color='#263238',lw=.8,label=names[channel])
        for points,color,label,marker in [(reference,'#2e7d32','Reference QRS','o'),
            (results['default']['predictions'],'#6b7c93','Default','+'),
            (results['enhanced']['predictions'],'#e65100','Enhanced','x')]:
            p=np.array([p for p in points if lo<=p<hi])
            ax.scatter(p/fs,signal[channel,p-c],color=color,label=label,marker=marker,s=45,zorder=5)
        ax.set(xlim=(lo/fs,hi/fs),ylabel='mV');ax.grid(alpha=.15);ax.legend(ncol=4,loc='upper right')
    axes[-1].set_xlabel('Original recording time (s)')
    axes[0].set_title('SVDB 862: QRS timing regression (post hoc; no time shift or algorithm tuning)')
    for extension in ('png','svg'):
        fig.savefig(out/f'svdb_862_timing_audit.{extension}',dpi=180)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()

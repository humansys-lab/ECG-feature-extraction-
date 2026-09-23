#!/usr/bin/env python3
"""Export static figures for the frozen external evaluation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wfdb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ecgfeat_cross_dataset_20260922'


def main():
    analysis=json.loads((OUT/'analysis.json').read_text())
    metrics=analysis['metrics']
    configurations=[
        ('Full-duration QRS detector (75 ms)', [('INCART','incartdb','detector_full','QRS','peak_75ms'),
                                       ('SVDB','svdb','detector_full','QRS','peak_75ms')]),
        ('End-to-end QRS extraction (75 ms)', [('INCART\n90 s / record','incartdb','pipeline_sampled','QRS','peak_75ms'),
                                       ('SVDB\n90 s / record','svdb','pipeline_sampled','QRS','peak_75ms'),
                                       ('PWAVE\nfull duration','pwave','pipeline_full','QRS','peak_75ms')]),
        ('PWAVE P peaks: timing tolerance', [('150 ms','pwave','pipeline_full','P_native','peak_150ms'),
                                             ('50 ms','pwave','pipeline_full','P_native','peak_50ms')]),
        ('PWAVE P outputs at 150 ms', [('Native','pwave','pipeline_full','P_native','peak_150ms'),
                                       ('Accepted','pwave','pipeline_full','P_accepted','peak_150ms'),
                                       ('Atrial candidates','pwave','pipeline_full','P_atrial','peak_150ms')])]
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    colors={'default':'#6b7c93','enhanced':'#147d92'}
    for ax,(title,items) in zip(axes.flat,configurations):
        x=np.arange(len(items))
        for offset,variant in [(-.19,'default'),(.19,'enhanced')]:
            values=[]
            for _,dataset,scope,lead,protocol in items:
                selected=[m for m in metrics if m['dataset']==dataset and m['scope']==scope and m['lead']==lead
                          and m['protocol']==protocol and m['method']==variant]
                assert len(selected)==1
                values.append(100*selected[0]['f1'])
            bars=ax.bar(x+offset,values,width=.36,color=colors[variant],label=variant.capitalize())
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set(title=title,xticks=x,xticklabels=[v[0] for v in items],ylim=(0,110),ylabel='F1 (%)')
        ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(loc='lower left')
    fig.suptitle('Frozen traditional ECG extraction | external datasets | 2026-09-22',fontsize=15)
    for extension in ('png','svg'):
        fig.savefig(OUT/f'validation_results.{extension}',dpi=180)
    plt.close(fig)

    record=wfdb.rdrecord(str(ROOT/'data/external_ecg/pwave/103'),sampto=1800)
    pa=wfdb.rdann(str(ROOT/'data/external_ecg/pwave/103'),'pwave')
    qa=wfdb.rdann(str(ROOT/'data/external_ecg/pwave/mitdb_reference/103'),'atr')
    checkpoint=json.loads((OUT/'pwave/checkpoints/pwave__103__pipeline_full__enhanced.json').read_text())
    qrs=[int(p) for p,symbol in zip(qa.sample,qa.symbol) if symbol in set('NLRBAaJSVrFejnE/fQ?')]
    fig,ax=plt.subplots(figsize=(12,3.5),layout='constrained')
    t=np.arange(len(record.p_signal))/360
    ax.plot(t,record.p_signal[:,0],color='#263238',lw=1,label='Raw MLII')
    for points,color,label,marker in [(pa.sample,'#2e7d32','Reference P','o'),
        (checkpoint['predictions']['P_native'],'#e65100','Detected P','x'),(qrs,'#1565c0','Reference QRS','v')]:
        p=np.array([x for x in points if 0<=x<len(t)])
        ax.scatter(p/360,record.p_signal[p,0],color=color,label=label,marker=marker,s=40,zorder=5)
    ax.set(xlabel='Time (s)',ylabel='mV',title='PWAVE 103: P peak localization (post-hoc inspection; no tuning)')
    ax.legend(ncol=4,loc='upper center');ax.grid(alpha=.15)
    for extension in ('png','svg'):
        fig.savefig(OUT/f'pwave_103_peak_audit.{extension}',dpi=180)


if __name__=='__main__':
    main()

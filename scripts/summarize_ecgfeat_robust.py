#!/usr/bin/env python3
"""Paired record/subject summaries for the traditional refinement experiment."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

DEVELOPMENT={"01","08","10","13","22","30","42","48"}


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def pooled(rows):
    values={key:sum(int(row[key]) for row in rows) for key in ("tp","fp","fn")}
    t,f,n=(values[k] for k in ("tp","fp","fn"))
    return dict(values,f1=2*t/(2*t+f+n) if 2*t+f+n else None,
                sensitivity=t/(t+n) if t+n else None,precision=t/(t+f) if t+f else None)


def paired_f1(rows, baseline, candidate, group="record"):
    names=sorted({row[group] for row in rows})
    def counts(method):
        return np.array([[sum(int(row[k]) for row in rows if row["method"]==method and row[group]==name)
                          for k in ("tp","fp","fn")] for name in names],dtype=float)
    a,b=counts(baseline),counts(candidate)
    def f1(x):
        t,f,n=np.moveaxis(x,-1,0)
        return np.divide(2*t,2*t+f+n,out=np.zeros_like(t),where=(2*t+f+n)>0)
    rng=np.random.default_rng(20260922)
    index=rng.integers(0,len(names),(5000,len(names)))
    distribution=f1(b[index].sum(axis=1))-f1(a[index].sum(axis=1))
    return dict(baseline=baseline,candidate=candidate,unit=group,n=len(names),
                pooled_f1_difference=float(f1(b.sum(axis=0))-f1(a.sum(axis=0))),
                ci95=np.quantile(distribution,[.025,.975]).tolist(),bootstrap_replicates=5000)


def summarize(root):
    output={}
    rows=csv_rows(root/"but_final/detection_by_record.csv")
    guarded=csv_rows(root/"but_release/detection_by_record.csv")
    replaced={r["method"] for r in guarded}
    rows=[r for r in rows if r["method"] not in replaced]+guarded
    output["but"]={}
    variants=sorted({r["method"] for r in rows})
    for cohort in ("all","development","remainder"):
        selected=[r for r in rows if cohort=="all" or (r["record"] in DEVELOPMENT)==(cohort=="development")]
        table=[]
        for wave,lead,tol in [("P",lead,tol) for lead in ("II","atrial_events","accepted_native") for tol in (50,150)]+[("QRS","II",75)]:
            scoped=[r for r in selected if r["wave"]==wave and r["lead"]==lead and r["protocol"]==f"peak_{tol}ms"]
            for variant in variants:
                table.append(dict(variant=variant,wave=wave,lead=lead,tolerance_ms=tol,
                                  **pooled([r for r in scoped if r["method"]==variant])))
        output["but"][cohort]=table
    output["but_paired_f1"]=[]
    for wave,lead,tol in [("P","II",50),("P","II",150),("P","atrial_events",50),("P","atrial_events",150),("QRS","II",75)]:
        selected=[r for r in rows if r["wave"]==wave and r["lead"]==lead and r["protocol"]==f"peak_{tol}ms"]
        for a,b in [("default","robust_p"),("limited","limited_p"),("limited","limited_qrs"),("limited","limited_robust")]:
            output["but_paired_f1"].append(dict(wave=wave,lead=lead,tolerance_ms=tol,**paired_f1(selected,a,b)))
    output["but_no_p"]= [r for r in rows if r["record"] in {"08","48"} and r["wave"]=="P" and r["protocol"]=="peak_150ms"]
    output["nstdb"]=[]
    for path in [root.parent/"ecgfeat_external_20260922/nstdb/summary.json",root/"nstdb_release/summary.json"]:
        output["nstdb"].extend(json.loads(path.read_text())["metrics"])
    output["isp"]=json.loads((root/"isp_t_final/summary.json").read_text())["metrics"]
    matches=csv_rows(root/"isp_t_final/matched_events.csv")
    output["isp_paired_t_offset"]=[]
    for split in ("train","test"):
        selected=[r for r in matches if r["wave"]=="T" and r["lead"]=="consensus_median" and r["split"]==split and r["offset_error_ms"]]
        def errors(method):
            return {(r["record"],r["gt_onset_sample"],r["gt_offset_sample"]):abs(float(r["offset_error_ms"]))
                    for r in selected if r["method"]==method}
        a=errors("default")
        for variant in ("t_projection_offset_only","t_sequence_offset_only","classical_offsets"):
            b=errors(variant);common=a.keys()&b.keys();names=sorted({k[0] for k in common})
            record_delta=np.array([np.mean([b[k]-a[k] for k in common if k[0]==name]) for name in names])
            rng=np.random.default_rng(20260922)
            ci=np.quantile(record_delta[rng.integers(0,len(names),(5000,len(names)))].mean(axis=1),[.025,.975])
            output["isp_paired_t_offset"].append(dict(split=split,variant=variant,common_matches=len(common),
                lost_matches=len(a.keys()-b.keys()),gained_matches=len(b.keys()-a.keys()),
                paired_beat_mean_delta_ms=float(np.mean([b[k]-a[k] for k in common])),
                paired_record_mean_delta_ms=float(record_delta.mean()),record_bootstrap_ci95=ci.tolist()))
    gudb=json.loads((root/"gudb_release/summary.json").read_text())
    output["gudb"]={key:gudb[key] for key in ("overall","metrics","missing_annotation_streams","completed_tasks","failures")}
    rows=csv_rows(root/"gudb_release/detection_by_record.csv")
    output["gudb_paired_f1"]=[]
    for stream in ("cables","chest_strap"):
        for tol in (50,75):
            selected=[r for r in rows if r["lead"]==stream and r["protocol"]==f"peak_{tol}ms"]
            output["gudb_paired_f1"].append(dict(stream=stream,tolerance_ms=tol,**paired_f1(selected,"default","adaptive",group="subject")))
    output["ludb"]={}
    for variant,path in [("default",root/"default_regression/summary.json"),
                         ("robust_p",root/"ludb_ablation/robust_p/summary.json"),
                         ("t_sequence_offset_only",root/"ludb_ablation/t_sequence_offset_only/summary.json"),
                         ("qrs_adaptive_consensus",root/"ludb_release/qrs_adaptive_consensus/summary.json")]:
        output["ludb"][variant]=json.loads(path.read_text())
    (root/"analysis.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("ecgfeat_robust_20260922"))
    args=parser.parse_args();summarize(args.root)

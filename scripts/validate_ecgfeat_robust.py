#!/usr/bin/env python3
"""Check completeness, source freeze, known regression and retained outputs."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
root=ROOT/"ecgfeat_robust_20260922"


def read(path):
    return json.loads(path.read_text())


def main():
    failures=[];experiments=[]
    for name,count in [("but_final",300),("but_release",100),("nstdb_release",12),
                       ("isp_t_final",1900),("gudb_release",458)]:
        summary=read(root/name/"summary.json")
        passed=summary["completed_tasks"]==count and not summary["failures"]
        experiments.append(dict(name=name,expected_tasks=count,completed_tasks=summary["completed_tasks"],passed=passed))
        if not passed:failures.append(name)
    for name,count in [("default_regression",200),("ludb_ablation/robust_p",200),
                       ("ludb_ablation/t_sequence_offset_only",200),("ludb_release/qrs_adaptive_consensus",200)]:
        records=[json.loads(line) for line in (root/name/"records.jsonl").read_text().splitlines()]
        passed=len(records)==count and not any(r["failures"] for r in records)
        experiments.append(dict(name=name,expected_tasks=count,completed_tasks=len(records),passed=passed))
        if not passed:failures.append(name)
    manifest=read(root/"but_release/manifest.json")
    source_differences=[p for p,h in manifest["source_sha256"].items()
                        if p.startswith("feature_extraction/ecgfeat/") and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if source_differences:failures.append("source_modified_after_release_freeze")
    regression=read(root/"default_regression/regression_gate.json")
    if not regression["passed"]:failures.append("default_ludb_regression")
    external=[]
    for dataset,folder in [("isp","isp_t_final"),("but-pdb","but_final")]:
        changes=[];count=0
        for p in (ROOT/"ecgfeat_external_20260922"/dataset/"checkpoints").glob("default__*.json"):
            a=read(p)["predictions"];b=read(root/folder/"checkpoints"/p.name)["predictions"];count+=1
            changes.extend([p.name,lead] for lead in a if a[lead]!=b[lead])
        if changes or count!={"isp":475,"but-pdb":50}[dataset]:failures.append(dataset+"_default_equivalence")
        external.append(dict(dataset=dataset,records=count,changed_outputs=changes))
    clean=[]
    for p in (ROOT/"ecgfeat_external_20260922/nstdb/checkpoints").glob("default__*.json"):
        q=root/"nstdb_release/checkpoints"/p.name.replace("default__","qrs_adaptive_consensus__")
        def clean_peaks(path):
            data=read(path);fs=data["fs"]
            return [e["peak"] for e in data["predictions"]["II"]["QRS"]
                    if e["peak"]/fs<300 or int((e["peak"]/fs-300)//120)%2!=0]
        a,b=clean_peaks(p),clean_peaks(q)
        clean.append(dict(record=p.stem,identical=a==b,lost=len(set(a)-set(b)),added=len(set(b)-set(a))))
        if a!=b:failures.append("clean_QRS_changed:"+p.stem)
    if len(clean)!=12:failures.append("missing_nstdb_clean_comparison")
    external_report=dict(default_external_equivalence=external,nstdb_clean_peak_equivalence=clean)
    (root/"external_regression.json").write_text(json.dumps(external_report,indent=2))
    block=read(root/"but_release/checkpoints/limited_robust__13.json")
    qrs=next(r for r in block["rows"] if r["wave"]=="QRS")
    if (qrs["tp"],qrs["fp"],qrs["fn"])!=(89,0,0):failures.append("nonconducted_p_misclassified_as_qrs")
    cache=read(root/"p_cache_equivalence.json")
    if not all(r["debug_equal"] and r["summary_equal"] for r in cache["records"]):failures.append("p_cache_not_equivalent")
    test_log=(root/"tests_release.log").read_text()
    test_match=re.search(r"(\d+) passed, (\d+) skipped,.*?(\d+) subtests passed",test_log)
    if test_match is None or "FAILED " in test_log:failures.append("feature_test_suite_not_passed")
    tests=dict(passed=int(test_match[1]),skipped=int(test_match[2]),subtests_passed=int(test_match[3])) if test_match else {}
    tests.update(scope="95 feature/extraction/evaluator modules; ecgagent and batch_medgemma excluded",
                 skip_reason="Two existing JS00059 regressions require separately generated demo artifacts")
    report=dict(passed=not failures,failures=failures,experiments=experiments,
                effective_final_comparison_tasks=3470,
                source_modified_after_freeze=source_differences,default_ludb=regression,
                external_regression=external_report,but13_qrs={k:qrs[k] for k in ("tp","fp","fn")},
                equivalent_p_cache_records=len(cache["records"]),
                unit_tests=tests,
                test_logs={"feature_suite":"tests_release.log","limited_export_suite":"tests_limited_export.log"},
                statistical_caveats=["BUT records can share source patients; record bootstrap is descriptive",
                    "NSTDB has two original sources","GUDB first frozen run preceded guard development on BUT13; final GUDB is repeated external validation",
                    "Naive cross-device GUDB fusion was invalidated by clock audit and excluded"])
    (root/"validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=report["passed"],failures=failures,effective_tasks=3470)))
    return int(not report["passed"])


if __name__=="__main__":
    raise SystemExit(main())

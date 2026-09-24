"""Audit ecgfeat outputs for use as conditioning variables in ECG generation.

Diagnostic quality and conditioning quality are different requirements. A
feature that abstains when unsure is safe for diagnosis and dangerous for
conditioning: if it abstains more often on abnormal records, dropping those
rows biases the training distribution, and imputing them invents structure.
So this measures, per candidate feature:

  coverage      non-null rate -- how much data survives if the feature is required
  coverage bias coverage on abnormal minus on normal; non-zero means dropping
                nulls skews the conditional distribution
  degeneracy    share taken by the single most common value (categoricals) or
                coefficient of variation (continuous); a near-constant feature
                gives a generator nothing to steer with
  spread        the usable dynamic range

Domains covered: rhythm, ST, waveform morphology, infarct territory.
"""
import ast
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/workspace/ecg_gemma")
sys.path.insert(0, "/workspace/ecg_gemma/feature_extraction")

import numpy as np
import pandas as pd

PTBXL = Path("/workspace/ecg_gemma/data/ptb-xl")
META = "/workspace/ecg_gemma/data/ptb-xl-metadata-1.0.1/ptbxl_database.csv"
OUT = "/workspace/ecg_gemma/ecgfeat_perf_raw_20260809/cond_feature_audit.csv"
N_PER_GROUP = 50

# record-level (global_features) candidates, by domain
GLOBAL_FIELDS = {
    "rhythm": ["heart_rate_bpm", "rr_cv", "rr_irregularity_class",
               "probable_af", "heart_rate_class", "pr_ms", "p_axis_normal"],
    "morphology": ["qrs_ms", "qt_ms", "qtc_ms", "qrs_axis_deg", "qrs_axis_class",
                   "t_axis_deg", "t_axis_class", "qrs_t_angle_deg", "p_dur_ms",
                   "p_amp_mv"],
}
# per-lead candidates
LEAD_FIELDS = {
    "st": ["st_hybrid_j_mv", "st_hybrid_60ms_mv", "st_hybrid_80ms_mv",
           "st_hybrid_slope_mv_per_ms", "st_hybrid_baseline_confidence",
           "st_pattern_class", "st_morphology"],
    "morphology": ["p_amp_mv", "p_dur_ms", "t_amp_mv", "t_polarity",
                   "qrs_ms", "r_amp_mv", "s_amp_mv", "q_amp_mv",
                   "q_duration_ms", "t_symmetry"],
}


def rpath(i):
    return PTBXL / f"{(i // 1000) * 1000:05d}" / f"{i:05d}_hr"


def process(args):
    ecg_id, group = args
    import wfdb
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    try:
        rec = wfdb.rdrecord(str(rpath(ecg_id)))
        feats = ECGFeatureExtractor(fs_internal=500, mains_freq=50.0).extract(
            np.asarray(rec.p_signal, dtype=float).T, fs=float(rec.fs)
        )
        row = {"ecg_id": ecg_id, "group": group}
        g = feats.global_features
        for dom, fields in GLOBAL_FIELDS.items():
            for f in fields:
                row[f"G::{dom}::{f}"] = getattr(g, f, None)

        # per-lead: record the fraction of the 12 leads that produced a value,
        # so a lead-wise conditioning vector's completeness is visible
        by_lead = {}
        for bf in feats.beat_features:
            by_lead.setdefault(bf.lead, []).append(bf)
        for dom, fields in LEAD_FIELDS.items():
            for f in fields:
                present, vals = 0, []
                for lead, items in by_lead.items():
                    v = [getattr(i, f, None) for i in items]
                    v = [x for x in v if x is not None and
                         (not isinstance(x, float) or np.isfinite(x))]
                    if v:
                        present += 1
                        if isinstance(v[0], (int, float)) and not isinstance(v[0], bool):
                            vals.append(float(np.median([float(x) for x in v])))
                row[f"L::{dom}::{f}::cov"] = present / 12.0
                if vals:
                    row[f"L::{dom}::{f}::spread"] = float(np.max(vals) - np.min(vals))

        # infarct territory
        interp = getattr(feats, "interpretation", None) or {}
        codes = []
        if isinstance(interp, dict):
            for k in ("statements", "findings", "codes"):
                v = interp.get(k)
                if isinstance(v, list):
                    codes += [str(x.get("code", x)) if isinstance(x, dict) else str(x)
                              for x in v]
        row["G::infarct::n_statements"] = len(codes)
        row["G::infarct::has_mi_code"] = any(
            "mi" in c.lower() or "infarct" in c.lower() or "stemi" in c.lower()
            for c in codes)
        return row
    except Exception as exc:  # noqa: BLE001
        return {"ecg_id": ecg_id, "group": group, "error": str(exc)[:80]}


def main():
    db = pd.read_csv(META, index_col="ecg_id")
    db["scp"] = db.scp_codes.apply(ast.literal_eval)
    av = lambda idx: [int(i) for i in idx if rpath(int(i)).with_suffix(".hea").exists()]
    mi = ("IMI", "AMI", "ASMI", "ILMI", "ALMI", "INJAS", "INJAL", "INJIN",
          "INJLA", "INJIL")
    isch = ("ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ISCAN")
    groups = {
        "NORM": av(db[db.scp.apply(lambda d: "NORM" in d)].index),
        "MI": av(db[db.scp.apply(lambda d: any(k in d for k in mi))].index),
        "ISCHEMIC": av(db[db.scp.apply(lambda d: any(k in d for k in isch))].index),
        "AFIB": av(db[db.scp.apply(lambda d: "AFIB" in d)].index),
        "CD": av(db[db.scp.apply(  # conduction disease: wide/abnormal QRS
            lambda d: any(k in d for k in ("CLBBB", "CRBBB", "IVCD", "LAFB", "1AVB")))].index),
    }
    jobs = [(i, g) for g, ids in groups.items() for i in ids[:N_PER_GROUP]]
    print(f"{len(jobs)} records: " +
          ", ".join(f"{g} {min(len(v), N_PER_GROUP)}" for g, v in groups.items()),
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(process, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if n % 40 == 0:
                print(f"[{n}/{len(jobs)}]", flush=True)

    d = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    ok = d[d.get("error").isna()] if "error" in d else d
    print(f"\nwrote {OUT}   ({len(ok)}/{len(d)} ok)\n")

    norm = ok[ok.group == "NORM"]
    abnorm = ok[ok.group != "NORM"]

    print("=== 记录级特征 ===")
    print(f"{'特征':<42}{'覆盖率':>8}{'缺失偏差':>10}{'退化度':>9}  说明")
    for col in [c for c in ok.columns if c.startswith("G::")]:
        s = ok[col]
        cov = s.notna().mean()
        bias = abnorm[col].notna().mean() - norm[col].notna().mean()
        nn = s.dropna()
        if nn.empty:
            continue
        if nn.dtype == object or nn.dtype == bool:
            top = Counter(nn.astype(str)).most_common(1)[0]
            deg = top[1] / len(nn)
            note = f"最常见值 {top[0][:18]!r}"
        else:
            v = pd.to_numeric(nn, errors="coerce").dropna()
            deg = float("nan") if v.empty or v.mean() == 0 else float(
                1 - min(abs(v.std() / (abs(v.mean()) + 1e-9)), 1))
            note = f"中位={v.median():.3g} 范围={v.min():.3g}~{v.max():.3g}" if len(v) else ""
        flag = "  ⚠缺失有偏" if abs(bias) >= 0.10 else ""
        print(f"{col.replace('G::',''):<42}{cov:>8.2f}{bias:>+10.2f}{deg:>9.2f}  {note}{flag}")

    print("\n=== 逐导联特征:12导联中产出值的比例 ===")
    print(f"{'特征':<42}{'平均覆盖':>10}{'缺失偏差':>10}")
    for col in [c for c in ok.columns if c.startswith("L::") and c.endswith("::cov")]:
        s = pd.to_numeric(ok[col], errors="coerce")
        bias = (pd.to_numeric(abnorm[col], errors="coerce").mean()
                - pd.to_numeric(norm[col], errors="coerce").mean())
        flag = "  ⚠缺失有偏" if abs(bias) >= 0.10 else ""
        print(f"{col.replace('L::','').replace('::cov',''):<42}{s.mean():>10.2f}{bias:>+10.2f}{flag}")

    print("\n  覆盖率=可用样本比例;缺失偏差=异常组减正常组(非零则丢弃空值会扭曲分布);"
          "\n  退化度→1 表示该特征近似常数,对生成没有可控性")


if __name__ == "__main__":
    main()

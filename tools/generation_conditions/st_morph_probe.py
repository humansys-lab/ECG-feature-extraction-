"""Does the ST morphology classifier mean anything on real ECGs?

`classify_st_pattern` names four clinical classes (J-point elevation, slow
upsloping, horizontal, downsloping) and has unit tests, but those only check
the decision tree against constructed inputs -- it has never been compared
with annotated recordings. PTB-XL carries no shape labels, so this tests the
next best thing: the classes' clinical meaning. Horizontal and downsloping ST
depression is the ischemia-specific pattern; upsloping and a notched elevated
J point are usually benign. If the classifier works, those distributions must
separate between ischemic and normal records. If they do not, the classes are
decoration.
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
OUT = "/workspace/ecg_gemma/ecgfeat_perf_raw_20260809/st_morph_probe.csv"
N_PER_GROUP = 40
CLASSES = ("j_point_elevation", "slow_upsloping", "horizontal",
           "downsloping", "unknown")


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
        # One class per lead: the modal class over that lead's beats, so a
        # long record cannot outvote a short one.
        by_lead = {}
        for bf in feats.beat_features:
            cls = getattr(bf, "st_pattern_class", None)
            if cls:
                by_lead.setdefault(bf.lead, []).append(cls)
        lead_class = {
            lead: Counter(v).most_common(1)[0][0] for lead, v in by_lead.items()
        }
        counts = Counter(lead_class.values())
        row = {"ecg_id": ecg_id, "group": group, "status": "ok",
               "n_leads": len(lead_class)}
        for c in CLASSES:
            row[c] = counts.get(c, 0)
        # also the raw slope-based trend, for comparison
        trend = Counter(
            getattr(bf, "st_morphology", None) for bf in feats.beat_features
        )
        for t in ("upsloping", "horizontal", "downsloping"):
            row[f"trend_{t}"] = trend.get(t, 0)
        # deepest ST deviation seen anywhere, as a sanity anchor
        devs = [getattr(bf, "st_hybrid_80ms_mv", None) for bf in feats.beat_features]
        devs = [d for d in devs if d is not None and np.isfinite(d)]
        row["st80_min_mv"] = float(np.min(devs)) if devs else None
        row["st80_max_mv"] = float(np.max(devs)) if devs else None
        return row
    except Exception as exc:  # noqa: BLE001
        return {"ecg_id": ecg_id, "group": group, "status": f"error: {exc}"}


def main():
    db = pd.read_csv(META, index_col="ecg_id")
    db["scp"] = db.scp_codes.apply(ast.literal_eval)
    av = lambda idx: [int(i) for i in idx if rpath(int(i)).with_suffix(".hea").exists()]

    ischemic = ("ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ISCAN")
    groups = {
        "NORM": av(db[db.scp.apply(lambda d: "NORM" in d and not any(
            k in d for k in ischemic + ("STD_", "STE_", "NST_")))].index),
        "STD_": av(db[db.scp.apply(lambda d: "STD_" in d)].index),
        "ISCHEMIC": av(db[db.scp.apply(
            lambda d: any(k in d for k in ischemic))].index),
        "STE_": av(db[db.scp.apply(lambda d: "STE_" in d)].index),
    }
    jobs = [(i, g) for g, ids in groups.items() for i in ids[:N_PER_GROUP]]
    print(f"{len(jobs)} records: " +
          ", ".join(f"{g} {min(len(ids), N_PER_GROUP)}" for g, ids in groups.items()),
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(process, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if n % 30 == 0:
                print(f"[{n}/{len(jobs)}]", flush=True)

    d = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    ok = d[d.status == "ok"].copy()
    print(f"\nwrote {OUT}  ({len(ok)} ok)\n")

    print("=== 每条记录12导联中,各形态类占多少个导联(均值)===")
    hdr = f"{'组':<10}{'n':>4}" + "".join(f"{c[:13]:>15}" for c in CLASSES)
    print(hdr)
    for g in ("NORM", "STD_", "ISCHEMIC", "STE_"):
        s = ok[ok.group == g]
        if not len(s):
            continue
        print(f"{g:<10}{len(s):>4}" +
              "".join(f"{s[c].mean():>15.2f}" for c in CLASSES))

    print("\n=== 缺血特异形态(水平型+下斜型)占该记录已分类导联的比例 ===")
    ok["classified"] = ok[list(CLASSES[:-1])].sum(axis=1)
    ok["isch_share"] = np.where(
        ok.classified > 0,
        (ok.horizontal + ok.downsloping) / ok.classified.replace(0, np.nan), np.nan)
    for g in ("NORM", "STD_", "ISCHEMIC", "STE_"):
        s = ok[ok.group == g].isch_share.dropna()
        if len(s):
            print(f"  {g:<10} n={len(s):>3}  中位={s.median():.3f}  "
                  f"p25={s.quantile(.25):.3f}  p75={s.quantile(.75):.3f}")

    def auc(a, b):
        a, b = a.dropna(), b.dropna()
        if len(a) < 5 or len(b) < 5:
            return float("nan")
        v = np.concatenate([a.values, b.values])
        r = pd.Series(v).rank().values
        return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))

    norm = ok[ok.group == "NORM"].isch_share
    print("\n=== 判别力 AUC(对照 NORM)===")
    for g in ("STD_", "ISCHEMIC", "STE_"):
        print(f"  {g:<10} AUC={auc(ok[ok.group == g].isch_share, norm):.3f}")
    print("\n  AUC≈0.5 表示该形态分类无法区分缺血与正常,即分类没有临床信息量")

    print("\n=== 未能分类(unknown)的导联占比 ===")
    for g in ("NORM", "STD_", "ISCHEMIC", "STE_"):
        s = ok[ok.group == g]
        if len(s):
            print(f"  {g:<10} {100 * s.unknown.sum() / max(s.n_leads.sum(), 1):.0f}%")


if __name__ == "__main__":
    main()

"""How trustworthy are the Q and S outputs, given no dataset annotates them?

LUDB and QTDB mark QRS onset / peak / offset only, never Q, R and S
separately, so there is no reference to score amplitudes against. Two things
can still be established:

1. Semantic validity. `_qrs_points` defines q_amp as the minimum before the R
   peak and s_amp as the minimum after it -- extrema, not deflections. A
   complex beginning with an r wave has no Q at all, yet still receives a
   `q_amp_mv`, and it can come out positive. Measure how often that happens,
   because a "Q amplitude" of +0.05 mV is not a Q wave and conditioning on the
   field would mix two different quantities.
2. Discriminative check. A pathological Q wave is an infarct marker, so Q
   depth / Q:R ratio / `significant_q` should separate MI from normal. If they
   do not, the field carries no usable information regardless of its
   definition.
"""
import ast
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/workspace/ecg_gemma")
sys.path.insert(0, "/workspace/ecg_gemma/feature_extraction")

import numpy as np
import pandas as pd

PTBXL = Path("/workspace/ecg_gemma/data/ptb-xl")
META = "/workspace/ecg_gemma/data/ptb-xl-metadata-1.0.1/ptbxl_database.csv"
OUT = "/workspace/ecg_gemma/ecgfeat_perf_raw_20260809/qs_wave_probe.csv"
N = 50


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
        rows = []
        by_lead = {}
        for bf in feats.beat_features:
            by_lead.setdefault(bf.lead, []).append(bf)
        for lead, items in by_lead.items():
            def med(attr):
                v = [getattr(i, attr, None) for i in items]
                v = [float(x) for x in v
                     if x is not None and isinstance(x, (int, float))
                     and not isinstance(x, bool) and np.isfinite(x)]
                return float(np.median(v)) if v else None
            rows.append({
                "ecg_id": ecg_id, "group": group, "lead": lead,
                "q_amp": med("q_amp_mv"), "r_amp": med("r_amp_mv"),
                "s_amp": med("s_amp_mv"), "q_dur": med("q_duration_ms"),
                "q_r_ratio": med("q_r_ratio"),
                "s_dur": med("s_duration_ms"),
            })
        # significant_q from the MI evidence block
        ev = getattr(feats.interpretation, "mi_evidence", None) or {}
        qbl = ev.get("q_by_lead", {}) if isinstance(ev, dict) else {}
        for r in rows:
            e = qbl.get(r["lead"], {})
            r["significant_q"] = bool(e.get("significant_q")) if e else None
            r["q_wave_mi_ratio"] = bool(e.get("q_wave_mi_ratio")) if e else None
        return rows
    except Exception:  # noqa: BLE001
        return []


def main():
    db = pd.read_csv(META, index_col="ecg_id")
    db["scp"] = db.scp_codes.apply(ast.literal_eval)
    ok = lambda i: rpath(int(i)).with_suffix(".hea").exists()
    mi = ("IMI", "AMI", "ASMI", "ILMI", "ALMI", "IPMI", "IPLMI", "LMI")
    groups = {
        "NORM": [int(i) for i, d in db.scp.items() if "NORM" in d and ok(i)][:N],
        "MI": [int(i) for i, d in db.scp.items()
               if any(c in d for c in mi) and ok(i)][:N],
        "LVH": [int(i) for i, d in db.scp.items() if "LVH" in d and ok(i)][:N],
    }
    jobs = [(i, g) for g, ids in groups.items() for i in ids]
    print(f"{len(jobs)} records", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(process, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if n % 30 == 0:
                print(f"[{n}/{len(jobs)}]", flush=True)

    d = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(d)} lead rows)\n")

    q = d.q_amp.dropna()
    s = d.s_amp.dropna()
    print("=== 语义有效性:Q/S 被定义为极值,不是负向波 ===")
    print(f"  q_amp 为正(该导联其实没有Q波)的比例: {100 * (q > 0).mean():.1f}%  "
          f"(n={len(q)})")
    print(f"  q_amp 中位={q.median():+.4f} mV   p10={q.quantile(.1):+.4f}  "
          f"p90={q.quantile(.9):+.4f}")
    print(f"  s_amp 为正(该导联其实没有S波)的比例: {100 * (s > 0).mean():.1f}%  "
          f"(n={len(s)})")
    print(f"  s_amp 中位={s.median():+.4f} mV   p10={s.quantile(.1):+.4f}  "
          f"p90={s.quantile(.9):+.4f}")
    print(f"\n  q_duration_ms 有值的比例: {100 * d.q_dur.notna().mean():.1f}%   "
          f"(需要真实Q波才算得出)")
    print(f"  其中 q_amp 为正却仍给出 q_duration 的: "
          f"{100 * ((d.q_amp > 0) & d.q_dur.notna()).sum() / max(d.q_dur.notna().sum(), 1):.1f}%"
          f"   <- 若不低,说明时限也算在了非Q波上")

    def auc(col, hi=True, grp="MI"):
        a = d[d.group == grp][col].dropna()
        b = d[d.group == "NORM"][col].dropna()
        if len(a) < 10 or len(b) < 10:
            return float("nan")
        v = np.concatenate([a.values, b.values])
        r = pd.Series(v).rank().values
        val = (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        return val if hi else 1 - val

    print("\n=== 判别力 AUC(逐导联,对照 NORM)===")
    print(f"  {'特征':<28}{'MI':>9}{'LVH':>9}")
    for col, hi in (("q_amp", False), ("q_r_ratio", True),
                    ("q_dur", True), ("s_amp", False)):
        print(f"  {col:<28}{auc(col, hi, 'MI'):>9.3f}{auc(col, hi, 'LVH'):>9.3f}")

    print("\n=== significant_q / q_wave_mi_ratio 的触发率 ===")
    for g in ("NORM", "MI", "LVH"):
        s2 = d[d.group == g]
        print(f"  {g:<6} significant_q {100 * s2.significant_q.fillna(False).mean():>5.1f}%   "
              f"q_wave_mi_ratio {100 * s2.q_wave_mi_ratio.fillna(False).mean():>5.1f}%")

    print("\n  AUC对 q_amp/s_amp 取的是'越负越像异常'的方向")


if __name__ == "__main__":
    main()

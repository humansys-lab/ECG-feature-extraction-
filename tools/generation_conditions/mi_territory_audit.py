"""Is the infarct-territory output usable as a generation condition?

Never evaluated in this repo. PTB-XL's MI codes carry the territory (IMI =
inferior, ASMI = anteroseptal, ALMI = anterolateral, ILMI = inferolateral,
AMI = anterior, LMI = lateral), so ecgfeat's own territory assignment can be
scored against them directly.

For conditioning, two things matter beyond accuracy: how often a territory is
produced at all (an empty territory list means the sample carries no usable
condition), and whether that emptiness tracks the label (which would bias the
generated distribution).
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
OUT = "/workspace/ecg_gemma/ecgfeat_perf_raw_20260809/mi_territory_audit.csv"
N = 50

# PTB-XL MI code -> territory as ecgfeat names them
CODE_TERRITORY = {
    "IMI": "inferior", "ILMI": "inferior", "IPMI": "inferior",
    "IPLMI": "inferior", "INJIN": "inferior", "INJIL": "inferior",
    "AMI": "anterior", "ASMI": "anterior", "INJAS": "anterior",
    "ALMI": "anterolateral", "INJAL": "anterolateral",
    "LMI": "lateral", "INJLA": "lateral",
    "PMI": "posterior",
}


def rpath(i):
    return PTBXL / f"{(i // 1000) * 1000:05d}" / f"{i:05d}_hr"


def process(args):
    ecg_id, group, truth = args
    import wfdb
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    try:
        rec = wfdb.rdrecord(str(rpath(ecg_id)))
        feats = ECGFeatureExtractor(fs_internal=500, mains_freq=50.0).extract(
            np.asarray(rec.p_signal, dtype=float).T, fs=float(rec.fs)
        )
        it = feats.interpretation
        ev = getattr(it, "mi_evidence", None) or {}
        terrs = ev.get("territories", {}) if isinstance(ev, dict) else {}

        q_terr, ste_terr = [], []
        for name, t in terrs.items():
            if not isinstance(t, dict):
                continue
            if t.get("q_wave_mi_pattern") or t.get("q_wave_leads"):
                q_terr.append(name)
            if t.get("st_elevation_leads"):
                ste_terr.append(name)

        stmts = getattr(it, "mi_statement_candidates", None) or []
        stmt_terr = sorted({s.get("territory") for s in stmts
                            if isinstance(s, dict) and s.get("territory")})

        return {
            "ecg_id": ecg_id, "group": group, "truth_territory": truth,
            "q_territories": "|".join(sorted(set(
                getattr(it, "q_wave_territories", None) or q_terr))),
            "st_depressed": "|".join(
                getattr(it, "st_territories_depressed", None) or []),
            "st_elevated": "|".join(
                getattr(it, "st_territories_elevated", None) or []),
            "stmt_territories": "|".join(stmt_terr),
            "n_statements": len(stmts),
            "n_st_dep_leads": len(getattr(it, "st_depression_leads", None) or {}),
            "n_st_ele_leads": len(getattr(it, "st_elevation_leads", None) or {}),
            "posterior_suspected": bool(getattr(it, "posterior_mi_suspected", False)),
            "mi_evidence_available": bool(ev.get("available")) if isinstance(ev, dict) else False,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ecg_id": ecg_id, "group": group, "error": str(exc)[:90]}


def main():
    db = pd.read_csv(META, index_col="ecg_id")
    db["scp"] = db.scp_codes.apply(ast.literal_eval)
    ok = lambda i: rpath(int(i)).with_suffix(".hea").exists()

    jobs = []
    for terr in ("inferior", "anterior", "anterolateral"):
        codes = [c for c, t in CODE_TERRITORY.items() if t == terr]
        ids = [int(i) for i, d in db.scp.items()
               if any(c in d for c in codes) and ok(i)][:N]
        jobs += [(i, f"MI_{terr}", terr) for i in ids]
    norm = [int(i) for i, d in db.scp.items() if "NORM" in d and ok(i)][:N]
    jobs += [(i, "NORM", "") for i in norm]
    print(f"{len(jobs)} records", flush=True)

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
    d = d[d.get("error").isna()] if "error" in d else d
    print(f"\nwrote {OUT}  ({len(d)} ok)\n")

    print("=== 条件可用性:每组产出非空区域的比例 ===")
    print(f"{'组':<20}{'n':>4}{'Q波区域':>10}{'ST压低区域':>12}{'ST抬高区域':>12}{'MI陈述':>9}")
    for g in sorted(d.group.unique()):
        s = d[d.group == g]
        f = lambda c: (s[c].fillna("").astype(str) != "").mean()
        print(f"{g:<20}{len(s):>4}{f('q_territories'):>10.2f}"
              f"{f('st_depressed'):>12.2f}{f('st_elevated'):>12.2f}"
              f"{(s.n_statements > 0).mean():>9.2f}")

    print("\n=== 区域判定是否命中 PTB-XL 标注的区域 ===")
    mi = d[d.group.str.startswith("MI_")].copy()
    for src in ("q_territories", "st_elevated", "stmt_territories"):
        hit = mi.apply(
            lambda r: bool(r.truth_territory) and r.truth_territory in
            str(r[src] or "").split("|"), axis=1)
        nonempty = (mi[src].fillna("").astype(str) != "")
        print(f"  {src:<20} 非空 {nonempty.mean():.2f}   "
              f"命中真值区域 {hit.mean():.2f}   "
              f"非空样本中的命中率 {hit[nonempty].mean() if nonempty.any() else float('nan'):.2f}")

    print("\n=== 各真值区域下,ecgfeat 给出的区域分布 ===")
    for terr in ("inferior", "anterior", "anterolateral"):
        s = mi[mi.truth_territory == terr]
        if not len(s):
            continue
        c = Counter(x for v in s.stmt_territories.fillna("") for x in str(v).split("|") if x)
        print(f"  真值 {terr:<15} n={len(s):>3}  陈述区域: {dict(c.most_common(4)) or '(全空)'}")

    print("\n  非空率低 = 多数样本没有可用的区域条件;命中率低 = 该条件与真实区域不对应")


if __name__ == "__main__":
    main()

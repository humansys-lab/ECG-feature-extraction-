"""Would qualifying the ST shape class by deviation actually help?

Measured: the four-class shape on its own discriminates ischemia worse than
plain deviation magnitude on 2 of 3 label groups, because the class carries no
amplitude -- a flat ST on the baseline and a flat ST depressed 2 mm are both
`horizontal`. The proposed fix is to report shape only where the segment is
actually displaced. Before writing that, check per lead whether
"shape AND displaced" beats either input alone; if it does not, the fix is not
worth the vocabulary change.
"""
import ast
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/ecg_gemma")
sys.path.insert(0, "/workspace/ecg_gemma/feature_extraction")

import numpy as np
import pandas as pd

PTBXL = Path("/workspace/ecg_gemma/data/ptb-xl")
META = "/workspace/ecg_gemma/data/ptb-xl-metadata-1.0.1/ptbxl_database.csv"
OUT = "/workspace/ecg_gemma/ecgfeat_perf_raw_20260809/st_joint_probe2.csv"
N_PER_GROUP = 45


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
        by_lead = {}
        for bf in feats.beat_features:
            cls = getattr(bf, "st_pattern_class", None)
            morph = getattr(bf, "st_morphology", None)
            dev = getattr(bf, "st_hybrid_80ms_mv", None)
            if dev is None or not np.isfinite(dev):
                dev = getattr(bf, "st_80ms_mv", None)
            by_lead.setdefault(bf.lead, {"cls": [], "dev": [], "morph": []})
            if cls:
                by_lead[bf.lead]["cls"].append(cls)
            if morph:
                by_lead[bf.lead]["morph"].append(morph)
            if dev is not None and np.isfinite(dev):
                by_lead[bf.lead]["dev"].append(float(dev))
        rows = []
        for lead, d in by_lead.items():
            if not d["cls"] or not d["dev"]:
                continue
            rows.append({
                "ecg_id": ecg_id, "group": group, "lead": lead,
                "cls": Counter(d["cls"]).most_common(1)[0][0],
                "morph": Counter(d["morph"]).most_common(1)[0][0] if d["morph"] else "",
                "dev": float(np.median(d["dev"])),
            })
        return rows
    except Exception:  # noqa: BLE001
        return []


def main():
    db = pd.read_csv(META, index_col="ecg_id")
    db["scp"] = db.scp_codes.apply(ast.literal_eval)
    av = lambda idx: [int(i) for i in idx if rpath(int(i)).with_suffix(".hea").exists()]
    isch = ("ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ISCAN")
    groups = {
        "NORM": av(db[db.scp.apply(lambda d: "NORM" in d and not any(
            k in d for k in isch + ("STD_", "STE_", "NST_")))].index),
        "STD_": av(db[db.scp.apply(lambda d: "STD_" in d)].index),
        "ISCHEMIC": av(db[db.scp.apply(lambda d: any(k in d for k in isch))].index),
    }
    jobs = [(i, g) for g, ids in groups.items() for i in ids[:N_PER_GROUP]]
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



if __name__ == "__main__":
    main()

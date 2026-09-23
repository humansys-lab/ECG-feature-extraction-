#!/usr/bin/env python3
"""Paired comparisons, error cases and stratification for the external protocol."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_ecgfeat_external import (
    WaveEvent, aggregate_metrics, discover, load_annotations, noise_phase, sparse_match,
)
import numpy as np
import wfdb


def read_tasks(root, dataset):
    return [json.loads(p.read_text()) for p in sorted((root / dataset / "checkpoints").glob("*.json"))]


def bootstrap(values):
    if not values:
        return dict(records=0, mean_delta=None, ci95=None)
    values = np.asarray(values)
    rng = np.random.default_rng(20260922)
    boot = np.mean(rng.choice(values, size=(5000, len(values))), axis=1)
    return dict(records=len(values), mean_delta=float(values.mean()),
                ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                improved=int(np.sum(values < -1e-9)), worsened=int(np.sum(values > 1e-9)),
                unchanged=int(np.sum(abs(values) <= 1e-9)))


def paired_isp(tasks):
    groups = defaultdict(dict)
    for t in tasks:
        groups[t["variant"]][t["record"]] = t
    output = []
    for variant, records in groups.items():
        if variant == "default":
            continue
        for split in ("train", "test"):
            for wave in ("P", "QRS", "T"):
                for boundary in ("onset", "offset"):
                    deltas, base_values, candidate_values = [], [], []
                    base_count = candidate_count = 0
                    for record, t in records.items():
                        if not record.startswith(split+"/"):
                            continue
                        def errors(task):
                            return {r["gt_onset_sample"]: abs(r[f"{boundary}_error_ms"])
                                    for r in task["matches"] if r["lead"] == "consensus_median" and r["wave"] == wave
                                    and r.get(f"{boundary}_error_ms") is not None}
                        base, candidate = errors(groups["default"][record]), errors(t)
                        base_count += len(base); candidate_count += len(candidate)
                        keys = base.keys() & candidate.keys()
                        if keys:
                            delta = [candidate[k]-base[k] for k in keys]
                            deltas.append(float(np.mean(delta)))
                            base_values.extend(base[k] for k in keys)
                            candidate_values.extend(candidate[k] for k in keys)
                    output.append(dict(variant=variant, split=split, wave=wave, boundary=boundary,
                                       base_matched=base_count, candidate_matched=candidate_count,
                                       common_events=len(base_values),
                                       base_paired_mae=float(np.mean(base_values)) if base_values else None,
                                       candidate_paired_mae=float(np.mean(candidate_values)) if candidate_values else None,
                                       record_bootstrap=bootstrap(deltas)))
    return output


def but_analysis(tasks, data_root):
    specs = {s["record"]: s for s in discover("but-pdb", data_root)}
    rows, matches, cases = [], [], []
    for t in tasks:
        tags = [s.strip() for s in t["diagnosis"].split(",") if s.strip()]
        for tag in tags + ["ALL"]:
            rows.extend(dict(r, pathology=tag) for r in t["rows"])
            matches.extend(dict(m, pathology=tag) for m in t["matches"])
        if t["variant"] != "default":
            continue
        fs = t["fs"]
        truth, _ = load_annotations("but-pdb", specs[t["record"]], fs, t["samples"])
        detected = {w: [WaveEvent(**e) for e in es] for w, es in t["predictions"].get("II", {}).items()}
        pairs = sparse_match(truth["P"], detected.get("P", []), round(.15*fs))
        qrs_pairs = sparse_match(truth["QRS"], detected.get("QRS", []), round(.075*fs))
        found_p, found_qrs = {a for a, _ in pairs}, {a for a, _ in qrs_pairs}
        missed = []
        for p in truth["P"]:
            if p in found_p:
                continue
            following = [q for q in truth["QRS"] if 0 < q.peak-p.peak <= .45*fs]
            reason = ("no_reference_QRS_within_next_450ms" if not following else
                      "following_QRS_missed" if following[0] not in found_qrs else "P_missed_despite_following_QRS_detected")
            missed.append(dict(sample=p.peak, category=reason))
        fp = [p.peak for p in detected.get("P", []) if p not in {b for _, b in pairs}]
        cases.append(dict(record=t["record"], diagnosis=t["diagnosis"], gt_p=len(truth["P"]),
                          tp=len(pairs), fp=len(fp), fn=len(missed), false_positive_samples=fp,
                          missed=missed, missed_categories=dict(Counter(p["category"] for p in missed)),
                          p_absent_record_fp_per_minute=len(fp)/(t["duration_seconds"]/60) if not truth["P"] else None))
    metrics = aggregate_metrics(rows, matches, ("method", "wave", "protocol", "pathology"))
    return dict(metrics=metrics, default_cases=cases,
                default_fn_categories=dict(sum((Counter(c["missed_categories"]) for c in cases), Counter())))


def figures(root, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    noise = summaries["nstdb"]["metrics"]
    for phase in ("noisy", "clean"):
        points = sorted((m["snr"], 100*m["f1"]) for m in noise if m["phase"] == phase)
        axes[0].plot(*zip(*points), marker="o", label=phase)
    axes[0].set(xlabel="SNR (dB)", ylabel="QRS F1 (%)", title="NSTDB: 2 source ECGs", ylim=(0, 101))
    axes[0].legend(); axes[0].grid(alpha=.2)
    isp = [m for m in summaries["isp"]["metrics"] if m["method"] == "default" and m["lead"] == "consensus_median" and m["split"] == "test"]
    x = np.arange(3)
    values = {m["wave"]: m for m in isp}
    axes[1].bar(x-.18, [100*values[w]["f1"] for w in ("P", "QRS", "T")], .36, label="Event F1 (IoU >= 0.5)")
    axes[1].bar(x+.18, [100*values[w]["sample_dice"] for w in ("P", "QRS", "T")], .36, label="Sample Dice")
    axes[1].set(xticks=x, xticklabels=["P", "QRS", "T"], ylabel="%", title="ISP official test: default", ylim=(0, 101))
    axes[1].legend(loc="lower right"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(root / "external_metrics.png", dpi=170)
    fig.savefig(root / "external_metrics.svg")
    plt.close(fig)


def case_figure(root, data_root, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
    for ax, name in zip(axes, ("08", "13", "42")):
        case = next(c for c in cases if c["record"] == name)
        item = json.loads((root / "but-pdb/checkpoints" / f"default__{name}.json").read_text())
        record = wfdb.rdrecord(str(data_root / "but-pdb" / name))
        # Error-selected visualization only; selection never affects extraction.
        samples = case["false_positive_samples"] if name == "08" else [c["sample"] for c in case["missed"]]
        bins = Counter(int(s / record.fs / 10) for s in samples)
        start = bins.most_common(1)[0][0]*10 if bins else 0
        end = start+10
        t = np.arange(record.sig_len)/record.fs
        mask = (t >= start) & (t < end)
        ax.plot(t[mask], record.p_signal[mask, 0], color="black", lw=.7)
        gt, _ = load_annotations("but-pdb", {"path": str(data_root / "but-pdb" / name)}, record.fs, record.sig_len)
        for events, color, marker, label in ((gt["P"], "seagreen", "o", "Reference P"),
                  ([WaveEvent(**e) for e in item["predictions"]["II"]["P"]], "darkorange", "x", "Native detected P")):
            indices = [e.peak for e in events if start <= e.peak/record.fs < end]
            ax.scatter(np.asarray(indices)/record.fs, record.p_signal[indices, 0], s=42, marker=marker,
                       color=color, facecolors="none" if marker == "o" else color, label=label, zorder=4)
        ax.set(xlim=(start, end), ylabel="Channel 0 (mV)", title=f"BUT PDB {name}: {case['diagnosis']} (selected error window)")
        ax.legend(loc="upper right"); ax.grid(alpha=.15)
    axes[-1].set_xlabel("Time from record start (s)")
    fig.savefig(root / "but_error_cases.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "ecgfeat_external_20260922")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external_ecg")
    args = parser.parse_args()
    summaries = {d: json.loads((args.root/d/"summary.json").read_text()) for d in ("isp", "but-pdb", "nstdb")}
    tasks = {d: read_tasks(args.root, d) for d in summaries}
    for d, ts in tasks.items():
        manifest = json.loads((args.root/d/"manifest.json").read_text())
        assert len(ts) == len(manifest["records"])*len(manifest["variants"]), f"incomplete {d}"
    result = dict(isp_paired=paired_isp(tasks["isp"]), but=but_analysis(tasks["but-pdb"], args.data_root))
    result["audit"] = {d: dict(records=len({t["record"] for t in ts}), tasks=len(ts),
                             annotation_audit={t["record"]: t["annotation_audit"] for t in ts if t["annotation_audit"]},
                             failures=[(t["record"], t["variant"], t["failures"]) for t in ts if t["failures"]])
                       for d, ts in tasks.items()}
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    figures(args.root, summaries)
    case_figure(args.root, args.data_root, result["but"]["default_cases"])


if __name__ == "__main__":
    main()

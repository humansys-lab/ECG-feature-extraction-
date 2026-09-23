#!/usr/bin/env python3
"""Exploratory BUT PDB audit of existing atrial outputs; no algorithm changes."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_ecgfeat_external as ev

native_events = ev.result_events


def diagnostic_events(features, dataset):
    result = native_events(features, dataset)
    events = features.metadata.get("rhythm_analysis", {}).get("atrial_events", [])
    result["atrial_events"] = {"P": [ev.WaveEvent(None, int(e["sample"]), None) for e in events]}
    accepted = {p.beat_id for p in features.p_wave_assessments if p.accepted}
    result["accepted_native"] = {"P": [ev.WaveEvent(b.p.onset, b.p.peak, b.p.offset)
                                         for b in features.beat_features if b.lead == "II" and b.beat_id in accepted and b.p.peak is not None]}
    return result


def task(payload):
    ev.result_events = diagnostic_events
    return ev.evaluate_task(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "ecgfeat_external_20260922")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external_ecg")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    source = json.loads((args.root / "but-pdb/manifest.json").read_text())
    for name, expected in source["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    manifest = dict(base_fingerprint=source["fingerprint"], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    purpose="Exploratory audit after native-output evaluation; no thresholds selected or fitted",
                    variants=["default"], base_protocol=ev.PROTOCOL)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = args.root / "but_atrial_audit"
    directory.mkdir(exist_ok=True); (directory / "checkpoints").mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("stale atrial audit")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    specs = ev.discover("but-pdb", args.data_root)
    tasks = [("but-pdb", s, "default", str(directory / "checkpoints" / f"default__{s['record']}.json"), fingerprint)
             for s in specs if not (directory / "checkpoints" / f"default__{s['record']}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, f in enumerate(as_completed([pool.submit(task, t) for t in tasks]), 1):
            print(n, f.result(), flush=True)
    rows, matches, failures = [], [], []
    for s in specs:
        item = json.loads((directory / "checkpoints" / f"default__{s['record']}.json").read_text())
        assert item["fingerprint"] == fingerprint
        truth, _ = ev.load_annotations("but-pdb", s, item["fs"], item["samples"])
        baseline = json.loads((args.root / "but-pdb/checkpoints" / f"default__{s['record']}.json").read_text())
        assert item["predictions"]["II"] == baseline["predictions"]["II"], f"native replay differs: {s['record']}"
        failures.extend(item["failures"])
        for lead in ("II", "atrial_events", "accepted_native"):
            predictions = [ev.WaveEvent(**e) for e in item["predictions"].get(lead, {}).get("P", [])]
            for tolerance in (150, 50):
                row, matched = ev.score(s["record"], "default", "P", lead, truth["P"], predictions,
                                        item["fs"], "partial_failure" if item["failures"] else "ok", tolerance,
                                        extra={"protocol": f"peak_{tolerance}ms"})
                rows.append(row); matches.extend(matched)
    ev.write_csv(directory / "detection_by_record.csv", rows)
    ev.write_csv(directory / "matched_events.csv", matches)
    summary = dict(records=len(specs), failures=failures, native_replay_exact=True,
                   metrics=ev.aggregate_metrics(rows, matches, ("method", "wave", "lead", "protocol")))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download pinned INCART, SVDB and PWAVE, including original MIT-BIH beats."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_ecgfeat_external import fetch, physionet


def mirror_dataset(name, root, workers):
    """Official public S3 mirror, validated against the PhysioNet SHA256 list."""
    base = f"https://physionet.org/files/{name}/1.0.0/"
    mirror = f"https://physionet-open.s3.amazonaws.com/{name}/1.0.0/"
    folder = root / name
    source = fetch(base + "SHA256SUMS.txt", folder / "SHA256SUMS.txt")
    entries = []
    for line in (folder / "SHA256SUMS.txt").read_text().splitlines():
        checksum, filename = line.split(maxsplit=1)
        filename = filename.lstrip(" *")
        if Path(filename).name != filename:
            continue
        if Path(filename).suffix in {".dat", ".hea", ".atr", ".pwave", ".txt"} or filename in {"RECORDS", "ANNOTATORS", "README"}:
            entries.append((filename, checksum))
    results = [source]
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, mirror+name, folder/name, expected=checksum) for name,checksum in entries]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 20 == 0 or i == len(futures):
                print(f"{name}: {i}/{len(futures)} verified", flush=True)
    (folder / "download_manifest.json").write_text(json.dumps(dict(dataset=name, version="1.0.0",
                    source=base, mirror=mirror, files=sorted(results,key=lambda r:r["path"])), indent=2))


def mitdb_references(root):
    folder = root / "pwave/mitdb_reference"
    base = "https://physionet.org/files/mitdb/1.0.0/"
    manifest = fetch(base + "SHA256SUMS.txt", folder / "SHA256SUMS.txt")
    checksums = {name.lstrip(" *"): checksum for checksum, name in
                 (line.split(maxsplit=1) for line in (folder / "SHA256SUMS.txt").read_text().splitlines())}
    results = [manifest]
    for header in sorted((root / "pwave").glob("*.hea")):
        name = header.stem + ".atr"
        results.append(fetch(base + name, folder / name, expected=checksums[name]))
    (folder / "download_manifest.json").write_text(json.dumps(
        dict(dataset="mitdb", version="1.0.0", purpose="PWAVE original reference beats", files=results), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/external_ecg")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", choices=("incartdb", "svdb", "pwave"),
                        default=["incartdb", "svdb", "pwave"])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(mirror_dataset, name, args.root, args.workers)
                   for name in args.datasets]
        for future in futures:
            future.result()
    if "pwave" in args.datasets:
        mitdb_references(args.root)


if __name__ == "__main__":
    main()

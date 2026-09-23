#!/usr/bin/env python3
"""Download version-pinned public ECG evaluation data and verify checksums."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, path, algorithm="sha256", expected=None):
    path = Path(path)
    def valid():
        return path.exists() and (expected is None or hashlib.new(algorithm, path.read_bytes()).hexdigest() == expected)
    if not valid():
        path.parent.mkdir(parents=True, exist_ok=True)
        error = None
        for attempt in range(4):
            try:
                with requests.get(url, stream=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    temporary = path.with_suffix(path.suffix + ".part")
                    with temporary.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            handle.write(block)
                if expected and hashlib.new(algorithm, temporary.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"checksum mismatch: {path.name}")
                temporary.replace(path)
                break
            except (requests.RequestException, ValueError) as exc:
                error = exc
                time.sleep(attempt + 1)
        if not valid():
            raise RuntimeError(f"failed to fetch {url}: {error}")
    return {"path": str(path), "url": url, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "upstream_checksum": f"{algorithm}:{expected}" if expected else None}


def physionet(name, root, workers):
    base = f"https://physionet.org/files/{name}/1.0.0/"
    folder = root / name
    manifest_file = folder / "SHA256SUMS.txt"
    source = fetch(base + "SHA256SUMS.txt", manifest_file)
    entries = []
    for line in manifest_file.read_text().splitlines():
        checksum, filename = line.split(maxsplit=1)
        filename = filename.lstrip(" *")
        if Path(filename).name != filename:
            continue
        suffix = Path(filename).suffix
        if suffix in {".dat", ".hea", ".atr", ".pwave", ".qrs"} or filename in {"README", "README.txt", "RECORDS", "LICENSE.txt"}:
            entries.append((filename, checksum))
    results = [source]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, base + filename, folder / filename, "sha256", checksum)
                   for filename, checksum in entries]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 20 == 0 or index == len(futures):
                print(f"{name}: {index}/{len(futures)} files verified", flush=True)
    results.sort(key=lambda item: item["path"])
    (folder / "download_manifest.json").write_text(json.dumps({"dataset": name, "version": "1.0.0", "files": results}, indent=2))


def isp(root):
    folder = root / "isp"
    item = fetch("https://zenodo.org/records/14679837/files/isp_delineation_dataset.zip?download=1",
                 root / "isp.zip", "md5", "a070d5f7d5ff88ce823dbedc7412ee28")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "isp.zip") as archive:
        for member in archive.infolist():
            destination = (folder / member.filename).resolve()
            if not destination.is_relative_to(folder.resolve()):
                raise ValueError("unsafe archive path")
        archive.extractall(folder)
    (folder / "download_manifest.json").write_text(json.dumps({"dataset": "ISP", "version": "v2", "zenodo_record": 14679837, "archive": item}, indent=2))
    print("ISP archive verified and extracted", flush=True)


def gudb(root):
    folder = root / "gudb"
    archive_path = folder / "gudb_1.1.0.zip"
    item = fetch("https://zenodo.org/api/records/10925903/files/berndporr/ECG-GUDB-1.1.0.zip/content",
                 archive_path, "md5", "9f50e13470715d58d2aa9675e15a8f54")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (folder / member.filename).resolve().is_relative_to(folder.resolve()):
                raise ValueError("unsafe archive path")
        archive.extractall(folder)
    (folder / "download_manifest.json").write_text(json.dumps(
        {"dataset": "GUDB", "version": "1.1.0", "zenodo_record": 10925903, "archive": item}, indent=2))
    print("GUDB archive verified and extracted", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("isp", "but-pdb", "nstdb", "gudb"), default=["isp", "but-pdb", "nstdb"])
    parser.add_argument("--root", type=Path, default=ROOT / "data/external_ecg")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    for name in args.datasets:
        if name in {"isp", "gudb"}:
            {"isp": isp, "gudb": gudb}[name](args.root)
        else:
            physionet(name, args.root, args.workers)


if __name__ == "__main__":
    main()

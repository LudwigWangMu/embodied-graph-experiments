"""Reconstruct the exact four replay inputs identified in the manuscript supplement."""
import argparse
import csv
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from prepare_physical import align_record, payload
from range_archive import Remote


def physical_records(cache, archive_path):
    with (ROOT / "data/physical_sources.csv").open(newline="") as manifest:
        rows = list(csv.DictReader(manifest))
    cache.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if not (cache / Path(r["archive_member"]).name).exists()]
    if missing:
        if archive_path:
            with ZipFile(archive_path) as archive:
                for r in missing:
                    name = r["archive_member"]
                    (cache / Path(name).name).write_bytes(archive.read(name))
        else:
            with ZipFile(Remote()) as archive:
                entries = [archive.getinfo(r["archive_member"]) for r in missing]
            def fetch(entry):
                raw = payload(entry)
                (cache / Path(entry.filename).name).write_bytes(raw)
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(fetch, entries))
    episodes = []
    for i, r in enumerate(rows):
        path = cache / Path(r["archive_member"]).name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != r["sha256"]:
            raise ValueError(f"Source hash mismatch in {path}. Replace that cache entry and retry.")
        pusher, landmarks, _ = align_record(raw)
        if i != int(r["episode"]) or len(pusher) != int(r["replay_frames"]):
            raise ValueError("Episode index or replay length differs from the source manifest")
        episodes.append((pusher, landmarks))
    np.savez_compressed(ROOT / "data/physical_tracks.npz",
                        pusher=np.concatenate([p for p, _ in episodes]),
                        landmarks=np.concatenate([l for _, l in episodes]),
                        ends=np.cumsum([len(p) for p, _ in episodes]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pusht", type=Path, default=ROOT / "cache/pusht.zip")
    parser.add_argument("--mcube-cache", type=Path, default=ROOT / "cache/mcube")
    parser.add_argument("--mcube-archive", type=Path, help="Optional original rect1_json.zip")
    args = parser.parse_args()
    archive = args.pusht.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    for script, options in [("prepare_pusht.py", ["--archive", archive]),
                            ("prepare_pusht_pose.py", [archive]),
                            ("render_alignment.py", ["--archive", archive])]:
        subprocess.run([sys.executable, str(ROOT / "code" / script), *options], cwd=ROOT, check=True)
    physical_records(args.mcube_cache.resolve(), args.mcube_archive)


if __name__ == "__main__":
    main()

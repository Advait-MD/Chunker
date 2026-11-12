"""Download the M2 benchmark files, pinned to exact Hugging Face commits.

Writes data/benchmarks/<name>/<file> and data/benchmarks/MANIFEST.json (url, commit, bytes, sha256).
Re-running skips files whose sha256 already matches the manifest. Raw data is gitignored; the
sampled question ids (apex/eval/splits/) are what make runs reproducible from a clean checkout.

Usage: python scripts/fetch_benchmarks.py [name ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "benchmarks"

# name -> (repo, pinned revision, [files])
SOURCES = {
    "qasper": ("allenai/qasper", "06806e4608976fc2fac0a090ac425d5b2b29caf4",  # refs/convert/parquet
               ["qasper/validation/0000.parquet", "qasper/test/0000.parquet"]),
    "musique": ("dgslibisey/MuSiQue", "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
                ["musique_ans_v1.0_dev.jsonl"]),
    "quality": ("tasksource/QuALITY", "4892b065e6a40f44fa61bc406df4c35fe91e9290",
                ["data/validation-00000-of-00001-c057a0698ad4d3c4.parquet"]),
    "hotpotqa": ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab",
                 ["distractor/validation-00000-of-00001.parquet"]),
    "2wiki": ("framolfese/2WikiMultihopQA", "fe713bfbd1afbca1a65246741a75890405d56a3a",
              ["data/validation-00000-of-00001.parquet"]),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "MANIFEST.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    for name in names or SOURCES:
        repo, rev, files = SOURCES[name]
        for rel in files:
            dest = OUT / name / rel.replace("/", "__")
            key = f"{name}/{dest.name}"
            if dest.exists() and manifest.get(key, {}).get("sha256") == sha256_file(dest):
                print(f"skip  {key} (sha256 matches)")
                continue
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{rel}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            manifest[key] = {"repo": repo, "revision": rev, "path": rel, "url": url,
                             "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}
            print(f"got   {key}  {dest.stat().st_size / 1e6:.1f} MB")
            man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

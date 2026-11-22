"""Download the NLI model pinned in the config into the local Hugging Face cache (M5).

    python scripts/fetch_nli.py [--config configs/base.yaml]

Only the files `transformers` needs are fetched, at the exact `nli.revision` commit. The scorer itself
loads with `local_files_only=True`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download  # noqa: E402

from apex.config import load_config  # noqa: E402

FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "spm.model", "added_tokens.json"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    n = load_config(a.config)["nli"]
    path = snapshot_download(n["model"], revision=n["revision"], allow_patterns=FILES)
    print(f"{n['model']}@{n['revision']} -> {path}")
    for f in sorted(Path(path).iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:9.1f} MB")


if __name__ == "__main__":
    main()

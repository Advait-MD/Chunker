"""Download the cross-encoder reranker pinned in the config into the local Hugging Face cache.

    python scripts/fetch_reranker.py [--config configs/base.yaml]

Only the files `transformers` needs are fetched (safetensors weights, config, tokenizer), at the
exact `rerank.revision` commit. The reranker itself loads with `local_files_only=True`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download  # noqa: E402

from apex.config import load_config  # noqa: E402

FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "sentencepiece.bpe.model"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    r = load_config(a.config)["rerank"]
    path = snapshot_download(r["model"], revision=r["revision"], allow_patterns=FILES)
    print(f"{r['model']}@{r['revision']} -> {path}")
    for f in sorted(Path(path).iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:9.1f} MB")


if __name__ == "__main__":
    main()

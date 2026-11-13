"""Local cross-encoder reranker (spec §2.4: `bge-reranker-v2-m3` over the fused top 20).

Runs on the CPU through `transformers` (Ollama does not serve cross-encoders). The model is
pinned to a Hugging Face revision in the config and loaded with `local_files_only=True`: a missing
model is a loud error telling you how to fetch it, never a silent download or a fallback model.

`rerank.quantize: int8` applies PyTorch dynamic int8 quantization to the Linear layers: on this
laptop's CPU it cuts bge-reranker-v2-m3 from ~15–25 s to ~7–9 s per query (20 passages), and the
top 5 matched fp32 on the probes we ran (docs/M3_NOTES.md). Batch size 1 avoids padding waste.

Scores are the raw relevance logits (higher = more relevant). They are deterministic, so they are
cached in SQLite by content hash of (model, revision, max_length, query, passage): an eval rerun,
or the separate depth-20 ranking used for retrieval metrics, reuses them.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from apex.config import resolve_path


class RerankerUnavailable(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(self, cfg: Dict[str, Any]):
        r = cfg["rerank"]
        self.model_name, self.revision = r["model"], r.get("revision")
        if not self.revision:
            raise ValueError("rerank.revision must pin a Hugging Face commit (spec: pin versions)")
        self.max_length, self.batch_size = int(r["max_length"]), int(r["batch_size"])
        self.threads = int(r.get("threads") or 0)
        self.quantize = r.get("quantize") or "none"
        if self.quantize not in ("none", "int8"):
            raise ValueError(f"rerank.quantize must be none or int8, got {self.quantize!r}")
        self._model = self._tok = None
        self._lock = threading.Lock()
        path = resolve_path(r["cache_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL)")
        self._db.commit()
        self.last: Dict[str, Any] = {}

    def _key(self, query: str, text: str) -> str:
        h = hashlib.sha256()
        for part in (self.model_name, self.revision, str(self.max_length), self.quantize, query, text):
            h.update(part.encode("utf-8")); h.update(b"\x00")
        return h.hexdigest()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        try:
            self._tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.revision, local_files_only=True)
        except OSError as e:
            raise RerankerUnavailable(
                f"Reranker {self.model_name}@{self.revision} is not in the local Hugging Face cache. "
                f"Fetch it with `python scripts/fetch_reranker.py` ({e})") from e
        self._model.eval()
        if self.quantize == "int8":
            self._model = torch.quantization.quantize_dynamic(self._model, {torch.nn.Linear}, dtype=torch.qint8)
        if self.threads:
            torch.set_num_threads(self.threads)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        keys = [self._key(query, t) for t in texts]
        got: Dict[str, float] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = f"SELECT key, score FROM scores WHERE key IN ({','.join('?' * len(part))})"
            got.update(dict(self._db.execute(q, part).fetchall()))
        todo = [i for i, k in enumerate(keys) if k not in got]
        t0 = time.perf_counter()
        if todo:
            import torch
            with self._lock:
                self._load()
                for s in range(0, len(todo), self.batch_size):
                    idx = todo[s:s + self.batch_size]
                    enc = self._tok([query] * len(idx), [texts[i] for i in idx], padding=True, truncation="only_second",
                                    max_length=self.max_length, return_tensors="pt")
                    with torch.inference_mode():
                        logits = self._model(**enc).logits.view(-1).float().tolist()
                    for i, v in zip(idx, logits):
                        got[keys[i]] = v
                self._db.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?)", [(keys[i], got[keys[i]]) for i in todo])
                self._db.commit()
        self.last = {"n": len(texts), "n_computed": len(todo), "latency_ms": round(1000 * (time.perf_counter() - t0), 2)}
        return [got[k] for k in keys]


def build_reranker(cfg: Dict[str, Any]) -> Optional[CrossEncoderReranker]:
    return CrossEncoderReranker(cfg) if cfg.get("rerank", {}).get("enabled") else None

"""Local NLI scorer for the M5 evidence filter, citation verifier and faithfulness metric (spec §2.6, §2.7).

Runs `nli.model` (default MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, MIT) on the CPU through
`transformers`, pinned to a Hugging Face revision and loaded with `local_files_only=True`: a missing model
is a loud error, never a silent download or another model.

fp32 only: PyTorch dynamic int8 quantization (which works for the bge reranker) breaks this DeBERTa-v3
model. On the M5 probe every pair came out about 80% "neutral", so `nli.quantize` is not offered.

`probs(premise, hypothesis)` returns {entailment, neutral, contradiction}. Results are deterministic and
cached in SQLite by content hash of (model, revision, max_length, premise, hypothesis). Long premises are
truncated from the end ("only_first"), so a hypothesis is never cut.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from typing import Any, Dict, List, Sequence, Tuple

from apex.config import resolve_path

LABELS = ("entailment", "neutral", "contradiction")


class NLIUnavailable(RuntimeError):
    pass


class NLIScorer:
    def __init__(self, cfg: Dict[str, Any]):
        n = cfg["nli"]
        self.model_name, self.revision = n["model"], n.get("revision")
        if not self.revision:
            raise ValueError("nli.revision must pin a Hugging Face commit (spec: pin versions)")
        self.max_length, self.batch_size = int(n["max_length"]), int(n["batch_size"])
        self.threads = int(n.get("threads") or 0)
        self._model = self._tok = None
        self._label_idx: Dict[str, int] = {}
        self._lock = threading.Lock()
        path = resolve_path(n["cache_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS nli (key TEXT PRIMARY KEY, e REAL, n REAL, c REAL)")
        self._db.commit()
        self.calls = 0          # pairs requested since construction
        self.computed = 0       # pairs actually run through the model
        self.seconds = 0.0

    def _key(self, premise: str, hypothesis: str) -> str:
        h = hashlib.sha256()
        for part in (self.model_name, self.revision, str(self.max_length), premise, hypothesis):
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
            raise NLIUnavailable(f"NLI model {self.model_name}@{self.revision} is not in the local Hugging Face "
                                 f"cache. Fetch it with `python scripts/fetch_nli.py` ({e})") from e
        self._model.eval()
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._label_idx = {lab: i for i, lab in id2label.items()}
        if set(self._label_idx) != set(LABELS):
            raise NLIUnavailable(f"NLI model labels {id2label} are not {LABELS}")
        if self.threads:
            torch.set_num_threads(self.threads)

    def probs_many(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        keys = [self._key(p, h) for p, h in pairs]
        got: Dict[str, Tuple[float, float, float]] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = f"SELECT key, e, n, c FROM nli WHERE key IN ({','.join('?' * len(part))})"
            got.update({k: (e, n, c) for k, e, n, c in self._db.execute(q, part).fetchall()})
        todo = [i for i, k in enumerate(keys) if k not in got]
        self.calls += len(pairs)
        if todo:
            import torch
            t0 = time.perf_counter()
            with self._lock:
                self._load()
                for s in range(0, len(todo), self.batch_size):
                    idx = todo[s:s + self.batch_size]
                    enc = self._tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                                    truncation="only_first", max_length=self.max_length, return_tensors="pt")
                    with torch.inference_mode():
                        pr = torch.softmax(self._model(**enc).logits.float(), dim=-1).tolist()
                    for i, row in zip(idx, pr):
                        got[keys[i]] = tuple(row[self._label_idx[lab]] for lab in LABELS)
                self._db.executemany("INSERT OR REPLACE INTO nli VALUES (?, ?, ?, ?)",
                                     [(keys[i], *got[keys[i]]) for i in todo])
                self._db.commit()
            self.computed += len(todo)
            self.seconds += time.perf_counter() - t0
        return [dict(zip(LABELS, got[k])) for k in keys]

    def probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        return self.probs_many([(premise, hypothesis)])[0]

    def stats(self) -> Dict[str, Any]:
        return {"pairs": self.calls, "computed": self.computed, "seconds": round(self.seconds, 2)}


_SHARED: Dict[Tuple[str, str, int], NLIScorer] = {}


def shared_nli(cfg: Dict[str, Any]) -> NLIScorer:
    """One scorer per model+revision+max_length per process (the model is ~740 MB in fp32 RAM)."""
    n = cfg["nli"]
    key = (n["model"], n.get("revision"), int(n["max_length"]))
    if key not in _SHARED:
        _SHARED[key] = NLIScorer(cfg)
    return _SHARED[key]

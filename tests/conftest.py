"""Shared fixtures: an offline fake Ollama transport and a tmp-isolated config."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List

import numpy as np
import pytest

from apex.config import load_config

DIM = 64


def bow_embedding(text: str) -> List[float]:
    """Deterministic hashed bag-of-words vector, so lexical overlap ⇒ cosine similarity."""
    v = np.zeros(DIM, dtype=np.float32)
    text = re.sub(r"^search_(document|query):\s*", "", text)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) < 3:
            continue
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    if not v.any():
        v[0] = 1.0
    return v.tolist()


class FakeOllama:
    """Records every request; answers /api/embed and /api/generate deterministically."""

    def __init__(self, answer_fn=None):
        self.calls: List[Dict[str, Any]] = []
        self.answer_fn = answer_fn or self.default_answer

    @staticmethod
    def default_answer(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"answer": "The capital of Freedonia is Fredville [1].", "citations": [1],
                "confidence": 0.9, "abstained": False}

    def __call__(self, endpoint: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        self.calls.append({"endpoint": endpoint, "payload": payload})
        if endpoint == "/api/embed":
            return {"embeddings": [bow_embedding(t) for t in payload["input"]], "prompt_eval_count": 7}
        if endpoint == "/api/generate":
            return {"response": json.dumps(self.answer_fn(payload)), "prompt_eval_count": 120,
                    "eval_count": 30, "total_duration": 5_000_000}
        raise AssertionError(f"unexpected endpoint {endpoint}")

    def count(self, endpoint: str) -> int:
        return sum(1 for c in self.calls if c["endpoint"] == endpoint)


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={
        "llm": {"cache_path": str(tmp_path / "cache.sqlite"), "log_path": str(tmp_path / "calls.jsonl")},
        "store": {"path": str(tmp_path / "db")},
        "trace": {"dir": str(tmp_path / "traces")},
    })


@pytest.fixture
def fake():
    return FakeOllama()

"""Embeddings through the logged Ollama wrapper, with nomic task prefixes and L2 normalisation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from apex.llm import OllamaClient


class Embedder:
    def __init__(self, cfg: Dict[str, Any], client: OllamaClient):
        e = cfg["embedding"]
        self.client = client
        self.model = cfg["models"]["embedder"]
        self.doc_prefix = e.get("doc_prefix", "")
        self.query_prefix = e.get("query_prefix", "")
        self.batch_size = int(e.get("batch_size", 32))
        self.normalize = bool(e.get("normalize", True))
        self.options = dict(e.get("options") or {})

    def _post(self, vecs: List[List[float]]) -> np.ndarray:
        arr = np.asarray(vecs, dtype=np.float32)
        if self.normalize and arr.size:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Zero-norm embedding returned by the embedding model")
            arr = arr / norms
        return arr

    def embed_documents(self, texts: List[str], caller: str = "ingest.embed") -> np.ndarray:
        vecs = self.client.embed([self.doc_prefix + t for t in texts], model=self.model,
                                 caller=caller, batch_size=self.batch_size, state="INGEST",
                                 options=self.options)
        return self._post(vecs)

    def embed_query(self, text: str, caller: str = "retrieve.embed_query") -> np.ndarray:
        vecs = self.client.embed([self.query_prefix + text], model=self.model, caller=caller, state="RETRIEVING",
                                 options=self.options)
        return self._post(vecs)[0]

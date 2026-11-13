"""FAISS IVF dense index built from the Chroma embeddings (spec §2.4; owner decision: FAISS IVF).

Chroma stays the source of truth for text, metadata and vectors. This is a derived view keyed by
the same node ids. Vectors are L2-normalised, so inner product = cosine similarity.

nlist: the spec's starting point is ≈ 4·√N, with k-means trained on ≥ 39·nlist vectors. Our
benchmark corpora are small (1.6k–4.9k leaves), so 4·√N would need more training vectors than
exist. nlist is therefore capped at N / 39 ([PROPOSED] deviation, see docs/M3_NOTES.md), and
k-means trains on all vectors. Training is single-threaded with a fixed seed, for reproducibility.

Scoped queries (one paper / one story) search only a few dozen to a few hundred vectors; they use
exact search over that subset, since IVF only makes sense for corpus-wide search.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np


def choose_nlist(n: int, factor: float = 4.0, min_points_per_centroid: int = 39) -> int:
    return max(1, min(int(round(factor * math.sqrt(n))), n // min_points_per_centroid))


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class IVFIndex:
    def __init__(self, ids: Sequence[str], vectors: np.ndarray, nlist: Optional[int] = None, seed: int = 42,
                 factor: float = 4.0, min_points_per_centroid: int = 39):
        if len(ids) != len(vectors):
            raise ValueError(f"{len(ids)} ids but {len(vectors)} vectors")
        if not len(ids):
            raise ValueError("IVFIndex needs at least one vector")
        self.ids = list(ids)
        self.vecs = _normalise(vectors)
        n, d = self.vecs.shape
        self.nlist = int(nlist) if nlist else choose_nlist(n, factor, min_points_per_centroid)
        if self.nlist > n:
            raise ValueError(f"nlist={self.nlist} exceeds the {n} vectors available for k-means")
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.cp.seed = int(seed)
        self.index.cp.min_points_per_centroid = 1  # we enforce the 39·nlist rule ourselves via choose_nlist
        threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)  # deterministic k-means
        try:
            self.index.train(self.vecs)
        finally:
            faiss.omp_set_num_threads(threads)
        self.index.add(self.vecs)
        self.build_s = time.perf_counter() - t0
        self.flat = faiss.IndexFlatIP(d)  # exact reference (and scoped search)
        self.flat.add(self.vecs)

    def __len__(self) -> int:
        return len(self.ids)

    def info(self) -> Dict[str, float]:
        return {"n": len(self.ids), "nlist": self.nlist, "build_s": round(self.build_s, 3)}

    def _out(self, scores: np.ndarray, idx: np.ndarray) -> List[Tuple[str, float]]:
        return [(self.ids[i], float(s)) for s, i in zip(scores, idx) if i >= 0]

    def search(self, query: np.ndarray, k: int, nprobe: int, allowed: Optional[np.ndarray] = None,
               exact: bool = False) -> List[Tuple[str, float]]:
        """Top-k (id, cosine). `allowed`: positions to restrict to (exact search over that subset)."""
        q = _normalise(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if allowed is not None:
            sub = self.vecs[allowed]
            sims = sub @ q[0]
            top = np.argsort(-sims, kind="stable")[:k]
            return [(self.ids[int(allowed[i])], float(sims[i])) for i in top]
        if exact:
            s, i = self.flat.search(q, min(k, len(self.ids)))
            return self._out(s[0], i[0])
        self.index.nprobe = max(1, min(int(nprobe), self.nlist))
        s, i = self.index.search(q, min(k, len(self.ids)))
        return self._out(s[0], i[0])

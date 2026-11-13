"""Hybrid retrieval: BM25 ‖ dense (FAISS IVF) → fusion (RRF) → optional cross-encoder rerank (spec §2.4).

Modes (`retrieval.mode`):
  dense      Chroma HNSW top-k: the M1 naive baseline (apex.retrieve.dense.DenseRetriever)
  dense_ivf  dense only, through `retrieval.dense_backend` (ivf | flat | hnsw); for the IVF-vs-HNSW ablation
  bm25       BM25 only
  hybrid     BM25 and dense, `retrieval.candidates` each, fused by `retrieval.fusion` (rrf | weighted),
             fused top `retrieval.fused_k` kept, then reranked when `rerank.enabled`

The generator receives the first `retrieval.top_k` hits of the final ranking. The full ranking
(depth `fused_k`) is what the eval's retrieval metrics score.

The BM25 and FAISS indexes are built in memory from the Chroma collection when the retriever is
constructed (≈1 s for our largest benchmark corpus), so they are always consistent with the store.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from apex.index.bm25 import BM25Index
from apex.index.chroma_store import Hit, NodeStore
from apex.index.embedder import Embedder
from apex.index.ivf import IVFIndex
from apex.retrieve import fusion
from apex.retrieve.rerank import CrossEncoderReranker

MODES = ("dense_ivf", "bm25", "hybrid")
BACKENDS = ("ivf", "flat", "hnsw")


class HybridRetriever:
    def __init__(self, cfg: Dict[str, Any], store: NodeStore, embedder: Embedder,
                 reranker: Optional[CrossEncoderReranker] = None):
        r = cfg["retrieval"]
        self.mode = r["mode"]
        if self.mode not in MODES:
            raise ValueError(f"HybridRetriever mode must be one of {MODES}, got {self.mode!r}")
        self.backend = r.get("dense_backend", "ivf")
        if self.backend not in BACKENDS:
            raise ValueError(f"retrieval.dense_backend must be one of {BACKENDS}, got {self.backend!r}")
        self.fuser = r.get("fusion", "rrf")
        if self.fuser not in fusion.FUSERS:
            raise ValueError(f"retrieval.fusion must be one of {sorted(fusion.FUSERS)}, got {self.fuser!r}")
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.top_k, self.candidates, self.fused_k = int(r["top_k"]), int(r["candidates"]), int(r["fused_k"])
        self.rrf_k, self.weights = int(r.get("rrf_k", 60)), dict(r.get("weights") or {})
        self.levels = list(r.get("levels", [0]))
        self.nprobe = int(cfg["ivf"]["nprobe"])

        where = {"level": {"$in": self.levels}} if self.levels else None
        t0 = time.perf_counter()
        nodes, vecs = store.all_nodes(where)
        if not nodes:
            raise RuntimeError(f"Collection {store.col.name} has no nodes at levels {self.levels}")
        self.nodes = {n.id: n for n in nodes}
        ids = [n.id for n in nodes]
        self._by_source: Dict[str, np.ndarray] = {}
        for i, n in enumerate(nodes):
            self._by_source.setdefault(n.source_path, []).append(i)
        self._by_source = {s: np.asarray(p, dtype=np.int64) for s, p in self._by_source.items()}
        self.bm25 = BM25Index(ids, [n.text for n in nodes], k1=float(cfg["bm25"]["k1"]), b=float(cfg["bm25"]["b"])) \
            if self.mode in ("bm25", "hybrid") else None
        self.ivf = None
        if self.mode in ("dense_ivf", "hybrid") and self.backend in ("ivf", "flat"):
            iv = cfg["ivf"]
            self.ivf = IVFIndex(ids, vecs, nlist=iv.get("nlist"), seed=int(cfg["run"]["seed"]),
                                factor=float(iv["nlist_factor"]), min_points_per_centroid=int(iv["min_points_per_centroid"]))
        self.build_info = {"n_nodes": len(ids), "build_s": round(time.perf_counter() - t0, 3),
                           **({"ivf": self.ivf.info()} if self.ivf else {})}
        self._memo: "OrderedDict[Tuple[str, Optional[str]], Tuple[List[Hit], Dict[str, Any]]]" = OrderedDict()
        self.last: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        parts = {"dense_ivf": [f"dense_{self.backend}"], "bm25": ["bm25"],
                 "hybrid": [f"bm25+dense_{self.backend}", self.fuser]}[self.mode]
        return "|".join(parts + (["rerank"] if self.reranker else []))

    def _allowed(self, source: Optional[str]) -> Optional[np.ndarray]:
        if not source:
            return None
        if source not in self._by_source:
            raise ValueError(f"No nodes from source {source!r} in collection {self.store.col.name}")
        return self._by_source[source]

    def _dense(self, query: str, k: int, source: Optional[str]) -> List[Tuple[str, float]]:
        emb = self.embedder.embed_query(query)
        if self.backend == "hnsw":
            clauses = ([{"level": {"$in": self.levels}}] if self.levels else []) + \
                      ([{"source_path": source}] if source else [])
            where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
            return [(h.node.id, h.score) for h in self.store.query(emb, k, where=where)]
        return self.ivf.search(emb, k, self.nprobe, allowed=self._allowed(source), exact=self.backend == "flat")

    def ranked(self, query: str, source: Optional[str] = None) -> List[Hit]:
        """The full final ranking (depth `fused_k`); memoised per (query, source)."""
        key = (query, source)
        if key in self._memo:
            self._memo.move_to_end(key)
            hits, info = self._memo[key]
            self.last = {**info, "memo_hit": True}
            return hits
        info: Dict[str, Any] = {"retriever": self.name}
        lists: Dict[str, List[Tuple[str, float]]] = {}
        if self.mode in ("dense_ivf", "hybrid"):
            t = time.perf_counter()
            lists["dense"] = self._dense(query, self.candidates, source)
            info["dense_ms"] = round(1000 * (time.perf_counter() - t), 2)
        if self.mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            lists["bm25"] = self.bm25.search(query, self.candidates, allowed=self._allowed(source))
            info["bm25_ms"] = round(1000 * (time.perf_counter() - t), 2)
        if len(lists) == 1:
            fused = next(iter(lists.values()))
        else:
            fused = fusion.rrf(lists, k=self.rrf_k) if self.fuser == "rrf" else fusion.weighted(lists, self.weights)
        fused = fused[:self.fused_k]
        ranks = {name: {nid: r for r, (nid, _) in enumerate(lst, 1)} for name, lst in lists.items()}
        info["n_candidates"] = {name: len(lst) for name, lst in lists.items()}
        info["n_fused"] = len(fused)

        def detail(nid: str, fused_score: float) -> Dict[str, Any]:
            d: Dict[str, Any] = {"fused": fused_score}
            for name in lists:
                d[f"{name}_rank"] = ranks[name].get(nid)
            return d

        hits_in = [(nid, s, detail(nid, s)) for nid, s in fused]
        if self.reranker and hits_in:
            scores = self.reranker.score(query, [self.nodes[nid].text for nid, _, _ in hits_in])
            info["rerank"] = dict(self.reranker.last)
            for r, ((_, _, d), s) in enumerate(zip(hits_in, scores), 1):
                d["fused_rank"], d["rerank"] = r, s
            hits_in = sorted(((nid, d["rerank"], d) for nid, _, d in hits_in), key=lambda x: (-x[1], x[0]))
        hits = [Hit(node=self.nodes[nid], score=float(s), rank=r, detail=d) for r, (nid, s, d) in enumerate(hits_in)]
        self._memo[key] = (hits, info)
        if len(self._memo) > 64:
            self._memo.popitem(last=False)
        self.last = info
        return hits

    def retrieve(self, query: str, k: Optional[int] = None, source: Optional[str] = None) -> List[Hit]:
        k = k or self.top_k
        if k > self.fused_k:
            raise ValueError(f"requested depth {k} > retrieval.fused_k={self.fused_k}")
        return self.ranked(query, source)[:k]

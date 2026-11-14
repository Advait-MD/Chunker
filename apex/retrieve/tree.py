"""RAPTOR tree-traversal retrieval: the spec's ablation against collapsed-tree search (M4).

Collapsed-tree search (the default) needs no special code: `retrieval.levels` lists every level and
the hybrid retriever searches all nodes as one flat pool.

Traversal follows the RAPTOR paper. The frontier starts as the roots: every node in scope without a
parent, so each tree's top level, plus any leaf that was never clustered. Each step scores the
frontier by cosine similarity to the query and keeps the top `retrieval.traversal_k`; their children
form the next frontier, until it runs out at the leaves. The nodes selected at all layers are then
ranked by similarity (the paper concatenates them; ranking them keeps the generator's `top_k` budget
comparable with the other rows) and optionally reranked like the hybrid ranking.

Exact search over the in-memory vectors: each frontier has at most a few hundred nodes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from apex.index.chroma_store import Hit, NodeStore
from apex.index.embedder import Embedder
from apex.retrieve.rerank import CrossEncoderReranker


class TreeTraversalRetriever:
    def __init__(self, cfg: Dict[str, Any], store: NodeStore, embedder: Embedder,
                 reranker: Optional[CrossEncoderReranker] = None):
        r = cfg["retrieval"]
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.top_k, self.fused_k = int(r["top_k"]), int(r["fused_k"])
        self.k_layer = int(r.get("traversal_k", 5))
        t0 = time.perf_counter()
        nodes, vecs = store.all_nodes()
        if not any(n.level > 0 for n in nodes):
            raise RuntimeError(f"Collection {store.col.name} has no summary nodes: tree traversal needs a RAPTOR tree")
        self.nodes = {n.id: n for n in nodes}
        self.row = {n.id: i for i, n in enumerate(nodes)}
        self.vecs = vecs
        self.roots: Dict[Optional[str], List[str]] = {None: [n.id for n in nodes if not n.parent_ids]}
        for nid in self.roots[None]:
            self.roots.setdefault(self.nodes[nid].source_path, []).append(nid)
        self.build_info = {"n_nodes": len(nodes), "n_roots": len(self.roots[None]),
                           "build_s": round(time.perf_counter() - t0, 3)}
        self.last: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "tree_traversal" + ("|rerank" if self.reranker else "")

    def ranked(self, query: str, source: Optional[str] = None) -> List[Hit]:
        if source is not None and source not in self.roots:
            raise ValueError(f"No nodes from source {source!r} in collection {self.store.col.name}")
        q = self.embedder.embed_query(query)
        frontier, picked, layers = list(self.roots[source]), {}, 0
        while frontier:
            sims = self.vecs[[self.row[n] for n in frontier]] @ q
            order = sorted(range(len(frontier)), key=lambda i: (-float(sims[i]), frontier[i]))[:self.k_layer]
            sel = [frontier[i] for i in order]
            for i in order:
                picked.setdefault(frontier[i], (float(sims[i]), layers))
            frontier = sorted({c for nid in sel for c in self.nodes[nid].child_ids if c not in picked})
            layers += 1
        ranking = sorted(picked.items(), key=lambda kv: (-kv[1][0], kv[0]))[:self.fused_k]
        info: Dict[str, Any] = {"retriever": self.name, "layers": layers, "n_selected": len(picked)}
        hits_in = [(nid, s, {"cosine": s, "layer": layer}) for nid, (s, layer) in ranking]
        if self.reranker and hits_in:
            scores = self.reranker.score(query, [self.nodes[nid].text for nid, _, _ in hits_in])
            info["rerank"] = dict(self.reranker.last)
            for r, ((_, _, d), s) in enumerate(zip(hits_in, scores), 1):
                d["pre_rank"], d["rerank"] = r, s
            hits_in = sorted(((nid, d["rerank"], d) for nid, _, d in hits_in), key=lambda x: (-x[1], x[0]))
        self.last = info
        return [Hit(node=self.nodes[nid], score=float(s), rank=r, detail=d) for r, (nid, s, d) in enumerate(hits_in)]

    def retrieve(self, query: str, k: Optional[int] = None, source: Optional[str] = None) -> List[Hit]:
        k = k or self.top_k
        if k > self.fused_k:
            raise ValueError(f"requested depth {k} > retrieval.fused_k={self.fused_k}")
        return self.ranked(query, source)[:k]

"""RAPTOR hierarchical summary tree (spec §2.3; Sarthi et al., ICLR 2024).

Leaves are the ingested chunks; each parent is an LLM summary of a cluster of its children.

Clustering follows the RAPTOR paper and reference code:
  1. global pass: UMAP to `umap_dim` dims (cosine, n_neighbors = sqrt(N - 1)) → GMM, k chosen by BIC;
  2. local pass inside each global cluster: UMAP (n_neighbors = `local_neighbors`) → GMM by BIC;
  3. soft assignment: a node joins every cluster whose posterior exceeds `threshold` (several parents);
  4. a cluster whose children exceed `cluster_max_tokens` is re-clustered recursively.
Summaries are embedded and become the next level; repeat until a level yields fewer than
`min_clusters` clusters or `max_depth` is reached.

The tree is written to its own Chroma collection, `<leaf collection>__<raptor.tag>_s<raptor.seed>`. It holds copies
of the leaves (stored embeddings reused, `parent_ids` filled in) plus the summary nodes, so
tree-less ablation rows keep reading the untouched leaf collection, and trees built with different
seeds or parameters never mix.

Determinism: UMAP and GMM take `raptor.seed`, and summaries are temperature-0 calls through the
logged, content-hash-cached LLM wrapper. Each summary node also records the spec's cache key,
sha256(sorted child ids + child texts) with the model and prompt, so a rebuild on unchanged input
is cache hits only.

Nothing degrades silently. Two deterministic rules cover cases the paper's code leaves open, and
the build report counts both:
  * argmax assignment: a node no cluster claims above `threshold` joins its most likely cluster
    (the reference code would drop it from the level);
  * forced split: a cluster over the token budget that re-clustering cannot split (too few nodes
    for UMAP, or GMM picks one component) is cut into consecutive id-ordered groups under the budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from apex.config import resolve_path
from apex.index.chroma_store import Node, NodeStore, tree_collection_name
from apex.index.embedder import Embedder
from apex.ingest.chunker import count_tokens
from apex.llm.ollama_client import OllamaClient

PROMPT_PATH = Path(__file__).resolve().parents[1] / "generate" / "prompts" / "raptor_summary.txt"
SCOPES = ("per_source", "collection")


def tree_collection(cfg: Dict[str, Any]) -> str:
    return tree_collection_name(cfg)


# =============================================================================
# Clustering
# =============================================================================
@dataclass
class ClusterParams:
    dim: int = 10
    local_neighbors: int = 10
    max_clusters: int = 50
    threshold: float = 0.1
    max_tokens: int = 3500
    seed: int = 42

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "ClusterParams":
        r = cfg["raptor"]
        return cls(dim=int(r["umap_dim"]), local_neighbors=int(r["local_neighbors"]),
                   max_clusters=int(r["max_clusters"]), threshold=float(r["threshold"]),
                   max_tokens=int(r["cluster_max_tokens"]), seed=int(r["seed"]))


@dataclass
class ClusterStats:
    argmax_assigned: int = 0
    forced_splits: int = 0
    recursions: int = 0


def _reduce(x: np.ndarray, dim: int, n_neighbors: int, seed: int) -> np.ndarray:
    import umap
    n = len(x)
    reducer = umap.UMAP(n_neighbors=max(2, min(n_neighbors, n - 1)), n_components=min(dim, n - 2),
                        metric="cosine", random_state=seed,
                        # spectral init needs a graph larger than the embedding dimension
                        init="spectral" if n > 4 * dim else "random")
    return reducer.fit_transform(x)


def _gmm(y: np.ndarray, p: ClusterParams, stats: ClusterStats) -> List[np.ndarray]:
    """Soft GMM labels per point; k by BIC over 1..min(max_clusters, n) - 1, as in the reference code."""
    from sklearn.mixture import GaussianMixture
    n = len(y)
    ks = list(range(1, max(2, min(p.max_clusters, n))))
    bics = [GaussianMixture(n_components=k, random_state=p.seed).fit(y).bic(y) for k in ks]
    k = ks[int(np.argmin(bics))]
    probs = GaussianMixture(n_components=k, random_state=p.seed).fit(y).predict_proba(y)
    labels = []
    for pr in probs:
        lab = np.where(pr > p.threshold)[0]
        if len(lab) == 0:
            lab = np.array([int(np.argmax(pr))])
            stats.argmax_assigned += 1
        labels.append(lab)
    return labels


def _two_stage(x: np.ndarray, p: ClusterParams, stats: ClusterStats) -> List[List[int]]:
    """RAPTOR global → local clustering; returns clusters as sorted lists of row indices."""
    n = len(x)
    if n <= p.dim + 1:
        return [list(range(n))]  # too few points for a dim-`dim` UMAP: one global cluster
    g_labels = _gmm(_reduce(x, p.dim, int(math.sqrt(n - 1)), p.seed), p, stats)
    n_global = 1 + max(int(l.max()) for l in g_labels)
    clusters: List[List[int]] = []
    for g in range(n_global):
        members = [i for i, l in enumerate(g_labels) if g in l]
        if not members:
            continue
        if len(members) <= p.dim + 1:
            clusters.append(members)
            continue
        l_labels = _gmm(_reduce(x[members], p.dim, p.local_neighbors, p.seed), p, stats)
        for j in range(1 + max(int(l.max()) for l in l_labels)):
            sub = [members[i] for i, l in enumerate(l_labels) if j in l]
            if sub:
                clusters.append(sub)
    return clusters


def _forced_split(idx: List[int], tokens: Sequence[int], max_tokens: int) -> List[List[int]]:
    groups, cur, tot = [], [], 0
    for i in sorted(idx):
        if cur and tot + tokens[i] > max_tokens:
            groups.append(cur); cur, tot = [], 0
        cur.append(i); tot += tokens[i]
    if cur:
        groups.append(cur)
    return groups


def cluster_nodes(x: np.ndarray, tokens: Sequence[int], p: ClusterParams,
                  stats: Optional[ClusterStats] = None) -> List[List[int]]:
    """Clusters of row indices (each sorted; clusters ordered by first member), every one within budget."""
    stats = stats if stats is not None else ClusterStats()

    def rec(idx: List[int]) -> List[List[int]]:
        out = []
        for c in _two_stage(x[idx], p, stats):
            members = sorted(idx[i] for i in c)
            if len(members) == 1 or sum(tokens[i] for i in members) <= p.max_tokens:
                out.append(members)
            elif len(members) < len(idx):
                stats.recursions += 1
                out.extend(rec(members))
            else:  # re-clustering could not split it
                stats.forced_splits += 1
                out.extend(_forced_split(members, tokens, p.max_tokens))
        return out

    if len(x) == 0:
        return []
    uniq = {tuple(c): None for c in sorted(rec(list(range(len(x)))), key=lambda c: (c[0], c))}
    return [list(c) for c in uniq]


# =============================================================================
# Tree build
# =============================================================================
@dataclass
class TreeReport:
    collection: str
    tree_hash: str
    groups: int = 0
    leaves: int = 0
    summaries_per_level: Dict[int, int] = field(default_factory=dict)
    children_per_summary: List[int] = field(default_factory=list)
    truncated_summaries: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_s: float = 0.0
    cluster_s: float = 0.0
    stats: ClusterStats = field(default_factory=ClusterStats)
    status: str = "built"

    def to_dict(self) -> Dict[str, Any]:
        c = self.children_per_summary
        return {"collection": self.collection, "tree_hash": self.tree_hash, "status": self.status,
                "groups": self.groups, "leaves": self.leaves,
                "summaries_per_level": {str(k): v for k, v in sorted(self.summaries_per_level.items())},
                "summaries": sum(self.summaries_per_level.values()),
                "children_per_summary_mean": round(float(np.mean(c)), 2) if c else None,
                "children_per_summary_max": max(c) if c else None,
                "truncated_summaries": self.truncated_summaries, "llm_calls": self.llm_calls,
                "llm_cache_hits": self.llm_cache_hits, "llm_s": round(self.llm_s, 1), "cluster_s": round(self.cluster_s, 1),
                "argmax_assigned": self.stats.argmax_assigned, "forced_splits": self.stats.forced_splits,
                "recursions": self.stats.recursions}


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for s in parts:
        h.update(s.encode("utf-8")); h.update(b"\x00")
    return h.hexdigest()


def tree_build_hash(cfg: Dict[str, Any], leaf_hashes: Sequence[str], prompt: str) -> str:
    r = cfg["raptor"]
    material = {k: r[k] for k in ("seed", "umap_dim", "local_neighbors", "max_clusters", "threshold",
                                  "cluster_max_tokens", "max_depth", "min_clusters", "scope",
                                  "summarizer", "summary_max_tokens", "num_ctx")}
    material.update({"leaves": sorted(set(leaf_hashes)), "prompt": _sha(prompt), "embedder": cfg["models"]["embedder"],
                     "schema": 1})
    return _sha(json.dumps(material, sort_keys=True))[:16]


class TreeBuilder:
    def __init__(self, cfg: Dict[str, Any], leaf_store: NodeStore, embedder: Embedder, client: OllamaClient,
                 log: Callable[[str], None] = print):
        r = cfg["raptor"]
        if r["scope"] not in SCOPES:
            raise ValueError(f"raptor.scope must be one of {SCOPES}, got {r['scope']!r}")
        self.cfg, self.r = cfg, r
        self.leaf_store, self.embedder, self.client, self.log = leaf_store, embedder, client, log
        self.params = ClusterParams.from_cfg(cfg)
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.model = r["summarizer"]

    def _manifest_path(self, collection: str) -> Path:
        return resolve_path(self.cfg["store"]["path"]) / "trees" / f"{collection}.json"

    def summarize(self, texts: List[str], rep: TreeReport) -> str:
        passages = "\n\n".join(f"[{i + 1}] {t.strip()}" for i, t in enumerate(texts))
        res = self.client.generate(self.prompt.replace("{passages}", passages), model=self.model,
                                   caller="raptor.summarize", temperature=0.0,
                                   max_tokens=int(self.r["summary_max_tokens"]), num_ctx=int(self.r["num_ctx"]))
        rep.llm_calls += 1
        rep.llm_cache_hits += int(res.cache_hit)
        rep.llm_s += res.latency_ms / 1000.0
        if res.raw.get("done_reason") == "length":
            rep.truncated_summaries += 1
        text = res.text.strip()
        if not text:
            raise RuntimeError(f"Empty summary from {self.model} (call {res.call_id})")
        return text

    def build(self, force: bool = False) -> Dict[str, Any]:
        cfg = self.cfg
        leaves, vecs = self.leaf_store.all_nodes({"level": 0})
        if not leaves:
            raise RuntimeError(f"Leaf collection {self.leaf_store.col.name} is empty: ingest first")
        collection = tree_collection(cfg)
        th = tree_build_hash(cfg, [n.build_hash for n in leaves], self.prompt)
        manifest = self._manifest_path(collection)
        if not force and manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("tree_hash") == th:
                self.log(f"  = {collection}: tree {th} up to date")
                return {**old, "status": "cached"}

        rep = TreeReport(collection=collection, tree_hash=th, leaves=len(leaves))
        leaf_by_id = {n.id: n for n in leaves}
        vec_by_id = {n.id: v for n, v in zip(leaves, vecs)}
        if self.r["scope"] == "per_source":
            groups: Dict[str, List[str]] = {}
            for n in leaves:
                groups.setdefault(n.source_path, []).append(n.id)
        else:
            groups = {"__collection__": [n.id for n in leaves]}
        rep.groups = len(groups)

        now = datetime.now(timezone.utc).isoformat()
        summaries: List[Node] = []
        sum_vecs: List[np.ndarray] = []
        t_start = time.perf_counter()
        for gi, (gname, ids) in enumerate(sorted(groups.items())):
            level_nodes = [leaf_by_id[i] for i in sorted(ids)]
            level_vecs = np.stack([vec_by_id[i] for i in sorted(ids)])
            first = level_nodes[0]
            doc_id = first.doc_id if self.r["scope"] == "per_source" else f"raptor:{self.leaf_store.col.name}"
            src = first.source_path if self.r["scope"] == "per_source" else "__raptor__"
            tag = _sha(gname)[:8]
            for level in range(1, int(self.r["max_depth"]) + 1):
                if len(level_nodes) < 2:
                    break
                t0 = time.perf_counter()
                clusters = cluster_nodes(level_vecs, [n.token_count for n in level_nodes], self.params, rep.stats)
                rep.cluster_s += time.perf_counter() - t0
                new_nodes: List[Node] = []
                for c in clusters:
                    kids = [level_nodes[i] for i in c]
                    text = self.summarize([k.text for k in kids], rep)
                    key = _sha(self.model, _sha(self.prompt), *[k.id for k in kids], *[k.text for k in kids])
                    node = Node(id=f"{self.r['tag']}:{tag}:L{level}:{key[:16]}", text=text, level=level, doc_id=doc_id,
                                source_path=src, child_ids=[k.id for k in kids], token_count=count_tokens(text),
                                created_at=now, build_hash=th)
                    for k in kids:
                        if node.id not in k.parent_ids:
                            k.parent_ids.append(node.id)
                    new_nodes.append(node)
                    rep.children_per_summary.append(len(kids))
                if not new_nodes:
                    break
                new_vecs = self.embedder.embed_documents([n.text for n in new_nodes], caller="raptor.embed")
                summaries += new_nodes
                sum_vecs += list(new_vecs)
                rep.summaries_per_level[level] = rep.summaries_per_level.get(level, 0) + len(new_nodes)
                if len(clusters) < int(self.r["min_clusters"]):
                    break
                level_nodes, level_vecs = new_nodes, np.asarray(new_vecs)
            if (gi + 1) % 10 == 0 or gi + 1 == len(groups):
                el = time.perf_counter() - t_start
                self.log(f"  [{collection}] {gi + 1}/{len(groups)} groups, {len(summaries)} summaries, "
                         f"{rep.llm_calls - rep.llm_cache_hits} new LLM calls, {el / 60:.1f} min")

        tree_store = NodeStore(cfg, collection=collection)
        tree_store.reset()
        all_nodes = [leaf_by_id[i] for i in sorted(leaf_by_id)] + summaries
        all_vecs = np.concatenate([np.stack([vec_by_id[n.id] for n in all_nodes[:len(leaves)]]),
                                   np.asarray(sum_vecs, dtype=np.float32).reshape(-1, vecs.shape[1])])
        tree_store.upsert(all_nodes, all_vecs)
        out = rep.to_dict()
        out["built_at"] = now
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(out, indent=1), encoding="utf-8")
        self.log(f"  + {collection}: {out['summaries']} summaries {out['summaries_per_level']}, "
                 f"{out['llm_calls'] - out['llm_cache_hits']} new LLM calls in {out['llm_s'] / 60:.1f} min")
        return out

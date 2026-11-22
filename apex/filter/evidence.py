"""Evidence filter between the reranker and the generator (spec §2.6, M5).

Stages, in order (each one is switchable in `filter:` in the config, so each can be ablated):
  1. relevance threshold  drop hits whose reranker score is below `rel_threshold`; always keep `min_keep`
  2. dedup                collapse near-duplicates (word-shingle Jaccard > `dedup.jaccard`); keep the leaf
                          over a summary (a leaf cites a precise span), otherwise the higher score
  3. sentence pruning     keep the sentences whose embedding similarity to the query passes
                          `prune.threshold`, plus `prune.neighbors` sentences on each side; a passage with
                          no passing sentence keeps its best one (it already passed the relevance stage)
  4. contradictions       optional NLI over passage pairs; contradicting pairs go to `EvidenceSet.conflicts`
  5. sufficiency          `apex.filter.sufficiency` (separate, because a failure triggers a retry)
  6. pack to budget       order by score, fit `generation.context_token_budget`, cite ids [1..n]

Exact Jaccard on 5-word shingles stands in for MinHash: the sets are a handful of passages, so the exact
value costs nothing. Sentence embeddings go through the logged, cached Ollama embedder.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from apex.generate.citations import split_sentences
from apex.generate.generator import EvidenceItem, EvidenceSet
from apex.index.chroma_store import Hit
from apex.ingest.chunker import count_tokens

_WORD = re.compile(r"\w+", re.U)


def shingles(text: str, n: int) -> set:
    w = _WORD.findall((text or "").lower())
    if len(w) < n:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def relevance_threshold(hits: Sequence[Hit], threshold: Optional[float], min_keep: int) -> Tuple[List[Hit], List[Hit]]:
    """Hits at or above `threshold` (None = stage off), topped up to `min_keep` by score. Returns (kept, dropped)."""
    ranked = sorted(hits, key=lambda h: -h.score)
    if threshold is None:
        return list(ranked), []
    kept = [h for h in ranked if h.score >= threshold]
    if len(kept) < min_keep:
        kept = ranked[:min(min_keep, len(ranked))]
    ids = {h.node.id for h in kept}
    return kept, [h for h in ranked if h.node.id not in ids]


def dedup(hits: Sequence[Hit], threshold: float, n: int) -> Tuple[List[Hit], List[Tuple[str, str, float]]]:
    """Greedy near-duplicate removal. Visits leaves first, then by score, so a leaf beats a summary.
    Returns (kept in the input order, [(dropped_id, kept_id, jaccard)])."""
    order = sorted(hits, key=lambda h: (h.node.level > 0, -h.score))
    sh = {h.node.id: shingles(h.node.text, n) for h in hits}
    kept: List[Hit] = []
    dropped: List[Tuple[str, str, float]] = []
    for h in order:
        dup = next(((k, j) for k in kept if (j := jaccard(sh[h.node.id], sh[k.node.id])) > threshold), None)
        if dup:
            dropped.append((h.node.id, dup[0].node.id, round(dup[1], 3)))
        else:
            kept.append(h)
    ids = {h.node.id for h in kept}
    return [h for h in hits if h.node.id in ids], dropped


def select_sentences(sims: Sequence[float], threshold: float, neighbors: int) -> List[int]:
    """Indices of the sentences to keep: those passing `threshold` plus `neighbors` on each side.
    With none passing, the best sentence and its neighbours."""
    if not len(sims):
        return []
    core = [i for i, s in enumerate(sims) if s >= threshold] or [int(np.argmax(sims))]
    keep = set()
    for i in core:
        keep.update(range(max(0, i - neighbors), min(len(sims), i + neighbors + 1)))
    return sorted(keep)


def join_kept(sents: Sequence[str], keep: Sequence[int]) -> str:
    """Join kept sentences; a gap between runs is marked with an ellipsis so the reader knows text was cut."""
    out, prev = [], None
    for i in keep:
        if prev is not None and i != prev + 1:
            out.append("…")
        out.append(sents[i])
        prev = i
    return " ".join(out)


@dataclass
class FilterResult:
    evidence: EvidenceSet
    kept_hits: List[Hit]
    stats: Dict[str, Any] = field(default_factory=dict)


class EvidenceFilter:
    def __init__(self, cfg: Dict[str, Any], embedder=None, nli=None):
        f = cfg["filter"]
        self.min_keep = int(f["min_keep"])
        self.rel_threshold = None if f.get("rel_threshold") is None else float(f["rel_threshold"])
        self.dedup_cfg, self.prune_cfg, self.conf_cfg = f["dedup"], f["prune"], f["conflicts"]
        self.budget = int(cfg["generation"]["context_token_budget"])
        self.embedder, self.nli = embedder, nli
        if self.prune_cfg["enabled"] and embedder is None:
            raise ValueError("filter.prune.enabled needs an embedder")
        if self.conf_cfg["enabled"] and nli is None:
            raise ValueError("filter.conflicts.enabled needs the NLI scorer")

    # ---- stage 3 ----
    def prune(self, query: str, hits: Sequence[Hit]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Pruned text per node id, and stats. Passages of one sentence are left whole."""
        p = self.prune_cfg
        sents = {h.node.id: split_sentences(h.node.text) for h in hits}
        flat = [(nid, i, s) for nid, ss in sents.items() if len(ss) > 1 for i, s in enumerate(ss)]
        texts = {h.node.id: h.node.text for h in hits}
        if not flat:
            return texts, {"sentences_in": sum(len(s) for s in sents.values()), "sentences_kept": None}
        q = self.embedder.embed_query(query, caller="filter.prune_query")
        vecs = self.embedder.embed_documents([s for _, _, s in flat], caller="filter.prune_sentences")
        sims = vecs @ q
        by_node: Dict[str, List[float]] = {}
        for (nid, _, _), v in zip(flat, sims):
            by_node.setdefault(nid, []).append(float(v))
        n_in = n_kept = 0
        for nid, ss in sents.items():
            if len(ss) <= 1:
                n_in += len(ss); n_kept += len(ss)
                continue
            keep = select_sentences(by_node[nid], float(p["threshold"]), int(p["neighbors"]))
            texts[nid] = join_kept(ss, keep)
            n_in += len(ss); n_kept += len(keep)
        return texts, {"sentences_in": n_in, "sentences_kept": n_kept}

    # ---- stage 4 ----
    def conflicts(self, items: Sequence[EvidenceItem]) -> List[Dict[str, Any]]:
        th = float(self.conf_cfg["threshold"])
        pairs, idx = [], []
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                pairs += [(items[a].text, items[b].text), (items[b].text, items[a].text)]
                idx += [(a, b), (b, a)]
        best: Dict[Tuple[int, int], float] = {}  # unordered pair → max contradiction over both directions
        for (a, b), pr in zip(idx, self.nli.probs_many(pairs)):
            key = tuple(sorted((items[a].cite_id, items[b].cite_id)))
            best[key] = max(best.get(key, 0.0), pr["contradiction"])
        return [{"cite_ids": list(k), "contradiction": round(v, 3)} for k, v in sorted(best.items()) if v >= th]

    # ---- stages 1-4 and 6 ----
    def run(self, query: str, hits: Sequence[Hit]) -> FilterResult:
        t0 = time.perf_counter()
        st: Dict[str, Any] = {"n_in": len(hits), "tokens_in": sum(h.node.token_count or count_tokens(h.node.text) for h in hits)}
        kept, dropped = relevance_threshold(hits, self.rel_threshold, self.min_keep)
        st["dropped_threshold"] = [h.node.id for h in dropped]
        if self.dedup_cfg["enabled"]:
            kept, dups = dedup(kept, float(self.dedup_cfg["jaccard"]), int(self.dedup_cfg["shingle"]))
            st["dropped_dedup"] = dups
        texts = {h.node.id: h.node.text for h in kept}
        if self.prune_cfg["enabled"] and kept:
            texts, pst = self.prune(query, kept)
            st.update(pst)
        # stage 6: pack by score to the token budget (always keeps the top passage)
        items: List[EvidenceItem] = []
        used = 0
        for h in sorted(kept, key=lambda x: -x.score):
            t = count_tokens(texts[h.node.id])
            if items and used + t > self.budget:
                continue
            used += t
            items.append(EvidenceItem(cite_id=len(items) + 1, node_id=h.node.id, text=texts[h.node.id], score=h.score,
                                      source=h.node.source_path, page=h.node.page, section=h.node.section,
                                      char_span=h.node.char_span))
        ev = EvidenceSet(items=items)
        if self.conf_cfg["enabled"] and len(items) > 1:
            ev.conflicts = self.conflicts(items)
        st.update({"n_out": len(items), "tokens_out": used, "ms": round(1000 * (time.perf_counter() - t0), 1)})
        kept_ids = {it.node_id for it in items}
        return FilterResult(evidence=ev, kept_hits=[h for h in kept if h.node.id in kept_ids], stats=st)

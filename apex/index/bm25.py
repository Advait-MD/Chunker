"""BM25 sparse index over node text (spec §2.4: k1 = 1.2, b = 0.75, `rank_bm25`).

Tokenisation: lowercase word tokens (`\\w+`), a small English stopword list removed, no stemming.
Exact tokens (names, numbers, acronyms) are what BM25 is here for, so stemming is left out.
IDF is computed over the whole collection; a scoped query (one paper or story) masks the scores
to that scope instead of re-indexing it.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have he her hers him his how i if in into
is it its me my no nor not of on or our ours she so than that the their theirs them then there these they this those
to too us was we were what when where which while who whom why will with would you your yours
""".split())


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        if len(ids) != len(texts):
            raise ValueError(f"{len(ids)} ids but {len(texts)} texts")
        if not ids:
            raise ValueError("BM25Index needs at least one document")
        self.ids = list(ids)
        self.pos = {nid: i for i, nid in enumerate(self.ids)}
        # rank_bm25 divides by zero on an empty corpus document; keep a placeholder token instead
        self.bm25 = BM25Okapi([tokenize(t) or ["∅"] for t in texts], k1=k1, b=b)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int, allowed: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """Top-k (id, score) with score > 0. `allowed`: positions (into self.ids) to restrict to."""
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        if allowed is not None:
            masked = np.full_like(scores, -np.inf)
            masked[allowed] = scores[allowed]
            scores = masked
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def positions(self, ids: Sequence[str]) -> np.ndarray:
        return np.asarray([self.pos[i] for i in ids], dtype=np.int64)

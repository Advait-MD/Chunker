"""Rank fusion for hybrid retrieval (spec §2.4).

`rrf`: Reciprocal Rank Fusion, score(d) = Σ_i 1 / (k + rank_i(d)) with 1-based ranks and k = 60.
No score calibration between BM25 and cosine is needed, which is why it is the default.
`weighted`: the ablation, Σ_i w_i · minmax_i(score), with per-list min-max normalisation.
Ties are broken by node id, so fused orders are deterministic.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

Ranked = Sequence[Tuple[str, float]]  # (node id, score), best first


def rrf(lists: Mapping[str, Ranked], k: int = 60) -> List[Tuple[str, float]]:
    fused: Dict[str, float] = {}
    for ranked in lists.values():
        for r, (nid, _) in enumerate(ranked, 1):
            fused[nid] = fused.get(nid, 0.0) + 1.0 / (k + r)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def weighted(lists: Mapping[str, Ranked], weights: Mapping[str, float]) -> List[Tuple[str, float]]:
    missing = set(lists) - set(weights)
    if missing:
        raise ValueError(f"no fusion weight for {sorted(missing)}")
    fused: Dict[str, float] = {}
    for name, ranked in lists.items():
        if not ranked:
            continue
        scores = [s for _, s in ranked]
        lo, hi = min(scores), max(scores)
        for nid, s in ranked:
            norm = (s - lo) / (hi - lo) if hi > lo else 1.0
            fused[nid] = fused.get(nid, 0.0) + float(weights[name]) * norm
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


FUSERS = {"rrf", "weighted"}

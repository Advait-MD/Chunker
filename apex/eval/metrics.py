"""Evaluation metrics (spec §2.8). Pure functions; no model calls.

Answer metrics use the official SQuAD normalisation (lowercase, strip punctuation and the articles
a/an/the, collapse whitespace) and take the max over gold aliases. Citation markers like "[2]" are
stripped from predictions first.

Retrieval is scored against gold *evidence text*, not chunk ids, so any chunker can be evaluated:
a retrieved chunk covers a gold passage when their word 5-gram shingles overlap by at least
`min_overlap` of the smaller of the two (handles chunks that split or merge paragraphs).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import stats

_CITE = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—"}


# ---- answer metrics ----------------------------------------------------------
def strip_citations(text: str) -> str:
    return _CITE.sub("", text or "")


def normalize_answer(text: str) -> str:
    text = strip_citations(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds)) if golds else 0.0


def _f1(pred_toks: List[str], gold_toks: List[str]) -> float:
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred_toks), same / len(gold_toks)
    return 2 * p * r / (p + r)


def token_f1(pred: str, golds: Sequence[str]) -> float:
    pt = normalize_answer(pred).split()
    return max((_f1(pt, normalize_answer(g).split()) for g in golds), default=0.0)


def answer_contains(pred: str, golds: Sequence[str]) -> float:
    """"Acc" used for LLM generators (e.g. Self-RAG, IRCoT): a normalised gold alias occurs in the prediction."""
    p = f" {normalize_answer(pred)} "
    return float(any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g)))


_LETTER = re.compile(r"(?:^|[\s(\[*])([A-D])(?=[)\].:,\s]|$)")


def parse_choice(pred: str) -> Optional[int]:
    """First standalone option letter A–D in the prediction → 0-based index, else None."""
    m = _LETTER.search(strip_citations(pred or "").strip())
    return "ABCD".index(m.group(1)) if m else None


# ---- retrieval metrics -------------------------------------------------------
def _shingles(text: str, n: int) -> set:
    toks = normalize_answer(text).split()
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def covers(chunk: str, evidence: str, n: int = 5, min_overlap: float = 0.5) -> bool:
    a, b = _shingles(chunk, n), _shingles(evidence, n)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= min_overlap


def evidence_hits(retrieved_texts: Sequence[str], gold_texts: Sequence[str], n: int = 5,
                  min_overlap: float = 0.5) -> List[List[int]]:
    """For each retrieved rank, the indices of gold passages it covers."""
    gold_sh = [_shingles(g, n) for g in gold_texts]
    out = []
    for t in retrieved_texts:
        a = _shingles(t, n)
        out.append([j for j, b in enumerate(gold_sh)
                    if a and b and len(a & b) / min(len(a), len(b)) >= min_overlap])
    return out


def recall_at_k(hits: List[List[int]], n_gold: int, k: int) -> float:
    """Fraction of gold passages covered by the top-k retrieved chunks."""
    if n_gold == 0:
        raise ValueError("recall_at_k needs at least one gold passage")
    found = {j for row in hits[:k] for j in row}
    return len(found) / n_gold


def union_recall_at_k(retrieved_texts: Sequence[str], gold_texts: Sequence[str], k: int, hits: List[List[int]],
                      n: int = 5, min_overlap: float = 0.5) -> float:
    """Recall@k where a gold passage also counts as found when the *union* of the top-k chunks covers it.

    Needed when chunk boundaries differ from the gold segmentation, e.g. a gold passage that crosses a
    page break is split over two chunks, neither of which alone reaches `min_overlap`.
    """
    if not gold_texts:
        raise ValueError("union_recall_at_k needs at least one gold passage")
    found = {j for row in hits[:k] for j in row}
    pool = set().union(*(_shingles(t, n) for t in retrieved_texts[:k])) if retrieved_texts[:k] else set()
    for j, g in enumerate(gold_texts):
        if j in found:
            continue
        gs = _shingles(g, n)
        if gs and len(gs & pool) / len(gs) >= min_overlap:
            found.add(j)
    return len(found) / len(gold_texts)


def mrr(hits: List[List[int]]) -> float:
    for rank, row in enumerate(hits, 1):
        if row:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: List[List[int]], n_gold: int, k: int) -> float:
    """Binary relevance; a chunk is relevant if it covers a gold passage not already credited."""
    seen, dcg = set(), 0.0
    for rank, row in enumerate(hits[:k], 1):
        new = [j for j in row if j not in seen]
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen.update(new)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_gold, k) + 1))
    return dcg / ideal if ideal else 0.0


# ---- abstention ----------------------------------------------------------------
def abstention_scores(abstained: Sequence[bool], answerable: Sequence[bool]) -> Dict[str, Optional[float]]:
    """Precision/recall of abstaining, treating 'unanswerable' as the positive class."""
    tp = sum(1 for a, ok in zip(abstained, answerable) if a and not ok)
    fp = sum(1 for a, ok in zip(abstained, answerable) if a and ok)
    fn = sum(1 for a, ok in zip(abstained, answerable) if not a and not ok)
    n_ans = sum(1 for ok in answerable if ok)
    return {
        "n_unanswerable": tp + fn,
        "abstain_precision": tp / (tp + fp) if tp + fp else None,
        "abstain_recall": tp / (tp + fn) if tp + fn else None,
        "false_abstain_rate": fp / n_ans if n_ans else None,
    }


# ---- statistics ------------------------------------------------------------------
def bootstrap_ci(values: Iterable[float], n: int = 1000, ci: float = 0.95, seed: int = 42) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
    a = (1 - ci) / 2
    return {"mean": float(arr.mean()), "lo": float(np.quantile(means, a)), "hi": float(np.quantile(means, 1 - a)),
            "n": int(arr.size)}


def paired_test(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Paired comparison of system A vs B on the same items: mean diff, 95% CI, bootstrap and Wilcoxon p."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        raise ValueError(f"paired_test needs equal-length samples of size >= 2 (got {x.size}, {y.size})")
    d = x - y
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    # two-sided bootstrap p: how often the resampled mean lands on the other side of 0
    p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    p_wil = 1.0 if not np.any(d) else float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
    return {"mean_diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "p_bootstrap": p_boot, "p_wilcoxon": p_wil, "n": int(d.size)}


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None

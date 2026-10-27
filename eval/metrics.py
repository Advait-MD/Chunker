"""
eval/metrics.py — Retrieval, Generation, and Cost Metrics Engine for APEX Evaluation Harness.
Strictly implements the metrics defined in HARNESS.md Section 7.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def compute_recall_at_k(retrieved_ids: List[str], gold_chunk_ids: List[str], k: int) -> float:
    """
    Fraction of gold chunks present in the top k retrieved chunks.
    Formula: |top_k_retrieved ∩ gold_chunks| / |gold_chunks|
    """
    if not gold_chunk_ids:
        return 0.0
    gold_set = set(gold_chunk_ids)
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & gold_set)
    return float(hits / len(gold_set))


def compute_mrr(retrieved_ids: List[str], gold_chunk_ids: List[str]) -> float:
    """
    Mean Reciprocal Rank of the first relevant gold chunk.
    Formula: 1 / rank (1-indexed) of the first gold chunk in retrieved_ids, or 0.0 if not found.
    """
    if not gold_chunk_ids or not retrieved_ids:
        return 0.0
    gold_set = set(gold_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute retrieval metrics over answerable cases only:
    - recall@1, recall@3, recall@5, recall@10
    - mrr
    - recall@5 split by single-hop (hops == 1) vs multi-hop (hops >= 2)
    """
    answerable_cases = [c for c in cases if c.get("answerable", True)]
    if not answerable_cases:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "recall@5_single_hop": 0.0,
            "recall@5_multi_hop": 0.0,
        }

    r1_list = []
    r3_list = []
    r5_list = []
    r10_list = []
    mrr_list = []

    r5_single_hop = []
    r5_multi_hop = []

    for c in answerable_cases:
        retrieved = c.get("retrieved_ids", [])
        gold = c.get("gold_chunk_ids", [])
        hops = c.get("hops", 1)

        r1 = compute_recall_at_k(retrieved, gold, 1)
        r3 = compute_recall_at_k(retrieved, gold, 3)
        r5 = compute_recall_at_k(retrieved, gold, 5)
        r10 = compute_recall_at_k(retrieved, gold, 10)
        mrr = compute_mrr(retrieved, gold)

        r1_list.append(r1)
        r3_list.append(r3)
        r5_list.append(r5)
        r10_list.append(r10)
        mrr_list.append(mrr)

        if hops == 1:
            r5_single_hop.append(r5)
        elif hops >= 2:
            r5_multi_hop.append(r5)

    return {
        "recall@1": float(np.mean(r1_list)) if r1_list else 0.0,
        "recall@3": float(np.mean(r3_list)) if r3_list else 0.0,
        "recall@5": float(np.mean(r5_list)) if r5_list else 0.0,
        "recall@10": float(np.mean(r10_list)) if r10_list else 0.0,
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "recall@5_single_hop": float(np.mean(r5_single_hop)) if r5_single_hop else 0.0,
        "recall@5_multi_hop": float(np.mean(r5_multi_hop)) if r5_multi_hop else 0.0,
    }


def compute_generation_metrics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute generation metrics from judge scores:
    - faithfulness: average binary score on answerable cases (excluding judge errors)
    - correctness: average binary score on answerable cases (excluding judge errors)
    - abstention_rate: on answerable: false cases, fraction where system abstained
    - hallucination_rate: on answerable: false cases, fraction where system answered anyway
    """
    answerable_cases = [c for c in cases if c.get("answerable", True)]
    unanswerable_cases = [c for c in cases if not c.get("answerable", True)]

    faithfulness_scores = [
        float(c["faithfulness"])
        for c in answerable_cases
        if c.get("faithfulness") is not None and not c.get("judge_error", False)
    ]
    correctness_scores = [
        float(c["correctness"])
        for c in answerable_cases
        if c.get("correctness") is not None and not c.get("judge_error", False)
    ]

    abstention_scores = [
        1.0 if c.get("abstained", False) else 0.0
        for c in unanswerable_cases
        if not c.get("judge_error", False)
    ]
    hallucination_scores = [
        1.0 if (not c.get("abstained", False)) else 0.0
        for c in unanswerable_cases
        if not c.get("judge_error", False)
    ]

    judge_errors = sum(1 for c in cases if c.get("judge_error", False))

    return {
        "faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
        "correctness": float(np.mean(correctness_scores)) if correctness_scores else 0.0,
        "abstention_rate": float(np.mean(abstention_scores)) if abstention_scores else 0.0,
        "hallucination_rate": float(np.mean(hallucination_scores)) if hallucination_scores else 0.0,
        "judge_errors": judge_errors,
    }


def compute_cost_metrics(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute cost and latency metrics:
    - mean_llm_calls: average LLM calls per query
    - p50_latency: 50th percentile (median) latency in seconds
    - p95_latency: 95th percentile latency in seconds
    """
    if not cases:
        return {
            "mean_llm_calls": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
        }

    llm_calls = [float(c.get("llm_calls", 0)) for c in cases]
    latencies = [float(c.get("latency", 0.0)) for c in cases]

    return {
        "mean_llm_calls": float(np.mean(llm_calls)) if llm_calls else 0.0,
        "p50_latency": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def compute_all_metrics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine retrieval, generation, and cost metrics into a single summary dict."""
    retrieval = compute_retrieval_metrics(cases)
    generation = compute_generation_metrics(cases)
    cost = compute_cost_metrics(cases)

    total_cases = len(cases)
    answerable_count = sum(1 for c in cases if c.get("answerable", True))
    unanswerable_count = total_cases - answerable_count

    return {
        "total_cases": total_cases,
        "answerable_cases": answerable_count,
        "unanswerable_cases": unanswerable_count,
        **retrieval,
        **generation,
        **cost,
    }


# ==================== ADVANCED TOP-VENUE RESEARCH METRICS ====================

import re
from collections import Counter
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _normalize_text(text: str) -> List[str]:
    """Normalize text into lowercase word tokens for academic lexical metrics."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def compute_token_f1(pred: str, gold: str) -> float:
    """
    Standard academic token-level F1 score (as in SQuAD, NarrativeQA, QASPER).
    Formula: 2 * (Precision * Recall) / (Precision + Recall)
    """
    pred_tokens = _normalize_text(pred)
    gold_tokens = _normalize_text(gold)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return float(2 * (precision * recall) / (precision + recall))


def compute_rouge_l(pred: str, gold: str) -> float:
    """
    Sentence-level ROUGE-L score based on Longest Common Subsequence (LCS).
    """
    pred_tokens = _normalize_text(pred)
    gold_tokens = _normalize_text(gold)
    m, n = len(pred_tokens), len(gold_tokens)
    if m == 0 or n == 0:
        return 1.0 if m == n else 0.0

    # Compute LCS length via dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == gold_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    r_lcs = lcs_len / n
    p_lcs = lcs_len / m
    beta = 1.2
    denom = r_lcs + (beta ** 2) * p_lcs
    return float(((1 + beta ** 2) * r_lcs * p_lcs) / denom) if denom > 0 else 0.0


def compute_bleu_4(pred: str, gold: str) -> float:
    """
    Cumulative 4-gram BLEU score with brevity penalty (Papineni et al., 2002).
    """
    pred_tokens = _normalize_text(pred)
    gold_tokens = _normalize_text(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Brevity penalty
    c = len(pred_tokens)
    r = len(gold_tokens)
    bp = 1.0 if c > r else np.exp(1 - r / max(c, 1e-9))

    precisions = []
    for n in range(1, 5):
        pred_ngrams = [tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)]
        gold_ngrams = [tuple(gold_tokens[i:i + n]) for i in range(len(gold_tokens) - n + 1)]
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        common = Counter(pred_ngrams) & Counter(gold_ngrams)
        overlap = sum(common.values())
        precisions.append((overlap + 1e-9) / (len(pred_ngrams) + 1e-9))

    geometric_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    return float(bp * geometric_mean)


def compute_bootstrap_ci(scores: List[float], 
                         n_bootstrap: int = 1000, 
                         ci: float = 0.95,
                         seed: int = 42) -> Dict[str, float]:
    """
    Compute non-parametric bootstrap confidence intervals (Piece 1 of Research Roadmap).
    Returns mean, lower bound, and upper bound at the specified CI (e.g. 95%).
    """
    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    arr = np.array(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_bootstrap)

    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0
    }


def compute_paired_significance(scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
    """
    Compute paired statistical hypothesis tests (Student's t-test and Wilcoxon signed-rank).
    Evaluates whether Model A significantly outperforms Model B at p < 0.05 and p < 0.01.
    """
    if len(scores_a) != len(scores_b) or len(scores_a) < 3:
        return {
            "error": "At least 3 paired samples required with identical lengths.",
            "is_significant_p05": False,
            "is_significant_p01": False
        }

    diffs = np.array(scores_a) - np.array(scores_b)
    t_stat, p_val_ttest = 0.0, 1.0
    w_stat, p_val_wilcoxon = 0.0, 1.0

    if SCIPY_AVAILABLE:
        try:
            t_res = stats.ttest_rel(scores_a, scores_b)
            t_stat = float(t_res.statistic)
            p_val_ttest = float(t_res.pvalue)
        except Exception:
            pass

        try:
            # If all differences are zero, Wilcoxon cannot be calculated
            if not np.all(diffs == 0):
                w_res = stats.wilcoxon(scores_a, scores_b)
                w_stat = float(w_res.statistic)
                p_val_wilcoxon = float(w_res.pvalue)
        except Exception:
            pass
    else:
        # Fallback pure-numpy paired t-test
        mean_d = np.mean(diffs)
        std_d = np.std(diffs, ddof=1)
        if std_d > 0:
            t_stat = float(mean_d / (std_d / np.sqrt(len(diffs))))
            # Normal distribution approximation for large N
            p_val_ttest = float(2 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(t_stat) / np.sqrt(2)))))

    return {
        "mean_diff": float(np.mean(diffs)),
        "t_statistic": t_stat,
        "p_value_ttest": p_val_ttest,
        "wilcoxon_statistic": w_stat,
        "p_value_wilcoxon": p_val_wilcoxon,
        "is_significant_p05": bool(p_val_ttest < 0.05),
        "is_significant_p01": bool(p_val_ttest < 0.01)
    }


def compute_claim_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """
    RAGAS-style claim verification and evidence grounding metric.
    Extracts individual factual sentences/clauses from the generated answer and checks
    lexical containment and content word support against the retrieved contexts.
    """
    if not answer or not context_chunks:
        return 0.0

    # Clean combined context
    corpus_tokens = set(_normalize_text(" ".join(context_chunks)))
    if not corpus_tokens:
        return 0.0

    # Split answer into sentence claims
    claims = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip().split()) >= 3]
    if not claims:
        return 0.0

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or", "that", "this", "it", "with"}
    grounded_claims = 0

    for claim in claims:
        claim_words = [w for w in _normalize_text(claim) if w not in stopwords]
        if not claim_words:
            grounded_claims += 1
            continue
        # Support test: at least 70% of non-trivial words in claim exist in context
        supported_count = sum(1 for w in claim_words if w in corpus_tokens)
        support_ratio = supported_count / len(claim_words)
        if support_ratio >= 0.70:
            grounded_claims += 1

    return float(grounded_claims / len(claims))


def compute_system_efficiency(cases: List[Dict[str, Any]], 
                              model_params_b: float = 1.0, 
                              cost_per_m_tokens: float = 0.15) -> Dict[str, Any]:
    """
    Systems & compute efficiency metrics (Piece 6 of Research Roadmap):
    - Context token consumption
    - Theoretical inference FLOPs per query (2 * params * tokens)
    - Projected API dollar cost per 1M queries
    """
    if not cases:
        return {
            "mean_context_tokens": 0.0,
            "mean_gflops_per_query": 0.0,
            "projected_cost_per_1m_queries": 0.0
        }

    token_counts = []
    for c in cases:
        # Check explicit token count or approximate from character length of retrieved chunks
        toks = c.get("context_tokens")
        if toks is None and "retrieved_chunks" in c:
            toks = sum(len(txt) // 4 for txt in c["retrieved_chunks"])
        elif toks is None and "chunks" in c:
            toks = sum(len(txt) // 4 for txt in c["chunks"])
        token_counts.append(float(toks or 0.0))

    mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    # GFLOPs estimate: 2 * (params in billions) * (tokens) * 10^9 / 10^9 = 2 * params_b * tokens
    mean_gflops = 2.0 * model_params_b * mean_tokens
    # Projected cost per 1M queries (in USD)
    projected_cost = (mean_tokens * 1_000_000 / 1_000_000) * cost_per_m_tokens

    return {
        "mean_context_tokens": mean_tokens,
        "mean_gflops_per_query": mean_gflops,
        "projected_cost_per_1m_queries": projected_cost
    }

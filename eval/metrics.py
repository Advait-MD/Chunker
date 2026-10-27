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

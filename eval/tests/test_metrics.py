"""
eval/tests/test_metrics.py — Unit tests for evaluation metrics.
Verifies recall@k, MRR, hop-splitting, and abstention/cost calculations on hand-computed examples.
"""

import pytest
from eval.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_retrieval_metrics,
    compute_generation_metrics,
    compute_cost_metrics,
    compute_all_metrics,
)


def test_recall_at_k_hand_computed():
    gold = ["c1", "c2"]
    retrieved = ["c3", "c1", "c4", "c2", "c5"]

    # k=1: ["c3"] -> 0 hits -> 0.0
    assert compute_recall_at_k(retrieved, gold, 1) == 0.0

    # k=2: ["c3", "c1"] -> 1 hit ("c1") -> 1/2 = 0.5
    assert compute_recall_at_k(retrieved, gold, 2) == 0.5

    # k=3: ["c3", "c1", "c4"] -> 1 hit -> 0.5
    assert compute_recall_at_k(retrieved, gold, 3) == 0.5

    # k=4: ["c3", "c1", "c4", "c2"] -> 2 hits -> 1.0
    assert compute_recall_at_k(retrieved, gold, 4) == 1.0

    # k=5: 2 hits -> 1.0
    assert compute_recall_at_k(retrieved, gold, 5) == 1.0


def test_recall_empty_gold():
    assert compute_recall_at_k(["c1", "c2"], [], 5) == 0.0
    assert compute_recall_at_k([], ["c1"], 5) == 0.0


def test_mrr_hand_computed():
    gold = ["c2", "c4"]

    # First gold is at rank 1
    assert compute_mrr(["c2", "c1", "c4"], gold) == 1.0

    # First gold is at rank 2
    assert compute_mrr(["c1", "c2", "c3"], gold) == 0.5

    # First gold is at rank 4
    assert compute_mrr(["c9", "c8", "c7", "c4"], gold) == 0.25

    # Gold not in retrieved
    assert compute_mrr(["c9", "c8", "c7"], gold) == 0.0

    # Empty retrieved or empty gold
    assert compute_mrr([], gold) == 0.0
    assert compute_mrr(["c1"], []) == 0.0


def test_retrieval_metrics_hop_splitting():
    cases = [
        # Single-hop: hops = 1
        {
            "answerable": True,
            "hops": 1,
            "gold_chunk_ids": ["c1"],
            "retrieved_ids": ["c1", "c2", "c3"],  # Recall@5 = 1.0, MRR = 1.0
        },
        {
            "answerable": True,
            "hops": 1,
            "gold_chunk_ids": ["c1"],
            "retrieved_ids": ["c9", "c8", "c7"],  # Recall@5 = 0.0, MRR = 0.0
        },
        # Multi-hop: hops = 2
        {
            "answerable": True,
            "hops": 2,
            "gold_chunk_ids": ["c1", "c2"],
            "retrieved_ids": ["c1", "c9"],  # Recall@5 = 0.5, MRR = 1.0
        },
        # Multi-hop: hops = 3
        {
            "answerable": True,
            "hops": 3,
            "gold_chunk_ids": ["c1", "c2"],
            "retrieved_ids": ["c1", "c2"],  # Recall@5 = 1.0, MRR = 1.0
        },
        # Unanswerable: should be excluded from retrieval metrics
        {
            "answerable": False,
            "hops": 1,
            "gold_chunk_ids": [],
            "retrieved_ids": ["c1", "c2"],
        },
    ]

    metrics = compute_retrieval_metrics(cases)

    # Single-hop average: (1.0 + 0.0) / 2 = 0.5
    assert metrics["recall@5_single_hop"] == pytest.approx(0.5)

    # Multi-hop average: (0.5 + 1.0) / 2 = 0.75
    assert metrics["recall@5_multi_hop"] == pytest.approx(0.75)

    # Overall Recall@5 on 4 answerable cases: (1.0 + 0.0 + 0.5 + 1.0) / 4 = 0.625
    assert metrics["recall@5"] == pytest.approx(0.625)

    # Overall MRR: (1.0 + 0.0 + 1.0 + 1.0) / 4 = 0.75
    assert metrics["mrr"] == pytest.approx(0.75)


def test_generation_and_cost_metrics():
    cases = [
        {
            "answerable": True,
            "faithfulness": True,
            "correctness": True,
            "llm_calls": 1,
            "latency": 2.0,
            "judge_error": False,
        },
        {
            "answerable": True,
            "faithfulness": True,
            "correctness": False,
            "llm_calls": 1,
            "latency": 4.0,
            "judge_error": False,
        },
        {
            "answerable": False,
            "abstained": True,
            "llm_calls": 0,
            "latency": 0.5,
            "judge_error": False,
        },
        {
            "answerable": False,
            "abstained": False,  # Hallucinated!
            "llm_calls": 1,
            "latency": 1.5,
            "judge_error": False,
        },
    ]

    gen = compute_generation_metrics(cases)
    assert gen["faithfulness"] == pytest.approx(1.0)
    assert gen["correctness"] == pytest.approx(0.5)
    assert gen["abstention_rate"] == pytest.approx(0.5)
    assert gen["hallucination_rate"] == pytest.approx(0.5)
    assert gen["judge_errors"] == 0

    cost = compute_cost_metrics(cases)
    assert cost["mean_llm_calls"] == pytest.approx(0.75)
    assert cost["p50_latency"] > 0
    assert cost["p95_latency"] >= cost["p50_latency"]


def test_academic_metrics_hand_computed():
    from eval.metrics import (
        compute_token_f1,
        compute_rouge_l,
        compute_bleu_4,
        compute_bootstrap_ci,
        compute_paired_significance,
        compute_claim_faithfulness,
        compute_system_efficiency,
    )

    # 1. Token F1
    # Overlap: "steve", "jobs", "founded", "apple" (4 tokens identical)
    pred = "Steve Jobs founded Apple Computer in California"
    gold = "Steve Jobs founded Apple"
    # pred len: 7, gold len: 4, overlap: 4 -> prec: 4/7, rec: 4/4=1 -> F1: 2*(4/7*1)/(4/7+1) = 8/11 = 0.72727
    f1 = compute_token_f1(pred, gold)
    assert f1 == pytest.approx(8 / 11)

    # Completely disjoint
    assert compute_token_f1("hello world", "foo bar") == 0.0
    # Exact match
    assert compute_token_f1("exact match text", "exact match text") == 1.0

    # 2. ROUGE-L
    rouge = compute_rouge_l("the quick brown fox", "the brown fox jumps")
    assert rouge > 0.5

    # 3. BLEU-4
    bleu = compute_bleu_4("the cat sat on the mat", "the cat sat on the mat")
    assert bleu == pytest.approx(1.0, abs=1e-3)

    # 4. Bootstrap CI
    sample_scores = [0.8, 0.82, 0.79, 0.85, 0.81, 0.83]
    ci = compute_bootstrap_ci(sample_scores, n_bootstrap=200)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
    assert ci["ci_lower"] > 0.75

    # 5. Paired Significance Testing
    model_a = [0.90, 0.92, 0.88, 0.95, 0.91, 0.94]
    model_b = [0.60, 0.62, 0.58, 0.65, 0.61, 0.64]
    sig = compute_paired_significance(model_a, model_b)
    assert sig["is_significant_p01"] is True
    assert sig["mean_diff"] > 0.25

    # 6. Claim Faithfulness
    context = ["Steve Jobs founded Apple in Cupertino with Steve Wozniak."]
    supported_ans = "Steve Jobs founded Apple."
    unsupported_ans = "Steve Jobs traveled to Mars on a rocket."
    assert compute_claim_faithfulness(supported_ans, context) == 1.0
    assert compute_claim_faithfulness(unsupported_ans, context) == 0.0

    # 7. System Efficiency
    case_dummies = [{"context_tokens": 400}, {"context_tokens": 600}]
    eff = compute_system_efficiency(case_dummies, model_params_b=1.0)
    assert eff["mean_context_tokens"] == 500.0
    assert eff["mean_gflops_per_query"] == 1000.0


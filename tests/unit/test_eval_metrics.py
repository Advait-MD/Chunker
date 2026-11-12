import pytest

from apex.eval import metrics as M


def test_squad_normalisation_and_citation_stripping():
    assert M.normalize_answer("The  Cat, sat! [2]") == "cat sat"
    assert M.exact_match("Paris [1].", ["paris"]) == 1.0
    assert M.exact_match("Paris is nice", ["Paris"]) == 0.0


def test_token_f1_max_over_golds():
    assert M.token_f1("Miquette Giraudy [3]", ["Steve Hillage", "Miquette Giraudy"]) == 1.0
    f = M.token_f1("Giraudy was his wife", ["Miquette Giraudy"])
    assert 0 < f < 1


def test_answer_contains_whole_tokens_only():
    assert M.answer_contains("He married Miquette Giraudy in 1975.", ["Miquette Giraudy"]) == 1.0
    assert M.answer_contains("The value was 1975.", ["75"]) == 0.0


@pytest.mark.parametrize("text,idx", [("(B) They treat him well [1].", 1), ("C. Because...", 2),
                                      ("The answer is D", 3), ("Insufficient evidence", None)])
def test_parse_choice(text, idx):
    assert M.parse_choice(text) == idx


def test_covers_handles_split_and_merged_chunks():
    para = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi"
    assert M.covers("intro words " + para + " outro words", para)        # chunk merges paragraphs
    assert M.covers(" ".join(para.split()[:9]), para)                    # chunk is half the paragraph
    assert not M.covers("completely unrelated text about something else entirely here", para)


def test_retrieval_metrics():
    hits = [[], [0], [0, 1], []]  # rank 2 covers gold 0, rank 3 covers golds 0 and 1
    assert M.recall_at_k(hits, 2, 1) == 0.0
    assert M.recall_at_k(hits, 2, 2) == 0.5
    assert M.recall_at_k(hits, 2, 3) == 1.0
    assert M.mrr(hits) == 0.5
    import math
    ideal = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(hits, 2, 10) == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / ideal)
    with pytest.raises(ValueError):
        M.recall_at_k(hits, 0, 5)


def test_union_recall_credits_gold_split_across_chunks():
    g = [f"g{i}" for i in range(20)]               # 16 gold 5-gram shingles
    filler = lambda p: [f"{p}{i}" for i in range(40)]  # noqa: E731
    halves = [" ".join(filler("a") + g[:10]), " ".join(g[10:] + filler("b"))]  # each holds 6/16 gold shingles
    gold = " ".join(g)
    hits = M.evidence_hits(halves, [gold])
    assert hits == [[], []]                                   # neither chunk alone reaches 50%
    assert M.union_recall_at_k(halves, [gold], 1, hits) == 0.0
    assert M.union_recall_at_k(halves, [gold], 2, hits) == 1.0  # together they cover 12/16


def test_abstention_scores():
    s = M.abstention_scores([True, True, False, False], [False, True, False, True])
    assert s["abstain_precision"] == 0.5 and s["abstain_recall"] == 0.5 and s["false_abstain_rate"] == 0.5


def test_bootstrap_and_paired_test():
    ci = M.bootstrap_ci([0, 1] * 50)
    assert ci["lo"] < 0.5 < ci["hi"] and ci["n"] == 100
    t = M.paired_test([1.0] * 30 + [0.0] * 10, [0.0] * 40)
    assert t["mean_diff"] == 0.75 and t["p_bootstrap"] < 0.01 and t["p_wilcoxon"] < 0.01
    assert M.paired_test([0.3, 0.4], [0.3, 0.4])["p_wilcoxon"] == 1.0
    with pytest.raises(ValueError):
        M.paired_test([1, 2], [1])

"""BM25, FAISS IVF and rank fusion (M3), offline."""

import numpy as np
import pytest

from apex.index.bm25 import BM25Index, tokenize
from apex.index.ivf import IVFIndex, choose_nlist
from apex.retrieve import fusion

TEXTS = {
    "a": "The capital of Freedonia is Fredville, on the river Marx.",
    "b": "Sylvania exports timber; its ambassador is Trentino.",
    "c": "Error code E1234 means the Fredville turbine overheated.",
    "d": "A long poem about rivers, mountains and the sea.",
    "e": "Brass instruments are played in the royal orchestra.",
    "f": "Duck soup is a traditional dish served in winter.",
}


def test_tokenize_drops_stopwords_keeps_codes():
    assert tokenize("What is the E1234 code of the Fredville turbine?") == ["e1234", "code", "fredville", "turbine"]


def test_bm25_exact_term_and_scope():
    idx = BM25Index(list(TEXTS), list(TEXTS.values()))
    assert idx.search("E1234", 3)[0][0] == "c"  # an exact code: BM25's reason to exist
    assert {i for i, _ in idx.search("Fredville", 4)} == {"a", "c"}
    only_a = idx.search("Fredville", 4, allowed=idx.positions(["a", "b"]))
    assert [i for i, _ in only_a] == ["a"]  # scoped: c is excluded, zero scores dropped
    assert idx.search("the of is", 3) == []  # only stopwords


def test_choose_nlist_caps_by_training_points():
    assert choose_nlist(4896) == 4896 // 39 == 125  # 4·√N = 280 would need 10,920 training vectors
    assert choose_nlist(1_000_000) == 4000  # large corpora get the spec's 4·√N
    assert choose_nlist(10) == 1


def _vecs(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_ivf_exact_and_full_probe_agree_and_is_deterministic():
    v = _vecs()
    ids = [f"n{i}" for i in range(len(v))]
    a, b = IVFIndex(ids, v, seed=7), IVFIndex(ids, v, seed=7)
    assert a.nlist == choose_nlist(len(v)) == 10
    q = v[3]
    exact = a.search(q, 10, 0, exact=True)
    assert exact[0][0] == "n3" and abs(exact[0][1] - 1.0) < 1e-5  # normalised: self-similarity 1
    assert [i for i, _ in a.search(q, 10, nprobe=a.nlist)] == [i for i, _ in exact]  # probing all cells is exact
    assert a.search(q, 10, nprobe=2) == b.search(q, 10, nprobe=2)  # same seed, same index


def test_ivf_scoped_search_is_exact_over_subset():
    v = _vecs()
    ids = [f"n{i}" for i in range(len(v))]
    idx = IVFIndex(ids, v, seed=1)
    allowed = np.arange(100, 150)
    res = idx.search(v[120], 5, nprobe=1, allowed=allowed)
    assert res[0][0] == "n120" and all(100 <= int(i[1:]) < 150 for i, _ in res)


def test_rrf_scores_and_ties():
    lists = {"dense": [("a", 0.9), ("b", 0.8)], "bm25": [("b", 7.0), ("c", 3.0)]}
    out = dict(fusion.rrf(lists, k=60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["a"] == pytest.approx(1 / 61) and out["c"] == pytest.approx(1 / 62)
    assert [i for i, _ in fusion.rrf(lists)] == ["b", "a", "c"]
    assert [i for i, _ in fusion.rrf({"x": [("z", 1)], "y": [("y", 1)]})] == ["y", "z"]  # tie → id order


def test_weighted_fusion_minmax():
    lists = {"dense": [("a", 0.9), ("b", 0.5)], "bm25": [("b", 10.0), ("c", 2.0)]}
    out = dict(fusion.weighted(lists, {"dense": 0.5, "bm25": 0.5}))
    assert out == {"a": 0.5, "b": 0.5, "c": 0.0}
    with pytest.raises(ValueError):
        fusion.weighted(lists, {"dense": 1.0})

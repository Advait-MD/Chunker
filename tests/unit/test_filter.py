"""M5 evidence filter stages, sufficiency, query rewrite and citation verifier, with fake embedder / NLI / LLM."""

import numpy as np
import pytest

from apex import config as C
from apex.filter.evidence import EvidenceFilter, dedup, jaccard, join_kept, relevance_threshold, select_sentences, shingles
from apex.filter.sufficiency import SufficiencyChecker
from apex.generate.generator import EvidenceItem, EvidenceSet, build_user_prompt
from apex.generate.verifier import CitationVerifier, score_sentences, strip_citations
from apex.index.chroma_store import Hit, Node


def hit(i, text, score, level=0):
    return Hit(node=Node(id=f"n{i}", text=text, level=level, doc_id="d", source_path="s.md", token_count=len(text.split())),
               score=score, rank=i)


def cfg(**filter_over):
    c = C.load_config(None)
    c["filter"] = C.deep_merge(c["filter"], filter_over)
    return c


class FakeEmbedder:
    """Query = unit vector e0; a sentence containing 'gold' embeds near e0, any other far from it."""

    def __init__(self):
        self.doc_calls = []

    def embed_query(self, text, caller=""):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_documents(self, texts, caller=""):
        self.doc_calls.append(list(texts))
        return np.array([[0.9, 0.436] if "gold" in t else [0.1, 0.995] for t in texts], dtype=np.float32)


class FakeNLI:
    """Entailed iff every word of the hypothesis occurs in the premise; contradiction if the premise says 'not'."""

    def __init__(self):
        self.pairs = []

    def probs_many(self, pairs):
        self.pairs += list(pairs)
        out = []
        for p, h in pairs:
            if " not " in f" {p} " and " not " not in f" {h} ":
                out.append({"entailment": 0.01, "neutral": 0.04, "contradiction": 0.95})
                continue
            ok = all(w in p.lower() for w in h.lower().rstrip(".").split())
            out.append({"entailment": 0.97 if ok else 0.1, "neutral": 0.02 if ok else 0.8, "contradiction": 0.01 if ok else 0.1})
        return out


# ---- stage 1: relevance threshold -------------------------------------------------
def test_threshold_drops_low_scores_but_keeps_min_keep():
    hs = [hit(1, "a", 3.0), hit(2, "b", -1.0), hit(3, "c", -6.0), hit(4, "d", -7.0)]
    kept, dropped = relevance_threshold(hs, threshold=-2.0, min_keep=2)
    assert [h.node.id for h in kept] == ["n1", "n2"] and [h.node.id for h in dropped] == ["n3", "n4"]
    kept, _ = relevance_threshold(hs, threshold=5.0, min_keep=2)  # nothing passes: top 2 by score survive
    assert [h.node.id for h in kept] == ["n1", "n2"]
    kept, dropped = relevance_threshold(hs, threshold=None, min_keep=2)  # stage off
    assert len(kept) == 4 and not dropped


# ---- stage 2: dedup --------------------------------------------------------------
def test_dedup_prefers_leaf_over_summary_and_keeps_order():
    t = "the quick brown fox jumps over the lazy dog near the river bank today"
    hs = [hit(1, t, 5.0, level=1), hit(2, "something else entirely about cats and dogs here", 4.0), hit(3, t + " again", 1.0)]
    kept, dropped = dedup(hs, threshold=0.8, n=5)
    assert [h.node.id for h in kept] == ["n2", "n3"]  # the higher-scored summary loses to the leaf copy
    assert dropped[0][:2] == ("n1", "n3") and dropped[0][2] > 0.8


def test_shingle_jaccard():
    a, b = shingles("one two three four five six", 5), shingles("one two three four five seven", 5)
    assert jaccard(a, a) == 1.0 and 0 < jaccard(a, b) < 1 and jaccard(a, set()) == 0.0
    assert shingles("too short", 5) == {("too", "short")}


# ---- stage 3: sentence pruning -----------------------------------------------------
def test_select_sentences_neighbors_and_fallback():
    assert select_sentences([0.1, 0.2, 0.9, 0.1, 0.1], 0.5, 1) == [1, 2, 3]
    assert select_sentences([0.9, 0.1, 0.1, 0.1, 0.8], 0.5, 0) == [0, 4]
    assert select_sentences([0.1, 0.3, 0.2], 0.5, 1) == [0, 1, 2]  # none pass: best (1) and its neighbours
    assert select_sentences([], 0.5, 1) == []
    assert join_kept(["A.", "B.", "C.", "D."], [0, 2, 3]) == "A. … C. D."


def test_filter_prunes_to_relevant_sentences_and_packs():
    c = cfg(prune={"enabled": True, "threshold": 0.5, "neighbors": 0}, rel_threshold=-3.0)
    emb = FakeEmbedder()
    f = EvidenceFilter(c, embedder=emb)
    hs = [hit(1, "Intro text here. The gold fact is 42. Unrelated tail.", 2.0),
          hit(2, "Only one sentence without the key", -1.0),
          hit(3, "Noise one. Noise two.", -5.0)]
    r = f.run("what is the fact?", hs)
    assert [it.node_id for it in r.evidence.items] == ["n1", "n2"]          # n3 below θ_rel
    assert r.evidence.items[0].text == "The gold fact is 42."                # pruned to the passing sentence
    assert r.evidence.items[1].text == "Only one sentence without the key"   # one-sentence passages stay whole
    assert [it.cite_id for it in r.evidence.items] == [1, 2]
    assert r.stats["sentences_in"] == 4 and r.stats["sentences_kept"] == 2
    assert r.stats["tokens_out"] < r.stats["tokens_in"]
    assert emb.doc_calls == [["Intro text here.", "The gold fact is 42.", "Unrelated tail."]]


def test_filter_respects_budget_and_flags_conflicts():
    c = cfg(prune={"enabled": False, "threshold": 0.5, "neighbors": 1}, conflicts={"enabled": True, "threshold": 0.9})
    c["generation"]["context_token_budget"] = 10
    f = EvidenceFilter(c, nli=FakeNLI())
    hs = [hit(1, "the river is long", 3.0), hit(2, "the river is not long", 2.0), hit(3, "a b c d e f g h i", 1.0)]
    r = f.run("q", hs)
    assert [it.node_id for it in r.evidence.items] == ["n1", "n2"]  # 4 + 5 tokens; n3 (9) would pass the 10-token budget
    assert r.evidence.conflicts and r.evidence.conflicts[0]["cite_ids"] == [1, 2]


def test_filter_config_errors_are_loud():
    with pytest.raises(ValueError, match="embedder"):
        EvidenceFilter(cfg(prune={"enabled": True}))
    with pytest.raises(ValueError, match="NLI"):
        EvidenceFilter(cfg(prune={"enabled": False}, conflicts={"enabled": True}))


# ---- stage 5: sufficiency ----------------------------------------------------------
class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        reply = self.reply

        class R:
            call_id = "c1"

            @staticmethod
            def json():
                return reply
        return R()


def ev(*scores):
    return EvidenceSet(items=[EvidenceItem(cite_id=i + 1, node_id=f"n{i}", text=f"t{i}", score=s, source="s")
                              for i, s in enumerate(scores)])


def test_sufficiency_bands_and_llm_tiebreak():
    c = cfg(sufficiency={"sufficient_above": 0.0, "insufficient_below": -4.0, "ambiguous": "llm"})
    cl = FakeClient({"sufficient": False})
    s = SufficiencyChecker(c, cl)
    assert s.check("q", ev(1.5, -9))["verdict"] == "sufficient"
    assert s.check("q", ev(-5.0, -9))["verdict"] == "insufficient"
    assert s.check("q", EvidenceSet())["verdict"] == "insufficient"
    assert not cl.calls
    r = s.check("q", ev(-2.0))
    assert r == {"verdict": "insufficient", "by": "llm", "best": -2.0, "call_id": "c1"} and len(cl.calls) == 1
    c2 = cfg(sufficiency={"ambiguous": "sufficient"})
    assert SufficiencyChecker(c2, cl).check("q", ev(-2.0))["verdict"] == "sufficient"
    with pytest.raises(ValueError):
        SufficiencyChecker(cfg(sufficiency={"sufficient_above": -5.0, "insufficient_below": 0.0}), cl)


# ---- citation verifier --------------------------------------------------------------
def test_verifier_scores_each_sentence_against_its_citations():
    nli = FakeNLI()
    texts = {1: "Paris is the capital of France", 2: "France is in Europe"}
    rows = score_sentences(nli, "Paris is the capital [1]. Paris is in Europe [1][2]. It is big.", texts, 0.5)
    # sentence 2 needs both passages: neither alone entails it, the two joined do
    assert [r["supported"] for r in rows] == [True, True, False]
    assert rows[2]["cites"] == [] and rows[2]["entail"] is None          # uncited
    assert rows[1]["cites"] == [1, 2]
    assert nli.probs_many([(texts[1], "Paris is in Europe")])[0]["entailment"] < 0.5
    assert ("Paris is the capital of France\nFrance is in Europe", "Paris is in Europe.") in nli.pairs
    assert strip_citations("Paris is big [1, 2].") == "Paris is big."


def test_verifier_remove_keeps_at_least_one_sentence():
    c = C.load_config(None)
    v = CitationVerifier(c, FakeNLI())
    e = ev(1.0, 1.0)
    e.items[0].text, e.items[1].text = "the sky is blue", "grass is green"
    rep = v.verify("The sky is blue [1]. Grass is red [2].", e)
    assert rep["n_supported"] == 1 and rep["unsupported"] == ["Grass is red [2]."]
    assert v.remove_unsupported("x", rep) == "The sky is blue [1]."
    rep2 = v.verify("Grass is red [2].", e)
    assert v.remove_unsupported("x", rep2) is None
    assert "Grass is red" in v.feedback(rep)


def test_feedback_prompt_is_identical_without_feedback():
    e = ev(1.0)
    assert build_user_prompt("q", e) == build_user_prompt("q", e, "")
    assert "NOTE" in build_user_prompt("q", e, "NOTE").split("Question: q")[1]

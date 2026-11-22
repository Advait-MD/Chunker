"""M5 end to end, offline: hybrid + (fake) rerank → evidence filter → sufficiency (rewrite, retry, abstain)
→ generation → (fake) NLI citation verifier, with the trace stages and the answer contract."""

import json

import pytest
import torch

from apex import config as C
from apex.ingest.pipeline import ingest_path
from apex.llm import OllamaClient
from apex.pipeline import Components, NaivePipeline
from apex.retrieve.rerank import CrossEncoderReranker
from tests.conftest import FakeOllama

DOCS = {
    "codes.txt": "Turbine error E1234 means overheating. The fan speed is then reduced. Error E9 means the door is open.",
    "freedonia.md": "# Freedonia\n\nThe capital of Freedonia is Fredville. Fredville sits on the river Marx.\n",
    "sylvania.txt": "Sylvania is a neighbouring kingdom. Its ambassador is Trentino. Sylvania exports timber.",
}


class _Tok:
    def __call__(self, qs, ps, **kw):
        return {"lengths": torch.tensor([float(sum(w in p.lower() for w in q.lower().split())) for q, p in zip(qs, ps)])}


class _Model:
    def __call__(self, lengths):
        return type("Out", (), {"logits": lengths.view(-1, 1)})()


class FakeNLI:
    """Entailed iff all content words of the hypothesis occur in the premise."""

    def probs_many(self, pairs):
        out = []
        for p, h in pairs:
            ok = all(w in p.lower() for w in h.lower().rstrip(".").split() if len(w) > 2)
            out.append({"entailment": 0.95 if ok else 0.05, "neutral": 0.04 if ok else 0.9, "contradiction": 0.01 if ok else 0.05})
        return out


def answer_fn(payload):
    sys_prompt = payload.get("system", "")
    if "rewrite a question" in sys_prompt:
        return {"query": "turbine error code meaning"}
    if "You check whether numbered evidence passages" in sys_prompt:
        return {"sufficient": True}
    if "not supported by the passages" in payload["prompt"]:  # the verifier's regeneration
        return {"answer": "Error E1234 means overheating [1].", "short_answer": "overheating", "citations": [1],
                "confidence": 0.9, "abstained": False}
    return {"answer": "Error E1234 means overheating [1]. It also melts the moon [1].", "short_answer": "overheating",
            "citations": [1], "confidence": 0.9, "abstained": False}


@pytest.fixture
def m5_cfg(cfg, tmp_path, monkeypatch):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(CrossEncoderReranker, "_load", lambda self: setattr(self, "_tok", _Tok()) or setattr(self, "_model", _Model()))
    monkeypatch.setattr("apex.pipeline.shared_nli", lambda cfg: FakeNLI())
    cfg = C.deep_merge(cfg, {
        "retrieval": {"mode": "hybrid", "top_k": 3},
        "rerank": {"enabled": True, "cache_path": str(tmp_path / "rerank.sqlite")},
        "filter": {"enabled": True, "pool_k": 3, "rel_threshold": 1.0,
                   "prune": {"enabled": True, "threshold": 0.3, "neighbors": 0},
                   "sufficiency": {"sufficient_above": 2.0, "insufficient_below": 1.0, "ambiguous": "sufficient"}},
        "verifier": {"enabled": True, "action": "regenerate"},
    })
    cfg["_corpus"] = d
    return cfg


def _pipe(cfg, fake):
    comps = Components(cfg, client=OllamaClient(cfg, transport=fake))
    ingest_path(cfg["_corpus"], cfg, comps.store, comps.embedder, log=lambda *_: None)
    return NaivePipeline(comps)


def test_filtered_answer_is_pruned_verified_and_regenerated(m5_cfg):
    fake = FakeOllama(answer_fn)
    pipe = _pipe(m5_cfg, fake)
    assert pipe.name == "apex_m5"
    res = pipe.answer("what does turbine error E1234 mean")
    assert not res["abstained"] and res["answer"] == "Error E1234 means overheating [1]."
    ev = res["evidence"]
    assert ev[0]["source"] == "codes.txt" and "E9" not in ev[0]["text"]         # pruned to the E1234 sentence
    v = res["verification"]
    assert v["first"] == {"n_sentences": 2, "n_supported": 1} and v["regenerated"] and v["regeneration_kept"]
    assert v["final"] == {"n_sentences": 1, "n_supported": 1}
    assert res["filter"]["sufficiency"][0]["verdict"] == "sufficient" and res["filter"]["rewritten_query"] is None
    tr = json.load(open(res["trace_path"], encoding="utf-8"))
    assert {"filter", "sufficiency", "generate", "verify"} <= set(tr["stages"])
    assert "VERIFYING" in [s["state"] if isinstance(s, dict) else s for s in tr["states"]]
    assert tr["stages"]["filter"]["tokens_out"] < tr["stages"]["filter"]["tokens_in"]
    callers = [c["caller"] for c in tr["llm_calls"] if c["kind"] == "generate"]
    assert callers == ["generate.answer", "generate.regenerate"]
    assert res["totals"]["generator_prompt_tokens"] == 120


def test_insufficient_evidence_rewrites_once_then_abstains(m5_cfg):
    cfg = C.deep_merge(m5_cfg, {"filter": {"sufficiency": {"sufficient_above": 100.0, "insufficient_below": 99.0}}})
    fake = FakeOllama(answer_fn)
    res = _pipe(cfg, fake).answer("what does turbine error E1234 mean")
    assert res["abstained"] and res["short_answer"] == "Unanswerable" and res["filter"]["abstained_by"] == "sufficiency"
    assert res["filter"]["rewritten_query"] == "turbine error code meaning"
    assert [s["verdict"] for s in res["filter"]["sufficiency"]] == ["insufficient", "insufficient"]
    gens = [c for c in fake.calls if c["endpoint"] == "/api/generate"]
    assert len(gens) == 1 and "rewrite a question" in gens[0]["payload"]["system"]  # no answer call after abstaining


def test_ambiguous_band_asks_the_llm(m5_cfg):
    cfg = C.deep_merge(m5_cfg, {"filter": {"sufficiency": {"sufficient_above": 100.0, "insufficient_below": 0.0,
                                                           "ambiguous": "llm"}}, "verifier": {"action": "flag"}})
    fake = FakeOllama(answer_fn)
    res = _pipe(cfg, fake).answer("what does turbine error E1234 mean")
    assert res["filter"]["sufficiency"][0]["by"] == "llm" and not res["abstained"]
    assert res["verification"]["regenerated"] is False and "melts the moon" in res["answer"]  # flag only


def test_filter_without_reranker_is_rejected(m5_cfg, fake):
    cfg = C.deep_merge(m5_cfg, {"rerank": {"enabled": False}})
    with pytest.raises(ValueError, match="rerank"):
        _pipe(cfg, fake)

"""Hybrid retrieval end-to-end (M3): ingest → BM25 ‖ IVF → RRF → (fake) rerank → cited answer, offline."""

import json

import pytest
import torch

from apex import config as C
from apex.ingest.pipeline import ingest_path
from apex.llm import OllamaClient
from apex.pipeline import Components, NaivePipeline
from apex.retrieve.rerank import CrossEncoderReranker, RerankerUnavailable

DOCS = {
    "freedonia.md": "# Freedonia\n\nThe capital of Freedonia is Fredville. Fredville sits on the river Marx.\n\n"
                    "## Economy\n\nFreedonia exports duck soup and brass instruments.\n",
    "sylvania.txt": "Sylvania is a neighbouring kingdom. Its ambassador is Trentino. Sylvania exports timber.",
    "codes.txt": "Turbine error E1234 means overheating. Error E9 means the door is open.",
}


@pytest.fixture
def hybrid_cfg(cfg, tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    cfg = C.deep_merge(cfg, {"retrieval": {"mode": "hybrid", "top_k": 2},
                             "rerank": {"cache_path": str(tmp_path / "rerank.sqlite")}})
    cfg["_corpus"] = d
    return cfg


def _comps(cfg, fake):
    comps = Components(cfg, client=OllamaClient(cfg, transport=fake))
    ingest_path(cfg["_corpus"], cfg, comps.store, comps.embedder, log=lambda *_: None)
    return comps


class _FakeTok:
    def __call__(self, qs, ps, **kw):
        return {"lengths": torch.tensor([float(sum(w in p.lower() for w in q.lower().split())) for q, p in zip(qs, ps)])}


class _FakeModel:
    calls = 0

    def __call__(self, lengths):
        _FakeModel.calls += 1
        return type("Out", (), {"logits": lengths.view(-1, 1)})()


def test_hybrid_pipeline_answers_with_trace(hybrid_cfg, fake):
    comps = _comps(hybrid_cfg, fake)
    pipe = NaivePipeline(comps)
    assert pipe.retriever_name == "bm25+dense_ivf|rrf"
    res = pipe.answer("What does error E1234 mean?")
    assert res["evidence"][0]["source"] == "codes.txt" and len(res["evidence"]) <= 2
    tr = json.load(open(res["trace_path"], encoding="utf-8"))["stages"]["retrieve"]
    assert tr["retriever"] == "bm25+dense_ivf|rrf" and tr["n_candidates"]["bm25"] >= 1
    assert {"fused", "dense_rank", "bm25_rank"} <= set(tr["hits"][0])
    deep = pipe.retrieve_ranked("What does error E1234 mean?", k=5)  # metrics depth reuses the memo
    assert [h.node.id for h in deep[:2]] == [e["node_id"] for e in res["evidence"]]
    with pytest.raises(ValueError):
        pipe.retrieve_ranked("x", k=hybrid_cfg["retrieval"]["fused_k"] + 1)


def test_scoped_hybrid_only_returns_that_source(hybrid_cfg, fake):
    pipe = NaivePipeline(_comps(hybrid_cfg, fake))
    hits = pipe.retrieve_ranked("Fredville capital exports", k=5, source="sylvania.txt")
    assert hits and {h.node.source_path for h in hits} == {"sylvania.txt"}
    with pytest.raises(ValueError):
        pipe.retrieve_ranked("anything", k=5, source="missing.txt")


def test_rerank_reorders_and_caches(hybrid_cfg, fake, monkeypatch):
    cfg = C.deep_merge(hybrid_cfg, {"rerank": {"enabled": True}})

    def fake_load(self):
        self._tok, self._model = _FakeTok(), _FakeModel()
    monkeypatch.setattr(CrossEncoderReranker, "_load", fake_load)
    pipe = NaivePipeline(_comps(cfg, fake))
    assert pipe.retriever_name.endswith("|rerank")
    hits = pipe.retriever.retrieve("turbine error overheating door", k=3)
    assert hits[0].node.source_path == "codes.txt" and "rerank" in hits[0].detail and hits[0].detail["fused_rank"]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    n = _FakeModel.calls
    pipe.retriever._memo.clear()
    pipe.retriever.retrieve("turbine error overheating door", k=3)
    assert _FakeModel.calls == n and pipe.retriever.last["rerank"]["n_computed"] == 0  # served from the SQLite cache


def test_missing_reranker_is_a_loud_error(hybrid_cfg, tmp_path):
    cfg = C.deep_merge(hybrid_cfg, {"rerank": {"enabled": True, "model": "apex-test/does-not-exist", "revision": "0" * 40}})
    with pytest.raises(RerankerUnavailable):
        CrossEncoderReranker(cfg).score("q", ["p"])


def test_dense_mode_rejects_rerank(hybrid_cfg, fake):
    cfg = C.deep_merge(hybrid_cfg, {"retrieval": {"mode": "dense"}, "rerank": {"enabled": True}})
    with pytest.raises(ValueError):
        NaivePipeline(Components(cfg, client=OllamaClient(cfg, transport=fake)))

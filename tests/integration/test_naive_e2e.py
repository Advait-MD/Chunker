"""End-to-end naive pipeline: ingest → dense retrieve → cited generation → trace, offline."""

import json

import pytest

from apex.ingest.pipeline import ingest_path
from apex.llm import OllamaClient
from apex.pipeline import Components, NaivePipeline
from tests.conftest import FakeOllama

DOCS = {
    "freedonia.md": "# Freedonia\n\nThe capital of Freedonia is Fredville. Fredville sits on the river Marx.\n\n"
                    "## Economy\n\nFreedonia exports duck soup and brass instruments.\n",
    "sylvania.txt": "Sylvania is a neighbouring kingdom. Its ambassador is Trentino. Sylvania exports timber.",
}


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


def _pipeline(cfg, transport):
    comps = Components(cfg, client=OllamaClient(cfg, transport=transport))
    return comps, NaivePipeline(comps)


def test_ingest_query_cite_trace(cfg, corpus, fake):
    comps, pipe = _pipeline(cfg, fake)
    rep = ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    assert rep["added_nodes"] == comps.store.count() >= 3

    res = pipe.answer("What is the capital of Freedonia?")
    assert not res["abstained"] and res["answer"]
    assert res["citations"], "answer must carry citations"
    stored = comps.store.get([c["node_id"] for c in res["citations"]])
    for c in res["citations"]:
        assert c["node_id"] in stored  # every citation resolves to a stored node
        assert c["source"] == "freedonia.md" and c["char_span"]
    assert "Fredville" in res["evidence"][0]["text"]

    # trace contract
    trace = json.load(open(res["trace_path"], encoding="utf-8"))
    assert [s["state"] for s in trace["states"]] == ["RECEIVED", "RETRIEVING", "FILTERING", "GENERATING", "DONE"]
    assert trace["route"] == "SINGLE_HOP" and trace["retrieved_ids"] and trace["filtered_ids"]
    assert trace["totals"]["llm_calls"] == 1 and trace["totals"]["prompt_tokens"] == 120
    for stage in ("retrieve", "pack", "generate"):
        assert "latency_ms" in trace["stages"][stage]

    # the generator saw numbered evidence and the JSON schema
    gen_payload = [c for c in fake.calls if c["endpoint"] == "/api/generate"][0]["payload"]
    assert "[1] (source: freedonia.md" in gen_payload["prompt"] and gen_payload["format"]["required"]


def test_reingest_unchanged_reuses_cache(cfg, corpus, fake):
    comps, _ = _pipeline(cfg, fake)
    ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    n_embed = fake.count("/api/embed")
    rep = ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    assert rep["added_nodes"] == 0 and rep["skipped_files"] == len(DOCS)
    assert fake.count("/api/embed") == n_embed


def test_changed_file_replaces_old_nodes(cfg, corpus, fake):
    comps, _ = _pipeline(cfg, fake)
    ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    (corpus / "sylvania.txt").write_text("Sylvania is gone.", encoding="utf-8")
    ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    syl = comps.store.col.get(where={"source_path": "sylvania.txt"}, include=["documents"])
    assert syl["documents"] == ["Sylvania is gone."]  # no stale nodes from the old version
    assert comps.store.count(where={"source_path": "freedonia.md"}) >= 2


def test_model_abstention_passes_through(cfg, corpus):
    fake = FakeOllama(lambda p: {"answer": "Insufficient evidence in the corpus to answer this question.",
                                 "citations": [], "confidence": 0.1, "abstained": True})
    comps, pipe = _pipeline(cfg, fake)
    ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    res = pipe.answer("Who won the 1998 World Cup?")
    assert res["abstained"] and res["citations"] == []
    assert json.load(open(res["trace_path"], encoding="utf-8"))["states"][-1]["state"] == "ABSTAIN"


def test_invalid_citation_reported(cfg, corpus):
    fake = FakeOllama(lambda p: {"answer": "Fredville [9].", "citations": [9], "confidence": 0.5, "abstained": False})
    comps, pipe = _pipeline(cfg, fake)
    ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    res = pipe.answer("capital of Freedonia")
    assert res["invalid_citations"] == [9] and res["citations"] == []
    assert res["uncited_sentences"] == ["Fredville [9]."]


def test_fastapi_query_endpoint(cfg, corpus, fake):
    from fastapi.testclient import TestClient
    from apex.api import server

    comps, pipe = _pipeline(cfg, fake)
    ingest_path(corpus, cfg, comps.store, comps.embedder, log=lambda *_: None)
    server.set_pipeline(pipe)
    try:
        r = TestClient(server.app).post("/query", json={"query": "What is the capital of Freedonia?"})
    finally:
        server.set_pipeline(None)
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] and body["citations"][0]["source"] == "freedonia.md"
    assert body["citations"][0]["page"] == 1

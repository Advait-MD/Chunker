"""End-to-end eval harness on a synthetic pooled dataset, offline (fake Ollama for generator and judge)."""

import json

import pytest

from apex.eval import datasets as D
from apex.eval.judge import SCHEMA, Judge, quote_in
from apex.ingest.parsers import parse_jsonl_doc
from apex.llm import OllamaClient

from tests.conftest import FakeOllama

ITEMS = [
    ("toy:q1", "What is the capital of Freedonia?", ["Fredville"], [("Freedonia", "Freedonia is a country. Its capital city is Fredville, founded in 1901 by settlers."),
                                                                    ("Sylvania", "Sylvania is a neighbour of Freedonia known for its long rivers and mountains.")], [0]),
    ("toy:q2", "Who founded the Fredville observatory?", ["Ada Quill"], [("Fredville Observatory", "The Fredville observatory was founded by the astronomer Ada Quill in 1920."),
                                                                         ("Marsh Hall", "Marsh Hall is a concert venue on the eastern bank of the Fredville river.")], [0]),
    ("toy:q3", "What colour is the Sylvanian flag?", ["green"], [("Sylvanian flag", "The flag of Sylvania is green with a white star placed at its centre."),
                                                                 ("Anthem", "The Sylvanian anthem was written in the nineteenth century by a poet.")], [0]),
    ("toy:q4", "When did Marsh Hall open?", ["1955"], [("Marsh Hall opening", "Marsh Hall opened to the public in 1955 after six years of building work."),
                                                       ("Bridges", "Several stone bridges cross the Fredville river near the old town.")], [0]),
]


def _toy_loader(_d):
    return D._pooled([(q, text, ans, paras, sup, {"hops": 1 if i % 2 else 2})
                      for i, (q, text, ans, paras, sup) in enumerate(ITEMS)], "toy")


def _fake_answer(payload):
    if payload.get("format") == SCHEMA:
        import re
        prompt = payload["prompt"]
        ids = sorted({int(x) for x in re.findall(r"^\[(\d+)\] ", prompt, re.M)})
        sents = re.findall(r"^S(\d+): .*\n    cites: (.*)$", prompt, re.M)
        quote = "Its capital city is Fredville"  # genuine for the Freedonia passage only
        return {"passages": [{"id": i, "relevant": i == 1} for i in ids],
                "gold_support": {"supported": True, "quote": quote},
                "sentences": [{"idx": int(i), "supported": True, "quote": quote,
                               "cited": [{"passage": int(c), "supported": True, "quote": quote}
                                         for c in re.findall(r"\[(\d+)\]", cites)]} for i, cites in sents],
                "answer_relevance": 5}
    prompt = payload["prompt"]
    if "Freedonia" in prompt and "capital" in prompt:
        return {"answer": "The capital is Fredville [1].", "short_answer": "Fredville", "citations": [1],
                "confidence": 0.9, "abstained": False}
    return {"answer": "Insufficient evidence in the corpus to answer this question.", "short_answer": "Unanswerable", "citations": [],
            "confidence": 0.0, "abstained": True}


@pytest.fixture
def toy(cfg, tmp_path, monkeypatch):
    monkeypatch.setitem(D.LOADERS, "toy", _toy_loader)
    monkeypatch.setattr(D, "SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(D, "POOLED", D.POOLED | {"toy"})
    data = tmp_path / "bench"
    data.mkdir()
    (data / "MANIFEST.json").write_text("{}", encoding="utf-8")
    cfg["eval"].update({"data_dir": str(data), "results_dir": str(tmp_path / "results"), "n_dev": 1, "n_test": 3,
                        "unanswerable_fraction": 0.34, "bootstrap": {"n": 200, "ci": 0.95},
                        "judge_enabled": True})  # off by default since M2; keep the judge path covered
    return cfg


def test_jsonl_doc_parser(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"section": "A", "text": "one  two"}\n\n{"section": null, "text": "three"}\n', encoding="utf-8")
    assert parse_jsonl_doc(p) == [{"page_number": 1, "section": "A", "text": "one two"},
                                  {"page_number": 2, "section": None, "text": "three"}]
    p.write_text('{"section": "A"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        parse_jsonl_doc(p)


def test_splits_are_fixed_and_injection_removes_support(toy):
    D.make_splits("toy", toy)
    with pytest.raises(FileExistsError):
        D.make_splits("toy", toy)
    sp = D.read_splits("toy")
    assert len(sp["dev"]) == 1 and len(sp["test"]) == 3 and not set(sp["dev"]) & set(sp["test"])
    assert len(sp["injected_unanswerable"]) == 1
    inj = sp["injected_unanswerable"][0]
    split = "dev" if inj in sp["dev"] else "test"
    item = next(i for i in D.load_split("toy", split, toy) if i.qid == inj)
    assert not item.answerable and item.answers == [D.ABSTAIN_GOLD] and item.meta["original_answers"]
    D.build_corpus("toy", toy)
    corpus = "".join(p.read_text(encoding="utf-8") for p in D.corpus_dir("toy", toy).glob("*.jsonl"))
    support_text = dict((q, paras[s[0]][1]) for q, _, _, paras, s in ITEMS)[inj]
    assert support_text not in corpus  # its supporting paragraph is gone, distractors stay


def test_quote_check():
    texts = ["The flag of Sylvania is green with a white star placed at its centre."]
    assert quote_in("flag of Sylvania is green", texts)
    assert quote_in("The flag of Sylvania is green with a white star", texts)
    assert not quote_in("Robin Wieruch wrote the book", texts)
    assert not quote_in("green", texts)  # too short to be evidence


def test_judge_overrules_unquoted_support(cfg):
    fake = FakeOllama(_fake_answer)
    j = Judge(cfg, OllamaClient(cfg, transport=fake))
    ev = [{"cite_id": 1, "text": "Sylvania has green flags and long rivers."}]
    out = j.judge("What is the capital of Freedonia?", "The capital is Fredville [1].", ev, "Fredville")
    sc = out["scores"]  # every 'supported' label quotes text that is not in this passage
    assert sc["context_recall"] == 0.0 and sc["faithfulness"] == 0.0 and sc["citation_precision"] == 0.0
    assert sc["overruled"] == sc["n_support_labels"] == 3


def test_judge_rejects_same_family(cfg, fake):
    cfg["models"]["judge"] = "qwen2.5:3b"
    with pytest.raises(ValueError):
        Judge(cfg, OllamaClient(cfg, transport=fake))


def test_eval_run_end_to_end(toy, monkeypatch, tmp_path):
    from apex.eval import run as R
    from apex.eval.suite import main as suite_main
    from apex import config as C

    fake = FakeOllama(_fake_answer)
    client = OllamaClient(toy, transport=fake)
    monkeypatch.setattr(R, "load_config", lambda path=None, overrides=None: C.deep_merge(toy, overrides or {}))
    D.make_splits("toy", toy)
    D.build_corpus("toy", toy)
    from apex.ingest.pipeline import ingest_path
    from apex.pipeline import Components
    c = Components(C.deep_merge(toy, D.bench_overrides("toy")), client=client)
    ingest_path(D.corpus_dir("toy", toy), c.cfg, c.store, c.embedder, log=lambda *_: None)

    run_dir = R.run_eval(None, ["toy"], split="test", run_id="t1", client=client, log=lambda *_: None)
    rows = [json.loads(l) for l in open(run_dir / "toy" / "per_query.jsonl", encoding="utf-8")]
    assert len(rows) == 3 and all(r["judged"] for r in rows)
    m = json.loads((run_dir / "toy" / "metrics.json").read_text(encoding="utf-8"))
    assert {"retrieval", "answer", "judge", "abstention", "cost"} <= set(m)
    assert m["retrieval"]["recall@5"]["mean"] is not None
    assert (run_dir / "table.md").read_text(encoding="utf-8").startswith("# Run `t1`")
    for r in rows:  # judge calls never count as system cost
        assert r["cost"]["llm_calls"] <= 1
    n_gen = fake.count("/api/generate")

    # resumable: a second run over the same id does no new work
    R.run_eval(None, ["toy"], split="test", run_id="t1", client=client, log=lambda *_: None)
    assert fake.count("/api/generate") == n_gen

    # paired comparison of a run with itself: zero difference everywhere
    from apex.eval.report import compare
    text = compare(run_dir, run_dir)
    assert "| toy | f1 |" in text and "+0.0" in text

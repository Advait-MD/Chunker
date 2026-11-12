import json

import pytest

from apex.llm import OllamaClient, OllamaError
from apex.trace import Tracer


def _lines(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def test_generate_cached_by_content_hash(cfg, fake):
    c = OllamaClient(cfg, transport=fake)
    a = c.generate("hello", model="m", caller="t")
    b = c.generate("hello", model="m", caller="t")
    c.generate("hello", model="m", caller="t", temperature=0.5)  # different options → miss
    assert fake.count("/api/generate") == 2
    assert not a.cache_hit and b.cache_hit
    assert a.text == b.text and a.prompt_hash == b.prompt_hash


def test_generate_sends_seed_and_temperature(cfg, fake):
    OllamaClient(cfg, transport=fake).generate("p", model="m", caller="t", temperature=0.0, json_schema={"type": "object"})
    opts = fake.calls[0]["payload"]["options"]
    assert opts["seed"] == cfg["run"]["seed"] and opts["temperature"] == 0.0
    assert fake.calls[0]["payload"]["format"] == {"type": "object"}


def test_every_call_logged_with_required_fields(cfg, fake):
    c = OllamaClient(cfg, transport=fake)
    c.generate("p", model="m", caller="unit", state="GENERATING")
    c.embed(["x", "y"], model="e", caller="unit.embed")
    recs = _lines(cfg["llm"]["log_path"])
    assert [r["kind"] for r in recs] == ["generate", "embed"]
    for r in recs:
        for f in ("prompt_hash", "model", "prompt_tokens", "completion_tokens", "latency_ms", "caller", "state", "cache_hit"):
            assert f in r
    assert recs[0]["state"] == "GENERATING" and recs[0]["prompt_tokens"] == 120


def test_trace_receives_calls_and_state(cfg, fake):
    c = OllamaClient(cfg, transport=fake)
    t = Tracer(cfg).new("q")
    t.enter("RETRIEVING")
    c.bind_trace(t)
    c.embed(["q"], model="e", caller="x")
    c.bind_trace(None)
    c.generate("p", model="m", caller="y")  # unbound: not on trace
    assert len(t.llm_calls) == 1 and t.llm_calls[0]["state"] == "RETRIEVING"
    assert t.llm_calls[0]["trace_id"] == t.trace_id


def test_embed_partial_cache(cfg, fake):
    c = OllamaClient(cfg, transport=fake)
    first = c.embed(["a1", "b2"], model="e", caller="t")
    second = c.embed(["b2", "c3"], model="e", caller="t")
    assert second[0] == first[1]
    sent = [call["payload"]["input"] for call in fake.calls]
    assert sent == [["a1", "b2"], ["c3"]]


def test_embed_count_mismatch_is_loud(cfg):
    c = OllamaClient(cfg, transport=lambda e, p, t: {"embeddings": []})
    with pytest.raises(OllamaError):
        c.embed(["a"], model="e", caller="t")


def test_non_json_output_is_loud(cfg):
    c = OllamaClient(cfg, transport=lambda e, p, t: {"response": "not json"})
    with pytest.raises(OllamaError):
        c.generate("p", model="m", caller="t").json()


def test_backend_is_part_of_generate_cache_key(cfg, fake):
    OllamaClient(cfg, transport=fake).generate("p", model="m", caller="t")
    other = dict(cfg, ollama=dict(cfg["ollama"], backend="cpu-test"))
    r = OllamaClient(other, transport=fake).generate("p", model="m", caller="t")
    assert not r.cache_hit and fake.count("/api/generate") == 2
    assert _lines(cfg["llm"]["log_path"])[-1]["backend"] == "cpu-test"


def test_embed_forwards_options(cfg, fake):
    OllamaClient(cfg, transport=fake).embed(["a"], model="e", caller="t", options={"num_batch": 2048})
    assert fake.calls[0]["payload"]["options"] == {"num_batch": 2048}

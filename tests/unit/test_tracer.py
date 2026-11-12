import json

from apex.trace import Tracer


def test_states_stages_totals_and_save(cfg):
    tr = Tracer(cfg)
    t = tr.new("what?")
    t.enter("RETRIEVING")
    with t.stage("retrieve", k=5) as st:
        st["n"] = 3
    t.record_llm({"kind": "generate", "prompt_tokens": 10, "completion_tokens": 4, "cache_hit": False})
    t.record_llm({"kind": "generate", "prompt_tokens": 5, "completion_tokens": 1, "cache_hit": True})
    t.record_llm({"kind": "embed", "prompt_tokens": 99})
    t.enter("DONE")

    assert [s["state"] for s in t.states] == ["RECEIVED", "RETRIEVING", "DONE"]
    assert t.stages["retrieve"]["k"] == 5 and "latency_ms" in t.stages["retrieve"]
    tot = t.totals()
    assert tot["llm_calls"] == 2 and tot["llm_calls_uncached"] == 1 and tot["embed_calls"] == 1
    assert tot["prompt_tokens"] == 15  # embeddings excluded from generation token cost

    data = json.load(open(tr.save(t), encoding="utf-8"))
    assert data["trace_id"] == t.trace_id and data["config_hash"] == cfg["_meta"]["config_hash"]
    assert data["git_sha"]

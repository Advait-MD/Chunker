"""RAPTOR tree (M4): clustering rules, tree build with a fake Ollama, cache reuse, collapsed-tree retrieval."""

import json

import numpy as np
import pytest

from apex import config as C
from apex.index.chroma_store import NodeStore, tree_collection_name
from apex.index.raptor import ClusterParams, ClusterStats, TreeBuilder, cluster_nodes
from apex.ingest.pipeline import ingest_path
from apex.llm import OllamaClient
from apex.pipeline import Components, build_retriever

RAPTOR = {"enabled": False, "tag": "raptor", "seed": 42, "scope": "collection", "summarizer": "qwen2.5:3b",
          "summary_max_tokens": 256, "num_ctx": 6144, "umap_dim": 10, "local_neighbors": 10, "max_clusters": 50,
          "threshold": 0.1, "cluster_max_tokens": 3500, "max_depth": 4, "min_clusters": 3}


def _blobs(n_per=30, k=3, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim)) * 10
    x = np.concatenate([c + rng.normal(size=(n_per, dim)) for c in centers]).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True), np.repeat(np.arange(k), n_per)


def test_clusters_recover_blobs_and_are_deterministic():
    x, truth = _blobs()
    p = ClusterParams(max_tokens=10_000)
    a = cluster_nodes(x, [100] * len(x), p)
    assert a == cluster_nodes(x, [100] * len(x), p)
    assert set(i for c in a for i in c) == set(range(len(x)))  # nobody dropped
    for c in a:  # every cluster is pure: soft assignment never mixes well-separated blobs
        assert len(set(truth[c])) == 1


def test_token_budget_recurses_or_force_splits():
    x, _ = _blobs(n_per=4, k=2)  # 8 points <= dim + 1: one cluster, too big for the budget
    st = ClusterStats()
    out = cluster_nodes(x, [1000] * 8, ClusterParams(max_tokens=3500), st)
    assert out == [[0, 1, 2], [3, 4, 5], [6, 7]] and st.forced_splits == 1
    x, _ = _blobs()
    out = cluster_nodes(x, [400] * len(x), ClusterParams(max_tokens=3500))
    assert all(len(c) * 400 <= 3500 or len(c) == 1 for c in out)


DOCS = {f"d{i}.txt": t for i, t in enumerate([
    "Freedonia's capital is Fredville, on the river Marx.", "Fredville was founded in 1820 by Rufus Firefly.",
    "Sylvania borders Freedonia; its ambassador is Trentino.", "Sylvania exports timber and wool.",
    "Turbine error E1234 means overheating.", "Error E9 on the turbine means the door is open."])}


@pytest.fixture
def tree_cfg(cfg, tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    cfg = C.deep_merge(cfg, {"raptor": RAPTOR, "retrieval": {"mode": "hybrid", "top_k": 3, "levels": [0, 1, 2, 3, 4]}})
    cfg["_corpus"] = d
    return cfg


def _summary(payload):
    return "Summary: " + " ".join(sorted(set(w for w in payload["prompt"].split() if w[:1].isupper())))[:300]


def test_tree_build_links_cache_and_collapsed_retrieval(tree_cfg, fake):
    fake.answer_fn = _summary
    client = OllamaClient(tree_cfg, transport=fake)
    leaf = Components(tree_cfg, client=client)
    ingest_path(tree_cfg["_corpus"], tree_cfg, leaf.store, leaf.embedder, log=lambda *_: None)

    with pytest.raises(RuntimeError, match="does not exist"):
        NodeStore(C.deep_merge(tree_cfg, {"raptor": {"enabled": True}}))

    rep = TreeBuilder(tree_cfg, leaf.store, leaf.embedder, client, log=lambda *_: None).build()
    assert rep["summaries_per_level"] == {"1": 1} and rep["leaves"] == 6  # 6 leaves <= dim + 1: one root summary
    n_gen = fake.count("/api/generate")
    gen = [c["payload"] for c in fake.calls if c["endpoint"] == "/api/generate"]
    assert n_gen == 1 and gen[0]["model"] == "qwen2.5:3b" and gen[0]["options"]["temperature"] == 0.0
    assert "Fredville" in gen[0]["prompt"] and "Passages:" in gen[0]["prompt"]

    tree = NodeStore(tree_cfg, collection=tree_collection_name(tree_cfg))
    nodes, _ = tree.all_nodes()
    by_level = {lvl: [n for n in nodes if n.level == lvl] for lvl in (0, 1)}
    root = by_level[1][0]
    assert len(by_level[0]) == 6 and sorted(root.child_ids) == sorted(n.id for n in by_level[0])
    assert all(n.parent_ids == [root.id] for n in by_level[0]) and root.source_path == "__raptor__"
    assert leaf.store.count({"level": 1}) == 0  # the leaf collection is untouched

    again = TreeBuilder(tree_cfg, leaf.store, leaf.embedder, client, log=lambda *_: None).build()
    assert again["status"] == "cached" and fake.count("/api/generate") == n_gen
    rebuilt = TreeBuilder(tree_cfg, leaf.store, leaf.embedder, client, log=lambda *_: None).build(force=True)
    assert rebuilt["llm_cache_hits"] == 1 and fake.count("/api/generate") == n_gen  # LLM cache, no new call

    on = C.deep_merge(tree_cfg, {"raptor": {"enabled": True}})
    ret = build_retriever(Components(on, client=client))
    assert ret.build_info["n_nodes"] == 7
    ids = [h.node.id for h in ret.retrieve("Freedonia Fredville Sylvania Trentino capital", k=7)]
    assert root.id in ids

    trav_cfg = C.deep_merge(on, {"retrieval": {"tree_search": "traversal", "traversal_k": 2}})
    trav = build_retriever(Components(trav_cfg, client=client))
    hits = trav.retrieve("Turbine error E1234 overheating", k=3)
    assert trav.name == "tree_traversal" and trav.last["layers"] == 2  # the root, then 2 of its 6 leaves
    assert {h.node.id for h in hits} == {root.id} | {h.node.id for h in hits if h.node.level == 0}
    assert len(hits) == 3 and any("E1234" in h.node.text for h in hits)

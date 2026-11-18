"""Query pipelines. M1: the naive baseline (flat leaves → dense top-k → cited generation).
M3: the retriever is chosen by `retrieval.mode` (dense HNSW, or BM25/IVF hybrid with fusion and rerank).

Later milestones add hybrid retrieval, the RAPTOR tree, the evidence filter, the DSM router
and the multi-hop controller as separate, ablatable stages behind the same `answer()` contract.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from apex.config import load_config
from apex.generate.citations import check_citations
from apex.generate.generator import Generator, pack_evidence
from apex.index.chroma_store import NodeStore
from apex.index.embedder import Embedder
from apex.llm import OllamaClient
from apex.retrieve.dense import DenseRetriever
from apex.retrieve.hybrid import HybridRetriever
from apex.retrieve.rerank import build_reranker
from apex.retrieve.tree import TreeTraversalRetriever
from apex.trace import Tracer


class Components:
    """Wires shared components once from a config dict."""

    def __init__(self, cfg: Dict[str, Any], client: Optional[OllamaClient] = None, store_path: Optional[str] = None):
        self.cfg = cfg
        self.client = client or OllamaClient(cfg)
        self.store = NodeStore(cfg, path=store_path)
        self.embedder = Embedder(cfg, self.client)
        self.tracer = Tracer(cfg)


def build_retriever(comps: Components):
    """`retrieval.mode: dense` keeps the M1 Chroma-HNSW retriever unchanged; other modes are hybrid.py."""
    cfg = comps.cfg
    if cfg["retrieval"]["mode"] == "dense":
        if cfg.get("rerank", {}).get("enabled"):
            raise ValueError("rerank.enabled needs retrieval.mode hybrid, dense_ivf or bm25")
        return DenseRetriever(cfg, comps.store, comps.embedder)
    if cfg["retrieval"].get("tree_search", "collapsed") == "traversal":
        return TreeTraversalRetriever(cfg, comps.store, comps.embedder, reranker=build_reranker(cfg))
    return HybridRetriever(cfg, comps.store, comps.embedder, reranker=build_reranker(cfg))


class NaivePipeline:
    name = "naive"

    def __init__(self, comps: Components):
        self.c = comps
        self.retriever = build_retriever(comps)
        self.retriever_name = getattr(self.retriever, "name", "dense_hnsw")
        self.generator = Generator(comps.cfg, comps.client)
        self.budget = int(comps.cfg["generation"]["context_token_budget"])

    def retrieve_ranked(self, query: str, k: int, source: Optional[str] = None):
        """Ranked retrieval at an arbitrary depth, for retrieval metrics (Recall@k, MRR, nDCG). Untraced."""
        return self.retriever.retrieve(query, k=k, source=source)

    def answer(self, query: str, source: Optional[str] = None, save_trace: bool = True) -> Dict[str, Any]:
        trace = self.c.tracer.new(query)
        trace.route, trace.routing_tier = "SINGLE_HOP", "none"  # no router in the naive baseline
        self.c.client.bind_trace(trace)
        try:
            trace.enter("RETRIEVING")
            with trace.stage("retrieve", retriever=self.retriever_name, k=self.retriever.top_k) as st:
                hits = self.retriever.retrieve(query, source=source)
                trace.retrieved_ids = [h.node.id for h in hits]
                st["n_hits"] = len(hits)
                st.update(getattr(self.retriever, "last", {}) or {})
                if hits and hits[0].detail:
                    st["hits"] = [{"id": h.node.id, "score": h.score, **h.detail} for h in hits]

            trace.enter("FILTERING")  # pass-through in M1; only budget packing
            with trace.stage("pack", token_budget=self.budget) as st:
                evidence = pack_evidence(hits, self.budget)
                trace.filtered_ids = [it.node_id for it in evidence.items]
                st["n_items"] = len(evidence.items)
                st["context_tokens"] = sum(h.node.token_count for h in hits if h.node.id in trace.filtered_ids)

            if not evidence.items:
                trace.enter("ABSTAIN")
                gen = {"answer": "Insufficient evidence in the corpus to answer this question.",
                       "short_answer": "Unanswerable", "citations": [], "confidence": 0.0, "abstained": True}
            else:
                trace.enter("GENERATING")
                with trace.stage("generate", model=self.generator.model):
                    gen = self.generator.generate(query, evidence)

            cites = check_citations(gen["answer"], gen.get("citations", []), len(evidence.items))
            by_id = {it.cite_id: it for it in evidence.items}
            citations = [{"cite_id": i, "node_id": by_id[i].node_id, "source": by_id[i].source,
                          "page": by_id[i].page, "section": by_id[i].section, "char_span": by_id[i].char_span}
                         for i in cites["valid"]]
            trace.extra["citation_check"] = {k: v for k, v in cites.items() if k != "valid"}
            trace.enter("DONE" if not gen.get("abstained") else "ABSTAIN")
        finally:
            self.c.client.bind_trace(None)

        result = {
            "query": query, "answer": gen["answer"], "short_answer": gen.get("short_answer"), "citations": citations,
            "confidence": gen.get("confidence"), "abstained": bool(gen.get("abstained")),
            "invalid_citations": cites["invalid"], "uncited_sentences": cites["uncited_sentences"],
            "evidence": [asdict(it) for it in evidence.items],
            "trace_id": trace.trace_id, "route": trace.route, "totals": trace.totals(), "pipeline": self.name,
        }
        if save_trace:
            result["trace_path"] = self.c.tracer.save(trace)
        result["_trace"] = trace
        return result


PIPELINES = {"naive": NaivePipeline}


def build_pipeline(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
                   client: Optional[OllamaClient] = None, store_path: Optional[str] = None):
    cfg = load_config(config_path, overrides)
    comps = Components(cfg, client=client, store_path=store_path)
    name = cfg.get("pipeline", "naive")
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline '{name}'. Available: {sorted(PIPELINES)}")
    return PIPELINES[name](comps)

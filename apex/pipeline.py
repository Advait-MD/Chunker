"""Query pipelines. M1: the naive baseline (flat leaves → dense top-k → cited generation).
M3: the retriever is chosen by `retrieval.mode` (dense HNSW, or BM25/IVF hybrid with fusion and rerank).
M5: `filter.enabled` adds the evidence filter (threshold, dedup, sentence pruning, sufficiency with one
rewrite-and-retry, abstain); `verifier.enabled` adds the NLI citation verifier after generation.

Later milestones add hybrid retrieval, the RAPTOR tree, the evidence filter, the DSM router
and the multi-hop controller as separate, ablatable stages behind the same `answer()` contract.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from apex.config import load_config
from apex.generate.citations import check_citations
from apex.filter.evidence import EvidenceFilter
from apex.filter.nli import shared_nli
from apex.filter.sufficiency import QueryRewriter, SufficiencyChecker
from apex.generate.generator import EvidenceSet, Generator, pack_evidence
from apex.generate.verifier import CitationVerifier
from apex.index.chroma_store import Hit, NodeStore
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
        cfg = comps.cfg
        self.c = comps
        self.retriever = build_retriever(comps)
        self.retriever_name = getattr(self.retriever, "name", "dense_hnsw")
        self.generator = Generator(cfg, comps.client)
        self.budget = int(cfg["generation"]["context_token_budget"])
        # M5: evidence filter, sufficiency check with one rewrite-and-retry, NLI citation verifier
        f, v = cfg.get("filter") or {}, cfg.get("verifier") or {}
        self.filter = self.sufficiency = self.rewriter = self.verifier = None
        nli = shared_nli(cfg) if (v.get("enabled") or (f.get("enabled") and f["conflicts"]["enabled"])) else None
        if f.get("enabled"):
            if getattr(self.retriever, "reranker", None) is None:
                raise ValueError("filter.enabled needs rerank.enabled: the threshold and sufficiency use reranker scores")
            self.pool_k = int(f["pool_k"])
            self.filter = EvidenceFilter(cfg, embedder=comps.embedder, nli=nli)
            self.sufficiency = SufficiencyChecker(cfg, comps.client)
            self.rewriter = QueryRewriter(cfg, comps.client) if f["rewrite"]["enabled"] else None
        if v.get("enabled"):
            self.verifier = CitationVerifier(cfg, nli)
        if self.filter or self.verifier:
            self.name = "apex_m5"

    def _filtered(self, query: str, source: Optional[str], trace) -> Tuple[EvidenceSet, Dict[str, Any]]:
        """Filter → sufficiency → (insufficient: rewrite, retrieve again, merge, re-filter, re-check once)."""
        with trace.stage("filter", pool_k=self.pool_k) as st:
            hits = self.retriever.retrieve(query, k=self.pool_k, source=source)
            fr = self.filter.run(query, hits)
            st.update(fr.stats)
        with trace.stage("sufficiency") as st:
            suff = self.sufficiency.check(query, fr.evidence)
            st.update(suff)
        info: Dict[str, Any] = {"sufficiency": [suff], "rewritten_query": None}
        if suff["verdict"] == "insufficient" and self.rewriter:
            with trace.stage("rewrite_retry") as st:
                q2 = self.rewriter.rewrite(query)
                info["rewritten_query"] = st["query"] = q2
                if q2 and q2 != query:
                    new = [h for h in self.retriever.retrieve(q2, k=self.pool_k, source=source)
                           if h.node.id not in {x.node.id for x in hits}]
                    if new:  # score the new hits against the original question, so all scores are comparable
                        sc = self.retriever.reranker.score(query, [h.node.text for h in new])
                        new = [Hit(node=h.node, score=float(s), rank=h.rank, detail={**h.detail, "retry": True})
                               for h, s in zip(new, sc)]
                    st["n_new"] = len(new)
                    merged = sorted(list(hits) + new, key=lambda h: -h.score)[:self.pool_k]
                    fr = self.filter.run(query, merged)
                    st["filter"] = fr.stats
                    suff = self.sufficiency.check(query, fr.evidence)
                    st["sufficiency"] = suff
                    info["sufficiency"].append(suff)
        fr.evidence.sufficient = suff["verdict"] == "sufficient"
        return fr.evidence, info

    def _verify(self, query, evidence, gen, trace, st) -> Dict[str, Any]:
        """Verify; per `verifier.action` flag, remove unsupported sentences, or regenerate once then remove."""
        rep = self.verifier.verify(gen["answer"], evidence)
        out: Dict[str, Any] = {"first": {k: rep[k] for k in ("n_sentences", "n_supported")}, "regenerated": False,
                               "removed": 0, "all_unsupported": False}
        if rep["unsupported"] and self.verifier.action == "regenerate":
            gen2 = self.generator.generate(query, evidence, feedback=self.verifier.feedback(rep))
            out["regenerated"] = True
            if not gen2.get("abstained"):
                rep2 = self.verifier.verify(gen2["answer"], evidence)
                # keep the regeneration only if it is at least as well supported (fraction of sentences)
                frac = lambda r: r["n_supported"] / r["n_sentences"] if r["n_sentences"] else 0.0
                if frac(rep2) >= frac(rep):
                    gen.clear(); gen.update(gen2); rep = rep2
                    out["regeneration_kept"] = True
                else:
                    out["regeneration_kept"] = False
            else:
                out["regeneration_kept"] = False
        if rep["unsupported"] and self.verifier.action in ("remove", "regenerate"):
            kept = self.verifier.remove_unsupported(gen["answer"], rep)
            if kept is None:
                out["all_unsupported"] = True
            else:
                out["removed"] = len(rep["unsupported"])
                gen["answer"] = kept
        out["final"] = {k: rep[k] for k in ("n_sentences", "n_supported")}
        out["sentences"] = rep["sentences"]
        st.update({k: v for k, v in out.items() if k != "sentences"})
        return out

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

            trace.enter("FILTERING")
            filt: Dict[str, Any] = {}
            if self.filter:
                evidence, filt = self._filtered(query, source, trace)
                trace.filtered_ids = [it.node_id for it in evidence.items]
            else:  # M1-M4: pass-through, budget packing only
                with trace.stage("pack", token_budget=self.budget) as st:
                    evidence = pack_evidence(hits, self.budget)
                    trace.filtered_ids = [it.node_id for it in evidence.items]
                    st["n_items"] = len(evidence.items)
                    st["context_tokens"] = sum(h.node.token_count for h in hits if h.node.id in trace.filtered_ids)

            verification = None
            if not evidence.items or not evidence.sufficient:
                trace.enter("ABSTAIN")
                gen = {"answer": "Insufficient evidence in the corpus to answer this question.",
                       "short_answer": "Unanswerable", "citations": [], "confidence": 0.0, "abstained": True,
                       "abstained_by": "sufficiency" if evidence.items else "no_evidence"}
            else:
                trace.enter("GENERATING")
                with trace.stage("generate", model=self.generator.model):
                    gen = self.generator.generate(query, evidence)
                if self.verifier and not gen.get("abstained"):
                    trace.enter("VERIFYING")
                    with trace.stage("verify", action=self.verifier.action) as st:
                        verification = self._verify(query, evidence, gen, trace, st)

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
        if self.filter:
            result["filter"] = {"stats": trace.stages.get("filter"), **filt, "conflicts": evidence.conflicts,
                                "abstained_by": gen.get("abstained_by")}
        if verification is not None:
            result["verification"] = verification
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

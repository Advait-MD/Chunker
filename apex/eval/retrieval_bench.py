"""Retrieval-only evaluation: no generation, minutes instead of hours (M3 tuning and ablations).

`compare_retrievers`: Recall@5/10/20, MRR and nDCG@10 (same scorer as the full eval) plus
retriever latency, for several retrieval configs on the same questions. Used to tune on dev and
for the IVF-vs-HNSW and RRF-vs-weighted comparisons.

`ivf_curve`: the spec's Recall@k vs latency curve for FAISS IVF over nprobe, with exact (flat)
search and Chroma HNSW as reference points. Only corpus-wide (pooled) datasets: scoped datasets
search a single document, where an ANN index has nothing to do. Query embeddings are computed once,
so the latency is index search time alone. Besides gold Recall@10 it reports ANN recall@10:
the overlap with the exact top 10, which isolates the error IVF itself introduces.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from apex import config as C
from apex.eval import datasets as D
from apex.eval import metrics as M
from apex.eval.run import retrieval_scores
from apex.pipeline import Components, build_retriever
from apex.retrieve.hybrid import HybridRetriever


def _bench_cfg(config_path: Optional[str], overrides: Dict[str, Any], name: str) -> Dict[str, Any]:
    return C.load_config(config_path, overrides=C.deep_merge(D.bench_overrides(name), overrides))


def _ci(vals: Sequence[float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    b = cfg["eval"]["bootstrap"]
    return M.bootstrap_ci(list(vals), n=int(b["n"]), ci=float(b["ci"]), seed=int(cfg["run"]["seed"]))


def compare_retrievers(variants: Dict[str, Dict[str, Any]], datasets: List[str], split: str = "dev",
                       config_path: Optional[str] = None, out_dir: Optional[Path] = None, limit: Optional[int] = None,
                       log: Callable[[str], None] = print, client=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split, "variants": variants, "datasets": {}}
    for name in datasets:
        out["datasets"][name] = {}
        for vname, ov in variants.items():
            cfg = _bench_cfg(config_path, ov, name)
            comps = Components(cfg, client=client)
            items = [it for it in D.load_split(name, split, cfg)[:limit] if it.answerable and it.evidence_sets]
            t0 = time.perf_counter()
            ret = build_retriever(comps)
            build_s = time.perf_counter() - t0
            depth = max(cfg["eval"]["retrieval_ks"])
            rows, lat = [], []
            for it in items:
                q = it.prompt()
                comps.embedder.embed_query(q)  # warm the embedding cache so latency is retrieval only
                t = time.perf_counter()
                hits = ret.retrieve(q, k=depth, source=it.scope)
                lat.append(1000 * (time.perf_counter() - t))
                rows.append(retrieval_scores([h.node.text for h in hits], it.evidence_sets, cfg))
            keys = list(rows[0]) if rows else []
            res = {k: _ci([r[k] for r in rows], cfg) for k in keys}
            res.update({"n": len(rows), "latency_ms_p50": M.percentile(lat, 50), "latency_ms_p95": M.percentile(lat, 95),
                        "retriever": getattr(ret, "name", "dense_hnsw"), "build_s": round(build_s, 2),
                        **({"build": ret.build_info} if isinstance(ret, HybridRetriever) else {})})
            out["datasets"][name][vname] = res
            log(f"[{name}] {vname:18s} n={len(rows)} R@10={100 * res['recall@10']['mean']:.1f} "
                f"MRR={100 * res['mrr']['mean']:.1f} p50={res['latency_ms_p50']:.1f} ms" if rows else f"[{name}] {vname}: no items")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "retrieval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (out_dir / "retrieval.md").write_text(retrieval_table(out), encoding="utf-8")
    return out


def _f(ci: Optional[Dict[str, Any]]) -> str:
    if not ci or ci.get("mean") is None:
        return "—"
    return f"{100 * ci['mean']:.1f} ({100 * ci['lo']:.1f}–{100 * ci['hi']:.1f})"


def retrieval_table(out: Dict[str, Any]) -> str:
    lines = [f"# Retrieval-only comparison ({out['split']} split)", "",
             "Answerable questions with gold evidence. % with 95% bootstrap CIs; latency is the retriever alone "
             "(query embedding cached), including the reranker when enabled.", "",
             "| Dataset | Variant | Retriever | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | p50 ms | p95 ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, vs in out["datasets"].items():
        for vname, r in vs.items():
            if not r.get("n"):
                continue
            lines.append(f"| {name} | {vname} | `{r['retriever']}` | {r['n']} | {_f(r['recall@5'])} | {_f(r['recall@10'])} | "
                         f"{_f(r['recall@20'])} | {_f(r['mrr'])} | {_f(r['ndcg@10'])} | {r['latency_ms_p50']:.1f} | "
                         f"{r['latency_ms_p95']:.1f} |")
    return "\n".join(lines) + "\n"


def ivf_curve(datasets: List[str], split: str = "dev", nprobes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
              config_path: Optional[str] = None, out_dir: Optional[Path] = None, repeats: int = 5,
              log: Callable[[str], None] = print, client=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split, "datasets": {}}
    for name in datasets:
        if name not in D.POOLED and name != "custom":
            raise ValueError(f"{name} is searched per document; the IVF curve needs a corpus-wide dataset")
        cfg = _bench_cfg(config_path, {"retrieval": {"mode": "dense_ivf", "dense_backend": "ivf"}}, name)
        comps = Components(cfg, client=client)
        ret = HybridRetriever(cfg, comps.store, comps.embedder)
        items = [it for it in D.load_split(name, split, cfg) if it.answerable and it.evidence_sets]
        qv = [comps.embedder.embed_query(it.prompt()) for it in items]
        k = 10
        exact = [ret.ivf.search(q, k, 0, exact=True) for q in qv]

        def measure(search) -> Dict[str, Any]:
            per_q, lat = [], []
            for q, it, ex in zip(qv, items, exact):
                ts = []
                for _ in range(repeats):
                    t = time.perf_counter(); res = search(q); ts.append(time.perf_counter() - t)
                lat.append(1000 * float(np.median(ts)))
                ids = [nid for nid, _ in res]
                texts = [ret.nodes[nid].text for nid in ids]
                sc = retrieval_scores(texts, it.evidence_sets, cfg)
                per_q.append((sc["recall@10"], len(set(ids) & {nid for nid, _ in ex}) / max(1, len(ex))))
            return {"recall@10": float(np.mean([a for a, _ in per_q])), "ann_recall@10": float(np.mean([b for _, b in per_q])),
                    "latency_ms_p50": M.percentile(lat, 50), "latency_ms_mean": float(np.mean(lat))}

        pts: Dict[str, Any] = {"flat": measure(lambda q: ret.ivf.search(q, k, 0, exact=True))}

        def hnsw(q):
            return [(h.node.id, h.score) for h in comps.store.query(q, k, where={"level": {"$in": ret.levels}})]
        pts["hnsw"] = measure(hnsw)
        sweep = []
        for p in sorted({min(int(x), ret.ivf.nlist) for x in nprobes}):
            r = measure(lambda q, p=p: ret.ivf.search(q, k, p))
            sweep.append({"nprobe": p, **r})
            log(f"[{name}] nprobe={p:3d}/{ret.ivf.nlist} R@10={100 * r['recall@10']:.1f} "
                f"ANN-R@10={100 * r['ann_recall@10']:.1f} p50={r['latency_ms_p50']:.3f} ms")
        out["datasets"][name] = {"n_questions": len(items), "ivf": ret.ivf.info(), "points": pts, "sweep": sweep}
        log(f"[{name}] flat R@10={100 * pts['flat']['recall@10']:.1f} p50={pts['flat']['latency_ms_p50']:.3f} ms; "
            f"hnsw R@10={100 * pts['hnsw']['recall@10']:.1f} p50={pts['hnsw']['latency_ms_p50']:.3f} ms")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ivf_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        plot_ivf_curve(out, out_dir / "ivf_curve.png")
    return out


def plot_ivf_curve(out: Dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(out["datasets"])
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        d = out["datasets"][name]
        xs = [p["latency_ms_p50"] for p in d["sweep"]]
        ax.plot(xs, [100 * p["recall@10"] for p in d["sweep"]], "o-", color="#3266ad", label="IVF: gold Recall@10")
        ax.plot(xs, [100 * p["ann_recall@10"] for p in d["sweep"]], "s--", color="#3266ad", alpha=0.55,
                label="IVF: ANN recall@10 vs exact")
        for p in d["sweep"]:
            ax.annotate(str(p["nprobe"]), (p["latency_ms_p50"], 100 * p["recall@10"]), textcoords="offset points",
                        xytext=(4, -10), fontsize=7)
        for key, mk, col in (("flat", "*", "#2a9d5c"), ("hnsw", "D", "#c0612b")):
            pt = d["points"][key]
            ax.plot([pt["latency_ms_p50"]], [100 * pt["recall@10"]], mk, color=col, markersize=9,
                    label=f"{key} (gold Recall@10)")
        ax.set_xscale("log")
        ax.set_title(f"{name}: N={d['ivf']['n']}, nlist={d['ivf']['nlist']}", fontsize=9)
        ax.set_xlabel("search latency p50 (ms, log)")
        ax.set_ylabel("%")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"FAISS IVF Recall@10 vs latency over nprobe ({out['split']} split); numbers = nprobe", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)

"""
Budget-matched, controlled benchmark for APEX.

Every system answers with the same generator, the same prompt and temperature 0.
The baselines get exactly as many context tokens per question as full APEX used,
so differences come from *what* is retrieved, not *how much*.

Conditions
  closed_book      no retrieval (measures parametric knowledge)
  dense_top5       flat dense top-5 leaves (the paper's naive baseline, unmatched budget)
  dense            flat dense leaves, budget-matched
  dense_srcfilter  dense restricted to the document APEX's router picked (gold set only)
  tree_ungated     RAPTOR-style collapsed tree: leaves + every summary ranked by distance
  tree_gated       same, over the silhouette-gated tree
  apex_ungated     full APEX retrieval over the ungated tree (retrieval-time gate disabled)
  apex_gated       full APEX retrieval over the gated tree (the claimed system)

Both trees are built by the same code from the same chunks. The gated tree reuses the
ungated level-0 summaries whose cluster silhouette passes the gate (so summarizer noise
is shared), then builds its upper levels from those survivors only.

Usage:
  python eval/run_fair_benchmark.py build --db-dir eval/fair_dbs
  python eval/run_fair_benchmark.py run   --db-dir eval/fair_dbs --out eval/fair_results
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import requests
import chromadb

from modules.config import config

config.retrieval.include_conversation_memory = False

SRC_DB = "hierarchical_chroma_db"
QUALITY_FILE = "benchmark_data/quality_stories.json"
GOLD_FILE = "eval/dataset/gold.jsonl"
GEN_MODEL = config.ollama.generation_model
OLLAMA = config.ollama.url.rstrip("/")


def toks(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


# ==================== TREE BUILDING ====================

def _processor(path):
    from main import HierarchicalClusterSummarizer
    return HierarchicalClusterSummarizer(chroma_path=path)


def _build_upper_levels(proc, start_level, max_levels, skip_bad):
    level = start_level
    while level < max_levels:
        emb, docs, metas, _ = proc.store.get_items_by_level(level)
        if emb is None or len(emb) < 3:
            break
        res = proc.perform_clustering(emb)
        if res is None:
            break
        stored = proc.store_cluster_summaries(res, docs, metas, level, skip_bad_clusters=skip_bad)
        if not stored and skip_bad:
            break
        level += 1


def _copy_gated_level0(ungated_path, gated_path, threshold):
    src = chromadb.PersistentClient(path=ungated_path).get_collection(config.chroma.summaries_collection)
    dst = chromadb.PersistentClient(path=gated_path).get_collection(config.chroma.summaries_collection)
    r = src.get(where={"level": 0}, include=["documents", "metadatas", "embeddings"])
    keep = [i for i, m in enumerate(r["metadatas"]) if float(m.get("silhouette_score", 0.0)) > threshold]
    if keep:
        dst.upsert(ids=[r["ids"][i] for i in keep],
                   documents=[r["documents"][i] for i in keep],
                   metadatas=[r["metadatas"][i] for i in keep],
                   embeddings=[r["embeddings"][i] for i in keep])
    print(f"  gated level 0: kept {len(keep)}/{len(r['ids'])} summaries (silhouette > {threshold})")


def _init_empty_db(path):
    """Fresh DB with cosine collections, matching the space of the main PDF DB's collections."""
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=path)
    ef = embedding_functions.DefaultEmbeddingFunction()
    for name in (config.chroma.chunks_collection, config.chroma.summaries_collection,
                 config.chroma.conversation_chunks_collection, config.chroma.conversation_summaries_collection):
        client.get_or_create_collection(name, embedding_function=ef, configuration={"hnsw": {"space": "cosine"}})
    return client


def _init_pdf_db(path):
    """Fresh DB holding the original PDF chunks and their stored embeddings (no summaries).
    Re-inserting avoids copying Chroma's on-disk files, which corrupted the copied index."""
    client = _init_empty_db(path)
    src = chromadb.PersistentClient(path=SRC_DB).get_collection(config.chroma.chunks_collection)
    dst = client.get_collection(config.chroma.chunks_collection)
    total = src.count()
    for off in range(0, total, 1000):
        r = src.get(limit=1000, offset=off, include=["documents", "metadatas", "embeddings"])
        dst.upsert(ids=r["ids"], documents=r["documents"], metadatas=r["metadatas"], embeddings=r["embeddings"])
    print(f"  {os.path.basename(path)}: copied {dst.count()}/{total} chunks")
    return _processor(path)


def _check_ollama():
    """The summarizer silently falls back to keyword summaries when Ollama is down; refuse to build then."""
    try:
        names = [m["name"] for m in requests.get(f"{OLLAMA}/api/tags", timeout=10).json()["models"]]
    except Exception as e:
        sys.exit(f"Ollama is not reachable at {OLLAMA} ({e}). Start it before building.")
    if not any(n.split(":")[0] == GEN_MODEL.split(":")[0] and (":" not in GEN_MODEL or n == GEN_MODEL) for n in names):
        sys.exit(f"Model {GEN_MODEL} is not pulled. Run: ollama pull {GEN_MODEL}")


def _init_quality_db(path):
    """Fresh DB holding the QuALITY articles, chunked with the pipeline's own chunker."""
    from modules.document_loader import create_chunks_from_pages
    _init_empty_db(path)
    proc = _processor(path)
    for story in json.load(open(QUALITY_FILE, encoding="utf-8")):
        chunks = create_chunks_from_pages([{"page_number": 1, "text": story["article"]}])
        proc.store.add_chunks(story["title"], chunks, doc_id=f"quality_{story['article_id']}")
    return proc


def build(db_dir, max_levels):
    _check_ollama()
    os.makedirs(db_dir, exist_ok=True)
    thr = config.clustering.min_cluster_silhouette
    for corpus in ("pdf", "quality"):
        ung, gat = os.path.join(db_dir, f"{corpus}_ungated"), os.path.join(db_dir, f"{corpus}_gated")
        for p in (ung, gat):
            if os.path.exists(p):
                shutil.rmtree(p)
        t0 = time.time()
        if corpus == "pdf":
            proc_u = _init_pdf_db(ung)
            _init_pdf_db(gat)
        else:
            proc_u = _init_quality_db(ung)
            _init_quality_db(gat)
        print(f"[{corpus}] building UNGATED tree...")
        proc_u.hierarchical_clustering(max_levels=max_levels, skip_bad_clusters=False, source_aware=True)
        print(f"[{corpus}] building GATED tree...")
        _copy_gated_level0(ung, gat, thr)
        _build_upper_levels(_processor(gat), 1, max_levels, skip_bad=True)
        for p in (ung, gat):
            col = chromadb.PersistentClient(path=p).get_collection(config.chroma.summaries_collection)
            got = col.get(include=["metadatas", "documents"])
            levels = [m.get("level") for m in got["metadatas"]]
            fallback = sum("Heuristic Fallback" in d for d in got["documents"])
            print(f"  {os.path.basename(p)}: {len(levels)} summaries, by level {dict(zip(*np.unique(levels, return_counts=True)))}")
            if fallback:
                sys.exit(f"{fallback} summaries in {p} are keyword fallbacks (Ollama failed mid-build). Rebuild.")
        print(f"[{corpus}] built in {time.time() - t0:.0f}s")


# ==================== RETRIEVAL CONDITIONS ====================

def _ungate_fusion():
    """Wrap face.collapsed_tree_fusion so the retrieval-time silhouette filter is disabled."""
    import face
    original = face.collapsed_tree_fusion

    def ungated(*args, **kwargs):
        metas = kwargs.get("summary_metas")
        if metas:
            kwargs["summary_metas"] = [dict(m, low_confidence=False, silhouette_score=1.0) if m else m for m in metas]
        return original(*args, **kwargs)
    return original, ungated


def apex_retrieve(rag, question, source, ungated=False):
    import face
    from modules.retrieval import retrieval_result_cache
    retrieval_result_cache.clear()  # cache key ignores the DB path, so clear between interfaces
    original, wrapped = _ungate_fusion()
    if ungated:
        face.collapsed_tree_fusion = wrapped
    try:
        ctx = rag.retrieve_context(question, source_filter=source)
    finally:
        face.collapsed_tree_fusion = original
    passages = list(ctx.get("chunks") or [])
    passages += [s.get("summary", "") for s in (ctx.get("cluster_summaries") or [])]
    ids = [m.get("chroma_id") for m in (ctx.get("metadatas") or []) if m]
    return dict(passages=passages, leaf_ids=ids, n_summaries=len(ctx.get("cluster_summaries") or []),
                mode=ctx.get("chat_mode", "document_rag"))


def _query(col, emb, n, where):
    n = min(n, col.count())
    if n == 0:
        return []
    r = col.query(query_embeddings=[emb], n_results=n, where=where, include=["documents", "distances"])
    return list(zip(r["ids"][0], r["documents"][0], r["distances"][0]))


def _pack(items, budget):
    """Greedy pack (id, text, is_leaf) in rank order up to `budget` tokens; always keep the first item."""
    out, used = [], 0
    for item in items:
        t = toks(item[1])
        if out and used + t > budget:
            continue
        out.append(item)
        used += t
        if used >= budget:
            break
    return dict(passages=[i[1] for i in out], leaf_ids=[i[0] for i in out if i[2]],
                n_summaries=sum(1 for i in out if not i[2]))


def baseline_retrieve(store, question, source, budget, use_tree, k=None):
    emb = store.get_query_embedding(question)
    leaf_where = {"source_file": source} if source else None
    leaves = [(i, d, dist, True) for i, d, dist in _query(store.chunks_collection, emb, 30, leaf_where)]
    pool = leaves
    if use_tree:
        sum_where = {"source_file": {"$in": [source, "multiple"]}} if source else None
        pool = leaves + [(i, d, dist, False) for i, d, dist in _query(store.summaries_collection, emb, 30, sum_where)]
    pool.sort(key=lambda x: x[2])
    items = [(i, d, is_leaf) for i, d, _, is_leaf in pool]
    if k is not None:
        items = items[:k]
        return dict(passages=[i[1] for i in items], leaf_ids=[i[0] for i in items if i[2]], n_summaries=0)
    return _pack(items, budget)


# ==================== GENERATION ====================

def generate(prompt, num_predict):
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": GEN_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "seed": 0, "num_predict": num_predict}}, timeout=600)
    r.raise_for_status()
    return r.json()["response"].strip()


def _context_block(passages):
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def open_prompt(q, passages):
    if not passages:
        return f"Answer the question concisely.\nQuestion: {q}\nAnswer:"
    return ("Use the passages to answer concisely. If the passages do not contain the answer, "
            f"say you don't know.\n\n{_context_block(passages)}\n\nQuestion: {q}\nAnswer:")


def mc_prompt(q, options, passages):
    opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(options))
    ctx = f"{_context_block(passages)}\n\n" if passages else ""
    return f"{ctx}Question: {q}\n{opts}\nAnswer with the letter of the correct option only.\nAnswer:"


def parse_letter(text):
    m = re.search(r"\b([ABCD])\b", text.upper())
    return "ABCD".index(m.group(1)) if m else -1


REFUSALS = ["does not mention", "do not contain", "doesn't contain", "no information", "i don't know",
            "not stated", "cannot answer", "not mentioned", "does not contain", "doesn't mention",
            "not provide", "no mention", "don't know"]


# ==================== RUN ====================

def run(db_dir, out_dir, limit):
    logging.disable(logging.CRITICAL)
    from face import RAGInterface
    from eval.metrics import compute_token_f1, compute_rouge_l, compute_bootstrap_ci, compute_paired_significance
    from modules.retrieval import detect_document_intent

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    def interfaces(corpus):
        r = {}
        for variant in ("gated", "ungated"):
            rag = RAGInterface(chroma_path=os.path.join(db_dir, f"{corpus}_{variant}"), verbose=False)
            rag.record_conversation_memory = False
            r[variant] = rag
        return r

    def run_case(corpus, rag, q, source, scorer, extra_conditions=()):
        rec = {}
        t = time.perf_counter()
        apex = apex_retrieve(rag["gated"], q, source)
        apex["ret_ms"] = (time.perf_counter() - t) * 1000
        budget = max(75, sum(toks(p) for p in apex["passages"]))
        conds = {"apex_gated": apex,
                 "apex_ungated": apex_retrieve(rag["ungated"], q, source, ungated=True),
                 "closed_book": dict(passages=[], leaf_ids=[], n_summaries=0),
                 "dense_top5": baseline_retrieve(rag["gated"].processor.store, q, source, None, False, k=5),
                 "dense": baseline_retrieve(rag["gated"].processor.store, q, source, budget, False),
                 "tree_ungated": baseline_retrieve(rag["ungated"].processor.store, q, source, budget, True),
                 "tree_gated": baseline_retrieve(rag["gated"].processor.store, q, source, budget, True)}
        for name, fn in extra_conditions:
            conds[name] = fn(budget)
        for name, c in conds.items():
            pred = generate(*scorer["prompt"](c["passages"]))
            rec[name] = dict(pred=pred, ctx_tokens=sum(toks(p) for p in c["passages"]),
                             n_summaries=c.get("n_summaries", 0), leaf_ids=c.get("leaf_ids", []),
                             mode=c.get("mode"), **scorer["score"](pred, c))
        return rec

    # ---- Gold set over the four PDFs (APEX's own routing decides the source filter) ----
    gold = [json.loads(l) for l in open(GOLD_FILE, encoding="utf-8")][:limit or None]
    rag = interfaces("pdf")
    srcs = rag["gated"].processor.get_indexed_sources()
    for i, g in enumerate(gold):
        q = g["question"]
        _, detected = detect_document_intent(q, srcs)

        def score(pred, c, g=g):
            refused = any(x in pred.lower() for x in REFUSALS)
            hit = float(any(x in c.get("leaf_ids", []) for x in g.get("gold_chunk_ids", []))) if g["answerable"] else None
            if not g["answerable"]:
                return dict(refused=refused, gold_hit=None, f1=None, rouge_l=None)
            return dict(refused=refused, gold_hit=hit, f1=compute_token_f1(pred, g["gold_answer"]),
                        rouge_l=compute_rouge_l(pred, g["gold_answer"]))
        scorer = dict(prompt=lambda p, q=q: (open_prompt(q, p), 160), score=score)
        extra = [("dense_srcfilter", lambda b, q=q, d=detected: baseline_retrieve(rag["gated"].processor.store, q, d, b, False))]
        rec = run_case("pdf", rag, q, None, scorer, extra)
        rows.append(dict(corpus="gold_pdf", id=g["id"], answerable=g["answerable"], conditions=rec))
        print(f"[gold {i + 1}/{len(gold)}] {g['id']} " + " ".join(
            f"{k}={v['f1']:.2f}" if v["f1"] is not None else f"{k}={'REF' if v['refused'] else 'ANS'}" for k, v in rec.items()), flush=True)

    # ---- QuALITY (multiple choice, per-article source filter for every system) ----
    rag = interfaces("quality")
    stories = json.load(open(QUALITY_FILE, encoding="utf-8"))
    qs = [(s, qq) for s in stories for qq in s["questions"]][:limit or None]
    for i, (s, qq) in enumerate(qs):
        def score(pred, c, qq=qq):
            return dict(correct=float(parse_letter(pred) == qq["gold_label"]))
        scorer = dict(prompt=lambda p, qq=qq: (mc_prompt(qq["question"], qq["options"], p), 8), score=score)
        rec = run_case("quality", rag, qq["question"], s["title"], scorer)
        rows.append(dict(corpus="quality", id=qq["question_id"], difficult=qq.get("difficult"), conditions=rec))
        print(f"[quality {i + 1}/{len(qs)}] " + " ".join(f"{k}={int(v['correct'])}" for k, v in rec.items()), flush=True)

    with open(os.path.join(out_dir, "per_question.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Summary ----
    def col(corpus, cond, key, filt=lambda r: True):
        return [r["conditions"][cond][key] for r in rows
                if r["corpus"] == corpus and cond in r["conditions"] and filt(r) and r["conditions"][cond][key] is not None]

    summary, lines = {}, []
    ans = lambda r: r.get("answerable")
    unans = lambda r: r.get("answerable") is False
    conds = list(rows[0]["conditions"].keys()) if rows else []
    n_ans = sum(1 for g in gold if g["answerable"])
    lines += [f"## Gold set (4 PDFs): {n_ans} answerable, {len(gold) - n_ans} unanswerable", "",
              "| Condition | Token F1 [95% CI] | ROUGE-L | Gold-chunk hit | Refused unanswerable | Context tok | Summaries used | p vs dense (F1) |",
              "|---|---|---|---|---|---|---|---|"]
    base_f1 = col("gold_pdf", "dense", "f1", ans)
    for c in conds:
        f1 = col("gold_pdf", c, "f1", ans)
        if not f1:
            continue
        ci = compute_bootstrap_ci(f1)
        p = compute_paired_significance(f1, base_f1).get("p_value_ttest") if c != "dense" else None
        s = dict(f1=float(np.mean(f1)), f1_ci=[ci["ci_lower"], ci["ci_upper"]],
                 rouge_l=float(np.mean(col("gold_pdf", c, "rouge_l", ans))),
                 gold_hit=float(np.mean(col("gold_pdf", c, "gold_hit", ans))),
                 refusal_unanswerable=float(np.mean(col("gold_pdf", c, "refused", unans))),
                 ctx_tokens=float(np.mean(col("gold_pdf", c, "ctx_tokens"))),
                 summaries_per_q=float(np.mean(col("gold_pdf", c, "n_summaries"))), p_vs_dense=p)
        summary[f"gold_pdf/{c}"] = s
        lines.append(f"| {c} | {s['f1']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] | {s['rouge_l']:.3f} | "
                     f"{s['gold_hit']:.2f} | {s['refusal_unanswerable']:.2f} | {s['ctx_tokens']:.0f} | "
                     f"{s['summaries_per_q']:.2f} | {'—' if p is None else f'{p:.3f}'} |")
    lines += ["", f"## QuALITY ({len(qs)} multiple-choice questions, 3 full articles)", "",
              "| Condition | Accuracy [95% CI] | Hard-question acc | Context tok | Summaries used | p vs dense |",
              "|---|---|---|---|---|---|"]
    base_acc = col("quality", "dense", "correct")
    for c in conds:
        acc = col("quality", c, "correct")
        if not acc:
            continue
        ci = compute_bootstrap_ci(acc)
        hard = col("quality", c, "correct", lambda r: r.get("difficult"))
        p = compute_paired_significance(acc, base_acc).get("p_value_ttest") if c != "dense" else None
        s = dict(acc=float(np.mean(acc)), acc_ci=[ci["ci_lower"], ci["ci_upper"]],
                 hard_acc=float(np.mean(hard)) if hard else None,
                 ctx_tokens=float(np.mean(col("quality", c, "ctx_tokens"))),
                 summaries_per_q=float(np.mean(col("quality", c, "n_summaries"))), p_vs_dense=p)
        summary[f"quality/{c}"] = s
        hard_str = "—" if s["hard_acc"] is None else f"{s['hard_acc']:.2f}"
        lines.append(f"| {c} | {s['acc']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] | "
                     f"{hard_str} | {s['ctx_tokens']:.0f} | "
                     f"{s['summaries_per_q']:.2f} | {'—' if p is None else f'{p:.3f}'} |")
    lines += ["", "Chance accuracy on QuALITY is 0.25. p-values: paired t-test against budget-matched dense."]
    json.dump(dict(generator=GEN_MODEL, summary=summary), open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "run"])
    ap.add_argument("--db-dir", default="eval/fair_dbs")
    ap.add_argument("--out", default="eval/fair_results")
    ap.add_argument("--max-levels", type=int, default=config.clustering.max_levels)
    ap.add_argument("--limit", type=int, default=0, help="Limit questions per dataset (smoke test)")
    a = ap.parse_args()
    if a.command == "build":
        build(a.db_dir, a.max_levels)
    else:
        run(a.db_dir, a.out, a.limit)

"""Run a pipeline config over benchmark splits and score it (spec §2.8 harness).

results/<run_id>/
  run.json                    git SHA, config hash + full config, model digests, Ollama version, backend
  <dataset>/per_query.jsonl   one row per question: answer, citations, evidence, retrieval hits, scores, cost
  <dataset>/metrics.json      aggregated metrics with 95% bootstrap CIs
  metrics.json, table.md      all datasets
  traces/                     one trace per query

Runs are resumable: rows already in per_query.jsonl are skipped. Judge scores can be filled later
(`judge_run`) so generation is not blocked on the judge model.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from apex.config import load_config, resolve_path
from apex.eval import datasets as D
from apex.eval import metrics as M
from apex.eval.judge import Judge, JudgeIncomplete
from apex.generate.generator import GenerationError
from apex.pipeline import Components, PIPELINES
from apex.trace.tracer import git_sha

ABSTAIN = D.ABSTAIN_GOLD


# ---- per-query scoring ---------------------------------------------------------
def retrieval_scores(texts: List[str], evidence_sets: List[List[str]], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Max over alternative gold evidence sets (QASPER has one per annotator)."""
    if not evidence_sets:
        return None
    e = cfg["eval"]
    n, ov = int(e["evidence_match"]["shingle"]), float(e["evidence_match"]["min_overlap"])
    best: Dict[str, float] = {}
    for gold in evidence_sets:
        hits = M.evidence_hits(texts, gold, n=n, min_overlap=ov)
        cur = {f"recall@{k}": M.union_recall_at_k(texts, gold, k, hits, n=n, min_overlap=ov) for k in e["retrieval_ks"]}
        cur["mrr"] = M.mrr(hits)
        cur[f"ndcg@{e['ndcg_k']}"] = M.ndcg_at_k(hits, len(gold), int(e["ndcg_k"]))
        for k, v in cur.items():
            best[k] = max(best.get(k, 0.0), v)
    return best


def answer_scores(item: D.EvalItem, answer: str, short: Optional[str], abstained: bool,
                  evidence_texts: List[str]) -> Dict[str, Any]:
    """EM/F1 on the short answer (primary); F1 on the long cited answer kept for continuity with M1.

    An abstention is the prediction "Unanswerable" for every metric. Acc: a gold alias occurs in the
    long or short answer. answer_in_context (deterministic context recall): a short gold alias
    (≤ 5 tokens, not yes/no) occurs in the packed evidence; None when no such gold exists.
    """
    s_pred = ABSTAIN if abstained else (short or "")
    l_pred = ABSTAIN if abstained else answer
    out: Dict[str, Any] = {
        "em": M.exact_match(s_pred, item.answers), "f1": M.token_f1(s_pred, item.answers),
        "f1_long": M.token_f1(l_pred, item.answers),
        "acc": M.exact_match(s_pred, item.answers) if abstained else
        max(M.answer_contains(l_pred, item.answers), M.answer_contains(s_pred, item.answers)),
    }
    if item.options is not None:
        choice, src = (None, None) if abstained else (M.parse_choice(short or ""), "short_answer")
        if choice is None and not abstained:
            choice, src = M.parse_choice(answer), "answer"
        out.update({"choice": choice, "choice_source": src, "mc_acc": float(choice == item.gold_option)})
    shorts = [g for g in item.answers if g != ABSTAIN and 0 < len(M.normalize_answer(g).split()) <= 5
              and M.normalize_answer(g) not in ("yes", "no")]
    out["answer_in_context"] = (M.answer_contains(" ".join(evidence_texts), shorts)
                                if item.answerable and shorts and item.options is None else None)
    return out


def gold_for_judge(item: D.EvalItem) -> Optional[str]:
    if not item.answerable:
        return None
    return next((a for a in item.answers if a != ABSTAIN), None)


def judge_row(judge: Judge, row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        j = judge.judge(row["question"], None if row["abstained"] else row["answer"], row["evidence"],
                        row.get("gold_for_judge"))
    except JudgeIncomplete as e:  # recorded and reported (judge_errors), excluded from judge metrics
        return {"judge_error": str(e), "judged": True}
    if not j:
        return {"judged": True}
    return {"judge": j["scores"], "judge_raw": j["raw"], "judge_call": j["call"], "judged": True}


# ---- aggregation -----------------------------------------------------------------
def _ci(vals, cfg):
    b = cfg["eval"]["bootstrap"]
    return M.bootstrap_ci(vals, n=int(b["n"]), ci=float(b["ci"]), seed=int(cfg["run"]["seed"]))


def aggregate(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows), "n_answerable": sum(r["answerable"] for r in rows),
                           "n_generation_errors": sum(1 for r in rows if r.get("generation_error"))}
    ret = [r["retrieval"] for r in rows if r.get("retrieval")]
    if ret:
        out["retrieval"] = {k: _ci([x[k] for x in ret], cfg) for k in ret[0]}
    ans = {k: _ci([r["scores"][k] for r in rows], cfg) for k in ("em", "f1", "acc")}
    ans["f1_answerable"] = _ci([r["scores"]["f1"] for r in rows if r["answerable"]], cfg)
    ans["f1_long"] = _ci([r["scores"]["f1_long"] for r in rows], cfg)
    aic = [r["scores"]["answer_in_context"] for r in rows if r["scores"].get("answer_in_context") is not None]
    if aic:
        out["answer_in_context"] = _ci(aic, cfg)
    if any("mc_acc" in r["scores"] for r in rows):
        ans["mc_acc"] = _ci([r["scores"]["mc_acc"] for r in rows], cfg)
        hard = [r["scores"]["mc_acc"] for r in rows if r["meta"].get("hard")]
        if hard:
            ans["mc_acc_hard"] = _ci(hard, cfg)
    out["answer"] = ans
    js = [r["judge"] for r in rows if r.get("judge")]
    if js:
        jd: Dict[str, Any] = {"n_judged": len(js), "n_judge_errors": sum(1 for r in rows if r.get("judge_error"))}
        for k in ("faithfulness", "citation_precision", "citation_recall", "answer_relevance",
                  "context_precision", "context_recall"):
            jd[k] = _ci([x[k] for x in js if x.get(k) is not None], cfg)
        n_lab = sum(x["n_support_labels"] for x in js)
        jd["overruled_rate"] = sum(x["overruled"] for x in js) / n_lab if n_lab else None
        out["judge"] = jd
    out["abstention"] = M.abstention_scores([r["abstained"] for r in rows], [r["answerable"] for r in rows])
    out["abstention"]["abstain_rate"] = sum(r["abstained"] for r in rows) / len(rows) if rows else None
    fresh = [r["cost"]["total_ms"] / 1000 for r in rows
             if not r["cost"]["gen_cache_hit"] and not r.get("generation_error")]
    out["cost"] = {
        "llm_calls_per_query": _ci([r["cost"]["llm_calls"] for r in rows], cfg)["mean"],
        "tokens_per_query": _ci([r["cost"]["prompt_tokens"] + r["cost"]["completion_tokens"] for r in rows], cfg)["mean"],
        "prompt_tokens_per_query": _ci([r["cost"]["prompt_tokens"] for r in rows], cfg)["mean"],
        "latency_s_p50": M.percentile(fresh, 50), "latency_s_p95": M.percentile(fresh, 95),
        "n_latency": len(fresh),
        "pct_routed_without_llm": None,  # no router before M6
    }
    hops = sorted({r["meta"].get("hops") for r in rows if r["meta"].get("hops")})
    if len(hops) > 1:
        out["by_hops"] = {str(h): {"n": sum(1 for r in rows if r["meta"].get("hops") == h),
                                   "f1": _ci([r["scores"]["f1"] for r in rows if r["meta"].get("hops") == h], cfg)["mean"]}
                          for h in hops}
    return out


def _failed_result(err: str) -> Dict[str, Any]:
    """Stand-in for a generation that produced no usable answer. Not an abstention: it scores 0 everywhere."""
    class _T:
        llm_calls: List[Dict[str, Any]] = [{"kind": "generate", "cache_hit": False}]
    return {"answer": "", "short_answer": "", "abstained": False, "citations": [], "invalid_citations": [],
            "uncited_sentences": [], "evidence": [], "trace_id": None, "_trace": _T(),
            "totals": {"llm_calls": 1, "llm_calls_uncached": 1, "embed_calls": 1, "prompt_tokens": 0,
                       "completion_tokens": 0, "total_ms": 0.0}, "generation_error": err}


# ---- run -------------------------------------------------------------------------
def _ollama_meta(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = cfg["ollama"]["url"].rstrip("/")
    tags = requests.get(base + "/api/tags", timeout=10).json().get("models", [])
    digests = {m["name"]: m.get("digest") for m in tags}
    want = {role: cfg["models"][role] for role in ("generator", "embedder", "judge")}

    def dig(name):
        return digests.get(name) or digests.get(name + ":latest")
    return {"version": requests.get(base + "/api/version", timeout=10).json().get("version"),
            "backend": cfg["ollama"].get("backend"),
            "models": {role: {"name": n, "digest": dig(n)} for role, n in want.items()}}


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def run_eval(config_path: Optional[str], datasets: List[str], split: str = "test", run_id: Optional[str] = None,
             limit: Optional[int] = None, judge: bool = True, log: Callable[[str], None] = print,
             client=None) -> Path:
    base_cfg = load_config(config_path)
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M}_{base_cfg['run']['name']}_{split}"
    run_dir = resolve_path(base_cfg["eval"]["results_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run.json"
    meta = {"run_id": run_id, "created": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(),
            "config_path": base_cfg["_meta"]["config_path"], "config_hash": base_cfg["_meta"]["config_hash"],
            "split": split, "datasets": datasets, "limit": limit,
            "ollama": _ollama_meta(base_cfg) if client is None else {"version": "fake"},
            "config": {k: v for k, v in base_cfg.items() if k != "_meta"}}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old["config_hash"] != meta["config_hash"]:
            raise ValueError(f"{run_dir} was started with config {old['config_hash']}, not {meta['config_hash']}")
        meta["created"] = old["created"]
        meta["datasets"] = sorted(set(old["datasets"]) | set(datasets))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    max_k = max(base_cfg["eval"]["retrieval_ks"])
    for name in datasets:
        cfg = load_config(config_path, overrides={**D.bench_overrides(name), "trace": {"dir": str(run_dir / "traces")}})
        comps = Components(cfg, client=client)
        if comps.store.count() == 0:
            raise RuntimeError(f"Collection {cfg['store']['collection']} is empty: run `python -m apex.eval.suite index {name}`")
        pipe = PIPELINES[cfg.get("pipeline", "naive")](comps)
        jdg = Judge(cfg, comps.client) if judge and cfg["eval"]["judge_enabled"] else None
        items = D.load_split(name, split, cfg)[:limit]
        out_dir = run_dir / name
        out_dir.mkdir(exist_ok=True)
        rows_path = out_dir / "per_query.jsonl"
        done = {r["qid"] for r in _read_rows(rows_path)}
        log(f"[{name}/{split}] {len(items)} questions, {len(done)} already done → {rows_path}")
        with open(rows_path, "a", encoding="utf-8") as f:
            for i, it in enumerate(items, 1):
                if it.qid in done:
                    continue
                t0 = time.perf_counter()
                q = it.prompt()
                try:
                    res = pipe.answer(q, source=it.scope)
                except GenerationError as e:  # a real system failure on this question: scored 0 and reported
                    res = _failed_result(str(e))
                    log(f"  !! {it.qid}: generation failed: {str(e)[:160]}")
                hits = pipe.retrieve_ranked(q, k=max_k, source=it.scope)
                gen_call = next((c for c in res["_trace"].llm_calls if c["kind"] == "generate"), {})
                row = {
                    "qid": it.qid, "dataset": name, "question": q, "answers": it.answers, "answerable": it.answerable,
                    "meta": it.meta, "answer": res["answer"], "short_answer": res.get("short_answer"),
                    "abstained": res["abstained"],
                    "citations": res["citations"], "invalid_citations": res["invalid_citations"],
                    "n_uncited_sentences": len(res["uncited_sentences"]),
                    "evidence": [{"cite_id": e["cite_id"], "node_id": e["node_id"], "source": e["source"],
                                  "section": e["section"], "text": e["text"], "score": e["score"]} for e in res["evidence"]],
                    "retrieved_ids": [h.node.id for h in hits],
                    "retrieval": retrieval_scores([h.node.text for h in hits], it.evidence_sets, cfg),
                    "scores": answer_scores(it, res["answer"], res.get("short_answer"), res["abstained"],
                                            [e["text"] for e in res["evidence"]]),
                    "gold_for_judge": gold_for_judge(it),
                    "cost": {**res["totals"], "gen_cache_hit": bool(gen_call.get("cache_hit"))},
                    "trace_id": res["trace_id"],
                }
                if res.get("generation_error"):
                    row["generation_error"] = res["generation_error"]
                if jdg:
                    row.update(judge_row(jdg, row))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                s = row["scores"]
                log(f"  {i:3d}/{len(items)} {it.qid[:40]:40s} {time.perf_counter() - t0:5.1f}s "
                    f"em={s['em']:.0f} f1={s['f1']:.2f} acc={s['acc']:.0f} abst={row['abstained']!s:5s}"
                    + (f" R@10={row['retrieval']['recall@10']:.2f}" if row["retrieval"] else ""))
        finalize(run_dir, name, cfg)
    write_summary(run_dir)
    return run_dir


def finalize(run_dir: Path, name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    rows = _read_rows(run_dir / name / "per_query.jsonl")
    m = aggregate(rows, cfg)
    (run_dir / name / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def judge_run(run_dir: Path, datasets: Optional[List[str]] = None, log: Callable[[str], None] = print,
              client=None) -> None:
    """Fill judge scores into an existing run's rows that lack them (rewrites per_query.jsonl atomically)."""
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for name in datasets or meta["datasets"]:
        cfg = load_config(meta["config_path"], overrides=D.bench_overrides(name))
        comps = Components(cfg, client=client)
        jdg = Judge(cfg, comps.client)
        path = run_dir / name / "per_query.jsonl"
        rows = _read_rows(path)
        todo = [r for r in rows if not r.get("judged")]
        log(f"[{name}] judging {len(todo)} of {len(rows)} rows with {jdg.model}")
        for i, r in enumerate(todo, 1):
            r.update(judge_row(jdg, r))
            if i % 10 == 0 or i == len(todo):
                tmp = path.with_suffix(".jsonl.tmp")
                tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
                tmp.replace(path)
                log(f"  {i}/{len(todo)}")
        finalize(run_dir, name, cfg)
    write_summary(run_dir)


def write_summary(run_dir: Path) -> None:
    from apex.eval.report import markdown_table

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    allm = {}
    for name in meta["datasets"]:
        p = run_dir / name / "metrics.json"
        if p.exists():
            allm[name] = json.loads(p.read_text(encoding="utf-8"))
    (run_dir / "metrics.json").write_text(json.dumps({"run": {k: v for k, v in meta.items() if k != "config"},
                                                      "datasets": allm}, indent=2), encoding="utf-8")
    (run_dir / "table.md").write_text(markdown_table(meta, allm), encoding="utf-8")

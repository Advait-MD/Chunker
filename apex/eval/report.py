"""Markdown tables for a run, and paired comparisons between two runs (spec §2.8 rigour)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from apex.eval import metrics as M


def _fmt(ci: Optional[Dict[str, Any]], scale: float = 100.0) -> str:
    if not ci or ci.get("mean") is None:
        return "—"
    return f"{ci['mean'] * scale:.1f} ({ci['lo'] * scale:.1f}–{ci['hi'] * scale:.1f})"


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _num(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def markdown_table(meta: Dict[str, Any], allm: Dict[str, Dict[str, Any]]) -> str:
    o = meta.get("ollama", {})
    gen = o.get("models", {}).get("generator", {}).get("name", meta.get("config", {}).get("models", {}).get("generator"))
    lines = [
        f"# Run `{meta['run_id']}`",
        "",
        f"split **{meta['split']}** · git `{meta['git_sha']}` · config `{meta['config_hash']}` · generator `{gen}` · "
        f"Ollama {o.get('version')} ({o.get('backend')})",
        "",
        "Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the "
        "answer \"Unanswerable\". Retrieval metrics cover answerable questions with gold evidence.",
        "",
        "## Answers",
        "",
        "EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).",
        "",
        "| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in allm.items():
        a = m["answer"]
        lines.append(f"| {name} | {m['n']} | {_fmt(a['em'])} | {_fmt(a['f1'])} | {_fmt(a['acc'])} | "
                     f"{_fmt(a['f1_answerable'])} | {_fmt(a.get('f1_long'))} | {_fmt(a.get('mc_acc'))} | "
                     f"{m.get('n_generation_errors', 0)} |")
    lines += ["", "## Retrieval", "",
              "Recall/MRR/nDCG: top-20 dense retrieval against gold evidence passages. *Answer in context*: a short "
              "gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).", "",
              "| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in allm.items():
        r, aic = m.get("retrieval"), m.get("answer_in_context")
        if r or aic:
            r = r or {}
            nr = r["recall@5"]["n"] if r else 0
            lines.append(f"| {name} | {nr} | {_fmt(r.get('recall@5'))} | {_fmt(r.get('recall@10'))} | "
                         f"{_fmt(r.get('recall@20'))} | {_fmt(r.get('mrr'))} | {_fmt(r.get('ndcg@10'))} | "
                         f"{_fmt(aic)} ({aic['n'] if aic else 0}) |")
    if any("judge" in m for m in allm.values()):
        lines += ["", "## Judge (faithfulness, citations, context)", "",
                  "| Dataset | Faithfulness | Citation precision | Citation recall | Answer relevance | "
                  "Context precision | Context recall | Overruled labels | Judge errors |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, m in allm.items():
            j = m.get("judge")
            if j:
                lines.append(f"| {name} | {_fmt(j['faithfulness'])} | {_fmt(j['citation_precision'])} | "
                             f"{_fmt(j['citation_recall'])} | {_fmt(j['answer_relevance'])} | "
                             f"{_fmt(j['context_precision'])} | {_fmt(j['context_recall'])} | {_pct(j.get('overruled_rate'))} | {j.get('n_judge_errors', 0)}/{j['n_judged'] + j.get('n_judge_errors', 0)} |")
    lines += ["", "## Abstention and cost", "",
              "| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | "
              "Tokens/q | Latency p50 | Latency p95 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in allm.items():
        ab, c = m["abstention"], m["cost"]
        lines.append(f"| {name} | {ab['n_unanswerable']} | {_pct(ab['abstain_precision'])} | {_pct(ab['abstain_recall'])} | "
                     f"{_pct(ab['false_abstain_rate'])} | {_num(c['llm_calls_per_query'], 2)} | "
                     f"{_num(c['tokens_per_query'], 0)} | {_num(c['latency_s_p50'])} s | {_num(c['latency_s_p95'])} s |")
    by_h = {n: m["by_hops"] for n, m in allm.items() if m.get("by_hops")}
    if by_h:
        lines += ["", "## F1 by hop count", ""]
        for n, h in by_h.items():
            lines.append(f"- **{n}:** " + ", ".join(f"{k}-hop {100 * v['f1']:.1f} (n={v['n']})" for k, v in h.items()))
    return "\n".join(lines) + "\n"


# ---- paired comparison -------------------------------------------------------------
PAIRED = {
    "f1": lambda r: r["scores"]["f1"],
    "answer_in_context": lambda r: r["scores"].get("answer_in_context"),
    "em": lambda r: r["scores"]["em"],
    "acc": lambda r: r["scores"]["acc"],
    "mc_acc": lambda r: r["scores"].get("mc_acc"),
    "recall@10": lambda r: (r.get("retrieval") or {}).get("recall@10"),
    "ndcg@10": lambda r: (r.get("retrieval") or {}).get("ndcg@10"),
    "faithfulness": lambda r: (r.get("judge") or {}).get("faithfulness"),
    "citation_recall": lambda r: (r.get("judge") or {}).get("citation_recall"),
    "prompt_tokens": lambda r: r["cost"]["prompt_tokens"],
}


def compare(run_a: Path, run_b: Path, datasets: Optional[List[str]] = None) -> str:
    """Paired test of run A (candidate) against run B (baseline) on shared question ids."""
    meta_a = json.loads((run_a / "run.json").read_text(encoding="utf-8"))
    meta_b = json.loads((run_b / "run.json").read_text(encoding="utf-8"))
    names = datasets or [d for d in meta_a["datasets"] if d in meta_b["datasets"]]
    lines = [f"# `{meta_a['run_id']}` vs baseline `{meta_b['run_id']}`", "",
             "Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, "
             "a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.", "",
             "| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        ra = {json.loads(l)["qid"]: json.loads(l) for l in open(run_a / name / "per_query.jsonl", encoding="utf-8")}
        rb = {json.loads(l)["qid"]: json.loads(l) for l in open(run_b / name / "per_query.jsonl", encoding="utf-8")}
        shared = sorted(set(ra) & set(rb))
        for metric, get in PAIRED.items():
            pairs = [(get(ra[q]), get(rb[q])) for q in shared]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 2:
                continue
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
            t = M.paired_test(xs, ys)
            s = 1.0 if metric == "prompt_tokens" else 100.0
            lines.append(f"| {name} | {metric} | {t['n']} | {s * sum(xs) / len(xs):.1f} | {s * sum(ys) / len(ys):.1f} | "
                         f"{s * t['mean_diff']:+.1f} ({s * t['lo']:+.1f}, {s * t['hi']:+.1f}) | "
                         f"{t['p_bootstrap']:.3f} | {t['p_wilcoxon']:.3f} |")
    return "\n".join(lines) + "\n"

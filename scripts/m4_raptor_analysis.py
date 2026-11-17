"""Why RAPTOR did not help (M4): trace the tree's effect through every pipeline stage.

Reads only finished artifacts (per-query results and the LLM call log), so it is safe to run beside other jobs:
  python scripts/m4_raptor_analysis.py --out results/m4_raptor_analysis

Runs compared (same questions, same generator, temperature 0):
  row 2  = M3 hybrid (no tree)            row 3 = row 2 + RAPTOR collapsed
  M3 row 4 = row 2 + rerank (no tree)     row 4 = row 3 + rerank
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS = {"row2": "m3_row2_hybrid_test100", "row3": "m4_row3_raptor_test100",
        "m3row4": "m3_row4_rerank_test100", "row4": "m4_row4_raptor_rerank_test100"}
DATASETS = ["qasper", "quality", "custom", "musique", "hotpotqa", "2wiki"]
KIND = {"qasper": "long-doc", "quality": "long-doc", "custom": "long-doc",
        "musique": "multi-hop", "hotpotqa": "multi-hop", "2wiki": "multi-hop"}
LEVEL = re.compile(r":L(\d+):")


def level(node_id: str) -> int:
    m = LEVEL.search(node_id)
    return int(m.group(1)) if m else 0


def load(run: str, d: str):
    p = Path("results") / run / d / "per_query.jsonl"
    return {r["qid"]: r for r in map(json.loads, p.open(encoding="utf-8"))}


def mean(xs):
    xs = list(xs)
    return 100 * st.mean(xs) if xs else float("nan")


def pct(a, b):
    return 100 * a / b if b else float("nan")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def answer_in(texts, answers) -> bool:
    t = norm(" ".join(texts))
    return any(a and norm(a) and norm(a) in t for a in answers if isinstance(a, str) and a != "Unanswerable")


def stage_index(log_path: Path):
    """Summary calls from the tree builds (13-15 Nov, seed 42): compression and length caps."""
    rows = []
    for line in log_path.open(encoding="utf-8"):
        if '"raptor.summarize"' not in line:
            continue
        r = json.loads(line)
        if r.get("cache_hit") or not ("2026-11-13" <= r["ts"][:10] <= "2026-11-15"):
            continue
        if r.get("model") != "qwen2.5:3b":
            continue
        rows.append(r)
    if not rows:
        return {}
    pt = [r["prompt_tokens"] for r in rows]
    ct = [r["completion_tokens"] for r in rows]
    with_reason = [r for r in rows if r.get("done_reason")]
    return {"calls": len(rows), "prompt_tokens_median": st.median(pt), "summary_tokens_median": st.median(ct),
            "compression_median": st.median(p / max(c, 1) for p, c in zip(pt, ct)),
            "length_capped_pct": pct(sum(r["done_reason"] == "length" for r in with_reason), len(with_reason)),
            "latency_s_median": st.median(r["latency_ms"] for r in rows) / 1000}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m4_raptor_analysis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = {k: {d: load(v, d) for d in DATASETS} for k, v in RUNS.items()}
    res = {"index": stage_index(Path("logs/llm_calls.jsonl")), "datasets": {}}
    examples = []

    for d in DATASETS:
        r2, r3, m4, r4 = (R[k][d] for k in ("row2", "row3", "m3row4", "row4"))
        q = [k for k in r3 if k in r2 and k in m4 and k in r4]
        D = res["datasets"][d] = {"n": len(q), "kind": KIND[d]}

        # --- retrieval: what the tree puts in the top 20 / top 5
        top20 = Counter(level(i) for k in q for i in r3[k]["retrieved_ids"][:20])
        D["top20_summary_share"] = pct(sum(v for lv, v in top20.items() if lv > 0), sum(top20.values()))
        D["top20_by_level"] = dict(sorted(top20.items()))
        first = [next((i for i, x in enumerate(r3[k]["retrieved_ids"]) if level(x) > 0), None) for k in q]
        D["first_summary_rank_median"] = st.median([f + 1 for f in first if f is not None]) if any(
            f is not None for f in first) else None
        D["q_summary_rank1_pct"] = pct(sum(f == 0 for f in first), len(q))
        rec = [k for k in q if r3[k]["retrieval"] and r2[k]["retrieval"]]
        D["recall5_row2"] = mean(r2[k]["retrieval"]["recall@5"] for k in rec)
        D["recall5_row3"] = mean(r3[k]["retrieval"]["recall@5"] for k in rec)
        D["q_lost_gold_in_top5_pct"] = pct(sum(r3[k]["retrieval"]["recall@5"] < r2[k]["retrieval"]["recall@5"]
                                               for k in rec), len(rec))
        D["q_gained_gold_in_top5_pct"] = pct(sum(r3[k]["retrieval"]["recall@5"] > r2[k]["retrieval"]["recall@5"]
                                                 for k in rec), len(rec))

        # --- context: the five passages the generator reads
        for name, run in (("row3", r3), ("row4", r4)):
            ev = [level(e["node_id"]) for k in q for e in run[k]["evidence"]]
            D[f"{name}_ctx_summary_share"] = pct(sum(lv > 0 for lv in ev), len(ev))
            D[f"{name}_ctx_by_level"] = dict(sorted(Counter(ev).items()))
        D["row4_vs_pool_summary_share"] = (D["row4_ctx_summary_share"], D["top20_summary_share"])
        D["ctx_tokens_row2"] = st.mean(r2[k]["cost"]["prompt_tokens"] for k in q)
        D["ctx_tokens_row3"] = st.mean(r3[k]["cost"]["prompt_tokens"] for k in q)

        # answer string present in the packed context: leaves vs summaries
        def ans_where(run, k):
            ev = run[k]["evidence"]
            ans = run[k]["answers"]
            leaf = answer_in([e["text"] for e in ev if level(e["node_id"]) == 0], ans)
            summ = answer_in([e["text"] for e in ev if level(e["node_id"]) > 0], ans)
            return leaf, summ
        aw2 = [answer_in([e["text"] for e in r2[k]["evidence"]], r2[k]["answers"]) for k in q]
        aw3 = [ans_where(r3, k) for k in q]
        D["answer_in_ctx_row2_pct"] = pct(sum(aw2), len(q))
        D["answer_in_ctx_row3_pct"] = pct(sum(l or s for l, s in aw3), len(q))
        D["answer_only_in_summary_row3_pct"] = pct(sum(s and not l for l, s in aw3), len(q))
        D["answer_lost_from_ctx_row3_pct"] = pct(sum(x and not (l or s) for x, (l, s) in zip(aw2, aw3)), len(q))

        # --- answer: paired F1 by how many summaries were packed
        f = lambda run, k: run[k]["scores"]["f1"]
        groups = defaultdict(list)
        for k in q:
            ns = sum(level(e["node_id"]) > 0 for e in r3[k]["evidence"])
            groups["0" if ns == 0 else "1" if ns == 1 else "2" if ns == 2 else "3+"].append(f(r3, k) - f(r2, k))
        D["dF1_row3_vs_row2_by_n_summaries"] = {g: (len(v), 100 * st.mean(v)) for g, v in sorted(groups.items())}
        maxlv = defaultdict(list)
        for k in q:
            m = max(level(e["node_id"]) for e in r3[k]["evidence"]) if r3[k]["evidence"] else 0
            maxlv["L0 only" if m == 0 else "L1" if m == 1 else "L2+"].append(f(r3, k) - f(r2, k))
        D["dF1_row3_vs_row2_by_top_level"] = {g: (len(v), 100 * st.mean(v)) for g, v in sorted(maxlv.items())}
        D["row3_vs_row2_win_loss_tie"] = (sum(f(r3, k) > f(r2, k) for k in q), sum(f(r3, k) < f(r2, k) for k in q),
                                          sum(f(r3, k) == f(r2, k) for k in q))
        D["row4_vs_m3row4_win_loss_tie"] = (sum(f(r4, k) > f(m4, k) for k in q), sum(f(r4, k) < f(m4, k) for k in q),
                                            sum(f(r4, k) == f(m4, k) for k in q))

        # --- citations
        for name, run in (("row3", r3), ("row4", r4)):
            cites = [c for k in q for c in run[k]["citations"]]
            D[f"{name}_cites_to_summary_pct"] = pct(sum(level(c["node_id"]) > 0 for c in cites), len(cites))
            D[f"{name}_cites_without_page_pct"] = pct(sum(c.get("page") is None and c.get("char_span") is None
                                                          for c in cites), len(cites))
            cs = [k for k in q if any(level(c["node_id"]) > 0 for c in run[k]["citations"])]
            ncs = [k for k in q if k not in set(cs)]
            D[f"{name}_f1_when_citing_summary"] = (len(cs), mean(f(run, k) for k in cs))
            D[f"{name}_f1_when_not"] = (len(ncs), mean(f(run, k) for k in ncs))
        D["abstain_rate_row2_row3"] = (pct(sum(r2[k]["abstained"] for k in q), len(q)),
                                       pct(sum(r3[k]["abstained"] for k in q), len(q)))

        # --- question types
        tg = defaultdict(list)
        for k in q:
            m = r3[k].get("meta") or {}
            if "hops" in m:
                tg[f"{m['hops']}-hop"].append(f(r3, k) - f(r2, k))
            if "type" in m:
                tg[f"type={m['type']}"].append(f(r3, k) - f(r2, k))
            if "hard" in m:
                tg["hard" if m["hard"] else "not hard"].append(f(r3, k) - f(r2, k))
        if tg:
            D["dF1_row3_vs_row2_by_question_type"] = {g: (len(v), 100 * st.mean(v)) for g, v in sorted(tg.items())}

        # --- QuALITY multiple choice: which questions flipped
        if d == "quality":
            mc = lambda run, k: run[k]["scores"].get("mc_acc")
            flips = [(k, mc(m4, k), mc(r4, k)) for k in q if mc(m4, k) is not None and mc(m4, k) != mc(r4, k)]
            lost = [k for k, a_, b_ in flips if a_ > b_]
            gained = [k for k, a_, b_ in flips if b_ > a_]
            D["mc_flips_m3row4_to_row4"] = {"lost": len(lost), "gained": len(gained),
                                             "lost_with_summary_ctx": sum(any(level(e["node_id"]) > 0
                                                                              for e in r4[k]["evidence"]) for k in lost),
                                             "lost_mean_summaries_ctx": st.mean(
                                                 sum(level(e["node_id"]) > 0 for e in r4[k]["evidence"]) for k in lost)
                                             if lost else None}

        # --- examples: tree lost the answer and the answer cited a summary
        for k in q:
            if f(r2, k) >= 0.99 and f(r3, k) == 0 and any(level(c["node_id"]) > 0 for c in r3[k]["citations"]):
                s = next(e for e in r3[k]["evidence"]
                         if level(e["node_id"]) > 0 and any(c["node_id"] == e["node_id"] for c in r3[k]["citations"]))
                examples.append({"dataset": d, "qid": k, "question": r3[k]["question"], "gold": r3[k]["answers"][:2],
                                 "row2_answer": r2[k]["short_answer"], "row3_answer": r3[k]["short_answer"],
                                 "cited_summary_level": level(s["node_id"]), "cited_summary": s["text"][:600]})

    res["examples"] = examples[:12]
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    L = ["# Why RAPTOR did not help: per-stage trace (M4)", "",
         "Test, first 100 questions per dataset plus custom 32; same questions in every run. "
         "Summary = a tree node above level 0. Answer-in-context is a normalised substring match of a gold answer.", ""]
    ix = res["index"]
    if ix:
        L += ["## Index (seed-42 tree builds, qwen2.5:3b)", "",
              f"{ix['calls']} summary calls; median prompt {ix['prompt_tokens_median']:.0f} tokens → median summary "
              f"{ix['summary_tokens_median']:.0f} tokens (compression {ix['compression_median']:.1f}×); "
              f"{ix['length_capped_pct']:.0f}% hit the 256-token cap (of calls with done_reason logged); "
              f"median {ix['latency_s_median']:.1f} s per summary.", ""]
    hdr = lambda cols: ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    ds = res["datasets"]
    L += ["## Retrieval (row 3: hybrid over leaves + summaries)", ""] + hdr(
        ["Dataset", "Summaries in top 20", "Summary at rank 1", "Recall@5 row 2 → row 3",
         "Questions losing gold from top 5", "gaining"])
    for d in DATASETS:
        x = ds[d]
        L.append(f"| {d} | {x['top20_summary_share']:.0f}% | {x['q_summary_rank1_pct']:.0f}% | "
                 f"{x['recall5_row2']:.1f} → {x['recall5_row3']:.1f} | {x['q_lost_gold_in_top5_pct']:.0f}% | "
                 f"{x['q_gained_gold_in_top5_pct']:.0f}% |")
    L += ["", "## Context (the 5 passages the generator reads)", ""] + hdr(
        ["Dataset", "Summaries in context, row 3", "row 4 (after rerank)", "Answer in context row 2 → row 3",
         "Answer only in a summary", "Answer pushed out", "Prompt tokens row 2 → row 3"])
    for d in DATASETS:
        x = ds[d]
        L.append(f"| {d} | {x['row3_ctx_summary_share']:.0f}% | {x['row4_ctx_summary_share']:.0f}% | "
                 f"{x['answer_in_ctx_row2_pct']:.0f}% → {x['answer_in_ctx_row3_pct']:.0f}% | "
                 f"{x['answer_only_in_summary_row3_pct']:.0f}% | {x['answer_lost_from_ctx_row3_pct']:.0f}% | "
                 f"{x['ctx_tokens_row2']:.0f} → {x['ctx_tokens_row3']:.0f} |")
    L += ["", "## Answers: F1 change row 3 − row 2 by how many summaries were packed (n, points)", ""] + hdr(
        ["Dataset", "0", "1", "2", "3+", "Win / loss / tie", "Row 4 vs M3 row 4: win / loss / tie"])
    for d in DATASETS:
        x = ds[d]
        g = x["dF1_row3_vs_row2_by_n_summaries"]
        cell = lambda k: f"{g[k][1]:+.1f} (n={g[k][0]})" if k in g else "—"
        L.append(f"| {d} | {cell('0')} | {cell('1')} | {cell('2')} | {cell('3+')} | "
                 f"{'/'.join(map(str, x['row3_vs_row2_win_loss_tie']))} | "
                 f"{'/'.join(map(str, x['row4_vs_m3row4_win_loss_tie']))} |")
    L += ["", "By the highest tree level packed (n, points):", ""]
    for d in DATASETS:
        g = ds[d]["dF1_row3_vs_row2_by_top_level"]
        L.append(f"- {d}: " + "; ".join(f"{k} {v[1]:+.1f} (n={v[0]})" for k, v in g.items()))
    L += ["", "## Citations", ""] + hdr(
        ["Dataset", "Cites to a summary, row 3 / row 4", "Cites with no page or span, row 3",
         "F1 when citing a summary (n)", "F1 otherwise (n)"])
    for d in DATASETS:
        x = ds[d]
        a_, b_ = x["row3_f1_when_citing_summary"], x["row3_f1_when_not"]
        L.append(f"| {d} | {x['row3_cites_to_summary_pct']:.0f}% / {x['row4_cites_to_summary_pct']:.0f}% | "
                 f"{x['row3_cites_without_page_pct']:.0f}% | {a_[1]:.1f} ({a_[0]}) | {b_[1]:.1f} ({b_[0]}) |")
    L += ["", "## Question types: F1 change row 3 − row 2 (n, points)", ""]
    for d in DATASETS:
        g = ds[d].get("dF1_row3_vs_row2_by_question_type")
        if g:
            L.append(f"- {d}: " + "; ".join(f"{k} {v[1]:+.1f} (n={v[0]})" for k, v in g.items()))
    mcf = ds["quality"].get("mc_flips_m3row4_to_row4")
    if mcf:
        L += ["", "## QuALITY multiple choice, M3 row 4 → row 4", "",
              f"{mcf['lost']} questions flipped right → wrong, {mcf['gained']} wrong → right; "
              f"{mcf['lost_with_summary_ctx']} of the {mcf['lost']} losses had a summary in context "
              f"(mean {mcf['lost_mean_summaries_ctx']:.1f} of 5 passages)."]
    L += ["", "## Examples: right without the tree, wrong with it, citing a summary", ""]
    for e in res["examples"][:6]:
        L += [f"**{e['dataset']} {e['qid']}**: {e['question']}",
              f"- gold {e['gold']}; row 2 said {e['row2_answer']!r}; row 3 said {e['row3_answer']!r}",
              f"- cited L{e['cited_summary_level']} summary: \"{e['cited_summary']}\"", ""]
    (out / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

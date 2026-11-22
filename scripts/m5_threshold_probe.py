"""M5 threshold probe (dev only): the relevance threshold θ_rel and the sufficiency bands, from
results/m5_dev_scores (row 4's reranked top 20 per dev question, with gold matches).

θ_rel, on the top `--pool` passages (at least `min_keep` = 2 kept):
  passages kept, token share kept, gold kept (share of gold-covering passages in the pool that survive),
  non-gold dropped (share of the other passages removed) — precision of what reaches the generator.
Sufficiency, on the best reranker score: its distribution for answerable questions with gold in the pool,
answerable without gold in the pool, and unanswerable questions; and per candidate band the share of each
group sent to "insufficient".
Writes results/m5_threshold_probe/{probe.json,probe.md}.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REL = [None, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
SUFF = [-8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--min-keep", type=int, default=2)
    ap.add_argument("--src", default="results/m5_dev_scores")
    ap.add_argument("--out", default="results/m5_threshold_probe")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, L = {}, [f"# M5 threshold probe (dev, top {a.pool} of row 4's reranked list)", ""]
    allrows = {}
    for p in sorted(Path(a.src).glob("*.jsonl")):
        allrows[p.stem] = [json.loads(l) for l in p.open(encoding="utf-8")]
    allrows["all"] = [r for rs in list(allrows.values()) for r in rs]
    L += ["## Relevance threshold θ_rel", "",
          "Per cell: passages kept per question / token share kept / gold passages kept / non-gold passages dropped.", "",
          "| θ_rel | " + " | ".join(f"{n} (n={len(rs)})" for n, rs in allrows.items()) + " |",
          "| --- |" + " --- |" * len(allrows)]
    for th in REL:
        cells = []
        for n, rs in allrows.items():
            kept = tok_in = tok_out = g_in = g_out = o_in = o_drop = 0
            for r in rs:
                pool = list(zip(r["scores"], r["gold"], r["tokens"]))[:a.pool]
                keep = [x for x in pool if th is None or x[0] >= th]
                if len(keep) < a.min_keep:
                    keep = pool[:a.min_keep]
                kept += len(keep)
                tok_in += sum(x[2] for x in pool); tok_out += sum(x[2] for x in keep)
                g_in += sum(x[1] for x in pool); g_out += sum(x[1] for x in keep)
                o_in += sum(not x[1] for x in pool); o_drop += sum(not x[1] for x in pool) - sum(not x[1] for x in keep)
            res.setdefault(n, {})[f"rel{th}"] = {"kept": kept / len(rs), "token_share": tok_out / tok_in,
                                                 "gold_kept": g_out / g_in if g_in else None,
                                                 "other_dropped": o_drop / o_in if o_in else None}
            c = res[n][f"rel{th}"]
            cells.append(f"{c['kept']:.1f} / {100 * c['token_share']:.0f}% / "
                         + ("—" if c["gold_kept"] is None else f"{100 * c['gold_kept']:.0f}%") + " / "
                         + ("—" if c["other_dropped"] is None else f"{100 * c['other_dropped']:.0f}%"))
        L.append(f"| {th if th is not None else 'off'} | " + " | ".join(cells) + " |")
    L += ["", "## Best reranker score in the pool (sufficiency signal)", "",
          "| Dataset | group | n | p10 | median | p90 |", "| --- | --- | --- | --- | --- | --- |"]
    groups = {}
    for n, rs in allrows.items():
        g = {"answerable, gold in pool": [], "answerable, no gold in pool": [], "unanswerable": []}
        for r in rs:
            best = max(r["scores"][:a.pool]) if r["scores"] else -99
            if not r["answerable"]:
                g["unanswerable"].append(best)
            elif r["has_gold"]:
                g["answerable, gold in pool" if any(r["gold"][:a.pool]) else "answerable, no gold in pool"].append(best)
        groups[n] = g
        for k, xs in g.items():
            if xs:
                L.append(f"| {n} | {k} | {len(xs)} | {q(xs, .1):.2f} | {st.median(xs):.2f} | {q(xs, .9):.2f} |")
    L += ["", "## Share sent to *insufficient* by `insufficient_below`", "",
          "| Dataset | group | " + " | ".join(f"< {t}" for t in SUFF) + " |", "| --- | --- |" + " --- |" * len(SUFF)]
    for n, g in groups.items():
        for k, xs in g.items():
            if xs:
                L.append(f"| {n} | {k} | " + " | ".join(f"{100 * sum(x < t for x in xs) / len(xs):.0f}%" for t in SUFF) + " |")
                res[n][f"insufficient|{k}"] = {str(t): sum(x < t for x in xs) / len(xs) for t in SUFF}
    (out / "probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "probe.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

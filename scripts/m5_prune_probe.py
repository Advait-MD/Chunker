"""M5 sentence-pruning probe (dev only): tokens kept vs gold answer kept, over θ_sent and neighbours.

Reads results/m5_dev_scores/<dataset>.jsonl (row 4's reranked ranking) and takes each question's top
`--pool` passages, after the relevance threshold `--rel` (keeping at least 2). It prunes them exactly as the
filter does (apex.filter.evidence.select_sentences on nomic-embed-text query-sentence cosine) and reports per
setting:
  token share kept       tokens after pruning / tokens before
  answer kept            of the questions whose short gold answer occurs in the unpruned passages, the share
                         where it still occurs after pruning (QuALITY has option answers, so it is skipped)
Writes results/m5_prune_probe/{probe.json,probe.md}.

  PYTHONPATH=. python scripts/m5_prune_probe.py [--pool 5] [--rel none]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from apex.config import load_config
from apex.eval import datasets as D
from apex.eval import metrics as M
from apex.filter.evidence import join_kept, select_sentences
from apex.generate.citations import split_sentences
from apex.index.chroma_store import NodeStore
from apex.ingest.chunker import count_tokens
from apex.pipeline import Components

THRESHOLDS = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
NEIGHBORS = [0, 1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="qasper,musique,hotpotqa,2wiki")
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--rel", default="none")
    ap.add_argument("--out", default="results/m5_prune_probe")
    a = ap.parse_args()
    rel = None if a.rel == "none" else float(a.rel)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name in a.datasets.split(","):
        src = Path("results/m5_dev_scores") / f"{name}.jsonl"
        rows = [json.loads(l) for l in src.open(encoding="utf-8")]
        cfg = load_config("configs/ablations/row4_rerank.yaml", overrides=D.bench_overrides(name))
        comps = Components(cfg)
        nodes, _ = NodeStore(cfg).all_nodes({"level": {"$in": [0]}})
        by_id = {n.id: n for n in nodes}
        items = {it.qid: it for it in D.load_split(name, "dev", cfg)}
        acc = {(t, nb): {"tok_in": 0, "tok_out": 0, "ans_n": 0, "ans_kept": 0} for t in THRESHOLDS for nb in NEIGHBORS}
        for r in rows:
            it = items[r["qid"]]
            ids = [i for i, s in zip(r["ids"], r["scores"]) if rel is None or s >= rel][:a.pool] or r["ids"][:2]
            if len(ids) < 2:
                ids = r["ids"][:2]
            q = comps.embedder.embed_query(it.prompt(), caller="probe.prune_query")
            shorts = [x for x in it.answers if x != D.ABSTAIN_GOLD and 0 < len(M.normalize_answer(x).split()) <= 5
                      and M.normalize_answer(x) not in ("yes", "no")]
            for nid in ids:
                text = by_id[nid].text
                ss = split_sentences(text)
                sims = (comps.embedder.embed_documents(ss, caller="probe.prune_sentences") @ q).tolist() if len(ss) > 1 else [1.0]
                had = bool(shorts) and M.answer_contains(text, shorts)
                for (t, nb), x in acc.items():
                    pruned = text if len(ss) <= 1 else join_kept(ss, select_sentences(sims, t, nb))
                    x["tok_in"] += count_tokens(text)
                    x["tok_out"] += count_tokens(pruned)
                    if had:
                        x["ans_n"] += 1
                        x["ans_kept"] += M.answer_contains(pruned, shorts)
        res[name] = {f"{t}|{nb}": {"token_share": x["tok_out"] / x["tok_in"],
                                    "answer_kept": x["ans_kept"] / x["ans_n"] if x["ans_n"] else None, "answer_n": x["ans_n"]}
                     for (t, nb), x in acc.items()}
        print(f"[{name}] done ({len(rows)} questions)")
    (out / "probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    L = [f"# M5 sentence-pruning probe (dev, top {a.pool} of row 4, relevance threshold {a.rel})", "",
         "Each cell: token share kept / share of gold-answer passages that still contain the answer after pruning.", ""]
    for nb in NEIGHBORS:
        L += [f"## neighbours = {nb}", "", "| θ_sent | " + " | ".join(res) + " |", "| --- |" + " --- |" * len(res)]
        for t in THRESHOLDS:
            cells = []
            for name in res:
                c = res[name][f"{t}|{nb}"]
                cells.append(f"{100 * c['token_share']:.0f}% / " + ("—" if c["answer_kept"] is None else f"{100 * c['answer_kept']:.0f}% (n={c['answer_n']})"))
            L.append(f"| {t:.2f} | " + " | ".join(cells) + " |")
        L.append("")
    (out / "probe.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""M5 tuning data (dev only): reranker scores of the row 4 ranking, with gold-evidence matches.

For every dev question of each dataset, the row 4 retriever (hybrid weighted fusion + bge rerank, no tree)
ranks the fused top 20. Per hit we record the reranker logit, whether the hit covers a gold evidence passage
(the eval's shingle match), and whether a short gold answer occurs in it. Per question: answerable or not.
Writes results/m5_dev_scores/<dataset>.jsonl; resumable. No generation, so it costs reranker time only
(cached from M3/M4 where the same query-passage pairs were scored).

  PYTHONPATH=. python scripts/m5_dev_scores.py [--datasets qasper,quality,...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from apex.config import load_config
from apex.eval import datasets as D
from apex.eval import metrics as M
from apex.pipeline import Components, build_retriever

DATASETS = ["qasper", "quality", "musique", "hotpotqa", "2wiki"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--config", default="configs/ablations/row4_rerank.yaml")
    ap.add_argument("--out", default="results/m5_dev_scores")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.datasets.split(","):
        cfg = load_config(a.config, overrides=D.bench_overrides(name))
        em = cfg["eval"]["evidence_match"]
        comps = Components(cfg)
        ret = build_retriever(comps)
        path = out / f"{name}.jsonl"
        done = {json.loads(l)["qid"] for l in path.open(encoding="utf-8")} if path.exists() else set()
        items = D.load_split(name, "dev", cfg)
        t0 = time.perf_counter()
        with path.open("a", encoding="utf-8") as f:
            for i, it in enumerate(items, 1):
                if it.qid in done:
                    continue
                q = it.prompt()
                hits = ret.retrieve(q, k=int(cfg["retrieval"]["fused_k"]), source=it.scope)
                texts = [h.node.text for h in hits]
                gold = [False] * len(hits)
                for g in it.evidence_sets or []:
                    for j, row in enumerate(M.evidence_hits(texts, g, n=int(em["shingle"]), min_overlap=float(em["min_overlap"]))):
                        gold[j] = gold[j] or bool(row)
                shorts = [x for x in it.answers if x != D.ABSTAIN_GOLD and 0 < len(M.normalize_answer(x).split()) <= 5
                          and M.normalize_answer(x) not in ("yes", "no")]
                ans_in = [bool(shorts) and M.answer_contains(t, shorts) for t in texts] if it.options is None else None
                f.write(json.dumps({"qid": it.qid, "answerable": it.answerable, "has_gold": bool(it.evidence_sets),
                                    "scores": [round(h.score, 4) for h in hits], "gold": gold, "answer_in": ans_in,
                                    "tokens": [h.node.token_count for h in hits], "ids": [h.node.id for h in hits]}) + "\n")
                f.flush()
                if i % 10 == 0:
                    print(f"[{name}] {i}/{len(items)} {time.perf_counter() - t0:.0f}s rerank={ret.last.get('rerank')}")
        print(f"[{name}] done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()

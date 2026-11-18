"""3b vs 7b RAPTOR summarizer probe (M4): the same level-1 clusters summarised by qwen2.5:3b and qwen2.5:7b.

For a fixed random sample of level-1 summaries in each seed-42 tree, the probe rebuilds the exact summary prompt
from the stored children and calls both models through the tree builder (same prompt, temperature 0, same cap,
same trimming). The 3b call is a cache hit, so it reproduces the stored summary. It records per summary:
  latency, summary tokens, whether the 256-token cap was hit, and an "unsupported detail" count: numbers and
  capitalised names in the summary that appear nowhere in its children (a deterministic hallucination proxy; it
  does not catch wrong relations between names that do appear, which need a reader or M5's NLI model).
Writes results/m4_summarizer_probe/{pairs.jsonl,probe.md}.

  python scripts/m4_summarizer_probe.py --per-dataset 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

from apex.config import deep_merge, load_config
from apex.eval import datasets as D
from apex.index.chroma_store import NodeStore, tree_collection_name
from apex.index.raptor import TreeBuilder, TreeReport
from apex.pipeline import Components

DATASETS = ["qasper", "quality", "custom", "musique", "hotpotqa", "2wiki"]
NUM = re.compile(r"\b\d[\d,.]*%?")
NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*)")
STOP = {"The", "This", "These", "Those", "A", "An", "In", "On", "It", "They", "He", "She", "His", "Her", "Their",
        "Table", "Figure", "Section", "However", "Overall", "Summary"}


def unsupported(summary: str, source: str):
    src = re.sub(r"\s+", " ", source.lower())
    items = set(NUM.findall(summary)) | {m for m in NAME.findall(summary) if m.split()[0] not in STOP}
    bad = sorted(i for i in items if i.lower().rstrip(".,") not in src)
    return len(items), bad


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=10)
    ap.add_argument("--out", default="results/m4_summarizer_probe")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs.jsonl"
    done = {json.loads(l)["node_id"] for l in pairs_path.open(encoding="utf-8")} if pairs_path.exists() else set()

    for n in DATASETS:
        cfg = load_config(None, overrides=deep_merge(D.bench_overrides(n), {"raptor": {"enabled": False}}))
        tree = NodeStore(cfg, collection=tree_collection_name(cfg))
        nodes, _ = tree.all_nodes()
        by_id = {x.id: x for x in nodes}
        l1 = sorted((x for x in nodes if x.level == 1), key=lambda x: x.id)
        sample = random.Random(0).sample(l1, min(a.per_dataset, len(l1)))
        comps = Components(cfg)
        b3 = TreeBuilder(cfg, comps.store, comps.embedder, comps.client, log=lambda *_: None)
        b7 = TreeBuilder(deep_merge(cfg, {"raptor": {"summarizer": "qwen2.5:7b"}}), comps.store, comps.embedder,
                         comps.client, log=lambda *_: None)
        for s in sample:
            if s.id in done:
                continue
            kids = [by_id[c] for c in s.child_ids]
            texts = [k.text for k in kids]
            row = {"dataset": n, "node_id": s.id, "children": len(kids),
                   "child_tokens": sum(k.token_count for k in kids)}
            for tag, b in (("3b", b3), ("7b", b7)):
                rep = TreeReport(collection="probe", tree_hash="probe")
                t = time.perf_counter()
                text = b.summarize(texts, rep)
                checked, bad = unsupported(text, " ".join(texts))
                row[tag] = {"text": text, "latency_s": time.perf_counter() - t, "cache_hit": bool(rep.llm_cache_hits),
                            "capped": bool(rep.truncated_summaries), "words": len(text.split()),
                            "details_checked": checked, "unsupported": bad}
            row["3b_matches_stored"] = row["3b"]["text"] == s.text
            with pairs_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{n}] {s.id}: 3b {row['3b']['words']}w {len(row['3b']['unsupported'])} unsupported | "
                  f"7b {row['7b']['words']}w {row['7b']['latency_s']:.0f}s {len(row['7b']['unsupported'])} unsupported",
                  flush=True)

    rows = [json.loads(l) for l in pairs_path.open(encoding="utf-8")]
    L = ["# 3b vs 7b RAPTOR summarizer probe (M4)", "",
         f"{len(rows)} level-1 clusters from the seed-42 trees, summarised by both models with the same prompt, "
         "temperature 0, 256-token cap and trimming. *Unsupported details*: numbers and capitalised names in the "
         "summary that never appear in its children (a deterministic proxy; wrong relations between names that do "
         "appear are not counted).", "",
         "| Dataset | n | Words 3b / 7b | Capped 3b / 7b | Unsupported details per summary 3b / 7b | "
         "Summaries with any, 3b / 7b | 7b s per summary |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for n in DATASETS + ["all"]:
        g = [r for r in rows if n == "all" or r["dataset"] == n]
        if not g:
            continue
        f = lambda m, k: st.mean(r[m][k] for r in g)
        u = lambda m: st.mean(len(r[m]["unsupported"]) for r in g)
        anyu = lambda m: 100 * st.mean(bool(r[m]["unsupported"]) for r in g)
        cap = lambda m: 100 * st.mean(r[m]["capped"] for r in g)
        lat = st.median(r["7b"]["latency_s"] for r in g if not r["7b"]["cache_hit"]) if any(
            not r["7b"]["cache_hit"] for r in g) else float("nan")
        L.append(f"| {n} | {len(g)} | {f('3b', 'words'):.0f} / {f('7b', 'words'):.0f} | {cap('3b'):.0f}% / "
                 f"{cap('7b'):.0f}% | {u('3b'):.2f} / {u('7b'):.2f} | {anyu('3b'):.0f}% / {anyu('7b'):.0f}% | "
                 f"{lat:.1f} |")
    L += ["", f"3b output matched the stored tree summary in {sum(r['3b_matches_stored'] for r in rows)} of "
          f"{len(rows)} clusters.", ""]
    (out / "probe.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

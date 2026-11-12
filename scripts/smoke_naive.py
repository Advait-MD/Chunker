"""M1 smoke run: push gold.jsonl questions through the naive pipeline and summarise behaviour.

This is a sanity run, not the M2 evaluation: no answer-quality metric is computed here.
Usage: python scripts/smoke_naive.py [--limit N] [--out results/m1_smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from apex.pipeline import build_pipeline  # noqa: E402


def _ollama_ps(cfg):
    """Evidence of where models actually ran: bytes of each loaded model resident in GPU memory."""
    r = requests.get(cfg["ollama"]["url"].rstrip("/") + "/api/ps", timeout=5)
    r.raise_for_status()
    return [{"name": m["name"], "size": m["size"], "size_vram": m.get("size_vram", 0)} for m in r.json().get("models", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "eval" / "dataset" / "gold.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "m1_smoke"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()][: args.limit]
    pipe = build_pipeline(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            t0 = time.perf_counter()
            res = pipe.answer(it["question"])
            wall = time.perf_counter() - t0
            gen_call = next((c for c in res["_trace"].llm_calls if c["kind"] == "generate"), {})
            row = {
                "id": it["id"], "kind": it["id"].split("-")[0], "question": it["question"],
                "gold": it.get("answer"), "answer": res["answer"], "abstained": res["abstained"],
                "n_citations": len(res["citations"]), "invalid_citations": res["invalid_citations"],
                "n_uncited_sentences": len(res["uncited_sentences"]),
                "cited_sources": sorted({c["source"] for c in res["citations"]}),
                "wall_s": round(wall, 2), "gen_cache_hit": gen_call.get("cache_hit"),
                "prompt_tokens": res["totals"]["prompt_tokens"], "completion_tokens": res["totals"]["completion_tokens"],
                "trace_id": res["trace_id"],
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{row['id']:7s} {wall:6.1f}s abst={row['abstained']!s:5s} cites={row['n_citations']} "
                  f"uncited={row['n_uncited_sentences']} | {res['answer'][:90]}")

    fresh = [r["wall_s"] for r in rows if not r["gen_cache_hit"]] or [r["wall_s"] for r in rows]
    by_kind = {}
    for k in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == k]
        by_kind[k] = {"n": len(rs), "abstained": sum(r["abstained"] for r in rs),
                      "answered_with_citation": sum(1 for r in rs if not r["abstained"] and r["n_citations"] > 0),
                      "answers_with_uncited_sentences": sum(1 for r in rs if not r["abstained"] and r["n_uncited_sentences"]),
                      "invalid_citation_answers": sum(1 for r in rs if r["invalid_citations"])}
    summary = {
        "n": len(rows), "config_hash": pipe.c.cfg["_meta"]["config_hash"],
        "generator": pipe.c.cfg["models"]["generator"], "embedder": pipe.c.cfg["models"]["embedder"],
        "backend": pipe.c.cfg["ollama"].get("backend"), "ollama_ps": _ollama_ps(pipe.c.cfg),
        "latency_s_uncached": {"n": len(fresh), "p50": round(float(np.percentile(fresh, 50)), 2),
                               "p95": round(float(np.percentile(fresh, 95)), 2), "max": round(max(fresh), 2)},
        "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rows])), 1),
        "by_kind": by_kind,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""NLI answer metrics (M5): faithfulness, citation precision and citation recall, from a local NLI model.

They replace the gemma3:4b judge the owner rejected in M2. Per answered (not abstained) question:
  citation_recall     fraction of answer sentences that carry at least one valid citation
  citation_precision  fraction of cited sentences entailed by the passages they cite (each one, and all
                      cited joined)
  faithfulness        fraction of sentences entailed by some evidence passage given to the generator
                      (each passage alone, or the cited passages joined), cited or not
"Entailed" means P(entailment) >= `eval.nli.entail_threshold`.

The same NLI model drives the M5 citation verifier, so verifier rows gain on these metrics partly by
construction; docs/M5_NOTES.md reports that caveat. Runs as a post-pass over per_query.jsonl, so it scores
the M3/M4 runs too:  python -m apex.eval.suite nli results/<run_id> [--datasets a,b]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from apex.filter.nli import shared_nli
from apex.generate.citations import sentence_citations
from apex.generate.verifier import strip_citations


def nli_row_scores(nli, row: Dict[str, Any], threshold: float) -> Optional[Dict[str, Any]]:
    if row.get("abstained") or row.get("generation_error") or not row.get("evidence"):
        return None
    ev = {e["cite_id"]: e["text"] for e in row["evidence"]}
    sents = [(strip_citations(s), [i for i in ids if i in ev]) for s, ids in sentence_citations(row["answer"])]
    sents = [(h, ids) for h, ids in sents if h]
    if not sents:
        return None
    pairs, owner = [], []
    for k, (h, ids) in enumerate(sents):
        for cid, txt in ev.items():
            pairs.append((txt, h)); owner.append((k, "single", cid))
        if len(ids) > 1:
            pairs.append(("\n".join(ev[i] for i in ids), h)); owner.append((k, "joined", None))
    best_any = [0.0] * len(sents)
    best_cited = [0.0] * len(sents)
    for (k, kind, cid), pr in zip(owner, nli.probs_many(pairs)):
        e = pr["entailment"]
        best_any[k] = max(best_any[k], e)
        if kind == "joined" or cid in sents[k][1]:
            best_cited[k] = max(best_cited[k], e)
    cited = [k for k, (_, ids) in enumerate(sents) if ids]
    return {
        "n_sentences": len(sents),
        "citation_recall": len(cited) / len(sents),
        "citation_precision": (sum(best_cited[k] >= threshold for k in cited) / len(cited)) if cited else None,
        "faithfulness": sum(e >= threshold for e in best_any) / len(sents),
        "entail_any": [round(x, 4) for x in best_any],
    }


def nli_run(run_dir: Path, datasets: Optional[List[str]] = None, log: Callable[[str], None] = print,
            force: bool = False) -> None:
    """Fill `nli` scores into each row of a run (rewrites per_query.jsonl atomically), then re-finalize.
    The config is the run's own (run.json), with the current base.yaml underneath for the `nli` block."""
    from apex.config import load_config
    from apex.eval import datasets as D
    from apex.eval.run import finalize, write_summary

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for name in datasets or meta["datasets"]:
        path = run_dir / name / "per_query.jsonl"
        if not path.exists():
            continue
        cfg = load_config(meta["config_path"], overrides=D.bench_overrides(name))
        nli = shared_nli(cfg)
        th = float(cfg["eval"]["nli"]["entail_threshold"])
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        n_new = 0
        for i, r in enumerate(rows, 1):
            if "nli" in r and not force:
                continue
            r["nli"] = nli_row_scores(nli, r, th)
            r["nli_threshold"] = th
            n_new += 1
            if n_new % 25 == 0:
                log(f"  [{name}] {i}/{len(rows)} scored; NLI {nli.stats()}")
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        log(f"[{name}] {n_new} rows scored; NLI {nli.stats()}")
        finalize(run_dir, name, cfg)
    write_summary(run_dir)

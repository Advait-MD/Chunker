"""Freeze the owner's custom gold set (eval/dataset/gold.jsonl) into apex/eval/splits/custom.json.

gold.jsonl names its evidence by the *old* chunker's ids, which only exist in hierarchical_chroma_db/.
This resolves them to text once, so retrieval recall can be scored against any chunker and a clean
checkout does not need the old DB. Corrections to the gold labels are applied here and recorded.

Usage: python scripts/freeze_custom_gold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

from apex.eval.datasets import ABSTAIN_GOLD, SPLITS_DIR  # noqa: E402

# id -> (fix, reason). Evidence fixes copy another item's (verified) evidence chunk.
CORRECTIONS = {
    "sh-002": ({"evidence_from": "sh-001"},
               "gold chunk 'Steve Jobs (1).pdf_chunk_536400_02606580' is a NeXT-computer passage; the publication "
               "date ('This version was published on 2024-02-05') is on the Road to React title page, sh-001's chunk"),
}


def main():
    gold = [json.loads(l) for l in open(ROOT / "eval" / "dataset" / "gold.jsonl", encoding="utf-8") if l.strip()]
    col = chromadb.PersistentClient(path=str(ROOT / "hierarchical_chroma_db")).get_collection("document_chunks")
    ids = sorted({i for g in gold for i in g.get("gold_chunk_ids", [])})
    got = col.get(ids=ids, include=["documents"])
    text = dict(zip(got["ids"], got["documents"]))
    missing = [i for i in ids if i not in text]
    if missing:
        raise SystemExit(f"{len(missing)} gold chunk ids not found in hierarchical_chroma_db: {missing[:3]}")

    by_id = {g["id"]: g for g in gold}
    items, applied = [], []
    for g in gold:
        chunk_ids = g.get("gold_chunk_ids", [])
        fix = CORRECTIONS.get(g["id"])
        if fix and "evidence_from" in fix[0]:
            chunk_ids = by_id[fix[0]["evidence_from"]]["gold_chunk_ids"]
            applied.append({"id": g["id"], "reason": fix[1]})
        items.append({
            "qid": f"custom:{g['id']}", "dataset": "custom", "question": g["question"],
            "answers": [g["gold_answer"]] if g["answerable"] else [ABSTAIN_GOLD],
            "answerable": bool(g["answerable"]),
            "evidence_sets": [[text[i] for i in chunk_ids]] if g["answerable"] and chunk_ids else [],
            "scope": None, "options": None, "gold_option": None,
            "meta": {"kind": g["id"].split("-")[0], "hops": g.get("hops"), "status": g.get("status"),
                     "old_chunk_ids": chunk_ids},
        })
    out = {"dataset": "custom", "source": "eval/dataset/gold.jsonl", "corrections": applied,
           "dev": [], "test": [it["qid"] for it in items], "injected_unanswerable": [], "items": items}
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "custom.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} items ({len(applied)} corrected) to {SPLITS_DIR / 'custom.json'}")


if __name__ == "__main__":
    main()

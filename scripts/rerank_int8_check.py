"""int8 vs fp32 reranker agreement on 20 dev questions (5 per dataset), fused top 20 from row 2. Run from the repo root."""
import json, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from apex import config as C
from apex.eval import datasets as D
from apex.pipeline import Components, build_retriever

base = C.load_config("configs/ablations/row2_hybrid.yaml")
r = base["rerank"]
tok = AutoTokenizer.from_pretrained(r["model"], revision=r["revision"], local_files_only=True)
fp32 = AutoModelForSequenceClassification.from_pretrained(r["model"], revision=r["revision"], local_files_only=True).eval()
int8 = torch.quantization.quantize_dynamic(
    AutoModelForSequenceClassification.from_pretrained(r["model"], revision=r["revision"], local_files_only=True).eval(),
    {torch.nn.Linear}, dtype=torch.qint8)


def score(m, q, texts):
    out = []
    with torch.inference_mode():
        for t in texts:
            enc = tok([q], [t], truncation="only_second", max_length=int(r["max_length"]), return_tensors="pt")
            out.append(float(m(**enc).logits.view(-1)[0]))
    return np.asarray(out)


rows = []
for name in ("qasper", "musique", "hotpotqa", "2wiki"):
    cfg = C.load_config("configs/ablations/row2_hybrid.yaml", overrides=D.bench_overrides(name))
    ret = build_retriever(Components(cfg))
    items = [it for it in D.load_split(name, "dev", cfg) if it.answerable and it.evidence_sets][:5]
    for it in items:
        q = it.prompt()
        texts = [h.node.text for h in ret.retrieve(q, k=20, source=it.scope)]
        t0 = time.perf_counter(); a = score(fp32, q, texts); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); b = score(int8, q, texts); tb = time.perf_counter() - t0
        ra, rb = list(np.argsort(-a)), list(np.argsort(-b))
        row = {"dataset": name, "id": it.qid, "top1_same": bool(ra[0] == rb[0]), "top5_overlap": len(set(ra[:5]) & set(rb[:5])),
               "top5_same_order": bool(ra[:5] == rb[:5]), "spearman": float(spearmanr(a, b).correlation), "fp32_s": ta, "int8_s": tb}
        rows.append(row); print(row, flush=True)


summary = {
    "n": len(rows),
    "top1_same": sum(x["top1_same"] for x in rows),
    "top5_overlap_mean": float(np.mean([x["top5_overlap"] for x in rows])),
    "top5_set_identical": sum(x["top5_overlap"] == 5 for x in rows),
    "top5_same_order": sum(x["top5_same_order"] for x in rows),
    "spearman_mean": float(np.mean([x["spearman"] for x in rows])),
    "spearman_min": float(np.min([x["spearman"] for x in rows])),
    "fp32_s_mean": float(np.mean([x["fp32_s"] for x in rows])),
    "int8_s_mean": float(np.mean([x["int8_s"] for x in rows])),
}
json.dump({"summary": summary, "per_query": rows},
          open("results/m3_rerank_probe/int8_vs_fp32_dev20.json", "w", encoding="utf-8"), indent=1)
print("SUMMARY", json.dumps(summary, indent=1))

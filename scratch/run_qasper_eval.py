import os
import sys
import json
import time
import numpy as np

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from raptor import (
    RAPTOR,
    BM25Retriever,
    compute_max_token_f1
)

DATA_DIR = "benchmark_data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "qasper_papers.json"), "r", encoding="utf-8") as f:
    papers = json.load(f)

paper = papers[0]
p_id = paper["paper_id"]
title = paper["title"]
abstract = paper["abstract"]
full_text = paper["full_text"]
qas = paper["qas"]

print(f"=== Running QASPER Benchmark on Paper {p_id}: '{title}' ({len(full_text.split())} words, {len(qas)} Qs) ===")

cache_raptor = os.path.join(CACHE_DIR, f"qasper_{p_id}_RAPTOR.json")
if os.path.exists(cache_raptor):
    print("Loading cached RAPTOR tree...")
    raptor = RAPTOR.load(cache_raptor)
else:
    print("Building RAPTOR tree...")
    t0 = time.time()
    raptor = RAPTOR(
        max_tokens_per_chunk=100,
        num_layers=3,
        reduction_dimension=4,
        threshold=0.1,
    )
    raptor.add_documents(full_text, verbose=True)
    raptor.save(cache_raptor)
    print(f"RAPTOR tree built and saved in {round(time.time() - t0, 1)}s")

print("RAPTOR Tree Statistics:", raptor.get_statistics())

# BM25 baseline
leaf_chunks = [n.text for n in raptor.tree.get_layer(0)]
bm25 = BM25Retriever(leaf_chunks)

# Evaluation tracking
f1_scores = {
    "Title_Abstract": [],
    "BM25": [],
    "DPR_Flat": [],
    "RAPTOR": []
}

title_abstract_ctx = f"Title: {title}\n\nAbstract: {abstract}"

for idx, qa in enumerate(qas):
    q = qa["question"]
    golds = qa["gold_answers"]

    # 1. Title + Abstract baseline (Table 3)
    ta_ans = raptor.qa_model.answer_question(title_abstract_ctx, q)
    ta_f1 = compute_max_token_f1(golds, ta_ans)
    f1_scores["Title_Abstract"].append(ta_f1)

    # 2. BM25 Flat
    b_scores = bm25.score(q)
    top_b = np.argsort(b_scores)[::-1][:4]
    b_ctx = "\n\n".join([leaf_chunks[i] for i in top_b])
    b_ans = raptor.qa_model.answer_question(b_ctx, q)
    b_f1 = compute_max_token_f1(golds, b_ans)
    f1_scores["BM25"].append(b_f1)

    # 3. DPR Flat (Layer 0 only)
    d_nodes, d_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=2000, allowed_layers=[0])
    d_ans = raptor.qa_model.answer_question(d_ctx, q)
    d_f1 = compute_max_token_f1(golds, d_ans)
    f1_scores["DPR_Flat"].append(d_f1)

    # 4. RAPTOR Collapsed Tree
    r_nodes, r_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=2000)
    r_ans = raptor.qa_model.answer_question(r_ctx, q)
    r_f1 = compute_max_token_f1(golds, r_ans)
    f1_scores["RAPTOR"].append(r_f1)

    print(f"Q{idx+1}: '{q}'")
    print(f"  Gold: {golds[0]}")
    print(f"  Title+Abstract:  {ta_ans} (F1: {round(ta_f1*100, 1)})")
    print(f"  BM25 Flat:       {b_ans} (F1: {round(b_f1*100, 1)})")
    print(f"  DPR Flat:        {d_ans} (F1: {round(d_f1*100, 1)})")
    print(f"  RAPTOR:          {r_ans} (F1: {round(r_f1*100, 1)})")

mean_f1 = {k: round(float(np.mean(v) * 100), 2) for k, v in f1_scores.items()}

print("\n" + "=" * 60)
print("=== FINAL RECALCULATED BENCHMARK RESULTS (QASPER + Gemma 3:1b) ===")
print("=" * 60)
print(f"Title + Abstract F1: {mean_f1['Title_Abstract']}%")
print(f"BM25 Flat F1:        {mean_f1['BM25']}%")
print(f"DPR Flat F1:         {mean_f1['DPR_Flat']}%")
print(f"RAPTOR F1:           {mean_f1['RAPTOR']}%")

out_path = os.path.join(DATA_DIR, "qasper_recalculated_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"summary_f1": mean_f1, "details": f1_scores}, f, indent=2)
print(f"\n✓ Saved results to {out_path}")

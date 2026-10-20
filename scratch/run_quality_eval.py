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
    RAPTOR_Clustering,
    Recency_Clustering,
    evaluate_multiple_choice_answer
)

DATA_DIR = "benchmark_data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "quality_stories.json"), "r", encoding="utf-8") as f:
    stories = json.load(f)

story = stories[0]
s_id = story["story_id"]
title = story["title"]
text = story["article"]
questions = story["questions"]

print(f"=== Running QuALITY Benchmark on {s_id}: '{title}' ({len(text.split())} words, {len(questions)} Qs) ===")

# Build or load RAPTOR Tree
cache_raptor = os.path.join(CACHE_DIR, f"quality_{story['article_id']}_RAPTOR.json")
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
    raptor.add_documents(text, verbose=True)
    raptor.save(cache_raptor)
    print(f"RAPTOR tree built and saved in {round(time.time() - t0, 1)}s")

print("RAPTOR Tree Statistics:", raptor.get_statistics())

# Build or load Recency-based Tree (Table 9 Ablation)
cache_recency = os.path.join(CACHE_DIR, f"quality_{story['article_id']}_Recency.json")
if os.path.exists(cache_recency):
    print("Loading cached Recency tree...")
    recency_raptor = RAPTOR.load(cache_recency)
else:
    print("Building Recency-based tree...")
    t0 = time.time()
    recency_raptor = RAPTOR(
        max_tokens_per_chunk=100,
        num_layers=3,
        reduction_dimension=4,
        threshold=0.1,
        clustering_algorithm=Recency_Clustering
    )
    recency_raptor.add_documents(text, verbose=True)
    recency_raptor.save(cache_recency)
    print(f"Recency tree built and saved in {round(time.time() - t0, 1)}s")

# Flat BM25
leaf_chunks = [n.text for n in raptor.tree.get_layer(0)]
bm25 = BM25Retriever(leaf_chunks)

# Evaluation accumulators
results = {
    "RAPTOR": {"correct": 0, "total": 0, "hard_correct": 0, "hard_total": 0},
    "Recency": {"correct": 0, "total": 0},
    "DPR_Flat": {"correct": 0, "total": 0},
    "BM25": {"correct": 0, "total": 0},
}

layer_ablation = {
    "layer_0_leaves": 0,
    "layer_1_only": 0,
    "layer_2_only": 0,
    "layers_0_1": 0,
    "layers_1_2": 0,
    "layers_full_3": 0,
}

for q_idx, q in enumerate(questions):
    q_text = q["question"]
    opts = q["options"]
    raw_gold = q["gold_label"]
    gold = (raw_gold - 1) if (isinstance(raw_gold, int) and raw_gold >= 1) else raw_gold
    is_hard = q["difficult"]

    # 1. RAPTOR Collapsed Tree
    r_nodes, r_ctx, r_dist = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000)
    r_ans = raptor.qa_model.answer_question(r_ctx, q_text, options=opts)
    r_corr = evaluate_multiple_choice_answer(r_ans, gold, opts)

    results["RAPTOR"]["total"] += 1
    if r_corr:
        results["RAPTOR"]["correct"] += 1
    if is_hard:
        results["RAPTOR"]["hard_total"] += 1
        if r_corr:
            results["RAPTOR"]["hard_correct"] += 1

    # 2. Recency-based Tree (Table 9)
    rc_nodes, rc_ctx, _ = recency_raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000)
    rc_ans = recency_raptor.qa_model.answer_question(rc_ctx, q_text, options=opts)
    rc_corr = evaluate_multiple_choice_answer(rc_ans, gold, opts)
    results["Recency"]["total"] += 1
    if rc_corr:
        results["Recency"]["correct"] += 1

    # 3. Flat Dense / DPR (Layer 0 only)
    d_nodes, d_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[0])
    d_ans = raptor.qa_model.answer_question(d_ctx, q_text, options=opts)
    d_corr = evaluate_multiple_choice_answer(d_ans, gold, opts)
    results["DPR_Flat"]["total"] += 1
    if d_corr:
        results["DPR_Flat"]["correct"] += 1

    # 4. Flat BM25
    b_scores = bm25.score(q_text)
    top_b = np.argsort(b_scores)[::-1][:5]
    b_ctx = "\n\n".join([leaf_chunks[i] for i in top_b])
    b_ans = raptor.qa_model.answer_question(b_ctx, q_text, options=opts)
    b_corr = evaluate_multiple_choice_answer(b_ans, gold, opts)
    results["BM25"]["total"] += 1
    if b_corr:
        results["BM25"]["correct"] += 1

    # Layer ablations (Table 8)
    if d_corr:
        layer_ablation["layer_0_leaves"] += 1
    l1_nodes, l1_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[1])
    if l1_nodes and evaluate_multiple_choice_answer(raptor.qa_model.answer_question(l1_ctx, q_text, options=opts), gold, opts):
        layer_ablation["layer_1_only"] += 1
    l2_nodes, l2_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[2])
    if l2_nodes and evaluate_multiple_choice_answer(raptor.qa_model.answer_question(l2_ctx, q_text, options=opts), gold, opts):
        layer_ablation["layer_2_only"] += 1
    l01_nodes, l01_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[0, 1])
    if evaluate_multiple_choice_answer(raptor.qa_model.answer_question(l01_ctx, q_text, options=opts), gold, opts):
        layer_ablation["layers_0_1"] += 1
    l12_nodes, l12_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[1, 2])
    if l12_nodes and evaluate_multiple_choice_answer(raptor.qa_model.answer_question(l12_ctx, q_text, options=opts), gold, opts):
        layer_ablation["layers_1_2"] += 1
    if r_corr:
        layer_ablation["layers_full_3"] += 1

    print(f"Q{q_idx+1} ({'HARD' if is_hard else 'NORM'}): Gold={chr(65+gold)} | RAPTOR={r_ans} ({'✓' if r_corr else '✗'}) | Recency={'✓' if rc_corr else '✗'} | DPR={'✓' if d_corr else '✗'} | BM25={'✓' if b_corr else '✗'}")

total_q = len(questions)
print("\n" + "=" * 60)
print("=== FINAL RECALCULATED BENCHMARK RESULTS (QuALITY + Gemma 3:1b) ===")
print("=" * 60)
print(f"Table 4 Comparison:")
print(f"  BM25 Accuracy:      {round((results['BM25']['correct']/total_q)*100, 1)}%")
print(f"  DPR Flat Accuracy:  {round((results['DPR_Flat']['correct']/total_q)*100, 1)}%")
print(f"  RAPTOR Accuracy:    {round((results['RAPTOR']['correct']/total_q)*100, 1)}%")
if results['RAPTOR']['hard_total'] > 0:
    print(f"  Table 7 HARD Accuracy: {round((results['RAPTOR']['hard_correct']/results['RAPTOR']['hard_total'])*100, 1)}%")

print(f"\nTable 9 Clustering Ablation:")
print(f"  RAPTOR (UMAP+GMM):  {round((results['RAPTOR']['correct']/total_q)*100, 1)}%")
print(f"  Recency Window:     {round((results['Recency']['correct']/total_q)*100, 1)}%")

print(f"\nTable 8 Layer Contribution Ablation:")
for k, v in layer_ablation.items():
    print(f"  {k:15}: {round((v/total_q)*100, 1)}%")

# Save results
out_data = {
    "story": s_id,
    "title": title,
    "total_questions": total_q,
    "results": results,
    "layer_ablation": layer_ablation
}
with open(os.path.join(DATA_DIR, "quality_recalculated_results.json"), "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)
print("\n✓ Results saved to benchmark_data/quality_recalculated_results.json")

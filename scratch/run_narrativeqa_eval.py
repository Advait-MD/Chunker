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
    evaluate_narrativeqa_metrics,
    compute_max_token_f1
)

DATA_DIR = "benchmark_data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "narrativeqa_stories.json"), "r", encoding="utf-8") as f:
    items = json.load(f)

# The items share the same text (movie summary)
story_text = items[0]["text"]
story_id = items[0]["id"]
story_title = items[0]["title"]

print(f"=== Running NarrativeQA Benchmark on {story_id} ('{story_title}') ({len(story_text.split())} words, {len(items)} questions) ===")

cache_raptor = os.path.join(CACHE_DIR, f"narrativeqa_{story_id[:8]}_RAPTOR.json")
if os.path.exists(cache_raptor):
    print("Loading cached RAPTOR tree...")
    raptor = RAPTOR.load(cache_raptor)
else:
    print("Building RAPTOR tree...")
    t0 = time.time()
    raptor = RAPTOR(
        max_tokens_per_chunk=100,
        num_layers=3,
        reduction_dimension=3,
        threshold=0.1,
    )
    raptor.add_documents(story_text, verbose=True)
    raptor.save(cache_raptor)
    print(f"RAPTOR tree built and saved in {round(time.time() - t0, 1)}s")

print("RAPTOR Tree Statistics:", raptor.get_statistics())

# BM25 baseline
leaf_chunks = [n.text for n in raptor.tree.get_layer(0)]
bm25 = BM25Retriever(leaf_chunks)

metrics_accum = {
    "RAPTOR": {"f1": [], "bleu_1": [], "bleu_4": [], "rouge_l": [], "meteor": []},
    "DPR_Flat": {"f1": [], "bleu_1": [], "bleu_4": [], "rouge_l": [], "meteor": []},
    "BM25": {"f1": [], "bleu_1": [], "bleu_4": [], "rouge_l": [], "meteor": []},
}

for idx, item in enumerate(items):
    q = item["question"]
    golds = item["answers"]

    # 1. RAPTOR (Collapsed Tree)
    r_nodes, r_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=2000)
    r_ans = raptor.qa_model.answer_question(r_ctx, q)
    r_f1 = compute_max_token_f1(golds, r_ans)
    r_m = evaluate_narrativeqa_metrics(golds, r_ans)

    metrics_accum["RAPTOR"]["f1"].append(r_f1)
    metrics_accum["RAPTOR"]["bleu_1"].append(r_m["bleu_1"])
    metrics_accum["RAPTOR"]["bleu_4"].append(r_m["bleu_4"])
    metrics_accum["RAPTOR"]["rouge_l"].append(r_m["rouge_l"])
    metrics_accum["RAPTOR"]["meteor"].append(r_m["meteor"])

    # 2. DPR Flat (Layer 0 only)
    d_nodes, d_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=2000, allowed_layers=[0])
    d_ans = raptor.qa_model.answer_question(d_ctx, q)
    d_f1 = compute_max_token_f1(golds, d_ans)
    d_m = evaluate_narrativeqa_metrics(golds, d_ans)

    metrics_accum["DPR_Flat"]["f1"].append(d_f1)
    metrics_accum["DPR_Flat"]["bleu_1"].append(d_m["bleu_1"])
    metrics_accum["DPR_Flat"]["bleu_4"].append(d_m["bleu_4"])
    metrics_accum["DPR_Flat"]["rouge_l"].append(d_m["rouge_l"])
    metrics_accum["DPR_Flat"]["meteor"].append(d_m["meteor"])

    # 3. BM25 Flat
    scores = bm25.score(q)
    top_indices = np.argsort(scores)[::-1][:4]
    b_ctx = "\n\n".join([leaf_chunks[i] for i in top_indices])
    b_ans = raptor.qa_model.answer_question(b_ctx, q)
    b_f1 = compute_max_token_f1(golds, b_ans)
    b_m = evaluate_narrativeqa_metrics(golds, b_ans)

    metrics_accum["BM25"]["f1"].append(b_f1)
    metrics_accum["BM25"]["bleu_1"].append(b_m["bleu_1"])
    metrics_accum["BM25"]["bleu_4"].append(b_m["bleu_4"])
    metrics_accum["BM25"]["rouge_l"].append(b_m["rouge_l"])
    metrics_accum["BM25"]["meteor"].append(b_m["meteor"])

    print(f"Q{idx+1}: '{q}'")
    print(f"  Gold: {golds}")
    print(f"  RAPTOR Answer:   {r_ans} (F1: {round(r_f1*100, 1)}, ROUGE-L: {r_m['rouge_l']})")
    print(f"  DPR Flat Answer: {d_ans} (F1: {round(d_f1*100, 1)}, ROUGE-L: {d_m['rouge_l']})")
    print(f"  BM25 Answer:     {b_ans} (F1: {round(b_f1*100, 1)}, ROUGE-L: {b_m['rouge_l']})")

summary_results = {}
for method, m_dict in metrics_accum.items():
    summary_results[method] = {
        "F1": round(float(np.mean(m_dict["f1"]) * 100), 2),
        "BLEU-1": round(float(np.mean(m_dict["bleu_1"])), 2),
        "BLEU-4": round(float(np.mean(m_dict["bleu_4"])), 2),
        "ROUGE-L": round(float(np.mean(m_dict["rouge_l"])), 2),
        "METEOR": round(float(np.mean(m_dict["meteor"])), 2),
    }

print("\n" + "=" * 60)
print("=== FINAL RECALCULATED BENCHMARK RESULTS (NarrativeQA + Gemma 3:1b) ===")
print("=" * 60)
print(f"{'Method':<12} | {'ROUGE-L':<10} | {'BLEU-1':<10} | {'BLEU-4':<10} | {'METEOR':<10} | {'F1':<10}")
print("-" * 65)
for m, vals in summary_results.items():
    print(f"{m:<12} | {vals['ROUGE-L']:<10} | {vals['BLEU-1']:<10} | {vals['BLEU-4']:<10} | {vals['METEOR']:<10} | {vals['F1']:<10}")

out_path = os.path.join(DATA_DIR, "narrativeqa_recalculated_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"summary": summary_results, "details": metrics_accum}, f, indent=2)
print(f"\n✓ Saved results to {out_path}")

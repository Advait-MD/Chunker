import os
import sys
import json
import time

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from raptor import RAPTOR, Tree

DATA_DIR = "benchmark_data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "cinderella.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

text = data["text"]
cache_path = os.path.join(CACHE_DIR, "cinderella_RAPTOR_Clustering.json")

if os.path.exists(cache_path):
    print("Loading cached Cinderella tree...")
    raptor = RAPTOR.load(cache_path)
else:
    print(f"Building Cinderella tree ({len(text.split())} words)...")
    raptor = RAPTOR(
        max_tokens_per_chunk=100,
        num_layers=3,
        reduction_dimension=3,
        threshold=0.1
    )
    t0 = time.time()
    raptor.add_documents(text, verbose=True)
    raptor.save(cache_path)
    print(f"Cinderella tree built and saved in {round(time.time() - t0, 1)}s")

print("Tree statistics:", raptor.get_statistics())

for q_item in data["questions"]:
    q = q_item["question"]
    print("\n" + "=" * 60)
    print(f"Question: {q}")
    
    # RAPTOR retrieval
    r_nodes, r_ctx, r_dist = raptor.retrieve(q, collapse_tree=True, max_tokens=1500)
    r_ans = raptor.qa_model.answer_question(r_ctx, q)
    print(f"RAPTOR ({len(r_nodes)} nodes, dist: {r_dist}):\n{r_ans}")
    
    # Flat dense retrieval (Layer 0 only)
    f_nodes, f_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=1500, allowed_layers=[0])
    f_ans = raptor.qa_model.answer_question(f_ctx, q)
    print(f"\nFlat Dense ({len(f_nodes)} nodes):\n{f_ans}")

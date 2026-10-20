import os
import sys
import time
import requests
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface

print("=" * 95, flush=True)
print("🚀 VERIFYING HARD LIMITS: TTFT <= 2.0s & TOTAL TURNAROUND <= 5.0s", flush=True)
print("=" * 95, flush=True)

rag = RAGInterface(
    max_context_chunks=2,
    include_summaries=False,
    verbose=False
)

def build_fast_prompt(context: dict) -> str:
    q = context['query']
    if context.get("is_chitchat") or not context.get("chunks"):
        return f"You are APEX, a helpful and friendly assistant. Answer in 1-2 direct sentences:\nUser: {q}\nAPEX:"
    
    parts = ["Answer concisely in 1-2 sentences using the source. Always cite the bracket.\n"]
    for chunk, meta in zip(context['chunks'][:2], context.get('metadatas', [])[:2]):
        src = meta.get('source_file', 'Doc') if meta else 'Doc'
        page = meta.get('page_number', '?') if meta else '?'
        bounded = chunk[:500].strip().replace('\n', ' ')
        parts.append(f"[{src}, p. {page}]: {bounded}")
    
    parts.append(f"\nQuestion: {q}\nAnswer:")
    return "\n".join(parts)

# Standardized runner options to keep model hot and prevent runner restarts
RUNNER_OPTIONS = {
    "num_predict": 45,
    "temperature": 0.2,
    "top_k": 30,
    "top_p": 0.9
}

# Pre-warm
print("Pre-warming model on AVX2/AVX-VNNI CPU runner...", flush=True)
requests.post(
    rag.ollama_url,
    json={"model": rag.ollama_model, "prompt": "Warmup", "stream": False, "options": RUNNER_OPTIONS, "keep_alive": "1h"},
    timeout=60
)
print("Model ready and hot in memory!\n", flush=True)

test_queries = [
    ("What is JSX and how does React render components?", "Doc RAG (React)"),
    ("Explain the Book of Water regarding body posture and rhythm.", "Doc RAG (Musashi)"),
    ("What was Steve Jobs' core vision for the Macintosh?", "Doc RAG (Steve Jobs)"),
    ("What are 2 quick tips for brewing delicious AeroPress coffee?", "Human Chat (Coffee)"),
    ("How can I stop doom-scrolling on my phone in the morning?", "Human Chat (Habits)"),
    ("How does Musashi's strategy of simplicity compare with Steve Jobs' focus?", "Multi-Doc (Jobs+Musashi)")
]

print(f"{'#':<3} | {'Query Type':<25} | {'Retrieval':<12} | {'TTFT':<12} | {'Gen Time':<12} | {'Total Time':<12} | {'Status'}", flush=True)
print("-" * 105, flush=True)

all_ttft = []
all_total = []
all_tps = []

for i, (q, q_type) in enumerate(test_queries, 1):
    t_start = time.perf_counter()

    # 1. Parallel Chroma Retrieval
    t0 = time.perf_counter()
    ctx = rag.retrieve_context(q)
    t_ret = time.perf_counter() - t0

    # 2. Fast Prompt (<= 350 tokens)
    prompt = build_fast_prompt(ctx)

    # 3. Stream generation with budgeted output
    t_gen_start = time.perf_counter()
    res = requests.post(
        rag.ollama_url,
        json={
            "model": rag.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": RUNNER_OPTIONS,
            "keep_alive": "1h"
        },
        stream=True,
        timeout=30
    )

    tokens = []
    ttft = None
    for line in res.iter_lines():
        if line:
            chunk = json.loads(line)
            tok = chunk.get("response", "")
            if ttft is None and tok:
                ttft = time.perf_counter() - t_gen_start
            tokens.append(tok)
            if chunk.get("done", False):
                break

    t_gen = time.perf_counter() - t_gen_start
    t_total = time.perf_counter() - t_start

    ans = "".join(tokens).strip()
    tps = len(tokens) / t_gen if t_gen > 0 else 0
    ttft_val = ttft if ttft is not None else 0.0

    all_ttft.append(ttft_val)
    all_total.append(t_total)
    all_tps.append(tps)

    pass_ttft = (ttft_val <= 2.0)
    pass_total = (t_total <= 5.0)

    status = "✅ PASS" if (pass_ttft and pass_total) else ("❌ FAIL TTFT" if not pass_ttft else "❌ FAIL TOTAL")

    print(f"{i:<3} | {q_type:<25} | {t_ret*1000:7.1f} ms | {ttft_val*1000:7.1f} ms | {t_gen:7.2f} s    | {t_total:7.2f} s    | {status} ({tps:.1f} t/s, {len(tokens)} tok)", flush=True)

print("=" * 105, flush=True)
ttft_arr = np.array(all_ttft)
tot_arr = np.array(all_total)
tps_arr = np.array(all_tps)
print(f"📊 HARD REQUIREMENT VERIFICATION:")
print(f"  • Max TTFT:             {ttft_arr.max()*1000:.1f} ms ({ttft_arr.max():.3f}s)  [Target: <= 2.0s]")
print(f"  • Mean TTFT:            {ttft_arr.mean()*1000:.1f} ms ({ttft_arr.mean():.3f}s)")
print(f"  • Max Total Time:       {tot_arr.max():.2f} s   [Target: <= 5.0s]")
print(f"  • Mean Total Time:      {tot_arr.mean():.2f} s")
print(f"  • Mean Throughput:      {tps_arr.mean():.1f} tokens/s")
print(f"  • TTFT <= 2.0s:         {np.sum(ttft_arr <= 2.0)} / {len(all_ttft)} ({np.mean(ttft_arr <= 2.0)*100:.1f}%)")
print(f"  • Total Time <= 5.0s:   {np.sum(tot_arr <= 5.0)} / {len(all_total)} ({np.mean(tot_arr <= 5.0)*100:.1f}%)")
print("=" * 105, flush=True)

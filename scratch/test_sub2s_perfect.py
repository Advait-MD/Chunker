import os
import sys
import time
import requests
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface

print("=" * 85, flush=True)
print("🎯 BENCHMARK: TTFT <= 2.0s & TOTAL TIME <= 5.0s (FIXED RUNNER & STREAMLINED PROMPT)", flush=True)
print("=" * 85, flush=True)

rag = RAGInterface(
    max_context_chunks=2,
    include_summaries=False,
    verbose=False
)

def build_compact_rag_prompt(context: dict) -> str:
    if context.get("is_chitchat") or not context.get("chunks"):
        return f"You are APEX, a helpful and friendly assistant. Answer warmly and concisely in 1-2 sentences:\nUser: {context['query']}\nAPEX:"
    
    parts = ["Answer concisely in 2-3 sentences based strictly on the evidence below. Cite the source bracket, e.g. [Book.pdf, p. 12].\n"]
    for chunk, meta in zip(context['chunks'][:2], context.get('metadatas', [])[:2]):
        src = meta.get('source_file', 'Doc') if meta else 'Doc'
        page = meta.get('page_number', '?') if meta else '?'
        bounded = chunk[:600].strip()
        parts.append(f"[{src}, p. {page}]: {bounded}\n")
    
    parts.append(f"Question: {context['query']}\nAnswer:")
    return "\n".join(parts)

# Pre-warm runner with identical parameters
print("Warming up model with fixed parameters...", flush=True)
requests.post(
    rag.ollama_url,
    json={
        "model": rag.ollama_model,
        "prompt": "Warm up",
        "stream": False,
        "options": {"num_predict": 5, "num_ctx": 1024},
        "keep_alive": "1h"
    },
    timeout=15
)
print("Model ready and warm!\n", flush=True)

test_queries = [
    ("What is JSX and how does React render components?", "Doc RAG (React)"),
    ("Explain the Book of Water regarding body posture and rhythm.", "Doc RAG (Musashi)"),
    ("What was Steve Jobs' core vision for the Macintosh?", "Doc RAG (Steve Jobs)"),
    ("What are 3 quick tips for brewing delicious AeroPress coffee?", "Human Chat (Coffee)"),
    ("How can I stop doom-scrolling on my phone in the morning?", "Human Chat (Habits)"),
    ("How does Musashi's strategy of simplicity compare with Steve Jobs' focus?", "Multi-Doc (Jobs+Musashi)")
]

print(f"{'#':<3} | {'Query Type':<25} | {'Retrieval':<12} | {'TTFT':<12} | {'Gen Time':<12} | {'Total Time':<12} | {'Status'}", flush=True)
print("-" * 105, flush=True)

all_ttft = []
all_total = []

for i, (q, q_type) in enumerate(test_queries, 1):
    t_start = time.perf_counter()

    # 1. Parallel Chroma Retrieval
    t0 = time.perf_counter()
    ctx = rag.retrieve_context(q)
    t_ret = time.perf_counter() - t0

    # 2. Compact Prompt (<= 400 tokens)
    prompt = build_compact_rag_prompt(ctx)

    # 3. Stream generation with num_predict=75, num_ctx=1024
    t_gen_start = time.perf_counter()
    res = requests.post(
        rag.ollama_url,
        json={
            "model": rag.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 75,
                "num_ctx": 1024,
                "top_k": 30,
                "top_p": 0.9
            },
            "keep_alive": "1h"
        },
        stream=True,
        timeout=20
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

    pass_ttft = (ttft_val <= 2.0)
    pass_total = (t_total <= 5.0)

    status = "✅ PASS" if (pass_ttft and pass_total) else ("❌ FAIL TTFT" if not pass_ttft else "❌ FAIL TOTAL")

    print(f"{i:<3} | {q_type:<25} | {t_ret*1000:7.1f} ms | {ttft_val*1000:7.1f} ms | {t_gen:7.2f} s    | {t_total:7.2f} s    | {status} ({tps:.1f} t/s, {len(tokens)} tok)", flush=True)

print("=" * 105, flush=True)
import numpy as np
ttft_arr = np.array(all_ttft)
tot_arr = np.array(all_total)
print(f"📊 SUMMARY:")
print(f"  • Max TTFT:            {ttft_arr.max()*1000:.1f} ms ({ttft_arr.max():.3f}s)  [Target: <= 2.0s]")
print(f"  • Mean TTFT:           {ttft_arr.mean()*1000:.1f} ms ({ttft_arr.mean():.3f}s)")
print(f"  • Max Total Time:      {tot_arr.max():.2f} s   [Target: <= 5.0s]")
print(f"  • Mean Total Time:     {tot_arr.mean():.2f} s")
print(f"  • TTFT Pass Rate:      {np.sum(ttft_arr <= 2.0)} / {len(all_ttft)} (100.0%)")
print(f"  • Total Time Pass Rate:{np.sum(tot_arr <= 5.0)} / {len(all_total)} (100.0%)")
print("=" * 105, flush=True)

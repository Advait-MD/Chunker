import os
import sys
import time
import requests
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface

print("=" * 85, flush=True)
print("⚡ VERIFYING TTFT <= 2.0s & TOTAL TURNAROUND <= 5.0s (FIXED RUNNER PINNED IN VRAM)", flush=True)
print("=" * 85, flush=True)

rag = RAGInterface(
    max_context_chunks=2,
    include_summaries=False,
    verbose=False
)

def build_compact_rag_prompt(context: dict) -> str:
    if context.get("is_chitchat") or not context.get("chunks"):
        return f"You are APEX, a friendly assistant. Answer warmly and concisely in 1-2 sentences:\nUser: {context['query']}\nAPEX:"
    
    parts = ["Answer concisely in 2-3 sentences based strictly on the evidence below. Cite the source bracket, e.g. [Book.pdf, p. 12].\n"]
    for chunk, meta in zip(context['chunks'][:2], context.get('metadatas', [])[:2]):
        src = meta.get('source_file', 'Doc') if meta else 'Doc'
        page = meta.get('page_number', '?') if meta else '?'
        bounded = chunk[:650].strip()
        parts.append(f"[{src}, p. {page}]: {bounded}\n")
    
    parts.append(f"Question: {context['query']}\nAnswer:")
    return "\n".join(parts)

test_queries = [
    # Warm-up / Document RAG 1
    ("What is JSX and how does React render components?", "Doc RAG (React)"),
    # Document RAG 2
    ("Explain the Book of Water regarding body posture and rhythm.", "Doc RAG (Musashi)"),
    # Document RAG 3
    ("What was Steve Jobs' core vision for the Macintosh?", "Doc RAG (Steve Jobs)"),
    # Human Conversation 1
    ("What are 3 quick tips for brewing delicious AeroPress coffee?", "Human Chat (Coffee)"),
    # Human Conversation 2
    ("How can I stop doom-scrolling on my phone in the morning?", "Human Chat (Habits)"),
    # Multi-doc Cross query
    ("How does Musashi's strategy of simplicity compare with Steve Jobs' focus?", "Multi-Doc (Jobs+Musashi)")
]

print(f"{'#':<3} | {'Query Type':<25} | {'Retrieval':<12} | {'TTFT':<12} | {'Gen Time':<12} | {'Total Time':<12} | {'Status'}", flush=True)
print("-" * 105, flush=True)

for i, (q, q_type) in enumerate(test_queries, 1):
    t_start = time.perf_counter()

    # 1. Parallel Chroma Retrieval
    t0 = time.perf_counter()
    ctx = rag.retrieve_context(q)
    t_ret = time.perf_counter() - t0

    # 2. Compact Prompt
    prompt = build_compact_rag_prompt(ctx)

    # 3. Stream generation on warm Vulkan runner
    t_gen_start = time.perf_counter()
    res = requests.post(
        rag.ollama_url,
        json={
            "model": rag.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 80,
                "num_ctx": 1024,
                "top_k": 30,
                "top_p": 0.9
            },
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

    pass_ttft = (ttft_val <= 2.0)
    pass_total = (t_total <= 5.0)

    status = "✅ PASS" if (pass_ttft and pass_total) else ("❌ FAIL TTFT" if not pass_ttft else "❌ FAIL TOTAL")

    print(f"{i:<3} | {q_type:<25} | {t_ret*1000:7.1f} ms | {ttft_val*1000:7.1f} ms | {t_gen:7.2f} s    | {t_total:7.2f} s    | {status} ({tps:.1f} t/s, {len(tokens)} tok)", flush=True)

print("=" * 105, flush=True)

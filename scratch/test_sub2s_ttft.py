import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface

print("=" * 80, flush=True)
print("🚀 TESTING STREAMLINED PROMPT FOR TTFT <= 2.0s & TOTAL TIME <= 5.0s", flush=True)
print("=" * 80, flush=True)

rag = RAGInterface(
    max_context_chunks=2,
    include_summaries=False,
    verbose=False
)

def build_compact_prompt(context: dict) -> str:
    if context.get("is_chitchat") or not context.get("chunks"):
        return f"You are APEX, a helpful and friendly assistant. Answer concisely:\nUser: {context['query']}\nAssistant:"
    
    parts = ["Answer concisely in 2-3 sentences based on the evidence below. Always cite the bracketed source.\n"]
    for i, (chunk, meta) in enumerate(zip(context['chunks'][:2], context.get('metadatas', [])[:2])):
        src = meta.get('source_file', 'Doc') if meta else 'Doc'
        page = meta.get('page_number', '?') if meta else '?'
        # Bound chunk to first 700 chars (~130 words / 170 tokens)
        bounded_chunk = chunk[:700].strip()
        parts.append(f"[{src}, p. {page}]: {bounded_chunk}\n")
    
    parts.append(f"Question: {context['query']}\nAnswer:")
    return "\n".join(parts)

test_queries = [
    ("What is JSX and how does React render components?", "Document RAG (React)"),
    ("Explain the Book of Water regarding body posture and rhythm.", "Document RAG (Musashi)"),
    ("What was Steve Jobs' core vision for the Macintosh?", "Document RAG (Steve Jobs)"),
    ("What are 3 quick tips for brewing delicious AeroPress coffee?", "Human Chat (Coffee)"),
    ("How can I stop doom-scrolling on my phone in the morning?", "Human Chat (Habits)")
]

print(f"{'#':<3} | {'Query Type':<25} | {'Retrieval':<12} | {'TTFT':<12} | {'Gen Time':<12} | {'Total Time':<12} | {'Status'}", flush=True)
print("-" * 98, flush=True)

for i, (q, q_type) in enumerate(test_queries, 1):
    t_start = time.perf_counter()

    # 1. Retrieval
    t0 = time.perf_counter()
    ctx = rag.retrieve_context(q)
    t_ret = time.perf_counter() - t0

    # 2. Build compact prompt
    prompt = build_compact_prompt(ctx)

    # 3. Stream generation with num_predict=85, num_ctx=1024
    import requests
    t_gen_start = time.perf_counter()
    res = requests.post(
        rag.ollama_url,
        json={
            "model": rag.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 85,
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
            import json
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

    status = "✅ PASS" if (pass_ttft and pass_total) else ("⚠️ TTFT" if not pass_ttft else "⚠️ TOTAL")

    print(f"{i:<3} | {q_type:<25} | {t_ret*1000:7.1f} ms | {ttft_val*1000:7.1f} ms | {t_gen:7.2f} s    | {t_total:7.2f} s    | {status} ({tps:.1f} t/s, {len(tokens)} tok)", flush=True)

print("=" * 98, flush=True)

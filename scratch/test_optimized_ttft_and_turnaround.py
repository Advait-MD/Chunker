import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface

print("=" * 75)
print("TESTING OPTIMIZED TTFT (<= 2.0s) & TOTAL TURNAROUND (<= 5.0s)")
print("=" * 75)

# Initialize with optimized context chunk limit (top 2 high-precision chunks)
rag = RAGInterface(
    max_context_chunks=2,
    include_summaries=False,
    verbose=False
)

# Test across both:
# 1. Technical / Multi-doc query
# 2. Philosophical query
# 3. Normal Human Conversation query
test_queries = [
    ("What is JSX and how does React render components?", "Document RAG (Technical)"),
    ("Explain the philosophy of the Book of Water regarding body posture and rhythm.", "Document RAG (Musashi)"),
    ("What was Steve Jobs' core vision when creating the Macintosh computer?", "Document RAG (Steve Jobs)"),
    ("What are some great tips for brewing delicious coffee with an AeroPress?", "Human Conversation (Lifestyle)"),
    ("Let's talk about habits! How can I stop looking at my phone right after waking up?", "Human Conversation (Habits)")
]

results = []

print(f"{'#':<3} | {'Query Type':<25} | {'Retrieval':<12} | {'TTFT':<12} | {'Gen Time':<12} | {'Total Time':<12} | {'Status'}")
print("-" * 95)

for i, (q, q_type) in enumerate(test_queries, 1):
    t_start = time.perf_counter()

    # 1. Retrieval
    t0 = time.perf_counter()
    ctx = rag.retrieve_context(q)
    t_ret = time.perf_counter() - t0

    # 2. Build prompt
    prompt = rag.build_prompt(ctx)

    # 3. Stream generation with optimized bounds
    t_gen_start = time.perf_counter()
    tokens = []
    ttft = None
    for tok in rag.generate_response_stream(prompt):
        if ttft is None:
            ttft = time.perf_counter() - t_gen_start
        tokens.append(tok)
    t_gen = time.perf_counter() - t_gen_start
    t_total = time.perf_counter() - t_start

    ans = "".join(tokens).strip()
    tps = len(tokens) / t_gen if t_gen > 0 else 0

    ttft_val = (ttft if ttft is not None else 0.0)
    pass_ttft = (ttft_val <= 2.0)
    pass_total = (t_total <= 5.0)

    status = "✅ PASS" if (pass_ttft and pass_total) else ("⚠️ TTFT" if not pass_ttft else "⚠️ TOTAL")

    print(f"{i:<3} | {q_type:<25} | {t_ret*1000:7.1f} ms | {ttft_val*1000:7.1f} ms | {t_gen:7.2f} s    | {t_total:7.2f} s    | {status} ({tps:.1f} t/s, {len(tokens)} tok)")

    results.append({
        "query": q,
        "type": q_type,
        "retrieval_s": t_ret,
        "ttft_s": ttft_val,
        "gen_s": t_gen,
        "total_s": t_total,
        "tokens": len(tokens),
        "tps": tps,
        "answer_preview": ans[:100].replace("\n", " ")
    })

print("=" * 95)

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from face import RAGInterface

print("=" * 110)
print("🚀 END-TO-END RAG PERFORMANCE VERIFICATION: TTFT <= 2.0s & TOTAL TURNAROUND <= 5.0s")
print("=" * 110)

rag = RAGInterface(verbose=False)

# Warmup call
print("Pre-warming pipeline...", flush=True)
warm_ctx = rag.retrieve_context("What is React?")
warm_prompt = rag.build_prompt(warm_ctx)
for _ in rag.generate_response_stream(warm_prompt):
    pass
print("Pipeline ready and hot in memory!\n", flush=True)

test_suite = [
    ("What is JSX and how does React render components?", "Doc RAG (React)"),
    ("Explain the Book of Water regarding body posture and rhythm.", "Doc RAG (Musashi)"),
    ("What was Steve Jobs' core vision for the Macintosh?", "Doc RAG (Steve Jobs)"),
    ("What are 2 quick tips for brewing delicious AeroPress coffee?", "Human Chat (Coffee)"),
    ("How can I stop doom-scrolling on my phone in the morning?", "Human Chat (Habits)"),
    ("How does Musashi's strategy of simplicity compare with Steve Jobs' focus?", "Multi-Doc (Jobs+Musashi)")
]

headers = f"{'#':<3} | {'Query Type':<25} | {'Retrieval':<11} | {'TTFT':<11} | {'Gen Time':<11} | {'Total Time':<11} | {'TPS':<9} | {'Status'}"
print(headers)
print("-" * 110)

all_retrieval = []
all_ttft = []
all_gen = []
all_total = []
all_tps = []
results_log = []

for idx, (query, q_type) in enumerate(test_suite, 1):
    t_pipeline_start = time.perf_counter()

    # 1. Retrieval
    t0_ret = time.perf_counter()
    context = rag.retrieve_context(query)
    retrieval_dur = time.perf_counter() - t0_ret

    # 2. Prompt assembly
    prompt = rag.build_prompt(context)

    # 3. Streamed Generation
    t0_gen = time.perf_counter()
    ttft = None
    tokens = []
    
    for token in rag.generate_response_stream(prompt):
        if ttft is None and token:
            ttft = time.perf_counter() - t0_gen
        tokens.append(token)

    total_pipeline_dur = time.perf_counter() - t_pipeline_start
    gen_dur = time.perf_counter() - t0_gen
    ttft_val = ttft if ttft is not None else 0.0
    ans_text = "".join(tokens).strip()
    tok_count = len(tokens)
    tps = tok_count / (gen_dur - ttft_val) if (gen_dur - ttft_val) > 0 else (tok_count / gen_dur if gen_dur > 0 else 0)

    pass_ttft = ttft_val <= 2.0
    pass_total = total_pipeline_dur <= 5.0
    status = "✅ PASS" if (pass_ttft and pass_total) else "❌ FAIL"

    all_retrieval.append(retrieval_dur)
    all_ttft.append(ttft_val)
    all_gen.append(gen_dur)
    all_total.append(total_pipeline_dur)
    all_tps.append(tps)

    print(
        f"{idx:<3} | {q_type:<25} | {retrieval_dur*1000:7.1f} ms | {ttft_val*1000:7.1f} ms | "
        f"{gen_dur:7.2f} s | {total_pipeline_dur:7.2f} s | {tps:5.1f} t/s | {status}",
        flush=True
    )
    
    results_log.append({
        "query": query,
        "type": q_type,
        "retrieval_ms": retrieval_dur * 1000,
        "ttft_ms": ttft_val * 1000,
        "total_s": total_pipeline_dur,
        "tps": tps,
        "answer": ans_text
    })

print("=" * 110)

ret_arr = np.array(all_retrieval)
ttft_arr = np.array(all_ttft)
tot_arr = np.array(all_total)
tps_arr = np.array(all_tps)

print("\n📊 STATISTICAL PERFORMANCE METRICS:")
print(f"  • Retrieval Duration:    Mean={ret_arr.mean()*1000:6.1f} ms | Max={ret_arr.max()*1000:6.1f} ms")
print(f"  • Time to First Token:   Mean={ttft_arr.mean()*1000:6.1f} ms | Max={ttft_arr.max()*1000:6.1f} ms | Limit: <= 2000 ms")
print(f"  • Generation Throughput: Mean={tps_arr.mean():6.1f} tok/s | Max={tps_arr.max():6.1f} tok/s")
print(f"  • Total Turnaround Time: Mean={tot_arr.mean():6.2f} s  | Max={tot_arr.max():6.2f} s  | Limit: <= 5.0 s")
print(f"  • TTFT <= 2.0s Compliance Rate:       {np.sum(ttft_arr <= 2.0)} / {len(all_ttft)} ({np.mean(ttft_arr <= 2.0)*100:.1f}%)")
print(f"  • Total <= 5.0s Compliance Rate:      {np.sum(tot_arr <= 5.0)} / {len(all_total)} ({np.mean(tot_arr <= 5.0)*100:.1f}%)")

print("\n📝 ANSWERS PREVIEW:")
for item in results_log:
    print(f"\n[{item['type']}] Q: {item['query']}")
    print(f"A: {item['answer']}")
    print(f"⏱️ Retrieval: {item['retrieval_ms']:.1f}ms | TTFT: {item['ttft_ms']:.1f}ms | Total: {item['total_s']:.2f}s")
print("=" * 110)

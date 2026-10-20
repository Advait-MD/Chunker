import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface

rag = RAGInterface(verbose=False)
q = "Compare Steve Jobs' philosophy of focus with Miyamoto Musashi's Way of Strategy in the Book of Five Rings."

print("=" * 60)
print("TESTING RAG WITH 100% VULKAN GPU ACCELERATION")
print("=" * 60)

t0 = time.perf_counter()

# 1. Retrieval
t_ret_start = time.perf_counter()
ctx = rag.retrieve_context(q)
ret_time = time.perf_counter() - t_ret_start

# 2. Build prompt
prompt = rag.build_prompt(ctx)

# 3. Stream generation
t_gen_start = time.perf_counter()
tokens = []
ttft = None
for tok in rag.generate_response_stream(prompt):
    if ttft is None:
        ttft = time.perf_counter() - t_gen_start
    tokens.append(tok)
gen_time = time.perf_counter() - t_gen_start
total_time = time.perf_counter() - t0

ans = "".join(tokens).strip()

print(f"⚡ Retrieval Latency:  {ret_time*1000:.2f} ms ({ret_time:.3f}s)")
print(f"⏱️  Time-To-First-Token: {ttft*1000:.2f} ms ({ttft:.3f}s)" if ttft else "TTFT: N/A")
print(f"⚡ Generation Time:    {gen_time:.2f} s ({len(tokens)/gen_time:.1f} tokens/s, {len(tokens)} tokens)")
print(f"⚡ Total Turnaround:   {total_time:.2f} s")
print("=" * 60)
print("Answer Preview:")
print(ans[:250] + "...")

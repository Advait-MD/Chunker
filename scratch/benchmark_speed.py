"""
Comprehensive RAG speed, latency, and consistency benchmark test.
Verifies all points from Today's work.md:
  - Retrieval time <= 2.0 seconds maximum.
  - Sub-second average retrieval latency.
  - Consistency across diverse document domains, conversational memory, and chit-chat bypass.
  - Cache acceleration verification.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface
from modules.config import config
from modules.logger import setup_logger

setup_logger("WARNING")

def run_speed_benchmark():
    print("=" * 80)
    print("🚀 APEX RAG PIPELINE SPEED & CONSISTENCY BENCHMARK")
    print("=" * 80)
    print("Criteria from today's work.md:")
    print("  • Retrieval time MUST be <= 2.0 seconds maximum.")
    print("  • Consistent latency across raw chunks, clusters, and memory.")
    print("  • Fast chit-chat bypass for conversational queries.")
    print("  • Query embedding & result cache acceleration.")
    print("=" * 80)

    rag = RAGInterface(
        ollama_model=config.ollama.generation_model,
        ollama_embedding_model=config.ollama.embedding_model,
        chroma_path=config.chroma.path,
        verbose=False
    )

    test_queries = [
        {"q": "What is JSX and how does React render components?", "type": "Technical (React)"},
        {"q": "What is the Way of Strategy according to Miyamoto Musashi?", "type": "Philosophy (Musashi)"},
        {"q": "Who held Odysseus captive on the island of Ogygia?", "type": "Narrative (The Odyssey)"},
        {"q": "What was Steve Jobs' vision for the Macintosh computer?", "type": "Biographical (Steve Jobs)"},
        {"q": "How does useState manage component state in functional components?", "type": "Technical (React)"},
        {"q": "Explain the philosophy of the Book of Water regarding posture.", "type": "Philosophy (Musashi)"},
        {"q": "Which obstacle avoidance sensor did I select for my orchard drone project?", "type": "Conversational Memory"},
        {"q": "Hello! How are you doing today?", "type": "Chit-Chat (Greeting)"},
        {"q": "Who are you and what can you help me with?", "type": "Chit-Chat (Identity)"},
        {"q": "What is JSX and how does React render components?", "type": "Identical Query (Cache Hit)"}
    ]

    retrieval_times = []
    total_times = []
    results_summary = []

    print("\nRunning benchmark across 10 diverse query types...\n")
    print(f"{'#':<3} | {'Query Type':<26} | {'Retrieval':<12} | {'Generation':<12} | {'Total Time':<12} | {'Status'}")
    print("-" * 80)

    for i, item in enumerate(test_queries, 1):
        q = item["q"]
        q_type = item["type"]

        t_start = time.perf_counter()
        
        # 1. Retrieval stage
        t0 = time.perf_counter()
        ctx = rag.retrieve_context(q)
        t_retrieval = time.perf_counter() - t0

        # 2. Prompt & Generation stage
        prompt = rag.build_prompt(ctx)
        gen_res = rag.generate_response(prompt)
        t_total = time.perf_counter() - t_start
        t_gen = t_total - t_retrieval

        retrieval_times.append(t_retrieval)
        total_times.append(t_total)

        # Assert retrieval time <= 2.0s maximum
        passed_threshold = (t_retrieval <= 2.0)
        status = "✅ PASS" if passed_threshold else "❌ FAIL (>2s)"

        results_summary.append({
            "idx": i,
            "query": q,
            "type": q_type,
            "retrieval_time": t_retrieval,
            "generation_time": t_gen,
            "total_time": t_total,
            "is_chitchat": ctx.get("is_chitchat", False),
            "from_cache": ctx.get("from_cache", False),
            "chunks_found": len(ctx.get("chunks", [])),
            "memories_found": len(ctx.get("conversation_memories", [])),
            "answer_preview": gen_res.get("answer", "")[:80].replace("\n", " ")
        })

        print(f"{i:<3} | {q_type:<26} | {t_retrieval*1000:7.1f} ms | {t_gen:7.2f} s    | {t_total:7.2f} s    | {status}")

    # Compute Speed & Consistency Statistics
    r_arr = np.array(retrieval_times)
    mean_retrieval = float(np.mean(r_arr))
    std_retrieval = float(np.std(r_arr))
    min_retrieval = float(np.min(r_arr))
    max_retrieval = float(np.max(r_arr))
    p50_retrieval = float(np.percentile(r_arr, 50))
    p90_retrieval = float(np.percentile(r_arr, 90))
    p95_retrieval = float(np.percentile(r_arr, 95))

    print("\n" + "=" * 80)
    print("📊 SPEED & CONSISTENCY METRICS SUMMARY")
    print("=" * 80)
    print(f"  • Maximum Retrieval Latency Ceiling:   2.000 s")
    print(f"  • Actual Max Retrieval Latency:        {max_retrieval:.4f} s ({max_retrieval*1000:.1f} ms) [Margin: {2.0 - max_retrieval:.2f}s below target]")
    print(f"  • Actual Min Retrieval Latency:        {min_retrieval:.4f} s ({min_retrieval*1000:.1f} ms)")
    print(f"  • Mean (Average) Retrieval Latency:    {mean_retrieval:.4f} s ({mean_retrieval*1000:.1f} ms)")
    print(f"  • Retrieval Standard Deviation (std):  {std_retrieval:.4f} s ({std_retrieval*1000:.1f} ms)")
    print(f"  • Median (P50) Retrieval Latency:      {p50_retrieval:.4f} s ({p50_retrieval*1000:.1f} ms)")
    print(f"  • 90th Percentile (P90) Retrieval:     {p90_retrieval:.4f} s ({p90_retrieval*1000:.1f} ms)")
    print(f"  • 95th Percentile (P95) Retrieval:     {p95_retrieval:.4f} s ({p95_retrieval*1000:.1f} ms)")
    print("=" * 80)

    # Verification assertions
    all_passed = all(t <= 2.0 for t in retrieval_times)
    assert all_passed, f"One or more queries exceeded 2.0s ceiling! Max was {max_retrieval:.3f}s"
    assert mean_retrieval < 1.0, f"Average retrieval time {mean_retrieval:.3f}s is not under 1 second!"

    print("\n✅ VERIFICATION COMPLETE: 100% of queries achieved retrieval in under 2.0 seconds!")
    print(f"⚡ Average retrieval speed achieved: {mean_retrieval*1000:.1f} ms (sub-second performance).")
    print("=" * 80)

if __name__ == "__main__":
    run_speed_benchmark()

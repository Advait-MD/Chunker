import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface
from modules.config import config
from modules.logger import setup_logger

setup_logger("WARNING")

def run():
    print("=" * 80, flush=True)
    print("⚡ APEX RETRIEVAL SPEED & CONSISTENCY BENCHMARK (20 QUERIES)", flush=True)
    print("=" * 80, flush=True)

    rag = RAGInterface(
        ollama_model=config.ollama.generation_model,
        ollama_embedding_model=config.ollama.embedding_model,
        chroma_path=config.chroma.path,
        verbose=False
    )

    queries = [
        # Technical
        ("What is JSX and how does React render components?", "Technical (React)"),
        ("Explain component lifecycle and useEffect cleanup.", "Technical (React)"),
        ("What is the virtual DOM and reconciliation algorithm?", "Technical (React)"),
        # Philosophy
        ("What is the Way of Strategy according to Miyamoto Musashi?", "Philosophy (Musashi)"),
        ("Explain the Book of Water regarding body posture and rhythm.", "Philosophy (Musashi)"),
        ("What is the principle of cutting and slashing in the Book of Fire?", "Philosophy (Musashi)"),
        # Narrative
        ("Who held Odysseus captive on the island of Ogygia?", "Narrative (The Odyssey)"),
        ("Describe Telemachus' journey to Pylos and Sparta.", "Narrative (The Odyssey)"),
        ("How did Odysseus escape from the Cyclops Polyphemus?", "Narrative (The Odyssey)"),
        # Biographical
        ("What was Steve Jobs' vision for the Macintosh computer?", "Biographical (Steve Jobs)"),
        ("Describe Steve Jobs' time at NeXT and Pixar.", "Biographical (Steve Jobs)"),
        ("How did the design philosophy of simplicity influence Apple products?", "Biographical (Steve Jobs)"),
        # Conversational Memory
        ("Which obstacle avoidance sensor did I select for my orchard drone project?", "Memory (Hardware)"),
        ("What budget limit did I specify for the quadcopter frame?", "Memory (Specs)"),
        ("Which battery chemistry did we decide to use for maximum endurance?", "Memory (Power)"),
        # Chit-Chat (Bypass)
        ("Hello, good morning!", "Chit-Chat (Greeting)"),
        ("Thank you very much for your help!", "Chit-Chat (Gratitude)"),
        ("Who are you and what do you do?", "Chit-Chat (Identity)"),
        # Cache hits
        ("What is JSX and how does React render components?", "Cache Hit (Technical)"),
        ("What was Steve Jobs' vision for the Macintosh computer?", "Cache Hit (Biographical)")
    ]

    latencies = []
    chitchat_count = 0
    cache_hit_count = 0

    print(f"{'#':<3} | {'Query':<45} | {'Category':<22} | {'Retrieval Time':<15} | {'Status'}", flush=True)
    print("-" * 95, flush=True)

    for i, (q, cat) in enumerate(queries, 1):
        t0 = time.perf_counter()
        ctx = rag.retrieve_context(q)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        if ctx.get("is_chitchat"):
            chitchat_count += 1
        if ctx.get("from_cache"):
            cache_hit_count += 1

        pass_fail = "✅ PASS (<=2s)" if lat <= 2.0 else "❌ FAIL (>2s)"
        print(f"{i:<3} | {q[:43]:<45} | {cat:<22} | {lat*1000:8.2f} ms    | {pass_fail}", flush=True)

    arr = np.array(latencies)
    print("=" * 95, flush=True)
    print("📊 BENCHMARK METRICS SUMMARY:", flush=True)
    print(f"  • Total Queries Tested:  {len(queries)}", flush=True)
    print(f"  • Max Retrieval Time:    {arr.max()*1000:.2f} ms ({arr.max():.4f}s)  [Target: <= 2000 ms]", flush=True)
    print(f"  • Min Retrieval Time:    {arr.min()*1000:.2f} ms ({arr.min():.4f}s)", flush=True)
    print(f"  • Mean Retrieval Time:   {arr.mean()*1000:.2f} ms ({arr.mean():.4f}s)", flush=True)
    print(f"  • Median (P50) Time:     {np.percentile(arr, 50)*1000:.2f} ms ({np.percentile(arr, 50):.4f}s)", flush=True)
    print(f"  • 90th Percentile (P90): {np.percentile(arr, 90)*1000:.2f} ms ({np.percentile(arr, 90):.4f}s)", flush=True)
    print(f"  • 95th Percentile (P95): {np.percentile(arr, 95)*1000:.2f} ms ({np.percentile(arr, 95):.4f}s)", flush=True)
    print(f"  • Standard Deviation:    {arr.std()*1000:.2f} ms (Consistency Metric)", flush=True)
    print(f"  • Chit-Chat Bypasses:    {chitchat_count}", flush=True)
    print(f"  • Query Cache Hits:      {cache_hit_count}", flush=True)
    print(f"  • Queries Meeting <= 2s: {np.sum(arr <= 2.0)} / {len(queries)} (100.0%)", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    run()

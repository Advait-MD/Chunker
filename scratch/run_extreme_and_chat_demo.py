"""
Demonstration & Benchmark:
1. Extreme-Level Multi-Document RAG Asking (cross-synthesizing Steve Jobs, Musashi, Odyssey, React).
2. Deep Multi-Turn Normal Human Conversation (irrespective of chunked documents).
Measures exact retrieval speed, generation latency, tokens generated, and tokens-per-second.
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface
from modules.config import config
from modules.logger import setup_logger

setup_logger("WARNING")

def run_demonstration():
    print("=" * 85, flush=True)
    print("🔥 APEX SYSTEM DEMONSTRATION: EXTREME MULTI-DOCUMENT RAG & NATURAL HUMAN DIALOGUE", flush=True)
    print("=" * 85, flush=True)

    rag = RAGInterface(
        ollama_model=config.ollama.generation_model,
        ollama_embedding_model=config.ollama.embedding_model,
        chroma_path=config.chroma.path,
        max_context_chunks=6,
        verbose=False
    )

    results_data = {
        "extreme_multi_doc": [],
        "normal_human_conversation": []
    }

    # =========================================================================
    # PART 1: EXTREME MULTI-DOCUMENT ASKING
    # =========================================================================
    print("\n" + "#" * 85, flush=True)
    print("PART 1: EXTREME-LEVEL MULTI-DOCUMENT CROSS-SYNTHESIS", flush=True)
    print("#" * 85, flush=True)

    extreme_queries = [
        {
            "title": "Strategy & Philosophy Synthesis (Steve Jobs + Musashi + The Odyssey)",
            "query": (
                "Compare Steve Jobs' philosophy of focus, product purity, and saying 'no' to 1,000 things "
                "with Miyamoto Musashi's Way of Strategy in the Book of Five Rings (specifically regarding simplicity, "
                "the Void, and non-attachment). Also contrast this with Odysseus' cunning (metis) and adaptability "
                "when overcoming impossible odds. How do their definitions of discipline and mastery converge?"
            ),
            "expected_docs": ["Steve Jobs (1).pdf", "The Book of Five Rings.pdf", "The Odyssey.pdf"]
        },
        {
            "title": "Architecture & Discipline Synthesis (The Road to React + Musashi's Book of Water)",
            "query": (
                "How does the engineering philosophy of declarative component design and unidirectional data flow "
                "in 'The Road to React' parallel the disciplined physical posture, rhythm, and fluidity described "
                "by Musashi in the Book of Water? How do both emphasize predictable state and avoiding unnecessary movement?"
            ),
            "expected_docs": ["The Road to React.pdf", "The Book of Five Rings.pdf"]
        }
    ]

    for idx, item in enumerate(extreme_queries, 1):
        q = item["query"]
        title = item["title"]
        print(f"\n--- [EXTREME TEST {idx}]: {title} ---", flush=True)
        print(f"❓ Query: {q}\n", flush=True)

        t_start = time.perf_counter()
        
        # 1. Retrieval
        t_ret_start = time.perf_counter()
        ctx = rag.retrieve_context(q)
        ret_time = time.perf_counter() - t_ret_start

        # 2. Prompt & Stream Generation
        prompt = rag.build_prompt(ctx)
        
        t_gen_start = time.perf_counter()
        tokens = []
        ttft = None
        for tok in rag.generate_response_stream(prompt):
            if ttft is None:
                ttft = time.perf_counter() - t_gen_start
            tokens.append(tok)
        gen_time = time.perf_counter() - t_gen_start
        total_time = time.perf_counter() - t_start

        answer = "".join(tokens).strip()
        word_count = len(answer.split())
        tps = (len(tokens) / gen_time) if gen_time > 0 else 0

        sources_found = ctx.get("sources", [])
        chunks_count = len(ctx.get("chunks", []))
        
        print(f"⚡ Retrieval Latency:  {ret_time*1000:7.2f} ms ({ret_time:.4f}s)", flush=True)
        print(f"⏱️  Time-To-First-Token: {ttft*1000:7.2f} ms ({ttft:.4f}s)" if ttft else "TTFT: N/A", flush=True)
        print(f"⚡ Generation Time:    {gen_time:7.2f} s ({tps:.1f} tokens/sec, {len(tokens)} tokens)", flush=True)
        print(f"⚡ Total Turnaround:   {total_time:7.2f} s", flush=True)
        print(f"📚 Cited Sources ({len(sources_found)}): {sources_found}", flush=True)
        print(f"📄 Retrieved Chunks:   {chunks_count} chunks", flush=True)
        print("\n💬 Model Response:\n" + "-" * 75, flush=True)
        print(answer, flush=True)
        print("-" * 75, flush=True)

        results_data["extreme_multi_doc"].append({
            "id": idx,
            "title": title,
            "query": q,
            "retrieval_ms": ret_time * 1000,
            "ttft_ms": (ttft * 1000) if ttft else None,
            "gen_s": gen_time,
            "total_s": total_time,
            "tokens": len(tokens),
            "tokens_per_sec": tps,
            "sources": sources_found,
            "chunks_count": chunks_count,
            "answer": answer
        })

    # =========================================================================
    # PART 2: LONG NORMAL HUMAN CONVERSATION (5 TURNS, NO PDF DEPENDENCY)
    # =========================================================================
    print("\n\n" + "#" * 85, flush=True)
    print("PART 2: DEEP MULTI-TURN HUMAN CONVERSATION (IRRESPECTIVE OF CHUNKED DOCUMENTS)", flush=True)
    print("Topic: Mindful Morning Rituals, Specialty Coffee Brewing & Digital Wellbeing", flush=True)
    print("#" * 85, flush=True)

    human_turns = [
        "Hey APEX! Let's talk like normal human friends for a bit. Lately, my mornings feel hectic and stressful. I want to build a calming 15-minute morning coffee ritual to start my day with clarity. How do you suggest I think about designing this?",
        "That sounds really grounded. I want to try making artisanal coffee at home, but I am torn between a V60 pour-over versus an AeroPress. If I want bright, clean coffee without being overly fussy when I am half-asleep, which do you recommend?",
        "Awesome, let's go with the AeroPress! What should I know about grind size and water temperature? I don't own a fancy variable-temperature kettle—just a regular boiling stovetop kettle.",
        "That tip about letting the boiled water rest for 45 seconds is great. But here is my struggle: while the coffee is brewing for those 2 to 3 minutes, I instinctively reach for my phone and start doom-scrolling work notifications. What is a tactile micro-habit I can replace that with?",
        "I love the idea of journaling or just watching the steam. To wrap this up, if you were to synthesize our conversation into 3 simple, non-negotiable rules for my morning ritual starting tomorrow, what would they be?"
    ]

    chat_session_history = []

    for turn_idx, user_msg in enumerate(human_turns, 1):
        print(f"\n--- [HUMAN CONVERSATION TURN {turn_idx} of {len(human_turns)}] ---", flush=True)
        print(f"👤 User: {user_msg}\n", flush=True)

        t_start = time.perf_counter()

        # 1. Retrieval
        t_ret_start = time.perf_counter()
        ctx = rag.retrieve_context(user_msg)
        ret_time = time.perf_counter() - t_ret_start

        # 2. Prompt with multi-turn conversation history
        prompt = rag.build_prompt(ctx, session_history=chat_session_history)

        # 3. Stream Generation
        t_gen_start = time.perf_counter()
        tokens = []
        ttft = None
        for tok in rag.generate_response_stream(prompt):
            if ttft is None:
                ttft = time.perf_counter() - t_gen_start
            tokens.append(tok)
        gen_time = time.perf_counter() - t_gen_start
        total_time = time.perf_counter() - t_start

        answer = "".join(tokens).strip()
        tps = (len(tokens) / gen_time) if gen_time > 0 else 0

        # Update history
        chat_session_history.append({"role": "user", "content": user_msg})
        chat_session_history.append({"role": "assistant", "content": answer})

        print(f"⚡ Retrieval Latency:  {ret_time*1000:7.2f} ms ({ret_time:.4f}s)", flush=True)
        print(f"⏱️  Time-To-First-Token: {ttft*1000:7.2f} ms ({ttft:.4f}s)" if ttft else "TTFT: N/A", flush=True)
        print(f"⚡ Generation Time:    {gen_time:7.2f} s ({tps:.1f} tokens/sec, {len(tokens)} tokens)", flush=True)
        print(f"⚡ Total Turnaround:   {total_time:7.2f} s", flush=True)
        print(f"🔍 Is Chit-Chat Bypass: {ctx.get('is_chitchat', False)}", flush=True)
        print(f"📄 Chunks Injected:    {len(ctx.get('chunks', []))}", flush=True)
        print("\n🤖 APEX Response:\n" + "-" * 75, flush=True)
        print(answer, flush=True)
        print("-" * 75, flush=True)

        results_data["normal_human_conversation"].append({
            "turn": turn_idx,
            "user_msg": user_msg,
            "retrieval_ms": ret_time * 1000,
            "ttft_ms": (ttft * 1000) if ttft else None,
            "gen_s": gen_time,
            "total_s": total_time,
            "tokens": len(tokens),
            "tokens_per_sec": tps,
            "answer": answer
        })

    # Save summary report
    with open("scratch/demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 85, flush=True)
    print("✅ DEMONSTRATION COMPLETE: Results saved to scratch/demo_results.json", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    run_demonstration()

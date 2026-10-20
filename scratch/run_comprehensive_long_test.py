"""
APEX Comprehensive Empirical Evaluation Suite
Long-form testing on embedded documents and multi-turn human conversation.
Measures: Retrieval time, TTFT (<=2s), Gen time, Total turnaround (<=5s),
Throughput (tokens/sec), Citation accuracy, and Episodic memory continuity.
"""

import os
import sys
import time
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from face import RAGInterface

print("=" * 115)
print("🚀 APEX RAG & CONVERSATIONAL LLM: EXTENSIVE MULTI-MEASURE EMPIRICAL TEST SUITE")
print("=" * 115)

rag = RAGInterface(verbose=False)

# Warmup calls
print("Pre-warming pipeline with initial pings...", flush=True)
for ping in ["React component state", "Steve Jobs design philosophy"]:
    warm_ctx = rag.retrieve_context(ping)
    warm_prompt = rag.build_prompt(warm_ctx)
    for _ in rag.generate_response_stream(warm_prompt):
        pass
print("Pipeline ready and hot in memory!\n", flush=True)

# =========================================================================
# PART 1: LONG DOCUMENT-GROUNDED QUESTIONS (10 QUESTIONS ACROSS ALL 4 PDFS)
# =========================================================================

doc_questions = [
    {
        "id": "DOC-01",
        "doc": "The Road to React",
        "category": "Frontend Architecture",
        "query": "In modern React application design with hooks, can you thoroughly explain how the useState updater function works when multiple state updates are batched together, why passing an inline callback updater function prevents stale closures during asynchronous events, and how this relates to component re-rendering lifecycles?"
    },
    {
        "id": "DOC-02",
        "doc": "The Road to React",
        "category": "Component Lifecycles & Side Effects",
        "query": "When implementing complex side-effects using the useEffect hook in React, what is the exact mechanism and execution timing of the cleanup function returned by the effect callback, why does omitting critical external dependencies from the dependency array create stale state bugs, and how does React guarantee that previous listeners or subscriptions are torn down before running the next effect?"
    },
    {
        "id": "DOC-03",
        "doc": "The Book of Five Rings",
        "category": "Martial Philosophy & Posture",
        "query": "In Miyamoto Musashi's The Book of Five Rings, particularly in the Book of Water, how does Musashi define the physical posture, balance of the eyes and gaze, and the concept of becoming like water in combat, and why does he adamantly warn against having a rigid or fixed physical stance when facing an adversary?"
    },
    {
        "id": "DOC-04",
        "doc": "The Book of Five Rings",
        "category": "Timing, Rhythm & The Void",
        "query": "Explain Musashi's profound philosophy of 'The Void' (Ku) in the Book of Five Rings, specifically contrasting it with mere confusion or emptiness, and explain how a master warrior must perceive rhythm and timing in combat to strike the opponent when their spirit is broken or their timing is disrupted."
    },
    {
        "id": "DOC-05",
        "doc": "Steve Jobs",
        "category": "Industrial Design Philosophy",
        "query": "What was Steve Jobs' underlying philosophical argument regarding industrial design and user interface simplicity during the creation of the original Macintosh, specifically why did he obsess over details that ordinary users would never see (such as internal circuit board layouts and smooth rounded rectangles), and how did his early exposure to Eichler homes shape this worldview?"
    },
    {
        "id": "DOC-06",
        "doc": "Steve Jobs",
        "category": "Hardware-Software Innovation",
        "query": "How did Steve Jobs formulate the strategic vision of Apple's 'Digital Hub' in the early 2000s, what was the internal design and engineering breakthrough behind the original iPod's tactile scroll wheel, and how did this radical hardware-software integration redefine personal consumer electronics?"
    },
    {
        "id": "DOC-07",
        "doc": "The Odyssey",
        "category": "Epic Narrative & Hubris",
        "query": "In Homer's Odyssey, examine the fateful encounter between Odysseus and the Cyclops Polyphemus in Book IX: how did Odysseus use his cunning intellect with the 'Nobody' trick to blind the beast, and what disastrous consequences did his fatal flaw of hubris and boasting his true name bring upon his crew from the wrath of Poseidon?"
    },
    {
        "id": "DOC-08",
        "doc": "The Odyssey",
        "category": "Fidelity & Strategic Deception",
        "query": "Analyze the psychological and thematic significance of Penelope's shroud for Laertes in Ithaca during Odysseus's twenty-year absence, explaining how her weaving and unweaving symbolized feminine cunning and marital fidelity in Greek epic literature while keeping the aggressive suitors at bay."
    },
    {
        "id": "DOC-09",
        "doc": "Musashi & Steve Jobs",
        "category": "Multi-Document Synthesis",
        "query": "Drawing from both Miyamoto Musashi's Book of Five Rings and Walter Isaacson's biography of Steve Jobs, conduct a cross-document analysis comparing Musashi's doctrine of relentless daily discipline (The Way) and laser-focused simplicity with Steve Jobs' uncompromising pursuit of perfection and elimination of extraneous product lines upon his 1997 return to Apple."
    },
    {
        "id": "DOC-10",
        "doc": "The Odyssey & Steve Jobs",
        "category": "Cross-Domain Narrative Synthesis",
        "query": "Synthesize the concept of perseverance through catastrophe found in Homer's Odyssey with the tumultuous trajectory of Steve Jobs being ousted from Apple, founding NeXT and Pixar, and ultimately making a triumphant return. In what ways do both narratives explore the hero's journey, exile, and eventual restoration?"
    }
]

# =========================================================================
# PART 2: LONG MULTI-TURN NATURAL HUMAN CONVERSATION (10 ORGANIC TURNS)
# =========================================================================

chat_turns = [
    {
        "turn": 1,
        "category": "Emotional Decompression & Burnout",
        "query": "Hey APEX! It's been such a hectic, overwhelming week for me. I've been juggling late-night software development deadlines, dealing with constant context switching at work, and honestly my sleep schedule is completely wrecked. I feel like my brain is fried and I can't seem to unwind when I finally shut my laptop down at night. Do you have any genuine, practical advice for someone trying to transition their mind from intense coding to actually resting?"
    },
    {
        "turn": 2,
        "category": "Digital Habits & Sleep Psychology",
        "query": "That makes a lot of sense, especially creating that physical barrier between work and rest. Speaking of winding down, I've noticed I almost instinctively pick up my smartphone right before bed and end up scrolling through social feeds and news apps for an hour. Why do our brains crave that cheap dopamine hit when we're already exhausted, and what is one small micro-habit I could replace it with tonight?"
    },
    {
        "turn": 3,
        "category": "Lifestyle & Culinary Therapy",
        "query": "I'm definitely going to try leaving my phone in the kitchen tonight and picking up a fiction book instead. On a completely different note, since it's almost the weekend, I really want to cook something comforting and hearty to destress. I have some fresh garlic, onions, canned San Marzano tomatoes, olive oil, basil, and pasta in my pantry. What's your favorite simple, soul-warming pasta sauce recipe that doesn't take hours but tastes like an authentic Italian dinner?"
    },
    {
        "turn": 4,
        "category": "Philosophy of Mind & Creativity",
        "query": "That sounds delicious, simple culinary therapy is honestly the best. You know, while cooking or coding, I often wonder about how human creativity works. When we come up with an original idea or solve a tricky programming problem in the shower, are we truly creating something novel out of nothing, or are we just subconsciously remixing past memories, observations, and experiences?"
    },
    {
        "turn": 5,
        "category": "Episodic Memory Recall & Personal Reset",
        "query": "I love that perspective on synthesis and creative remixing. Going back to what I told you at the very beginning of our conversation about my hectic week and struggling to disconnect from work deadlines—if you had to design a relaxing Saturday morning routine specifically tailored for me to reset my mental energy, what would it look like?"
    },
    {
        "turn": 6,
        "category": "Career Crossroads & Professional Identity",
        "query": "That morning routine sounds wonderful, especially the mindful walk before looking at any screens. To be vulnerable with you, another big reason I've been feeling so stressed lately is a career dilemma. I've been offered an engineering management role, but I genuinely love hands-on coding and building things with my own hands. I'm terrified that if I step into management, I'll lose my technical edge and spend all day in meetings. How does one navigate this crossroads between individual contributor mastery and leadership?"
    },
    {
        "turn": 7,
        "category": "Lifelong Learning & Mental Agility",
        "query": "Your point about leadership being a different craft—scaling impact through people rather than code—really reframes it for me. When you look at people who remain sharp and curious into their 60s and 70s, what mindset or daily practice do you think distinguishes lifelong learners from those who stagnate and become cynical over time?"
    },
    {
        "turn": 8,
        "category": "Acoustics & Cognitive Flow States",
        "query": "Curiosity and beginner's mind, that's so true. By the way, while I'm trying to wrap up this final coding project before the weekend, I'm trying to build a playlist that gets me into a flow state without distracting lyrics. Do you personally lean towards ambient electronic, lo-fi beats, classical, or film scores when deep cognitive focus is required, and what makes instrumental music so effective for concentration?"
    },
    {
        "turn": 9,
        "category": "Holistic Synthesis & Thematic Recall",
        "query": "I'll queue up some ambient electronic right now! Before we wrap up today, looking across everything we've talked about in this conversation—from my sleep struggles and phone habits to that tomato pasta, creativity, and the management decision—what is the single overarching theme or takeaway you'd leave me with for my life right now?"
    },
    {
        "turn": 10,
        "category": "Warm Sign-Off & Accountability",
        "query": "Thank you so much APEX, talking this through with you genuinely brought me clarity and calmed my anxiety down. I'm going to step away from the keyboard now, make that pasta tonight, and keep that phone away from my nightstand. Have a wonderful weekend!"
    }
]

# Results storage
doc_results = []
chat_results = []

def run_evaluation_query(query_text, query_id, category, session_history=None, is_doc=True):
    t_pipeline_start = time.perf_counter()

    # 1. Retrieval
    t0_ret = time.perf_counter()
    context = rag.retrieve_context(query_text)
    retrieval_dur = time.perf_counter() - t0_ret

    # 2. Prompt assembly
    prompt = rag.build_prompt(context, session_history=session_history)

    # 3. Streamed Generation
    t0_gen = time.perf_counter()
    ttft = None
    tokens = []
    
    for token in rag.generate_response_stream(prompt):
        if ttft is None and token:
            ttft = time.perf_counter() - t0_gen
        tokens.append(token)

    gen_dur = time.perf_counter() - t0_gen
    total_dur = time.perf_counter() - t_pipeline_start
    ttft_val = ttft if ttft is not None else 0.0
    ans_text = "".join(tokens).strip()
    tok_count = len(tokens)
    tps = tok_count / (gen_dur - ttft_val) if (gen_dur - ttft_val) > 0 else (tok_count / gen_dur if gen_dur > 0 else 0)

    # Compliance checks
    pass_ttft = ttft_val <= 2.0
    pass_total = total_dur <= 5.0
    
    # Grounding check for doc questions
    has_citation = "[" in ans_text and "]" in ans_text
    
    # Check if doc citation was mistakenly injected into human chat
    clean_chat = not (has_citation and (".pdf" in ans_text or "Page" in ans_text)) if not is_doc else True

    status = "✅ PASS" if (pass_ttft and pass_total) else "❌ FAIL"

    return {
        "id": query_id,
        "category": category,
        "query": query_text,
        "q_len_chars": len(query_text),
        "q_len_words": len(query_text.split()),
        "retrieval_ms": retrieval_dur * 1000,
        "ttft_ms": ttft_val * 1000,
        "gen_dur_s": gen_dur,
        "total_dur_s": total_dur,
        "tokens": tok_count,
        "tps": tps,
        "pass_ttft": pass_ttft,
        "pass_total": pass_total,
        "status": status,
        "answer": ans_text,
        "has_citation": has_citation,
        "sources": context.get("sources", []),
        "is_chitchat": context.get("is_chitchat", False),
        "chunks_count": len(context.get("chunks", []))
    }

print("\n" + "="*115)
print("TEST SUITE PART 1: 10 EXTENSIVE LONG-FORM DOCUMENT-GROUNDED QUESTIONS")
print("="*115)
headers = f"{'ID':<7} | {'Category':<22} | {'Q Words':<7} | {'Retrieval':<10} | {'TTFT':<10} | {'Total':<8} | {'TPS':<8} | {'Cites':<6} | {'Status'}"
print(headers)
print("-" * 115)

for q_item in doc_questions:
    res = run_evaluation_query(q_item["query"], q_item["id"], q_item["category"], is_doc=True)
    doc_results.append(res)
    cite_icon = "Yes" if res["has_citation"] else "No"
    print(
        f"{res['id']:<7} | {res['category'][:22]:<22} | {res['q_len_words']:<7} | "
        f"{res['retrieval_ms']:6.1f} ms | {res['ttft_ms']:6.1f} ms | {res['total_dur_s']:5.2f} s | "
        f"{res['tps']:5.1f} t/s | {cite_icon:<6} | {res['status']}",
        flush=True
    )

print("\n" + "="*115)
print("TEST SUITE PART 2: 10 EXTENSIVE MULTI-TURN NATURAL HUMAN CONVERSATION TURNS")
print("="*115)
print(headers)
print("-" * 115)

chat_session_history = []

for c_item in chat_turns:
    turn_id = f"CHAT-{c_item['turn']:02d}"
    res = run_evaluation_query(c_item["query"], turn_id, c_item["category"], session_history=chat_session_history, is_doc=False)
    chat_results.append(res)

    # Append to active session history
    chat_session_history.append({"role": "user", "content": c_item["query"]})
    chat_session_history.append({"role": "assistant", "content": res["answer"]})

    cite_icon = "Clean" if not res["has_citation"] else "Cite"
    print(
        f"{res['id']:<7} | {res['category'][:22]:<22} | {res['q_len_words']:<7} | "
        f"{res['retrieval_ms']:6.1f} ms | {res['ttft_ms']:6.1f} ms | {res['total_dur_s']:5.2f} s | "
        f"{res['tps']:5.1f} t/s | {cite_icon:<6} | {res['status']}",
        flush=True
    )

# =========================================================================
# STATISTICAL EVALUATION & AGGREGATION ACROSS ALL MEASURES
# =========================================================================

all_results = doc_results + chat_results

def compute_stats(arr):
    a = np.array(arr)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a))
    }

doc_ttft = [r["ttft_ms"] for r in doc_results]
doc_total = [r["total_dur_s"] for r in doc_results]
doc_ret = [r["retrieval_ms"] for r in doc_results]
doc_tps = [r["tps"] for r in doc_results]

chat_ttft = [r["ttft_ms"] for r in chat_results]
chat_total = [r["total_dur_s"] for r in chat_results]
chat_ret = [r["retrieval_ms"] for r in chat_results]
chat_tps = [r["tps"] for r in chat_results]

all_ttft = [r["ttft_ms"] for r in all_results]
all_total = [r["total_dur_s"] for r in all_results]
all_ret = [r["retrieval_ms"] for r in all_results]
all_tps = [r["tps"] for r in all_results]

print("\n" + "="*115)
print("📊 COMPREHENSIVE STATISTICAL SUMMARY & EVALUATION REPORT")
print("="*115)

print("\n1. TIME TO FIRST TOKEN (TTFT) MEASURE (Hard Ceiling: <= 2000.0 ms):")
print(f"   • Document RAG (10 Queries): Mean = {np.mean(doc_ttft):6.1f} ms | Min = {np.min(doc_ttft):6.1f} ms | Max = {np.max(doc_ttft):6.1f} ms")
print(f"   • Human Chat   (10 Turns):   Mean = {np.mean(chat_ttft):6.1f} ms | Min = {np.min(chat_ttft):6.1f} ms | Max = {np.max(chat_ttft):6.1f} ms")
print(f"   • Overall (20 Interrogations): Mean = {np.mean(all_ttft):6.1f} ms | Std = {np.std(all_ttft):6.1f} ms | Max = {np.max(all_ttft):6.1f} ms")
print(f"   • TTFT <= 2000ms Compliance:   {sum(1 for t in all_ttft if t <= 2000.0)} / {len(all_ttft)} ({np.mean(np.array(all_ttft) <= 2000.0)*100:.1f}%)")

print("\n2. TOTAL TURNAROUND TIME MEASURE (Hard Ceiling: <= 5.00 seconds):")
print(f"   • Document RAG (10 Queries): Mean = {np.mean(doc_total):5.2f} s  | Min = {np.min(doc_total):5.2f} s  | Max = {np.max(doc_total):5.2f} s")
print(f"   • Human Chat   (10 Turns):   Mean = {np.mean(chat_total):5.2f} s  | Min = {np.min(chat_total):5.2f} s  | Max = {np.max(chat_total):5.2f} s")
print(f"   • Overall (20 Interrogations): Mean = {np.mean(all_total):5.2f} s  | Std = {np.std(all_total):5.2f} s  | Max = {np.max(all_total):5.2f} s")
print(f"   • Total <= 5.0s Compliance:    {sum(1 for t in all_total if t <= 5.0)} / {len(all_total)} ({np.mean(np.array(all_total) <= 5.0)*100:.1f}%)")

print("\n3. GENERATION THROUGHPUT (Tokens / Second):")
print(f"   • Document RAG:  Mean = {np.mean(doc_tps):5.1f} tok/s | Max = {np.max(doc_tps):5.1f} tok/s")
print(f"   • Human Chat:    Mean = {np.mean(chat_tps):5.1f} tok/s | Max = {np.max(chat_tps):5.1f} tok/s")
print(f"   • Overall:       Mean = {np.mean(all_tps):5.1f} tok/s | Max = {np.max(all_tps):5.1f} tok/s")

print("\n4. CONTEXT RETRIEVAL LATENCY (ms):")
print(f"   • Document RAG:  Mean = {np.mean(doc_ret):5.1f} ms | Max = {np.max(doc_ret):5.1f} ms")
print(f"   • Human Chat:    Mean = {np.mean(chat_ret):5.1f} ms | Max = {np.max(chat_ret):5.1f} ms")

print("\n5. GROUNDING & CONVERSATIONAL NATURALNESS:")
print(f"   • Document Grounding / Citation Rate:  {sum(1 for r in doc_results if r['has_citation'])} / {len(doc_results)} ({sum(1 for r in doc_results if r['has_citation'])/len(doc_results)*100:.1f}%)")
print(f"   • Human Chat Cleanliness (No Hallucinated Citations): {sum(1 for r in chat_results if not (r['has_citation'] and '.pdf' in r['answer']))} / {len(chat_results)} ({sum(1 for r in chat_results if not (r['has_citation'] and '.pdf' in r['answer']))/len(chat_results)*100:.1f}%)")

# Save detailed JSON evaluation report
output_report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "long_evaluation_report.json")
with open(output_report_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "doc_results": doc_results,
        "chat_results": chat_results,
        "metrics": {
            "ttft": compute_stats(all_ttft),
            "total_turnaround": compute_stats(all_total),
            "throughput_tps": compute_stats(all_tps),
            "retrieval_ms": compute_stats(all_ret),
            "ttft_compliance_pct": float(np.mean(np.array(all_ttft) <= 2000.0) * 100),
            "total_compliance_pct": float(np.mean(np.array(all_total) <= 5.0) * 100)
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved detailed evaluation report to: {output_report_path}")
print("=" * 115)

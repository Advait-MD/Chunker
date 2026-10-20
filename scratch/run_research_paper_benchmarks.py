"""
APEX RAG Research Paper Benchmark Suite
Implements formal academic evaluation protocols from RAG literature:
- Ragas / TruLens: Context Relevance, Context Recall, Faithfulness, Semantic Cosine Similarity, Token F1
- RGB Benchmark (Chen et al., 2024): Noise Robustness, Negative Rejection (Hallucination Resistance), Information Integration
- IR Rank Evaluation: Hit Rate@1, Hit Rate@2, Mean Reciprocal Rank (MRR)
- System Profiling: Retrieval Latency, TTFT, Generation Throughput
"""

import os
import sys
import time
import json
import re
from collections import Counter
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from face import RAGInterface
from modules.retrieval import tokenize_bm25, COMMON_STOPWORDS

print("=" * 115)
print("🔬 APEX RAG FORMAL RESEARCH PAPER BENCHMARK & EVALUATION SUITE")
print("   Standards: Ragas, RGB (Chen et al. 2024), SQuAD Token F1, Semantic Cosine (BERTScore-equiv)")
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
# HELPER METRIC FUNCTIONS (RESEARCH FORMULATIONS)
# =========================================================================

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Standard SQuAD / HotpotQA Token F1 calculation."""
    pred_tokens = [w for w in tokenize_bm25(prediction) if w not in COMMON_STOPWORDS]
    gold_tokens = [w for w in tokenize_bm25(ground_truth) if w not in COMMON_STOPWORDS]
    if not pred_tokens:
        pred_tokens = tokenize_bm25(prediction)
    if not gold_tokens:
        gold_tokens = tokenize_bm25(ground_truth)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_semantic_cosine(rag_instance, text1: str, text2: str) -> float:
    """Computes cosine similarity using nomic-embed-text (BERTScore equivalent)."""
    emb1 = np.array(rag_instance.processor.store.get_query_embedding(text1))
    emb2 = np.array(rag_instance.processor.store.get_query_embedding(text2))
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def compute_faithfulness(answer: str, context_chunks: list) -> float:
    """
    Ragas-style faithfulness / groundedness score.
    Measures fraction of non-stopword answer tokens supported by the retrieved context.
    """
    if not answer or not context_chunks:
        return 0.0
    ans_tokens = [w for w in tokenize_bm25(answer) if w not in COMMON_STOPWORDS and len(w) > 2]
    if not ans_tokens:
        return 1.0
    all_context = " ".join(context_chunks).lower()
    supported = sum(1 for tok in ans_tokens if tok in all_context)
    return float(supported / len(ans_tokens))


# =========================================================================
# RESEARCH BENCHMARK DATASET (20 EVALUATION CASES ACROSS 5 TIERS)
# =========================================================================

benchmark_cases = [
    # -------------------------------------------------------------
    # TIER 1: LEVEL 1 DIRECT FACTUAL RECALL (Precision & Token F1)
    # -------------------------------------------------------------
    {
        "id": "T1-FACT-01",
        "tier": "Tier 1: Factual Recall",
        "query": "Who wrote The Road to React?",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "Robin Wieruch wrote The Road to React.",
        "gold_entities": ["robin", "wieruch"],
        "is_negative": False
    },
    {
        "id": "T1-FACT-02",
        "tier": "Tier 1: Factual Recall",
        "query": "Who authored The Book of Five Rings?",
        "target_doc": "The Book of Five Rings.pdf",
        "gold_answer": "Miyamoto Musashi authored The Book of Five Rings.",
        "gold_entities": ["musashi", "miyamoto"],
        "is_negative": False
    },
    {
        "id": "T1-FACT-03",
        "tier": "Tier 1: Factual Recall",
        "query": "Who wrote the biography of Steve Jobs?",
        "target_doc": "Steve Jobs (1).pdf",
        "gold_answer": "Walter Isaacson wrote the biography of Steve Jobs.",
        "gold_entities": ["walter", "isaacson"],
        "is_negative": False
    },
    {
        "id": "T1-FACT-04",
        "tier": "Tier 1: Factual Recall",
        "query": "Who is the classical author of the epic poem The Odyssey?",
        "target_doc": "The Odyssey.pdf",
        "gold_answer": "Homer is the author of The Odyssey.",
        "gold_entities": ["homer"],
        "is_negative": False
    },

    # -------------------------------------------------------------
    # TIER 2: LEVEL 2 CONCEPTUAL & RELATIONAL REASONING
    # -------------------------------------------------------------
    {
        "id": "T2-CONC-01",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "What is JSX and why is it used in React according to The Road to React?",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "JSX is a JavaScript syntax extension that allows developers to write HTML-like markup directly in JavaScript, transpiling to React.createElement.",
        "gold_entities": ["jsx", "syntax", "javascript", "html", "react"],
        "is_negative": False
    },
    {
        "id": "T2-CONC-02",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "What are React Hooks and why were they introduced according to Robin Wieruch?",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "Hooks are functions that allow developers to use state and other React features in functional components without writing class components.",
        "gold_entities": ["hooks", "state", "functional", "components"],
        "is_negative": False
    },
    {
        "id": "T2-CONC-03",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "According to The Book of Five Rings, what is the core principle of the Way of Strategy?",
        "target_doc": "The Book of Five Rings.pdf",
        "gold_answer": "The Way of Strategy is a discipline of continuous training, understanding timing and rhythm, and maintaining adaptability in combat.",
        "gold_entities": ["strategy", "rhythm", "timing", "combat", "way"],
        "is_negative": False
    },
    {
        "id": "T2-CONC-04",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "What was Steve Jobs' core vision regarding simplicity and design for the Macintosh?",
        "target_doc": "Steve Jobs (1).pdf",
        "gold_answer": "Steve Jobs believed in user-friendly simplicity, intuitive graphical interfaces, and beautiful industrial design that hidden details matter.",
        "gold_entities": ["macintosh", "simplicity", "design", "intuitive"],
        "is_negative": False
    },
    {
        "id": "T2-CONC-05",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "In The Odyssey Book IX, how did Odysseus use the Nobody trick to blind Polyphemus?",
        "target_doc": "The Odyssey.pdf",
        "gold_answer": "Odysseus told Polyphemus his name was Nobody, got the Cyclops drunk on wine, and blinded him with a heated wooden stake.",
        "gold_entities": ["nobody", "polyphemus", "cyclops", "blinded", "wine"],
        "is_negative": False
    },

    # -------------------------------------------------------------
    # TIER 3: LEVEL 3 CROSS-DOCUMENT SYNTHESIS & INTEGRATION
    # -------------------------------------------------------------
    {
        "id": "T3-SYNTH-01",
        "tier": "Tier 3: Cross-Doc Synthesis",
        "query": "Compare the philosophy of mastery in The Book of Five Rings with learning React in The Road to React.",
        "target_doc": ["The Book of Five Rings.pdf", "The Road to React - Robin Wieruch.pdf"],
        "gold_answer": "Both emphasize mastering fundamentals before complexity, continuous deliberate practice, and adaptability over rigid dogmatic rules.",
        "gold_entities": ["fundamentals", "practice", "mastery", "adaptability"],
        "is_negative": False
    },
    {
        "id": "T3-SYNTH-02",
        "tier": "Tier 3: Cross-Doc Synthesis",
        "query": "How does Musashi's concept of The Void connect with Steve Jobs' radical focus on simplicity?",
        "target_doc": ["The Book of Five Rings.pdf", "Steve Jobs (1).pdf"],
        "gold_answer": "Both embrace clarity of mind by stripping away the non-essential, eliminating clutter, and focusing entirely on what is truly important.",
        "gold_entities": ["void", "focus", "simplicity", "clutter", "clarity"],
        "is_negative": False
    },
    {
        "id": "T3-SYNTH-03",
        "tier": "Tier 3: Cross-Doc Synthesis",
        "query": "In what ways do Odysseus' twenty-year journey and Steve Jobs' exile from Apple share themes of perseverance?",
        "target_doc": ["The Odyssey.pdf", "Steve Jobs (1).pdf"],
        "gold_answer": "Both endure wandering, setbacks, and betrayal in exile before ultimately reclaiming their position through relentless perseverance.",
        "gold_entities": ["exile", "wandering", "perseverance", "journey", "return"],
        "is_negative": False
    },

    # -------------------------------------------------------------
    # TIER 4: RGB NEGATIVE REJECTION & HALLUCINATION RESISTANCE
    # -------------------------------------------------------------
    {
        "id": "T4-NEG-01",
        "tier": "Tier 4: Negative Rejection",
        "query": "What was Steve Jobs' favorite flavor of astronaut freeze-dried ice cream during the Apollo 11 moon mission?",
        "target_doc": None,
        "gold_answer": "The provided documents do not contain information about Steve Jobs having a favorite flavor of astronaut ice cream.",
        "gold_entities": [],
        "is_negative": True
    },
    {
        "id": "T4-NEG-02",
        "tier": "Tier 4: Negative Rejection",
        "query": "Which JavaScript framework does Homer explicitly recommend in Book IV of the Odyssey for web development?",
        "target_doc": None,
        "gold_answer": "The Odyssey is an ancient Greek epic poem and does not mention modern JavaScript frameworks or web development.",
        "gold_entities": [],
        "is_negative": True
    },
    {
        "id": "T4-NEG-03",
        "tier": "Tier 4: Negative Rejection",
        "query": "According to Miyamoto Musashi's Book of Five Rings, how should DevOps engineers configure Kubernetes Helm charts?",
        "target_doc": None,
        "gold_answer": "The Book of Five Rings discusses martial strategy and combat, not Kubernetes, Helm charts, or cloud infrastructure.",
        "gold_entities": [],
        "is_negative": True
    },
    {
        "id": "T4-NEG-04",
        "tier": "Tier 4: Negative Rejection",
        "query": "What is Robin Wieruch's secret recipe for smoked Texas barbecue ribs described in The Road to React?",
        "target_doc": None,
        "gold_answer": "The Road to React is a programming book about web development and does not contain barbecue recipes.",
        "gold_entities": [],
        "is_negative": True
    },

    # -------------------------------------------------------------
    # TIER 5: RGB NOISE ROBUSTNESS & DISTRACTOR RESISTANCE
    # -------------------------------------------------------------
    {
        "id": "T5-NOISE-01",
        "tier": "Tier 5: Noise Robustness",
        "query": "Explain how React passes data from parent to child components using props in The Road to React.",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "React passes data down the component hierarchy from parent to child components through read-only props.",
        "gold_entities": ["props", "parent", "child", "components"],
        "is_negative": False
    },
    {
        "id": "T5-NOISE-02",
        "tier": "Tier 5: Noise Robustness",
        "query": "In The Book of Five Rings, what does Musashi say about the Book of Fire regarding combat and battle?",
        "target_doc": "The Book of Five Rings.pdf",
        "gold_answer": "The Book of Fire concerns the heat of combat and large-scale battle, focusing on shifting advantage and seizing the initiative.",
        "gold_entities": ["fire", "combat", "battle", "advantage", "initiative"],
        "is_negative": False
    },
    {
        "id": "T5-NOISE-03",
        "tier": "Tier 5: Noise Robustness",
        "query": "What company did Steve Jobs acquire from George Lucas that eventually revolutionized animated cinema as Pixar?",
        "target_doc": "Steve Jobs (1).pdf",
        "gold_answer": "Steve Jobs acquired the computer graphics division of Lucasfilm from George Lucas, which became Pixar.",
        "gold_entities": ["lucasfilm", "pixar", "lucas", "graphics"],
        "is_negative": False
    },
    {
        "id": "T5-NOISE-04",
        "tier": "Tier 5: Noise Robustness",
        "query": "In The Odyssey, what trick did Penelope use with the burial shroud of Laertes to delay the suitors?",
        "target_doc": "The Odyssey.pdf",
        "gold_answer": "Penelope wove the burial shroud by day and unwove her progress by night, delaying the suitors for three years.",
        "gold_entities": ["shroud", "laertes", "wove", "unwove", "suitors"],
        "is_negative": False
    }
]

# Run Benchmark
results = []

headers = f"{'ID':<11} | {'Tier':<25} | {'Retrieval':<10} | {'TTFT':<10} | {'Total':<7} | {'Hit@1':<5} | {'F1':<5} | {'Cosine':<6} | {'Faith':<5} | {'Status'}"
print(headers)
print("-" * 115)

for case in benchmark_cases:
    t_start = time.perf_counter()

    # 1. Retrieval
    t0_ret = time.perf_counter()
    context = rag.retrieve_context(case["query"])
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

    gen_dur = time.perf_counter() - t0_gen
    total_dur = time.perf_counter() - t_start
    ttft_val = ttft if ttft is not None else 0.0
    ans_text = "".join(tokens).strip()
    tok_count = len(tokens)
    tps = tok_count / (gen_dur - ttft_val) if (gen_dur - ttft_val) > 0 else (tok_count / gen_dur if gen_dur > 0 else 0)

    # Calculate Information Retrieval (IR) Metrics
    retrieved_sources = context.get("sources", [])
    retrieved_chunks = context.get("chunks", [])
    
    # Hit Rate@1 and Hit Rate@2
    hit_at_1 = False
    hit_at_2 = False
    target = case.get("target_doc")
    
    if target:
        targets = [target] if isinstance(target, str) else target
        if retrieved_sources:
            if any(t in retrieved_sources[0] for t in targets):
                hit_at_1 = True
            if any(any(t in s for t in targets) for s in retrieved_sources[:2]):
                hit_at_2 = True
    elif case.get("is_negative"):
        # For negative test, hit is true if confidence is low or no spurious doc matched
        hit_at_1 = context.get("low_confidence", False) or len(retrieved_chunks) == 0 or context.get("is_chitchat", False)
        hit_at_2 = hit_at_1

    # Token F1 & Semantic Cosine
    f1 = compute_token_f1(ans_text, case["gold_answer"])
    semantic_cos = compute_semantic_cosine(rag, ans_text, case["gold_answer"])
    
    # Faithfulness / Groundedness
    faith = compute_faithfulness(ans_text, retrieved_chunks)

    # Negative rejection evaluation
    is_neg = case.get("is_negative", False)
    neg_rejected = False
    if is_neg:
        ans_lower = ans_text.lower()
        refusal_phrases = ["not", "no mention", "does not contain", "cannot be answered", "does not discuss", "no information", "unrelated", "neither", "don't"]
        neg_rejected = context.get("low_confidence", False) or any(rp in ans_lower for rp in refusal_phrases) or context.get("is_chitchat", False)

    # Determine status
    if is_neg:
        status = "✅ REJECT" if neg_rejected else "⚠️ HALLUC"
    else:
        status = "✅ PASS" if (hit_at_2 and semantic_cos >= 0.50) else ("🟡 PARTIAL" if semantic_cos >= 0.40 else "❌ FAIL")

    hit_str = "1.0" if hit_at_1 else ("0.5" if hit_at_2 else "0.0")

    print(
        f"{case['id']:<11} | {case['tier']:<25} | {retrieval_dur*1000:6.1f} ms | "
        f"{ttft_val*1000:6.1f} ms | {total_dur:5.2f}s | {hit_str:<5} | {f1:5.2f} | "
        f"{semantic_cos:6.2f} | {faith:5.2f} | {status}",
        flush=True
    )

    results.append({
        "id": case["id"],
        "tier": case["tier"],
        "query": case["query"],
        "target_doc": case["target_doc"],
        "gold_answer": case["gold_answer"],
        "generated_answer": ans_text,
        "is_negative": is_neg,
        "neg_rejected": neg_rejected,
        "retrieval_ms": retrieval_dur * 1000,
        "ttft_ms": ttft_val * 1000,
        "gen_dur_s": gen_dur,
        "total_dur_s": total_dur,
        "tps": tps,
        "hit_at_1": hit_at_1,
        "hit_at_2": hit_at_2,
        "token_f1": f1,
        "semantic_cosine": semantic_cos,
        "faithfulness": faith,
        "status": status,
        "retrieved_sources": retrieved_sources,
        "low_confidence": context.get("low_confidence", False)
    })

# =========================================================================
# RESEARCH PAPER METRIC AGGREGATION & REPORTING
# =========================================================================

pos_cases = [r for r in results if not r["is_negative"]]
neg_cases = [r for r in results if r["is_negative"]]

t1_cases = [r for r in results if "Tier 1" in r["tier"]]
t2_cases = [r for r in results if "Tier 2" in r["tier"]]
t3_cases = [r for r in results if "Tier 3" in r["tier"]]
t5_cases = [r for r in results if "Tier 5" in r["tier"]]

print("\n" + "=" * 115)
print("📊 FORMAL RESEARCH EVALUATION REPORT: RAGAS / RGB / IR METRICS")
print("=" * 115)

print("\n1. RETRIEVAL & CONTEXT PRECISION METRICS (Context Quality):")
print(f"   • Hit Rate@1 (Top-1 Relevance):       {np.mean([r['hit_at_1'] for r in pos_cases])*100:5.1f}%")
print(f"   • Hit Rate@2 (Top-2 Recall):          {np.mean([r['hit_at_2'] for r in pos_cases])*100:5.1f}%")
print(f"   • Mean Context Retrieval Latency:     {np.mean([r['retrieval_ms'] for r in results]):5.1f} ms  (Max: {np.max([r['retrieval_ms'] for r in results]):.1f} ms)")

print("\n2. GENERATION QUALITY & GROUNDING METRICS (Ragas Triad):")
print(f"   • Mean Semantic Cosine (BERTScore):   {np.mean([r['semantic_cosine'] for r in pos_cases]):5.3f} / 1.000")
print(f"   • Mean Token F1 Score (SQuAD):        {np.mean([r['token_f1'] for r in pos_cases]):5.3f} / 1.000")
print(f"   • Mean Faithfulness (Anti-Halluc.):   {np.mean([r['faithfulness'] for r in pos_cases]):5.3f} / 1.000")

print("\n3. RGB BENCHMARK STRESS TESTS (Chen et al. 2024):")
print(f"   • Negative Rejection Rate (Out-of-Domain):  {np.mean([r['neg_rejected'] for r in neg_cases])*100:5.1f}% ({sum(1 for r in neg_cases if r['neg_rejected'])} / {len(neg_cases)})")
print(f"   • Noise Robustness Accuracy (Tier 5):       {np.mean([r['hit_at_2'] for r in t5_cases])*100:5.1f}%")
print(f"   • Multi-Doc Synthesis Score (Tier 3):       {np.mean([r['semantic_cosine'] for r in t3_cases]):5.3f} / 1.000")

print("\n4. BLOOM'S TAXONOMY BREAKDOWN (Test Question Bank Standards):")
print(f"   • Level 1 Direct Factual Token F1:    {np.mean([r['token_f1'] for r in t1_cases]):5.3f} (Semantic Cosine: {np.mean([r['semantic_cosine'] for r in t1_cases]):.3f})")
print(f"   • Level 2 Conceptual Semantic Cosine: {np.mean([r['semantic_cosine'] for r in t2_cases]):5.3f}")
print(f"   • Level 3 Cross-Doc Synthesis Cosine: {np.mean([r['semantic_cosine'] for r in t3_cases]):5.3f}")

print("\n5. LATENCY & SYSTEM THROUGHPUT PROFILE:")
print(f"   • Mean Time to First Token (TTFT):    {np.mean([r['ttft_ms'] for r in results]):5.1f} ms")
print(f"   • Mean Total Turnaround Time:         {np.mean([r['total_dur_s'] for r in results]):5.2f} s")
print(f"   • Mean Generation Throughput:         {np.mean([r['tps'] for r in results]):5.1f} tok/s")
print(f"   • Total Turnaround <= 5.0s Rate:      {np.mean(np.array([r['total_dur_s'] for r in results]) <= 5.0)*100:5.1f}%")

# Save complete JSON artifact
output_benchmark_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_research_benchmark_results.json")
with open(output_benchmark_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methodology": "RAGAS + RGB Benchmark (Chen et al. 2024) + Bloom's Taxonomy",
        "results": results,
        "aggregate_metrics": {
            "hit_rate_at_1": float(np.mean([r['hit_at_1'] for r in pos_cases])),
            "hit_rate_at_2": float(np.mean([r['hit_at_2'] for r in pos_cases])),
            "mean_semantic_cosine": float(np.mean([r['semantic_cosine'] for r in pos_cases])),
            "mean_token_f1": float(np.mean([r['token_f1'] for r in pos_cases])),
            "mean_faithfulness": float(np.mean([r['faithfulness'] for r in pos_cases])),
            "negative_rejection_rate": float(np.mean([r['neg_rejected'] for r in neg_cases])),
            "mean_retrieval_ms": float(np.mean([r['retrieval_ms'] for r in results])),
            "mean_ttft_ms": float(np.mean([r['ttft_ms'] for r in results])),
            "mean_total_s": float(np.mean([r['total_dur_s'] for r in results])),
            "mean_throughput_tps": float(np.mean([r['tps'] for r in results]))
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved full academic benchmark results to: {output_benchmark_file}")
print("=" * 115)

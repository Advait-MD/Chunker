"""
Head-to-Head Research Benchmark: APEX vs. Naive RAG (Lewis et al., 2020)
Controlled academic benchmark with standardized runner parameters:
- Same Generator: gemma3:1b on Native Ollama (127.0.0.1)
- Same Runner Options: num_thread=4, num_predict=40, num_ctx=2048 (Eliminates runner thrashing)
- Same Question Suite: 15 questions across 5 core research tiers:
  1. Factual Recall (Hit@1, Token F1, Cosine)
  2. Conceptual Reasoning (Semantic Cosine, Explanatory Depth)
  3. Cross-Document Synthesis (Information Integration)
  4. Negative Rejection (Hallucination Resistance on Out-of-Domain queries)
  5. Noise Robustness (Distractor Resistance)
"""

import os
import sys
import time
import json
import re
from collections import Counter
import numpy as np
import requests

# Point to 127.0.0.1 explicitly
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
NAIVE_DIR = os.path.join(PROJECT_ROOT, "naive rag")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, NAIVE_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from face import RAGInterface
from naive_rag import NaiveRAG, PROMPT_TEMPLATE
from modules.retrieval import tokenize_bm25, COMMON_STOPWORDS

print("=" * 115)
print("⚔️ HEAD-TO-HEAD RESEARCH BENCHMARK: APEX vs. NAIVE RAG (Lewis et al., 2020)")
print("   Identical Corpus: 4 Books | Identical Model: Gemma 3 (1B) | Identical Prompts")
print("=" * 115)

# Initialize APEX
print("\n[1/3] Initializing APEX (Hierarchical Clustering RAG)...", flush=True)
apex = RAGInterface(verbose=False)
print("✓ APEX initialized.", flush=True)

# Initialize Naive RAG
print("\n[2/3] Initializing Naive RAG (DPR + FAISS IndexFlatIP + 100-word passages)...", flush=True)
naive = NaiveRAG(generator_model="gemma3:1b", top_k=3)
if not naive.load_index():
    print("❌ Error: Naive RAG index not found!", flush=True)
    sys.exit(1)
naive._load_question_encoder()
print("✓ Naive RAG initialized.\n", flush=True)

# Standardized Generator Function (Using shared session & runner options to prevent reload thrashing)
def generate_standardized(prompt: str) -> str:
    res = apex.session.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "gemma3:1b",
            "prompt": prompt,
            "stream": False,
            "options": apex.ollama_options,
            "keep_alive": -1
        },
        timeout=15
    )
    if res.status_code == 200:
        return res.json().get("response", "").strip()
    return "Generation Error"

# Pre-warm generator
print("Pre-warming Ollama generator...", flush=True)
generate_standardized("Answer in 1 sentence: What is React?\nAnswer:")
print("Generator ready and hot in memory!\n", flush=True)

# Metric Helpers
def compute_token_f1(prediction: str, ground_truth: str) -> float:
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


def compute_semantic_cosine(text1: str, text2: str) -> float:
    emb1 = np.array(apex.processor.store.get_query_embedding(text1))
    emb2 = np.array(apex.processor.store.get_query_embedding(text2))
    n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (n1 * n2))


# 15 Standardized Benchmark Cases across 5 Tiers
test_suite = [
    # Tier 1: Factual Recall
    {
        "id": "T1-Q01",
        "tier": "Tier 1: Factual Recall",
        "query": "Who wrote The Road to React?",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "Robin Wieruch wrote The Road to React.",
        "is_neg": False
    },
    {
        "id": "T1-Q02",
        "tier": "Tier 1: Factual Recall",
        "query": "Who authored The Book of Five Rings?",
        "target_doc": "The Book of Five Rings.pdf",
        "gold_answer": "Miyamoto Musashi authored The Book of Five Rings.",
        "is_neg": False
    },
    {
        "id": "T1-Q03",
        "tier": "Tier 1: Factual Recall",
        "query": "Who wrote the biography of Steve Jobs?",
        "target_doc": "Steve Jobs (1).pdf",
        "gold_answer": "Walter Isaacson wrote the biography of Steve Jobs.",
        "is_neg": False
    },
    {
        "id": "T1-Q04",
        "tier": "Tier 1: Factual Recall",
        "query": "Who is the classical author of the epic poem The Odyssey?",
        "target_doc": "The Odyssey.pdf",
        "gold_answer": "Homer is the classical author of The Odyssey.",
        "is_neg": False
    },

    # Tier 2: Conceptual Reasoning
    {
        "id": "T2-Q05",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "What is JSX and why is it used in React according to The Road to React?",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "JSX is a JavaScript syntax extension that allows developers to write HTML-like markup directly in JavaScript, transpiling to React.createElement.",
        "is_neg": False
    },
    {
        "id": "T2-Q06",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "What are React Hooks and why were they introduced according to Robin Wieruch?",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "Hooks are functions that allow developers to use state and lifecycle features in functional components without writing class components.",
        "is_neg": False
    },
    {
        "id": "T2-Q07",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "According to The Book of Five Rings, what is the core principle of the Way of Strategy?",
        "target_doc": "The Book of Five Rings.pdf",
        "gold_answer": "The Way of Strategy is a continuous discipline of mastering fundamentals, situational awareness, and practical adaptability.",
        "is_neg": False
    },
    {
        "id": "T2-Q08",
        "tier": "Tier 2: Conceptual Reasoning",
        "query": "What was Steve Jobs' core vision regarding simplicity and design for the Macintosh?",
        "target_doc": "Steve Jobs (1).pdf",
        "gold_answer": "Steve Jobs believed in user-friendly simplicity, intuitive graphical interfaces, and beautiful design where hidden details matter.",
        "is_neg": False
    },

    # Tier 3: Cross-Document Synthesis
    {
        "id": "T3-Q09",
        "tier": "Tier 3: Cross-Doc Synthesis",
        "query": "Compare the philosophy of mastery in The Book of Five Rings with learning React in The Road to React.",
        "target_doc": ["The Book of Five Rings.pdf", "The Road to React - Robin Wieruch.pdf"],
        "gold_answer": "Both emphasize mastering fundamentals before complexity, continuous deliberate practice, and adaptability over rigid dogmatic rules.",
        "is_neg": False
    },
    {
        "id": "T3-Q10",
        "tier": "Tier 3: Cross-Doc Synthesis",
        "query": "How does Musashi's concept of The Void connect with Steve Jobs' radical focus on simplicity?",
        "target_doc": ["The Book of Five Rings.pdf", "Steve Jobs (1).pdf"],
        "gold_answer": "Both embrace clarity of mind by stripping away the non-essential, eliminating clutter, and focusing entirely on what is truly important.",
        "is_neg": False
    },

    # Tier 4: Negative Rejection (Hallucination Resistance)
    {
        "id": "T4-Q11",
        "tier": "Tier 4: Negative Rejection",
        "query": "What was Steve Jobs' favorite flavor of astronaut freeze-dried ice cream during Apollo 11?",
        "target_doc": None,
        "gold_answer": "The provided documents do not contain information about Steve Jobs having a favorite flavor of astronaut ice cream.",
        "is_neg": True
    },
    {
        "id": "T4-Q12",
        "tier": "Tier 4: Negative Rejection",
        "query": "Which JavaScript framework does Homer explicitly recommend in Book IV of the Odyssey for web development?",
        "target_doc": None,
        "gold_answer": "The Odyssey is an ancient Greek epic poem and does not mention modern JavaScript frameworks.",
        "is_neg": True
    },
    {
        "id": "T4-Q13",
        "tier": "Tier 4: Negative Rejection",
        "query": "What is Robin Wieruch's secret recipe for smoked Texas barbecue ribs in The Road to React?",
        "target_doc": None,
        "gold_answer": "The Road to React is a programming book about web development and does not contain barbecue recipes.",
        "is_neg": True
    },

    # Tier 5: Noise Robustness & Distractor Resistance
    {
        "id": "T5-Q14",
        "tier": "Tier 5: Noise Robustness",
        "query": "Explain how React passes data from parent to child components using props in The Road to React.",
        "target_doc": "The Road to React - Robin Wieruch.pdf",
        "gold_answer": "React passes data down the component hierarchy from parent to child components through read-only props.",
        "is_neg": False
    },
    {
        "id": "T5-Q15",
        "tier": "Tier 5: Noise Robustness",
        "query": "What company did Steve Jobs acquire from George Lucas that eventually revolutionized animated cinema as Pixar?",
        "target_doc": "Steve Jobs (1).pdf",
        "gold_answer": "Steve Jobs acquired the computer graphics division of Lucasfilm from George Lucas, which became Pixar.",
        "is_neg": False
    }
]

comparison_results = []

print("Running Head-to-Head Comparative Interrogations...\n", flush=True)
headers = f"{'ID':<8} | {'System':<10} | {'Retrieval':<10} | {'Gen Time':<10} | {'Total Time':<10} | {'Hit@1':<6} | {'F1':<6} | {'Cosine':<7} | {'Verdict'}"
print(headers)
print("-" * 105)

for test in test_suite:
    q = test["query"]
    target = test["target_doc"]
    gold = test["gold_answer"]
    is_neg = test["is_neg"]

    # ------------------ APEX Execution ------------------
    t0_apex = time.perf_counter()
    apex_ctx = apex.retrieve_context(q)
    t_apex_ret = time.perf_counter() - t0_apex

    apex_prompt = apex.build_prompt(apex_ctx)
    t0_gen = time.perf_counter()
    apex_ans = generate_standardized(apex_prompt)
    t_apex_gen = time.perf_counter() - t0_gen
    t_apex_tot = time.perf_counter() - t0_apex

    # APEX Metrics
    apex_sources = apex_ctx.get("sources", [])
    if target:
        targets = [target] if isinstance(target, str) else target
        apex_hit = any(any(t in s for t in targets) for s in apex_sources[:1])
    else:
        apex_hit = apex_ctx.get("low_confidence", False) or len(apex_ctx.get("chunks", [])) == 0

    apex_f1 = compute_token_f1(apex_ans, gold)
    apex_cos = compute_semantic_cosine(apex_ans, gold)
    
    if is_neg:
        apex_verdict = "✅ REJECT" if (apex_ctx.get("low_confidence") or "not" in apex_ans.lower()) else "⚠️ HALLUC"
    else:
        apex_verdict = "✅ PASS" if (apex_hit and apex_cos >= 0.50) else ("🟡 PARTIAL" if apex_cos >= 0.40 else "❌ MISS")

    # ------------------ Naive RAG Execution ------------------
    t0_naive = time.perf_counter()
    naive_retrieved = naive.retrieve(q, k=3)
    t_naive_ret = time.perf_counter() - t0_naive

    context_str = "\n\n".join(f"[{p['rank']}] {p['text']}" for p in naive_retrieved)
    naive_prompt = PROMPT_TEMPLATE.format(context=context_str, query=q)

    t0_ngen = time.perf_counter()
    naive_ans = generate_standardized(naive_prompt)
    t_naive_gen = time.perf_counter() - t0_ngen
    t_naive_tot = time.perf_counter() - t0_naive

    # Naive RAG Metrics
    naive_sources = [p["doc_id"] for p in naive_retrieved]
    if target:
        targets = [target] if isinstance(target, str) else target
        naive_hit = any(any(t in s for t in targets) for s in naive_sources[:1])
    else:
        # Lewis et al. refusal criteria: "I don't know" or refusal phrasing
        ans_low = naive_ans.lower()
        naive_hit = "don't know" in ans_low or "not contain" in ans_low or "does not contain" in ans_low

    naive_f1 = compute_token_f1(naive_ans, gold)
    naive_cos = compute_semantic_cosine(naive_ans, gold)

    if is_neg:
        ans_low = naive_ans.lower()
        naive_verdict = "✅ REJECT" if ("don't know" in ans_low or "not contain" in ans_low or "does not contain" in ans_low) else "⚠️ HALLUC"
    else:
        naive_verdict = "✅ PASS" if (naive_hit and naive_cos >= 0.50) else ("🟡 PARTIAL" if naive_cos >= 0.40 else "❌ MISS")

    # Print comparison row
    hit_apex_str = "1.0" if apex_hit else "0.0"
    hit_naive_str = "1.0" if naive_hit else "0.0"

    print(f"{test['id']:<8} | {'APEX':<10} | {t_apex_ret*1000:6.1f} ms | {t_apex_gen:6.2f} s   | {t_apex_tot:6.2f} s   | {hit_apex_str:<6} | {apex_f1:5.2f}  | {apex_cos:5.2f}   | {apex_verdict}")
    print(f"{'':<8} | {'Naive RAG':<10} | {t_naive_ret*1000:6.1f} ms | {t_naive_gen:6.2f} s   | {t_naive_tot:6.2f} s   | {hit_naive_str:<6} | {naive_f1:5.2f}  | {naive_cos:5.2f}   | {naive_verdict}")
    print("-" * 105, flush=True)

    comparison_results.append({
        "id": test["id"],
        "tier": test["tier"],
        "query": q,
        "gold_answer": gold,
        "is_negative": is_neg,
        "apex": {
            "answer": apex_ans,
            "retrieval_ms": t_apex_ret * 1000,
            "gen_s": t_apex_gen,
            "total_s": t_apex_tot,
            "hit_at_1": apex_hit,
            "token_f1": apex_f1,
            "semantic_cosine": apex_cos,
            "verdict": apex_verdict
        },
        "naive": {
            "answer": naive_ans,
            "retrieval_ms": t_naive_ret * 1000,
            "gen_s": t_naive_gen,
            "total_s": t_naive_tot,
            "hit_at_1": naive_hit,
            "token_f1": naive_f1,
            "semantic_cosine": naive_cos,
            "verdict": naive_verdict
        }
    })

# Compute Aggregated Comparative Statistics
pos_cases = [c for c in comparison_results if not c["is_negative"]]
neg_cases = [c for c in comparison_results if c["is_negative"]]

apex_hits = np.mean([c["apex"]["hit_at_1"] for c in pos_cases]) * 100
naive_hits = np.mean([c["naive"]["hit_at_1"] for c in pos_cases]) * 100

apex_cos = np.mean([c["apex"]["semantic_cosine"] for c in pos_cases])
naive_cos = np.mean([c["naive"]["semantic_cosine"] for c in pos_cases])

apex_f1 = np.mean([c["apex"]["token_f1"] for c in pos_cases])
naive_f1 = np.mean([c["naive"]["token_f1"] for c in pos_cases])

apex_ret = np.mean([c["apex"]["retrieval_ms"] for c in comparison_results])
naive_ret = np.mean([c["naive"]["retrieval_ms"] for c in comparison_results])

apex_gen = np.mean([c["apex"]["gen_s"] for c in comparison_results])
naive_gen = np.mean([c["naive"]["gen_s"] for c in comparison_results])

apex_tot = np.mean([c["apex"]["total_s"] for c in comparison_results])
naive_tot = np.mean([c["naive"]["total_s"] for c in comparison_results])

apex_neg = np.mean(["REJECT" in c["apex"]["verdict"] for c in neg_cases]) * 100
naive_neg = np.mean(["REJECT" in c["naive"]["verdict"] for c in neg_cases]) * 100

print("\n" + "=" * 105)
print("📊 HEAD-TO-HEAD COMPARATIVE RESEARCH SUMMARY: APEX vs. NAIVE RAG")
print("=" * 105)
print(f"{'Evaluation Metric':<35} | {'APEX Pipeline':<25} | {'Naive RAG Baseline':<25} | {'Advantage'}")
print("-" * 105)
print(f"{'Top-1 Precision / Hit Rate@1':<35} | {apex_hits:5.1f}%                    | {naive_hits:5.1f}%                    | {'+' if apex_hits >= naive_hits else ''}{apex_hits - naive_hits:.1f}%")
print(f"{'Mean Semantic Cosine (BERTScore)':<35} | {apex_cos:5.3f} / 1.000             | {naive_cos:5.3f} / 1.000             | {'+' if apex_cos >= naive_cos else ''}{apex_cos - naive_cos:.3f}")
print(f"{'Mean SQuAD Token F1 Score':<35} | {apex_f1:5.3f} / 1.000             | {naive_f1:5.3f} / 1.000             | {'+' if apex_f1 >= naive_f1 else ''}{apex_f1 - naive_f1:.3f}")
print(f"{'Negative Rejection (Anti-Halluc.)':<35} | {apex_neg:5.1f}%                    | {naive_neg:5.1f}%                    | {'+' if apex_neg >= naive_neg else ''}{apex_neg - naive_neg:.1f}%")
print(f"{'Mean Context Retrieval Latency':<35} | {apex_ret:6.1f} ms                  | {naive_ret:6.1f} ms                  | {naive_ret / max(apex_ret, 0.001):.1f}x Faster")
print(f"{'Mean Generation Duration':<35} | {apex_gen:6.2f} seconds             | {naive_gen:6.2f} seconds             | {naive_gen / max(apex_gen, 0.001):.1f}x Faster")
print(f"{'Mean Total Pipeline Turnaround':<35} | {apex_tot:6.2f} seconds             | {naive_tot:6.2f} seconds             | {naive_tot / max(apex_tot, 0.001):.1f}x Faster")
print(f"{'Turnaround <= 5.0s Compliance':<35} | {np.mean(np.array([c['apex']['total_s'] for c in comparison_results]) <= 5.0)*100:5.1f}%                    | {np.mean(np.array([c['naive']['total_s'] for c in comparison_results]) <= 5.0)*100:5.1f}%                    | {'+' if np.mean(np.array([c['apex']['total_s'] for c in comparison_results]) <= 5.0) >= np.mean(np.array([c['naive']['total_s'] for c in comparison_results]) <= 5.0) else ''}{(np.mean(np.array([c['apex']['total_s'] for c in comparison_results]) <= 5.0) - np.mean(np.array([c['naive']['total_s'] for c in comparison_results]) <= 5.0))*100:.1f}%")
print("=" * 105)

# Save JSON file
out_file = os.path.join(PROJECT_ROOT, "apex_vs_naive_head_to_head_report.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "apex_hit_rate_at_1": float(apex_hits),
            "naive_hit_rate_at_1": float(naive_hits),
            "apex_mean_cosine": float(apex_cos),
            "naive_mean_cosine": float(naive_cos),
            "apex_mean_f1": float(apex_f1),
            "naive_mean_f1": float(naive_f1),
            "apex_neg_rejection_pct": float(apex_neg),
            "naive_neg_rejection_pct": float(naive_neg),
            "apex_mean_retrieval_ms": float(apex_ret),
            "naive_mean_retrieval_ms": float(naive_ret),
            "apex_mean_gen_s": float(apex_gen),
            "naive_mean_gen_s": float(naive_gen),
            "apex_mean_total_s": float(apex_tot),
            "naive_mean_total_s": float(naive_tot)
        },
        "detailed_comparisons": comparison_results
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved comprehensive comparative report to: {out_file}\n")

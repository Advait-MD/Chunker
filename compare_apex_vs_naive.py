"""
compare_apex_vs_naive.py — Rigorous Head-to-Head Evaluation Benchmark: APEX vs. Naive RAG (Lewis et al., 2020)

Evaluates both systems on identical corpus and questions derived from:
- test docs/Test Questions.md (Levels 1, 2, 3)
- Adversarial Hallucination Trap (Chronological impossibility)

Conditions strictly standardized (Lewis et al., Section 6 & 7):
- Identical documents: The Road to React, The Book of Five Rings, The Odyssey, Steve Jobs
- Identical generator: Ollama gemma3:1b
- Identical generation temperature: 0.0 (fully deterministic)
- Identical retrieval budget: top_k = 5
"""

import os
import sys
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Any

# Ensure project root and naive rag directory are importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
NAIVE_DIR = os.path.join(PROJECT_ROOT, "naive rag")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, NAIVE_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from face import RAGInterface
from modules.config import config
from naive_rag import NaiveRAG

# ==================== EVALUATION QUESTIONS BANK ====================

EVAL_SUITE = [
    # ---------- LEVEL 1: DIRECT FACTUAL RETRIEVAL ----------
    {
        "id": "L1-Q1",
        "level": "Level 1: Factual Recall",
        "target_capability": "Direct Token/Entity Lookup",
        "question": "Who wrote 'The Road to React'?",
        "expected_answer": "Robin Wieruch",
        "primary_source": "The Road to React - Robin Wieruch.pdf",
        "gold_keywords": ["Robin Wieruch", "Wieruch"],
        "eval_type": "factual"
    },
    {
        "id": "L1-Q2",
        "level": "Level 1: Factual Recall",
        "target_capability": "Specific Date/Number Extraction",
        "question": "What is the publication date of 'The Road to React'?",
        "expected_answer": "February 5, 2024 (2024-02-05)",
        "primary_source": "The Road to React - Robin Wieruch.pdf",
        "gold_keywords": ["2024", "February", "2024-02-05"],
        "eval_type": "factual"
    },
    {
        "id": "L1-Q3",
        "level": "Level 1: Factual Recall",
        "target_capability": "Author & Historical Entity Lookup",
        "question": "Who authored 'The Book of Five Rings'?",
        "expected_answer": "Miyamoto Musashi",
        "primary_source": "The Book of Five Rings.pdf",
        "gold_keywords": ["Miyamoto Musashi", "Musashi"],
        "eval_type": "factual"
    },

    # ---------- LEVEL 2: CONCEPTUAL & RELATIONAL REASONING ----------
    {
        "id": "L2-Q4",
        "level": "Level 2: Conceptual Understanding",
        "target_capability": "Semantic Synthesis across Chunks",
        "question": "What is JSX and why is it used in React?",
        "expected_answer": "JSX is a JavaScript syntax extension allowing HTML-like markup in components; it improves readability and transpiles to React.createElement() calls at build time.",
        "primary_source": "The Road to React - Robin Wieruch.pdf",
        "gold_keywords": ["syntax", "JavaScript", "HTML", "createElement", "component"],
        "eval_type": "conceptual"
    },
    {
        "id": "L2-Q5",
        "level": "Level 2: Conceptual Understanding",
        "target_capability": "Abstract Doctrine / Relational Extraction",
        "question": "According to 'The Book of Five Rings', what is the 'Way of Strategy'?",
        "expected_answer": "A path of continuous self-improvement, mastery of fundamentals, situational awareness, and practical adaptability applicable to all crafts.",
        "primary_source": "The Book of Five Rings.pdf",
        "gold_keywords": ["strategy", "way", "fundamentals", "practice", "spirit", "warrior"],
        "eval_type": "conceptual"
    },
    {
        "id": "L2-Q6",
        "level": "Level 2: Conceptual Understanding",
        "target_capability": "Feature Motivation & Architecture Reason",
        "question": "What are React Hooks and why were they introduced?",
        "expected_answer": "Hooks are functions providing state and lifecycle capabilities in functional components, eliminating class component complexity and enabling stateful logic reuse.",
        "primary_source": "The Road to React - Robin Wieruch.pdf",
        "gold_keywords": ["hooks", "state", "functional", "components", "classes", "reuse"],
        "eval_type": "conceptual"
    },

    # ---------- LEVEL 3: ANALYTICAL & CROSS-DOCUMENT SYNTHESIS ----------
    {
        "id": "L3-Q7",
        "level": "Level 3: Cross-Document Synthesis",
        "target_capability": "Multi-Document Analogy & Principle Mapping",
        "question": "Compare the philosophical approach to mastery in 'The Book of Five Rings' with the learning approach to mastering React. What similarities do they share?",
        "expected_answer": "Both prioritize fundamentals before advanced techniques, learning through active practice over passive theory, principles over rote memorization, and continuous iterative refinement.",
        "primary_source": "Both (The Book of Five Rings + The Road to React)",
        "gold_keywords": ["fundamentals", "practice", "principles", "doing", "mastery", "Musashi", "React"],
        "eval_type": "cross_document"
    },
    {
        "id": "L3-Q8",
        "level": "Level 3: Cross-Document Synthesis",
        "target_capability": "High-Order Domain Projection",
        "question": "How does the concept of 'emptiness' (void) in Musashi's philosophy relate to writing clean, maintainable React code?",
        "expected_answer": "The void represents clarity without obstruction; in code, this translates to single-responsibility components, elimination of unnecessary abstractions, clean separation of concerns, and pruning dead code.",
        "primary_source": "Both (The Book of Five Rings + The Road to React)",
        "gold_keywords": ["void", "emptiness", "clarity", "unnecessary", "clean", "components", "code"],
        "eval_type": "cross_document"
    },
    {
        "id": "L3-Q9",
        "level": "Level 3: Cross-Document Synthesis",
        "target_capability": "Cross-Domain Advice Synthesis",
        "question": "Based on both books, what advice would Musashi give to a React developer about mastering their craft?",
        "expected_answer": "Know the fundamentals (master JS mechanics first), practice daily, study multiple paths, do nothing unnecessary, and treat training/shipping as the way.",
        "primary_source": "Both (The Book of Five Rings + The Road to React)",
        "gold_keywords": ["fundamentals", "practice", "unnecessary", "study", "paths", "training", "code"],
        "eval_type": "cross_document"
    },

    # ---------- LEVEL 4: ADVERSARIAL TRAP & HALLUCINATION RESISTANCE ----------
    {
        "id": "L4-Q10",
        "level": "Level 4: Adversarial Stress Test",
        "target_capability": "Hallucination Resistance on Chronological Trap",
        "question": "According to the provided documents, what specific React hooks did Steve Jobs use when developing the Macintosh in 1984?",
        "expected_answer": "Refusal / Not mentioned (React hooks did not exist in 1984; Steve Jobs did not use React hooks).",
        "primary_source": "Steve Jobs (1).pdf / Adversarial",
        "refusal_keywords": ["don't know", "don't have enough information", "not mentioned", "does not mention", "did not use", "not exist", "no mention", "cannot", "impossible"],
        "eval_type": "adversarial_refusal"
    }
]


def compute_token_f1(gold: str, pred: str) -> float:
    """Token-level F1 metric."""
    def tokenize(s):
        return re.findall(r"\w+", s.lower())
    gold_toks = tokenize(gold)
    pred_toks = tokenize(pred)
    if not gold_toks or not pred_toks:
        return 0.0
    common = set(gold_toks) & set(pred_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec = len(common) / len(gold_toks)
    return (2 * prec * rec) / (prec + rec)


def evaluate_system_response(test_case: Dict, answer: str, retrieved_sources: List[str]) -> Dict[str, Any]:
    """Score individual answer based on ground-truth criteria."""
    ans_lower = answer.lower().replace("’", "'").replace("‘", "'")
    eval_type = test_case["eval_type"]
    gold_keywords = test_case.get("gold_keywords", [])
    
    hits = [kw for kw in gold_keywords if kw.lower() in ans_lower]
    keyword_coverage = len(hits) / max(len(gold_keywords), 1)
    
    f1 = compute_token_f1(test_case["expected_answer"], answer)
    
    score = 0.0
    notes = []
    
    if eval_type == "factual":
        if hits:
            score = 70.0 + (30.0 * (len(hits) / len(gold_keywords)))
            notes.append(f"Hit key factual terms: {hits}")
        else:
            score = 15.0 if len(answer.strip()) > 10 else 0.0
            notes.append("Missed target entity/fact.")
            
    elif eval_type == "conceptual":
        score = min(100.0, 30.0 + (70.0 * keyword_coverage))
        notes.append(f"Conceptual coverage: {len(hits)}/{len(gold_keywords)} concepts ({', '.join(hits)})")
        
    elif eval_type == "cross_document":
        distinct_sources = len(set(retrieved_sources))
        cross_source_bonus = 25.0 if distinct_sources >= 2 else 0.0
        score = min(100.0, (keyword_coverage * 65.0) + cross_source_bonus + 10.0)
        notes.append(f"Synthesis coverage: {len(hits)}/{len(gold_keywords)} terms, distinct sources retrieved: {distinct_sources}")
        
    elif eval_type == "adversarial_refusal":
        refusal_terms = test_case.get("refusal_keywords", [])
        refused = any(ref.lower() in ans_lower for ref in refusal_terms)
        hallucinated = any(h in ans_lower for h in ["usestate", "useeffect", "usecontext", "usereducer", "custom hook"])
        
        if refused and not hallucinated:
            score = 100.0
            notes.append("EXCELLENT: Refused adversarial trap cleanly without hallucinating.")
        elif refused and hallucinated:
            score = 50.0
            notes.append("Partial refusal: Stated hesitation but named modern React hooks.")
        elif hallucinated:
            score = 0.0
            notes.append("CRITICAL FAILURE: Hallucinated that Steve Jobs used React hooks in 1984.")
        else:
            score = 40.0
            notes.append("Unclear response.")
            
    return {
        "score": round(score, 1),
        "token_f1": round(f1, 4),
        "keyword_hits": hits,
        "keyword_coverage": round(keyword_coverage, 2),
        "notes": "; ".join(notes)
    }


def run_benchmark():
    report_path = os.path.join(PROJECT_ROOT, "apex_vs_naive_rag_benchmark.json")

    print("=" * 80, flush=True)
    print("APEX vs. NAIVE RAG (Lewis et al., 2020) — HEAD-TO-HEAD COMPARATIVE BENCHMARK", flush=True)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize APEX Interface with temperature=0.0
    print("\n[1/3] Initializing APEX (Hierarchical Clustering RAG)...", flush=True)
    config.retrieval.include_conversation_memory = False
    apex = RAGInterface(
        ollama_model="gemma3:1b",
        temperature=0.0,  # Zero temperature for deterministic evaluation
        max_context_chunks=5,
        verbose=False
    )
    apex.record_conversation_memory = False
    print("✓ APEX initialized.", flush=True)

    # 2. Initialize Naive RAG Interface with temperature=0.0
    print("\n[2/3] Initializing Naive RAG (DPR + FAISS IndexFlatIP + 100-word passages)...", flush=True)
    naive = NaiveRAG(generator_model="gemma3:1b", top_k=5)
    if not naive.load_index():
        print("❌ Error: Naive RAG index not found! Run 'python \"naive rag/naive_rag.py\" --index' first.", flush=True)
        sys.exit(1)
    naive._load_question_encoder()
    print("✓ Naive RAG initialized.", flush=True)

    # Load existing results if resuming
    results = []
    completed_ids = set()
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                results = prev_data.get("detailed_results", [])
                completed_ids = {r["test_case"]["id"] for r in results}
                print(f"Found existing benchmark file with {len(results)} completed question(s). Resuming...", flush=True)
        except Exception:
            results = []
            completed_ids = set()

    # 3. Run Benchmark Suite
    print(f"\n[3/3] Running {len(EVAL_SUITE)} Evaluation Questions across 4 Difficulty Levels...\n", flush=True)
    
    for i, test in enumerate(EVAL_SUITE, 1):
        q_id = test["id"]
        q_text = test["question"]

        if q_id in completed_ids:
            print(f"[{i}/{len(EVAL_SUITE)}] Skipping {q_id} (already evaluated).", flush=True)
            continue

        print(f"--------------------------------------------------------------------------------", flush=True)
        print(f"[{i}/{len(EVAL_SUITE)}] {q_id} ({test['level']}):", flush=True)
        print(f"Question: {q_text}", flush=True)
        
        # --- APEX Execution ---
        t_apex_start = time.perf_counter()
        apex_ctx = apex.retrieve_context(q_text)
        t_apex_ret = apex_ctx.get("retrieval_time", 0.0)
        
        apex_prompt = apex.build_prompt(apex_ctx)
        t_gen_start = time.perf_counter()
        apex_ans = ""
        for attempt in range(3):
            apex_gen_res = apex.generate_response(apex_prompt)
            if apex_gen_res.get("success"):
                apex_ans = apex_gen_res.get("answer", "").strip()
                break
            time.sleep(2.0)
        if not apex_ans:
            apex_ans = apex_gen_res.get("error", "Error")
            
        t_apex_gen = time.perf_counter() - t_gen_start
        t_apex_total = time.perf_counter() - t_apex_start
        
        apex_sources = [m.get("source_file", "unknown") for m in apex_ctx.get("metadatas", [])]
        apex_eval = evaluate_system_response(test, apex_ans, apex_sources)
        
        # --- Naive RAG Execution ---
        t_naive_start = time.perf_counter()
        naive_res = naive.rag_answer(q_text, k=5)
        t_naive_total = time.perf_counter() - t_naive_start
        t_naive_ret = naive_res["retrieval_time"]
        t_naive_gen = naive_res["generation_time"]
        
        naive_ans = naive_res["answer"]
        if "Error communicating with Ollama" in naive_ans:
            for _ in range(3):
                time.sleep(2.0)
                naive_ans = naive.generate(q_text, naive_res["retrieved"])
                if "Error communicating with Ollama" not in naive_ans:
                    break
        naive_sources = [p["doc_id"] for p in naive_res["retrieved"]]
        
        naive_eval = evaluate_system_response(test, naive_ans, naive_sources)
        
        # Print summary for this question
        print(f"\n  ➤ APEX Score: {apex_eval['score']}/100 | Latency: {t_apex_total:.2f}s (ret: {t_apex_ret:.3f}s)", flush=True)
        print(f"    Answer: {apex_ans[:140]}...", flush=True)
        print(f"  ➤ Naive RAG Score: {naive_eval['score']}/100 | Latency: {t_naive_total:.2f}s (ret: {t_naive_ret:.3f}s)", flush=True)
        print(f"    Answer: {naive_ans[:140]}...", flush=True)
        
        results.append({
            "test_case": test,
            "apex": {
                "answer": apex_ans,
                "sources": apex_sources,
                "score": apex_eval["score"],
                "token_f1": apex_eval["token_f1"],
                "keyword_coverage": apex_eval["keyword_coverage"],
                "notes": apex_eval["notes"],
                "retrieval_time": t_apex_ret,
                "generation_time": t_apex_gen,
                "total_time": t_apex_total,
                "chunks_retrieved": len(apex_ctx.get("chunks", [])),
                "summaries_retrieved": len(apex_ctx.get("cluster_summaries", []))
            },
            "naive": {
                "answer": naive_ans,
                "sources": naive_sources,
                "score": naive_eval["score"],
                "token_f1": naive_eval["token_f1"],
                "keyword_coverage": naive_eval["keyword_coverage"],
                "notes": naive_eval["notes"],
                "retrieval_time": t_naive_ret,
                "generation_time": t_naive_gen,
                "total_time": t_naive_total,
                "passages_retrieved": len(naive_res["retrieved"])
            }
        })

        # Save progress immediately after each question
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "detailed_results": results
            }, f, ensure_ascii=False, indent=2)
        
    # ==================== AGGREGATE METRICS ====================
    apex_avg_score = sum(r["apex"]["score"] for r in results) / len(results)
    naive_avg_score = sum(r["naive"]["score"] for r in results) / len(results)
    
    apex_avg_f1 = sum(r["apex"]["token_f1"] for r in results) / len(results)
    naive_avg_f1 = sum(r["naive"]["token_f1"] for r in results) / len(results)
    
    apex_avg_ret_time = sum(r["apex"]["retrieval_time"] for r in results) / len(results)
    naive_avg_ret_time = sum(r["naive"]["retrieval_time"] for r in results) / len(results)
    
    apex_avg_total_time = sum(r["apex"]["total_time"] for r in results) / len(results)
    naive_avg_total_time = sum(r["naive"]["total_time"] for r in results) / len(results)

    print("\n" + "=" * 80, flush=True)
    print("FINAL BENCHMARK SCOREBOARD", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Metric':<35} | {'APEX':<20} | {'Naive RAG':<20} | {'Delta':<12}", flush=True)
    print("-" * 95, flush=True)
    print(f"{'Overall Score (0-100)':<35} | {apex_avg_score:<20.1f} | {naive_avg_score:<20.1f} | {apex_avg_score - naive_avg_score:+5.1f}", flush=True)
    print(f"{'Mean Token F1':<35} | {apex_avg_f1:<20.4f} | {naive_avg_f1:<20.4f} | {apex_avg_f1 - naive_avg_f1:+5.4f}", flush=True)
    print(f"{'Avg Retrieval Latency':<35} | {f'{apex_avg_ret_time*1000:.1f} ms':<20} | {f'{naive_avg_ret_time*1000:.1f} ms':<20} | {f'{(apex_avg_ret_time - naive_avg_ret_time)*1000:+5.1f} ms'}", flush=True)
    print(f"{'Avg Total Latency':<35} | {f'{apex_avg_total_time:.2f} s':<20} | {f'{naive_avg_total_time:.2f} s':<20} | {f'{apex_avg_total_time - naive_avg_total_time:+5.2f} s'}", flush=True)

    # Final complete JSON report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "apex_avg_score": round(apex_avg_score, 1),
                "naive_avg_score": round(naive_avg_score, 1),
                "apex_avg_f1": round(apex_avg_f1, 4),
                "naive_avg_f1": round(naive_avg_f1, 4),
                "apex_avg_ret_latency_ms": round(apex_avg_ret_time * 1000, 1),
                "naive_avg_ret_latency_ms": round(naive_avg_ret_time * 1000, 1),
            },
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed benchmark report exported to: {report_path}", flush=True)

    return results


if __name__ == "__main__":
    run_benchmark()

"""
APEX RAG - Multi-Tier Model Stress Benchmark (Easy, Medium, Hard, Extreme)
Evaluates the native RAG model (Gemma 3 1B + nomic-embed-text + ChromaDB) across:
  - Easy: Factual Lookups & Direct Entity Extraction
  - Medium: Conceptual Understanding & Relational Reasoning
  - Hard: Cross-Document Synthesis & Philosophical Parallels
  - Extreme: Adversarial Traps, Hallucination Resistance & Abstract Mapping
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from face import RAGInterface
from modules.logger import setup_logger, get_logger

logger = get_logger("benchmark")


BENCHMARK_SUITE = [
    # ========================== EASY TIER ==========================
    {
        "id": "E1",
        "difficulty": "Easy",
        "title": "React Author & Book Identification",
        "question": "Who wrote 'The Road to React'?",
        "source_filter": "The Road to React",
        "expected_keywords": ["Robin Wieruch", "Wieruch"],
        "eval_type": "factual",
        "description": "Tests direct factual retrieval of author name from metadata and text chunks."
    },
    {
        "id": "E2",
        "difficulty": "Easy",
        "title": "Five Rings Author & Style",
        "question": "Who authored 'The Book of Five Rings' and what martial style or school did he found?",
        "source_filter": "The Book of Five Rings",
        "expected_keywords": ["Musashi", "Miyamoto", "Two Heavens", "Niten", "style"],
        "eval_type": "factual",
        "description": "Tests direct entity and school lookup in historical text."
    },

    # ========================== MEDIUM TIER ==========================
    {
        "id": "M1",
        "difficulty": "Medium",
        "title": "React JSX & Build-Time Transpilation",
        "question": "What is JSX in React, and what does it transpile to at build time?",
        "source_filter": "The Road to React",
        "expected_keywords": ["syntax", "JavaScript", "createElement", "React.createElement", "HTML"],
        "eval_type": "conceptual",
        "description": "Tests conceptual explanation and specific build-time API recall."
    },
    {
        "id": "M2",
        "difficulty": "Medium",
        "title": "Book of Water Metaphor & Combat Stance",
        "question": "According to The Book of Five Rings, what is the core philosophy and metaphor of the Book of Water regarding combat and strategy?",
        "source_filter": "The Book of Five Rings",
        "expected_keywords": ["water", "spirit", "receptacle", "shape", "vessel", "fluid", "swordsmanship", "strategy", "two-heavens"],
        "eval_type": "conceptual",
        "description": "Tests multi-chunk synthesis of a philosophical and strategic metaphor."
    },

    # ========================== HARD TIER ==========================
    {
        "id": "H1",
        "difficulty": "Hard",
        "title": "Cross-Document Mastery: Musashi vs. Learning React",
        "question": "Compare the philosophy of mastery in 'The Book of Five Rings' with the recommended approach to learning React in 'The Road to React'. What fundamental principles do they share?",
        "source_filter": None, # Requires searching across documents
        "expected_keywords": ["fundamentals", "practice", "principles", "doing", "components", "strategy", "Musashi"],
        "eval_type": "cross_document",
        "description": "Tests cross-document retrieval and synthesis of educational and martial philosophies."
    },
    {
        "id": "H2",
        "difficulty": "Hard",
        "title": "Design Minimalism: Steve Jobs Focus vs. Musashi Simplicity",
        "question": "Compare Steve Jobs's philosophy of product design and focus with Musashi's principle of 'do nothing which is of no use' in The Book of Five Rings.",
        "source_filter": None, # Requires multi-source synthesis
        "expected_keywords": ["simplicity", "focus", "nothing", "use", "unnecessary", "Jobs", "Musashi", "design"],
        "eval_type": "cross_document",
        "description": "Tests multi-source analytical synthesis contrasting biography with martial doctrine."
    },

    # ========================== EXTREME TIER ==========================
    {
        "id": "X1",
        "difficulty": "Extreme",
        "title": "Adversarial Hallucination Trap (Chronological Impossible Hook)",
        "question": "According to the provided documents, what specific React hooks did Steve Jobs use when developing the Macintosh in 1984?",
        "source_filter": None,
        "refusal_keywords": [
            "don't have enough information",
            "not mentioned",
            "does not mention",
            "did not use",
            "no mention",
            "impossible",
            "not exist",
            "unrelated",
            "cannot find"
        ],
        "eval_type": "adversarial_refusal",
        "description": "Tests hallucination resistance against a counterfactual trap combining 1984 Mac history with modern React."
    },
    {
        "id": "X2",
        "difficulty": "Extreme",
        "title": "Abstract Domain Mapping: The Void to Software Architecture",
        "question": "In 'The Book of Five Rings', how does the concept of 'The Book of the Void' (emptiness) translate to the principles of writing clean, maintainable software and component architecture?",
        "source_filter": None,
        "expected_keywords": ["void", "emptiness", "clean", "complexity", "unnecessary", "clarity", "component", "code"],
        "eval_type": "abstract_mapping",
        "description": "Tests high-order cognitive projection mapping ancient philosophical emptiness to modern clean code."
    }
]


def evaluate_response(test_case: Dict, result: Dict) -> Dict[str, Any]:
    """Score the response based on test case evaluation criteria."""
    answer = result.get('answer', '')
    answer_lower = answer.lower().replace("’", "'").replace("‘", "'")
    eval_type = test_case['eval_type']
    retrieved_chunks = result.get('context_chunks', 0)
    sources = result.get('sources', [])
    scores = result.get('relevance_scores', [])
    
    avg_relevance = sum(scores) / len(scores) if scores else 0.0
    
    score = 0.0
    notes = []
    
    if eval_type == "factual":
        # Check keyword hits
        hits = [kw for kw in test_case['expected_keywords'] if kw.lower() in answer_lower]
        keyword_ratio = len(hits) / len(test_case['expected_keywords'])
        
        # In factual lookups, at least one key entity is required
        if hits:
            score = 70.0 + (30.0 * min(1.0, len(hits) / max(1, len(test_case['expected_keywords']) // 2)))
            score = min(100.0, score)
            notes.append(f"Matched key terms: {hits}")
        else:
            score = 20.0 if retrieved_chunks > 0 else 0.0
            notes.append("Failed to identify exact target entity.")
            
    elif eval_type == "conceptual":
        hits = [kw for kw in test_case['expected_keywords'] if kw.lower() in answer_lower]
        hit_ratio = len(hits) / len(test_case['expected_keywords'])
        score = min(100.0, 40.0 + (60.0 * hit_ratio))
        notes.append(f"Conceptual coverage: {len(hits)}/{len(test_case['expected_keywords'])} concepts ({', '.join(hits[:4])})")
        
    elif eval_type == "cross_document":
        hits = [kw for kw in test_case['expected_keywords'] if kw.lower() in answer_lower]
        hit_ratio = len(hits) / len(test_case['expected_keywords'])
        # Cross document must bridge concepts
        multi_source_bonus = 20.0 if len(sources) >= 2 else 0.0
        score = min(100.0, (hit_ratio * 70.0) + multi_source_bonus + 10.0)
        notes.append(f"Cross-doc synthesis hits: {len(hits)} terms; sources consulted: {len(sources)}")
        
    elif eval_type == "adversarial_refusal":
        # The model MUST reject or deny the premise, NOT hallucinate a hook!
        refused = any(ref.lower() in answer_lower for ref in test_case['refusal_keywords'])
        hallucinated_hook = any(h in answer_lower for h in ["usestate", "useeffect", "usecontext", "usereducer"])
        
        if refused and not hallucinated_hook:
            score = 100.0
            notes.append("EXCELLENT: Refused the adversarial premise cleanly without hallucinating.")
        elif refused and hallucinated_hook:
            score = 60.0
            notes.append("Partial refusal: Noted lack of connection but mentioned React hooks.")
        elif "steve jobs" in answer_lower and "1984" in answer_lower and not hallucinated_hook:
            score = 80.0
            notes.append("Good: Grounded in 1984 context without inventing hooks.")
        else:
            score = 15.0
            notes.append("FAILED: Hallucinated or accepted false premise.")
            
    elif eval_type == "abstract_mapping":
        hits = [kw for kw in test_case['expected_keywords'] if kw.lower() in answer_lower]
        hit_ratio = len(hits) / len(test_case['expected_keywords'])
        score = min(100.0, 30.0 + (70.0 * hit_ratio))
        notes.append(f"Abstract mapping coverage: {len(hits)} concepts")

    # Format verdict
    if score >= 75.0:
        verdict = "PASS"
    elif score >= 50.0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "avg_relevance": round(avg_relevance, 3),
        "notes": "; ".join(notes)
    }


def run_benchmark(output_file: str = "benchmark_results.json"):
    print("=" * 80)
    print("🚀 STARTING APEX RAG MULTI-TIER MODEL STRESS TEST")
    print("=" * 80)
    print("Difficulties: Easy (2) | Medium (2) | Hard (2) | Extreme (2)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    setup_logger(level="WARNING")
    interface = RAGInterface(verbose=True)

    results = []
    tier_stats = {
        "Easy": {"scores": [], "passes": 0, "total": 0},
        "Medium": {"scores": [], "passes": 0, "total": 0},
        "Hard": {"scores": [], "passes": 0, "total": 0},
        "Extreme": {"scores": [], "passes": 0, "total": 0},
    }

    total_start = time.time()

    for idx, tc in enumerate(BENCHMARK_SUITE):
        print("\n" + "#" * 80)
        print(f"[{idx+1}/{len(BENCHMARK_SUITE)}] [{tc['difficulty'].upper()}] {tc['id']}: {tc['title']}")
        print(f"❓ Question: {tc['question']}")
        if tc.get('source_filter'):
            print(f"🎯 Target Source Filter: '{tc['source_filter']}'")
        print("#" * 80)

        t_start = time.time()
        res = interface.query(tc['question'], source_filter=tc.get('source_filter'))
        elapsed = time.time() - t_start

        eval_res = evaluate_response(tc, res)

        tier = tc['difficulty']
        tier_stats[tier]['scores'].append(eval_res['score'])
        tier_stats[tier]['total'] += 1
        if eval_res['verdict'] == "PASS":
            tier_stats[tier]['passes'] += 1

        record = {
            "id": tc['id'],
            "difficulty": tc['difficulty'],
            "title": tc['title'],
            "question": tc['question'],
            "source_filter": tc.get('source_filter'),
            "sources_consulted": res.get('sources', []),
            "context_chunks_count": res.get('context_chunks', 0),
            "relevance_scores": [round(s, 3) for s in res.get('relevance_scores', [])],
            "response_time_sec": round(elapsed, 2),
            "answer": res.get('answer', ''),
            "score": eval_res['score'],
            "verdict": eval_res['verdict'],
            "evaluation_notes": eval_res['notes']
        }
        results.append(record)

        print(f"\n📊 Evaluation Verdict: [{eval_res['verdict']}] - Score: {eval_res['score']}/100")
        print(f"📝 Evaluation Notes: {eval_res['notes']}")
        print(f"⏱️ Time: {elapsed:.2f}s | Chunks: {res.get('context_chunks', 0)}")
        time.sleep(1.0)

    total_duration = time.time() - total_start

    # Summary calculations
    all_scores = [r['score'] for r in results]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total_passes = sum(1 for r in results if r['verdict'] == "PASS")

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "total_duration_seconds": round(total_duration, 2),
        "total_tests": len(results),
        "overall_average_score": round(overall_avg, 1),
        "overall_pass_rate": f"{round((total_passes / len(results)) * 100, 1)}%",
        "tier_breakdown": {
            tier: {
                "average_score": round(sum(d['scores']) / len(d['scores']), 1) if d['scores'] else 0.0,
                "pass_rate": f"{d['passes']}/{d['total']} ({round((d['passes']/d['total'])*100, 1)}%)"
            }
            for tier, d in tier_stats.items()
        },
        "test_results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("🏁 BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total Duration: {total_duration:.1f}s")
    print(f"Overall Average Score: {overall_avg:.1f} / 100")
    print(f"Overall Pass Rate: {total_passes}/{len(results)} ({round((total_passes/len(results))*100, 1)}%)")
    print("\n📈 Difficulty Breakdown:")
    for tier, stats in summary_data['tier_breakdown'].items():
        print(f"  • {tier:8s}: Avg Score = {stats['average_score']:5.1f} | Pass Rate = {stats['pass_rate']}")
    print(f"\n💾 Full JSON report exported to: {output_file}")
    print("=" * 80)

    return summary_data


if __name__ == "__main__":
    run_benchmark("benchmark_results.json")

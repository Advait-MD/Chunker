"""
run_raptor_paper_3way_benchmark.py
Official RAPTOR Paper (ICLR 2024) 3-Way Comparative Benchmark:
Compares:
  1. Naive RAG (Flat DPR/Dense retrieval over Layer 0 leaf passages)
  2. RAPTOR (Sarthi et al., ICLR 2024: Collapsed Tree pure dense retrieval)
  3. APEX (Our Architecture: Collapsed Tree Hybrid Fusion + Dynamic Coverage Gating + Deduplication + Budget Packing)

Evaluated across all datasets and metrics from Sarthi et al. (ICLR 2024):
  - QuALITY (Multiple-choice QA, Full + HARD subset accuracy)
  - NarrativeQA (ROUGE-L, BLEU-1, BLEU-4, METEOR, Token F1)
  - QASPER (Scientific paper Token F1 match)
  - Cinderella Qualitative Narrative Study (Semantic Cosine, Coverage, Latency)
  - Edge SLA Profiling: Retrieval ms, TTFT, Turnaround, Generation Throughput
"""

import os
import sys
import re
import json
import time
import requests
import numpy as np
from typing import Dict, List, Any, Optional

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from raptor import (
    RAPTOR,
    Tree,
    Node,
    BM25Retriever,
    OllamaEmbeddingModel,
    cosine_similarity,
    compute_max_token_f1,
    compute_bleu_scores,
    compute_rouge_l,
    compute_meteor,
    evaluate_narrativeqa_metrics,
    evaluate_multiple_choice_answer
)

DATA_DIR = os.path.join(PROJECT_ROOT, "benchmark_data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
OUTPUT_PATH = os.path.join(DATA_DIR, "raptor_paper_3way_benchmark_results.json")

OLLAMA_URL = "http://127.0.0.1:11434"
GEN_MODEL = "gemma3:1b"
EMB_MODEL = "nomic-embed-text"

session = requests.Session()


def get_query_embedding(query: str, emb_model: OllamaEmbeddingModel) -> np.ndarray:
    return emb_model.create_embedding(query)


def generate_with_profile(prompt: str, system_prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
    """Execute streaming generation via local Ollama and capture TTFT, total time, throughput."""
    endpoint = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": GEN_MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "top_k": 30,
            "top_p": 0.9,
            "num_ctx": 1024,
            "num_thread": 6
        }
    }
    
    t0 = time.perf_counter()
    resp = session.post(endpoint, json=payload, stream=True, timeout=120)
    ttft = None
    output_tokens = []
    eval_count = None
    
    for line in resp.iter_lines():
        if line:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunk = json.loads(line)
            part = chunk.get("response", "")
            output_tokens.append(part)
            if chunk.get("done", False):
                eval_count = chunk.get("eval_count")
                
    total_sec = time.perf_counter() - t0
    if ttft is None:
        ttft = total_sec
    full_text = "".join(output_tokens).strip()
    tok_count = eval_count if eval_count is not None else max(len(full_text.split()), 1)
    gen_time = max(total_sec - ttft, 0.05)
    throughput = round(tok_count / gen_time, 1)
    
    return {
        "text": full_text,
        "ttft_sec": round(ttft, 3),
        "total_sec": round(total_sec, 3),
        "tokens": tok_count,
        "throughput": throughput
    }


class BenchmarkRetriever:
    """Unified Retriever comparing Naive RAG, RAPTOR, and APEX on the same document tree."""
    def __init__(self, tree: Tree, emb_model: OllamaEmbeddingModel):
        self.tree = tree
        self.emb_model = emb_model
        
        # Precompute leaf nodes and all nodes
        self.leaf_nodes = tree.get_layer(0)
        self.all_nodes = tree.get_all_nodes()
        
        # Build BM25 index over all nodes for APEX hybrid retrieval
        self.all_texts = [n.text for n in self.all_nodes]
        self.bm25 = BM25Retriever(self.all_texts)
        
        # Pre-cache node embeddings
        self.all_embeddings = np.array([n.get_embedding("EMB") for n in self.all_nodes], dtype=np.float32)
        self.leaf_embeddings = np.array([n.get_embedding("EMB") for n in self.leaf_nodes], dtype=np.float32)

    def retrieve_naive(self, query: str, max_tokens: int = 1500) -> Dict[str, Any]:
        """
        Naive RAG (Flat DPR/Dense):
        Dense cosine similarity strictly over Layer 0 (leaf nodes).
        Top-k nodes packed up to max_tokens.
        """
        t0 = time.perf_counter()
        q_emb = get_query_embedding(query, self.emb_model)
        
        sims = [cosine_similarity(q_emb, emb) for emb in self.leaf_embeddings]
        ranked_indices = np.argsort(sims)[::-1]
        
        retrieved_nodes = []
        total_tok = 0
        for idx in ranked_indices:
            node = self.leaf_nodes[idx]
            if total_tok + node.token_size <= max_tokens or not retrieved_nodes:
                retrieved_nodes.append(node)
                total_tok += node.token_size
            if total_tok >= max_tokens:
                break
                
        context = "\n\n".join([n.text for n in retrieved_nodes])
        ret_time = (time.perf_counter() - t0) * 1000
        return {
            "model": "Naive_RAG",
            "nodes": retrieved_nodes,
            "context": context,
            "retrieval_ms": round(ret_time, 2),
            "num_nodes": len(retrieved_nodes),
            "token_count": total_tok,
            "layer_dist": {0: len(retrieved_nodes)}
        }

    def retrieve_raptor(self, query: str, max_tokens: int = 1500) -> Dict[str, Any]:
        """
        RAPTOR (Collapsed Dense Tree Retrieval - Sarthi et al., ICLR 2024 Algorithm 2):
        Dense cosine similarity across all tree nodes simultaneously.
        Greedily accumulates nodes until max_tokens budget.
        """
        t0 = time.perf_counter()
        q_emb = get_query_embedding(query, self.emb_model)
        
        sims = [cosine_similarity(q_emb, emb) for emb in self.all_embeddings]
        ranked_indices = np.argsort(sims)[::-1]
        
        retrieved_nodes = []
        total_tok = 0
        layer_dist = {}
        for idx in ranked_indices:
            node = self.all_nodes[idx]
            if total_tok + node.token_size <= max_tokens or not retrieved_nodes:
                retrieved_nodes.append(node)
                total_tok += node.token_size
                layer_dist[node.layer] = layer_dist.get(node.layer, 0) + 1
            if total_tok >= max_tokens:
                break
                
        context = "\n\n".join([n.text for n in retrieved_nodes])
        ret_time = (time.perf_counter() - t0) * 1000
        return {
            "model": "RAPTOR",
            "nodes": retrieved_nodes,
            "context": context,
            "retrieval_ms": round(ret_time, 2),
            "num_nodes": len(retrieved_nodes),
            "token_count": total_tok,
            "layer_dist": layer_dist
        }

    def retrieve_apex(self, query: str, budget_tokens: int = 750) -> Dict[str, Any]:
        """
        APEX (Collapsed Tree Hybrid Fusion + Dynamic Coverage Gating + Deduplication + Budget Packing):
        1. Dense cosine similarity over all tree nodes
        2. BM25 keyword score over all tree nodes
        3. Query term coverage gating
        4. Composite score: 0.50 * Dense + 0.35 * BM25 + 0.15 * Coverage
        5. Jaccard deduplication on leaf chunks
        6. Budget packing (~750 tokens max for sub-2s TTFT)
        """
        t0 = time.perf_counter()
        q_emb = get_query_embedding(query, self.emb_model)
        
        # Dense scores
        dense_sims = [cosine_similarity(q_emb, emb) for emb in self.all_embeddings]
        
        # BM25 scores
        bm25_raw = self.bm25.score(query)
        max_b = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0
        bm25_norm = [s / max_b for s in bm25_raw]
        
        # Query coverage
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        q_len = max(len(q_words), 1)
        
        candidates = []
        for i, node in enumerate(self.all_nodes):
            d_score = dense_sims[i]
            b_score = bm25_norm[i]
            
            node_words = set(re.findall(r'\b\w+\b', node.text.lower()))
            cov = len(q_words & node_words) / q_len
            
            apex_score = 0.50 * d_score + 0.35 * b_score + 0.15 * cov
            candidates.append({
                "node": node,
                "score": apex_score,
                "dense": d_score,
                "bm25": b_score,
                "coverage": cov,
                "layer": node.layer,
                "is_summary": (node.layer > 0)
            })
            
        # Dynamic coverage gating: require cov >= 0.20 for long queries or 0.30 for short,
        # waived for tree summaries
        cov_thresh = 0.20 if len(query.split()) > 15 else 0.30
        gated = [
            c for c in candidates 
            if (c["coverage"] >= cov_thresh or c["is_summary"]) and c["score"] >= 0.25
        ]
        if not gated:
            gated = sorted(candidates, key=lambda x: x["score"], reverse=True)[:2]
            
        gated.sort(key=lambda x: x["score"], reverse=True)
        
        # Deduplication for leaf chunks (Jaccard > 0.65)
        passed_leaves = [c for c in gated if not c["is_summary"]]
        deduped_leaves = []
        for cand in passed_leaves:
            c_words = set(cand["node"].text.lower().split())
            duplicate = False
            for prev in deduped_leaves:
                p_words = set(prev["node"].text.lower().split())
                inter = len(c_words & p_words)
                union = len(c_words | p_words)
                if union > 0 and (inter / union) > 0.65:
                    duplicate = True
                    break
            if not duplicate:
                deduped_leaves.append(cand)
                
        # Recombine summaries and deduped leaves
        final_pool = [c for c in gated if c["is_summary"]] + deduped_leaves
        final_pool.sort(key=lambda x: x["score"], reverse=True)
        
        # APEX Budget Packing: compact salient token budget (~750 tokens max)
        retrieved_nodes = []
        total_tok = 0
        layer_dist = {}
        for c in final_pool:
            node = c["node"]
            if total_tok + node.token_size <= budget_tokens or not retrieved_nodes:
                retrieved_nodes.append(node)
                total_tok += node.token_size
                layer_dist[node.layer] = layer_dist.get(node.layer, 0) + 1
            if total_tok >= budget_tokens:
                break
                
        context = "\n\n".join([n.text for n in retrieved_nodes])
        ret_time = (time.perf_counter() - t0) * 1000
        return {
            "model": "APEX",
            "nodes": retrieved_nodes,
            "context": context,
            "retrieval_ms": round(ret_time, 2),
            "num_nodes": len(retrieved_nodes),
            "token_count": total_tok,
            "layer_dist": layer_dist
        }


def run_quality_suite(emb_model: OllamaEmbeddingModel) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("▶ 1. EVALUATING QuALITY BENCHMARK (Multiple-Choice Story QA - Tables 2, 4, 7)")
    print("="*80)
    
    with open(os.path.join(DATA_DIR, "quality_stories.json"), "r", encoding="utf-8") as f:
        stories = json.load(f)
        
    story = stories[0]
    s_id = story["story_id"]
    title = story["title"]
    questions = story["questions"]
    print(f"Dataset: QuALITY Story 1 ({s_id}: '{title}', {len(questions)} Questions)")
    
    # Load cached tree
    cache_path = os.path.join(CACHE_DIR, f"quality_{story['article_id']}_RAPTOR.json")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(CACHE_DIR, f"quality_{story['article_id']}_RAPTOR_Clustering.json")
    raptor = RAPTOR.load(cache_path)
    retriever = BenchmarkRetriever(raptor.tree, emb_model)
    
    results = {
        "Naive_RAG": {"correct": 0, "total": 0, "hard_correct": 0, "hard_total": 0, "ret_ms": [], "ttft": [], "total_s": [], "tokens": [], "throughputs": []},
        "RAPTOR": {"correct": 0, "total": 0, "hard_correct": 0, "hard_total": 0, "ret_ms": [], "ttft": [], "total_s": [], "tokens": [], "throughputs": []},
        "APEX": {"correct": 0, "total": 0, "hard_correct": 0, "hard_total": 0, "ret_ms": [], "ttft": [], "total_s": [], "tokens": [], "throughputs": []},
    }
    details = []
    
    for q_idx, q in enumerate(questions):
        q_text = q["question"]
        opts = q["options"]
        raw_gold = q["gold_label"]
        gold = (raw_gold - 1) if (isinstance(raw_gold, int) and raw_gold >= 1) else raw_gold
        is_hard = q["difficult"]
        opt_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(opts)])
        
        print(f"\n[QuALITY Q{q_idx+1}] '{q_text[:60]}...' | Gold: ({chr(65+gold)}) | Hard: {is_hard}")
        
        q_detail = {"question": q_text, "gold": chr(65+gold), "is_hard": is_hard, "evaluations": {}}
        
        for name, get_ctx_fn in [
            ("Naive_RAG", lambda: retriever.retrieve_naive(q_text, max_tokens=1200)),
            ("RAPTOR", lambda: retriever.retrieve_raptor(q_text, max_tokens=1200)),
            ("APEX", lambda: retriever.retrieve_apex(q_text, budget_tokens=450)),
        ]:
            ret_data = get_ctx_fn()
            
            prompt = (
                f"Given Context:\n{ret_data['context']}\n\n"
                f"Question:\n{q_text}\n\n"
                f"Options:\n{opt_str}\n\n"
                f"Select the single best correct option. Output ONLY the letter (A, B, C, or D) corresponding to the correct answer."
            )
            gen = generate_with_profile(prompt, "You are Question Answering Portal", max_tokens=15)
            
            # Evaluate multiple-choice accuracy
            pred = gen["text"]
            correct = evaluate_multiple_choice_answer(pred, gold, opts)
            
            results[name]["total"] += 1
            if correct:
                results[name]["correct"] += 1
            if is_hard:
                results[name]["hard_total"] += 1
                if correct:
                    results[name]["hard_correct"] += 1
                    
            results[name]["ret_ms"].append(ret_data["retrieval_ms"])
            results[name]["ttft"].append(gen["ttft_sec"])
            results[name]["total_s"].append(gen["total_sec"])
            results[name]["tokens"].append(ret_data["token_count"])
            results[name]["throughputs"].append(gen["throughput"])
            
            q_detail["evaluations"][name] = {
                "pred": pred,
                "correct": correct,
                "ret_ms": ret_data["retrieval_ms"],
                "ttft_sec": gen["ttft_sec"],
                "total_sec": gen["total_sec"],
                "tokens": ret_data["token_count"],
                "throughput": gen["throughput"],
                "layer_dist": ret_data["layer_dist"]
            }
            status = "✓" if correct else "✗"
            print(f"  [{name:9}] Pred: {pred[:6]:6} ({status}) | Ret: {ret_data['retrieval_ms']:5.1f}ms | TTFT: {gen['ttft_sec']:.2f}s | Turnaround: {gen['total_sec']:.2f}s | Ctx Tok: {ret_data['token_count']}")
            
        details.append(q_detail)
        
    summary = {}
    for name, d in results.items():
        summary[name] = {
            "accuracy_pct": round((d["correct"] / max(d["total"], 1)) * 100, 2),
            "hard_accuracy_pct": round((d["hard_correct"] / max(d["hard_total"], 1)) * 100, 2),
            "mean_retrieval_ms": round(float(np.mean(d["ret_ms"])), 2),
            "mean_ttft_sec": round(float(np.mean(d["ttft"])), 3),
            "mean_turnaround_sec": round(float(np.mean(d["total_s"])), 3),
            "mean_context_tokens": round(float(np.mean(d["tokens"])), 1),
            "mean_throughput_tok_s": round(float(np.mean(d["throughputs"])), 1),
        }
        
    return {"summary": summary, "details": details}


def run_narrativeqa_suite(emb_model: OllamaEmbeddingModel) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("▶ 2. EVALUATING NarrativeQA BENCHMARK (Story QA - Tables 1 & 6)")
    print("="*80)
    
    with open(os.path.join(DATA_DIR, "narrativeqa_stories.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
        
    story_id = items[0]["id"]
    story_title = items[0]["title"]
    print(f"Dataset: NarrativeQA ({story_id[:8]}: '{story_title}', {len(items[:4])} Questions evaluated)")
    
    cache_path = os.path.join(CACHE_DIR, f"narrativeqa_{story_id[:8]}_RAPTOR.json")
    raptor = RAPTOR.load(cache_path)
    retriever = BenchmarkRetriever(raptor.tree, emb_model)
    
    methods = ["Naive_RAG", "RAPTOR", "APEX"]
    metrics_accum = {
        m: {"f1": [], "bleu_1": [], "bleu_4": [], "rouge_l": [], "meteor": [], "ret_ms": [], "ttft": [], "total_s": [], "tokens": [], "throughputs": []}
        for m in methods
    }
    details = []
    
    for idx, item in enumerate(items[:4]):
        q = item["question"]
        golds = item["answers"]
        print(f"\n[NarrativeQA Q{idx+1}] '{q}'")
        print(f"  Gold Ref: {golds[0][:60]}...")
        
        q_detail = {"question": q, "golds": golds, "evaluations": {}}
        
        for name, get_ctx_fn in [
            ("Naive_RAG", lambda: retriever.retrieve_naive(q, max_tokens=1000)),
            ("RAPTOR", lambda: retriever.retrieve_raptor(q, max_tokens=1000)),
            ("APEX", lambda: retriever.retrieve_apex(q, budget_tokens=450)),
        ]:
            ret_data = get_ctx_fn()
            prompt = (
                f"Given Context:\n{ret_data['context']}\n\n"
                f"Question: {q}\n\n"
                f"Answer the question directly and concisely based solely on the provided context:"
            )
            gen = generate_with_profile(prompt, "You are Question Answering Portal", max_tokens=50)
            ans = gen["text"]
            
            # Official NLP evaluation metrics
            f1 = compute_max_token_f1(golds, ans) * 100
            bleu = compute_bleu_scores(golds, ans)
            rouge_l = compute_rouge_l(golds, ans)
            meteor = compute_meteor(golds, ans)
            
            metrics_accum[name]["f1"].append(f1)
            metrics_accum[name]["bleu_1"].append(bleu["bleu_1"])
            metrics_accum[name]["bleu_4"].append(bleu["bleu_4"])
            metrics_accum[name]["rouge_l"].append(rouge_l)
            metrics_accum[name]["meteor"].append(meteor)
            metrics_accum[name]["ret_ms"].append(ret_data["retrieval_ms"])
            metrics_accum[name]["ttft"].append(gen["ttft_sec"])
            metrics_accum[name]["total_s"].append(gen["total_sec"])
            metrics_accum[name]["tokens"].append(ret_data["token_count"])
            metrics_accum[name]["throughputs"].append(gen["throughput"])
            
            q_detail["evaluations"][name] = {
                "answer": ans,
                "f1": round(f1, 2),
                "bleu_1": bleu["bleu_1"],
                "bleu_4": bleu["bleu_4"],
                "rouge_l": rouge_l,
                "meteor": meteor,
                "ret_ms": ret_data["retrieval_ms"],
                "ttft_sec": gen["ttft_sec"],
                "total_sec": gen["total_sec"],
                "tokens": ret_data["token_count"],
                "throughput": gen["throughput"],
                "layer_dist": ret_data["layer_dist"]
            }
            print(f"  [{name:9}] F1: {f1:5.1f} | ROUGE-L: {rouge_l:5.1f} | Ret: {ret_data['retrieval_ms']:5.1f}ms | TTFT: {gen['ttft_sec']:.2f}s | Turnaround: {gen['total_sec']:.2f}s | Ans: {ans[:40]}...")
            
        details.append(q_detail)
        
    summary = {}
    for name, d in metrics_accum.items():
        summary[name] = {
            "mean_f1": round(float(np.mean(d["f1"])), 2),
            "mean_bleu_1": round(float(np.mean(d["bleu_1"])), 2),
            "mean_bleu_4": round(float(np.mean(d["bleu_4"])), 2),
            "mean_rouge_l": round(float(np.mean(d["rouge_l"])), 2),
            "mean_meteor": round(float(np.mean(d["meteor"])), 2),
            "mean_retrieval_ms": round(float(np.mean(d["ret_ms"])), 2),
            "mean_ttft_sec": round(float(np.mean(d["ttft"])), 3),
            "mean_turnaround_sec": round(float(np.mean(d["total_s"])), 3),
            "mean_context_tokens": round(float(np.mean(d["tokens"])), 1),
            "mean_throughput_tok_s": round(float(np.mean(d["throughputs"])), 1),
        }
        
    return {"summary": summary, "details": details}


def run_qasper_suite(emb_model: OllamaEmbeddingModel) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("▶ 3. EVALUATING QASPER BENCHMARK (Scientific Paper QA - Tables 3 & 5)")
    print("="*80)
    
    with open(os.path.join(DATA_DIR, "qasper_papers.json"), "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    paper = papers[0]
    p_id = paper["paper_id"]
    title = paper["title"]
    qas = paper["qas"]
    print(f"Dataset: QASPER Paper ({p_id}: '{title[:50]}...', {len(qas[:4])} Questions evaluated)")
    
    cache_path = os.path.join(CACHE_DIR, f"qasper_{p_id}_RAPTOR.json")
    raptor = RAPTOR.load(cache_path)
    retriever = BenchmarkRetriever(raptor.tree, emb_model)
    
    methods = ["Naive_RAG", "RAPTOR", "APEX"]
    metrics_accum = {
        m: {"f1": [], "ret_ms": [], "ttft": [], "total_s": [], "tokens": [], "throughputs": []}
        for m in methods
    }
    details = []
    
    for idx, q_entry in enumerate(qas[:4]):
        q = q_entry["question"]
        golds = q_entry["gold_answers"]
        print(f"\n[QASPER Q{idx+1}] '{q}'")
        print(f"  Gold Ref: {golds[0][:60]}...")
        
        q_detail = {"question": q, "golds": golds, "evaluations": {}}
        
        for name, get_ctx_fn in [
            ("Naive_RAG", lambda: retriever.retrieve_naive(q, max_tokens=1000)),
            ("RAPTOR", lambda: retriever.retrieve_raptor(q, max_tokens=1000)),
            ("APEX", lambda: retriever.retrieve_apex(q, budget_tokens=450)),
        ]:
            ret_data = get_ctx_fn()
            prompt = (
                f"Given Context:\n{ret_data['context']}\n\n"
                f"Question: {q}\n\n"
                f"Answer the question directly and concisely based solely on the provided context:"
            )
            gen = generate_with_profile(prompt, "You are Question Answering Portal", max_tokens=50)
            ans = gen["text"]
            
            f1 = compute_max_token_f1(golds, ans) * 100
            
            metrics_accum[name]["f1"].append(f1)
            metrics_accum[name]["ret_ms"].append(ret_data["retrieval_ms"])
            metrics_accum[name]["ttft"].append(gen["ttft_sec"])
            metrics_accum[name]["total_s"].append(gen["total_sec"])
            metrics_accum[name]["tokens"].append(ret_data["token_count"])
            metrics_accum[name]["throughputs"].append(gen["throughput"])
            
            q_detail["evaluations"][name] = {
                "answer": ans,
                "f1": round(f1, 2),
                "ret_ms": ret_data["retrieval_ms"],
                "ttft_sec": gen["ttft_sec"],
                "total_sec": gen["total_sec"],
                "tokens": ret_data["token_count"],
                "throughput": gen["throughput"],
                "layer_dist": ret_data["layer_dist"]
            }
            print(f"  [{name:9}] Token F1: {f1:5.1f}% | Ret: {ret_data['retrieval_ms']:5.1f}ms | TTFT: {gen['ttft_sec']:.2f}s | Turnaround: {gen['total_sec']:.2f}s | Ans: {ans[:40]}...")
            
        details.append(q_detail)
        
    summary = {}
    for name, d in metrics_accum.items():
        summary[name] = {
            "mean_token_f1": round(float(np.mean(d["f1"])), 2),
            "mean_retrieval_ms": round(float(np.mean(d["ret_ms"])), 2),
            "mean_ttft_sec": round(float(np.mean(d["ttft"])), 3),
            "mean_turnaround_sec": round(float(np.mean(d["total_s"])), 3),
            "mean_context_tokens": round(float(np.mean(d["tokens"])), 1),
            "mean_throughput_tok_s": round(float(np.mean(d["throughputs"])), 1),
        }
        
    return {"summary": summary, "details": details}


def run_cinderella_suite(emb_model: OllamaEmbeddingModel) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("▶ 4. EVALUATING Cinderella THEMATIC STUDY (Tables 12 & 13 / Appendix G)")
    print("="*80)
    
    with open(os.path.join(DATA_DIR, "cinderella.json"), "r", encoding="utf-8") as f:
        cind_data = json.load(f)
        
    questions = cind_data["questions"]
    print(f"Dataset: Cinderella ({len(questions)} Thematic Multi-Hop Questions)")
    
    cache_path = os.path.join(CACHE_DIR, "cinderella_RAPTOR_Clustering.json")
    raptor = RAPTOR.load(cache_path)
    retriever = BenchmarkRetriever(raptor.tree, emb_model)
    
    methods = ["Naive_RAG", "RAPTOR", "APEX"]
    metrics_accum = {
        m: {"cos_sim": [], "ret_ms": [], "ttft": [], "total_s": [], "tokens": [], "throughputs": []}
        for m in methods
    }
    details = []
    
    # Ground truth reference concepts for semantic cosine scoring
    gold_thematic_refs = {
        "What is the central theme of the story?": [
            "Goodness and piety are rewarded while wickedness, cruelty, and falsehood are punished.",
            "Humility, perseverance, and purity overcome adversity and triumph over malicious step-relations."
        ],
        "How did Cinderella find a happy ending?": [
            "Through her piety at her mother's grave, the magical assistance of the doves and hazel tree, and the golden slipper fitting her foot.",
            "The king's son discovered her true identity after the shoe fit only her, and they were married while the evil sisters were punished."
        ]
    }
    
    for idx, q_item in enumerate(questions):
        q = q_item["question"]
        golds = gold_thematic_refs.get(q, ["Cinderella triumphs over wickedness."])
        print(f"\n[Cinderella Q{idx+1}] '{q}'")
        
        q_detail = {"question": q, "golds": golds, "evaluations": {}}
        
        for name, get_ctx_fn in [
            ("Naive_RAG", lambda: retriever.retrieve_naive(q, max_tokens=1000)),
            ("RAPTOR", lambda: retriever.retrieve_raptor(q, max_tokens=1000)),
            ("APEX", lambda: retriever.retrieve_apex(q, budget_tokens=450)),
        ]:
            ret_data = get_ctx_fn()
            prompt = (
                f"Given Context:\n{ret_data['context']}\n\n"
                f"Question: {q}\n\n"
                f"Answer the question directly and concisely based solely on the provided context:"
            )
            gen = generate_with_profile(prompt, "You are Question Answering Portal", max_tokens=60)
            ans = gen["text"]
            
            # Semantic cosine similarity against gold concepts
            ans_emb = get_query_embedding(ans, emb_model)
            gold_embs = [get_query_embedding(g, emb_model) for g in golds]
            cos_scores = [cosine_similarity(ans_emb, g_e) for g_e in gold_embs]
            max_cos = max(cos_scores) if cos_scores else 0.0
            
            metrics_accum[name]["cos_sim"].append(max_cos)
            metrics_accum[name]["ret_ms"].append(ret_data["retrieval_ms"])
            metrics_accum[name]["ttft"].append(gen["ttft_sec"])
            metrics_accum[name]["total_s"].append(gen["total_sec"])
            metrics_accum[name]["tokens"].append(ret_data["token_count"])
            metrics_accum[name]["throughputs"].append(gen["throughput"])
            
            q_detail["evaluations"][name] = {
                "answer": ans,
                "cosine_similarity": round(max_cos, 4),
                "ret_ms": ret_data["retrieval_ms"],
                "ttft_sec": gen["ttft_sec"],
                "total_sec": gen["total_sec"],
                "tokens": ret_data["token_count"],
                "throughput": gen["throughput"],
                "layer_dist": ret_data["layer_dist"]
            }
            print(f"  [{name:9}] Cosine: {max_cos:.3f} | Ret: {ret_data['retrieval_ms']:5.1f}ms | TTFT: {gen['ttft_sec']:.2f}s | Turnaround: {gen['total_sec']:.2f}s | Ans: {ans[:40]}...")
            
        details.append(q_detail)
        
    summary = {}
    for name, d in metrics_accum.items():
        summary[name] = {
            "mean_cosine_similarity": round(float(np.mean(d["cos_sim"])), 4),
            "mean_retrieval_ms": round(float(np.mean(d["ret_ms"])), 2),
            "mean_ttft_sec": round(float(np.mean(d["ttft"])), 3),
            "mean_turnaround_sec": round(float(np.mean(d["total_s"])), 3),
            "mean_context_tokens": round(float(np.mean(d["tokens"])), 1),
            "mean_throughput_tok_s": round(float(np.mean(d["throughputs"])), 1),
        }
        
    return {"summary": summary, "details": details}


def main():
    print("="*80)
    print("OFFICIAL RAPTOR PAPER (ICLR 2024) 3-WAY COMPARATIVE BENCHMARK")
    print("Architectures: Naive RAG vs. RAPTOR vs. APEX")
    print(f"Hardware & Models: Native Ollama {GEN_MODEL} (pinned) & {EMB_MODEL}")
    print("="*80)
    
    t_global_start = time.time()
    emb_model = OllamaEmbeddingModel(model_name=EMB_MODEL)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": GEN_MODEL,
        "embedding_model": EMB_MODEL,
        "quality_benchmark": run_quality_suite(emb_model),
        "narrativeqa_benchmark": run_narrativeqa_suite(emb_model),
        "qasper_benchmark": run_qasper_suite(emb_model),
        "cinderella_benchmark": run_cinderella_suite(emb_model),
        "benchmark_duration_sec": round(time.time() - t_global_start, 2)
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print(f"🎉 3-WAY BENCHMARK COMPLETE IN {results['benchmark_duration_sec']}s!")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()

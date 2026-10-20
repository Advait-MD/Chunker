"""
run_raptor_benchmarks.py — Comprehensive Benchmark Suite & Empirical Recalculation
Recreating all experiments, tables, and ablations from Sarthi et al. (ICLR 2024)
using local Gemma 3:1b and nomic-embed-text / DPR / BM25.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from raptor import (
    RAPTOR,
    Tree,
    Node,
    OllamaEmbeddingModel,
    BM25Retriever,
    GemmaSummarizer,
    GemmaQAModel,
    RAPTOR_Clustering,
    Recency_Clustering,
    split_text_into_chunks,
    compute_token_f1,
    compute_max_token_f1,
    compute_bleu_scores,
    compute_rouge_l,
    compute_meteor,
    evaluate_narrativeqa_metrics,
    evaluate_multiple_choice_answer,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("raptor.benchmark")

DATA_DIR = os.path.join(PROJECT_ROOT, "benchmark_data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_or_build_tree(
    name: str,
    text: str,
    clustering_algo=RAPTOR_Clustering,
    max_tokens: int = 100,
    reduction_dim: int = 5,
    threshold: float = 0.1
) -> RAPTOR:
    """Build or load cached RAPTOR tree."""
    cache_path = os.path.join(CACHE_DIR, f"{name}_{clustering_algo.__name__}.json")
    emb_model = OllamaEmbeddingModel()
    qa_model = GemmaQAModel()

    if os.path.exists(cache_path):
        logger.info(f"Loading cached tree for '{name}' from {cache_path}")
        return RAPTOR.load(cache_path, embedding_model=emb_model, qa_model=qa_model)

    logger.info(f"Building tree for '{name}' ({len(text.split())} words)...")
    raptor = RAPTOR(
        embedding_model=emb_model,
        clustering_algorithm=clustering_algo,
        max_tokens_per_chunk=max_tokens,
        reduction_dimension=reduction_dim,
        threshold=threshold,
        num_layers=4,
    )
    raptor.add_documents(text, verbose=True)
    raptor.save(cache_path)
    logger.info(f"Tree for '{name}' saved to {cache_path}")
    return raptor


def run_cinderella_qualitative_benchmark() -> Dict[str, Any]:
    """
    Recreates Tables 12 & 13 / Appendix G:
    Thematic and multi-hop questions over Cinderella fairytale.
    Comprehends the whole narrative arc (RAPTOR vs Flat DPR/Dense).
    """
    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARK: Cinderella Qualitative Study (Tables 12 & 13)")
    logger.info("=" * 60)

    cind_path = os.path.join(DATA_DIR, "cinderella.json")
    with open(cind_path, "r", encoding="utf-8") as f:
        cind_data = json.load(f)

    text = cind_data["text"]
    raptor = get_or_build_tree("cinderella", text, reduction_dim=3)

    # Flat baseline: leaf nodes only (DPR/Dense flat retrieval)
    leaf_nodes = raptor.tree.get_layer(0)
    leaf_texts = [n.text for n in leaf_nodes]

    results = []
    for q_item in cind_data["questions"]:
        q = q_item["question"]
        logger.info(f"Evaluating Question: '{q}'")

        # 1. RAPTOR Collapsed Tree Retrieval
        r_nodes, r_ctx, r_dist = raptor.retrieve(q, collapse_tree=True, max_tokens=1500)
        r_ans = raptor.qa_model.answer_question(r_ctx, q)

        # 2. Flat Dense Retrieval (Leaf nodes only)
        flat_nodes, flat_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=1500, allowed_layers=[0])
        flat_ans = raptor.qa_model.answer_question(flat_ctx, q)

        results.append({
            "question": q,
            "type": q_item["type"],
            "raptor": {
                "retrieved_nodes_count": len(r_nodes),
                "layer_distribution": r_dist,
                "retrieved_excerpts": [n.text[:180] + "..." for n in r_nodes[:3]],
                "answer_gemma": r_ans,
            },
            "flat_dense": {
                "retrieved_nodes_count": len(flat_nodes),
                "retrieved_excerpts": [n.text[:180] + "..." for n in flat_nodes[:3]],
                "answer_gemma": flat_ans,
            }
        })
        logger.info(f"  RAPTOR Answer: {r_ans[:120]}...")
        logger.info(f"  Flat Answer:   {flat_ans[:120]}...")

    return {"cinderella_qualitative": results}


def run_quality_benchmarks() -> Dict[str, Any]:
    """
    Recreates Tables 2, 4, 7, 8, 9, 14:
    Multiple-Choice Question Answering on QuALITY long-form stories.
    Includes:
    - Tables 2, 4, 7: Controlled comparisons (BM25 vs DPR vs RAPTOR) + Overall & HARD subset
    - Table 8 / 14: Tree Structure Layer Ablation (Layer 0, Layer 1, Layer 2, 2 layers, 3 layers)
    - Table 9: Clustering Mechanism Ablation (RAPTOR GMM vs Recency contiguous tree)
    """
    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARK: QuALITY Long-Context QA & Layer Ablations (Tables 2, 4, 7, 8, 9)")
    logger.info("=" * 60)

    quality_path = os.path.join(DATA_DIR, "quality_stories.json")
    with open(quality_path, "r", encoding="utf-8") as f:
        stories = json.load(f)

    all_eval_results = []
    layer_ablation_records = {}
    clustering_ablation_records = {"raptor_gmm": {"correct": 0, "total": 0}, "recency": {"correct": 0, "total": 0}}
    retriever_comparison = {
        "BM25": {"correct": 0, "total": 0},
        "DPR_Flat": {"correct": 0, "total": 0},
        "RAPTOR": {"correct": 0, "total": 0, "hard_correct": 0, "hard_total": 0}
    }

    # For layer percentage tracking
    layer_hits = {0: 0, 1: 0, 2: 0, 3: 0}
    total_retrieved_nodes = 0

    for story_idx, story in enumerate(stories[:2]):  # Stories 1 & 2
        s_id = story["story_id"]
        title = story["title"]
        text = story["article"]
        questions = story["questions"]

        logger.info(f"\n--- Processing {s_id}: '{title}' ({len(text.split())} words, {len(questions)} Qs) ---")

        # Build RAPTOR Tree (GMM soft clustering)
        raptor = get_or_build_tree(f"quality_{story['article_id']}", text, clustering_algo=RAPTOR_Clustering, reduction_dim=4)

        # Build Recency-based Tree for Table 9 Ablation
        recency_raptor = get_or_build_tree(f"quality_{story['article_id']}_recency", text, clustering_algo=Recency_Clustering, reduction_dim=4)

        # Build BM25 flat retriever over leaf chunks
        leaf_chunks = [n.text for n in raptor.tree.get_layer(0)]
        bm25 = BM25Retriever(leaf_chunks)

        # --- ABLATION: Tree Structure Layer Ablation (Table 8 / 14) ---
        layer_ablation_records[s_id] = {
            "layer_0_leaves": 0,
            "layer_1_only": 0,
            "layer_2_only": 0,
            "layers_0_1": 0,
            "layers_1_2": 0,
            "layers_full_3": 0,
            "total": len(questions)
        }

        for q_idx, q in enumerate(questions):
            q_text = q["question"]
            opts = q["options"]
            gold = q["gold_label"]
            is_hard = q["difficult"]

            # 1. RAPTOR Full Collapsed Tree
            r_nodes, r_ctx, r_dist = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000)
            r_pred = raptor.qa_model.answer_question(r_ctx, q_text, options=opts)
            r_correct = evaluate_multiple_choice_answer(r_pred, gold, opts)

            retriever_comparison["RAPTOR"]["total"] += 1
            if r_correct:
                retriever_comparison["RAPTOR"]["correct"] += 1
            if is_hard:
                retriever_comparison["RAPTOR"]["hard_total"] += 1
                if r_correct:
                    retriever_comparison["RAPTOR"]["hard_correct"] += 1

            for l, cnt in r_dist.items():
                layer_hits[l] = layer_hits.get(l, 0) + cnt
                total_retrieved_nodes += cnt

            # 2. Recency-based Tree (Table 9)
            rec_nodes, rec_ctx, _ = recency_raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000)
            rec_pred = recency_raptor.qa_model.answer_question(rec_ctx, q_text, options=opts)
            rec_correct = evaluate_multiple_choice_answer(rec_pred, gold, opts)

            clustering_ablation_records["raptor_gmm"]["total"] += 1
            clustering_ablation_records["recency"]["total"] += 1
            if r_correct:
                clustering_ablation_records["raptor_gmm"]["correct"] += 1
            if rec_correct:
                clustering_ablation_records["recency"]["correct"] += 1

            # 3. Flat Dense / DPR baseline (Layer 0 only)
            dpr_nodes, dpr_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[0])
            dpr_pred = raptor.qa_model.answer_question(dpr_ctx, q_text, options=opts)
            dpr_correct = evaluate_multiple_choice_answer(dpr_pred, gold, opts)
            retriever_comparison["DPR_Flat"]["total"] += 1
            if dpr_correct:
                retriever_comparison["DPR_Flat"]["correct"] += 1

            # 4. Flat BM25 baseline
            bm25_scores = bm25.score(q_text)
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:5]
            bm25_ctx = "\n\n".join([leaf_chunks[i] for i in top_bm25_idx])
            bm25_pred = raptor.qa_model.answer_question(bm25_ctx, q_text, options=opts)
            bm25_correct = evaluate_multiple_choice_answer(bm25_pred, gold, opts)
            retriever_comparison["BM25"]["total"] += 1
            if bm25_correct:
                retriever_comparison["BM25"]["correct"] += 1

            # 5. Layer combinations for Table 8 / 14
            # Layer 0 only
            if dpr_correct:
                layer_ablation_records[s_id]["layer_0_leaves"] += 1
            # Layer 1 only
            l1_nodes, l1_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[1])
            if l1_nodes:
                l1_pred = raptor.qa_model.answer_question(l1_ctx, q_text, options=opts)
                if evaluate_multiple_choice_answer(l1_pred, gold, opts):
                    layer_ablation_records[s_id]["layer_1_only"] += 1
            # Layer 2 only
            l2_nodes, l2_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[2])
            if l2_nodes:
                l2_pred = raptor.qa_model.answer_question(l2_ctx, q_text, options=opts)
                if evaluate_multiple_choice_answer(l2_pred, gold, opts):
                    layer_ablation_records[s_id]["layer_2_only"] += 1
            # Layers 0 + 1
            l01_nodes, l01_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[0, 1])
            l01_pred = raptor.qa_model.answer_question(l01_ctx, q_text, options=opts)
            if evaluate_multiple_choice_answer(l01_pred, gold, opts):
                layer_ablation_records[s_id]["layers_0_1"] += 1
            # Layers 1 + 2
            l12_nodes, l12_ctx, _ = raptor.retrieve(q_text, collapse_tree=True, max_tokens=2000, allowed_layers=[1, 2])
            if l12_nodes:
                l12_pred = raptor.qa_model.answer_question(l12_ctx, q_text, options=opts)
                if evaluate_multiple_choice_answer(l12_pred, gold, opts):
                    layer_ablation_records[s_id]["layers_1_2"] += 1
            # Full 3 layers
            if r_correct:
                layer_ablation_records[s_id]["layers_full_3"] += 1

            logger.info(f"  Q{q_idx+1}: '{q_text[:50]}...' -> Gold: {chr(65+gold)} | RAPTOR: {r_pred} ({'✓' if r_correct else '✗'}) | DPR: {'✓' if dpr_correct else '✗'} | BM25: {'✓' if bm25_correct else '✗'}")

    # Compute percentage accuracies
    def pct(d):
        return round((d["correct"] / max(d["total"], 1)) * 100, 2)

    quality_summary = {
        "retriever_comparison": {
            "BM25_Accuracy": pct(retriever_comparison["BM25"]),
            "DPR_Flat_Accuracy": pct(retriever_comparison["DPR_Flat"]),
            "RAPTOR_Accuracy": pct(retriever_comparison["RAPTOR"]),
            "RAPTOR_HARD_Subset_Accuracy": round((retriever_comparison["RAPTOR"]["hard_correct"] / max(retriever_comparison["RAPTOR"]["hard_total"], 1)) * 100, 2),
        },
        "clustering_ablation_table_9": {
            "raptor_gmm_soft_clustering": pct(clustering_ablation_records["raptor_gmm"]),
            "recency_based_window_tree": pct(clustering_ablation_records["recency"]),
        },
        "layer_ablation_tables_8_14": {
            s_id: {k: round((v / max(records["total"], 1)) * 100, 2) for k, v in records.items() if k != "total"}
            for s_id, records in layer_ablation_records.items()
        },
        "layer_retrieval_distribution": {
            l: round((cnt / max(total_retrieved_nodes, 1)) * 100, 2) for l, cnt in layer_hits.items()
        }
    }

    return quality_summary


def run_narrativeqa_benchmarks() -> Dict[str, Any]:
    """
    Recreates Table 1 & Table 6:
    NarrativeQA Reading Comprehension benchmark evaluating ROUGE-L, BLEU-1, BLEU-4, METEOR
    comparing With RAPTOR vs Without RAPTOR across SBERT, BM25, DPR with reader Gemma 3:1b.
    """
    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARK: NarrativeQA (Table 1 & Table 6)")
    logger.info("=" * 60)

    nqa_path = os.path.join(DATA_DIR, "narrativeqa_stories.json")
    with open(nqa_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # We evaluate 4 test stories
    methods = ["SBERT_with_RAPTOR", "SBERT_without_RAPTOR", "BM25_with_RAPTOR", "BM25_without_RAPTOR"]
    metrics_accum = {m: {"rouge_l": [], "bleu_1": [], "bleu_4": [], "meteor": []} for m in methods}

    for idx, item in enumerate(items[:4]):
        title = item["title"]
        text = item["text"]
        q = item["question"]
        gold_refs = item["answers"]

        logger.info(f"\nNarrativeQA Item {idx+1}: {title} | Q: '{q}'")

        raptor = get_or_build_tree(f"nqa_{idx}", text, reduction_dim=3)
        leaf_chunks = [n.text for n in raptor.tree.get_layer(0)]
        bm25 = BM25Retriever(leaf_chunks)

        # 1. SBERT With RAPTOR (Collapsed tree)
        r_nodes, r_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=1500)
        ans_sbert_raptor = raptor.qa_model.answer_question(r_ctx, q)
        m_sr = evaluate_narrativeqa_metrics(gold_refs, ans_sbert_raptor)

        # 2. SBERT Without RAPTOR (Flat leaf chunks)
        flat_nodes, flat_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=1500, allowed_layers=[0])
        ans_sbert_flat = raptor.qa_model.answer_question(flat_ctx, q)
        m_sf = evaluate_narrativeqa_metrics(gold_refs, ans_sbert_flat)

        # 3. BM25 Without RAPTOR
        scores = bm25.score(q)
        top_idx = np.argsort(scores)[::-1][:5]
        bm25_ctx = "\n\n".join([leaf_chunks[i] for i in top_idx])
        ans_bm25_flat = raptor.qa_model.answer_question(bm25_ctx, q)
        m_bf = evaluate_narrativeqa_metrics(gold_refs, ans_bm25_flat)

        # 4. BM25 With RAPTOR (BM25 over all flattened tree nodes)
        all_tree_texts = [n.text for n in raptor.tree.get_all_nodes()]
        bm25_tree = BM25Retriever(all_tree_texts)
        scores_tree = bm25_tree.score(q)
        top_tree_idx = np.argsort(scores_tree)[::-1][:5]
        bm25_tree_ctx = "\n\n".join([all_tree_texts[i] for i in top_tree_idx])
        ans_bm25_tree = raptor.qa_model.answer_question(bm25_tree_ctx, q)
        m_br = evaluate_narrativeqa_metrics(gold_refs, ans_bm25_tree)

        for m_key, scores_dict in [
            ("SBERT_with_RAPTOR", m_sr),
            ("SBERT_without_RAPTOR", m_sf),
            ("BM25_with_RAPTOR", m_br),
            ("BM25_without_RAPTOR", m_bf),
        ]:
            for metric in ["rouge_l", "bleu_1", "bleu_4", "meteor"]:
                metrics_accum[m_key][metric].append(scores_dict[metric])

        logger.info(f"  SBERT+RAPTOR Answer: {ans_sbert_raptor[:80]}... (ROUGE-L: {m_sr['rouge_l']})")
        logger.info(f"  SBERT-Flat   Answer: {ans_sbert_flat[:80]}... (ROUGE-L: {m_sf['rouge_l']})")

    summary_table = {}
    for m, m_data in metrics_accum.items():
        summary_table[m] = {
            metric: round(float(np.mean(vals)), 2) for metric, vals in m_data.items()
        }

    return {"narrativeqa_table_1": summary_table}


def run_qasper_benchmarks() -> Dict[str, Any]:
    """
    Recreates Table 3 & Table 5:
    QASPER Information-Seeking Benchmark evaluating Token F1 Match
    comparing Title+Abstract vs BM25 vs DPR vs RAPTOR with reader Gemma 3:1b.
    """
    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARK: QASPER (Table 3 & Table 5)")
    logger.info("=" * 60)

    qasper_path = os.path.join(DATA_DIR, "qasper_papers.json")
    with open(qasper_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    scores_accum = {
        "Title_and_Abstract": [],
        "BM25": [],
        "DPR_Flat": [],
        "RAPTOR": []
    }

    for p_idx, paper in enumerate(papers[:2]):
        title = paper["title"]
        abstract = paper["abstract"]
        full_text = paper["full_text"]
        qas = paper["qas"]

        logger.info(f"\nProcessing QASPER Paper {p_idx+1}: '{title}' ({len(qas)} questions)")
        raptor = get_or_build_tree(f"qasper_{p_idx}", full_text, reduction_dim=4)

        leaf_chunks = [n.text for n in raptor.tree.get_layer(0)]
        bm25 = BM25Retriever(leaf_chunks)

        for q_entry in qas[:3]:  # Top questions per paper
            q = q_entry["question"]
            gold_refs = q_entry["gold_answers"]

            # 1. Title + Abstract
            title_abs_ctx = f"Title: {title}\nAbstract: {abstract}"
            ans_ta = raptor.qa_model.answer_question(title_abs_ctx, q)
            f1_ta = compute_max_token_f1(gold_refs, ans_ta) * 100

            # 2. BM25
            scores = bm25.score(q)
            top_idx = np.argsort(scores)[::-1][:5]
            bm25_ctx = "\n\n".join([leaf_chunks[i] for i in top_idx])
            ans_bm25 = raptor.qa_model.answer_question(bm25_ctx, q)
            f1_bm25 = compute_max_token_f1(gold_refs, ans_bm25) * 100

            # 3. Flat Dense / DPR
            dpr_nodes, dpr_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=1500, allowed_layers=[0])
            ans_dpr = raptor.qa_model.answer_question(dpr_ctx, q)
            f1_dpr = compute_max_token_f1(gold_refs, ans_dpr) * 100

            # 4. RAPTOR Collapsed Tree
            r_nodes, r_ctx, _ = raptor.retrieve(q, collapse_tree=True, max_tokens=1500)
            ans_raptor = raptor.qa_model.answer_question(r_ctx, q)
            f1_raptor = compute_max_token_f1(gold_refs, ans_raptor) * 100

            scores_accum["Title_and_Abstract"].append(f1_ta)
            scores_accum["BM25"].append(f1_bm25)
            scores_accum["DPR_Flat"].append(f1_dpr)
            scores_accum["RAPTOR"].append(f1_raptor)

            logger.info(f"  Q: '{q[:50]}...' -> F1: RAPTOR={f1_raptor:.1f} | DPR={f1_dpr:.1f} | BM25={f1_bm25:.1f} | Title+Abs={f1_ta:.1f}")

    qasper_summary = {
        k: round(float(np.mean(vals)), 2) for k, vals in scores_accum.items()
    }
    return {"qasper_table_3": qasper_summary}


def run_compression_statistics_benchmark() -> Dict[str, Any]:
    """
    Recreates Table 10 (Appendix C):
    Dataset Statistics, Average Summary Length, Child Node Length, Child Count, and Compression Ratio.
    """
    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARK: Tree Compression Statistics (Table 10)")
    logger.info("=" * 60)

    # Collect statistics from cached trees
    dataset_trees = {
        "Cinderella": os.path.join(CACHE_DIR, "cinderella_RAPTOR_Clustering.json"),
        "QuALITY": os.path.join(CACHE_DIR, "quality_00057_RAPTOR_Clustering.json"),
        "NarrativeQA": os.path.join(CACHE_DIR, "nqa_0_RAPTOR_Clustering.json"),
        "QASPER": os.path.join(CACHE_DIR, "qasper_0_RAPTOR_Clustering.json"),
    }

    stats_table = {}
    for d_name, path in dataset_trees.items():
        if os.path.exists(path):
            tree = Tree.load(path)
            stats = tree.get_statistics()
            stats_table[d_name] = {
                "avg_summary_length_tokens": stats["avg_summary_length_tokens"],
                "avg_child_node_length_tokens": stats["avg_child_node_length_tokens"],
                "avg_children_per_parent": stats["avg_children_per_parent"],
                "avg_compression_ratio": stats["avg_compression_ratio"],
                "total_nodes": stats["total_nodes"],
                "num_layers": stats["num_layers"],
            }

    return {"compression_statistics_table_10": stats_table}


def main():
    logger.info("=" * 80)
    logger.info("RAPTOR BENCHMARK SUITE — FULL LOCAL GEMMA 3:1B RECALCULATION")
    logger.info("=" * 80)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": "gemma3:1b",
        "embedding_model": "nomic-embed-text",
        "temperature": 0.0,
    }

    # 1. Cinderella Qualitative Benchmark (Tables 12 & 13)
    t0 = time.time()
    cind_results = run_cinderella_qualitative_benchmark()
    all_results.update(cind_results)
    logger.info(f"Cinderella qualitative completed in {round(time.time() - t0, 1)}s")

    # 2. QuALITY Benchmark & Ablations (Tables 2, 4, 7, 8, 9, 14, 18)
    t0 = time.time()
    quality_results = run_quality_benchmarks()
    all_results.update(quality_results)
    logger.info(f"QuALITY benchmark completed in {round(time.time() - t0, 1)}s")

    # 3. NarrativeQA Benchmark (Table 1 & Table 6)
    t0 = time.time()
    nqa_results = run_narrativeqa_benchmarks()
    all_results.update(nqa_results)
    logger.info(f"NarrativeQA benchmark completed in {round(time.time() - t0, 1)}s")

    # 4. QASPER Benchmark (Table 3 & Table 5)
    t0 = time.time()
    qasper_results = run_qasper_benchmarks()
    all_results.update(qasper_results)
    logger.info(f"QASPER benchmark completed in {round(time.time() - t0, 1)}s")

    # 5. Tree Compression & Scaling Statistics (Table 10)
    t0 = time.time()
    comp_results = run_compression_statistics_benchmark()
    all_results.update(comp_results)
    logger.info(f"Compression statistics completed in {round(time.time() - t0, 1)}s")

    # Save to JSON
    output_path = os.path.join(PROJECT_ROOT, "raptor_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"ALL BENCHMARKS SUCCESSFULLY RECALCULATED AND SAVED TO: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

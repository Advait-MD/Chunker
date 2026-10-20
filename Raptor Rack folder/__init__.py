"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (ICLR 2024).
Local Gemma 3:1b & Ollama Implementation.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    from .tree_structures import Node, Tree
    from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig, split_text_into_chunks
    from .cluster_utils import (
        RAPTOR_Clustering,
        Recency_Clustering,
        GMM_cluster,
        get_optimal_clusters,
        perform_clustering,
        global_cluster_embeddings,
        local_cluster_embeddings,
    )
    from .tree_retriever import TreeRetriever, cosine_similarity
    from .models import (
        BaseEmbeddingModel,
        OllamaEmbeddingModel,
        SBertEmbeddingModel,
        DPREmbeddingModel,
        BM25Retriever,
        BaseSummarizationModel,
        GemmaSummarizer,
        BaseQAModel,
        GemmaQAModel,
    )
    from .raptor_system import RAPTOR
    from .eval_metrics import (
        compute_token_f1,
        compute_max_token_f1,
        compute_bleu_scores,
        compute_rouge_l,
        compute_meteor,
        evaluate_narrativeqa_metrics,
        evaluate_multiple_choice_answer,
    )
except (ImportError, ValueError):
    from tree_structures import Node, Tree
    from cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig, split_text_into_chunks
    from cluster_utils import (
        RAPTOR_Clustering,
        Recency_Clustering,
        GMM_cluster,
        get_optimal_clusters,
        perform_clustering,
        global_cluster_embeddings,
        local_cluster_embeddings,
    )
    from tree_retriever import TreeRetriever, cosine_similarity
    from models import (
        BaseEmbeddingModel,
        OllamaEmbeddingModel,
        SBertEmbeddingModel,
        DPREmbeddingModel,
        BM25Retriever,
        BaseSummarizationModel,
        GemmaSummarizer,
        BaseQAModel,
        GemmaQAModel,
    )
    from raptor_system import RAPTOR
    from eval_metrics import (
        compute_token_f1,
        compute_max_token_f1,
        compute_bleu_scores,
        compute_rouge_l,
        compute_meteor,
        evaluate_narrativeqa_metrics,
        evaluate_multiple_choice_answer,
    )

__all__ = [
    "RAPTOR",
    "Node",
    "Tree",
    "ClusterTreeBuilder",
    "ClusterTreeConfig",
    "TreeRetriever",
    "RAPTOR_Clustering",
    "Recency_Clustering",
    "OllamaEmbeddingModel",
    "SBertEmbeddingModel",
    "DPREmbeddingModel",
    "BM25Retriever",
    "GemmaSummarizer",
    "GemmaQAModel",
    "split_text_into_chunks",
    "compute_token_f1",
    "compute_max_token_f1",
    "compute_bleu_scores",
    "compute_rouge_l",
    "compute_meteor",
    "evaluate_narrativeqa_metrics",
    "evaluate_multiple_choice_answer",
]

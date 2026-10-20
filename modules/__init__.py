"""
APEX Modular RAG Pipeline Modules.
"""

from .config import (
    config,
    AppConfig,
    OllamaConfig,
    ChromaConfig,
    ChunkingConfig,
    ClusteringConfig,
    RetrievalConfig
)
from .logger import setup_logger, get_logger
from .document_loader import (
    read_pdf,
    read_pdf_pages,
    create_chunks,
    create_chunks_from_pages,
    compute_document_hash,
    count_tokens,
    split_into_sentences,
    clean_extracted_text
)
from .clustering import perform_clustering
from .summarizer import OllamaSummarizer, ClusterSummary
from .vector_store import ChromaStore
from .retrieval import (
    BM25Scorer,
    reciprocal_rank_fusion,
    deduplicate_retrieved_chunks,
    rerank_hybrid_candidates,
    jaccard_word_similarity,
    is_chitchat_query,
    query_embedding_cache,
    retrieval_result_cache
)

__all__ = [
    "config",
    "AppConfig",
    "OllamaConfig",
    "ChromaConfig",
    "ChunkingConfig",
    "ClusteringConfig",
    "RetrievalConfig",
    "setup_logger",
    "get_logger",
    "read_pdf",
    "read_pdf_pages",
    "create_chunks",
    "create_chunks_from_pages",
    "compute_document_hash",
    "count_tokens",
    "split_into_sentences",
    "clean_extracted_text",
    "perform_clustering",
    "OllamaSummarizer",
    "ClusterSummary",
    "ChromaStore",
    "BM25Scorer",
    "reciprocal_rank_fusion",
    "deduplicate_retrieved_chunks",
    "rerank_hybrid_candidates",
    "jaccard_word_similarity",
    "is_chitchat_query",
    "query_embedding_cache",
    "retrieval_result_cache"
]

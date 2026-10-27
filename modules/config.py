import os
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class OllamaConfig:
    url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    generation_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma3:1b"))
    embedding_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    timeout: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "120")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("OLLAMA_MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("OLLAMA_RETRY_DELAY", "2.0")))


@dataclass
class ChromaConfig:
    path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./hierarchical_chroma_db"))
    host: Optional[str] = field(default_factory=lambda: os.getenv("CHROMA_HOST", None))
    port: Optional[int] = field(default_factory=lambda: int(os.getenv("CHROMA_PORT")) if os.getenv("CHROMA_PORT") else None)
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHROMA_CHUNKS_COL", "document_chunks"))
    summaries_collection: str = field(default_factory=lambda: os.getenv("CHROMA_SUMMARIES_COL", "hierarchical_summaries"))
    conversation_chunks_collection: str = field(default_factory=lambda: os.getenv("CHROMA_CONV_CHUNKS_COL", "conversation_chunks"))
    conversation_summaries_collection: str = field(default_factory=lambda: os.getenv("CHROMA_CONV_SUMMARIES_COL", "conversation_summaries"))


@dataclass
class ChunkingConfig:
    target_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_TARGET_TOKENS", "300")))
    overlap_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_TOKENS", "40")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_BATCH_SIZE", "100")))


@dataclass
class ClusteringConfig:
    max_levels: int = field(default_factory=lambda: int(os.getenv("CLUSTERING_MAX_LEVELS", "3")))
    min_silhouette_threshold: float = field(default_factory=lambda: float(os.getenv("MIN_SILHOUETTE_THRESHOLD", "0.05")))
    min_cluster_silhouette: float = field(default_factory=lambda: float(os.getenv("MIN_CLUSTER_SILHOUETTE", "0.01")))
    improvement_threshold: float = field(default_factory=lambda: float(os.getenv("IMPROVEMENT_THRESHOLD", "0.05")))
    max_pca_components: int = field(default_factory=lambda: int(os.getenv("PCA_COMPONENTS", "50")))
    source_aware: bool = field(default_factory=lambda: os.getenv("CLUSTERING_SOURCE_AWARE", "true").lower() in ("true", "1", "yes"))
    enable_soft_clustering: bool = field(default_factory=lambda: os.getenv("ENABLE_SOFT_CLUSTERING", "true").lower() in ("true", "1", "yes"))
    soft_threshold: float = field(default_factory=lambda: float(os.getenv("SOFT_CLUSTER_THRESHOLD", "0.20")))
    bic_selection: bool = field(default_factory=lambda: os.getenv("CLUSTERING_BIC_SELECTION", "true").lower() in ("true", "1", "yes"))
    max_k: int = field(default_factory=lambda: int(os.getenv("CLUSTERING_MAX_K", "8")))
    min_k: int = field(default_factory=lambda: int(os.getenv("CLUSTERING_MIN_K", "2")))


@dataclass
class RetrievalConfig:
    max_context_chunks: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHUNKS", "2")))
    include_summaries: bool = field(default_factory=lambda: os.getenv("INCLUDE_SUMMARIES", "true").lower() in ("true", "1", "yes"))
    enable_collapsed_retrieval: bool = field(default_factory=lambda: os.getenv("ENABLE_COLLAPSED_RETRIEVAL", "true").lower() in ("true", "1", "yes"))
    collapsed_token_budget: int = field(default_factory=lambda: int(os.getenv("COLLAPSED_TOKEN_BUDGET", "350")))
    max_summary_candidates: int = field(default_factory=lambda: int(os.getenv("MAX_SUMMARY_CANDIDATES", "4")))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    min_relevance_score: float = field(default_factory=lambda: float(os.getenv("MIN_RELEVANCE_SCORE", "0.25")))
    min_summary_score: float = field(default_factory=lambda: float(os.getenv("MIN_SUMMARY_SCORE", "0.30")))
    include_conversation_memory: bool = field(default_factory=lambda: os.getenv("INCLUDE_CONV_MEMORY", "true").lower() in ("true", "1", "yes"))
    max_memory_chunks: int = field(default_factory=lambda: int(os.getenv("MAX_MEMORY_CHUNKS", "2")))
    dense_weight: float = field(default_factory=lambda: float(os.getenv("DENSE_WEIGHT", "0.50")))
    bm25_weight: float = field(default_factory=lambda: float(os.getenv("BM25_WEIGHT", "0.30")))
    coverage_weight: float = field(default_factory=lambda: float(os.getenv("COVERAGE_WEIGHT", "0.20")))
    enable_jaccard_dedup: bool = field(default_factory=lambda: os.getenv("ENABLE_JACCARD_DEDUP", "true").lower() in ("true", "1", "yes"))
    max_jaccard_overlap: float = field(default_factory=lambda: float(os.getenv("MAX_JACCARD_OVERLAP", "0.65")))


@dataclass
class AppConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# Global singleton instance
config = AppConfig()

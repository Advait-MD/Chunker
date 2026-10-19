"""
APEX RAG - Hybrid Retrieval, Reranking, and Deduplication Module.
Provides BM25Okapi lexical retrieval, Reciprocal Rank Fusion (RRF),
composite post-retrieval reranking, and Jaccard chunk deduplication.
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from .config import config
from .logger import get_logger

logger = get_logger("retrieval")


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase words for lexical analysis."""
    if not text:
        return []
    return [w.lower() for w in re.findall(r'\b\w+\b', text)]


class BM25Scorer:
    """Lightweight BM25Okapi scoring wrapper for candidate reranking."""
    
    def __init__(self, corpus_texts: List[str]):
        self.corpus_texts = corpus_texts
        self.tokenized_corpus = [tokenize_bm25(doc) for doc in corpus_texts]
        if self.tokenized_corpus and any(len(toks) > 0 for toks in self.tokenized_corpus):
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def score(self, query: str) -> np.ndarray:
        """Compute min-max normalized BM25 scores across candidate texts."""
        if self.bm25 is None or not self.corpus_texts:
            return np.zeros(len(self.corpus_texts))
            
        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return np.zeros(len(self.corpus_texts))
            
        raw_scores = self.bm25.get_scores(query_tokens)
        max_s = float(np.max(raw_scores)) if len(raw_scores) > 0 else 0.0
        min_s = float(np.min(raw_scores)) if len(raw_scores) > 0 else 0.0
        
        if max_s > min_s:
            return (raw_scores - min_s) / (max_s - min_s)
        elif max_s > 0:
            return np.ones(len(raw_scores)) * 0.5
        return np.zeros(len(raw_scores))


def reciprocal_rank_fusion(dense_ranks: List[int], bm25_ranks: List[int], k: int = 60) -> List[float]:
    """
    Standard Reciprocal Rank Fusion (RRF) formula:
      RRF = (1 / (k + rank_dense)) + (1 / (k + rank_bm25))
    """
    rrf_scores = []
    for r_dense, r_bm25 in zip(dense_ranks, bm25_ranks):
        score = (1.0 / (k + r_dense + 1)) + (1.0 / (k + r_bm25 + 1))
        rrf_scores.append(score)
    return rrf_scores


def jaccard_word_similarity(text_a: str, text_b: str) -> float:
    """Compute token Jaccard similarity between two texts."""
    tokens_a = set(tokenize_bm25(text_a))
    tokens_b = set(tokenize_bm25(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return intersection / union if union > 0 else 0.0


def deduplicate_retrieved_chunks(chunks: List[str], 
                                 metadatas: List[Dict], 
                                 scores: List[float], 
                                 max_jaccard_overlap: float = 0.65) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Deduplicate retrieved chunks before prompt construction (Item 27):
      1. Drops adjacent sliding-window chunks from the same document if redundant.
      2. Drops chunks with word Jaccard similarity >= max_jaccard_overlap with an accepted chunk.
    Preserves descending ranking order.
    """
    if not chunks:
        return [], [], []

    deduped_chunks = []
    deduped_metas = []
    deduped_scores = []
    seen_indices_by_source = {}

    for chunk, meta, score in zip(chunks, metadatas, scores):
        source = meta.get('source_file', 'unknown') if meta else 'unknown'
        chunk_idx = meta.get('chunk_index') if meta else None
        
        # 1. Check adjacent sliding-window overlap
        if chunk_idx is not None:
            prev_indices = seen_indices_by_source.setdefault(source, set())
            if (chunk_idx - 1 in prev_indices) or (chunk_idx + 1 in prev_indices):
                is_adjacent_duplicate = False
                for accepted_chunk in deduped_chunks:
                    if jaccard_word_similarity(chunk, accepted_chunk) > 0.40:
                        is_adjacent_duplicate = True
                        break
                if is_adjacent_duplicate:
                    continue

        # 2. Check high lexical Jaccard overlap
        too_similar = False
        for accepted_chunk in deduped_chunks:
            if jaccard_word_similarity(chunk, accepted_chunk) >= max_jaccard_overlap:
                too_similar = True
                break
        if too_similar:
            continue

        deduped_chunks.append(chunk)
        deduped_metas.append(meta)
        deduped_scores.append(score)
        if chunk_idx is not None:
            seen_indices_by_source[source].add(chunk_idx)

    return deduped_chunks, deduped_metas, deduped_scores


def compute_query_coverage(query: str, text: str) -> float:
    """Calculate the fraction of non-trivial query keywords present in document."""
    q_tokens = [w for w in tokenize_bm25(query) if len(w) > 2]
    if not q_tokens:
        return 0.5
    doc_tokens = set(tokenize_bm25(text))
    matched = sum(1 for q in q_tokens if q in doc_tokens)
    return matched / len(q_tokens)


def rerank_hybrid_candidates(query: str,
                             candidates: List[Dict],
                             dense_weight: float = 0.50,
                             bm25_weight: float = 0.30,
                             coverage_weight: float = 0.20) -> List[Dict]:
    """
    Post-Retrieval Reranking (Item 24 & Item 25):
    Scores candidate chunks using a multi-factor composite formula:
      - Dense semantic similarity (nomic-embed-text)
      - Lexical term match (BM25Okapi)
      - Non-trivial query term coverage ratio
    """
    if not candidates:
        return []

    docs = [c['document'] for c in candidates]
    bm25_scorer = BM25Scorer(docs)
    bm25_norm_scores = bm25_scorer.score(query)

    for i, c in enumerate(candidates):
        dense_score = c.get('dense_score', 0.0)
        bm25_score = float(bm25_norm_scores[i]) if i < len(bm25_norm_scores) else 0.0
        coverage_score = compute_query_coverage(query, c['document'])
        
        composite_score = (
            dense_score * dense_weight +
            bm25_score * bm25_weight +
            coverage_score * coverage_weight
        )
        
        c['bm25_score'] = bm25_score
        c['coverage_score'] = coverage_score
        c['rerank_score'] = composite_score

    # Re-sort descending by composite reranking score
    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    return candidates

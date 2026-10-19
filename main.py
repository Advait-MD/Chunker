import os
import sys
import uuid
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from modules.config import config
from modules.logger import get_logger, setup_logger

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = get_logger("main")

# Import modular pipeline components
from modules.document_loader import (
    read_pdf, 
    read_pdf_pages, 
    create_chunks, 
    create_chunks_from_pages, 
    compute_document_hash,
    count_tokens
)
from modules.clustering import perform_clustering
from modules.summarizer import OllamaSummarizer
from modules.vector_store import ChromaStore


class HierarchicalClusterSummarizer:
    """
    Coordinates document processing, ChromaDB storage, GMM clustering,
    and hierarchical cluster summarization.
    """
    def __init__(self, 
                 ollama_model: Optional[str] = None,
                 ollama_embedding_model: Optional[str] = None,
                 ollama_url: Optional[str] = None,
                 chroma_path: Optional[str] = None,
                 chroma_host: Optional[str] = None,
                 chroma_port: Optional[int] = None,
                 min_silhouette_threshold: Optional[float] = None,
                 min_cluster_silhouette: Optional[float] = None):
        
        self.ollama_model = ollama_model or config.ollama.generation_model
        self.ollama_url = ollama_url or config.ollama.url
        self.batch_size = config.chunking.batch_size
        self.min_silhouette_threshold = min_silhouette_threshold if min_silhouette_threshold is not None else config.clustering.min_silhouette_threshold
        self.min_cluster_silhouette = min_cluster_silhouette if min_cluster_silhouette is not None else config.clustering.min_cluster_silhouette
        self.clustering_history = []
        
        # Initialize modular components
        self.summarizer = OllamaSummarizer(
            ollama_model=self.ollama_model,
            ollama_url=self.ollama_url
        )
        
        self.store = ChromaStore(
            embedding_model=ollama_embedding_model or config.ollama.embedding_model,
            ollama_url=self.ollama_url,
            chroma_path=chroma_path or config.chroma.path,
            chroma_host=chroma_host or config.chroma.host,
            chroma_port=chroma_port or config.chroma.port
        )
        
        # Backward-compatible collection references for face.py and legacy callers
        self.client = self.store.client
        self.embedding_fn = self.store.embedding_fn
        self.chunks_collection = self.store.chunks_collection
        self.summaries_collection = self.store.summaries_collection

    def get_indexed_sources(self) -> List[str]:
        """Return list of distinct source document filenames indexed in ChromaDB."""
        return self.store.get_indexed_sources()

    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        return read_pdf(file_path)

    def read_pdf_pages(self, file_path: str) -> List[Dict]:
        """Extract text page-by-page from PDF file with citation metadata."""
        return read_pdf_pages(file_path)

    def create_chunks(self, text_or_pages, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Split text or pages into chunks."""
        if isinstance(text_or_pages, list):
            target_tokens = max(50, chunk_size // 4)
            overlap_tokens = max(10, overlap // 4)
            return create_chunks_from_pages(text_or_pages, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        return create_chunks(text_or_pages, chunk_size=chunk_size, overlap=overlap)

    def add_document(self, file_path: str, document_name: str = None, force: bool = False,
                     target_tokens: int = None, overlap_tokens: int = None) -> int:
        """Read PDF, perform page-aware & token-aware chunking, and add chunks to vector store with citation metadata."""
        if document_name is None:
            document_name = os.path.basename(file_path)
        
        target_tokens = target_tokens or config.chunking.target_tokens
        overlap_tokens = overlap_tokens or config.chunking.overlap_tokens
        doc_id = compute_document_hash(file_path)
        
        # Early check: skip expensive PDF extraction and chunking if already in database
        if not force:
            try:
                existing_doc = self.chunks_collection.get(where={"doc_id": doc_id}, limit=1)
                if existing_doc and existing_doc.get('ids') and len(existing_doc['ids']) > 0:
                    logger.info(f"Document '{document_name}' [doc_id: {doc_id}] already exists in database. Skipping duplicate ingestion.")
                    return 0
                existing_src = self.chunks_collection.get(where={"source_file": document_name}, limit=1)
                if existing_src and existing_src.get('ids') and len(existing_src['ids']) > 0:
                    logger.info(f"Document '{document_name}' already exists in database. Skipping duplicate ingestion.")
                    return 0
            except Exception as e:
                logger.warning(f"Warning during pre-check: {e}")
        
        pages = read_pdf_pages(file_path)
        chunks = create_chunks_from_pages(pages, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        return self.store.add_chunks(document_name, chunks, doc_id=doc_id, force=force, batch_size=self.batch_size)

    def get_items_by_level(self, level: int = 0, source_file: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[Dict]], Optional[List[str]]]:
        """Get all items (embeddings, documents, metadata, ids) for a specific level."""
        return self.store.get_items_by_level(level=level, source_file=source_file)

    def perform_clustering(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> Optional[Dict]:
        """Perform GMM clustering on embeddings."""
        return perform_clustering(embeddings, n_clusters=n_clusters)

    def generate_summary_with_ollama(self, 
                                     texts: List[str], 
                                     max_chunks: int = 15,
                                     source_doc: Optional[str] = None,
                                     page_range: Optional[Tuple[int, int]] = None):
        """Generate dual full & dense summary using Ollama local LLM with source attribution constraints."""
        return self.summarizer.generate_summary(texts, max_chunks=max_chunks, source_doc=source_doc, page_range=page_range)

    def _generate_fallback_summary(self, texts: List[str], source_doc: Optional[str] = None):
        """Generate heuristic fallback summary."""
        return self.summarizer.generate_fallback_summary(texts, source_doc=source_doc)

    def store_cluster_summaries(self, clustering_results: Dict, 
                                documents: List[str],
                                metadatas: List[Dict],
                                level: int,
                                min_cluster_silhouette: Optional[float] = None,
                                skip_bad_clusters: bool = True,
                                source_doc: Optional[str] = None) -> Dict:
        """Store summaries for each cluster at a given level into hierarchical_summaries with silhouette quality gate and source attribution."""
        labels = clustering_results['labels']
        n_clusters = clustering_results['n_clusters']
        overall_silhouette = clustering_results.get('silhouette_score', 0.0)
        cluster_silhouettes = clustering_results.get('cluster_silhouettes', {})
        
        min_sil = self.min_cluster_silhouette if min_cluster_silhouette is None else min_cluster_silhouette
        cluster_summaries = {}
        
        source_label = f" for '{source_doc}'" if source_doc else ""
        logger.info(f"Evaluating and storing summaries for Level {level}{source_label} (Overall Silhouette: {overall_silhouette:.3f}, Gate Threshold: {min_sil:.3f})")
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_texts = [documents[i] for i in cluster_indices]
            cluster_metas = [metadatas[i] for i in cluster_indices]
            cluster_sil = cluster_silhouettes.get(cluster_id, overall_silhouette)
            
            logger.info(f"Cluster {cluster_id} at Level {level}: {len(cluster_texts)} chunks | Cluster Silhouette: {cluster_sil:.3f}")
            
            # Quality Gate: stop summarizing bad clusters (silhouette near zero or negative)
            is_bad_cluster = (cluster_sil <= min_sil)
            
            if is_bad_cluster and skip_bad_clusters:
                logger.warning(f"Quality Gate: Cluster {cluster_id} silhouette ({cluster_sil:.3f} <= {min_sil:.3f}) is near-zero or negative. Skipping LLM summary.")
                continue
            
            # Determine sources and page range for attribution constraint (Item 22)
            sources = list(set([m.get('source_file', 'unknown') for m in cluster_metas if m.get('source_file')]))
            effective_source = source_doc or (sources[0] if len(sources) == 1 else None)
            
            pages = [m.get('page_number') for m in cluster_metas if m.get('page_number') is not None]
            page_range = (min(pages), max(pages)) if pages else None
            
            logger.info(f"Generating summary for Cluster {cluster_id} ({effective_source or 'multi-doc'}) with {self.ollama_model}...")
            summary = self.generate_summary_with_ollama(
                cluster_texts,
                source_doc=effective_source,
                page_range=page_range
            )
            dense_summary = getattr(summary, 'dense', '') or str(summary)[:200]
            
            summary_id = f"level_{level}_cluster_{cluster_id}_{uuid.uuid4().hex[:8]}"
            avg_confidence = float(np.mean([max(clustering_results['probabilities'][i]) 
                                           for i in cluster_indices]))
            
            summary_metadata = {
                'type': 'cluster_summary',
                'level': level,
                'cluster_id': cluster_id,
                'chunks_count': len(cluster_texts),
                'source_file': effective_source or 'multiple',
                'source_files': [effective_source] if effective_source else sources,
                'cluster_type': 'intra_document' if effective_source else 'cross_document',
                'avg_confidence': avg_confidence,
                'silhouette_score': cluster_sil,
                'overall_silhouette': overall_silhouette,
                'low_confidence': is_bad_cluster,
                'dense_summary': dense_summary,
                'page_range': f"{page_range[0]}-{page_range[1]}" if page_range else "",
                'created_at': pd.Timestamp.now().isoformat(),
                'summary_preview': str(summary)[:100]
            }
            
            try:
                self.store.save_summary_record(
                    summary_id=summary_id,
                    summary=str(summary),
                    summary_metadata=summary_metadata
                )
                
                cluster_summaries[cluster_id] = {
                    'summary': str(summary),
                    'dense_summary': dense_summary,
                    'summary_id': summary_id,
                    'chunks_count': len(cluster_texts),
                    'sources': [effective_source] if effective_source else sources,
                    'source_file': effective_source or 'multiple',
                    'silhouette_score': cluster_sil,
                    'low_confidence': is_bad_cluster
                }
                logger.info(f"✓ Summary for Cluster {cluster_id} stored in {self.store.summaries_col_name}")
                
            except Exception as e:
                logger.error(f"Error storing summary for cluster {cluster_id}: {e}")
        
        return cluster_summaries

    def hierarchical_clustering(self, max_levels: int = None, 
                                improvement_threshold: float = None,
                                skip_bad_clusters: bool = True,
                                source_aware: Optional[bool] = None) -> Dict:
        """
        Perform hierarchical clustering recursively across levels with silhouette quality gating.
        When source_aware=True (Item 19 & 20):
          - Level 0 clusters chunks within each source document independently (prevents narrative bleeding).
          - Level 1+ clusters the resulting document-level summaries for macro cross-document synthesis.
        """
        max_levels = max_levels or config.clustering.max_levels
        improvement_threshold = improvement_threshold or config.clustering.improvement_threshold
        if source_aware is None:
            source_aware = config.clustering.source_aware
        
        logger.info(f"Starting hierarchical clustering (max_levels={max_levels}, source_aware={source_aware})")
        
        history = {
            'levels': [],
            'convergence': False,
            'final_level': 0
        }
        
        # --- LEVEL 0: INTRA-DOCUMENT CLUSTERING (Source-Aware) ---
        if source_aware:
            indexed_sources = self.store.get_indexed_sources()
            logger.info(f"Source-aware clustering across {len(indexed_sources)} documents: {indexed_sources}")
            
            all_level_0_summaries = {}
            total_items_level_0 = 0
            
            for src in indexed_sources:
                logger.info(f"--- LEVEL 0 INTRA-DOCUMENT CLUSTERING: '{src}' ---")
                embeddings, documents, metadatas, ids = self.store.get_items_by_level(level=0, source_file=src)
                
                if embeddings is None or len(embeddings) < 3:
                    logger.warning(f"Skipping intra-document clustering for '{src}' (insufficient chunks: {len(embeddings) if embeddings is not None else 0})")
                    continue
                
                total_items_level_0 += len(embeddings)
                n_clusters = max(2, min(6, len(embeddings) // 40))
                clustering_results = self.perform_clustering(embeddings, n_clusters)
                
                if clustering_results is None:
                    continue
                
                summaries = self.store_cluster_summaries(
                    clustering_results,
                    documents,
                    metadatas,
                    level=0,
                    min_cluster_silhouette=self.min_cluster_silhouette,
                    skip_bad_clusters=skip_bad_clusters,
                    source_doc=src
                )
                
                for cid, s_info in summaries.items():
                    all_level_0_summaries[f"{src}_c{cid}"] = s_info
            
            history['levels'].append({
                'level': 0,
                'source_aware': True,
                'sources_processed': len(indexed_sources),
                'n_items': total_items_level_0,
                'summaries': all_level_0_summaries,
                'summaries_stored': len(all_level_0_summaries)
            })
            
            current_level = 1
        else:
            current_level = 0
        
        # --- LEVEL 1+ (MACRO / CROSS-DOCUMENT CLUSTERING) ---
        previous_silhouette = 0
        while current_level < max_levels:
            logger.info(f"--- LEVEL {current_level} CLUSTERING ---")
            
            embeddings, documents, metadatas, ids = self.store.get_items_by_level(level=current_level)
            
            if embeddings is None or len(embeddings) < 3:
                logger.warning(f"No/insufficient items found at level {current_level} (found: {len(embeddings) if embeddings is not None else 0})")
                break
            
            logger.info(f"Found {len(embeddings)} items at level {current_level}")
            
            n_clusters = max(2, min(8, len(embeddings) // 4))
            clustering_results = self.perform_clustering(embeddings, n_clusters)
            
            if clustering_results is None:
                logger.error("Clustering failed")
                break
            
            current_silhouette = clustering_results['silhouette_score']
            cluster_summaries = self.store_cluster_summaries(
                clustering_results,
                documents,
                metadatas,
                current_level,
                min_cluster_silhouette=self.min_cluster_silhouette,
                skip_bad_clusters=skip_bad_clusters
            )
            
            history['levels'].append({
                'level': current_level,
                'n_clusters': clustering_results['n_clusters'],
                'silhouette_score': clustering_results['silhouette_score'],
                'n_items': len(embeddings),
                'summaries': cluster_summaries,
                'summaries_stored': len(cluster_summaries)
            })
            
            if len(cluster_summaries) == 0 and skip_bad_clusters:
                logger.warning(f"Quality Gate: No clusters met the minimum silhouette threshold at Level {current_level}. Halting hierarchy recursion.")
                history['final_level'] = current_level
                break
            
            current_level += 1
        
        history['final_level'] = current_level - 1 if current_level > 0 else 0
        return history

    def get_summary_tree(self) -> pd.DataFrame:
        """Get all summaries organized as a tree dataframe."""
        return self.store.get_summary_tree()

    def visualize_hierarchy(self) -> None:
        """Print hierarchical structure of summaries."""
        df = self.get_summary_tree()
        
        if df.empty:
            logger.info("No summaries found in hierarchy")
            return
        
        print("\n" + "="*80)
        print("HIERARCHICAL SUMMARY STRUCTURE")
        print("="*80)
        
        for level in sorted(df['level'].unique()):
            level_df = df[df['level'] == level]
            print(f"\n📚 Level {level} ({len(level_df)} clusters):")
            print("-" * 60)
            
            for _, row in level_df.iterrows():
                print(f"  Cluster {row['cluster_id']}:")
                print(f"    • {row['chunks_count']} chunks")
                print(f"    • Silhouette: {row['silhouette_score']:.3f}")
                print(f"    • Preview: {row['summary_preview'][:80]}...")
                print()


# ========== MAIN EXECUTION ==========

def main():
    parser = argparse.ArgumentParser(description="APEX RAG - Hierarchical Clustering Ingestion & Orchestration")
    parser.add_argument("--pdf", nargs="+", default=["Steve Jobs (1).pdf"], help="PDF file(s) to ingest")
    parser.add_argument("--max-levels", type=int, default=config.clustering.max_levels, help="Max hierarchy levels")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion of documents")
    parser.add_argument("--log-level", default=config.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--source-aware", action=argparse.BooleanOptionalAction, default=True, help="Perform intra-document clustering first (prevents narrative bleeding)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported CSV reports")
    args = parser.parse_args()

    setup_logger(level=args.log_level)
    
    processor = HierarchicalClusterSummarizer()
    
    # Preflight Ollama check
    is_ready, msg = processor.summarizer.check_health([processor.ollama_model, processor.store.embedding_model])
    if not is_ready:
        logger.error(f"Ollama preflight check failed: {msg}")
        logger.error("Make sure Ollama is running ('ollama serve') and required models are pulled:")
        logger.error(f"  ollama pull {processor.ollama_model}")
        logger.error(f"  ollama pull {processor.store.embedding_model}")
        return
    logger.info("✓ Ollama and required models are healthy")
    
    for file_path in args.pdf:
        if os.path.exists(file_path):
            logger.info(f"Processing PDF: {file_path}")
            processor.add_document(file_path, force=args.force)
        else:
            logger.warning(f"File not found: {file_path}")
    
    total_chunks = processor.store.count_chunks()
    if total_chunks == 0:
        logger.warning("No documents found in database. Ingest PDF files first.")
        return
    
    logger.info(f"Total chunks in database: {total_chunks}")
    
    history = processor.hierarchical_clustering(
        max_levels=args.max_levels,
        source_aware=args.source_aware
    )
    
    processor.visualize_hierarchy()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    summaries_df = processor.get_summary_tree()
    if not summaries_df.empty:
        output_file = os.path.join(args.output_dir, "hierarchical_summaries.csv")
        summaries_df.to_csv(output_file, index=False)
        logger.info(f"Summaries exported to: {output_file}")
    
    history_df = pd.DataFrame(history['levels'])
    if not history_df.empty:
        history_file = os.path.join(args.output_dir, "clustering_history.csv")
        history_df.to_csv(history_file, index=False)
        logger.info(f"Clustering history exported to: {history_file}")


if __name__ == "__main__":
    main()

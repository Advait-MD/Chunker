import os
import sys
import uuid
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import separated modules
from document_loader import read_pdf, create_chunks, compute_document_hash
from clustering import perform_clustering
from summarizer import OllamaSummarizer
from vector_store import ChromaStore


class HierarchicalClusterSummarizer:
    """
    Coordinates document processing, ChromaDB storage, GMM clustering,
    and hierarchical cluster summarization.
    """
    def __init__(self, 
                 ollama_model: str = "gemma3:1b",
                 ollama_embedding_model: str = "nomic-embed-text",
                 ollama_url: str = "http://localhost:11434",
                 chroma_path: Optional[str] = "./hierarchical_chroma_db",
                 chroma_host: Optional[str] = None,
                 chroma_port: Optional[int] = None):
        
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.batch_size = 100
        self.clustering_history = []
        
        # Initialize modular components
        self.summarizer = OllamaSummarizer(
            ollama_model=ollama_model,
            ollama_url=ollama_url
        )
        
        self.store = ChromaStore(
            embedding_model=ollama_embedding_model,
            ollama_url=ollama_url,
            chroma_path=chroma_path,
            chroma_host=chroma_host,
            chroma_port=chroma_port
        )
        
        # Backward-compatible collection references for face.py and legacy callers
        self.client = self.store.client
        self.embedding_fn = self.store.embedding_fn
        self.chunks_collection = self.store.chunks_collection
        self.summaries_collection = self.store.summaries_collection

    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        return read_pdf(file_path)

    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Split text into overlapping chunks."""
        return create_chunks(text, chunk_size=chunk_size, overlap=overlap)

    def add_document(self, file_path: str, document_name: str = None, force: bool = False) -> int:
        """Read PDF, chunk it, and add chunks to vector store with duplicate-ingestion protection."""
        if document_name is None:
            document_name = os.path.basename(file_path)
        
        doc_id = compute_document_hash(file_path)
        
        # Early check: skip expensive PDF extraction and chunking if already in database
        if not force:
            try:
                existing_doc = self.chunks_collection.get(where={"doc_id": doc_id}, limit=1)
                if existing_doc and existing_doc.get('ids') and len(existing_doc['ids']) > 0:
                    print(f"⏩ Document '{document_name}' [doc_id: {doc_id}] already exists in database. Skipping duplicate ingestion.")
                    return 0
                existing_src = self.chunks_collection.get(where={"source_file": document_name}, limit=1)
                if existing_src and existing_src.get('ids') and len(existing_src['ids']) > 0:
                    print(f"⏩ Document '{document_name}' already exists in database. Skipping duplicate ingestion.")
                    return 0
            except Exception as e:
                print(f"  Warning during pre-check: {e}")
        
        text = self.read_pdf(file_path)
        chunks = self.create_chunks(text)
        return self.store.add_chunks(document_name, chunks, doc_id=doc_id, force=force, batch_size=self.batch_size)

    def get_items_by_level(self, level: int = 0) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[Dict]], Optional[List[str]]]:
        """Retrieve items for a specific level from vector store."""
        return self.store.get_items_by_level(level=level)

    def perform_clustering(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> Optional[Dict]:
        """Perform GMM clustering on vector embeddings."""
        return perform_clustering(embeddings, n_clusters=n_clusters)

    def generate_summary_with_ollama(self, texts: List[str], max_chunks: int = 15) -> str:
        """Generate cluster summary using Ollama LLM."""
        return self.summarizer.generate_summary(texts, max_chunks=max_chunks)

    def _generate_fallback_summary(self, texts: List[str]) -> str:
        """Generate heuristic fallback summary."""
        return self.summarizer.generate_fallback_summary(texts)

    def store_cluster_summaries(self, clustering_results: Dict, 
                                documents: List[str],
                                metadatas: List[Dict],
                                level: int) -> Dict:
        """Store summaries for each cluster at a given level into hierarchical_summaries."""
        labels = clustering_results['labels']
        n_clusters = clustering_results['n_clusters']
        
        cluster_summaries = {}
        
        print(f"\n{'='*80}")
        print(f"STORING SUMMARIES FOR LEVEL {level}")
        print(f"{'='*80}")
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_texts = [documents[i] for i in cluster_indices]
            cluster_metas = [metadatas[i] for i in cluster_indices]
            
            print(f"\n📊 Cluster {cluster_id} at Level {level}:")
            print(f"   Chunks: {len(cluster_texts)}")
            
            print(f"   Generating summary with {self.ollama_model}...")
            summary = self.generate_summary_with_ollama(cluster_texts)
            
            summary_id = f"level_{level}_cluster_{cluster_id}_{uuid.uuid4().hex[:8]}"
            sources = list(set([m.get('source_file', 'unknown') for m in cluster_metas]))
            
            avg_confidence = float(np.mean([max(clustering_results['probabilities'][i]) 
                                           for i in cluster_indices]))
            
            summary_metadata = {
                'type': 'cluster_summary',
                'level': level,
                'cluster_id': cluster_id,
                'chunks_count': len(cluster_texts),
                'source_files': sources,
                'avg_confidence': avg_confidence,
                'silhouette_score': clustering_results.get('silhouette_score', 0),
                'created_at': pd.Timestamp.now().isoformat(),
                'summary_preview': summary[:100]
            }
            
            try:
                self.store.save_summary_record(
                    summary_id=summary_id,
                    summary=summary,
                    summary_metadata=summary_metadata
                )
                
                cluster_summaries[cluster_id] = {
                    'summary': summary,
                    'summary_id': summary_id,
                    'chunks_count': len(cluster_texts),
                    'sources': sources
                }
                
                print(f"   ✓ Summary stored in hierarchical_summaries")
                print(f"   Summary preview: {summary[:150]}...")
                
            except Exception as e:
                print(f"   ✗ Error storing summary: {e}")
        
        return cluster_summaries

    def hierarchical_clustering(self, max_levels: int = 3, 
                                improvement_threshold: float = 0.05) -> Dict:
        """Perform hierarchical clustering recursively across levels."""
        print("\n" + "="*80)
        print("STARTING HIERARCHICAL CLUSTERING")
        print("="*80)
        
        history = {
            'levels': [],
            'convergence': False,
            'final_level': 0
        }
        
        current_level = 0
        previous_silhouette = 0
        
        while current_level < max_levels:
            print(f"\n{'='*80}")
            print(f"LEVEL {current_level} CLUSTERING")
            print(f"{'='*80}")
            
            embeddings, documents, metadatas, ids = self.get_items_by_level(level=current_level)
            
            if embeddings is None or len(embeddings) == 0:
                print(f"No items found at level {current_level}")
                break
            
            print(f"Found {len(embeddings)} items at level {current_level}")
            
            n_clusters = max(2, min(8, len(embeddings) // 50))
            clustering_results = self.perform_clustering(embeddings, n_clusters)
            
            if clustering_results is None:
                print("Clustering failed")
                break
            
            cluster_summaries = self.store_cluster_summaries(
                clustering_results,
                documents,
                metadatas,
                current_level
            )
            
            history['levels'].append({
                'level': current_level,
                'n_clusters': clustering_results['n_clusters'],
                'silhouette_score': clustering_results['silhouette_score'],
                'n_items': len(embeddings),
                'summaries': cluster_summaries
            })
            
            current_silhouette = clustering_results['silhouette_score']
            improvement = current_silhouette - previous_silhouette if current_level > 0 else 1
            
            print(f"\n📊 Level {current_level} Summary:")
            print(f"   Silhouette Score: {current_silhouette:.3f}")
            print(f"   Improvement: {improvement:.3f}")
            print(f"   Number of clusters: {clustering_results['n_clusters']}")
            
            if current_level > 0 and improvement < improvement_threshold:
                print(f"\n✅ Convergence reached at level {current_level}")
                history['convergence'] = True
                history['final_level'] = current_level
                break
            
            if clustering_results['n_clusters'] <= 1:
                print(f"\n✅ Only one cluster formed at level {current_level}")
                history['convergence'] = True
                history['final_level'] = current_level
                break
            
            previous_silhouette = current_silhouette
            current_level += 1
        
        if current_level >= max_levels:
            print(f"\n⚠️ Reached maximum levels ({max_levels})")
            history['final_level'] = max_levels - 1
        
        return history

    def get_summary_tree(self) -> pd.DataFrame:
        """Get all summaries organized as a tree dataframe."""
        return self.store.get_summary_tree()

    def visualize_hierarchy(self) -> None:
        """Print hierarchical structure of summaries."""
        df = self.get_summary_tree()
        
        if df.empty:
            print("No summaries found")
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
    processor = HierarchicalClusterSummarizer(
        ollama_model="gemma3:1b",
        ollama_embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
        chroma_path="./hierarchical_chroma_db"
    )
    
    # Preflight Ollama check
    is_ready, msg = processor.summarizer.check_health(["gemma3:1b", "nomic-embed-text"])
    if not is_ready:
        print(f"⚠️ Ollama check failed: {msg}")
        print("   Make sure Ollama is running ('ollama serve') and models are pulled:")
        print("   ollama pull gemma3:1b")
        print("   ollama pull nomic-embed-text")
        return
    print("✓ Ollama and required models are available")
    
    files_to_add = [
        "Steve Jobs (1).pdf"
    ]
    
    for file_path in files_to_add:
        if os.path.exists(file_path):
            print(f"\n📖 Processing: {file_path}")
            processor.add_document(file_path)
        else:
            print(f"\n⚠️ File not found: {file_path}")
    
    total_chunks = processor.store.count_chunks()
    if total_chunks == 0:
        print("\n⚠️ No documents found. Please add PDF files to process.")
        return
    
    print(f"\n📊 Total chunks in database: {total_chunks}")
    print("   (Each chunk has an auto-generated embedding from nomic-embed-text)")
    
    history = processor.hierarchical_clustering(
        max_levels=3,
        improvement_threshold=0.05
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\n📈 Clustering History:")
    for level_info in history['levels']:
        print(f"  Level {level_info['level']}: {level_info['n_clusters']} clusters, "
              f"Silhouette: {level_info['silhouette_score']:.3f}, "
              f"Items: {level_info['n_items']}")
    
    print(f"\n✅ Convergence reached: {history['convergence']}")
    print(f"📊 Final level: {history['final_level']}")
    
    processor.visualize_hierarchy()
    
    summaries_df = processor.get_summary_tree()
    if not summaries_df.empty:
        output_file = "hierarchical_summaries.csv"
        summaries_df.to_csv(output_file, index=False)
        print(f"\n💾 Summaries exported to: {output_file}")
    
    history_df = pd.DataFrame(history['levels'])
    if not history_df.empty:
        history_file = "clustering_history.csv"
        history_df.to_csv(history_file, index=False)
        print(f"📊 Clustering history exported to: {history_file}")


if __name__ == "__main__":
    main()

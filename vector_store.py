import os
import sys
import uuid
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class ChromaStore:
    """Manages local or remote ChromaDB vector store collections for chunks and hierarchical summaries."""
    
    def __init__(self,
                 embedding_model: str = "nomic-embed-text",
                 ollama_url: str = "http://localhost:11434",
                 chroma_path: Optional[str] = "./hierarchical_chroma_db",
                 chroma_host: Optional[str] = None,
                 chroma_port: Optional[int] = None):
        
        # Configure embedding function
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name=embedding_model,
            url=f"{ollama_url.rstrip('/')}/api/embeddings"
        )
        
        # Initialize client (local embedded by default)
        if chroma_host and chroma_port:
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            print(f"✓ Connected to remote ChromaDB at {chroma_host}:{chroma_port}")
        else:
            db_path = chroma_path or "./hierarchical_chroma_db"
            self.client = chromadb.PersistentClient(path=db_path)
            print(f"✓ Using local ChromaDB at: {os.path.abspath(db_path)}")
        
        # Initialize collections
        self._init_collections()

    def _init_collections(self):
        """Get or initialize document_chunks and hierarchical_summaries collections."""
        existing_collections = [col.name for col in self.client.list_collections()]
        print(f"Existing collections: {existing_collections}")
        
        # 1. document_chunks collection
        if "document_chunks" in existing_collections:
            try:
                self.chunks_collection = self.client.get_collection("document_chunks")
                print("✓ Using existing document_chunks collection")
            except ValueError as e:
                if "embedding function conflict" in str(e):
                    print("⚠️ Embedding function conflict detected. Recreating document_chunks collection...")
                    self.client.delete_collection("document_chunks")
                    self.chunks_collection = self.client.create_collection(
                        name="document_chunks",
                        embedding_function=self.embedding_fn
                    )
                    print("✓ Created new document_chunks collection with embedding function")
                else:
                    raise
        else:
            self.chunks_collection = self.client.create_collection(
                name="document_chunks",
                embedding_function=self.embedding_fn
            )
            print("✓ Created new document_chunks collection with embedding function")
        
        # 2. hierarchical_summaries collection
        if "hierarchical_summaries" in existing_collections:
            try:
                self.summaries_collection = self.client.get_collection("hierarchical_summaries")
                print("✓ Using existing hierarchical_summaries collection")
            except ValueError as e:
                if "embedding function conflict" in str(e):
                    print("⚠️ Embedding function conflict detected. Recreating hierarchical_summaries collection...")
                    self.client.delete_collection("hierarchical_summaries")
                    self.summaries_collection = self.client.create_collection(
                        name="hierarchical_summaries",
                        embedding_function=self.embedding_fn
                    )
                    print("✓ Created new hierarchical_summaries collection with embedding function")
                else:
                    raise
        else:
            self.summaries_collection = self.client.create_collection(
                name="hierarchical_summaries",
                embedding_function=self.embedding_fn
            )
            print("✓ Created new hierarchical_summaries collection with embedding function")

    def count_chunks(self) -> int:
        """Return total number of chunks in database."""
        return self.chunks_collection.count()

    def count_summaries(self) -> int:
        """Return total number of summaries in database."""
        return self.summaries_collection.count()

    def add_chunks(self, document_name: str, chunks: List[Dict], doc_id: Optional[str] = None, force: bool = False, batch_size: int = 100) -> int:
        """
        Add chunks to ChromaDB document_chunks collection with duplicate-ingestion protection.
        
        Args:
            document_name: Human-readable name or filename of the document.
            chunks: List of chunk dictionaries containing 'text', 'chunk_id', 'chunk_index'.
            doc_id: Deterministic hash or ID of the document (prevents duplicate ingestion).
            force: If True, overwrite/re-ingest existing chunks for this document.
            batch_size: Batch size for ChromaDB insertion.
        """
        if not chunks:
            print("No chunks provided")
            return 0
        
        # 1. Duplicate-ingestion check
        if not force:
            if doc_id:
                try:
                    existing = self.chunks_collection.get(where={"doc_id": doc_id}, limit=1)
                    if existing and existing.get('ids') and len(existing['ids']) > 0:
                        print(f"⏩ Document '{document_name}' [doc_id: {doc_id}] already exists in database. Skipping duplicate ingestion.")
                        return 0
                except Exception as e:
                    print(f"  Warning checking doc_id: {e}")
            
            if document_name:
                try:
                    existing = self.chunks_collection.get(where={"source_file": document_name}, limit=1)
                    if existing and existing.get('ids') and len(existing['ids']) > 0:
                        print(f"⏩ Document '{document_name}' already exists in database. Skipping duplicate ingestion.")
                        return 0
                except Exception as e:
                    print(f"  Warning checking source_file: {e}")
        else:
            # Force mode: purge prior chunks for this document to avoid orphans
            try:
                if doc_id:
                    self.chunks_collection.delete(where={"doc_id": doc_id})
                    print(f"🗑️ Purged prior chunks matching doc_id: {doc_id}")
                elif document_name:
                    self.chunks_collection.delete(where={"source_file": document_name})
                    print(f"🗑️ Purged prior chunks matching source_file: {document_name}")
            except Exception as e:
                print(f"  Note during forced cleanup: {e}")
        
        print(f"\nAdding {len(chunks)} chunks to ChromaDB (document_chunks)...")
        print("(ChromaDB will automatically generate embeddings using nomic-embed-text)")
        
        clean_prefix = doc_id if doc_id else os.path.splitext(document_name)[0].replace(' ', '_')
        all_ids = []
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            
            for chunk in batch_chunks:
                # Deterministic chunk ID based on document hash/name + chunk identifier
                chroma_id = f"{clean_prefix}_{chunk['chunk_id']}"
                batch_ids.append(chroma_id)
                batch_documents.append(chunk['text'])
                
                metadata = {
                    'doc_id': doc_id or clean_prefix,
                    'source_file': document_name,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk.get('chunk_index', 0),
                    'level': 0,
                    'chunk_preview': chunk['text'][:50].replace('\n', ' '),
                    'added_timestamp': pd.Timestamp.now().isoformat()
                }
                batch_metadatas.append(metadata)
            
            try:
                self.chunks_collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                all_ids.extend(batch_ids)
                print(f"  Upserted batch {batch_start//batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"Successfully upserted {len(all_ids)} chunks (embeddings generated automatically)")
        return len(all_ids)

    def get_items_by_level(self, level: int = 0) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[Dict]], Optional[List[str]]]:
        """
        Get all items (embeddings, documents, metadata, ids) for a specific level.
        - Level 0 retrieves raw source chunks from document_chunks.
        - Level > 0 retrieves generated cluster summaries from hierarchical_summaries where level == level - 1.
        """
        if level == 0:
            target_col = self.chunks_collection
            filter_dict = None  # All items in chunks_collection are Level 0 source text
            desc = "document_chunks (Level 0 source text)"
        else:
            target_col = self.summaries_collection
            filter_dict = {"level": level - 1}
            desc = f"hierarchical_summaries (Level {level - 1} summaries for Level {level} clustering)"
        
        total_count = target_col.count()
        print(f"Retrieving items from {desc}. Total collection items: {total_count}")
        
        all_embeddings = []
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        batch_size = 1000
        for offset in range(0, max(1, total_count), batch_size):
            try:
                kwargs = {
                    "limit": batch_size,
                    "offset": offset,
                    "include": ["embeddings", "documents", "metadatas"]
                }
                if filter_dict:
                    kwargs["where"] = filter_dict
                
                result = target_col.get(**kwargs)
                
                if not result or not result.get('ids'):
                    break
                
                ids = result['ids']
                docs = result.get('documents', [])
                metas = result.get('metadatas', [])
                embs = result.get('embeddings')
                
                for i in range(len(ids)):
                    all_ids.append(ids[i])
                    all_documents.append(docs[i] if i < len(docs) else "")
                    all_metadatas.append(metas[i] if i < len(metas) else {})
                    if embs is not None and len(embs) > i and embs[i] is not None:
                        all_embeddings.append(embs[i])
                    else:
                        all_embeddings.append(np.zeros(768))
                
                if len(ids) < batch_size:
                    break
            except Exception as e:
                print(f"  Error retrieving batch at offset {offset}: {e}")
                break
        
        if len(all_embeddings) == 0:
            print(f"No items found for clustering at level {level}")
            return None, None, None, None
        
        print(f"Found {len(all_embeddings)} items for level {level}")
        return np.array(all_embeddings), all_documents, all_metadatas, all_ids

    def save_summary_record(self, summary_id: str, summary: str, summary_metadata: Dict,
                            summary_chunk_id: Optional[str] = None, chunk_metadata: Optional[Dict] = None):
        """
        Save a cluster summary to summaries_collection ONLY.
        Strictly avoids injecting synthetic summaries back into document_chunks.
        """
        self.summaries_collection.upsert(
            ids=[summary_id],
            documents=[summary],
            metadatas=[summary_metadata]
        )

    def get_summary_tree(self) -> pd.DataFrame:
        """Get all summaries organized as a tree dataframe."""
        result = self.summaries_collection.get(
            include=["documents", "metadatas"]
        )
        
        if not result['ids']:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'summary_id': result['ids'],
            'level': [m.get('level', -1) for m in result['metadatas']],
            'cluster_id': [m.get('cluster_id', -1) for m in result['metadatas']],
            'chunks_count': [m.get('chunks_count', 0) for m in result['metadatas']],
            'silhouette_score': [m.get('silhouette_score', 0) for m in result['metadatas']],
            'summary_preview': [doc[:100] for doc in result['documents']],
            'created_at': [m.get('created_at', '') for m in result['metadatas']]
        })
        
        return df.sort_values(['level', 'cluster_id'])

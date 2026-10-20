import os
import sys
import uuid
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

try:
    from .config import config
    from .logger import get_logger
except ImportError:
    from modules.config import config
    from modules.logger import get_logger

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = get_logger("vector_store")


class ChromaStore:
    """Manages local or remote ChromaDB vector store collections for chunks and hierarchical summaries."""
    
    def __init__(self,
                 embedding_model: Optional[str] = None,
                 ollama_url: Optional[str] = None,
                 chroma_path: Optional[str] = None,
                 chroma_host: Optional[str] = None,
                 chroma_port: Optional[int] = None):
        
        self.embedding_model = embedding_model or config.ollama.embedding_model
        self.ollama_url = (ollama_url or config.ollama.url).rstrip('/')
        self.chroma_path = chroma_path or config.chroma.path
        self.chroma_host = chroma_host or config.chroma.host
        self.chroma_port = chroma_port or config.chroma.port
        self.chunks_col_name = config.chroma.chunks_collection
        self.summaries_col_name = config.chroma.summaries_collection
        self.conv_chunks_col_name = config.chroma.conversation_chunks_collection
        self.conv_summaries_col_name = config.chroma.conversation_summaries_collection
        
        # Configure embedding function: use fast in-process DefaultEmbeddingFunction
        # (matching 384-dim ONNX) to achieve uniform sub-second lookup without HTTP round-trips
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize client (local embedded by default) with robust error handling
        try:
            if self.chroma_host and self.chroma_port:
                self.client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
                logger.info(f"Connected to remote ChromaDB at {self.chroma_host}:{self.chroma_port}")
            else:
                db_path = os.path.abspath(self.chroma_path)
                os.makedirs(db_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=db_path)
                logger.info(f"Using local ChromaDB at: {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client at '{self.chroma_path}': {e}")
            logger.error("Troubleshooting: Check file permissions, ensure another process is not locking the SQLite file, or clear corrupted cache.")
            raise
        
        # Initialize collections
        self._init_collections()

    def get_query_embedding(self, query: str) -> List[float]:
        """Compute query embedding using the fast unified in-process embedding function with LRU caching."""
        from .retrieval import query_embedding_cache
        return query_embedding_cache.get_or_compute(query, self.embedding_fn)

    def _get_or_create_col(self, col_name: str, existing_cols: List[str]):
        """Helper to get or create collection with automatic conflict resolution."""
        if col_name in existing_cols:
            try:
                col = self.client.get_collection(col_name)
                if not getattr(col, '_embedding_function', None):
                    col._embedding_function = self.embedding_fn
                logger.info(f"Using existing '{col_name}' collection")
                return col
            except ValueError as e:
                if "embedding function conflict" in str(e):
                    logger.warning(f"Embedding function conflict in '{col_name}'. Recreating collection...")
                    self.client.delete_collection(col_name)
                    col = self.client.create_collection(name=col_name, embedding_function=self.embedding_fn)
                    logger.info(f"Created new '{col_name}' collection with embedding function")
                    return col
                raise
        else:
            col = self.client.create_collection(name=col_name, embedding_function=self.embedding_fn)
            logger.info(f"Created new '{col_name}' collection with embedding function")
            return col

    def _init_collections(self):
        """Get or initialize document_chunks, hierarchical_summaries, and conversation collections."""
        try:
            existing_collections = [col.name for col in self.client.list_collections()]
            logger.debug(f"Existing Chroma collections: {existing_collections}")
        except Exception as e:
            logger.error(f"Error listing ChromaDB collections: {e}")
            raise
        
        # 1. Document collections (PDF knowledge base)
        self.chunks_collection = self._get_or_create_col(self.chunks_col_name, existing_collections)
        self.summaries_collection = self._get_or_create_col(self.summaries_col_name, existing_collections)
        
        # 2. Conversational Memory collections (chat exchanges & conversation topic clusters)
        self.conv_chunks_collection = self._get_or_create_col(self.conv_chunks_col_name, existing_collections)
        self.conv_summaries_collection = self._get_or_create_col(self.conv_summaries_col_name, existing_collections)

    def count_chunks(self) -> int:
        """Return total number of chunks in database."""
        return self.chunks_collection.count()

    def count_summaries(self) -> int:
        """Return total number of summaries in database."""
        return self.summaries_collection.count()

    def add_chunks(self, document_name: str, chunks: List[Dict], doc_id: Optional[str] = None, force: bool = False, batch_size: int = None) -> int:
        """
        Add chunks to ChromaDB document_chunks collection with duplicate-ingestion protection.
        
        Args:
            document_name: Human-readable name or filename of the document.
            chunks: List of chunk dictionaries containing 'text', 'chunk_id', 'chunk_index'.
            doc_id: Deterministic hash or ID of the document (prevents duplicate ingestion).
            force: If True, overwrite/re-ingest existing chunks for this document.
            batch_size: Batch size for ChromaDB insertion (defaults to config).
        """
        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return 0
            
        if batch_size is None:
            batch_size = config.chunking.batch_size
        
        # 1. Duplicate-ingestion check
        if not force:
            if doc_id:
                try:
                    existing = self.chunks_collection.get(where={"doc_id": doc_id}, limit=1)
                    if existing and existing.get('ids') and len(existing['ids']) > 0:
                        logger.info(f"Document '{document_name}' [doc_id: {doc_id}] already exists in database. Skipping duplicate ingestion.")
                        return 0
                except Exception as e:
                    logger.warning(f"Note checking doc_id: {e}")
            
            if document_name:
                try:
                    existing = self.chunks_collection.get(where={"source_file": document_name}, limit=1)
                    if existing and existing.get('ids') and len(existing['ids']) > 0:
                        logger.info(f"Document '{document_name}' already exists in database. Skipping duplicate ingestion.")
                        return 0
                except Exception as e:
                    logger.warning(f"Note checking source_file: {e}")
        else:
            # Force mode: purge prior chunks for this document to avoid orphans
            try:
                if document_name:
                    self.chunks_collection.delete(where={"source_file": document_name})
                    logger.info(f"Purged prior chunks matching source_file: {document_name}")
                if doc_id:
                    self.chunks_collection.delete(where={"doc_id": doc_id})
                    logger.info(f"Purged prior chunks matching doc_id: {doc_id}")
            except Exception as e:
                logger.warning(f"Note during forced cleanup: {e}")
        
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB ({self.chunks_col_name})...")
        
        clean_prefix = doc_id if doc_id else os.path.splitext(document_name)[0].replace(' ', '_')
        all_ids = []
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            
            for chunk in batch_chunks:
                page_num = chunk.get('page_number', 1)
                # Deterministic chunk ID incorporating document prefix, page number, and chunk identifier
                chroma_id = f"{clean_prefix}_p{page_num}_{chunk['chunk_id']}"
                batch_ids.append(chroma_id)
                batch_documents.append(chunk['text'])
                
                metadata = {
                    'doc_id': doc_id or clean_prefix,
                    'source_file': document_name,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk.get('chunk_index', 0),
                    'page_number': page_num,
                    'start_char': chunk.get('start_char', 0),
                    'end_char': chunk.get('end_char', 0),
                    'token_count': chunk.get('token_count', 0),
                    'char_count': chunk.get('char_count', len(chunk['text'])),
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
                logger.debug(f"  Upserted batch {batch_start//batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                logger.error(f"Error during batch upsert into ChromaDB: {e}")
                raise
        
        logger.info(f"Successfully upserted {len(all_ids)} chunks (embeddings generated automatically via {self.embedding_model})")
        return len(all_ids)

    def get_indexed_sources(self) -> List[str]:
        """Return list of distinct source_file names currently stored in document_chunks."""
        try:
            total_count = self.chunks_collection.count()
            if total_count == 0:
                return []
            
            sources = set()
            batch_size = 1000
            for offset in range(0, total_count, batch_size):
                res = self.chunks_collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
                )
                if not res or not res.get('metadatas'):
                    break
                for m in res['metadatas']:
                    if m and 'source_file' in m and m['source_file']:
                        sources.add(m['source_file'])
                if len(res['metadatas']) < batch_size:
                    break
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error fetching indexed sources: {e}")
            return []

    def get_items_by_level(self, level: int = 0, source_file: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[Dict]], Optional[List[str]]]:
        """
        Get all items (embeddings, documents, metadata, ids) for a specific level.
        - Level 0 retrieves raw source chunks from document_chunks (can be filtered by source_file).
        - Level > 0 retrieves generated cluster summaries from hierarchical_summaries where level == level - 1.
        """
        if level == 0:
            target_col = self.chunks_collection
            filter_dict = {"source_file": source_file} if source_file else None
            src_str = f" for '{source_file}'" if source_file else ""
            desc = f"{self.chunks_col_name} (Level 0 source text{src_str})"
        else:
            target_col = self.summaries_collection
            filter_dict = {"level": level - 1}
            if source_file:
                filter_dict["source_file"] = source_file
            src_str = f" for '{source_file}'" if source_file else ""
            desc = f"{self.summaries_col_name} (Level {level - 1} summaries{src_str} for Level {level} clustering)"
        
        total_count = target_col.count()
        logger.info(f"Retrieving items from {desc}. Total collection items: {total_count}")
        
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
                logger.error(f"Error retrieving batch at offset {offset}: {e}")
                break
        
        if len(all_embeddings) == 0:
            logger.warning(f"No items found for clustering at level {level}")
            return None, None, None, None
        
        logger.info(f"Found {len(all_embeddings)} items for level {level}")
        return np.array(all_embeddings), all_documents, all_metadatas, all_ids

    def save_summary_record(self, summary_id: str, summary: str, summary_metadata: Dict,
                            summary_chunk_id: Optional[str] = None, chunk_metadata: Optional[Dict] = None):
        """
        Save a cluster summary to summaries_collection ONLY.
        Strictly avoids injecting synthetic summaries back into document_chunks.
        """
        try:
            self.summaries_collection.upsert(
                ids=[summary_id],
                documents=[summary],
                metadatas=[summary_metadata]
            )
        except Exception as e:
            logger.error(f"Failed to save summary record {summary_id}: {e}")
            raise

    def get_summary_tree(self) -> pd.DataFrame:
        """Get all summaries organized as a tree dataframe."""
        try:
            result = self.summaries_collection.get(
                include=["documents", "metadatas"]
            )
            
            if not result['ids']:
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'summary_id': result['ids'],
                'level': [m.get('level', -1) for m in result['metadatas']],
                'cluster_id': [m.get('cluster_id', -1) for m in result['metadatas']],
                'source_file': [m.get('source_file') or (m.get('source_files', [''])[0] if isinstance(m.get('source_files'), list) and m.get('source_files') else 'unknown') for m in result['metadatas']],
                'chunks_count': [m.get('chunks_count', 0) for m in result['metadatas']],
                'silhouette_score': [m.get('silhouette_score', 0) for m in result['metadatas']],
                'dense_summary': [m.get('dense_summary', '') for m in result['metadatas']],
                'summary_preview': [doc[:100] for doc in result['documents']],
                'created_at': [m.get('created_at', '') for m in result['metadatas']]
            })
            
            return df.sort_values(['level', 'cluster_id'])
        except Exception as e:
            logger.error(f"Error fetching summary tree: {e}")
            return pd.DataFrame()

    def add_conversation_turn(self, chat_id: str, user_text: str, assistant_text: str, 
                              turn_index: int = 0, chat_title: Optional[str] = None) -> str:
        """
        Index a conversation turn into the conversation_chunks collection.
        Enables the system to recall past user conversations, preferences, and personal details.
        """
        turn_id = f"conv_{chat_id}_turn_{turn_index}"
        text_content = f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"
        
        metadata = {
            "chat_id": chat_id,
            "chat_title": chat_title or "Conversation",
            "turn_index": turn_index,
            "timestamp": pd.Timestamp.now().isoformat(),
            "level": 0,
            "type": "conversation_turn",
            "source_file": f"chat_{chat_id}",
            "user_preview": user_text[:120],
            "assistant_preview": assistant_text[:120]
        }
        
        # Check if already present to prevent duplicate accumulation
        existing = self.conv_chunks_collection.get(ids=[turn_id])
        if existing and existing.get('ids'):
            self.conv_chunks_collection.update(
                ids=[turn_id],
                documents=[text_content],
                metadatas=[metadata]
            )
            logger.info(f"Updated conversation memory: {turn_id}")
        else:
            self.conv_chunks_collection.add(
                ids=[turn_id],
                documents=[text_content],
                metadatas=[metadata]
            )
            logger.info(f"Stored conversation memory: {turn_id}")
            
        return turn_id

    def get_conversation_items(self, level: int = 0) -> Tuple[np.ndarray, List[str], List[Dict], List[str]]:
        """Retrieve conversation items with embeddings for clustering."""
        target_col = self.conv_chunks_collection if level == 0 else self.conv_summaries_collection
        
        count = target_col.count()
        if count == 0:
            return np.array([]), [], [], []
            
        result = target_col.get(
            include=["documents", "metadatas", "embeddings"],
            limit=10000
        )
        
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for i, (cid, doc, meta) in enumerate(zip(result['ids'], result['documents'], result['metadatas'])):
            if level > 0 and meta.get('level', 0) != level:
                continue
                
            emb = result['embeddings'][i] if result.get('embeddings') is not None and len(result['embeddings']) > i else None
            if emb is None:
                # Generate embedding on the fly if missing
                emb = self.embedding_fn([doc])[0]
                
            embeddings.append(emb)
            documents.append(doc)
            metadatas.append(meta or {})
            ids.append(cid)
            
        logger.info(f"Retrieved {len(embeddings)} conversation items for level {level}")
        return np.array(embeddings), documents, metadatas, ids

    def save_conversation_summary_record(self, summary_id: str, summary: str, summary_metadata: Dict):
        """Save a hierarchical summary of conversation clusters."""
        try:
            self.conv_summaries_collection.upsert(
                ids=[summary_id],
                documents=[summary],
                metadatas=[summary_metadata]
            )
            logger.info(f"Saved conversation cluster summary {summary_id}")
        except Exception as e:
            logger.error(f"Error saving conversation summary {summary_id}: {e}")
            raise

    def count_conversation_chunks(self) -> int:
        """Return total number of conversation turns in memory."""
        return self.conv_chunks_collection.count()

    def count_conversation_summaries(self) -> int:
        """Return total number of conversation cluster summaries."""
        return self.conv_summaries_collection.count()

    def get_conversation_summary_tree(self) -> pd.DataFrame:
        """Get all conversation summaries organized as a tree dataframe."""
        try:
            result = self.conv_summaries_collection.get(include=["documents", "metadatas"])
            if not result['ids']:
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'summary_id': result['ids'],
                'level': [m.get('level', -1) for m in result['metadatas']],
                'cluster_id': [m.get('cluster_id', -1) for m in result['metadatas']],
                'chunks_count': [m.get('chunks_count', 0) for m in result['metadatas']],
                'silhouette_score': [m.get('silhouette_score', 0) for m in result['metadatas']],
                'dense_summary': [m.get('dense_summary', '') for m in result['metadatas']],
                'summary_preview': [doc[:100] for doc in result['documents']],
                'created_at': [m.get('created_at', '') for m in result['metadatas']]
            })
            return df.sort_values(['level', 'cluster_id'])
        except Exception as e:
            logger.error(f"Error fetching conversation summary tree: {e}")
            return pd.DataFrame()

import requests
import re
import pandas as pd
from typing import List, Dict, Optional
import time
from datetime import datetime
import json
import os
import sys

from modules.config import config
from modules.logger import get_logger, setup_logger
import concurrent.futures
from modules.retrieval import (
    BM25Scorer,
    reciprocal_rank_fusion,
    deduplicate_retrieved_chunks,
    rerank_hybrid_candidates,
    is_chitchat_query,
    is_common_chat_query,
    retrieve_conversation_memories,
    retrieval_result_cache,
    collapsed_tree_fusion,
    pack_collapsed_context
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = get_logger("face")

# Import backend from main.py
try:
    from main import HierarchicalClusterSummarizer
except ImportError:
    print("❌ Error: Could not import HierarchicalClusterSummarizer from main.py")
    print("   Make sure main.py is in the same directory")
    sys.exit(1)


class RAGInterface:
    """
    Conversational RAG Interface using main.py for retrieval and Ollama for generation
    Runs locally using ChromaDB PersistentClient and native Ollama
    """
    
    def __init__(self, 
                 ollama_model: Optional[str] = None, 
                 ollama_embedding_model: Optional[str] = None,
                 ollama_url: Optional[str] = None,
                 chroma_path: Optional[str] = None,
                 chroma_host: Optional[str] = None,
                 chroma_port: Optional[int] = None,
                 max_context_chunks: Optional[int] = None,
                 include_summaries: Optional[bool] = None,
                 temperature: Optional[float] = None,
                 min_score: Optional[float] = None,
                 source_filter: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize the RAG Interface with local services
        """
        self.verbose = verbose
        self.ollama_model = ollama_model or config.ollama.generation_model
        ollama_url_clean = (ollama_url or config.ollama.url).rstrip('/')
        self.ollama_url = f"{ollama_url_clean}/api/generate"
        self.max_context_chunks = max_context_chunks if max_context_chunks is not None else config.retrieval.max_context_chunks
        self.include_summaries = include_summaries if include_summaries is not None else config.retrieval.include_summaries
        self.temperature = temperature if temperature is not None else config.retrieval.temperature
        self.min_score = min_score if min_score is not None else config.retrieval.min_relevance_score
        self.source_filter = source_filter
        self.conversation_history = []
        self.current_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # High-performance persistent connection pooling (eliminates Windows TCP handshaking delays)
        self.session = requests.Session()
        cpu_threads = min(12, os.cpu_count() or 4)
        self.ollama_options = {
            "temperature": self.temperature,
            "num_predict": 28,
            "top_k": 30,
            "top_p": 0.9,
            "num_ctx": 512,
            "num_thread": cpu_threads
        }

        # Initialize the processor with local ChromaDB & Ollama
        self.processor = HierarchicalClusterSummarizer(
            ollama_model=self.ollama_model,
            ollama_embedding_model=ollama_embedding_model or config.ollama.embedding_model,
            ollama_url=ollama_url_clean,
            chroma_path=chroma_path or config.chroma.path,
            chroma_host=chroma_host or config.chroma.host,
            chroma_port=chroma_port or config.chroma.port
        )
        
        # Check connections
        self._check_ollama()
        self._show_database_info()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            url_clean = config.ollama.url.rstrip('/')
            response = self.session.get(f"{url_clean}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if any(self.ollama_model in name for name in model_names):
                    if self.verbose:
                        print(f"✅ Ollama is running with model: {self.ollama_model}")
                    return True
                else:
                    print(f"⚠️ Model {self.ollama_model} not found. Run: ollama pull {self.ollama_model}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("   Please start Ollama with: ollama serve (or launch Ollama from the Start Menu)")
            return False
    
    def _show_database_info(self):
        """Show information about conversational memory and ChromaDB"""
        try:
            conv_count = self.processor.conv_chunks_collection.count() if hasattr(self.processor, 'conv_chunks_collection') else 0
            conv_sum_count = self.processor.conv_summaries_collection.count() if hasattr(self.processor, 'conv_summaries_collection') else 0
            total_chunks = self.processor.chunks_collection.count() if hasattr(self.processor, 'chunks_collection') else 0
            total_summaries = self.processor.summaries_collection.count() if hasattr(self.processor, 'summaries_collection') else 0
            
            if self.verbose:
                print(f"🧠 Conversational Memory: {conv_count} past turn(s) indexed, {conv_sum_count} memory cluster(s)")
                if total_chunks > 0:
                    print(f"📚 Auxiliary Documents: {total_chunks} chunks, {total_summaries} summaries")
        except Exception as e:
            print(f"Database info error: {e}")
            print(f"⚠️ Could not get database info: {e}")
    
    def retrieve_context(self, query: str, source_filter: Optional[str] = None) -> Dict:
        """
        High-performance context retrieval:
          - Fast chit-chat bypass (0 ms) for pure conversational queries (Item 2.3).
          - Retrieval result LRU caching (Item 2.4).
          - Single query embedding computation with LRU cache.
          - Concurrent collection queries via ThreadPoolExecutor (Item 3.3).
          - Hybrid BM25 + dense reranking, score thresholding, and deduplication.
        Target retrieval latency: <= 0.5s (max 2.0s).
        """
        if not query or not query.strip():
            return {'success': False, 'message': 'Empty or blank query provided'}

        t_start = time.perf_counter()
        clean_q = query.strip()
        effective_source = source_filter if source_filter is not None else self.source_filter

        # Check if user explicitly requested a specific document source filter (e.g. /source Steve Jobs.pdf):
        is_explicit_document_search = bool(
            effective_source and effective_source.strip().lower() not in ('none', 'chat', 'general', 'direct', 'normal', 'off', 'common', 'all')
        )

        # 0. DEFAULT MODE: Conversational AI with Episodic Memory Retrieval
        # (Zero dependency on embedded documents)
        if not is_explicit_document_search:
            conv_memories, conv_summaries = retrieve_conversation_memories(
                query=clean_q,
                conv_chunks_col=self.processor.conv_chunks_collection if hasattr(self.processor, 'conv_chunks_collection') else None,
                conv_summaries_col=self.processor.conv_summaries_collection if hasattr(self.processor, 'conv_summaries_collection') else None,
                store=self.processor.store
            )

            has_recalled_memory = bool(conv_memories)
            if self.verbose:
                if has_recalled_memory:
                    print(f"🧠 Recalled {len(conv_memories)} relevant past conversation turn(s) from memory.")
                else:
                    print(f"💬 Conversational mode (no past chat memory needed).")

            return {
                'success': True,
                'is_chitchat': True,
                'is_common_chat': True,
                'has_recalled_memory': has_recalled_memory,
                'chat_mode': 'conversational_memory' if has_recalled_memory else 'common_chat',
                'chunks': [],
                'metadatas': [],
                'scores': [],
                'cluster_summaries': [],
                'conversation_memories': conv_memories,
                'conversation_cluster_summaries': conv_summaries,
                'sources': [],
                'query': clean_q,
                'source_filter': None,
                'low_confidence': False,
                'retrieval_time': time.perf_counter() - t_start
            }

        if effective_source:
            indexed_sources = self.processor.get_indexed_sources()
            if effective_source not in indexed_sources:
                clean_target = effective_source[:-4].strip().lower() if effective_source.lower().endswith('.pdf') else effective_source.strip().lower()
                matches = [s for s in indexed_sources if clean_target in s.lower()]
                if matches:
                    effective_source = matches[0]

        # 1. Check Retrieval Result Cache (Item 2.4)
        cache_key = (clean_q.lower(), effective_source, self.max_context_chunks, self.min_score, self.include_summaries)
        cached_res = retrieval_result_cache.get(cache_key)
        if cached_res is not None:
            cached_res['retrieval_time'] = time.perf_counter() - t_start
            return cached_res

        if self.verbose:
            filter_str = f" [Filter: '{effective_source}']" if effective_source else ""
            print(f"\n🔍 Retrieving context for: {clean_q[:80]}...{filter_str}")

        try:
            if self.processor.chunks_collection.count() == 0:
                return {'success': False, 'message': 'Database is empty. Please ingest documents with main.py first.'}

            # 2. Compute query embedding ONCE with LRU caching (Item 2.4 & 3.3)
            query_emb = self.processor.store.get_query_embedding(clean_q)
            candidate_pool_size = max(self.max_context_chunks * 3, 12)

            # 3. Parallel Query Execution across Collections (Item 3.3)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Task A: Primary document chunks
                chunks_where = {"source_file": effective_source} if effective_source else None
                fut_chunks = executor.submit(
                    self.processor.chunks_collection.query,
                    query_embeddings=[query_emb],
                    n_results=candidate_pool_size,
                    where=chunks_where,
                    include=["documents", "metadatas", "distances"]
                )

                # Task B: Hierarchical document summaries
                fut_summaries = None
                if self.include_summaries and self.processor.summaries_collection.count() > 0:
                    sums_where = {"source_file": effective_source} if effective_source else None
                    fut_summaries = executor.submit(
                        self.processor.summaries_collection.query,
                        query_embeddings=[query_emb],
                        n_results=3,
                        where=sums_where,
                        include=["documents", "metadatas", "distances"]
                    )

                # Task C: Conversation memory chunks
                fut_conv = None
                if config.retrieval.include_conversation_memory and hasattr(self.processor, 'conv_chunks_collection') and self.processor.conv_chunks_collection.count() > 0:
                    fut_conv = executor.submit(
                        self.processor.conv_chunks_collection.query,
                        query_embeddings=[query_emb],
                        n_results=min(config.retrieval.max_memory_chunks, self.processor.conv_chunks_collection.count()),
                        include=["documents", "metadatas", "distances"]
                    )

                # Task D: Conversation cluster summaries
                fut_csum = None
                if hasattr(self.processor, 'conv_summaries_collection') and self.processor.conv_summaries_collection.count() > 0:
                    fut_csum = executor.submit(
                        self.processor.conv_summaries_collection.query,
                        query_embeddings=[query_emb],
                        n_results=min(2, self.processor.conv_summaries_collection.count()),
                        include=["documents", "metadatas", "distances"]
                    )

                try:
                    results = fut_chunks.result()
                except Exception as e:
                    logger.warning(f"Error querying chunks collection: {e}")
                    results = None

                try:
                    summary_results = fut_summaries.result() if fut_summaries else None
                except Exception as e:
                    logger.warning(f"Error querying summaries collection: {e}")
                    summary_results = None

                try:
                    mem_results = fut_conv.result() if fut_conv else None
                except Exception as e:
                    logger.warning(f"Error querying conversation memory collection: {e}")
                    mem_results = None

                try:
                    csum_results = fut_csum.result() if fut_csum else None
                except Exception as e:
                    logger.warning(f"Error querying conversation summaries collection: {e}")
                    csum_results = None

            if not results or not results.get('documents') or not results['documents'][0]:
                return {'success': False, 'message': f"No relevant chunks found{f' for source {effective_source}' if effective_source else ''}."}

            raw_chunks = results['documents'][0]
            raw_metadatas = results['metadatas'][0]
            raw_distances = results['distances'][0]

            # 4. Collapsed Tree Fusion (RAPTOR Upgrade - ICLR 2024)
            if config.retrieval.enable_collapsed_retrieval:
                s_docs = summary_results['documents'][0] if summary_results and summary_results.get('documents') and summary_results['documents'][0] else None
                s_metas = summary_results['metadatas'][0] if summary_results and summary_results.get('metadatas') and summary_results['metadatas'][0] else None
                s_dists = summary_results['distances'][0] if summary_results and summary_results.get('distances') and summary_results['distances'][0] else None
                
                candidates = collapsed_tree_fusion(
                    raw_chunks=raw_chunks,
                    raw_metas=raw_metadatas,
                    raw_dists=raw_distances,
                    summary_docs=s_docs,
                    summary_metas=s_metas,
                    summary_dists=s_dists
                )
            else:
                candidates = []
                for doc, meta, dist in zip(raw_chunks, raw_metadatas, raw_distances):
                    dense_score = max(0.0, min(1.0, 1.0 - float(dist)))
                    candidates.append({
                        'document': doc,
                        'metadata': meta,
                        'distance': dist,
                        'dense_score': dense_score,
                        'node_type': 'leaf_chunk'
                    })

            # Hybrid Reranking (BM25 + Dense Vector + Query Term Coverage across collapsed tree)
            reranked_candidates = rerank_hybrid_candidates(clean_q, candidates)

            # Score Thresholding: strict relevance and keyword coverage check
            threshold = max(self.min_score, 0.35)
            q_word_count = len(clean_q.split())
            cov_threshold = 0.20 if q_word_count > 15 else 0.35

            passed_candidates = [
                c for c in reranked_candidates 
                if c['rerank_score'] >= threshold and (c.get('coverage_score', 0) >= cov_threshold or (c.get('node_type') == 'tree_summary' and c.get('coverage_score', 0) >= 0.15))
            ]
            low_confidence = (len(passed_candidates) == 0)
            if low_confidence:
                top_score = reranked_candidates[0]['rerank_score'] if reranked_candidates else 0.0
                top_cov = reranked_candidates[0].get('coverage_score', 0.0) if reranked_candidates else 0.0

                # Unrelated query detection:
                # If no source was explicitly filtered by user, and top candidate has poor semantic relevance / coverage,
                # then this query is UNRELATED to the memory documents! Transition cleanly to Common Chat mode.
                if not effective_source and (top_score < 0.40 or top_cov < 0.25):
                    if self.verbose:
                        print(f"💬 Query has low document grounding (top_score={top_score:.1%}, cov={top_cov:.1%}). Transitioning to Common Chat.")
                    return {
                        'success': True,
                        'is_chitchat': True,
                        'is_common_chat': True,
                        'chat_mode': 'common_chat',
                        'chunks': [],
                        'metadatas': [],
                        'scores': [],
                        'cluster_summaries': [],
                        'conversation_memories': [],
                        'conversation_cluster_summaries': [],
                        'sources': [],
                        'query': clean_q,
                        'source_filter': None,
                        'low_confidence': False,
                        'unrelated_to_documents': True,
                        'retrieval_time': time.perf_counter() - t_start
                    }
                elif reranked_candidates and top_score >= 0.35 and top_cov >= 0.25:
                    passed_candidates = reranked_candidates[:1]
                else:
                    passed_candidates = []

            # Chunk Deduplication for leaf chunks
            cand_docs = [c['document'] for c in passed_candidates if c.get('node_type') == 'leaf_chunk']
            cand_metas = [c['metadata'] for c in passed_candidates if c.get('node_type') == 'leaf_chunk']
            cand_scores = [c['rerank_score'] for c in passed_candidates if c.get('node_type') == 'leaf_chunk']

            deduped_chunks, deduped_metas, deduped_scores = deduplicate_retrieved_chunks(
                cand_docs, cand_metas, cand_scores, max_jaccard_overlap=0.65
            )

            # Re-integrate deduped chunks into unified pool alongside tree summaries
            final_candidates = []
            for d_chunk, d_meta, d_score in zip(deduped_chunks, deduped_metas, deduped_scores):
                final_candidates.append({
                    'document': d_chunk,
                    'metadata': d_meta,
                    'rerank_score': d_score,
                    'node_type': 'leaf_chunk'
                })
            for c in passed_candidates:
                if c.get('node_type') == 'tree_summary':
                    final_candidates.append(c)

            final_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

            # 5. Budget-Aware Context Packing (guarantees TTFT <= 1.8s)
            packed = pack_collapsed_context(
                final_candidates,
                max_tokens=config.retrieval.collapsed_token_budget,
                max_chunks=self.max_context_chunks,
                max_summaries=1
            )

            retrieved_chunks = packed['chunks']
            retrieved_metadatas = packed['metadatas']
            relevance_scores = packed['scores']
            cluster_summaries = packed['cluster_summaries']
            collapsed_items = packed['collapsed_items']

            sources = []
            for meta in retrieved_metadatas:
                if meta and 'source_file' in meta:
                    sources.append(meta['source_file'])
            sources = sorted(list(set(sources)))

            # 6. Process Conversational Memories
            conversation_memories = []
            if mem_results and mem_results.get('documents') and mem_results['documents'][0]:
                for m_doc, m_meta, m_dist in zip(
                    mem_results['documents'][0],
                    mem_results['metadatas'][0],
                    mem_results['distances'][0]
                ):
                    m_score = max(0.0, min(1.0, 1.0 - float(m_dist)))
                    if m_score >= 0.20:
                        conversation_memories.append({
                            'text': m_doc,
                            'chat_id': m_meta.get('chat_id'),
                            'chat_title': m_meta.get('chat_title', 'Previous Chat'),
                            'turn_index': m_meta.get('turn_index', 0),
                            'timestamp': m_meta.get('timestamp', ''),
                            'score': m_score
                        })

            # 7. Process Conversational Cluster Themes
            conv_cluster_summaries = []
            if csum_results and csum_results.get('documents') and csum_results['documents'][0]:
                for cs_doc, cs_meta, cs_dist in zip(
                    csum_results['documents'][0],
                    csum_results['metadatas'][0],
                    csum_results['distances'][0]
                ):
                    cs_score = max(0.0, min(1.0, 1.0 - float(cs_dist)))
                    if cs_score >= 0.25:
                        conv_cluster_summaries.append({
                            'summary': cs_doc,
                            'dense_summary': cs_meta.get('dense_summary', ''),
                            'chats_represented': cs_meta.get('chats_represented', 'Chats'),
                            'score': cs_score,
                            'level': cs_meta.get('level', 0)
                        })

            elapsed = time.perf_counter() - t_start
            result_dict = {
                'success': True,
                'is_chitchat': False,
                'chunks': retrieved_chunks,
                'metadatas': retrieved_metadatas,
                'scores': relevance_scores,
                'cluster_summaries': cluster_summaries,
                'conversation_memories': conversation_memories,
                'conversation_cluster_summaries': conv_cluster_summaries,
                'sources': sources,
                'query': clean_q,
                'source_filter': effective_source,
                'low_confidence': low_confidence,
                'collapsed_items': collapsed_items,
                'retrieval_time': elapsed
            }

            retrieval_result_cache.put(cache_key, result_dict)
            return result_dict

        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return {'success': False, 'message': f'Error retrieving context: {str(e)}', 'query': clean_q}

    def build_prompt(self, context: Dict, session_history: Optional[List[Dict]] = None) -> str:
        """
        Build a concise, evidence-dense prompt prioritizing primary source evidence
        and conversational memory. Strictly bounds character count to guarantee
        TTFT <= 2.0s and Total Turnaround <= 5.0s.
        """
        user_query = context.get('query', '')
        
        # Conversational Mode (Default): Pure conversation or conversational memory recall
        if context.get('is_common_chat') or context.get('is_chitchat') or (not context.get('chunks') and not context.get('cluster_summaries')):
            prompt_parts = []
            conv_memories = context.get('conversation_memories', [])
            if conv_memories:
                prompt_parts.append("You are APEX, a thoughtful, articulate, and friendly AI companion with memory of past conversations.")
                prompt_parts.append("Relevant memory recalled from your previous discussions with the user is provided below:")
                prompt_parts.append("\n[Recalled Memory from Previous Conversation]:")
                for mem in conv_memories[:2]:
                    doc_text = mem.get('text', '').strip()
                    prompt_parts.append(f"{doc_text}")
                prompt_parts.append("\nInstructions:")
                prompt_parts.append("- Directly and warmly answer the user's question using this recalled memory.")
                prompt_parts.append("- Speak naturally like a friend remembering prior chats (e.g., 'As you mentioned earlier...', 'You told me before that...').")
                prompt_parts.append("- Keep your answer direct, warm, and concise (under 2-3 sentences).")
                prompt_parts.append("- Do not mention document restrictions or say 'the context does not contain'.")
            else:
                prompt_parts.append("You are APEX, a thoughtful, articulate, and friendly AI companion.")
                prompt_parts.append("Answer directly, conversationally, and helpfully using broad general knowledge.")
                prompt_parts.append("Keep your answer concise and engaging (under 2-3 sentences).")
                prompt_parts.append("Speak naturally like an intelligent partner. Do not mention document restrictions or say 'the context does not contain'.")

            # Recent in-session chat
            history_to_use = session_history if session_history is not None else self.conversation_history
            if history_to_use and len(user_query.split()) <= 35:
                prompt_parts.append("\nRecent chat in this session:")
                for turn in history_to_use[-2:]:
                    if isinstance(turn, dict):
                        q = turn.get('question') or turn.get('content', '')
                        a = turn.get('answer', '')
                        if q and a:
                            prompt_parts.append(f"User: {q[:100]}")
                            prompt_parts.append(f"APEX: {a[:100]}")

            prompt_parts.append(f"\nUser: {user_query}")
            prompt_parts.append("APEX:")
            return "\n".join(prompt_parts)

        prompt_parts = [
            "Directly answer in 1-2 sentences using source. Cite [Doc, Page]:\n"
        ]
        
        # Primary source evidence (strictly budgeted so prompt length never exceeds 75 tokens)
        chunks_to_use = context.get('chunks', [])[:self.max_context_chunks]
        per_chunk_limit = 95 if len(chunks_to_use) <= 1 else 55
        for i, (chunk, meta) in enumerate(zip(
            chunks_to_use, 
            context.get('metadatas', [])[:self.max_context_chunks]
        )):
            source = meta.get('source_file', 'Doc') if meta else 'Doc'
            page_num = meta.get('page_number', '?') if meta else '?'
            clean_chunk = chunk[:per_chunk_limit].strip().replace('\n', ' ')
            prompt_parts.append(f"[{source}, Page {page_num}]: {clean_chunk}")
        
        # Thematic summary (if present and enabled)
        if context.get('cluster_summaries') and self.include_summaries:
            for summary_info in context['cluster_summaries'][:1]:
                summary_text = (summary_info.get('dense_summary') or summary_info.get('summary', ''))[:150].strip().replace('\n', ' ')
                prompt_parts.append(f"[Theme]: {summary_text}")
        
        # Conversational memory (bounded)
        if context.get('conversation_memories'):
            for mem in context['conversation_memories'][:2]:
                clean_mem = mem['text'][:150].strip().replace('\n', ' ')
                prompt_parts.append(f"[Memory]: {clean_mem}")

        prompt_parts.append(f"\nQuestion: {user_query}\nAnswer:")
        return "\n".join(prompt_parts)
    
    def generate_response(self, prompt: str, is_common_chat: bool = False, user_query: Optional[str] = None) -> Dict:
        """
        Generate response using native Ollama LLM with keep-alive model pinning,
        tuned context size, thread count, and retry backoff (Item 2.2).
        """
        if self.verbose:
            print(" Generating response...")

        max_retries = config.ollama.max_retries
        retry_delay = config.ollama.retry_delay
        timeout = config.ollama.timeout

        # Adaptive options based on mode: Common Chat balances speed and completeness
        options = dict(self.ollama_options)
        if is_common_chat:
            if user_query:
                q_text = user_query
            else:
                m = re.search(r'(?:User|Question):\s*(.+)', prompt)
                q_text = m.group(1) if m else prompt
            is_coding = bool(re.search(r'\b(code|function|script|python|javascript|program|algorithm|class|def|import)\b', q_text, re.IGNORECASE))
            has_memory = "[Recalled Memory from Previous Conversation]" in prompt
            if is_coding:
                options["num_predict"] = 120
                options["temperature"] = 0.3
            elif has_memory:
                options["num_predict"] = 45
                options["temperature"] = 0.3
            else:
                options["num_predict"] = 35
                options["temperature"] = 0.4
        else:
            options["num_predict"] = 65
            options["temperature"] = self.temperature

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": options,
                        "keep_alive": -1
                    },
                    timeout=timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()

                    return {
                        'success': True,
                        'answer': answer,
                        'tokens': len(answer.split()),
                        'model': self.ollama_model
                    }
                else:
                    last_error = f"Ollama API returned HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Generation attempt {attempt}/{max_retries} failed: {last_error}")

            except requests.exceptions.Timeout:
                last_error = f"Ollama request timed out after {timeout}s (attempt {attempt}/{max_retries})."
                logger.warning(last_error)
            except requests.exceptions.ConnectionError:
                last_error = f"Cannot connect to Ollama on {self.ollama_url}. Make sure 'ollama serve' is running."
                logger.warning(last_error)
            except Exception as e:
                last_error = f"Unexpected generation error: {str(e)}"
                logger.error(last_error)
                break

            if attempt < max_retries:
                time.sleep(retry_delay * (1.5 ** (attempt - 1)))

        return {
            'success': False,
            'error': f"{last_error} (Troubleshooting: Run 'ollama serve' and ensure '{self.ollama_model}' is pulled)."
        }

    def generate_response_stream(self, prompt: str, is_common_chat: bool = False, user_query: Optional[str] = None):
        """
        Stream response tokens from native Ollama in real-time (Item 2.2).
        Yields token strings as they arrive from Ollama.
        """
        options = dict(self.ollama_options)
        if is_common_chat:
            if user_query:
                q_text = user_query
            else:
                m = re.search(r'(?:User|Question):\s*(.+)', prompt)
                q_text = m.group(1) if m else prompt
            is_coding = bool(re.search(r'\b(code|function|script|python|javascript|program|algorithm|class|def|import)\b', q_text, re.IGNORECASE))
            has_memory = "[Recalled Memory from Previous Conversation]" in prompt
            if is_coding:
                options["num_predict"] = 120
                options["temperature"] = 0.3
            elif has_memory:
                options["num_predict"] = 45
                options["temperature"] = 0.3
            else:
                options["num_predict"] = 35
                options["temperature"] = 0.4
        else:
            options["num_predict"] = 65
            options["temperature"] = self.temperature

        try:
            with self.session.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": options,
                    "keep_alive": -1
                },
                stream=True,
                timeout=config.ollama.timeout
            ) as response:
                for line in response.iter_lines(chunk_size=1):
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        yield token
                        if chunk.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n[Streaming error: {e}]" 
    
    def query(self, question: str, source_filter: Optional[str] = None) -> Dict:
        """
        Complete RAG query: Retrieve using main.py, Augment, Generate
        
        Args:
            question: User question
            source_filter: Optional source_file name to restrict search (Item 23)
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print(f"📝 Question: {question}")
        print("="*80)
        
        # Caution check: empty or whitespace question
        if not question or not str(question).strip():
            msg = "⚠️ Caution: Empty question provided. Please provide a non-empty question."
            print(f"\n{msg}")
            return {
                'question': question,
                'answer': msg,
                'sources': [],
                'context_chunks': 0,
                'response_time': 0.0
            }
        
        context = self.retrieve_context(question, source_filter=source_filter)
        
        if not context.get('success'):
            return {
                'question': question,
                'answer': f"Error: {context.get('message', 'Unknown error')}",
                'sources': [],
                'context_chunks': 0,
                'response_time': time.time() - start_time
            }
        
        has_recalled_memory = bool(context.get('conversation_memories'))
        is_common_chat = context.get('is_common_chat', False) or context.get('is_chitchat', False) or (not context.get('chunks') and not context.get('cluster_summaries'))

        if has_recalled_memory:
            mem_count = len(context['conversation_memories'])
            print(f"🧠 Mode: Conversational Memory (Recalled {mem_count} relevant past turn{'s' if mem_count > 1 else ''})")
            if self.verbose:
                for idx, m in enumerate(context['conversation_memories']):
                    text_prev = m.get('text', '').replace('\n', ' ')[:90]
                    print(f"   • Memory {idx+1}: \"{text_prev}...\" (Sim: {m.get('score', 0):.1%})")
        elif is_common_chat:
            print("💬 Mode: Conversational (General Knowledge / Dialogue)")
        elif context.get('low_confidence'):
            print("\n⚠️ Caution: Low similarity confidence (< 25%). Context may not directly answer this question.")
        
        prompt = self.build_prompt(context)
        response = self.generate_response(prompt, is_common_chat=is_common_chat, user_query=question)
        
        elapsed_time = time.time() - start_time
        
        if response.get('success'):
            answer = response['answer']
            
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'sources': context.get('sources', [])
            })
            
            if hasattr(self.processor, 'add_conversation_turn'):
                import threading
                threading.Thread(
                    target=self.processor.add_conversation_turn,
                    kwargs={
                        'chat_id': self.current_session,
                        'user_text': question,
                        'assistant_text': answer,
                        'turn_index': len(self.conversation_history),
                        'chat_title': 'Session Chat'
                    },
                    daemon=True
                ).start()
            
            print(f"\n✅ Response ({elapsed_time:.2f}s, {response.get('tokens', 0)} tokens):")
            print("-"*80)
            print(answer)
            print("-"*80)
            
            if context.get('sources'):
                print(f"\n📚 Sources: {', '.join(context['sources'])}")
            
            if self.verbose and not is_common_chat:
                print(f"\n📊 Retrieved {len(context.get('chunks', []))} primary chunks:")
                for i, (chunk, score, meta) in enumerate(zip(
                    context.get('chunks', [])[:10],
                    context.get('scores', [])[:10],
                    context.get('metadatas', [])[:10]
                )):
                    page_str = f" (p. {meta.get('page_number', '?')})" if meta and 'page_number' in meta else ""
                    chunk_id = meta.get('chunk_id', 'Unknown') if meta and 'chunk_id' in meta else 'Unknown'
                    print(f"   [{i+1}] {score:.2%} [Chunk: {chunk_id}] - {meta.get('source_file', 'Unknown')[:40]}{page_str}")
                
                if context.get('cluster_summaries'):
                    print(f"\n🌐 Auxiliary Thematic Context:")
                    for s in context['cluster_summaries']:
                        srcs = ", ".join(s.get('sources', []))
                        print(f"   • Level {s.get('level', 0)} Cluster {s.get('cluster_id')} | Relevance: {s.get('score', 0):.2%} | Silhouette: {s.get('silhouette_score', 0):.3f} | Sources: [{srcs}]")
            
            return {
                'question': question,
                'answer': answer,
                'sources': context.get('sources', []),
                'context_chunks': len(context.get('chunks', [])),
                'relevance_scores': context.get('scores', []),
                'conversation_memories': context.get('conversation_memories', []),
                'has_recalled_memory': has_recalled_memory,
                'is_common_chat': is_common_chat,
                'chat_mode': 'conversational_memory' if has_recalled_memory else ('common_chat' if is_common_chat else 'document_rag'),
                'unrelated_to_documents': context.get('unrelated_to_documents', False),
                'response_time': elapsed_time,
                'tokens_generated': response.get('tokens', 0),
                'conversation_turns': len(self.conversation_history)
            }
        else:
            return {
                'question': question,
                'answer': f"Error: {response.get('error', 'Unknown error')}",
                'sources': [],
                'context_chunks': 0,
                'has_recalled_memory': has_recalled_memory,
                'chat_mode': 'conversational_memory' if has_recalled_memory else ('common_chat' if is_common_chat else 'document_rag'),
                'response_time': elapsed_time
            }
    
    def interactive_mode(self):
        """Run interactive conversation mode"""
        print("\n" + "="*80)
        print("🤖 APEX - Conversational AI with Episodic Memory")
        print("="*80)
        conv_c = self.processor.conv_chunks_collection.count() if hasattr(self.processor, 'conv_chunks_collection') else 0
        print(f"🧠 Memory: {conv_c} past turns remembered across sessions")
        print(f"💬 Conversational Mode: Active (No document dependencies)")
        print(f"🤖 Ollama Model: {self.ollama_model}")
        print("\nCommands:")
        print("  /memories       - Show recent conversation memories remembered in ChromaDB")
        print("  /clear          - Clear current session memory")
        print("  /stats          - Show memory statistics")
        print("  /source <name>  - (Optional) Filter queries to a specific source document")
        print("  /help           - Show this help")
        print("  /quit           - Exit")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n❓ You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == '/quit':
                    print("\n👋 Goodbye!")
                    break
                    
                elif user_input.lower() == '/memories':
                    self.show_memories()
                    continue

                elif user_input.lower().startswith('/source'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        if self.source_filter:
                            print(f"📌 Active source filter: '{self.source_filter}'")
                        else:
                            print("📌 No source filter active (conversational mode).")
                    else:
                        sub = parts[1].strip()
                        if sub.lower() in ('clear', 'none', 'reset', 'all'):
                            self.source_filter = None
                            print("✅ Source filter cleared. Conversational mode active.")
                        else:
                            self.source_filter = sub
                            print(f"✅ Source filter set to: '{self.source_filter}'")
                    continue
                    
                elif user_input.lower() == '/clear':
                    self.conversation_history = []
                    print("✅ Conversation history cleared")
                    continue
                    
                elif user_input.lower() == '/stats':
                    self.show_stats()
                    continue
                    
                elif user_input.lower() == '/sources':
                    self.show_sources()
                    continue
                    
                elif user_input.lower() == '/clusters':
                    self.show_clusters()
                    continue
                    
                elif user_input.lower() == '/help':
                    self.show_help()
                    continue
                
                result = self.query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_stats(self):
        """Show database statistics from local ChromaDB"""
        try:
            total_chunks = self.processor.chunks_collection.count()
            total_summaries = self.processor.summaries_collection.count()
            
            print("\n📊 DATABASE STATISTICS (ChromaDB)")
            print("="*60)
            print(f"Total chunks: {total_chunks}")
            print(f"Total summaries: {total_summaries}")
            print(f"Session queries: {len(self.conversation_history)}")
            
            if total_chunks > 0:
                result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
                sources = {}
                for meta in result['metadatas']:
                    source = meta.get('source_file', 'Unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"\nDocument sources:")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    print(f"  • {source}: {count} chunks")
                
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    def show_memories(self):
        """Show recent conversation memories remembered in ChromaDB"""
        try:
            col = self.processor.conv_chunks_collection
            count = col.count()
            print("\n🧠 CONVERSATIONAL MEMORY (ChromaDB)")
            print("="*60)
            print(f"Total turns remembered: {count}")
            if count > 0:
                res = col.get(limit=10, include=["documents", "metadatas"])
                for i, (cid, doc, meta) in enumerate(zip(res['ids'], res['documents'], res['metadatas'])):
                    title = meta.get('chat_title', 'Chat')
                    ts = meta.get('timestamp', '')[:19]
                    print(f"\n  [{i+1}] Session: {title} ({ts})")
                    for line in doc.split('\n')[:2]:
                        print(f"      {line[:85]}")
            print("="*60)
        except Exception as e:
            print(f"Error fetching memories: {e}")
    
    def show_sources(self):
        """Show all documents in the database"""
        try:
            result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
            sources = {}
            for meta in result['metadatas']:
                source = meta.get('source_file', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
            
            print("\n📚 DOCUMENTS IN DATABASE (ChromaDB)")
            print("="*60)
            for source, count in sorted(sources.items()):
                print(f"  • {source}: {count} chunks")
                
        except Exception as e:
            print(f"Error getting sources: {e}")
    
    def show_clusters(self):
        """Show cluster summaries"""
        try:
            df = self.processor.get_summary_tree()
            if df.empty:
                print("\n📊 No cluster summaries found")
                return
            
            print("\n📊 CLUSTER SUMMARIES")
            print("="*60)   
            
            for level in sorted(df['level'].unique()):
                level_df = df[df['level'] == level]
                print(f"\nLevel {level} ({len(level_df)} clusters):")
                print("-"*40)
                
                for _, row in level_df.iterrows():
                    print(f"\n  Cluster {row['cluster_id']}:")
                    print(f"    Chunks: {row['chunks_count']}")
                    print(f"    Silhouette: {row['silhouette_score']:.3f}")
                    print(f"    Preview: {row['summary_preview'][:100]}...")
                    
        except Exception as e:
            print(f"Error getting clusters: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n📖 HELP")
        print("="*60)
        print("Commands:")
        print("  /source <name>  - Filter queries to a specific source document")
        print("  /source clear   - Clear active source filter (search all)")
        print("  /clear          - Clear conversation history")
        print("  /stats          - Show database statistics")
        print("  /sources        - Show all documents in database")
        print("  /clusters       - Show cluster summaries")
        print("  /help           - Show this help")
        print("  /quit           - Exit")
        print("\nTips:")
        print("  • Ask questions about your documents")
        print("  • The system retrieves relevant chunks using HNSW search")
        print("  • Responses include source attribution")
        print("  • Conversation history provides context for follow-up questions")
    
    def batch_query(self, questions: List[str], output_file: str = None) -> List[Dict]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            output_file: Optional file to save results
            
        Returns:
            List of results
        """
        results = []
        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] Processing...")
            result = self.query(question)
            results.append(result)
            time.sleep(0.5)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to {output_file}")
        
        return results
    
    def export_conversation(self, filename: str = None):
        """Export conversation history to file"""
        if not self.conversation_history:
            print("No conversation history to export")
            return
        
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'session': self.current_session,
            'model': self.ollama_model,
            'total_queries': len(self.conversation_history),
            'conversation': self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Conversation exported to {filename}")


# ========== MAIN EXECUTION ==========

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FACE - RAG Interface for Hierarchical Knowledge Base (Local Version)')
    parser.add_argument('--model', default=config.ollama.generation_model, help='Ollama model name')
    parser.add_argument('--embedding-model', default=config.ollama.embedding_model, help='Ollama embedding model')
    parser.add_argument('--ollama-url', default=config.ollama.url, help='Ollama API URL')
    parser.add_argument('--chroma-path', default=config.chroma.path, help='Local ChromaDB directory path')
    parser.add_argument('--chroma-host', default=config.chroma.host, help='ChromaDB host (optional, for remote server)')
    parser.add_argument('--chroma-port', type=int, default=config.chroma.port, help='ChromaDB port (optional, for remote server)')
    parser.add_argument('--chunks', type=int, default=config.retrieval.max_context_chunks, help='Number of chunks to retrieve')
    parser.add_argument('--no-summaries', action='store_true', help='Disable cluster summaries')
    parser.add_argument('--temperature', type=float, default=config.retrieval.temperature, help='LLM temperature')
    parser.add_argument('--log-level', default=config.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--source', '--filter', type=str, default=None, help='Filter retrieval to a specific document/source_file')
    parser.add_argument('--min-score', type=float, default=config.retrieval.min_relevance_score, help='Minimum composite score threshold for context chunks')
    parser.add_argument('--query', type=str, help='Single query (non-interactive mode)')
    parser.add_argument('--batch', type=str, help='File with list of questions (one per line)')
    parser.add_argument('--export', type=str, help='Export conversation to file')
    
    args = parser.parse_args()
    
    log_lvl = "WARNING" if args.quiet else args.log_level
    setup_logger(level=log_lvl)
    
    # Initialize interface with local services
    interface = RAGInterface(
        ollama_model=args.model,
        ollama_embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        chroma_path=args.chroma_path,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        max_context_chunks=args.chunks,
        include_summaries=not args.no_summaries,
        temperature=args.temperature,
        min_score=args.min_score,
        source_filter=args.source,
        verbose=not args.quiet
    )
    
    # Check if database has content
    total_chunks = interface.processor.chunks_collection.count()
    if total_chunks == 0:
        print("\n⚠️ Database is empty!")
        print("Please add documents first using main.py")
        print("Example: python main.py")
        return
    
    # Handle different modes
    if args.query:
        result = interface.query(args.query)
        print(f"\n{result['answer']}")
        
    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"\n❌ Batch file not found: {args.batch}")
            return
            
        with open(args.batch, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(questions)} questions...")
        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = interface.batch_query(questions, output_file)
        
    else:
        interface.interactive_mode()
        
        if args.export:
            interface.export_conversation(args.export)


if __name__ == "__main__":
    main()
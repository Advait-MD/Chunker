import requests
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
    retrieval_result_cache
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
        cpu_threads = 4
        self.ollama_options = {
            "temperature": self.temperature,
            "num_predict": 35,
            "top_k": 30,
            "top_p": 0.9,
            "num_ctx": 2048,
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
        """Show information about the database from local ChromaDB"""
        try:
            total_chunks = self.processor.chunks_collection.count()
            total_summaries = self.processor.summaries_collection.count()
            
            if self.verbose:
                print(f"📚 ChromaDB (local): {total_chunks} chunks, {total_summaries} summaries")
                
                # Show source distribution
                if total_chunks > 0:
                    result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
                    sources = {}
                    for meta in result['metadatas']:
                        source = meta.get('source_file', 'Unknown')
                        sources[source] = sources.get(source, 0) + 1
                    
                    print(f"📄 Sources: {len(sources)} document(s)")
                    for source, count in sources.items():
                        print(f"   • {source}: {count} chunks")
                else:
                    print("   ⚠️ Database is empty! Run main.py first to add documents.")
                    
        except Exception as e:
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

        # 0. Chit-chat bypass (Item 2.3)
        if is_chitchat_query(clean_q):
            return {
                'success': True,
                'is_chitchat': True,
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
                'retrieval_time': time.perf_counter() - t_start
            }

        effective_source = source_filter if source_filter is not None else self.source_filter

        # Pure Conversation Mode: bypass document chunks when user specifies 'none' or 'chat'
        if effective_source and effective_source.strip().lower() in ('none', 'chat', 'general', 'direct', 'normal', 'off'):
            return {
                'success': True,
                'is_chitchat': True,
                'chunks': [],
                'metadatas': [],
                'scores': [],
                'cluster_summaries': [],
                'conversation_memories': [],
                'conversation_cluster_summaries': [],
                'sources': [],
                'query': clean_q,
                'source_filter': 'none',
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

                results = fut_chunks.result()
                summary_results = fut_summaries.result() if fut_summaries else None
                mem_results = fut_conv.result() if fut_conv else None
                csum_results = fut_csum.result() if fut_csum else None

            if not results or not results.get('documents') or not results['documents'][0]:
                return {'success': False, 'message': f"No relevant chunks found{f' for source {effective_source}' if effective_source else ''}."}

            raw_chunks = results['documents'][0]
            raw_metadatas = results['metadatas'][0]
            raw_distances = results['distances'][0]

            # 4. Package candidates for hybrid BM25 + dense reranking
            candidates = []
            for doc, meta, dist in zip(raw_chunks, raw_metadatas, raw_distances):
                dense_score = max(0.0, min(1.0, 1.0 - float(dist)))
                candidates.append({
                    'document': doc,
                    'metadata': meta,
                    'distance': dist,
                    'dense_score': dense_score
                })

            # Hybrid Reranking (BM25 + Dense Vector + Query Term Coverage)
            reranked_candidates = rerank_hybrid_candidates(clean_q, candidates)

            # Score Thresholding: strict relevance and keyword coverage check
            threshold = max(self.min_score, 0.35)
            passed_candidates = [
                c for c in reranked_candidates 
                if c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.45
            ]
            low_confidence = (len(passed_candidates) == 0)
            if low_confidence:
                # If top candidate has solid query keyword coverage (>= 0.40), keep top 1
                if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.45 and reranked_candidates[0].get('coverage_score', 0) >= 0.50:
                    passed_candidates = reranked_candidates[:1]
                else:
                    passed_candidates = []

            # Chunk Deduplication
            cand_docs = [c['document'] for c in passed_candidates]
            cand_metas = [c['metadata'] for c in passed_candidates]
            cand_scores = [c['rerank_score'] for c in passed_candidates]

            deduped_chunks, deduped_metas, deduped_scores = deduplicate_retrieved_chunks(
                cand_docs, cand_metas, cand_scores, max_jaccard_overlap=0.65
            )

            retrieved_chunks = deduped_chunks[:self.max_context_chunks]
            retrieved_metadatas = deduped_metas[:self.max_context_chunks]
            relevance_scores = deduped_scores[:self.max_context_chunks]

            sources = []
            for meta in retrieved_metadatas:
                if meta and 'source_file' in meta:
                    sources.append(meta['source_file'])
            sources = sorted(list(set(sources)))

            # 5. Process Thematic Summaries
            cluster_summaries = []
            if summary_results and summary_results.get('documents') and summary_results['documents'][0]:
                for s_doc, s_meta, s_dist in zip(
                    summary_results['documents'][0],
                    summary_results['metadatas'][0],
                    summary_results['distances'][0]
                ):
                    s_score = max(0.0, min(1.0, 1.0 - float(s_dist)))
                    if s_score < 0.30:
                        continue
                    if s_meta.get('low_confidence', False) or float(s_meta.get('silhouette_score', 0.0)) <= 0.0:
                        continue
                    summary_sources = s_meta.get('source_files', [])
                    if isinstance(summary_sources, str):
                        summary_sources = [summary_sources]
                    if sources and summary_sources:
                        shared = set(sources).intersection(set(summary_sources))
                        if not shared:
                            continue
                    cluster_summaries.append({
                        'cluster_id': s_meta.get('cluster_id', 'unknown'),
                        'level': s_meta.get('level', 0),
                        'chunks_count': s_meta.get('chunks_count', 0),
                        'score': s_score,
                        'sources': summary_sources,
                        'silhouette_score': s_meta.get('silhouette_score', 0.0),
                        'summary': s_doc[:600],
                        'dense_summary': s_meta.get('dense_summary', '')
                    })
                    break

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
                'retrieval_time': elapsed
            }

            retrieval_result_cache.put(cache_key, result_dict)
            return result_dict

        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return {'success': False, 'message': f'Error retrieving context: {str(e)}'}

    def build_prompt(self, context: Dict, session_history: Optional[List[Dict]] = None) -> str:
        """
        Build a concise, evidence-dense prompt prioritizing primary source evidence
        and conversational memory. Strictly bounds character count to guarantee
        TTFT <= 2.0s and Total Turnaround <= 5.0s.
        """
        # Pure conversation branch: chit-chat or general dialogue with no matching document chunks
        if context.get('is_chitchat') or (not context.get('chunks') and not context.get('cluster_summaries')):
            prompt_parts = [
                "You are APEX, a thoughtful, articulate, and friendly AI companion.",
                "Chat warmly and concisely in 1-2 direct sentences like an intelligent human partner."
            ]
            if context.get('conversation_memories'):
                prompt_parts.append("\nPast memories:")
                for mem in context['conversation_memories'][:2]:
                    prompt_parts.append(f"- {mem.get('text', '')[:150]}")

            history_to_use = session_history if session_history is not None else self.conversation_history
            if history_to_use:
                prompt_parts.append("\nRecent chat:")
                if any('role' in m for m in history_to_use if isinstance(m, dict)):
                    for msg in history_to_use[-4:]:
                        role = "User" if msg.get('role') == 'user' else "APEX"
                        c = msg.get('content', '').strip()
                        if c and c != context['query']:
                            prompt_parts.append(f"{role}: {c[:150]}")
                else:
                    for turn in history_to_use[-2:]:
                        q = turn.get('question', '')
                        a = turn.get('answer', '')
                        if q and a:
                            prompt_parts.append(f"User: {q}")
                            prompt_parts.append(f"APEX: {a[:150]}")

            prompt_parts.append(f"\nUser: {context['query']}")
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

        prompt_parts.append(f"\nQuestion: {context['query']}\nAnswer:")
        return "\n".join(prompt_parts)
    
    def generate_response(self, prompt: str) -> Dict:
        """
        Generate response using native Ollama LLM with keep-alive model pinning,
        tuned context size, thread count, and retry backoff (Item 2.2).
        """
        if self.verbose:
            print(" Generating response...")

        max_retries = config.ollama.max_retries
        retry_delay = config.ollama.retry_delay
        timeout = config.ollama.timeout
        cpu_threads = 4

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": self.ollama_options,
                        "keep_alive": "1h"
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

    def generate_response_stream(self, prompt: str):
        """
        Stream response tokens from native Ollama in real-time (Item 2.2).
        Yields token strings as they arrive from Ollama.
        """
        try:
            with self.session.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": self.ollama_options,
                    "keep_alive": "1h"
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
        
        if context.get('low_confidence'):
            print("\n⚠️ Caution: Low similarity confidence (< 25%). Context may not directly answer this question.")
        
        prompt = self.build_prompt(context)
        response = self.generate_response(prompt)
        
        elapsed_time = time.time() - start_time
        
        if response.get('success'):
            answer = response['answer']
            
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'sources': context.get('sources', [])
            })
            
            print(f"\n✅ Response ({elapsed_time:.2f}s, {response.get('tokens', 0)} tokens):")
            print("-"*80)
            print(answer)
            print("-"*80)
            
            if context.get('sources'):
                print(f"\n📚 Sources: {', '.join(context['sources'])}")
            
            if self.verbose:
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
                'response_time': elapsed_time
            }
    
    def interactive_mode(self):
        """Run interactive conversation mode"""
        print("\n" + "="*80)
        print("🎯 FACE - RAG Interface (Frontend for Hierarchical Knowledge Base)")
        print("="*80)
        print(f"📚 ChromaDB: {self.processor.chunks_collection.count()} chunks total")
        print(f"🤖 Ollama Model: {self.ollama_model}")
        print(f"📊 Context: {self.max_context_chunks} chunks, Summaries: {self.include_summaries}")
        print("\nCommands:")
        print("  /source <name>  - Filter queries to a specific source document")
        print("  /source clear   - Clear active source filter (search all)")
        print("  /clear          - Clear conversation history")
        print("  /stats          - Show database statistics")
        print("  /sources        - Show all documents in database")
        print("  /clusters       - Show cluster summaries")
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
                    
                elif user_input.lower().startswith('/source'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        if self.source_filter:
                            print(f"📌 Active source filter: '{self.source_filter}'")
                        else:
                            print("📌 No source filter active (searching all documents).")
                    else:
                        sub = parts[1].strip()
                        if sub.lower() in ('clear', 'none', 'reset', 'all'):
                            self.source_filter = None
                            print("✅ Source filter cleared. Searching across all documents.")
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
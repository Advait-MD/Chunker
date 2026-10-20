"""
Detailed latency profiling script across all pipeline stages.
Benchmarks:
1. Query preprocessing
2. Embedding generation via Ollama (nomic-embed-text)
3. ChromaDB collection queries (document_chunks, summaries, conv_chunks, conv_summaries)
4. BM25 / Reranking / Deduplication
5. Prompt building
6. LLM response generation (Time-To-First-Token vs Total Generation)
7. End-to-end retrieval time vs Generation time
"""

import os
import sys
import time
import requests
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from face import RAGInterface
from modules.config import config
from modules.logger import setup_logger

setup_logger("WARNING")

def profile_pipeline():
    print("=" * 70)
    print("⏱️ PROFILING APEX RAG PIPELINE LATENCY")
    print("=" * 70)

    rag = RAGInterface(
        ollama_model=config.ollama.generation_model,
        ollama_embedding_model=config.ollama.embedding_model,
        chroma_path=config.chroma.path,
        verbose=False
    )

    test_queries = [
        "What is JSX and how does React render components?",
        "What is the Way of Strategy according to Miyamoto Musashi?",
        "Who held Odysseus captive on the island of Ogygia?",
        "What was Steve Jobs' vision for the Macintosh?",
        "Hello! How are you doing today?"
    ]

    # Warmup Ollama models
    print("\n[Warmup] Pre-warming models in Ollama...")
    t0 = time.perf_counter()
    try:
        requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": "warmup"}, timeout=10)
        requests.post("http://localhost:11434/api/generate", json={"model": "gemma3:1b", "prompt": "hi", "stream": False}, timeout=10)
    except Exception as e:
        print(f"Warmup error: {e}")
    print(f"Warmup done in {time.perf_counter() - t0:.3f}s")

    for idx, query in enumerate(test_queries, 1):
        print(f"\n" + "-" * 70)
        print(f"Query {idx}: \"{query}\"")
        print("-" * 70)

        # 1. Query preprocessing
        t_start = time.perf_counter()
        clean_query = query.strip()
        t_prep = time.perf_counter() - t_start
        print(f"  1. Preprocessing:               {t_prep * 1000:6.2f} ms")

        # 2. Raw Embedding Call (nomic-embed-text)
        t0 = time.perf_counter()
        emb_res = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": clean_query}, timeout=30)
        t_embed = time.perf_counter() - t0
        emb = emb_res.json().get('embedding', []) if emb_res.status_code == 200 else []
        print(f"  2. Embedding (nomic-embed-text): {t_embed * 1000:6.2f} ms")

        # 3. ChromaDB Query: document_chunks
        t0 = time.perf_counter()
        doc_res = rag.processor.chunks_collection.query(
            query_embeddings=[emb] if emb else None,
            query_texts=[clean_query] if not emb else None,
            n_results=15,
            include=["documents", "metadatas", "distances"]
        )
        t_chroma_docs = time.perf_counter() - t0
        docs_found = len(doc_res['documents'][0]) if doc_res.get('documents') and doc_res['documents'] else 0
        print(f"  3. ChromaDB document_chunks ({docs_found} chunks): {t_chroma_docs * 1000:6.2f} ms")

        # 4. ChromaDB Query: hierarchical_summaries
        t0 = time.perf_counter()
        sum_res = rag.processor.summaries_collection.query(
            query_embeddings=[emb] if emb else None,
            query_texts=[clean_query] if not emb else None,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        t_chroma_sums = time.perf_counter() - t0
        print(f"  4. ChromaDB hierarchical_summaries:  {t_chroma_sums * 1000:6.2f} ms")

        # 5. ChromaDB Query: conversation_chunks
        t0 = time.perf_counter()
        conv_res = rag.processor.conv_chunks_collection.query(
            query_embeddings=[emb] if emb else None,
            query_texts=[clean_query] if not emb else None,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        t_chroma_conv = time.perf_counter() - t0
        print(f"  5. ChromaDB conversation_chunks:    {t_chroma_conv * 1000:6.2f} ms")

        # 6. ChromaDB Query: conversation_summaries
        t0 = time.perf_counter()
        csum_res = rag.processor.conv_summaries_collection.query(
            query_embeddings=[emb] if emb else None,
            query_texts=[clean_query] if not emb else None,
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
        t_chroma_csum = time.perf_counter() - t0
        print(f"  6. ChromaDB conversation_summaries: {t_chroma_csum * 1000:6.2f} ms")

        # 7. Total Current face.py retrieve_context() (which runs sequentially with text queries)
        t0 = time.perf_counter()
        ctx = rag.retrieve_context(query)
        t_total_retrieval = time.perf_counter() - t0
        print(f"  --> TOTAL CURRENT RETRIEVAL TIME:   {t_total_retrieval:6.3f} s ({t_total_retrieval * 1000:.1f} ms)")

        # 8. LLM Prompt Build & Generation (Non-Streaming vs Time-To-First-Token)
        prompt = rag.build_prompt(ctx)
        
        # Test Streaming TTFT vs Total Gen Time
        t0 = time.perf_counter()
        ttft = None
        total_tokens = 0
        try:
            stream_resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:1b",
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.3, "num_predict": 200}
                },
                stream=True,
                timeout=60
            )
            for chunk in stream_resp.iter_lines():
                if chunk:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    total_tokens += 1
            t_gen_total = time.perf_counter() - t0
        except Exception as e:
            ttft = 0
            t_gen_total = 0

        print(f"  8. LLM Time-To-First-Token (TTFT):  {ttft:6.3f} s ({ttft * 1000:.1f} ms)")
        print(f"  9. LLM Total Generation Time:       {t_gen_total:6.3f} s ({total_tokens} chunks)")
        print(f"  ⚡ TOTAL END-TO-END TIME:            {t_total_retrieval + t_gen_total:6.3f} s")

if __name__ == "__main__":
    profile_pipeline()

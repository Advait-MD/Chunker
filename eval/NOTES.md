# APEX Repository Architecture & Harness Inspection Notes (`eval/NOTES.md`)

## 1. Pipeline Interface Inspection (`face.py`)
- **Current Entrypoint**: The primary RAG class in the repository is `RAGInterface` inside `face.py`.
- **Existing Query Method**: `RAGInterface.query(question: str, source_filter: Optional[str] = None) -> Dict`.
- **Existing Return Signature**:
  - `query()` returns a dictionary with keys:
    - `answer`: generated text string
    - `sources`: list of unique document filenames
    - `chunks`: list of chunk text passages passed to the prompt
    - `context_chunks`: count of retrieved text passages
    - `relevance_scores`: composite reranked scores
    - `conversation_memories`: episodic conversation items
    - `web_results`: web search items
    - `chat_mode`: routing state (`document_rag`, `conversational_memory`, `web_search`, `common_chat`)
    - `response_time`: elapsed time in seconds
    - `tokens_generated`: count of generated output tokens
- **Gaps Identified per Section 4**:
  - Missing method: `pipeline.answer(question: str) -> dict` does not exist.
  - Missing key: `retrieved_ids` (list of chunk IDs in ranked order) is not exposed by `query()`.
  - Missing key: `llm_calls` (integer count of LLM generation calls) is not explicitly reported.
- **Remediation Plan**:
  - Add `answer(self, question: str) -> Dict` to `RAGInterface` in `face.py`.
  - In `retrieve_context()`, ensure candidate objects retain their ChromaDB `id` so `retrieved_ids` is properly populated and ordered by post-rerank relevance.
  - Count LLM calls: `0` for chit-chat / low-confidence / empty query abstentions, `1` for standard single-turn LLM generation.
  - Perform this in an isolated step/commit labeled `expose harness outputs`.

## 2. Vector Indexing & Hybrid Retrieval
- **Vector Store**: Local embedded ChromaDB (`chromadb.PersistentClient`) situated at `./hierarchical_chroma_db`.
- **Index Type**: ChromaDB uses an HNSW graph index internally rather than an Inverted File (IVF) index. Parameters like `ivf_nprobe` in `configs/*.yaml` represent the candidate search exploration depth (mapped to Chroma's candidate retrieval limits).
- **Hybrid Retrieval**: Handled in `modules/retrieval.py` via `BM25Okapi` lexical scoring and Reciprocal Rank Fusion (`reciprocal_rank_fusion` with $k=60$).
- **RAPTOR Hierarchical Indexing**: APEX uses Gaussian Mixture Models (GMM) with Bayesian Information Criterion (BIC) in `modules/clustering.py` and collapsed tree fusion in `face.py`. When `raptor: true`, both raw leaf chunks (`document_chunks`) and cluster summaries (`hierarchical_summaries`) are queried and fused; when `raptor: false`, only leaf chunks are retrieved.

## 3. Local Model Architecture & Rule 5 Compliance
- **Generator Model**: `gemma3:1b` via local Ollama (`http://localhost:11434`).
- **Judge Model**: `llama3.2:1b` via local Ollama.
- **Rule 5 Compliance**: Hard Rule 5 mandates that the judge model must differ from the generator model. Using `llama3.2:1b` as judge and `gemma3:1b` as generator ensures strict compliance with zero self-preference bias.
- **Temperature**: Pinned to `0` for deterministic reproducibility.

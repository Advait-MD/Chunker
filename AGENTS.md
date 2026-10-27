# AGENTS.md

Flat Python research repository for a hierarchical-clustering RAG pipeline ("APEX") over dense PDF corpora.

## Architecture (what's actually active)

- **`main.py`** — Backend & pipeline coordinator. Coordinates the modular pipeline via `HierarchicalClusterSummarizer` for document ingestion, hierarchical clustering, and conversation memory clustering (`cluster_conversations()`).
- **`app.py`** — Modern AI Chat Web UI application (FastAPI + HTML5/CSS3). Features sidebar chat history, "+ New Chat", session persistence, document source filters, episodic memory persistence, on-demand conversation clustering, and interactive evidence drawers (`http://localhost:7860`).
- **`face.py`** — Terminal/CLI frontend. Imports backend components. Connects to local ChromaDB and native Ollama. Run interactively (`python face.py`) or `--query "..."` / `--batch file.txt`. Retrieves both ground-truth PDF evidence and episodic conversation memories.
- **`inspector.py`** — Environment smoke-test checking local ChromaDB and native Ollama connectivity.
- **`modules/`** — Dedicated package directory for core pipeline components:
  - **`modules/__init__.py`** — Package initializer exporting core classes and functions.
  - **`modules/document_loader.py`** — PDF text extraction (`read_pdf`, `read_pdf_pages`) and page/token-aware chunking (`create_chunks_from_pages`).
  - **`modules/clustering.py`** — PCA dimensionality reduction, StandardScaler normalization, GMM clustering, and silhouette evaluation (`perform_clustering`).
  - **`modules/summarizer.py`** — Local Ollama LLM summarization with exponential backoff and keyword frequency fallback (`OllamaSummarizer`).
  - **`modules/vector_store.py`** — ChromaDB `PersistentClient` collection management (`document_chunks`, `hierarchical_summaries`, `conversation_chunks`, `conversation_summaries`), batch chunk insertion with citation metadata, and hierarchy retrieval (`ChromaStore`).
  - **`modules/retrieval.py`** — Hybrid retrieval (`BM25Scorer`), Reciprocal Rank Fusion (`reciprocal_rank_fusion`), composite reranking (`rerank_hybrid_candidates`), and chunk deduplication (`deduplicate_retrieved_chunks`).
  - **`modules/config.py`** — Centralized typed dataclass configuration with environment variable overrides.
  - **`modules/logger.py`** — Standardized timestamped logging infrastructure.
- **`eval/`** — Rigorous evaluation harness with LLM-as-a-judge (`judge.py`), golden dataset (`dataset/gold.jsonl`), automated metrics (`metrics.py`), and test runners.
- **`naive rag/`** — Strict reproduction baseline of Lewis et al. (2020) DPR + FAISS IndexFlatIP for head-to-head benchmarking.
- **`raptor/`** — RAPTOR (Sarthi et al., ICLR 2024) tree clustering reference implementation for comparative analysis.

## Running (100% Local / Zero Docker)

> See **[`README.md`](README.md)** for the complete guide to all ingestion, querying, ablation, and testing commands.

```powershell
# 1. Start Ollama locally (if not already running from Windows Start Menu)
ollama serve

# 2. Ensure models are pulled locally
ollama pull gemma3:1b          # generation model
ollama pull nomic-embed-text   # embedding model

# 3. Check environment health
python inspector.py

# 4. Ingest PDFs & Hierarchical Clustering (optional if DB already populated)
python main.py

# 5. Interactive RAG query (or --query "..." / --batch file.txt)
python face.py

# 6. Launch Web UI
python app.py
```

Both `main.py` and `face.py` preflight Ollama (`/api/tags`) and print the exact pull command if a model is missing.

## Key Considerations

- **Single Active Database**: The sole active local store is `hierarchical_chroma_db/`. All legacy Chroma stores (`chroma_db/`, `pdf_chroma_db/`, `document_db/`, etc.) have been pruned and cleaned.
- **Duplicate Protection**: File SHA-256 hashes prevent duplicate ingestion, and deterministic chunk IDs enable idempotent upserts.
- **Strict Separation of Truth vs. Summary**: `document_chunks` strictly contains primary PDF text. `hierarchical_summaries` holds synthetic cluster summaries.
- **Dependencies**: Listed in `requirements.txt` and `pyproject.toml`. Run `pip install -r requirements.txt`.
- **Environment Overrides**: All pipeline hyperparameters (chunk size, GMM parameters, retrieval weights, models) can be configured via `.env` without modifying source code.

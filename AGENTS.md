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
- **`eval/`** — Comprehensive academic evaluation suite:
  - **`eval/metrics.py`** — Formal metrics (Token F1, ROUGE-L, BLEU-4, 95% bootstrap CIs via $1,000\times$ resampling, paired Student's $t$-test & Wilcoxon signed-rank tests, RAGAS claim faithfulness).
  - **`eval/run_ablation_matrix.py`** — Programmatic 7-row architectural ablation matrix runner generating Markdown and camera-ready LaTeX tables.
  - **`eval/run_multi_model_benchmark.py`** — Cross-model scaling evaluator (Google Gemma 3 vs. Meta Llama 3.2).
  - **`eval/systems_profiling.py`** — Systems, theoretical forward FLOPs ($2 \times N_{\text{params}} \times T_{\text{total}}$), and inference dollar cost profiler.
  - **`eval/THEORETICAL_PROOFS.md`** — Mathematical proofs for Theorem 1 (Hallucination Suppression), Theorem 2 (BIC Consistency), and Theorem 3 (Bounded Regret of Collapsed Tree RRF).
  - **`eval/judge.py`** & **`eval/harness.py`** — LLM-as-a-judge harness and response cache.
  - **`eval/dataset/gold.jsonl`** — Standardized evaluation dataset with verified ground truth citations.
- **`paper/`** — Camera-ready academic conference manuscript package targeting ACL / EMNLP / ICLR / NeurIPS / MLSys:
  - **`paper/main.tex`** — Complete two-column LaTeX paper with TikZ vector architecture diagrams, theoretical proofs, and empirical tables.
  - **`paper/references.bib`** — Comprehensive BibTeX bibliography.
  - **`paper/README.md`** — Compilation instructions for `pdflatex`, `latexmk`, and Overleaf.
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

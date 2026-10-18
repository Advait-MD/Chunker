# AGENTS.md

Flat Python research repo for a hierarchical-clustering RAG pipeline ("APEX") over PDFs. No package layout, no README, no requirements.txt, no tests, no lint/CI — don't go looking for them.

## Architecture (what's actually active)

- **`main.py`** — current backend & coordinator. Coordinates the modular pipeline via `HierarchicalClusterSummarizer` and provides the execution pipeline (`main()`).
- **`document_loader.py`** — PDF text extraction (`read_pdf`) and text chunking (`create_chunks`).
- **`clustering.py`** — PCA dimensionality reduction, StandardScaler normalization, and GMM clustering (`perform_clustering`).
- **`summarizer.py`** — Local Ollama LLM summarization and keyword frequency fallback (`OllamaSummarizer`).
- **`vector_store.py`** — ChromaDB `PersistentClient` collection management, batch chunk insertion, and hierarchy retrieval (`ChromaStore`).
- **`face.py`** — frontend/CLI. Imports `HierarchicalClusterSummarizer` from `main.py`. Connects to local ChromaDB and native Ollama. Run interactively (`python face.py`) or `--query "..."` / `--batch file.txt`.
- **Legacy experiments** (don't confuse with the active path): `mk2.py`, `mk2_max.py` use `chromadb.PersistentClient` with **legacy dirs** (`./pdf_chroma_db`, `./document_db`). `chunk.py` / `ecs.py` use `sentence_transformers`. `fixed_chunk.py` is pure text chunking. `chroma.py`, `inspector.py` are smoke tests.
- **`docker-compose.yml`** — legacy container orchestration file (Docker is no longer required; the stack runs 100% locally on native Python and native Ollama).

## Running (100% Local / No Docker)

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
```

Both `main.py` and `face.py` preflight Ollama (`/api/tags`) and print the exact pull command if a model is missing.

## Gotchas

- **Two ChromaDBs coexist**: the local store in `hierarchical_chroma_db/` is what `main.py`/`face.py` use. The on-disk dirs (`chroma_db/`, `pdf_chroma_db/`, `document_db/`, `improved_chroma_db/`, `my_chroma_data/`) are legacy stores from mk2-era scripts.
- **Re-running `main.py` duplicates chunks**: ingest IDs include `uuid4().hex[:8]`, so every run re-adds the same PDF's chunks. Collections are also silently **deleted and recreated** on an "embedding function conflict" error — data loss is intentional there.
- **Embeddings** are generated server-side by Chroma using Ollama's `nomic-embed-text` (768-dim). Never pass raw `embeddings=` to `chunks_collection.add()` or you get a conflict.
- **No dependency lockfile**. Implicit deps: `chromadb`, `scikit-learn`, `pandas`, `numpy`, `requests`, `PyPDF2` (+ `sentence_transformers` only for `chunk.py`/`ecs.py`).
- **Git tracks artifacts**: `__pycache__/*.pyc`, Chroma sqlite/binaries, Ollama model blobs, `anythingllm_data/` (contains private keys), and generated CSVs (`hierarchical_summaries.csv`, `clustering_history.csv`, `cluster_report.csv`) are all committed; there is no `.gitignore`. Avoid needlessly staging binary/data churn.
- **`test docs/Test Questions.md`** is the evaluation question bank (3 Bloom's-taxonomy levels, Token F1 / Context Relevance targets) — prose, not runnable tests. Feed similar questions to `face.py --batch`.
- `mk2.py` sets `LOKY_MAX_CPU_COUNT=4` before sklearn work; other scripts don't.

# APEX RAG — Execution & Command Guide (`run.md`)

This guide provides the complete set of commands to run, configure, evaluate, and test the APEX Hierarchical-Clustering RAG pipeline.

All commands run **100% locally** on Windows / PowerShell (or bash) with zero external cloud dependencies.

---

## Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Health Check & Diagnostics](#2-health-check--diagnostics)
3. [Document Ingestion & Hierarchical Clustering (`main.py`)](#3-document-ingestion--hierarchical-clustering-mainpy)
4. [Querying & RAG Retrieval Interface (`face.py`)](#4-querying--rag-retrieval-interface-facepy)
5. [Evaluation & Verification Test Suites](#5-evaluation--verification-test-suites)
6. [Mobile Telegram Bot Interface (`telegram_bot.py`)](#6-mobile-telegram-bot-interface-telegram_botpy)
7. [Environment Variables Reference](#7-environment-variables-reference)

---

## 1. Prerequisites & Environment Setup

### 1.1 Install Python Dependencies
```powershell
pip install -r requirements.txt
```
*(Installs `chromadb`, `scikit-learn`, `requests`, `PyPDF2`, `pandas`, `numpy`)*

### 1.2 Start Local Ollama Server
If not already running from the Windows Start Menu, launch the background daemon:
```powershell
ollama serve
```

### 1.3 Pull Required Local Models
```powershell
# Generation model (Gemma 3 1B)
ollama pull gemma3:1b

# Embedding model (768-dim embeddings)
ollama pull nomic-embed-text
```

### 1.4 Verify Models Are Ready
```powershell
ollama list
```

---

## 2. Health Check & Diagnostics

Run the environment inspector to smoke-test ChromaDB and Ollama connectivity before running any ingestion or query tasks:

```powershell
python inspector.py
```

**What it does:**
- Verifies local ChromaDB persistent storage at `./hierarchical_chroma_db`.
- Lists existing collections (`document_chunks`, `hierarchical_summaries`) and their chunk counts.
- Pings Ollama at `http://localhost:11434` and validates that `gemma3:1b` and `nomic-embed-text` are installed.

---

## 3. Document Ingestion & Hierarchical Clustering (`main.py`)

`main.py` is the backend coordinator. It reads PDFs, extracts text with page numbers, builds token-bounded chunks, indexes them into ChromaDB, performs GMM clustering with silhouette quality gating, and stores hierarchical cluster summaries.

### 3.1 Standard Ingestion (All Default PDFs)
Ingests default PDF documents found in the workspace (`The Road to React`, `The Book of Five Rings`, `The Odyssey`, `Steve Jobs`):
```powershell
python main.py
```
> **Duplicate Protection Note:** If a document is already ingested (matching SHA-256 hash), `main.py` skips re-ingestion to prevent chunk duplication.

### 3.2 Ingest a Specific PDF
```powershell
python main.py --pdf "The Book of Five Rings.pdf"
```

### 3.3 Ingest Multiple Specific PDFs
```powershell
python main.py --pdf "The Odyssey.pdf" "Steve Jobs (1).pdf"
```

### 3.4 Force Re-Ingestion (Overwrite / Re-Chunk)
Bypasses duplicate SHA-256 detection and re-indexes the document:
```powershell
python main.py --pdf "The Book of Five Rings.pdf" --force
```

### 3.5 Control Hierarchy Depth
Limit the maximum number of recursive clustering levels (default: 3):
```powershell
python main.py --max-levels 2
```

### 3.6 Enable Detailed Debug Logging
```powershell
python main.py --log-level DEBUG
```

### 3.7 Custom Report Export Folder
Export clustering histories and summaries CSV files to a specific directory (default: root):
```powershell
python main.py --output-dir outputs
```

### 3.8 Source-Aware Clustering (Prevent Narrative Bleeding)
Clusters chunks within individual documents independently before cross-document macro synthesis (enabled by default):
```powershell
python main.py --source-aware
# Or disable to cluster across all documents simultaneously:
python main.py --no-source-aware
```

---

## 4. Querying & RAG Retrieval Interface

You can query your knowledge base using either the **Modern Web UI** (recommended for browser-based interactive chat) or the **CLI Interface** (`face.py`).

---

### 4.0 Modern Web Chat UI Interface (`app.py`) — Recommended!
Launches a modern, full-featured AI chat web application (like ChatGPT / Claude) with persistent conversation history, previous chats sidebar, document filter selector, and expandable citation drawers:

```powershell
python app.py
```
*Access the interface locally at: **`http://localhost:7860`***

**Key Web UI Features:**
- **+ New Chat**: Start a fresh conversation with clean context anytime.
- **Previous Chats Sidebar**: View, search, load, rename, or delete past conversations. All sessions are saved automatically to the `chats/` directory.
- **Document Source Filter**: Filter queries across *All Documents* or restrict search strictly to an individual book (e.g. *The Road to React*, *The Book of Five Rings*, *The Odyssey*, *Steve Jobs*).
- **Interactive Evidence Drawer**: Click on any assistant message to expand the exact retrieved chunks, page numbers, chunk IDs, and composite relevance scores.
- **Settings Modal**: Dynamically adjust context chunks ($1 - 10$), score threshold, LLM temperature, and thematic summaries toggle.

---

### 4.1 Interactive Terminal Mode (`face.py`)
Starts an interactive terminal session where you can ask questions continuously:
```powershell
python face.py
```

**Commands available inside the interactive session:**
- `exit` or `quit` — Exit the program.
- `/sources` — Display all indexed documents and chunk statistics.
- `/chunks <N>` — Change number of retrieved context chunks dynamically (e.g. `/chunks 4`).
- `/export <filename>` — Export the current conversation session to a JSON or text file.
- `/help` — View the in-session help menu.

---

### 4.2 Single Query Execution (Direct CLI)
Ask a single question and receive an immediate answer with citations and exit:
```powershell
python face.py --query "What is JSX and how does React render elements?"
```

```powershell
python face.py --query "What is the Way of Strategy according to Miyamoto Musashi?"
```

```powershell
python face.py --query "Who held Odysseus captive on the island of Ogygia?"
```

---

### 4.3 Batch Question Evaluation Mode
Run a list of questions stored in a text file (one question per line):
```powershell
python face.py --batch "questions.txt"
```

---

### 4.4 Ablation Study: Vanilla RAG vs. Hierarchical RAG
Compare answers with and without hierarchical cluster summaries:

```powershell
# Vanilla RAG (retrieves raw document chunks ONLY, no cluster summaries)
python face.py --no-summaries --query "Explain the philosophy of the Book of Water"

# Hierarchical RAG (retrieves raw chunks + high-level thematic cluster summaries)
python face.py --query "Explain the philosophy of the Book of Water"
```

---

### 4.5 Retrieval & Model Customization Flags

| Flag | Description | Example |
|---|---|---|
| `--source <NAME>`, `--filter <NAME>` | Filter retrieval strictly to a specific document (Item 23) | `python face.py --source "The Book of Five Rings.pdf" --query "..."` |
| `--min-score <FLOAT>` | Minimum composite reranking score threshold (default: 0.25) (Item 26) | `python face.py --min-score 0.35 --query "..."` |
| `--chunks <N>` | Number of primary chunks to retrieve (default: 5) | `python face.py --chunks 3 --query "..."` |
| `--temperature <T>` | LLM sampling temperature (default: 0.3) | `python face.py --temperature 0.1 --query "..."` |
| `--no-summaries` | Disable auxiliary cluster summaries | `python face.py --no-summaries --query "..."` |
| `--model <NAME>` | Use an alternate Ollama generation model | `python face.py --model llama3.2 --query "..."` |
| `--embedding-model <NAME>` | Use an alternate embedding model | `python face.py --embedding-model nomic-embed-text` |
| `--quiet` | Suppress retrieval headers and metadata; output only the final answer | `python face.py --quiet --query "..."` |
| `--log-level <LVL>` | Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `python face.py --log-level WARNING` |
| `--export <FILE>` | Export question-answer session to a file | `python face.py --export session.json` |

---

### 4.6 Document-Filtered Search (`--source`)
Restricts retrieval exclusively to a single target book, preventing cross-document contamination. Supports full filenames or partial titles:
```powershell
# Filter by exact or partial book title:
python face.py --source "The Book of Five Rings" --query "What is the Book of Water?"
python face.py --source "React" --query "How does useState work?"
python face.py --source "The Odyssey" --query "Who is Polyphemus?"
python face.py --source "Steve Jobs" --query "What was the NeXT computer?"
```
*In interactive mode, use `/source <name>` to set the active filter and `/source clear` to reset.*

---

### 4.7 Score Thresholding & Low-Confidence Gating (`--min-score`)
Rejects candidate chunks whose composite relevance score (dense similarity + BM25 + keyword coverage) falls below the threshold:
```powershell
python face.py --min-score 0.40 --query "What is the doctrine of Void?"
```

---

### 4.8 Conversational Episodic Memory & Hierarchical Chat Clustering
APEX automatically retains conversational exchanges across chat sessions (irrespective of PDF knowledge base) and builds hierarchical topic clusters over user conversations:
- **Turn Recording**: Every question-and-answer exchange is automatically indexed into ChromaDB's `conversation_chunks` collection.
- **Hierarchical Chat Clustering**: Groups conversation turns into Level 0 topic clusters and Level 1+ macro themes using GMM + PCA and generates dual dense/full topic summaries with Ollama.
- **Unified RAG Recall**: When you ask questions about past discussions, preferences, or personal projects, APEX simultaneously retrieves both PDF ground-truth facts and relevant conversation memories.
- **Web UI Controls**: Click **"Cluster Memory"** in the sidebar footer to run clustering on demand; inspect recalled turns in the **"🧠 Recalled Conversational Memory"** evidence drawer.

Run the test suite verifying memory persistence, clustering, and recall:
```powershell
python scratch/test_conversation_clustering.py
```

---

## 5. Evaluation & Verification Test Suites

### 5.1 Multi-Domain RAG Benchmark Test
Runs end-to-end questions spanning technical, philosophical, narrative, and out-of-domain topics:
```powershell
python test_rag_pipeline.py
```

### 5.2 Code Cleanup & Architecture Verification
Validates `requirements.txt`, `.gitignore`, `config.py` overrides, modular imports, and Ollama retry backoff:
```powershell
python -u "C:\Users\hp\.gemini\antigravity\brain\a369e0f2-b110-458c-9aaa-169c55a7710f\scratch\test_code_cleanup.py"
```

### 5.3 Token & Page-Aware Chunking Unit Test
Validates sentence splitting, token boundaries, and citation metadata extraction:
```powershell
python -u "C:\Users\hp\.gemini\antigravity\brain\a369e0f2-b110-458c-9aaa-169c55a7710f\scratch\verify_chunking.py"
```

### 5.4 Quality Gate & Retrieval Separation Test
Validates that low/negative silhouette clusters are filtered out and that primary source evidence is placed first in prompts:
```powershell
python -u "C:\Users\hp\.gemini\antigravity\brain\a369e0f2-b110-458c-9aaa-169c55a7710f\scratch\test_items_5_6.py"
```

### 5.5 Data Quality & Clustering Logic Test Suite (Items 18 - 22)
Validates text cleaning, ligature normalization, de-hyphenation, intra-document clustering purity, dual dense summaries, and source attribution constraints:
```powershell
python -u "C:\Users\hp\.gemini\antigravity\brain\a369e0f2-b110-458c-9aaa-169c55a7710f\scratch\test_items_18_22.py"
```

### 5.6 Retrieval Improvements Test Suite (Items 23 - 28)
Validates source-filtered retrieval, BM25 keyword scoring, hybrid candidate reranking, score threshold gating, chunk deduplication, and prompt citation formatting:
```powershell
python -u "C:\Users\hp\.gemini\antigravity\brain\a369e0f2-b110-458c-9aaa-169c55a7710f\scratch\test_items_23_28.py"
```

---

## 6. Mobile Telegram Bot Interface (`telegram_bot.py`)

Run your laptop as a local APEX server and interact with your entire RAG pipeline from your phone via Telegram.

### 6.1 Setup Credentials
1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram and copy the API token.
2. (Recommended) Get your numeric Telegram User ID from [@userinfobot](https://t.me/userinfobot) to restrict access to only you.
3. Add credentials to `.env`:
```ini
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_ALLOWED_USERS=your_user_id
```

### 6.2 Launch Bot Server
Launch using the one-click batch script or directly with python:
```powershell
# Option A: One-click runner
.\run_bot.bat

# Option B: Direct Python execution
python telegram_bot.py
```

### 6.3 Mobile Phone Capabilities
- **Chat & Query:** Send any text message from your phone; receive grounded answers with exact source citations and latency metrics.
- **Upload PDFs:** Forward or attach any `.pdf` in the chat $\rightarrow$ the server downloads, token-chunks, and indexes it into ChromaDB.
- **Filter Sources:** Tap `[📚 Documents]` or `/sources` to toggle interactive inline buttons filtering search to specific books.
- **Inspect Clusters & Stats:** Tap `[📊 Stats]` or `[🧩 Clusters]` or run `/cluster_run` on-demand.

---

## 7. Environment Variables Reference

You can customize system parameters without modifying code by setting environment variables in your terminal session before execution.

### PowerShell Example:
```powershell
$env:OLLAMA_MODEL = "gemma3:1b"
$env:CHUNK_TARGET_TOKENS = "350"
$env:MIN_SILHOUETTE_THRESHOLD = "0.08"
$env:LOG_LEVEL = "INFO"

python main.py
```

### Supported Configuration Keys:

| Environment Variable | Default Value | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Native Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:1b` | Primary generation LLM |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | 768-dim embedding model |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `OLLAMA_MAX_RETRIES` | `3` | Max retries with exponential backoff |
| `CHROMA_PATH` | `./hierarchical_chroma_db` | Path to persistent ChromaDB folder |
| `CHROMA_CHUNKS_COL` | `document_chunks` | Collection for raw PDF text chunks |
| `CHROMA_SUMMARIES_COL` | `hierarchical_summaries` | Collection for cluster summaries |
| `CHROMA_CONV_CHUNKS_COL` | `conversation_chunks` | Collection for raw conversational exchanges |
| `CHROMA_CONV_SUMMARIES_COL` | `conversation_summaries` | Collection for hierarchical conversation cluster summaries |
| `CHUNK_TARGET_TOKENS` | `300` | Target tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `40` | Overlap tokens between chunks |
| `CLUSTERING_MAX_LEVELS` | `3` | Maximum clustering hierarchy levels |
| `MIN_SILHOUETTE_THRESHOLD` | `0.05` | Global silhouette score quality gate |
| `MIN_CLUSTER_SILHOUETTE` | `0.01` | Per-cluster silhouette score threshold |
| `MAX_CONTEXT_CHUNKS` | `5` | Default number of retrieved chunks |
| `MIN_RELEVANCE_SCORE` | `0.25` | Default minimum composite score threshold for context chunks |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

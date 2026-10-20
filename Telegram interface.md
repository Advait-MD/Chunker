# APEX RAG — Telegram Interface Execution Plan (`Telegram interface.md`)

This document provides the complete, step-by-step execution roadmap to configure, launch, test, and operate the **APEX Mobile Telegram Interface**.

---

## 1. System Overview & Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       📱 SMARTPHONE                         │
│   Telegram App (Voice/Text Prompts, PDF Uploads, Menus)     │
└──────────────────────────────▲──────────────────────────────┘
                               │ Encrypted Chat
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    ☁️ TELEGRAM CLOUD                        │
│                     api.telegram.org                        │
└──────────────────────────────▲──────────────────────────────┘
                               │ Outbound HTTPS Long-Polling
                               │ (Zero Port Forwarding / NAT-Friendly)
┌──────────────────────────────▼──────────────────────────────┐
│                    💻 LAPTOP (AI SERVER)                     │
│                                                             │
│  1. telegram_bot.py (Async Engine, Whitelist Guard, Worker) │
│  2. face.py (Hybrid BM25 + Dense RAG Retrieval Engine)       │
│  3. main.py (PDF Chunking & GMM Hierarchical Clustering)    │
│  4. Local ChromaDB (./hierarchical_chroma_db)               │
│  5. Local Ollama (gemma3:1b & nomic-embed-text)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Pre-Execution Checklist

Before launching the interface, confirm the following prerequisites on your laptop:

- [ ] **Python Environment:** Python 3.10+ installed with dependencies:
  ```powershell
  pip install -r requirements.txt
  ```
- [ ] **Ollama Service Running:**
  ```powershell
  # Check if Ollama is responsive:
  curl http://localhost:11434/api/tags
  # If not running, launch from Start Menu or:
  ollama serve
  ```
- [ ] **Required Models Downloaded:**
  ```powershell
  ollama list
  # Must show:
  # - gemma3:1b (Generation LLM)
  # - nomic-embed-text (Embedding model)
  ```
- [ ] **Database Sanity Check:**
  ```powershell
  python inspector.py
  # Verifies local ChromaDB persistent collections
  ```

---

## 3. Phase 1: Bot Creation & Authorization Setup

### Step 1.1: Register Bot with @BotFather (on Phone)
1. Open the Telegram app on your phone.
2. In the search bar, search for **`@BotFather`** (verified blue checkmark).
3. Send the command:
   ```text
   /newbot
   ```
4. Choose a display name for your bot:
   ```text
   APEX RAG Assistant
   ```
5. Choose a unique bot username ending in `bot`:
   ```text
   my_apex_rag_bot
   ```
6. **Copy the HTTP API Token** provided by BotFather (format: `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).

---

### Step 1.2: Get Your Numeric User ID (on Phone)
1. In Telegram, search for **`@userinfobot`**.
2. Send any message or `/start`.
3. The bot will reply with your numeric **Id** (e.g. `987654321`). Copy this number.

---

### Step 1.3: Configure `.env` on Your Laptop
In the project directory `c:\Users\hp\Chunker\`, create or edit `.env`:
```ini
# Telegram Bot Token from @BotFather
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ

# Your personal Telegram User ID from @userinfobot (Security Whitelist)
TELEGRAM_ALLOWED_USERS=987654321

# Fast mobile ingestion (false = chunk & index immediately, cluster on-demand)
TELEGRAM_AUTO_CLUSTER=false

# Local Ollama & ChromaDB configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PATH=./hierarchical_chroma_db
```

> 🔒 **Security Notice:** `TELEGRAM_ALLOWED_USERS` ensures that only *your* phone can interact with your laptop. Any other user who finds your bot handle will be rejected immediately.

---

## 4. Phase 2: Launching the Server on Your Laptop

### Option A: One-Click Runner (Recommended)
Double-click `run_bot.bat` or run in PowerShell:
```powershell
.\run_bot.bat
```

### Option B: Direct Python Execution
```powershell
python telegram_bot.py
```

### Expected Successful Output:
```text
======================================================================
🤖 APEX Mobile Telegram Interface Server
======================================================================
📡 Model: gemma3:1b
📚 Vector DB: ./hierarchical_chroma_db
🔒 Whitelist: 1 authorized user(s)

Starting Telegram Polling Engine (press Ctrl+C to stop)...
```

---

## 5. Phase 3: Verification & Mobile Testing Protocol

Once the laptop terminal shows polling has started, perform the following verification sequence directly from your phone:

### Test 1: Handshake & Welcome Menu
- **Action:** Open your bot in Telegram and tap **START** (or send `/start`).
- **Expected Result:** Bot replies with a welcome card showing active model (`gemma3:1b`), indexed chunk count, security status, and docks the persistent keyboard:
  ```
  ┌─────────────────────────┬─────────────────────────┐
  │     📚 Documents        │     📊 Stats            │
  ├─────────────────────────┼─────────────────────────┤
  │     🧩 Clusters         │     🧹 Clear Context    │
  └─────────────────────────┴─────────────────────────┘
  ```

---

### Test 2: Conversational RAG Query
- **Action:** Send a test question:
  ```text
  What was Steve Jobs's philosophy on product design?
  ```
- **Expected Result:**
  1. The bot pulses the `typing...` status at the top.
  2. Returns a grounded answer within 3–8 seconds.
  3. Includes exact source citations (e.g., `Steve Jobs (1).pdf`, page number, chunk ID).
  4. Shows latency and token count metrics.

---

### Test 3: Document Source Filtering
- **Action:** Tap **`📚 Documents`** (or send `/sources`).
- **Expected Result:** Bot displays an interactive inline keyboard of all indexed PDFs.
- **Action:** Tap `📄 The Book of Five Rings.pdf`.
- **Expected Result:** Bot confirms `Active filter: The Book of Five Rings.pdf`.
- **Action:** Ask: *"What is the Way of Strategy?"*
- **Expected Result:** Search is strictly confined to Musashi's text, preventing cross-document contamination.
- **Action:** Tap `🌐 All Documents (No Filter)` to reset.

---

### Test 4: Mobile PDF Document Upload & Ingestion
- **Action:** Tap the paperclip icon 📎 in Telegram and upload any new `.pdf` file from your phone.
- **Expected Result:**
  1. Status updates to `📥 [1/3] Downloading file to local server...`
  2. Status updates to `⚙️ [2/3] Extracting text & building token-bounded chunks...`
  3. Status updates to `✅ Successfully Ingested! X chunks indexed in ChromaDB.`
- **Action:** Ask a question specific to the uploaded PDF.
- **Expected Result:** The bot retrieves context from the newly uploaded document immediately.

---

### Test 5: Hierarchical Cluster Inspection
- **Action:** Tap **`🧩 Clusters`** (or send `/clusters`).
- **Expected Result:** Displays the hierarchical summary tree (Level 0 intra-document summaries, silhouette scores, and previews).
- **Action:** Send `/cluster_run`.
- **Expected Result:** Server executes source-aware GMM clustering in the background and reports progress and summary count.

---

### Test 6: Database Metrics & Context Reset
- **Action:** Tap **`📊 Stats`** (or send `/stats`).
- **Expected Result:** Displays total chunks, total summaries, and per-document breakdown.
- **Action:** Tap **`🧹 Clear Context`** (or send `/clear`).
- **Expected Result:** Clears conversation history memory for fresh queries.

---

## 6. Phase 4: Operational Best Practices & Remote Access

### 6.1 Prevent Laptop Sleep During Operation
Because your laptop acts as the server, it must stay awake to answer queries from your phone:
- **Windows Power Settings:** Set *"When plugged in, put my device to sleep after"* $\rightarrow$ **Never**.
- **Close Lid Behavior:** Set *"Choose what closing the lid does"* $\rightarrow$ **Do nothing** (when plugged in).

### 6.2 Running Headless / Background on Windows
To run the bot in the background without keeping a visible terminal open:
```powershell
# Start silently in background:
Start-Process python -ArgumentList "telegram_bot.py" -WindowStyle Hidden

# To stop the background process when done:
Get-Process python | Where-Object { $_.CommandLine -like "*telegram_bot.py*" } | Stop-Process
```

### 6.3 Log Monitoring
All bot activities, queries, and upload logs are automatically recorded to:
```text
c:\Users\hp\Chunker\apex_rag.log
```

---

## 7. Command Reference Card

| Command | Action / Behavior |
| :--- | :--- |
| `/start` | Welcome overview & bottom menu dock |
| `<any text>` | Query the knowledge base via RAG |
| `<send .pdf>` | Upload and index new PDF document |
| `/sources` | Interactive inline button document filter |
| `/source <name>` | Manually set document search filter |
| `/source clear` | Reset filter to search all documents |
| `/stats` | View chunk counts, summaries, and source metrics |
| `/clusters` | Inspect GMM hierarchical summary cards |
| `/cluster_run` | Execute hierarchical clustering on server |
| `/clear` | Reset multi-turn conversation memory |
| `/health` | Check local Ollama and ChromaDB connectivity |
| `/help` | Detailed command help guide |

# 📱 APEX Mobile Telegram Interface Guide (`Telegram.md`)

This guide provides the complete instructions to configure, run, and use the **APEX Mobile Telegram Interface**. It turns your local APEX pipeline into a private, secure mobile assistant accessible from your smartphone anywhere, running entirely on your local machine with zero third-party cloud hosting costs.

---

## ⚡ Quick Start: 3 Steps to Run

### 1. Configure `.env`
Ensure your `.env` file in the project root includes your credentials:
```ini
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TELEGRAM_ALLOWED_USERS=987654321
```

### 2. Verify Local Ollama
```powershell
ollama list
# Ensure gemma3:1b and nomic-embed-text are installed
```

### 3. Launch the Server
```powershell
# Option A: One-click runner
.\run_bot.bat

# Option B: Direct Python
python telegram_bot.py
```

Once running, open Telegram on your phone and send `/start` to begin!

---

## 🏗️ System Overview & Architecture

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
│                    💻 LOCAL MACHINE (AI SERVER)             │
│                                                             │
│  1. telegram_bot.py (Async Engine, Whitelist Guard, Worker) │
│  2. face.py (Hybrid BM25 + Dense RAG Retrieval Engine)       │
│  3. main.py (PDF Chunking & GMM Hierarchical Clustering)    │
│  4. Local ChromaDB (./hierarchical_chroma_db)               │
│  5. Local Ollama (gemma3:1b & in-process ONNX embeddings)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Bot Registration & Security Setup

### Step 1: Register Bot with @BotFather (on Phone)
1. In Telegram, search for **`@BotFather`** (verified blue checkmark).
2. Send `/newbot`.
3. Choose a display name (e.g. `APEX Assistant`).
4. Choose a unique bot username ending in `bot` (e.g. `my_apex_rag_bot`).
5. Copy the **HTTP API Token** provided.

### Step 2: Get Your Numeric User ID
1. In Telegram, search for **`@userinfobot`**.
2. Send any message or `/start`.
3. Copy your numeric **Id** (e.g. `987654321`).

### Step 3: Populate `.env`
```ini
TELEGRAM_BOT_TOKEN=your_token_from_botfather
TELEGRAM_ALLOWED_USERS=your_numeric_user_id
TELEGRAM_AUTO_CLUSTER=false
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
CHROMA_PATH=./hierarchical_chroma_db
```

> 🔒 **Security Whitelist:** `TELEGRAM_ALLOWED_USERS` strictly restricts access to your numeric ID. Any unauthorized user attempting to message your bot handle will be rejected immediately.

---

## 📱 Mobile Command & Feature Reference

| Command / Action | Description |
| :--- | :--- |
| `/start` | Open welcome dashboard & dock persistent mobile keyboard |
| `<any text query>` | Ask questions across your ingested documents |
| `<upload .pdf>` | Upload and automatically chunk & index a new PDF from phone |
| `/sources` | Interactive inline button menu to filter retrieval to a single PDF |
| `/source clear` | Reset document filter to search across all documents |
| `/clusters` | Inspect GMM hierarchical summary cards and cluster quality scores |
| `/cluster_run` | Trigger source-aware hierarchical clustering in background |
| `/stats` | View chunk counts, summary counts, and document breakdowns |
| `/clear` | Clear current multi-turn conversation memory |
| `/health` | Check local Ollama and ChromaDB connectivity |
| `/help` | Display command help guide |

---

## ⚙️ Operational Tips

### Prevent Machine Sleep
To allow remote queries while away from your desk:
- **Windows Power Settings:** Set *"When plugged in, put my device to sleep after"* $\rightarrow$ **Never**.
- **Close Lid Behavior:** Set *"Choose what closing the lid does"* $\rightarrow$ **Do nothing** (when plugged in).

### Background Execution
```powershell
# Start silently in background:
Start-Process python -ArgumentList "telegram_bot.py" -WindowStyle Hidden

# Stop background process:
Get-Process python | Where-Object { $_.CommandLine -like "*telegram_bot.py*" } | Stop-Process
```

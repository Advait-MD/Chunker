# APEX Telegram Quick Run Guide (`Telegram.md`)

> **Full Detailed Execution Plan:** See **[`Telegram interface.md`](Telegram%20interface.md)**

---

## ⚡ Quick Start: 3 Steps to Run

### 1. Configure `.env`
Ensure your `.env` file in `c:\Users\hp\Chunker\` has your credentials:
```ini
TELEGRAM_BOT_TOKEN=your_token_from_botfather
TELEGRAM_ALLOWED_USERS=your_numeric_user_id
```

### 2. Verify Local Ollama
```powershell
ollama list
# Ensure gemma3:1b and nomic-embed-text are installed
```

### 3. Launch Bot Server on Laptop
```powershell
# Option A: One-click runner
.\run_bot.bat

# Option B: Direct Python
python telegram_bot.py
```

Once running, open Telegram on your phone and send `/start` to begin!

---

## 📱 Mobile Cheat Sheet

- **Ask Questions:** Send any text message
- **Upload Document:** Send any `.pdf` file directly in chat
- **Filter Sources:** Tap `[📚 Documents]` or send `/sources`
- **View Clusters:** Tap `[🧩 Clusters]` or send `/clusters`
- **Check Database:** Tap `[📊 Stats]` or send `/stats`
- **Reset History:** Tap `[🧹 Clear Context]` or send `/clear`

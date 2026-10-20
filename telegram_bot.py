"""
APEX Telegram Bot Interface
Enables full remote control and conversational RAG over local documents
directly from your mobile phone using Telegram.

Architecture:
  - Phone Telegram App <-> Telegram Bot API <-> Local PC (Long Polling)
  - Non-blocking async execution using asyncio.to_thread
  - Ingests PDFs sent directly in chat
  - Real-time typing status and inline interactive buttons
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, Optional, Set
from functools import wraps

# Set terminal UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    constants
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from modules.config import config
from modules.logger import get_logger, setup_logger
from modules.telegram_helpers import (
    split_message,
    format_rag_response,
    format_stats_response,
    format_clusters_response
)
from face import RAGInterface
from main import HierarchicalClusterSummarizer

logger = get_logger("telegram_bot")

# Track per-user active source filters in memory
# e.g. {user_id: "The Book of Five Rings.pdf" or None}
USER_SOURCE_FILTERS: Dict[int, Optional[str]] = {}

# Lazy singletons
_rag_instance: Optional[RAGInterface] = None
_processor_instance: Optional[HierarchicalClusterSummarizer] = None


def get_rag() -> RAGInterface:
    """Lazy initialize and return singleton RAGInterface."""
    global _rag_instance
    if _rag_instance is None:
        logger.info("Initializing RAG Interface for Telegram...")
        _rag_instance = RAGInterface(
            ollama_model=config.ollama.generation_model,
            ollama_embedding_model=config.ollama.embedding_model,
            chroma_path=config.chroma.path,
            max_context_chunks=config.retrieval.max_context_chunks,
            temperature=config.retrieval.temperature,
            min_score=config.retrieval.min_relevance_score,
            verbose=False
        )
    return _rag_instance


def get_processor() -> HierarchicalClusterSummarizer:
    """Return the underlying processor for ingestion and clustering."""
    rag = get_rag()
    return rag.processor


# ================= SECURITY / AUTH DECORATOR =================

def restricted(func):
    """
    Decorator to restrict access only to allowed user IDs configured in TELEGRAM_ALLOWED_USERS.
    If no whitelist is configured, warns the user and shows their ID for easy setup.
    """
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        if not user:
            return

        allowed_ids = config.telegram.allowed_user_ids

        # If whitelist is empty, inform user about their ID so they can configure it
        if not allowed_ids:
            logger.warning(f"No TELEGRAM_ALLOWED_USERS configured. Request from User ID {user.id} ({user.first_name}).")
            # We allow interaction but display setup reminder on start
            return await func(update, context, *args, **kwargs)

        if user.id not in allowed_ids:
            logger.warning(f"⛔ Unauthorized access attempt from User ID {user.id} ({user.first_name})")
            if update.message:
                await update.message.reply_text(
                    f"⛔ <b>Access Denied</b>\n\n"
                    f"Your Telegram User ID is: <code>{user.id}</code>\n"
                    f"This bot is private. Add your User ID to <code>TELEGRAM_ALLOWED_USERS</code> in <code>.env</code> on your server to authorize access.",
                    parse_mode=constants.ParseMode.HTML
                )
            elif update.callback_query:
                await update.callback_query.answer("⛔ Unauthorized access.", show_alert=True)
            return

        return await func(update, context, *args, **kwargs)
    return wrapped


# ================= KEYBOARD BUILDERS =================

def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Persistent mobile bottom keyboard."""
    keyboard = [
        [KeyboardButton("📚 Documents"), KeyboardButton("📊 Stats")],
        [KeyboardButton("🧩 Clusters"), KeyboardButton("🧹 Clear Context")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def build_sources_inline_keyboard(sources: list, current_filter: Optional[str]) -> InlineKeyboardMarkup:
    """Build clickable inline buttons for indexed documents."""
    buttons = []
    
    # All documents option
    all_prefix = "✅ " if not current_filter else ""
    buttons.append([InlineKeyboardButton(f"{all_prefix}🌐 All Documents (No Filter)", callback_data="src_filter:__all__")])
    
    # Document options
    for src in sources:
        prefix = "✅ " if current_filter == src else ""
        short_name = src if len(src) <= 30 else src[:27] + "..."
        buttons.append([InlineKeyboardButton(f"{prefix}📄 {short_name}", callback_data=f"src_filter:{src}")])
        
    return InlineKeyboardMarkup(buttons)


# ================= ASYNC TYPING ACTION =================

async def send_typing_heartbeat(bot, chat_id: int, stop_event: asyncio.Event):
    """Periodically pulse typing indicator while Ollama or clustering is computing."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue


# ================= COMMAND HANDLERS =================

@restricted
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start and welcome user."""
    user = update.effective_user
    rag = get_rag()
    total_chunks = rag.processor.chunks_collection.count()
    allowed_ids = config.telegram.allowed_user_ids
    
    auth_status = "🔒 Whitelist Active" if allowed_ids else f"⚠️ Whitelist Empty (Your ID: <code>{user.id}</code>)"
    
    welcome_text = (
        f"👋 <b>Welcome to APEX RAG Mobile, {user.first_name}!</b>\n\n"
        f"Your local server is online and running on this machine.\n\n"
        f"🤖 <b>LLM:</b> <code>{rag.ollama_model}</code>\n"
        f"📚 <b>Vector DB:</b> {total_chunks} chunks indexed\n"
        f"🛡️ <b>Security:</b> {auth_status}\n\n"
        f"<b>How to use:</b>\n"
        f"• <b>Ask anything:</b> Just send any message directly\n"
        f"• <b>Add documents:</b> Send or forward any <code>.pdf</code> file here\n"
        f"• <b>Filter by source:</b> Tap <i>📚 Documents</i> or use <code>/sources</code>\n"
        f"• <b>View clusters:</b> Tap <i>🧩 Clusters</i> or use <code>/clusters</code>\n"
        f"• <b>Commands:</b> <code>/stats</code>, <code>/clear</code>, <code>/help</code>"
    )
    
    await update.message.reply_text(
        welcome_text,
        parse_mode=constants.ParseMode.HTML,
        reply_markup=get_main_keyboard()
    )


@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed help message."""
    help_text = (
        "📖 <b>APEX Mobile Commands Guide</b>\n"
        "─────────────────────\n"
        "• <b>Any text message</b> — Ask questions to the RAG pipeline\n"
        "• <b>Upload any PDF</b> — Ingests and chunks the document into ChromaDB\n"
        "• <code>/sources</code> — View indexed PDFs & switch search filters\n"
        "• <code>/source &lt;name&gt;</code> — Set active source filter\n"
        "• <code>/source clear</code> — Reset filter (search all documents)\n"
        "• <code>/stats</code> — View database chunk & summary metrics\n"
        "• <code>/clusters</code> — Inspect hierarchical cluster tree\n"
        "• <code>/cluster_run</code> — Trigger hierarchical GMM clustering\n"
        "• <code>/clear</code> — Reset conversation history context\n"
        "• <code>/health</code> — Check local Ollama and ChromaDB status\n"
        "• <code>/help</code> — Show this help message"
    )
    await update.message.reply_text(help_text, parse_mode=constants.ParseMode.HTML)


@restricted
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show system metrics and indexed sources."""
    rag = get_rag()
    
    def fetch_stats():
        chunks = rag.processor.chunks_collection.count()
        summaries = rag.processor.summaries_collection.count()
        sources = {}
        if chunks > 0:
            result = rag.processor.chunks_collection.get(include=["metadatas"], limit=10000)
            for meta in result.get('metadatas', []):
                src = meta.get('source_file', 'Unknown')
                sources[src] = sources.get(src, 0) + 1
        return {
            "chunks": chunks,
            "summaries": summaries,
            "sources": sources,
            "model": rag.ollama_model
        }

    stats = await asyncio.to_thread(fetch_stats)
    text = format_stats_response(stats)
    await update.message.reply_text(text, parse_mode=constants.ParseMode.HTML)


@restricted
async def sources_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show indexed sources with clickable inline filter buttons."""
    user_id = update.effective_user.id
    current_filter = USER_SOURCE_FILTERS.get(user_id)
    rag = get_rag()
    
    sources = await asyncio.to_thread(rag.processor.get_indexed_sources)
    
    if not sources:
        await update.message.reply_text(
            "📚 <b>Documents</b>\n\nNo documents have been indexed yet. Send a <code>.pdf</code> file to begin!",
            parse_mode=constants.ParseMode.HTML
        )
        return

    filter_info = f"<code>{current_filter}</code>" if current_filter else "<i>None (Searching all documents)</i>"
    text = (
        f"📚 <b>Indexed Documents ({len(sources)})</b>\n\n"
        f"📌 <b>Active Filter:</b> {filter_info}\n\n"
        f"<i>Tap a document below to restrict your questions to it, or select 'All Documents' to search everything:</i>"
    )
    
    keyboard = build_sources_inline_keyboard(sources, current_filter)
    await update.message.reply_text(text, parse_mode=constants.ParseMode.HTML, reply_markup=keyboard)


@restricted
async def source_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text /source <name> or /source clear."""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        current = USER_SOURCE_FILTERS.get(user_id)
        if current:
            await update.message.reply_text(f"📌 Active source filter: <code>{current}</code>", parse_mode=constants.ParseMode.HTML)
        else:
            await update.message.reply_text("📌 No filter active (searching all documents). Use <code>/sources</code> to select one.", parse_mode=constants.ParseMode.HTML)
        return

    sub = " ".join(args).strip()
    if sub.lower() in ('clear', 'none', 'reset', 'all'):
        USER_SOURCE_FILTERS[user_id] = None
        await update.message.reply_text("✅ Source filter cleared. Searching across all documents.")
    else:
        USER_SOURCE_FILTERS[user_id] = sub
        await update.message.reply_text(f"✅ Source filter set to: <code>{sub}</code>", parse_mode=constants.ParseMode.HTML)


@restricted
async def clusters_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display hierarchical cluster summaries."""
    rag = get_rag()
    
    # Send quick status
    status_msg = await update.message.reply_text("⏳ Fetching cluster hierarchy...")
    
    df = await asyncio.to_thread(rag.processor.get_summary_tree)
    messages = format_clusters_response(df)
    
    await status_msg.delete()
    for msg in messages:
        await update.message.reply_text(msg, parse_mode=constants.ParseMode.HTML)


@restricted
async def cluster_run_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger hierarchical clustering on server."""
    rag = get_rag()
    status_msg = await update.message.reply_text("🧩 <b>Initiating Hierarchical Clustering...</b>\nRunning source-aware GMM clustering on your server.", parse_mode=constants.ParseMode.HTML)
    
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(send_typing_heartbeat(context.bot, update.effective_chat.id, stop_typing))
    
    try:
        history = await asyncio.to_thread(
            rag.processor.hierarchical_clustering,
            max_levels=config.clustering.max_levels,
            source_aware=True
        )
        stop_typing.set()
        await typing_task
        
        levels = len(history.get('levels', []))
        final_level = history.get('final_level', 0)
        
        await status_msg.edit_text(
            f"✅ <b>Hierarchical Clustering Complete!</b>\n\n"
            f"• Processed levels: {levels}\n"
            f"• Final hierarchy level: {final_level}\n\n"
            f"Use <code>/clusters</code> to view the generated summary tree.",
            parse_mode=constants.ParseMode.HTML
        )
    except Exception as e:
        stop_typing.set()
        await typing_task
        logger.error(f"Error during clustering: {e}")
        await status_msg.edit_text(f"❌ Error during clustering: <code>{e}</code>", parse_mode=constants.ParseMode.HTML)


@restricted
async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history."""
    rag = get_rag()
    rag.conversation_history = []
    await update.message.reply_text("🧹 <b>Conversation history cleared.</b> Prior questions won't influence follow-up responses.", parse_mode=constants.ParseMode.HTML)


@restricted
async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run environment health check."""
    rag = get_rag()
    
    def check_health():
        import requests
        ollama_ok = False
        ollama_models = []
        try:
            r = requests.get(f"{config.ollama.url}/api/tags", timeout=5)
            if r.status_code == 200:
                ollama_ok = True
                ollama_models = [m.get('name', '') for m in r.json().get('models', [])]
        except Exception:
            pass
            
        chroma_ok = False
        chunks_count = 0
        try:
            chunks_count = rag.processor.chunks_collection.count()
            chroma_ok = True
        except Exception:
            pass
            
        return ollama_ok, ollama_models, chroma_ok, chunks_count

    ollama_ok, models, chroma_ok, count = await asyncio.to_thread(check_health)
    
    ollama_icon = "✅" if ollama_ok else "❌"
    chroma_icon = "✅" if chroma_ok else "❌"
    
    health_text = (
        "🩺 <b>APEX Server Health Status</b>\n"
        "─────────────────────\n"
        f"{ollama_icon} <b>Ollama Server:</b> {'Online' if ollama_ok else 'Offline (Check laptop)'}\n"
        f"   • Active Model: <code>{config.ollama.generation_model}</code>\n"
        f"   • Installed: {len(models)} model(s)\n\n"
        f"{chroma_icon} <b>ChromaDB:</b> {'Online' if chroma_ok else 'Error'}\n"
        f"   • Path: <code>{config.chroma.path}</code>\n"
        f"   • Chunks: {count}\n"
    )
    await update.message.reply_text(health_text, parse_mode=constants.ParseMode.HTML)


# ================= INLINE CALLBACK HANDLER =================

@restricted
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle tap events on inline buttons."""
    query = update.callback_query
    await query.answer()
    
    data = query.data or ""
    user_id = update.effective_user.id
    
    if data.startswith("src_filter:"):
        selected = data[len("src_filter:"):]
        rag = get_rag()
        sources = await asyncio.to_thread(rag.processor.get_indexed_sources)
        
        if selected == "__all__":
            USER_SOURCE_FILTERS[user_id] = None
            msg_text = "✅ Filter cleared. Now searching across <b>all documents</b>."
        else:
            USER_SOURCE_FILTERS[user_id] = selected
            msg_text = f"✅ Filter active: <code>{selected}</code>"
            
        # Update the keyboard checkmarks
        keyboard = build_sources_inline_keyboard(sources, USER_SOURCE_FILTERS.get(user_id))
        
        filter_info = f"<code>{USER_SOURCE_FILTERS.get(user_id)}</code>" if USER_SOURCE_FILTERS.get(user_id) else "<i>None (Searching all documents)</i>"
        header = (
            f"📚 <b>Indexed Documents ({len(sources)})</b>\n\n"
            f"📌 <b>Active Filter:</b> {filter_info}\n\n"
            f"{msg_text}"
        )
        try:
            await query.edit_message_text(text=header, parse_mode=constants.ParseMode.HTML, reply_markup=keyboard)
        except Exception:
            pass


# ================= DOCUMENT UPLOAD HANDLER =================

@restricted
async def handle_pdf_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle PDF files sent directly from the phone."""
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("⚠️ Please send a valid <code>.pdf</code> document.", parse_mode=constants.ParseMode.HTML)
        return

    file_name = doc.file_name
    file_size_mb = (doc.file_size or 0) / (1024 * 1024)
    
    status_msg = await update.message.reply_text(
        f"📥 <b>Receiving PDF:</b> <code>{file_name}</code> ({file_size_mb:.1f} MB)\n"
        f"⏳ [1/3] Downloading file to local server...",
        parse_mode=constants.ParseMode.HTML
    )
    
    upload_dir = os.path.join(BASE_DIR, "incoming_pdfs")
    os.makedirs(upload_dir, exist_ok=True)
    local_path = os.path.join(upload_dir, file_name)
    
    try:
        # Download file from Telegram
        tg_file = await context.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(local_path)
        
        await status_msg.edit_text(
            f"📄 <b>Processing:</b> <code>{file_name}</code>\n"
            f"⚙️ [2/3] Extracting text & building token-bounded chunks...",
            parse_mode=constants.ParseMode.HTML
        )
        
        rag = get_rag()
        
        # Add document into ChromaDB in worker thread
        chunks_added = await asyncio.to_thread(
            rag.processor.add_document,
            file_path=local_path,
            document_name=file_name,
            force=False
        )
        
        if chunks_added == 0:
            await status_msg.edit_text(
                f"ℹ️ <b>Document Already Exists</b>\n\n"
                f"<code>{file_name}</code> is already indexed in the ChromaDB database with matching SHA-256 hash. No duplicate chunks were added.",
                parse_mode=constants.ParseMode.HTML
            )
            return

        # Check if auto-clustering is configured
        if config.telegram.auto_cluster_on_upload:
            await status_msg.edit_text(
                f"🧩 <b>Indexing Complete:</b> {chunks_added} chunks added.\n"
                f"⚙️ [3/3] Running Hierarchical GMM Clustering...",
                parse_mode=constants.ParseMode.HTML
            )
            await asyncio.to_thread(
                rag.processor.hierarchical_clustering,
                max_levels=config.clustering.max_levels,
                source_aware=True
            )
            await status_msg.edit_text(
                f"✅ <b>Successfully Ingested & Clustered!</b>\n\n"
                f"• Document: <code>{file_name}</code>\n"
                f"• Chunks: {chunks_added}\n"
                f"• Status: Searchable with hierarchical summaries ready.\n\n"
                f"You can now ask questions about this document!",
                parse_mode=constants.ParseMode.HTML
            )
        else:
            await status_msg.edit_text(
                f"✅ <b>Successfully Ingested!</b>\n\n"
                f"• Document: <code>{file_name}</code>\n"
                f"• Chunks: {chunks_added} indexed in ChromaDB\n\n"
                f"You can immediately ask questions about this document!\n"
                f"<i>(Run <code>/cluster_run</code> if you want to generate high-level cluster summaries)</i>",
                parse_mode=constants.ParseMode.HTML
            )
            
    except Exception as e:
        logger.error(f"Error processing uploaded PDF {file_name}: {e}")
        await status_msg.edit_text(
            f"❌ <b>Error processing PDF:</b>\n<code>{e}</code>",
            parse_mode=constants.ParseMode.HTML
        )


# ================= TEXT MESSAGE / RAG QUERY HANDLER =================

@restricted
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle questions or bottom keyboard button clicks."""
    text = (update.message.text or "").strip()
    if not text:
        return

    # Check for bottom keyboard button shortcuts
    if text == "📚 Documents":
        return await sources_command(update, context)
    elif text == "📊 Stats":
        return await stats_command(update, context)
    elif text == "🧩 Clusters":
        return await clusters_command(update, context)
    elif text == "🧹 Clear Context":
        return await clear_command(update, context)

    # Otherwise, it's a RAG question!
    user_id = update.effective_user.id
    source_filter = USER_SOURCE_FILTERS.get(user_id)
    chat_id = update.effective_chat.id
    
    filter_label = f" [Filter: <code>{source_filter}</code>]" if source_filter else ""
    logger.info(f"Query from user {user_id}: '{text[:60]}' {filter_label}")
    
    # Start typing heartbeat
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(send_typing_heartbeat(context.bot, chat_id, stop_typing))
    
    rag = get_rag()
    
    try:
        # Run query in worker thread
        result = await asyncio.to_thread(
            rag.query,
            question=text,
            source_filter=source_filter
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        result = {
            'question': text,
            'answer': f"An error occurred while generating your answer: {str(e)}",
            'sources': [],
            'context_chunks': 0,
            'response_time': 0.0
        }
    finally:
        stop_typing.set()
        await typing_task

    # Format into clean HTML
    formatted_html = format_rag_response(result)
    
    # Split if exceeds Telegram's limit
    chunks = split_message(formatted_html, max_length=config.telegram.max_message_length)
    
    for chunk in chunks:
        await update.message.reply_text(
            chunk,
            parse_mode=constants.ParseMode.HTML,
            reply_markup=get_main_keyboard()
        )


# ================= SERVER BOOTSTRAP =================

def main():
    """Main Telegram bot server entrypoint."""
    setup_logger(level=config.log_level)
    
    print("=" * 70)
    print("🤖 APEX Mobile Telegram Interface Server")
    print("=" * 70)
    
    token = config.telegram.token
    if not token or token == "your_bot_token_here":
        print("\n❌ Error: TELEGRAM_BOT_TOKEN is not configured!")
        print("   1. Open Telegram and search for @BotFather")
        print("   2. Send /newbot and copy the API token")
        print("   3. Add the token to .env or set environment variable:")
        print("      TELEGRAM_BOT_TOKEN=your_token_here")
        print("   4. (Recommended) Add your User ID from @userinfobot:")
        print("      TELEGRAM_ALLOWED_USERS=your_user_id")
        sys.exit(1)
        
    print(f"📡 Model: {config.ollama.generation_model}")
    print(f"📚 Vector DB: {config.chroma.path}")
    if config.telegram.allowed_user_ids:
        print(f"🔒 Whitelist: {len(config.telegram.allowed_user_ids)} authorized user(s)")
    else:
        print("⚠️ Warning: TELEGRAM_ALLOWED_USERS is empty. Anyone who messages this bot can use it.")
    
    print("\nStarting Telegram Polling Engine (press Ctrl+C to stop)...")
    
    # Build application
    app = ApplicationBuilder().token(token).build()
    
    # Add Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("sources", sources_command))
    app.add_handler(CommandHandler("source", source_command))
    app.add_handler(CommandHandler("clusters", clusters_command))
    app.add_handler(CommandHandler("cluster_run", cluster_run_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("health", health_command))
    
    # Add Callback Query Handler (inline buttons)
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    
    # Add PDF Document Upload Handler
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf_upload))
    
    # Add Text Message Handler (queries & keyboard buttons)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    # Run polling loop
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

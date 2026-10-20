"""
APEX RAG Web Application Server
Modern AI Chat UI with session history, multi-document filtering,
expandable evidence drawers, and conversational persistence.
"""

import os
import sys
import json
import time
import uuid
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import asyncio
from pydantic import BaseModel
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from face import RAGInterface
from modules.config import config
from modules.logger import setup_logger, get_logger

logger = get_logger("webapp")

# Directory setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CHATS_DIR = os.path.join(BASE_DIR, "chats")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="APEX AI Chat Interface",
    description="Conversational UI for Hierarchical Clustering RAG over local documents",
    version="1.0.0"
)

# Global RAG Interface Instance
rag_interface: Optional[RAGInterface] = None


def get_rag() -> RAGInterface:
    """Lazy initialize and return singleton RAGInterface."""
    global rag_interface
    if rag_interface is None:
        logger.info("Initializing RAG Interface for Web UI...")
        rag_interface = RAGInterface(
            ollama_model=config.ollama.generation_model,
            ollama_embedding_model=config.ollama.embedding_model,
            chroma_path=config.chroma.path,
            max_context_chunks=config.retrieval.max_context_chunks,
            temperature=config.retrieval.temperature,
            min_score=config.retrieval.min_relevance_score,
            verbose=False
        )
    return rag_interface


# ================= DATA MODELS =================

class CreateChatRequest(BaseModel):
    title: Optional[str] = "New Conversation"
    source_filter: Optional[str] = None


class UpdateChatRequest(BaseModel):
    title: str


class ChatQueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    source_filter: Optional[str] = None
    max_chunks: Optional[int] = None
    min_score: Optional[float] = None
    temperature: Optional[float] = None
    include_summaries: Optional[bool] = None


# ================= SESSION PERSISTENCE HELPERS =================

def load_all_sessions_meta() -> List[Dict[str, Any]]:
    """Scan chats directory and return summary metadata sorted newest first."""
    sessions = []
    if not os.path.exists(CHATS_DIR):
        return []

    for fname in os.listdir(CHATS_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(CHATS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
                    preview = messages[-1]["content"][:80] if messages else "Empty conversation"
                    sessions.append({
                        "id": data.get("id", fname[:-5]),
                        "title": data.get("title", "Untitled Chat"),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "source_filter": data.get("source_filter"),
                        "message_count": len(messages),
                        "preview": preview
                    })
            except Exception as e:
                logger.warning(f"Error reading session file {fname}: {e}")

    # Sort descending by updated_at
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def load_session(chat_id: str) -> Optional[Dict[str, Any]]:
    """Load a full conversation session from disk."""
    fpath = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading chat {chat_id}: {e}")
        return None


def save_session(session_data: Dict[str, Any]):
    """Persist conversation session to disk."""
    chat_id = session_data.get("id")
    if not chat_id:
        return
    session_data["updated_at"] = datetime.now().isoformat()
    fpath = os.path.join(CHATS_DIR, f"{chat_id}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)


# ================= API ROUTES =================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main single-page chat application."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>Error: templates/index.html not found</h2>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/info")
async def get_system_info():
    """Return database statistics, available sources, and active models."""
    rag = get_rag()
    try:
        sources = rag.processor.get_indexed_sources()
        total_chunks = rag.processor.chunks_collection.count()
        total_summaries = rag.processor.summaries_collection.count()
        conv_chunks = rag.processor.count_conversation_chunks()
        conv_summaries = rag.processor.count_conversation_summaries()
        return {
            "success": True,
            "model": rag.ollama_model,
            "embedding_model": config.ollama.embedding_model,
            "total_chunks": total_chunks,
            "total_summaries": total_summaries,
            "conv_chunks": conv_chunks,
            "conv_summaries": conv_summaries,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error fetching system info: {e}")
        return {"success": False, "error": str(e), "sources": []}


@app.get("/api/chats")
async def list_chats():
    """List all previous chat sessions."""
    chats = load_all_sessions_meta()
    return {"success": True, "chats": chats}


@app.post("/api/chats")
async def create_chat(req: CreateChatRequest):
    """Create a new chat conversation."""
    chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    now = datetime.now().isoformat()
    session = {
        "id": chat_id,
        "title": req.title or "New Conversation",
        "created_at": now,
        "updated_at": now,
        "source_filter": req.source_filter,
        "messages": []
    }
    save_session(session)
    return {"success": True, "chat": session}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Load a specific chat conversation with its full message history."""
    session = load_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"success": True, "chat": session}


@app.put("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, req: UpdateChatRequest):
    """Rename a conversation title."""
    session = load_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    session["title"] = req.title.strip() or "Untitled Chat"
    save_session(session)
    return {"success": True, "chat": session}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a conversation from disk."""
    fpath = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(fpath):
        os.remove(fpath)
        return {"success": True, "deleted": chat_id}
    raise HTTPException(status_code=404, detail="Chat session not found")


@app.post("/api/chat")
async def process_chat_message(req: ChatQueryRequest):
    """
    Main RAG chat endpoint:
      1. Loads or creates conversation session.
      2. Records user question.
      3. Executes separated hybrid retrieval + reranking.
      4. Synthesizes evidence-grounded response via native Ollama.
      5. Persists session with retrieved chunks, citations, and metadata.
    """
    question = req.question.strip()
    if not question:
        return {"success": False, "message": "Empty question provided."}

    # 1. Retrieve or initialize chat session
    chat_id = req.chat_id
    session = None
    if chat_id:
        session = load_session(chat_id)
    
    if not session:
        chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()
        session = {
            "id": chat_id,
            "title": "New Conversation",
            "created_at": now,
            "updated_at": now,
            "source_filter": req.source_filter,
            "messages": []
        }

    # Append user message
    user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    session["messages"].append({
        "id": user_msg_id,
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat()
    })

    # 2. Configure RAG parameters for this query
    rag = get_rag()
    original_chunks = rag.max_context_chunks
    original_min_score = rag.min_score
    original_temp = rag.temperature
    original_summaries = rag.include_summaries

    if req.max_chunks is not None:
        rag.max_context_chunks = req.max_chunks
    if req.min_score is not None:
        rag.min_score = req.min_score
    if req.temperature is not None:
        rag.temperature = req.temperature
    if req.include_summaries is not None:
        rag.include_summaries = req.include_summaries

    effective_filter = req.source_filter or session.get("source_filter")

    start_time = time.time()
    try:
        # 3. Retrieve Context
        context = rag.retrieve_context(question, source_filter=effective_filter)
        if not context.get("success"):
            answer = f"Could not retrieve context: {context.get('message', 'Unknown error')}"
            assistant_msg = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat(),
                "sources": [],
                "chunks": [],
                "response_time": time.time() - start_time
            }
        else:
            # 4. Generate Response
            prompt = rag.build_prompt(context)
            gen_res = rag.generate_response(prompt)
            elapsed = time.time() - start_time

            if gen_res.get("success"):
                answer = gen_res.get("answer", "")
                tokens = gen_res.get("tokens", len(answer.split()))
            else:
                answer = f"Generation error: {gen_res.get('error', 'Ollama service error')}"
                tokens = 0

            assistant_msg = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat(),
                "sources": context.get("sources", []),
                "chunks": context.get("chunks", []),
                "scores": context.get("scores", []),
                "metadatas": context.get("metadatas", []),
                "cluster_summaries": context.get("cluster_summaries", []),
                "conversation_memories": context.get("conversation_memories", []),
                "conversation_cluster_summaries": context.get("conversation_cluster_summaries", []),
                "response_time": elapsed,
                "tokens": tokens,
                "retrieval_time": context.get("retrieval_time", 0.0),
                "low_confidence": context.get("low_confidence", False)
            }

        # Append assistant message
        session["messages"].append(assistant_msg)

        # Auto-index conversation turn into episodic memory
        try:
            turn_idx = len([m for m in session["messages"] if m.get("role") == "user"])
            rag.processor.add_conversation_turn(
                chat_id=chat_id,
                user_text=question,
                assistant_text=answer,
                turn_index=turn_idx,
                chat_title=session.get("title", "Conversation")
            )
        except Exception as mem_err:
            logger.warning(f"Failed to auto-index conversation memory turn: {mem_err}")

        # Auto-generate title if default
        if session.get("title") in ("New Conversation", "Untitled Chat"):
            words = question.split()
            session["title"] = " ".join(words[:6]) + ("..." if len(words) > 6 else "")

        # Save session
        save_session(session)

        return {
            "success": True,
            "chat_id": chat_id,
            "chat_title": session["title"],
            "message": assistant_msg
        }

    except Exception as e:
        logger.error(f"Error in chat processing: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Server Error: {str(e)}"
        }
    finally:
        # Restore original RAG parameters
        rag.max_context_chunks = original_chunks
        rag.min_score = original_min_score
        rag.temperature = original_temp
        rag.include_summaries = original_summaries



@app.post("/api/chat/stream")
async def process_chat_message_stream(req: ChatQueryRequest):
    """
    Streaming RAG chat endpoint using Server-Sent Events (SSE) (Item 2.2):
      - Sends retrieval metadata first (sources, chunks, memories, retrieval_time, is_chitchat).
      - Streams tokens in real-time as they arrive from Ollama.
      - Saves session and indexes conversation turn asynchronously on completion.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question provided.")

    chat_id = req.chat_id
    session = None
    if chat_id:
        session = load_session(chat_id)

    if not session:
        chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()
        session = {
            "id": chat_id,
            "title": "New Conversation",
            "created_at": now,
            "updated_at": now,
            "source_filter": req.source_filter,
            "messages": []
        }

    user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    session["messages"].append({
        "id": user_msg_id,
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat()
    })

    rag = get_rag()
    original_chunks = rag.max_context_chunks
    original_min_score = rag.min_score
    original_temp = rag.temperature
    original_summaries = rag.include_summaries

    if req.max_chunks is not None:
        rag.max_context_chunks = req.max_chunks
    if req.min_score is not None:
        rag.min_score = req.min_score
    if req.temperature is not None:
        rag.temperature = req.temperature
    if req.include_summaries is not None:
        rag.include_summaries = req.include_summaries

    effective_filter = req.source_filter or session.get("source_filter")

    async def event_generator():
        start_time = time.time()
        try:
            # 1. Retrieve Context
            context = await asyncio.to_thread(rag.retrieve_context, question, effective_filter)

            # Send initial metadata event
            meta_event = {
                "type": "metadata",
                "chat_id": chat_id,
                "chat_title": session["title"],
                "sources": context.get("sources", []),
                "chunks": context.get("chunks", []),
                "scores": context.get("scores", []),
                "metadatas": context.get("metadatas", []),
                "cluster_summaries": context.get("cluster_summaries", []),
                "conversation_memories": context.get("conversation_memories", []),
                "conversation_cluster_summaries": context.get("conversation_cluster_summaries", []),
                "retrieval_time": context.get("retrieval_time", 0.0),
                "is_chitchat": context.get("is_chitchat", False),
                "low_confidence": context.get("low_confidence", False)
            }
            yield f"data: {json.dumps(meta_event)}\n\n"

            # 2. Build prompt
            prompt = rag.build_prompt(context)

            # 3. Stream tokens
            full_answer = []
            token_count = 0

            for token in rag.generate_response_stream(prompt):
                full_answer.append(token)
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            answer_text = "".join(full_answer).strip()
            elapsed = time.time() - start_time

            # 4. Save assistant message
            assistant_msg = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": answer_text,
                "timestamp": datetime.now().isoformat(),
                "sources": context.get("sources", []),
                "chunks": context.get("chunks", []),
                "scores": context.get("scores", []),
                "metadatas": context.get("metadatas", []),
                "cluster_summaries": context.get("cluster_summaries", []),
                "conversation_memories": context.get("conversation_memories", []),
                "conversation_cluster_summaries": context.get("conversation_cluster_summaries", []),
                "response_time": elapsed,
                "retrieval_time": context.get("retrieval_time", 0.0),
                "tokens": token_count,
                "low_confidence": context.get("low_confidence", False)
            }
            session["messages"].append(assistant_msg)

            # Auto-title
            if session.get("title") in ("New Conversation", "Untitled Chat"):
                words = question.split()
                session["title"] = " ".join(words[:6]) + ("..." if len(words) > 6 else "")

            save_session(session)

            # Index conversation turn
            try:
                turn_idx = len([m for m in session["messages"] if m.get("role") == "user"])
                rag.processor.add_conversation_turn(
                    chat_id=chat_id,
                    user_text=question,
                    assistant_text=answer_text,
                    turn_index=turn_idx,
                    chat_title=session.get("title", "Conversation")
                )
            except Exception as mem_err:
                logger.warning(f"Error auto-indexing conversation turn: {mem_err}")

            # Send done event
            done_event = {
                "type": "done",
                "chat_id": chat_id,
                "chat_title": session["title"],
                "response_time": elapsed,
                "tokens": token_count
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            rag.max_context_chunks = original_chunks
            rag.min_score = original_min_score
            rag.temperature = original_temp
            rag.include_summaries = original_summaries

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ================= CONVERSATION CLUSTERING ROUTES =================

@app.post("/api/cluster-conversations")
async def cluster_conversations_endpoint():
    """
    Trigger hierarchical clustering over all stored conversation turns.
    Clusters chat exchanges, runs silhouette quality gating, and stores topic summaries.
    """
    rag = get_rag()
    try:
        result = rag.processor.cluster_conversations()
        conv_chunks = rag.processor.count_conversation_chunks()
        conv_summaries = rag.processor.count_conversation_summaries()
        return {
            "success": result.get("success", True),
            "message": result.get("message", "Clustering completed successfully"),
            "conv_chunks": conv_chunks,
            "conv_summaries": conv_summaries,
            "levels": result.get("levels", [])
        }
    except Exception as e:
        logger.error(f"Error clustering conversations: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.get("/api/conversation-clusters")
async def get_conversation_clusters_endpoint():
    """Return summary tree of conversational clusters organized by hierarchy level."""
    rag = get_rag()
    try:
        tree_df = rag.processor.get_conversation_summary_tree()
        records = tree_df.to_dict(orient="records") if not tree_df.empty else []
        return {
            "success": True,
            "count": len(records),
            "clusters": records
        }
    except Exception as e:
        logger.error(f"Error fetching conversation clusters: {e}", exc_info=True)
        return {"success": False, "error": str(e), "clusters": []}


# ================= ENTRYPOINT =================

def main():
    parser = argparse.ArgumentParser(description="APEX RAG Web Application Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open browser on launch")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    args = parser.parse_args()

    setup_logger(level=args.log_level)

    print("\n" + "=" * 70)
    print("🚀 APEX AI — Hierarchical RAG Chat Web Interface")
    print("=" * 70)
    print(f"🌐 Running locally at: http://{args.host}:{args.port}")
    print(f"📚 ChromaDB Store:    {config.chroma.path}")
    print(f"🤖 Generation Model:  {config.ollama.generation_model}")
    print(f"🔤 Embedding Model:   {config.ollama.embedding_model}")
    print(f"💬 Chat History:      {CHATS_DIR}")
    print("=" * 70)
    print("👉 Open your browser at the URL above to start chatting!\n")

    if args.open_browser:
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

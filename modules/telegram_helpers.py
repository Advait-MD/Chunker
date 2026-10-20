"""
Telegram helper utilities for APEX RAG:
Message splitting, HTML safe formatting, and response formatters.
"""

import html
import re
from typing import List, Dict, Any, Optional
import pandas as pd


def split_message(text: str, max_length: int = 4000) -> List[str]:
    """
    Split long text into chunks that fit within Telegram's 4096-character limit.
    Splits preferentially at paragraph breaks, then line breaks, then sentence boundaries.
    """
    if not text:
        return [""]
    
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = []
    current_length = 0

    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para_len = len(para) + 2  # account for \n\n
        if current_length + para_len <= max_length:
            current_chunk.append(para)
            current_length += para_len
        else:
            # If current chunk has content, flush it
            if current_chunk:
                chunks.append("\n\n".join(current_chunk).strip())
                current_chunk = []
                current_length = 0

            # If the paragraph itself is longer than max_length, split by lines or sentences
            if len(para) > max_length:
                lines = para.split("\n")
                for line in lines:
                    line_len = len(line) + 1
                    if current_length + line_len <= max_length:
                        current_chunk.append(line)
                        current_length += line_len
                    else:
                        if current_chunk:
                            chunks.append("\n".join(current_chunk).strip())
                            current_chunk = []
                            current_length = 0
                        
                        # If a single line is still too long, split by words
                        if len(line) > max_length:
                            words = line.split(" ")
                            for word in words:
                                word_len = len(word) + 1
                                if current_length + word_len <= max_length:
                                    current_chunk.append(word)
                                    current_length += word_len
                                else:
                                    if current_chunk:
                                        chunks.append(" ".join(current_chunk).strip())
                                        current_chunk = []
                                        current_length = 0
                                    current_chunk.append(word)
                                    current_length = len(word)
                        else:
                            current_chunk.append(line)
                            current_length = len(line)
            else:
                current_chunk.append(para)
                current_length = len(para)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return [c for c in chunks if c.strip()]


def format_rag_response(result: Dict[str, Any]) -> str:
    """
    Format RAG query result dictionary into clean, mobile-friendly HTML.
    """
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    response_time = result.get("response_time", 0.0)
    tokens = result.get("tokens_generated", 0)
    low_confidence = result.get("low_confidence", False)
    
    parts = []
    
    if low_confidence:
        parts.append("⚠️ <b>Notice:</b> Low confidence score (&lt;25%). The retrieved context may only partially address this query.\n")
    
    # Escape HTML in the answer text, but keep readable formatting
    safe_answer = html.escape(answer)
    parts.append(safe_answer)
    
    parts.append("\n\n─────────────────────")
    
    # Add source citations
    if sources:
        parts.append(f"📚 <b>Sources Cited:</b> {', '.join([f'<code>{html.escape(s)}</code>' for s in sources])}")
    else:
        parts.append("📚 <b>Sources:</b> None directly cited")
        
    chunks_count = result.get("context_chunks", 0)
    parts.append(f"⏱️ <i>{response_time:.2f}s | {chunks_count} context chunk(s) | {tokens} tokens</i>")
    
    return "\n".join(parts)


def format_clusters_response(df: pd.DataFrame) -> List[str]:
    """
    Format the hierarchical cluster tree into readable cards.
    Returns a list of messages (split if long).
    """
    if df.empty:
        return ["📊 <b>Cluster Hierarchy</b>\n\nNo cluster summaries found in the database. Run <code>/cluster_run</code> to generate them."]

    lines = ["📊 <b>Hierarchical Cluster Summaries</b>\n"]
    
    for level in sorted(df['level'].unique()):
        level_df = df[df['level'] == level]
        lines.append(f"\n<b>Level {level} ({len(level_df)} clusters):</b>")
        lines.append("─────────────────────")
        
        for _, row in level_df.iterrows():
            cid = row.get('cluster_id', '?')
            chunks = row.get('chunks_count', 0)
            sil = row.get('silhouette_score', 0.0)
            preview = html.escape(str(row.get('summary_preview', ''))[:160])
            src = html.escape(str(row.get('source_file', 'multi-doc')))
            
            lines.append(f"• <b>Cluster {cid}</b> [{src}]: {chunks} chunks | Sil: <code>{sil:.3f}</code>")
            lines.append(f"  <i>\"{preview}...\"</i>\n")
            
    full_text = "\n".join(lines)
    return split_message(full_text, max_length=3800)


def format_stats_response(stats: Dict[str, Any]) -> str:
    """Format system database statistics into an HTML card."""
    chunks = stats.get("chunks", 0)
    summaries = stats.get("summaries", 0)
    sources = stats.get("sources", {})
    model = html.escape(stats.get("model", "Unknown"))
    
    lines = [
        "📊 <b>APEX System & Database Statistics</b>",
        "─────────────────────",
        f"🤖 <b>Active Model:</b> <code>{model}</code>",
        f"📦 <b>Total Indexed Chunks:</b> {chunks}",
        f"🧩 <b>Hierarchical Summaries:</b> {summaries}",
        f"📚 <b>Indexed Documents:</b> {len(sources)}",
        ""
    ]
    
    if sources:
        lines.append("<b>Document Breakdown:</b>")
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  • <code>{html.escape(src)}</code>: {count} chunks")
    else:
        lines.append("<i>No documents currently indexed. Send a PDF to begin!</i>")
        
    return "\n".join(lines)

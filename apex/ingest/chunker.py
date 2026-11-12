"""Sentence-boundary chunking (~400 tokens, ~50 overlap), reusing modules.document_loader."""

from __future__ import annotations

from typing import Dict, List

from modules.document_loader import count_tokens, create_chunks_from_pages

__all__ = ["chunk_pages", "count_tokens"]


def chunk_pages(pages: List[Dict], target_tokens: int, overlap_tokens: int) -> List[Dict]:
    """Chunk page units; chunks never cross a page/section boundary.

    Each chunk carries page_number, section, start_char/end_char (offsets inside its page) and token_count.
    """
    section_by_page = {p["page_number"]: p.get("section") for p in pages}
    chunks = create_chunks_from_pages(pages, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    for c in chunks:
        c["section"] = section_by_page.get(c["page_number"])
    return chunks

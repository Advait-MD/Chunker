"""One-command ingestion: parse → chunk → embed → upsert leaves. Cache-aware and idempotent."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from apex.index.chroma_store import Node, NodeStore
from apex.index.embedder import Embedder
from apex.ingest.chunker import chunk_pages
from apex.ingest.parsers import file_sha256, parse_file


def leaf_build_hash(cfg: Dict[str, Any]) -> str:
    """Hash of everything that changes leaf nodes: chunker params, embedder and prefix."""
    material = {"chunk": [cfg["ingest"]["chunk_target_tokens"], cfg["ingest"]["chunk_overlap_tokens"]],
                "embedder": cfg["models"]["embedder"], "doc_prefix": cfg["embedding"]["doc_prefix"],
                "normalize": cfg["embedding"]["normalize"], "schema": 1}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def discover(folder: str | Path, extensions: List[str]) -> List[Path]:
    root = Path(folder)
    if root.is_file():
        return [root]
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def ingest_path(folder: str | Path, cfg: Dict[str, Any], store: NodeStore, embedder: Embedder,
                force: bool = False, log=print) -> Dict[str, Any]:
    build_hash = leaf_build_hash(cfg)
    files = discover(folder, cfg["ingest"]["extensions"])
    if not files:
        raise FileNotFoundError(f"No ingestible files under {folder} (extensions {cfg['ingest']['extensions']})")
    report = {"files": [], "build_hash": build_hash, "added_nodes": 0, "skipped_files": 0}
    for path in files:
        doc_id = file_sha256(path)[:16]
        if not force and store.has_build(doc_id, build_hash):
            report["skipped_files"] += 1
            report["files"].append({"path": str(path), "doc_id": doc_id, "status": "cached"})
            log(f"  = {path.name}: unchanged, reusing index")
            continue
        store.delete_doc(doc_id)  # stale build for this content, if any
        store.delete_source(path.name, keep_doc_id=doc_id)  # previous versions of this file
        pages = parse_file(path)
        chunks = chunk_pages(pages, cfg["ingest"]["chunk_target_tokens"], cfg["ingest"]["chunk_overlap_tokens"])
        if not chunks:
            report["files"].append({"path": str(path), "doc_id": doc_id, "status": "empty"})
            log(f"  ! {path.name}: no text extracted")
            continue
        now = datetime.now(timezone.utc).isoformat()
        nodes = [Node(id=f"{doc_id}:L0:{c['chunk_index']:06d}", text=c["text"], level=0, doc_id=doc_id,
                      source_path=path.name, page=c["page_number"], section=c.get("section"),
                      char_span=[c["start_char"], c["end_char"]], token_count=c["token_count"],
                      created_at=now, build_hash=build_hash) for c in chunks]
        log(f"  + {path.name}: {len(pages)} pages → {len(nodes)} chunks, embedding…")
        embs = embedder.embed_documents([n.text for n in nodes])
        store.upsert(nodes, embs)
        report["added_nodes"] += len(nodes)
        report["files"].append({"path": str(path), "doc_id": doc_id, "status": "ingested", "chunks": len(nodes)})
    return report

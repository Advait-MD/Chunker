"""ChromaDB node store: one record per tree node (spec §2.3 node schema).

Follows the teammate's `modules.vector_store.ChromaStore` conventions (PersistentClient,
deterministic ids, file-hash dedup, batched upserts) but stores explicit embeddings computed
through the logged wrapper, instead of Chroma's hard-wired MiniLM embedding function.
List-valued fields (parent_ids, child_ids, char_span) are JSON strings in Chroma metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import chromadb
import numpy as np

from apex.config import resolve_path

_LIST_FIELDS = ("parent_ids", "child_ids", "char_span")


@dataclass
class Node:
    id: str
    text: str
    level: int
    doc_id: str
    source_path: str
    page: Optional[int] = None
    section: Optional[str] = None
    char_span: Optional[List[int]] = None
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    token_count: int = 0
    created_at: str = ""
    build_hash: str = ""

    def metadata(self) -> Dict[str, Any]:
        md = {
            "level": self.level, "doc_id": self.doc_id, "source_path": self.source_path,
            "page": self.page if self.page is not None else -1, "section": self.section or "",
            "token_count": self.token_count, "created_at": self.created_at, "build_hash": self.build_hash,
        }
        for f in _LIST_FIELDS:
            md[f] = json.dumps(getattr(self, f) or [])
        return md

    @classmethod
    def from_record(cls, node_id: str, text: str, md: Dict[str, Any]) -> "Node":
        lists = {f: json.loads(md.get(f) or "[]") for f in _LIST_FIELDS}
        page = md.get("page", -1)
        return cls(id=node_id, text=text, level=int(md.get("level", 0)), doc_id=md.get("doc_id", ""),
                   source_path=md.get("source_path", ""), page=None if page == -1 else page,
                   section=md.get("section") or None, char_span=lists["char_span"] or None,
                   parent_ids=lists["parent_ids"], child_ids=lists["child_ids"],
                   token_count=int(md.get("token_count", 0)), created_at=md.get("created_at", ""),
                   build_hash=md.get("build_hash", ""))


@dataclass
class Hit:
    node: Node
    score: float  # final ranking score: cosine similarity for dense, RRF/fused or reranker score for hybrid
    rank: int
    detail: Dict[str, Any] = field(default_factory=dict)  # per-stage scores and ranks (hybrid), for traces


class NodeStore:
    def __init__(self, cfg: Dict[str, Any], path: Optional[str] = None):
        s = cfg["store"]
        self.path = resolve_path(path or s["path"])
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.space = s.get("space", "cosine")
        if self.space != "cosine":
            raise ValueError("NodeStore scores assume cosine space; set store.space: cosine")
        self.col = self.client.get_or_create_collection(
            name=s["collection"], embedding_function=None,
            configuration={"hnsw": {"space": self.space}},
        )

    def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        if where is None:
            return self.col.count()
        return len(self.col.get(where=where, include=[])["ids"])

    def has_build(self, doc_id: str, build_hash: str) -> bool:
        res = self.col.get(where={"$and": [{"doc_id": doc_id}, {"build_hash": build_hash}]}, limit=1, include=[])
        return bool(res["ids"])

    def delete_doc(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def delete_source(self, source_path: str, keep_doc_id: Optional[str] = None) -> None:
        where: Dict[str, Any] = {"source_path": source_path}
        if keep_doc_id:
            where = {"$and": [where, {"doc_id": {"$ne": keep_doc_id}}]}
        self.col.delete(where=where)

    def upsert(self, nodes: List[Node], embeddings: np.ndarray, batch_size: int = 500) -> None:
        if len(nodes) != len(embeddings):
            raise ValueError(f"{len(nodes)} nodes but {len(embeddings)} embeddings")
        for i in range(0, len(nodes), batch_size):
            part = nodes[i:i + batch_size]
            self.col.upsert(ids=[n.id for n in part], documents=[n.text for n in part],
                            metadatas=[n.metadata() for n in part],
                            embeddings=np.asarray(embeddings[i:i + batch_size], dtype=np.float32).tolist())

    def query(self, embedding: np.ndarray, k: int, where: Optional[Dict[str, Any]] = None) -> List[Hit]:
        res = self.col.query(query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()], n_results=k,
                             where=where, include=["documents", "metadatas", "distances"])
        hits = []
        for r, (nid, doc, md, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                     res["metadatas"][0], res["distances"][0])):
            hits.append(Hit(node=Node.from_record(nid, doc, md), score=1.0 - float(dist), rank=r))
        return hits

    def get(self, ids: Iterable[str]) -> Dict[str, Node]:
        ids = list(ids)
        if not ids:
            return {}
        res = self.col.get(ids=ids, include=["documents", "metadatas"])
        return {nid: Node.from_record(nid, doc, md) for nid, doc, md in zip(res["ids"], res["documents"], res["metadatas"])}

    def all_nodes(self, where: Optional[Dict[str, Any]] = None, batch_size: int = 2000):
        """Every node matching `where`, with its stored embedding, in a stable (id-sorted) order.

        The source of truth for the BM25 and FAISS indexes, which are derived views keyed by node id.
        """
        ids, docs, mds, embs = [], [], [], []
        offset = 0
        while True:
            res = self.col.get(where=where, include=["documents", "metadatas", "embeddings"],
                               limit=batch_size, offset=offset)
            if not res["ids"]:
                break
            ids += res["ids"]; docs += res["documents"]; mds += res["metadatas"]; embs += list(res["embeddings"])
            offset += len(res["ids"])
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        nodes = [Node.from_record(ids[i], docs[i], mds[i]) for i in order]
        vecs = np.asarray([embs[i] for i in order], dtype=np.float32).reshape(len(order), -1)
        return nodes, vecs

    def sources(self) -> Dict[str, int]:
        res = self.col.get(where={"level": 0}, include=["metadatas"])
        out: Dict[str, int] = {}
        for md in res["metadatas"]:
            out[md["source_path"]] = out.get(md["source_path"], 0) + 1
        return out

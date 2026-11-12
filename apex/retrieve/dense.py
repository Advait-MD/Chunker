"""Dense retrieval over Chroma (HNSW). The M1 naive baseline; IVF arrives in M3."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from apex.index.chroma_store import Hit, NodeStore
from apex.index.embedder import Embedder


class DenseRetriever:
    def __init__(self, cfg: Dict[str, Any], store: NodeStore, embedder: Embedder):
        r = cfg["retrieval"]
        self.store, self.embedder = store, embedder
        self.top_k = int(r["top_k"])
        self.levels = list(r.get("levels", [0]))

    def _where(self, source: Optional[str]) -> Optional[Dict[str, Any]]:
        clauses: List[Dict[str, Any]] = []
        if self.levels:
            clauses.append({"level": {"$in": self.levels}})
        if source:
            clauses.append({"source_path": source})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def retrieve(self, query: str, k: Optional[int] = None, source: Optional[str] = None) -> List[Hit]:
        emb = self.embedder.embed_query(query)
        return self.store.query(emb, k or self.top_k, where=self._where(source))

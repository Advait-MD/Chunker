"""Minimal FastAPI endpoint.  Run: python -m uvicorn apex.api.server:app --port 8000"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from apex.pipeline import build_pipeline

app = FastAPI(title="APEX", version="0.3.0")
_pipe = None
_lock = threading.Lock()  # one query at a time: a single local GPU/CPU model


def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = build_pipeline(os.getenv("APEX_CONFIG"))
    return _pipe


def set_pipeline(pipe) -> None:
    """Test hook."""
    global _pipe
    _pipe = pipe


class QueryRequest(BaseModel):
    query: str
    source: Optional[str] = None


class Citation(BaseModel):
    cite_id: int
    node_id: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    char_span: Optional[List[int]] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: Optional[float]
    abstained: bool
    trace_id: str
    route: Optional[str]
    totals: Dict[str, Any]
    uncited_sentences: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    with _lock:
        res = get_pipeline().answer(req.query, source=req.source)
    return QueryResponse(**{k: res[k] for k in QueryResponse.model_fields})

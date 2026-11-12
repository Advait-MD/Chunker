"""Per-query trace: route, tier, states visited, ids, LLM calls, tokens, per-stage latency."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

from apex.config import REPO_ROOT, resolve_path


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (sha or "unknown") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@dataclass
class Trace:
    query: str
    config_hash: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    route: Optional[str] = None
    routing_tier: Optional[str] = None
    state: str = "RECEIVED"
    states: List[Dict[str, Any]] = field(default_factory=list)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retrieved_ids: List[str] = field(default_factory=list)
    filtered_ids: List[str] = field(default_factory=list)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    git_sha: str = field(default_factory=git_sha)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def __post_init__(self):
        self.states.append({"state": self.state, "t_ms": 0.0})

    def enter(self, state: str) -> None:
        self.state = state
        self.states.append({"state": state, "t_ms": round((time.perf_counter() - self._t0) * 1000, 2)})

    @contextmanager
    def stage(self, name: str, **info):
        t = time.perf_counter()
        rec = dict(info)
        self.stages[name] = rec
        try:
            yield rec
        finally:
            rec["latency_ms"] = round((time.perf_counter() - t) * 1000, 2)

    def record_llm(self, record: Dict[str, Any]) -> None:
        self.llm_calls.append(record)

    def totals(self) -> Dict[str, Any]:
        gen = [c for c in self.llm_calls if c.get("kind") == "generate"]
        return {
            "llm_calls": len(gen),
            "llm_calls_uncached": sum(1 for c in gen if not c.get("cache_hit")),
            "embed_calls": sum(1 for c in self.llm_calls if c.get("kind") == "embed"),
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in gen),
            "completion_tokens": sum(c.get("completion_tokens", 0) for c in gen),
            "total_ms": round((time.perf_counter() - self._t0) * 1000, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id, "created_at": self.created_at, "query": self.query,
            "route": self.route, "routing_tier": self.routing_tier,
            "states": self.states, "stages": self.stages,
            "retrieved_ids": self.retrieved_ids, "filtered_ids": self.filtered_ids,
            "llm_calls": self.llm_calls, "totals": self.totals(),
            "config_hash": self.config_hash, "git_sha": self.git_sha, "extra": self.extra,
        }


class Tracer:
    def __init__(self, cfg: Dict[str, Any]):
        self.dir = resolve_path(cfg["trace"]["dir"])
        self.config_hash = cfg.get("_meta", {}).get("config_hash", "unknown")

    def new(self, query: str) -> Trace:
        return Trace(query=query, config_hash=self.config_hash)

    def save(self, trace: Trace) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{trace.trace_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

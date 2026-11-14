"""The single logged wrapper for every Ollama call (spec rule 2 and 3).

Every generate/embed call goes through `OllamaClient`, which
  * caches responses by content hash (model + prompt + options) in SQLite,
  * appends one JSONL record per call: prompt hash, model, tokens, latency,
    cache hit, caller and the caller's pipeline state,
  * attaches the same record to the active Trace, so per-query cost is exact.

The HTTP transport is injectable so tests can run without Ollama.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from apex.config import resolve_path

Transport = Callable[[str, Dict[str, Any], float], Dict[str, Any]]


class OllamaError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cache_hit: bool
    call_id: str
    prompt_hash: str
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            raise OllamaError(f"Model {self.model} returned non-JSON output (call {self.call_id}): {self.text[:200]!r}") from e


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _http_transport(base_url: str, max_retries: int) -> Transport:
    session = requests.Session()

    def post(endpoint: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        url = f"{base_url.rstrip('/')}{endpoint}"
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = session.post(url, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    raise OllamaError(f"Ollama {endpoint} HTTP {resp.status_code}: {resp.text[:300]}")
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise OllamaError(f"Cannot reach Ollama at {url} after {max_retries + 1} attempts: {last_err}")

    return post


class _SqliteCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT)")
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_many(self, items: Dict[str, Dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                [(k, json.dumps(v), now) for k, v in items.items()],
            )
            self._conn.commit()


class OllamaClient:
    def __init__(self, cfg: Dict[str, Any], transport: Optional[Transport] = None):
        self.cfg = cfg
        o = cfg["ollama"]
        self.timeout = float(o["timeout_s"])
        self.keep_alive = o.get("keep_alive", "30m")
        self.seed = int(cfg["run"]["seed"])
        self.backend = str(o.get("backend", "unspecified"))
        self._http = transport is None
        self._version: Optional[str] = None
        self._post = transport or _http_transport(o["url"], int(o.get("max_retries", 2)))
        llm = cfg["llm"]
        self.cache = _SqliteCache(resolve_path(llm["cache_path"])) if llm.get("cache_enabled", True) else None
        self.log_path = resolve_path(llm["log_path"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()
        self._local = threading.local()

    @property
    def server_version(self) -> str:
        """Ollama server version, fetched once. Part of the generate cache key: a new runtime may change outputs."""
        if self._version is None:
            if not self._http:
                self._version = "injected-transport"
            else:
                try:
                    r = requests.get(f"{self.cfg['ollama']['url'].rstrip('/')}/api/version", timeout=10)
                    r.raise_for_status()
                    self._version = str(r.json()["version"])
                except (requests.RequestException, KeyError, ValueError) as e:
                    raise OllamaError(f"Cannot read the Ollama server version: {e}") from e
        return self._version

    # ---- trace binding -------------------------------------------------
    def bind_trace(self, trace) -> None:
        """Attach a Trace to the current thread; subsequent calls are recorded on it."""
        self._local.trace = trace

    def _trace(self):
        return getattr(self._local, "trace", None)

    # ---- logging -------------------------------------------------------
    def _log(self, record: Dict[str, Any]) -> None:
        trace = self._trace()
        if trace is not None:
            record["trace_id"] = trace.trace_id
            if record.get("state") is None:
                record["state"] = trace.state
            trace.record_llm(record)
        with self._log_lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ---- generate ------------------------------------------------------
    def generate(self, prompt: str, *, model: str, caller: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512, num_ctx: Optional[int] = None,
                 json_schema: Optional[Dict[str, Any]] = None, state: Optional[str] = None) -> LLMResult:
        options: Dict[str, Any] = {"temperature": float(temperature), "seed": self.seed, "num_predict": int(max_tokens)}
        if num_ctx:
            options["num_ctx"] = int(num_ctx)
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False,
                                   "options": options, "keep_alive": self.keep_alive}
        if system:
            payload["system"] = system
        if json_schema is not None:
            payload["format"] = json_schema

        key_fields = {k: payload.get(k) for k in ("model", "prompt", "system", "options", "format")}
        key_fields["backend"] = self.backend  # CPU and GPU numerics differ; never serve one for the other
        key_fields["ollama_version"] = self.server_version  # nor across runtime upgrades
        key_material = json.dumps(key_fields, sort_keys=True)
        key = "gen:" + sha256_text(key_material)
        prompt_hash = sha256_text((system or "") + "\x00" + prompt)[:16]
        call_id = uuid.uuid4().hex[:12]

        t0 = time.perf_counter()
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            data, hit = cached, True
        else:
            data, hit = self._post("/api/generate", payload, self.timeout), False
            if self.cache:
                self.cache.put_many({key: data})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = LLMResult(
            text=data.get("response", ""), model=model,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            latency_ms=latency_ms, cache_hit=hit, call_id=call_id, prompt_hash=prompt_hash, raw=data,
        )
        self._log({
            "ts": datetime.now(timezone.utc).isoformat(), "call_id": call_id, "kind": "generate",
            "caller": caller, "state": state, "model": model, "prompt_hash": prompt_hash,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "latency_ms": round(latency_ms, 2), "server_ms": round((data.get("total_duration") or 0) / 1e6, 2),
            "cache_hit": hit, "temperature": options["temperature"], "json_mode": json_schema is not None,
            "backend": self.backend, "ollama_version": self.server_version,
            "done_reason": data.get("done_reason"),
        })
        return result

    # ---- embed ---------------------------------------------------------
    def embed(self, texts: List[str], *, model: str, caller: str, batch_size: int = 32,
              state: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """Embed texts; each text is cached individually by (model, text) hash."""
        if not texts:
            return []
        keys = [f"emb:{model}:" + sha256_text(t) for t in texts]
        out: List[Optional[List[float]]] = [None] * len(texts)
        if self.cache:
            for i, k in enumerate(keys):
                hit = self.cache.get(k)
                if hit is not None:
                    out[i] = hit["e"]
        missing = [i for i, v in enumerate(out) if v is None]

        t0 = time.perf_counter()
        prompt_tokens = 0
        for start in range(0, len(missing), batch_size):
            idx = missing[start:start + batch_size]
            payload = {"model": model, "input": [texts[i] for i in idx], "keep_alive": self.keep_alive, "truncate": True}
            if options:
                payload["options"] = dict(options)
            data = self._post("/api/embed", payload, self.timeout)
            embs = data.get("embeddings") or []
            if len(embs) != len(idx):
                raise OllamaError(f"/api/embed returned {len(embs)} vectors for {len(idx)} inputs")
            prompt_tokens += int(data.get("prompt_eval_count", 0) or 0)
            new = {}
            for i, e in zip(idx, embs):
                out[i] = e
                new[keys[i]] = {"e": e}
            if self.cache:
                self.cache.put_many(new)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._log({
            "ts": datetime.now(timezone.utc).isoformat(), "call_id": uuid.uuid4().hex[:12], "kind": "embed",
            "caller": caller, "state": state, "model": model,
            "prompt_hash": sha256_text("\x00".join(texts))[:16], "n_inputs": len(texts),
            "n_computed": len(missing), "prompt_tokens": prompt_tokens, "completion_tokens": 0,
            "latency_ms": round(latency_ms, 2), "cache_hit": len(missing) == 0, "backend": self.backend,
        })
        return out  # type: ignore[return-value]

    # ---- health --------------------------------------------------------
    def installed_models(self) -> List[str]:
        resp = requests.get(f"{self.cfg['ollama']['url'].rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

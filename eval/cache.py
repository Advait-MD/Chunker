"""
eval/cache.py — Content-Addressed Disk Cache for LLM Generation and Judge Calls.
Implements SHA-256 hashing over (model, temperature, prompt) per HARNESS.md Section 9.
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime


class ResponseCache:
    """Thread-safe disk cache storing raw LLM text responses indexed by content hash."""

    def __init__(self, cache_dir: str = "eval/cache", enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def compute_key(model: str, temperature: float, prompt: str) -> str:
        """Key: SHA-256 of (model, temperature, prompt)."""
        key_raw = f"{model}:{float(temperature):.4f}:{prompt}".encode("utf-8")
        return hashlib.sha256(key_raw).hexdigest()

    def get(self, model: str, temperature: float, prompt: str) -> Optional[str]:
        """Retrieve cached response string if hit; returns None on miss or if disabled."""
        if not self.enabled:
            return None
        key = self.compute_key(model, temperature, prompt)
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("response")
        except Exception:
            return None

    def put(self, model: str, temperature: float, prompt: str, response: str) -> None:
        """Store response string to disk."""
        if not self.enabled:
            return
        key = self.compute_key(model, temperature, prompt)
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        payload = {
            "key": key,
            "model": model,
            "temperature": float(temperature),
            "prompt": prompt,
            "response": response,
            "created_at": datetime.now().isoformat()
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

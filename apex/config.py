"""YAML config loading with deep-merged overlays and a stable config hash."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in (overlay or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def config_hash(cfg: Dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configs/base.yaml, then an optional overlay file, then in-code overrides.

    An overlay may declare `extends: <file>`; otherwise it extends base.yaml.
    The env var APEX_CONFIG selects the overlay when `path` is not given.
    """
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    path = path or os.getenv("APEX_CONFIG")
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        with open(p, "r", encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        parent = overlay.pop("extends", None)
        if parent:
            cfg = load_config(str((p.parent / parent).resolve()))
            cfg.pop("_meta", None)
        if p.resolve() != DEFAULT_CONFIG.resolve():
            cfg = deep_merge(cfg, overlay)

    if overrides:
        cfg = deep_merge(cfg, overrides)

    cfg["_meta"] = {"config_hash": config_hash({k: v for k, v in cfg.items() if k != "_meta"}),
                    "config_path": str(path or DEFAULT_CONFIG)}
    return cfg


def resolve_path(cfg_path: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(cfg_path)
    return p if p.is_absolute() else REPO_ROOT / p

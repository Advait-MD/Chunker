"""
eval/tests/test_cache.py — Unit tests for LLM Response Cache.
Validates cache hits, content-addressed hashing, temperature sensitivity, and bypass flag.
"""

import os
import shutil
import pytest
from eval.cache import ResponseCache


@pytest.fixture
def temp_cache(tmp_path):
    cache_dir = tmp_path / "test_cache"
    cache = ResponseCache(cache_dir=str(cache_dir), enabled=True)
    yield cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_cache_hit_returns_identical_response(temp_cache):
    model = "gemma3:1b"
    temp = 0.0
    prompt = "What is React?"
    expected = "React is a JavaScript library for building user interfaces."

    # Cache miss initially
    assert temp_cache.get(model, temp, prompt) is None

    # Put into cache
    temp_cache.put(model, temp, prompt, expected)

    # Cache hit returns identical response
    hit = temp_cache.get(model, temp, prompt)
    assert hit == expected


def test_cache_key_changes_with_temperature(temp_cache):
    model = "gemma3:1b"
    prompt = "What is React?"

    key_t0 = temp_cache.compute_key(model, 0.0, prompt)
    key_t1 = temp_cache.compute_key(model, 0.7, prompt)

    assert key_t0 != key_t1

    temp_cache.put(model, 0.0, prompt, "Answer T0")
    assert temp_cache.get(model, 0.7, prompt) is None


def test_cache_key_changes_with_model(temp_cache):
    prompt = "What is React?"
    key_gemma = temp_cache.compute_key("gemma3:1b", 0.0, prompt)
    key_llama = temp_cache.compute_key("llama3.2:1b", 0.0, prompt)

    assert key_gemma != key_llama


def test_disabled_cache_never_hits(tmp_path):
    cache_dir = tmp_path / "disabled_cache"
    cache = ResponseCache(cache_dir=str(cache_dir), enabled=False)

    cache.put("m", 0.0, "p", "resp")
    assert cache.get("m", 0.0, "p") is None

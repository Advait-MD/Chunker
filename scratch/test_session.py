import requests
import time
import json

s = requests.Session()

# Warm connection
s.get("http://localhost:11434/api/tags")

for i in range(4):
    t0 = time.perf_counter()
    r = s.post(
        'http://localhost:11434/api/generate',
        json={'model': 'gemma3:1b', 'prompt': 'What is React? Answer in 10 words.', 'stream': False},
        timeout=10
    )
    t_tot = time.perf_counter() - t0
    d = r.json()
    load_s = d.get('load_duration', 0) / 1e9
    pe_s = d.get('prompt_eval_duration', 0) / 1e9
    eval_s = d.get('eval_duration', 0) / 1e9
    ollama_s = d.get('total_duration', 0) / 1e9
    overhead_s = t_tot - ollama_s
    print(f"Run {i}: Python total: {t_tot:.3f}s | Ollama total: {ollama_s:.3f}s (Load: {load_s:.3f}s, PE: {pe_s:.3f}s, Eval: {eval_s:.3f}s) | HTTP Overhead: {overhead_s:.3f}s")

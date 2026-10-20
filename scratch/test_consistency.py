import requests
import time

url = "http://localhost:11434/api/generate"
model = "gemma3:1b"

payload_base = {
    "model": model,
    "keep_alive": "1h",
    "stream": False,
    "options": {
        "num_predict": 30,
        "temperature": 0.2,
        "top_k": 30,
        "top_p": 0.9
    }
}

# 1. Warmup
print("1. Sending warmup request...", flush=True)
t0 = time.perf_counter()
r0 = requests.post(url, json={**payload_base, "prompt": "Warmup"}).json()
print(f"Warmup done in {time.perf_counter() - t0:.2f}s (load: {r0.get('load_duration',0)/1e9:.2f}s, pe: {r0.get('prompt_eval_duration',0)/1e9:.2f}s, eval: {r0.get('eval_duration',0)/1e9:.2f}s)\n", flush=True)

# 2. Benchmark 5 consecutive queries
queries = [
    "What is JSX in React? Answer in 15 words.",
    "Explain the Book of Water in 15 words.",
    "What was Steve Jobs' vision for Macintosh in 15 words?",
    "How to brew AeroPress coffee in 15 words?",
    "How to avoid morning procrastination in 15 words?"
]

print(f"{'#':<3} | {'Prompt Eval (s)':<17} | {'Eval (s)':<12} | {'Load (s)':<12} | {'Total (s)':<12} | {'Tokens/s'}", flush=True)
print("-" * 75, flush=True)

for i, q in enumerate(queries, 1):
    t_start = time.perf_counter()
    r = requests.post(url, json={**payload_base, "prompt": q}).json()
    t_tot = time.perf_counter() - t_start
    
    pe_dur = r.get('prompt_eval_duration', 0) / 1e9
    pe_cnt = r.get('prompt_eval_count', 0)
    eval_dur = r.get('eval_duration', 0) / 1e9
    eval_cnt = r.get('eval_count', 0)
    load_dur = r.get('load_duration', 0) / 1e9
    tps = eval_cnt / eval_dur if eval_dur > 0 else 0
    
    print(f"{i:<3} | {pe_dur:10.3f}s ({pe_cnt:<3}t) | {eval_dur:7.3f}s     | {load_dur:7.3f}s     | {t_tot:7.3f}s     | {tps:5.1f} t/s", flush=True)

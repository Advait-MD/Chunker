import requests
import time
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

session = requests.Session()

# Pre-warm runner once with exact options
print("Pre-warming runner...", flush=True)
with session.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'gemma3:1b',
        'prompt': 'Warmup',
        'stream': False,
        'options': {'num_predict': 10},
        'keep_alive': '1h'
    }
) as r:
    pass
print("Warmup complete.\n", flush=True)

test_cases = [
    ("What is JSX in React? Answer in 10 words.", "React JSX"),
    ("Explain the Book of Water in 10 words.", "Musashi Water"),
    ("What was Steve Jobs' core vision in 10 words?", "Jobs Vision"),
    ("How to brew good coffee in 10 words?", "Coffee"),
    ("How to avoid procrastination in 10 words?", "Procrastination")
]

print(f"{'#':<3} | {'Query':<18} | {'TTFT':<12} | {'Total':<10} | {'Throughput':<12} | {'Load':<10} | {'PE':<10}")
print("-" * 75)

for i, (q, name) in enumerate(test_cases, 1):
    t0 = time.perf_counter()
    with session.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'gemma3:1b',
            'prompt': q,
            'stream': True,
            'options': {'num_predict': 25, 'temperature': 0.2, 'top_k': 30, 'top_p': 0.9},
            'keep_alive': '1h'
        },
        stream=True
    ) as res:
        ttft = None
        tokens = []
        final_data = {}
        for line in res.iter_lines():
            if line:
                d = json.loads(line)
                tok = d.get("response", "")
                if ttft is None and tok:
                    ttft = time.perf_counter() - t0
                tokens.append(tok)
                if d.get("done"):
                    final_data = d
                    break
        res.close()
        
    t_tot = time.perf_counter() - t0
    eval_dur = final_data.get('eval_duration', 0) / 1e9
    eval_cnt = final_data.get('eval_count', 0)
    load_s = final_data.get('load_duration', 0) / 1e9
    pe_s = final_data.get('prompt_eval_duration', 0) / 1e9
    tps = eval_cnt / eval_dur if eval_dur > 0 else 0
    ttft_val = ttft if ttft is not None else 0.0
    
    print(f"{i:<3} | {name:<18} | {ttft_val*1000:6.1f} ms | {t_tot:6.3f}s   | {tps:5.1f} tok/s  | {load_s:6.3f}s   | {pe_s:6.3f}s", flush=True)

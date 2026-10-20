import requests
import time

print("=" * 65)
print("TESTING WHAT UNLOCKS MAXIMUM GENERATION THROUGHPUT ON OLLAMA")
print("=" * 65)

configs = [
    ("Default Options", {}),
    ("Flash Attention True (via request)", {"flash_attention": True}),
    ("Batch Size 512, Threads 4", {"num_thread": 4, "num_batch": 512}),
    ("Batch Size 512, Threads 8", {"num_thread": 8, "num_batch": 512}),
    ("Num Ctx 512", {"num_ctx": 512}),
    ("Temperature 0.7, top_k 40", {"temperature": 0.7, "top_k": 40}),
]

for name, opts in configs:
    payload = {
        'model': 'gemma3:1b',
        'prompt': 'Write 30 words about speed and momentum.',
        'stream': False,
        'keep_alive': '1h'
    }
    if opts:
        payload['options'] = opts

    t0 = time.perf_counter()
    res = requests.post('http://localhost:11434/api/generate', json=payload)
    elapsed = time.perf_counter() - t0
    data = res.json()
    pe_dur = data.get('prompt_eval_duration', 0) / 1e9
    ev_cnt = data.get('eval_count', 0)
    ev_dur = data.get('eval_duration', 0) / 1e9
    tps = ev_cnt / ev_dur if ev_dur > 0 else 0

    print(f"{name:<35} | {tps:5.1f} tok/s | {pe_dur*1000:6.1f} ms pe | {elapsed:5.2f} s")

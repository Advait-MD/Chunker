import requests
import time

print("=" * 75, flush=True)
print("TESTING EXACT PROMPT EVAL & GENERATION SPEED ON INTEL ARC VULKAN", flush=True)
print("=" * 75, flush=True)

prompts = [
    ("Short Chat (~40 tokens)", "You are APEX, a helpful assistant. User: What is the capital of France? APEX:"),
    ("Medium RAG (~180 tokens)", (
        "Answer in 2 sentences based on evidence below:\n"
        "[The Road to React, p. 24]: JSX stands for JavaScript XML. It is a syntax extension for React "
        "that allows writing HTML elements in JavaScript and placing them in the DOM tree.\n"
        "Question: What is JSX and how does it work?\nAnswer:"
    )),
    ("Full Bounded RAG (~300 tokens)", (
        "Answer in 2-3 sentences based on evidence below:\n"
        "[The Road to React, p. 24]: JSX stands for JavaScript XML. It allows writing HTML elements in JavaScript.\n"
        "[The Book of Five Rings, p. 45]: The Book of Water explains the spirit of water: water adopts the shape of its receptacle. It is fluid and adaptable.\n"
        "Question: How does Musashi describe the nature of water?\nAnswer:"
    ))
]

print(f"{'Prompt Type':<30} | {'Prompt Eval':<15} | {'Gen Speed':<15} | {'Total Time':<12}", flush=True)
print("-" * 80, flush=True)

for label, p in prompts:
    t0 = time.perf_counter()
    res = requests.post('http://localhost:11434/api/generate', json={
        'model': 'gemma3:1b',
        'prompt': p,
        'stream': False,
        'options': {
            'num_predict': 50,
            'num_ctx': 1024,
            'temperature': 0.2
        },
        'keep_alive': '1h'
    })
    elapsed = time.perf_counter() - t0
    data = res.json()
    pe_cnt = data.get('prompt_eval_count', 0)
    pe_dur = data.get('prompt_eval_duration', 0) / 1e9
    ev_cnt = data.get('eval_count', 0)
    ev_dur = data.get('eval_duration', 0) / 1e9
    tps = ev_cnt / ev_dur if ev_dur > 0 else 0

    print(f"{label:<30} | {pe_dur*1000:6.1f} ms ({pe_cnt}t) | {tps:5.1f} tok/s ({ev_cnt}t) | {elapsed:5.2f} s", flush=True)

print("=" * 80, flush=True)

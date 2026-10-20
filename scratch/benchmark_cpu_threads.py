import time
import requests

prompt = "Answer in 2 sentences with source [The Road to React, p. 24]: JSX is an extension for JavaScript that allows writing HTML-like tags inside JavaScript files. What is JSX?"

print("=" * 65)
print("TESTING CPU RUNNER WITH DIFFERENT THREAD COUNTS")
print("=" * 65)
print(f"{'Threads':<10} | {'Prompt Eval (s)':<18} | {'Eval Tok/s':<15} | {'Total Time (s)'}")
print("-" * 65)

for threads in [4, 6, 8, 10, 12]:
    t0 = time.perf_counter()
    res = requests.post('http://localhost:11434/api/generate', json={
        'model': 'gemma3:1b',
        'prompt': prompt,
        'stream': False,
        'options': {
            'num_predict': 50,
            'num_ctx': 512,
            'num_thread': threads
        },
        'keep_alive': '1h'
    })
    elapsed = time.perf_counter() - t0
    data = res.json()
    pe_count = data.get('prompt_eval_count', 0)
    pe_dur_s = data.get('prompt_eval_duration', 0) / 1e9
    eval_count = data.get('eval_count', 0)
    eval_dur_s = data.get('eval_duration', 0) / 1e9
    tps = eval_count / eval_dur_s if eval_dur_s > 0 else 0

    print(f"{threads:<10} | {pe_dur_s:6.3f}s ({pe_count} tok)    | {tps:5.1f} tok/s       | {elapsed:6.3f}s")

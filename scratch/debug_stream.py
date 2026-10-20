import requests
import json
import time

t0 = time.perf_counter()
res = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'gemma3:1b',
        'prompt': 'Hi, who are you?',
        'stream': True,
        'options': {'num_gpu': 0, 'num_thread': 8, 'num_ctx': 1024, 'num_predict': 20},
        'keep_alive': '1h'
    },
    stream=True
)
print(f"HTTP connect time: {time.perf_counter() - t0:.3f}s")

for i, line in enumerate(res.iter_lines()):
    if line:
        t_now = time.perf_counter() - t0
        data = json.loads(line)
        resp = data.get("response")
        done = data.get("done")
        if i < 5 or done:
            print(f"Chunk {i} at {t_now:.3f}s: response={repr(resp)} done={done}")
        if done:
            print(f"--- Final Stats ---")
            print(f"Load duration:        {data.get('load_duration', 0) / 1e9:.3f}s")
            print(f"Prompt eval count:    {data.get('prompt_eval_count')}")
            print(f"Prompt eval duration: {data.get('prompt_eval_duration', 0) / 1e9:.3f}s")
            print(f"Eval count:           {data.get('eval_count')}")
            print(f"Eval duration:        {data.get('eval_duration', 0) / 1e9:.3f}s")
            print(f"Total duration:       {data.get('total_duration', 0) / 1e9:.3f}s")

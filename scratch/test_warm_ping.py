import requests
import time

t0 = time.perf_counter()
res = requests.post('http://localhost:11434/api/generate', json={
    'model': 'gemma3:1b',
    'prompt': 'Answer in 10 words: What is JSX in React?',
    'stream': False,
    'options': {'num_predict': 30, 'num_ctx': 1024},
    'keep_alive': '1h'
})
t_total = time.perf_counter() - t0
data = res.json()
print('Status:', res.status_code)
print('Prompt Eval Count:', data.get('prompt_eval_count'))
print('Prompt Eval Duration:', f"{data.get('prompt_eval_duration', 0) / 1e9:.3f}s")
print('Eval Count:', data.get('eval_count'))
print('Eval Duration:', f"{data.get('eval_duration', 0) / 1e9:.3f}s")
if data.get('eval_duration', 0) > 0:
    tps = data.get('eval_count') / (data.get('eval_duration') / 1e9)
    print(f"Throughput: {tps:.1f} tokens/s")
print('Total Time:', f'{t_total:.3f}s')
print('Answer:', data.get('response'))

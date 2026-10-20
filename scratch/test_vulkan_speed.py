import time
import requests

print("Pinging Ollama on Vulkan...")
t0 = time.perf_counter()
res = requests.post('http://localhost:11434/api/generate', json={
    'model': 'gemma3:1b',
    'prompt': 'Write a 20-word haiku about the speed of light.',
    'stream': False,
    'keep_alive': '1h'
})
elapsed = time.perf_counter() - t0
data = res.json()
print('Status:', res.status_code)
print('Total Time:', f'{elapsed:.2f}s')
print('Prompt Eval Count:', data.get('prompt_eval_count'))
print('Prompt Eval Duration:', f"{data.get('prompt_eval_duration', 0) / 1e9:.3f}s")
print('Eval Count:', data.get('eval_count'))
print('Eval Duration:', f"{data.get('eval_duration', 0) / 1e9:.3f}s")
if data.get('eval_duration', 0) > 0:
    tps = data.get('eval_count') / (data.get('eval_duration') / 1e9)
    print('Tokens/sec:', f'{tps:.1f}')
print('Answer:', data.get('response'))

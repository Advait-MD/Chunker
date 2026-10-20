import requests
import time

lengths = [100, 250, 400, 600, 800, 1200, 1600, 2000]
sample_sentence = "The quick brown fox jumps over the lazy dog and explores the philosophical nature of strategy and technology. "

print("=" * 65)
print("PROMPT EVALUATION LATENCY PROFILING ON INTEL ARC VULKAN")
print("=" * 65)
print(f"{'Target Tokens':<15} | {'Prompt Eval Time':<18} | {'Tokens/Sec'}")
print("-" * 65)

for n in lengths:
    text = (sample_sentence * (n // 15 + 1))[:n * 4]
    res = requests.post('http://localhost:11434/api/generate', json={
        'model': 'gemma3:1b',
        'prompt': f"Read this: {text}\nSummarize in 5 words.",
        'stream': False,
        'options': {'num_predict': 10},
        'keep_alive': '1h'
    })
    data = res.json()
    pe_count = data.get('prompt_eval_count', 0)
    pe_dur_s = data.get('prompt_eval_duration', 0) / 1e9
    pe_rate = pe_count / pe_dur_s if pe_dur_s > 0 else 0
    print(f"{pe_count:<15} | {pe_dur_s*1000:7.1f} ms ({pe_dur_s:.3f}s) | {pe_rate:6.1f} tok/s")

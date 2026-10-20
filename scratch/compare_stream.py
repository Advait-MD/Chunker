import requests
import time
import json

url = "http://localhost:11434/api/generate"
model = "gemma3:1b"

sample_text = (
    "In React, JSX is a syntax extension for JavaScript that looks similar to XML or HTML. "
    "JSX produces React elements that represent what you want to see on the screen. "
    "React embraces the fact that rendering logic is inherently coupled with other UI logic: "
    "how events are handled, how the state changes over time, and how the data is prepared for display. "
    "Instead of artificially separating technologies by putting markup and logic in separate files, "
    "React separates concerns with loosely coupled units called components that contain both. "
    "When JSX is compiled by Babel, each JSX element is transformed into React.createElement calls. "
    "React then creates an in-memory virtual DOM representation of the user interface. "
    "When a component's state or props change, React compares the new virtual DOM tree with the previous one "
    "using a fast reconciliation algorithm, and then only updates the specific DOM nodes that actually changed. "
)

prompt = f"Answer in 1 sentence using the source:\n[React.pdf, p. 12]: {sample_text}\n\nQuestion: What is JSX?\nAnswer:"
print(f"Prompt character length: {len(prompt)} chars (~{len(prompt.split())} words)")

payload_base = {
    "model": model,
    "keep_alive": "1h",
    "options": {
        "num_predict": 35,
        "temperature": 0.2,
        "top_k": 30,
        "top_p": 0.9
    }
}

# Test 1: stream=False
print("\n--- TEST 1: stream=False ---")
t0 = time.perf_counter()
r_nonstream = requests.post(url, json={**payload_base, "prompt": prompt, "stream": False}).json()
t_total_ns = time.perf_counter() - t0
print(f"Total time:       {t_total_ns:.3f}s")
print(f"Load duration:    {r_nonstream.get('load_duration', 0) / 1e9:.3f}s")
print(f"Prompt eval count:{r_nonstream.get('prompt_eval_count')}")
print(f"Prompt eval dur:  {r_nonstream.get('prompt_eval_duration', 0) / 1e9:.3f}s")
print(f"Eval count:       {r_nonstream.get('eval_count')}")
print(f"Eval duration:    {r_nonstream.get('eval_duration', 0) / 1e9:.3f}s")
tps_ns = r_nonstream.get('eval_count', 0) / (r_nonstream.get('eval_duration', 1) / 1e9)
print(f"Throughput:       {tps_ns:.1f} tok/s")
print(f"Response: {r_nonstream.get('response', '').strip()}")

# Test 2: stream=True
print("\n--- TEST 2: stream=True ---")
t0 = time.perf_counter()
res = requests.post(url, json={**payload_base, "prompt": prompt, "stream": True}, stream=True)
ttft = None
tokens = []
final_data = {}
for line in res.iter_lines():
    if line:
        data = json.loads(line)
        tok = data.get("response", "")
        if ttft is None and tok:
            ttft = time.perf_counter() - t0
        tokens.append(tok)
        if data.get("done"):
            final_data = data
t_total_s = time.perf_counter() - t0

print(f"TTFT (client):    {ttft:.3f}s" if ttft else "No tokens")
print(f"Total time:       {t_total_s:.3f}s")
print(f"Load duration:    {final_data.get('load_duration', 0) / 1e9:.3f}s")
print(f"Prompt eval count:{final_data.get('prompt_eval_count')}")
print(f"Prompt eval dur:  {final_data.get('prompt_eval_duration', 0) / 1e9:.3f}s")
print(f"Eval count:       {final_data.get('eval_count')}")
print(f"Eval duration:    {final_data.get('eval_duration', 0) / 1e9:.3f}s")
tps_s = final_data.get('eval_count', 0) / (final_data.get('eval_duration', 1) / 1e9)
print(f"Throughput:       {tps_s:.1f} tok/s")
print(f"Response: {''.join(tokens).strip()}")

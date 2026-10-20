import requests
import time
import json
import numpy as np
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

session = requests.Session()
session.get("http://localhost:11434/api/tags")

test_cases = [
    ("Explain the concept of Virtual DOM in React and why it is fast.",
     "[React.pdf, p. 14]: Virtual DOM is an in-memory representation of real DOM elements. When state changes, React updates the virtual tree, diffs it with the previous snapshot using a heuristic O(n) tree algorithm, and batches minimal DOM updates to avoid expensive browser reflows.",
     "React Virtual DOM"),
     
    ("What does Musashi say about perceiving the enemy's spirit and collapsing them?",
     "[Musashi.pdf, p. 88]: The spirit of defeat comes when an enemy is rattled or thrown off balance. When you perceive their spirit collapsing, you must pursue without letting them catch their breath, striking decisively before they recover composure.",
     "Musashi Collapse"),
     
    ("What was Steve Jobs' philosophy regarding product focus and saying no?",
     "[SteveJobs.pdf, p. 210]: Deciding what not to do is as important as deciding what to do. Focus means saying no to a hundred other good ideas. Steve Jobs reduced Apple's product line from thirty-five models to just four desktop and portable computers.",
     "Jobs Focus"),
     
    ("Give me 2 practical tips to make the smoothest pour-over coffee.",
     "",
     "Chat Coffee"),
     
    ("How do I overcome procrastination when starting a difficult technical project?",
     "",
     "Chat Procrastination"),
     
    ("How does Musashi's single-minded focus compare with Steve Jobs' ruthless curation?",
     "[Musashi.pdf, p. 45]: In strategy you must eliminate unnecessary movements and wield the sword directly. [Jobs.pdf, p. 195]: Simplicity is the ultimate sophistication; strip away every extraneous feature until only the pure essence remains.",
     "Cross-Doc Synthesis")
]

print("=" * 100)
print("🚀 VERIFICATION ACROSS 6 DISTINCT COLD PROMPTS (Target: TTFT <= 2.0s, Total <= 5.0s)")
print("=" * 100)
print(f"{'#':<3} | {'Query Category':<22} | {'TTFT':<14} | {'Total Time':<12} | {'Throughput':<12} | {'Tokens':<8} | {'Status'}")
print("-" * 100)

all_ttft = []
all_total = []
all_tps = []

for i, (q, context_text, cat) in enumerate(test_cases, 1):
    if context_text:
        prompt = f"Answer concisely in 1-2 direct sentences using the source:\n{context_text}\n\nQuestion: {q}\nAnswer:"
    else:
        prompt = f"You are APEX, a helpful assistant. Answer concisely in 1-2 direct sentences:\nUser: {q}\nAPEX:"
        
    t0 = time.perf_counter()
    res = session.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'gemma3:1b',
            'prompt': prompt,
            'stream': True,
            'options': {
                'num_predict': 40,
                'temperature': 0.2,
                'top_k': 30,
                'top_p': 0.9
            },
            'keep_alive': '1h'
        },
        stream=True
    )
    
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
                
    t_total = time.perf_counter() - t0
    eval_dur = final_data.get('eval_duration', 0) / 1e9
    eval_cnt = final_data.get('eval_count', 0)
    tps = eval_cnt / eval_dur if eval_dur > 0 else 0
    ttft_val = ttft if ttft is not None else 0.0
    
    all_ttft.append(ttft_val)
    all_total.append(t_total)
    all_tps.append(tps)
    
    pass_ttft = ttft_val <= 2.0
    pass_total = t_total <= 5.0
    status = "✅ PASS" if (pass_ttft and pass_total) else "❌ FAIL"
    
    print(f"{i:<3} | {cat:<22} | {ttft_val*1000:7.1f} ms     | {t_total:6.3f} s     | {tps:5.1f} tok/s  | {eval_cnt:<8} | {status}", flush=True)

print("=" * 100)
ttft_arr = np.array(all_ttft)
tot_arr = np.array(all_total)
tps_arr = np.array(all_tps)
print(f"📊 SUMMARY REPORT:")
print(f"  • Max TTFT:             {ttft_arr.max()*1000:.1f} ms ({ttft_arr.max():.3f}s)  [Limit: <= 2.0s]")
print(f"  • Mean TTFT:            {ttft_arr.mean()*1000:.1f} ms ({ttft_arr.mean():.3f}s)")
print(f"  • Max Total Time:       {tot_arr.max():.3f}s  [Limit: <= 5.0s]")
print(f"  • Mean Total Time:      {tot_arr.mean():.3f}s")
print(f"  • Mean Throughput:      {tps_arr.mean():.1f} tokens/s")
print(f"  • Success Rate (TTFT):  {np.sum(ttft_arr <= 2.0)} / {len(all_ttft)} ({np.mean(ttft_arr <= 2.0)*100:.1f}%)")
print(f"  • Success Rate (Total): {np.sum(tot_arr <= 5.0)} / {len(all_total)} ({np.mean(tot_arr <= 5.0)*100:.1f}%)")
print("=" * 100)

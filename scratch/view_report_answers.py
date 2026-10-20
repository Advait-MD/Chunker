import json

with open("long_evaluation_report.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("PART 1: SAMPLE DOCUMENT ANSWERS")
print("=" * 80)
for d in data["doc_results"]:
    print(f"\n[{d['id']}] Category: {d['category']} | Q words: {d['q_len_words']}")
    print(f"Q: {d['query']}")
    print(f"A: {d['answer']}")
    print(f"Metrics: Retrieval {d['retrieval_ms']:.1f}ms | TTFT: {d['ttft_ms']:.1f}ms | Total: {d['total_dur_s']:.2f}s | TPS: {d['tps']:.1f}")

print("\n" + "=" * 80)
print("PART 2: MULTI-TURN CHAT DIALOGUE")
print("=" * 80)
for c in data["chat_results"]:
    print(f"\n[{c['id']}] Category: {c['category']} | Q words: {c['q_len_words']}")
    print(f"User: {c['query']}")
    print(f"APEX: {c['answer']}")
    print(f"Metrics: Retrieval {c['retrieval_ms']:.1f}ms | TTFT: {c['ttft_ms']:.1f}ms | Total: {c['total_dur_s']:.2f}s | TPS: {c['tps']:.1f}")

import json

with open("long_evaluation_report.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 90)
print("PART 1: LONG DOCUMENT RAG EVALUATION (10 QUESTIONS)")
print("=" * 90)
for d in data["doc_results"]:
    print(f"\n[{d['id']}] {d['category']} | Length: {d['q_len_words']} words ({d['q_len_chars']} chars)")
    print(f"Q: {d['query']}")
    print(f"A: {d['answer']}")
    print(f"Performance: Retrieval {d['retrieval_ms']:.1f}ms | TTFT: {d['ttft_ms']:.1f}ms | Gen: {d['gen_dur_s']:.2f}s | Total: {d['total_dur_s']:.2f}s | TPS: {d['tps']:.1f}")

print("\n" + "=" * 90)
print("PART 2: LONG MULTI-TURN NATURAL HUMAN CONVERSATION (10 TURNS)")
print("=" * 90)
for c in data["chat_results"]:
    print(f"\n[{c['id']}] {c['category']} | Length: {c['q_len_words']} words ({c['q_len_chars']} chars)")
    print(f"User: {c['query']}")
    print(f"APEX: {c['answer']}")
    print(f"Performance: Retrieval {c['retrieval_ms']:.1f}ms | TTFT: {c['ttft_ms']:.1f}ms | Gen: {c['gen_dur_s']:.2f}s | Total: {c['total_dur_s']:.2f}s | TPS: {c['tps']:.1f}")

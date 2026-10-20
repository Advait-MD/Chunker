import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('rag_research_benchmark_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for r in d['results']:
    print(f"[{r['id']}] {r['tier']} | Hit@1: {r['hit_at_1']} | Cosine: {r['semantic_cosine']:.2f} | Status: {r['status']}")
    print(f"  Q: {r['query']}")
    print(f"  A: {r['generated_answer']}")
    print(f"  Gold: {r['gold_answer']}")
    print("-" * 80)

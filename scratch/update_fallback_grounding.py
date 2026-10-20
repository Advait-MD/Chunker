with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

target = "if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.38 and reranked_candidates[0].get('coverage_score', 0) >= 0.40:"
replacement = "if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.45 and reranked_candidates[0].get('coverage_score', 0) >= 0.50:"

assert target in content, "target line not found in face.py"
content = content.replace(target, replacement, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated fallback grounding threshold successfully!")

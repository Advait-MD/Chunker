with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

target = "c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.38"
replacement = "c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.45"
assert target in content, "target not found"
content = content.replace(target, replacement, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated coverage threshold to 0.45 successfully!")

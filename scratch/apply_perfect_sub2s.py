with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update coverage threshold to 0.38
target_coverage = "c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.34"
replacement_coverage = "c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.38"
assert target_coverage in content, "target_coverage not found in face.py"
content = content.replace(target_coverage, replacement_coverage, 1)

# 2. Update per_chunk_limit in build_prompt
target_limit = "per_chunk_limit = 140 if len(chunks_to_use) <= 1 else 90"
replacement_limit = "per_chunk_limit = 120 if len(chunks_to_use) <= 1 else 75"
assert target_limit in content, "target_limit not found in face.py"
content = content.replace(target_limit, replacement_limit, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Applied perfect sub-2s limits to face.py successfully!")

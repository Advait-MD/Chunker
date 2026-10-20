with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update rerank_score threshold for general conversation vs document RAG
old_thresh = "if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.18:"
new_thresh = "if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.32:"
assert old_thresh in content, "old_thresh not found"
content = content.replace(old_thresh, new_thresh, 1)

# 2. Update chunk bound to 300 characters in build_prompt
old_chunk_bound = "clean_chunk = chunk[:450].strip().replace('\\n', ' ')"
new_chunk_bound = "clean_chunk = chunk[:300].strip().replace('\\n', ' ')"
assert old_chunk_bound in content, "old_chunk_bound not found"
content = content.replace(old_chunk_bound, new_chunk_bound, 1)

# 3. Update num_predict to 40 in ollama_options
old_opt = '"num_predict": 45,'
new_opt = '"num_predict": 40,'
assert old_opt in content, "old_opt not found"
content = content.replace(old_opt, new_opt, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully applied precision thresholds and bounds to face.py!")

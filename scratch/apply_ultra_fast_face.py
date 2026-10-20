with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Score Thresholding in retrieve_context
target_filter = """            # Score Thresholding
            threshold = self.min_score
            passed_candidates = [c for c in reranked_candidates if c['rerank_score'] >= threshold]
            low_confidence = (len(passed_candidates) == 0)
            if low_confidence:
                # If top candidate is at least moderately related (>= 0.18), keep with low_confidence flag.
                # If completely unrelated (< 0.18), leave empty to enable seamless natural conversation.
                if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.32:
                    passed_candidates = reranked_candidates[:1]
                else:
                    passed_candidates = []"""

replacement_filter = """            # Score Thresholding: strict relevance and lexical grounding check
            threshold = max(self.min_score, 0.35)
            passed_candidates = [
                c for c in reranked_candidates 
                if c['rerank_score'] >= threshold and (c.get('bm25_score', 0) > 0 or c.get('dense_score', 0) >= 0.60)
            ]
            low_confidence = (len(passed_candidates) == 0)
            if low_confidence:
                # If top candidate is at least moderately related (>= 0.38) with BM25 match, keep 1
                if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.38 and reranked_candidates[0].get('bm25_score', 0) > 0:
                    passed_candidates = reranked_candidates[:1]
                else:
                    passed_candidates = []"""

assert target_filter in content, "target_filter not found"
content = content.replace(target_filter, replacement_filter, 1)

# 2. Update chunk bound to 220 characters in build_prompt
target_chunk = "clean_chunk = chunk[:300].strip().replace('\\n', ' ')"
replacement_chunk = "clean_chunk = chunk[:220].strip().replace('\\n', ' ')"
assert target_chunk in content, "target_chunk not found"
content = content.replace(target_chunk, replacement_chunk, 1)

# 3. Update num_predict to 35 in ollama_options
target_opt = '"num_predict": 40,'
replacement_opt = '"num_predict": 35,'
assert target_opt in content, "target_opt not found"
content = content.replace(target_opt, replacement_opt, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Applied ultra-fast optimizations to face.py successfully!")

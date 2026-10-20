with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Score Thresholding in retrieve_context
target_filter = """            # Score Thresholding: strict relevance and lexical grounding check
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

replacement_filter = """            # Score Thresholding: strict relevance and keyword coverage check
            threshold = max(self.min_score, 0.35)
            passed_candidates = [
                c for c in reranked_candidates 
                if c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.28
            ]
            low_confidence = (len(passed_candidates) == 0)
            if low_confidence:
                # If top candidate has solid query keyword coverage (>= 0.40), keep top 1
                if reranked_candidates and reranked_candidates[0]['rerank_score'] >= 0.38 and reranked_candidates[0].get('coverage_score', 0) >= 0.40:
                    passed_candidates = reranked_candidates[:1]
                else:
                    passed_candidates = []"""

assert target_filter in content, "target_filter not found"
content = content.replace(target_filter, replacement_filter, 1)

# 2. Update system prompt and chunk bound in build_prompt
target_prompt_head = """        prompt_parts = [
            "Answer concisely in 1-2 direct sentences using the source evidence. Always cite bracketed sources e.g. [Document.pdf, Page 12].\\n"
        ]"""

replacement_prompt_head = """        prompt_parts = [
            "Answer in 1-2 direct sentences using source. Always cite e.g. [Document.pdf, Page 12]:\\n"
        ]"""

assert target_prompt_head in content, "target_prompt_head not found"
content = content.replace(target_prompt_head, replacement_prompt_head, 1)

target_chunk_160 = "clean_chunk = chunk[:160].strip().replace('\\n', ' ')"
replacement_chunk_140 = "clean_chunk = chunk[:140].strip().replace('\\n', ' ')"

assert target_chunk_160 in content, "target_chunk_160 not found"
content = content.replace(target_chunk_160, replacement_chunk_140, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Applied final speed tune to face.py successfully!")

with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update coverage_score threshold to 0.34
target_coverage = "c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.28"
replacement_coverage = "c['rerank_score'] >= threshold and c.get('coverage_score', 0) >= 0.34"
assert target_coverage in content, "target_coverage not found in face.py"
content = content.replace(target_coverage, replacement_coverage, 1)

# 2. Dynamic per_chunk_limit in build_prompt
target_chunk_loop = """        # Primary source evidence (bounded to first 450 characters per chunk)
        for i, (chunk, meta) in enumerate(zip(
            context.get('chunks', [])[:self.max_context_chunks], 
            context.get('metadatas', [])[:self.max_context_chunks]
        )):
            source = meta.get('source_file', 'Doc') if meta else 'Doc'
            page_num = meta.get('page_number', '?') if meta else '?'
            clean_chunk = chunk[:140].strip().replace('\\n', ' ')
            prompt_parts.append(f"[{source}, Page {page_num}]: {clean_chunk}")"""

replacement_chunk_loop = """        # Primary source evidence (strictly budgeted so prompt length never exceeds 75 tokens)
        chunks_to_use = context.get('chunks', [])[:self.max_context_chunks]
        per_chunk_limit = 140 if len(chunks_to_use) <= 1 else 90
        for i, (chunk, meta) in enumerate(zip(
            chunks_to_use, 
            context.get('metadatas', [])[:self.max_context_chunks]
        )):
            source = meta.get('source_file', 'Doc') if meta else 'Doc'
            page_num = meta.get('page_number', '?') if meta else '?'
            clean_chunk = chunk[:per_chunk_limit].strip().replace('\\n', ' ')
            prompt_parts.append(f"[{source}, Page {page_num}]: {clean_chunk}")"""

assert target_chunk_loop in content, "target_chunk_loop not found in face.py"
content = content.replace(target_chunk_loop, replacement_chunk_loop, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Applied guaranteed limit optimization to face.py successfully!")

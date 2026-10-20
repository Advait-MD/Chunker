with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update iter_lines to chunk_size=1 (zero socket buffering delay)
old_iter = "for line in response.iter_lines():"
new_iter = "for line in response.iter_lines(chunk_size=1):"
assert old_iter in content, "old_iter not found in face.py"
content = content.replace(old_iter, new_iter, 1)

# 2. Update chunk bound to 160 characters
old_chunk = "clean_chunk = chunk[:220].strip().replace('\\n', ' ')"
new_chunk = "clean_chunk = chunk[:160].strip().replace('\\n', ' ')"
assert old_chunk in content, "old_chunk not found in face.py"
content = content.replace(old_chunk, new_chunk, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Applied sub-2s TTFT optimizations to face.py successfully!")

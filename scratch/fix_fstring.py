with open('face.py', 'r', encoding='utf-8') as f:
    c = f.read()

target = "prompt_parts.append(f\"[Memory]: {mem['text'][:150].strip().replace('\\n', ' ')}\")"
replacement = "clean_mem = mem['text'][:150].strip().replace('\\n', ' ')\n                prompt_parts.append(f\"[Memory]: {clean_mem}\")"

assert target in c, "target not found"
c = c.replace(target, replacement, 1)

with open('face.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Successfully fixed f-string!")

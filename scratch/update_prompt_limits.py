with open('face.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = 'per_chunk_limit = 120 if len(chunks_to_use) <= 1 else 75'
replacement = 'per_chunk_limit = 95 if len(chunks_to_use) <= 1 else 55'

if target in text:
    text = text.replace(target, replacement)
    print("Replaced per_chunk_limit successfully.")
else:
    print("Target per_chunk_limit not found!")

target_preamble = '"Answer in 1-2 direct sentences using source. Always cite e.g. [Document.pdf, Page 12]:\\n"'
repl_preamble = '"Directly answer in 1-2 sentences using source. Cite [Doc, Page]:\\n"'

if target_preamble in text:
    text = text.replace(target_preamble, repl_preamble)
    print("Replaced preamble successfully.")
else:
    print("Target preamble not found!")

with open('face.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Saved face.py.")

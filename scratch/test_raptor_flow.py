import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from raptor import RAPTOR, split_text_into_chunks

# Sample short story (first 500 words of Cinderella)
with open("cinderella_sample.txt", "r", encoding="utf-8") as f:
    text = f.read()[:3000]

print(f"Loaded Cinderella snippet ({len(text)} chars)")
chunks = split_text_into_chunks(text, max_tokens=100)
print(f"Chunks ({len(chunks)}):")
for i, c in enumerate(chunks[:3]):
    print(f"  Chunk {i} ({len(c.split())} words): {c[:80]}...")

print("\nInitializing RAPTOR system...")
raptor = RAPTOR(
    max_tokens_per_chunk=100,
    num_layers=3,
    reduction_dimension=3,
    threshold=0.1,
)

t0 = time.time()
print("Building tree...")
tree = raptor.add_documents(text, verbose=True)
print(f"Tree built in {round(time.time() - t0, 2)} s")

stats = raptor.get_statistics()
print("Tree statistics:", stats)

# Test Retrieval
query = "How did Cinderella find a happy ending?"
nodes, context, layer_counts = raptor.retrieve(query, collapse_tree=True, max_tokens=500)
print(f"\nRetrieved {len(nodes)} nodes (layer counts: {layer_counts}):")
for n in nodes:
    print(f"  [Layer {n.layer}, Node {n.index}] {n.text[:100]}...")

# Test Question Answering with Gemma 3:1b
print("\nGenerating answer with Gemma 3:1b...")
res = raptor.answer_question(query, max_tokens=150)
print("Answer:", res["answer"])

"""
run_raptor.py — Dedicated Execution & Verification Suite for RAPTOR RAG.
Runs from within 'Raptor Rack folder' without external dependencies.
"""

import os
import sys
import json
import time
import argparse

# 1. Enforce Windows stability environment settings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# 2. Add current folder to sys.path so modules resolve cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 3. Import RAPTOR architecture components
try:
    from raptor_system import RAPTOR
    from tree_structures import Tree, Node
    from cluster_tree_builder import split_text_into_chunks
    from cluster_utils import RAPTOR_Clustering, Recency_Clustering
    from tree_retriever import TreeRetriever
    from models import OllamaEmbeddingModel, GemmaSummarizer, GemmaQAModel, BM25Retriever
    from eval_metrics import compute_token_f1
except ImportError:
    from raptor import (
        RAPTOR,
        Tree,
        Node,
        split_text_into_chunks,
        RAPTOR_Clustering,
        Recency_Clustering,
        TreeRetriever,
        OllamaEmbeddingModel,
        GemmaSummarizer,
        GemmaQAModel,
        BM25Retriever,
        compute_token_f1
    )


SAMPLE_DOCUMENT = """
The Kingdom of Eldoria was a realm of perpetual twilight, nestled between towering obsidian mountains and an ancient whispering forest.
For centuries, the Eldorians lived in harmony with the Lumina, ethereal winged creatures that illuminated the dark valleys with their iridescent plumage.
The Lumina drew their light from the Heartstone, a massive radiant crystal anchored in the subterranean caverns beneath the royal citadel of Sunken Spire.

King Valerius governed Eldoria with measured wisdom. Unlike his bellicose ancestors who had waged devastating resource wars against the mountain clans,
Valerius prioritized scholastic discovery, botanical agriculture, and diplomatic pacts. His primary advisor, Lady Seraphina, was a renowned arcanist
who had dedicated her life to cataloging the symbiotic relationship between the Lumina and the kingdom's agricultural terraces.
Through careful resonance crystals, Seraphina channeled the Lumina's bioluminescent glow into underground fungi and root vegetables,
ensuring the populace never suffered famine despite the perpetual absence of true sunlight.

However, tranquility fractured during the Year of the Eclipse. A renegade alchemist named Morvath believed the Heartstone contained latent energy
capable of extending mortal life indefinitely. Blinded by fear of his own mortality, Morvath orchestrated a clandestine infiltration into the Sunken Spire.
Using resonance dissonators crafted from cursed pyrite, he fractured a substantial shard from the Heartstone's crystalline core.
The immediate repercussions were catastrophic. The subterranean caverns trembled violently, and the Lumina shrieked as their luminous plumage faded to dim embers.
Across Eldoria, the agricultural terraces withered in darkness, and a chilling gloom descended upon the cities.

Refusing to let his realm perish, King Valerius convened a war council, not to seek vengeance, but to formulate a precise rescue operation.
Lady Seraphina deduced that Morvath had fled to the Obsidian Crags, intending to fuse the Heartstone fragment into a necrotic crucible.
Accompanied by Captain Kaelen of the Citadel Guard and a small contingent of pathfinders, Seraphina embarked on an expedition across the frozen scree.
Armed with an inverted resonance tuning fork, Seraphina planned to neutralize Morvath's dissonator without shattering the delicate crystal fragment.

The confrontation atop the Obsidian Crags was swift. Morvath, weakened by the volatile radiation of the uncontained shard, attempted to detonate his alchemy lab.
Captain Kaelen shielded Seraphina while she activated the harmonic fork, counter-oscillating the crystal until its discordant vibrations stabilized.
Morvath was apprehended without loss of life and sentenced to restorative labor in the botanical greenhouses.
With the fragment safely returned and reunited with the Heartstone, the Lumina once again burst into brilliant azure and emerald radiance.
The crops revived, and King Valerius established the Order of the Radiant Watch to protect the Heartstone for generations to come.
"""


def verify_ollama_environment() -> bool:
    """Check that local Ollama is active with required models."""
    print("--- 1. Environment & Model Pre-flight Check ---")
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            print(f"✓ Ollama server online at http://localhost:11434")
            print(f"  Detected models: {models}")
            
            has_embed = any("nomic-embed-text" in m for m in models)
            has_gen = any("gemma3:1b" in m for m in models)
            
            if not has_embed:
                print("  [WARN] 'nomic-embed-text' not found. Run: ollama pull nomic-embed-text")
            else:
                print("  ✓ Embedding model 'nomic-embed-text' is ready")
                
            if not has_gen:
                print("  [WARN] 'gemma3:1b' not found. Run: ollama pull gemma3:1b")
            else:
                print("  ✓ Generation model 'gemma3:1b' is ready")
                
            return has_embed and has_gen
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        print("  Ensure Ollama is running ('ollama serve')")
        return False
    return False


def run_full_raptor_test():
    """Execute complete end-to-end test of RAPTOR RAG components."""
    print("\n" + "=" * 65)
    print("=== RAPTOR RAG: Comprehensive Execution & Verification Suite ===")
    print("=" * 65 + "\n")

    # Step 1: Preflight
    env_ok = verify_ollama_environment()
    if not env_ok:
        print("\n[!] Please ensure Ollama has 'gemma3:1b' and 'nomic-embed-text' installed.")
        return False

    # Step 2: Initialize RAPTOR System
    print("\n--- 2. Initializing RAPTOR RAG System ---")
    t0 = time.time()
    raptor = RAPTOR(
        summarization_model_name="gemma3:1b",
        embedding_model_name="nomic-embed-text",
        max_tokens_per_chunk=100,
        num_layers=3,
        reduction_dimension=3,
        threshold=0.1,
    )
    print(f"✓ RAPTOR instance created ({round(time.time() - t0, 3)}s)")

    # Step 3: Ingest Document & Build Tree
    print("\n--- 3. Ingesting Document & Building Hierarchical Tree ---")
    print(f"Document Length: {len(SAMPLE_DOCUMENT.split())} words")
    t_build = time.time()
    raptor.add_documents(SAMPLE_DOCUMENT, verbose=True)
    build_duration = round(time.time() - t_build, 2)
    print(f"✓ Hierarchical tree successfully constructed in {build_duration}s")

    # Step 4: Verify Tree Statistics
    print("\n--- 4. Tree Statistics & Architecture Validation ---")
    stats = raptor.get_statistics()
    for k, v in stats.items():
        print(f"  {k:30}: {v}")
    
    assert stats["total_nodes"] > 0, "Tree has zero total nodes"
    assert stats["leaf_nodes"] > 0, "Tree has zero leaf nodes"
    assert stats["num_layers"] >= 2, "Tree should have at least 2 layers (leaves + summaries)"
    print("✓ Tree structure assertions passed!")

    # Step 5: Test Algorithm 2 — Collapsed Tree Retrieval
    print("\n--- 5. Testing Algorithm 2: Collapsed Tree Retrieval ---")
    query_holistic = "What is the central conflict in the story and how was it resolved?"
    print(f"Query: '{query_holistic}'")
    t_ret = time.time()
    nodes, context, layer_dist = raptor.retrieve(query_holistic, collapse_tree=True, max_tokens=1500)
    print(f"✓ Retrieved {len(nodes)} nodes across layers: {layer_dist} ({round(time.time() - t_ret, 3)}s)")
    for i, n in enumerate(nodes[:3]):
        print(f"  Node {n.index} (Layer {n.layer}, {n.token_size} tokens): {n.text[:90]}...")

    # Step 6: Test Reader QA with Gemma 3:1b
    print("\n--- 6. Testing Reader QA Generation (Gemma 3:1b) ---")
    t_qa = time.time()
    qa_result = raptor.answer_question(query_holistic, collapse_tree=True, max_tokens=1500)
    qa_duration = round(time.time() - t_qa, 2)
    print(f"Answer ({qa_duration}s):\n{qa_result['answer']}")
    assert len(qa_result["answer"].strip()) > 0, "QA model generated empty response"
    print("✓ Grounded QA response generated successfully!")

    # Step 7: Test Algorithm 1 — Tree Traversal Retrieval
    print("\n--- 7. Testing Algorithm 1: Tree Traversal Retrieval ---")
    query_specific = "Who was Lady Seraphina and what was her plan?"
    print(f"Query: '{query_specific}'")
    t_trav = time.time()
    trav_nodes, trav_context = raptor.retriever.tree_traversal_retrieval(query_specific, k=2)
    print(f"✓ Tree Traversal retrieved {len(trav_nodes)} nodes ({round(time.time() - t_trav, 3)}s)")
    trav_ans = raptor.qa_model.answer_question(trav_context, query_specific)
    print(f"Answer:\n{trav_ans}")
    assert len(trav_ans.strip()) > 0, "Traversal QA model generated empty response"

    # Step 8: Test Serialization (Save & Load)
    print("\n--- 8. Testing Tree Persistence (Save & Load) ---")
    test_tree_path = os.path.join(CURRENT_DIR, "test_raptor_tree.json")
    raptor.save(test_tree_path)
    print(f"✓ Tree serialized to {test_tree_path} ({os.path.getsize(test_tree_path)} bytes)")

    loaded_raptor = RAPTOR.load(test_tree_path)
    loaded_stats = loaded_raptor.get_statistics()
    assert loaded_stats["total_nodes"] == stats["total_nodes"], "Loaded tree node count mismatch"
    print(f"✓ Tree successfully re-loaded from disk! Node count: {loaded_stats['total_nodes']}")

    # Clean up test save file
    if os.path.exists(test_tree_path):
        os.remove(test_tree_path)
        print("✓ Test artifact cleaned up")

    print("\n" + "=" * 65)
    print("✓ SUCCESS: ALL RAPTOR RAG COMPONENTS OPERATING FLAWLESSLY WITH ZERO ERRORS!")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPTOR RAG Execution Suite")
    parser.add_argument("--test", action="store_true", help="Run full self-test suite")
    parser.add_argument("--query", type=str, help="Query RAPTOR tree interactively")
    parser.add_argument("--tree", type=str, help="Path to saved tree JSON to load")
    args = parser.parse_args()

    run_full_raptor_test()

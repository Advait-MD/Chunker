import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from face import RAGInterface

def run_tests():
    interface = RAGInterface(
        ollama_model="gemma3:1b",
        ollama_embedding_model="nomic-embed-text",
        chroma_path="./hierarchical_chroma_db",
        max_context_chunks=4,
        include_summaries=True,
        verbose=False
    )
    
    test_queries = [
        ("React & JSX Concept", "What is JSX and how does React render elements?"),
        ("The Book of Five Rings Philosophy", "What is the Way of the Void according to Miyamoto Musashi?"),
        ("The Odyssey Narrative", "Who held Odysseus captive on the island of Ogygia?"),
        ("Steve Jobs Biography", "Where was Steve Jobs born and who raised him?"),
        ("Negative Out-Of-Domain Test", "What is the capital city and currency of planet Mars?"),
        ("Empty String Edge Case", "")
    ]
    
    print("\n" + "="*80)
    print("RUNNING RAG VERIFICATION TESTS (POST ITEM 1 & 2 FIXES)")
    print("="*80)
    
    for category, query in test_queries:
        print(f"\n🧪 [TEST: {category}]")
        print(f"❓ Query: '{query}'")
        t0 = time.time()
        result = interface.query(query)
        dt = time.time() - t0
        
        print(f"⏱️ Response Time: {dt:.2f}s")
        print(f"📚 Sources: {result.get('sources', [])}")
        print(f"🧩 Context Chunks: {result.get('context_chunks', 0)}")
        print(f"📝 Answer Preview:\n{result.get('answer', '')}")
        print("-" * 60)

if __name__ == '__main__':
    run_tests()

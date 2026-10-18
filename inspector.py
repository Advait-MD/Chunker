import os
import sys
import requests
import chromadb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("🔍 Inspecting Local Environment Setup...")

# 1. Test Local ChromaDB
chroma_path = "./hierarchical_chroma_db"
try:
    print(f"\n[1/2] Checking ChromaDB at '{chroma_path}'...")
    client = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()
    print(f"✅ Local ChromaDB is working! Found {len(collections)} collections:")
    for col in collections:
        c = client.get_collection(col.name)
        print(f"   • {col.name}: {c.count()} items")
except Exception as e:
    print(f"❌ ChromaDB Error: {e}")

# 2. Test Local Ollama
try:
    print(f"\n[2/2] Checking Ollama at 'http://localhost:11434'...")
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    if r.status_code == 200:
        models = [m.get('name', '') for m in r.json().get('models', [])]
        print(f"✅ Ollama is running! Available models:")
        for m in models:
            print(f"   • {m}")
    else:
        print(f"⚠️ Ollama responded with status code {r.status_code}")
except Exception as e:
    print(f"❌ Cannot connect to Ollama: {e}")
    print("   Make sure Ollama is running: start 'ollama serve' in background or launch from Start Menu")
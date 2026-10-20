# Raptor RAG ("Raptor Rack folder")
### Recursive Abstractive Processing for Tree-Organized Retrieval (ICLR 2024)

Complete local implementation of RAPTOR RAG using **Google Gemma 3:1b** and **Nomic Embed Text** via native Ollama.

---

## Folder Structure

```
Raptor Rack folder/
├── run_raptor.py            # Primary standalone execution and verification suite
├── raptor_system.py         # Main RAPTOR facade class (add_documents, retrieve, answer_question)
├── cluster_tree_builder.py  # 100-token sentence chunker & recursive bottom-up builder
├── cluster_utils.py         # UMAP reduction, GMM with BIC selection, soft clustering (tau=0.1)
├── tree_retriever.py        # Algorithm 1 (Tree Traversal) & Algorithm 2 (Collapsed Tree Retrieval)
├── tree_structures.py       # Node and Tree DAG data structures with JSON serialization
├── models.py                # GemmaSummarizer, GemmaQAModel, OllamaEmbeddingModel, BM25Retriever
├── eval_metrics.py          # Token F1, BLEU-1, BLEU-4, ROUGE-L, METEOR, MC Accuracy
├── __init__.py              # Package initializer with multi-mode relative/absolute imports
└── raptor/                  # Standard nested package for 'from raptor import RAPTOR'
```

---

## Running RAPTOR from this Folder

To verify and run RAPTOR from within this folder:

```powershell
cd "c:\Users\hp\Chunker\Raptor Rack folder"
python run_raptor.py
```

### Programmatic Usage

```python
from raptor_system import RAPTOR

# 1. Initialize
raptor = RAPTOR(
    summarization_model_name="gemma3:1b",
    embedding_model_name="nomic-embed-text",
    max_tokens_per_chunk=100,
    num_layers=3,
    reduction_dimension=3,
    threshold=0.1
)

# 2. Ingest document & build tree
raptor.add_documents(my_text, verbose=True)

# 3. Query with Collapsed Tree retrieval (Algorithm 2)
response = raptor.answer_question(
    question="What is the central theme of the story?",
    collapse_tree=True,
    max_tokens=1500
)
print("Answer:", response["answer"])
print("Retrieved nodes:", response["retrieved_nodes_count"])

# 4. Save and Load
raptor.save("my_tree.json")
loaded_raptor = RAPTOR.load("my_tree.json")
```

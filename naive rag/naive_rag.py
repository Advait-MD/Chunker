"""
naive_rag.py — Lewis et al. (2020)-style RAG Baseline for APEX.

Source: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020.
Paper: https://arxiv.org/abs/2005.11401

Key characteristics:
- Passage unit: 100-word passages, no overlap
- Dense Passage Retrieval (DPR):
    * Context encoder: facebook/dpr-ctx_encoder-single-nq-base (frozen)
    * Question encoder: facebook/dpr-question_encoder-single-nq-base (frozen)
- Vector store: FAISS IndexFlatIP (exact maximum inner product search, no normalization)
- Top-k: k = 5
- Generator: Ollama model matching APEX (gemma3:1b) with temperature = 0.0
- Deterministic evaluation harness with token F1, Recall@k, and MRR.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
import ollama
import PyPDF2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ---- Configuration -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(SCRIPT_DIR, "naive_rag_index.faiss")
PASSAGES_FILE = os.path.join(SCRIPT_DIR, "passages.json")

CTX_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
Q_MODEL = "facebook/dpr-question_encoder-single-nq-base"
PASSAGE_WORDS = 100
TOP_K = 5
GENERATOR = "gemma3:1b"  # Matches APEX default Ollama model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimize CPU threads if running on CPU
if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)

# ---- Generation Prompt (Lewis et al. standard zero-shot constraint) ----------
PROMPT_TEMPLATE = """Answer the question using only the passages below. If the passages do not contain the answer, say "I don't know."

{context}

Question: {query}
Answer:"""


class NaiveRAG:
    """
    Lewis et al. (2020) Naive RAG implementation with FAISS IndexFlatIP,
    frozen DPR encoders, 100-word chunking, and deterministic Ollama generation.
    """

    def __init__(self,
                 generator_model: str = GENERATOR,
                 device: str = DEVICE,
                 top_k: int = TOP_K):
        self.generator_model = generator_model
        self.device = device
        self.top_k = top_k

        self.passages: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None

        self._ctx_tok = None
        self._ctx_enc = None
        self._q_tok = None
        self._q_enc = None

    def _load_question_encoder(self):
        """Lazy load question encoder."""
        if self._q_tok is None or self._q_enc is None:
            print(f"Loading DPR Question Encoder ({Q_MODEL}) on {self.device}...")
            self._q_tok = DPRQuestionEncoderTokenizer.from_pretrained(Q_MODEL)
            self._q_enc = DPRQuestionEncoder.from_pretrained(Q_MODEL).to(self.device).eval()

    def _load_context_encoder(self):
        """Lazy load context encoder."""
        if self._ctx_tok is None or self._ctx_enc is None:
            print(f"Loading DPR Context Encoder ({CTX_MODEL}) on {self.device}...")
            self._ctx_tok = DPRContextEncoderTokenizer.from_pretrained(CTX_MODEL)
            self._ctx_enc = DPRContextEncoder.from_pretrained(CTX_MODEL).to(self.device).eval()

    @torch.inference_mode()
    def encode_passages(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode passage texts into dense 768-d embeddings using DPR context encoder."""
        self._load_context_encoder()
        out = []
        total = len(texts)
        t_start = time.perf_counter()

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch = self._ctx_tok(
                batch_texts,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            emb = self._ctx_enc(**batch).pooler_output.cpu().numpy()
            out.append(emb)

            done = min(i + batch_size, total)
            elapsed = time.perf_counter() - t_start
            rate = done / max(elapsed, 0.001)
            print(f"  [DPR Encoding] {done}/{total} passages ({done/total*100:.1f}%) | {rate:.1f} passages/sec", end="\r")

        print()
        return np.vstack(out).astype("float32")

    @torch.inference_mode()
    def encode_query(self, query: str) -> np.ndarray:
        """Encode single query using DPR question encoder."""
        self._load_question_encoder()
        batch = self._q_tok(query, return_tensors="pt").to(self.device)
        return self._q_enc(**batch).pooler_output.cpu().numpy().astype("float32")

    @staticmethod
    def chunk_words(text: str, n: int = PASSAGE_WORDS) -> List[str]:
        """Split text into consecutive 100-word passages with zero overlap."""
        words = text.split()
        if not words:
            return []
        return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]

    def extract_document_text(self, pdf_path: str) -> str:
        """Extract clean text from PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PyPDF2.PdfReader(pdf_path)
        pages_text = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages_text.append(t.strip())
        return "\n\n".join(pages_text)

    def build_index_from_documents(self, documents: List[Dict[str, str]], save_to_disk: bool = True):
        """
        Build FAISS IndexFlatIP index over documents: list of {"id": doc_name, "text": text}
        """
        print(f"--- Chunking {len(documents)} document(s) into {PASSAGE_WORDS}-word passages (no overlap) ---")
        self.passages = []
        for doc in documents:
            doc_id = doc["id"]
            chunks = self.chunk_words(doc["text"], n=PASSAGE_WORDS)
            print(f"  • {doc_id}: {len(chunks)} passages")
            for chunk_idx, chunk in enumerate(chunks):
                self.passages.append({
                    "idx": len(self.passages),
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "text": chunk
                })

        total_passages = len(self.passages)
        print(f"Total passages to encode: {total_passages}")

        texts = [p["text"] for p in self.passages]
        t0 = time.perf_counter()
        embeddings = self.encode_passages(texts)
        t_enc = time.perf_counter() - t0
        print(f"Passage encoding completed in {t_enc:.2f}s ({total_passages / max(t_enc, 0.001):.1f} passages/sec)")

        dim = embeddings.shape[1]
        print(f"Creating FAISS IndexFlatIP (dim={dim}) for exact inner-product search...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index populated: {self.index.ntotal} vectors.")

        if save_to_disk:
            self.save_index(INDEX_FILE, PASSAGES_FILE)

    def save_index(self, index_path: str = INDEX_FILE, passages_path: str = PASSAGES_FILE):
        """Save FAISS index and passage metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        print(f"Saving FAISS index to {index_path}...")
        faiss.write_index(self.index, index_path)
        print(f"Saving passages metadata ({len(self.passages)} items) to {passages_path}...")
        with open(passages_path, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)
        print("✓ Index and metadata saved successfully.")

    def load_index(self, index_path: str = INDEX_FILE, passages_path: str = PASSAGES_FILE) -> bool:
        """Load precomputed FAISS index and passages from disk."""
        if not (os.path.exists(index_path) and os.path.exists(passages_path)):
            return False

        print(f"Loading cached FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Loading passages metadata from {passages_path}...")
        with open(passages_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)
        print(f"✓ Successfully loaded {self.index.ntotal} index vectors and {len(self.passages)} passages.")
        return True

    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        DPR Question Encoder + FAISS IndexFlatIP top-k search.
        Returns list of retrieved passages with rank, score, doc_id, and text.
        """
        if self.index is None or not self.passages:
            raise ValueError("Index is not loaded or built.")
        if k is None:
            k = self.top_k

        t0 = time.perf_counter()
        q_emb = self.encode_query(query)
        scores, ids = self.index.search(q_emb, k)
        t_ret = time.perf_counter() - t0

        retrieved = []
        for r, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx < 0 or idx >= len(self.passages):
                continue
            p = self.passages[idx]
            retrieved.append({
                "rank": r + 1,
                "score": float(score),
                "doc_id": p["doc_id"],
                "idx": int(idx),
                "text": p["text"]
            })
        return retrieved

    def generate(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """
        Deterministic generator call via native Ollama (temperature=0.0).
        """
        context = "\n\n".join(f"[{p['rank']}] {p['text']}" for p in retrieved)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)

        try:
            response = ollama.generate(
                model=self.generator_model,
                prompt=prompt,
                options={
                    "temperature": 0.0,  # Deterministic for reproducible evaluation
                    "num_predict": 150,
                    "num_ctx": 2048,
                }
            )
            return response.get("response", "").strip()
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"

    def rag_answer(self, query: str, k: int = None) -> Dict[str, Any]:
        """
        Full pipeline: Retrieve top-k passages via DPR/FAISS and generate answer.
        """
        t0 = time.perf_counter()
        retrieved = self.retrieve(query, k=k)
        t_ret = time.perf_counter() - t0

        t1 = time.perf_counter()
        answer = self.generate(query, retrieved)
        t_gen = time.perf_counter() - t1

        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieved,
            "retrieval_time": t_ret,
            "generation_time": t_gen,
            "total_time": t_ret + t_gen
        }


# ---- Standalone Helper Functions Matching Paper Specification ---------------

def chunk_words(text: str, n: int = PASSAGE_WORDS) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def recall_at_k(retrieved: List[Dict], gold_doc_ids: List[str], k: int = TOP_K) -> float:
    """Fraction of gold documents retrieved in top-k."""
    if not gold_doc_ids:
        return 0.0
    got = {p["doc_id"] for p in retrieved[:k]}
    return len(got & set(gold_doc_ids)) / len(gold_doc_ids)


def mrr(retrieved: List[Dict], gold_doc_ids: List[str]) -> float:
    """Mean Reciprocal Rank of first gold document."""
    for p in retrieved:
        if p["doc_id"] in gold_doc_ids:
            return 1.0 / p["rank"]
    return 0.0


def token_f1(gold: str, pred: str) -> float:
    """Standard Token F1 score for QA evaluation."""
    def tokenize(s):
        import re
        return re.findall(r"\w+", s.lower())

    gold_toks = tokenize(gold)
    pred_toks = tokenize(pred)
    if not gold_toks or not pred_toks:
        return 0.0

    common = set(gold_toks) & set(pred_toks)
    if not common:
        return 0.0

    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


# ---- CLI & Indexing Entrypoint -----------------------------------------------

def index_corpus(docs_dir: str = SCRIPT_DIR, force: bool = False) -> NaiveRAG:
    """Index all PDFs found in docs_dir."""
    rag = NaiveRAG()

    if not force and rag.load_index():
        return rag

    # Discover PDFs
    pdf_files = [
        f for f in os.listdir(docs_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF documents found in {docs_dir}")

    print(f"Found {len(pdf_files)} PDF document(s) in {docs_dir}:")
    for f in pdf_files:
        print(f"  • {f}")

    documents = []
    for pdf_name in pdf_files:
        full_path = os.path.join(docs_dir, pdf_name)
        text = rag.extract_document_text(full_path)
        documents.append({
            "id": pdf_name,
            "text": text
        })

    rag.build_index_from_documents(documents, save_to_disk=True)
    return rag


def main():
    parser = argparse.ArgumentParser(description="Naive RAG (Lewis et al., 2020) Baseline for APEX")
    parser.add_argument("--index", action="store_true", help="Force building FAISS index from PDFs")
    parser.add_argument("--query", type=str, default=None, help="Run single query")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-k passages to retrieve")
    parser.add_argument("--interactive", action="store_true", help="Start interactive CLI loop")
    args = parser.parse_args()

    rag = NaiveRAG()
    has_cached = rag.load_index()

    if args.index or not has_cached:
        print("\n=== Building Naive RAG FAISS Index ===")
        rag = index_corpus(SCRIPT_DIR, force=True)

    if args.query:
        print(f"\nQuery: {args.query}")
        res = rag.rag_answer(args.query, k=args.k)
        print("\nRetrieved Passages:")
        for p in res["retrieved"]:
            print(f"  [{p['rank']}] Score: {p['score']:.4f} | Source: {p['doc_id']}")
            print(f"      Text: {p['text'][:120]}...\n")
        print("Generated Answer:")
        print(f"{res['answer']}")
        print(f"\n[Retrieval: {res['retrieval_time']:.3f}s | Generation: {res['generation_time']:.3f}s]")

    elif args.interactive:
        print("\n=== Naive RAG (Lewis et al., 2020) Interactive Terminal ===")
        print("Type 'exit' or 'quit' to quit.\n")
        while True:
            try:
                q = input("\nQuery > ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    break
                res = rag.rag_answer(q, k=args.k)
                print("\nTop Retrieved:")
                for p in res["retrieved"]:
                    print(f"  [{p['rank']}] ({p['score']:.2f}) [{p['doc_id']}] {p['text'][:90]}...")
                print(f"\nAnswer:\n{res['answer']}\n")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()

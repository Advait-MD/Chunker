"""
raptor/models.py — Embedding models, Summarization model, and Reader/QA model
integrated with local Ollama (Gemma 3:1b and nomic-embed-text), HuggingFace SBERT/DPR,
and BM25, adhering to Sarthi et al. (ICLR 2024).
"""

import os
import sys
import time
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import requests

logger = logging.getLogger("raptor.models")


# ==================== EMBEDDING MODELS ====================

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> np.ndarray:
        pass

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.array([self.create_embedding(t) for t in texts], dtype=np.float32)


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    Local embedding model powered by Ollama nomic-embed-text (768 dimensions).
    Fast, fully offline, and includes session reuse and batching.
    """
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        timeout: int = 30,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.cache: Dict[str, np.ndarray] = {}

    def create_embedding(self, text: str) -> np.ndarray:
        text_clean = text.strip()
        if not text_clean:
            return np.zeros(768, dtype=np.float32)

        cache_key = f"{self.model_name}:{hash(text_clean)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"{self.ollama_url}/api/embed"
        payload = {"model": self.model_name, "input": text_clean}

        for attempt in range(3):
            try:
                resp = self.session.post(endpoint, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    embs = data.get("embeddings", [])
                    if embs:
                        emb = np.array(embs[0], dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        self.cache[cache_key] = emb
                        return emb
                # Fallback to legacy /api/embeddings
                legacy_ep = f"{self.ollama_url}/api/embeddings"
                resp = self.session.post(legacy_ep, json={"model": self.model_name, "prompt": text_clean}, timeout=self.timeout)
                data = resp.json()
                emb = np.array(data.get("embedding", []), dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self.cache[cache_key] = emb
                return emb
            except Exception as e:
                time.sleep(0.5 * (attempt + 1))
                if attempt == 2:
                    logger.error(f"Ollama embedding failed for text snippet: {e}")
                    return np.zeros(768, dtype=np.float32)

        return np.zeros(768, dtype=np.float32)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch embedding with /api/embed."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        # Separate cached and uncached texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for idx, t in enumerate(texts):
            clean_t = t.strip()
            if not clean_t:
                results[idx] = np.zeros(768, dtype=np.float32)
                continue
            key = f"{self.model_name}:{hash(clean_t)}"
            if key in self.cache:
                results[idx] = self.cache[key]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(clean_t)

        # Batch request for uncached
        if uncached_texts:
            endpoint = f"{self.ollama_url}/api/embed"
            batch_size = 32
            for i in range(0, len(uncached_texts), batch_size):
                sub_batch = uncached_texts[i:i + batch_size]
                sub_indices = uncached_indices[i:i + batch_size]
                try:
                    resp = self.session.post(endpoint, json={"model": self.model_name, "input": sub_batch}, timeout=self.timeout)
                    if resp.status_code == 200:
                        embs = resp.json().get("embeddings", [])
                        for sub_idx, raw_emb in zip(sub_indices, embs):
                            arr = np.array(raw_emb, dtype=np.float32)
                            norm = np.linalg.norm(arr)
                            if norm > 0:
                                arr = arr / norm
                            key = f"{self.model_name}:{hash(texts[sub_idx].strip())}"
                            self.cache[key] = arr
                            results[sub_idx] = arr
                except Exception as e:
                    logger.warning(f"Batch embed failed ({e}), falling back to single embed.")
                    for sub_idx in sub_indices:
                        results[sub_idx] = self.create_embedding(texts[sub_idx])

        # Fill any remaining None with single embed
        for idx in range(len(texts)):
            if results[idx] is None:
                results[idx] = self.create_embedding(texts[idx])

        return np.array(results, dtype=np.float32)


class SBertEmbeddingModel(BaseEmbeddingModel):
    """
    Sentence-BERT embedding model using HuggingFace AutoModel and AutoTokenizer
    (default: multi-qa-mpnet-base-cos-v1 or all-MiniLM-L6-v2) with mean pooling.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        device: Optional[str] = None,
        fallback_ollama: bool = True
    ):
        self.model_name = model_name
        self.fallback_ollama = fallback_ollama
        self.cache: Dict[str, np.ndarray] = {}
        self.tokenizer = None
        self.model = None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info(f"Loaded SBert model '{model_name}' on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace model '{model_name}' ({e}). Falling back to Ollama nomic-embed-text.")
            self.tokenizer = None
            self.model = None
            self.ollama_fallback = OllamaEmbeddingModel()

    def create_embedding(self, text: str) -> np.ndarray:
        text_clean = text.strip()
        if not text_clean:
            return np.zeros(768, dtype=np.float32)

        if text_clean in self.cache:
            return self.cache[text_clean]

        if self.model is None or self.tokenizer is None:
            return self.ollama_fallback.create_embedding(text)

        import torch
        inputs = self.tokenizer(
            text_clean,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            emb = mean_pooled.cpu().numpy()[0].astype(np.float32)

            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

        self.cache[text_clean] = emb
        return emb


class DPREmbeddingModel:
    """
    Dense Passage Retrieval (DPR) encoder pair:
    Context encoder (for passages/nodes) and Question encoder (for queries).
    """
    def __init__(
        self,
        ctx_model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        q_model_name: str = "facebook/dpr-question_encoder-single-nq-base",
        device: Optional[str] = None
    ):
        import torch
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizer,
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizer,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model_name)
        self.ctx_model = DPRContextEncoder.from_pretrained(ctx_model_name).to(self.device)
        self.ctx_model.eval()

        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_model_name)
        self.q_model = DPRQuestionEncoder.from_pretrained(q_model_name).to(self.device)
        self.q_model.eval()
        self.cache: Dict[str, np.ndarray] = {}

    def embed_context(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        import torch
        inputs = self.ctx_tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.ctx_model(**inputs).pooler_output.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self.cache[text] = emb
        return emb

    def embed_question(self, text: str) -> np.ndarray:
        import torch
        inputs = self.q_tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.q_model(**inputs).pooler_output.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


class BM25Retriever:
    """
    Standard BM25 keyword matching retriever (Robertson et al. 1995/2009).
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avg_dl = sum(self.doc_lengths) / max(len(corpus), 1)
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._build_idf()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_idf(self):
        N = len(self.corpus)
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        for term, freq in self.doc_freqs.items():
            self.idf[term] = np.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.corpus), dtype=np.float32)

        for i, doc in enumerate(self.tokenized_corpus):
            doc_len = self.doc_lengths[i]
            doc_counts = {}
            for t in doc:
                doc_counts[t] = doc_counts.get(t, 0) + 1

            for qt in query_tokens:
                if qt in doc_counts:
                    freq = doc_counts[qt]
                    idf = self.idf.get(qt, 0.0)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_dl, 1e-6)))
                    scores[i] += idf * (numerator / denominator)

        return scores


# ==================== SUMMARIZATION MODEL ====================

class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        pass


class GemmaSummarizer(BaseSummarizationModel):
    """
    Local LLM summarizer using Gemma 3:1b via Ollama.
    Employs the exact prompt from RAPTOR Paper Table 11:
      System: 'You are a Summarizing Text Portal'
      User: 'Write a summary of the following, including as many key details as possible: {context}:'
    """
    def __init__(
        self,
        model_name: str = "gemma3:1b",
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def summarize(self, context: str, max_tokens: int = 200) -> str:
        if not context or not context.strip():
            return ""

        endpoint = f"{self.ollama_url}/api/generate"
        system_prompt = "You are a Summarizing Text Portal"
        user_prompt = f"Write a summary of the following, including as many key details as possible: {context.strip()}:"

        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
            }
        }

        for attempt in range(3):
            try:
                resp = self.session.post(endpoint, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json().get("response", "").strip()
                if result:
                    # Clean out conversational filler
                    cleaned = re.sub(
                        r"(?i)^(?:(?:okay|sure|certainly)[,\.\!]?\s*)?(?:here\s*(?:is|'s)\s*(?:a\s*)?(?:summary|the\s*summary)[^:\n]*:?\s*|\*\*summary:?\*\*\s*|summary:\s*)",
                        "",
                        result
                    ).strip()
                    return cleaned if cleaned else result
            except Exception as e:
                logger.warning(f"Gemma summarization attempt {attempt+1} failed: {e}")
                time.sleep(1.0 * (attempt + 1))

        # Fallback heuristic summary: take first 2-3 salient sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        return " ".join(sentences[:3]) if sentences else context[:300]


# ==================== QA / READER MODEL ====================

class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context: str, question: str, **kwargs) -> str:
        pass


class GemmaQAModel(BaseQAModel):
    """
    Local Reader / QA model using Gemma 3:1b via Ollama.
    Supports both multiple-choice (QuALITY) and open/abstractive question answering (NarrativeQA, QASPER).
    """
    def __init__(
        self,
        model_name: str = "gemma3:1b",
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def answer_question(
        self,
        context: str,
        question: str,
        options: Optional[List[str]] = None,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate answer from retrieved context.
        Zero temperature ensures deterministic reproducible evaluation.
        """
        endpoint = f"{self.ollama_url}/api/generate"

        if options:
            # Multiple-choice format (QuALITY benchmark)
            formatted_options = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
            system_prompt = "You are Question Answering Portal"
            user_prompt = (
                f"Given Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{formatted_options}\n\n"
                f"Select the single best correct option. Output ONLY the letter (A, B, C, or D) corresponding to the correct answer."
            )
        else:
            # Free-text QA format (NarrativeQA, QASPER, Cinderella)
            system_prompt = "You are Question Answering Portal"
            user_prompt = (
                f"Given Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer the question directly and concisely based solely on the provided context:"
            )

        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        for attempt in range(3):
            try:
                resp = self.session.post(endpoint, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                ans = resp.json().get("response", "").strip()
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Gemma QA attempt {attempt+1} failed: {e}")
                time.sleep(1.0 * (attempt + 1))

        return ""

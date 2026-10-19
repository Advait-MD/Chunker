import sys
import time
from typing import List, Tuple, Optional
from collections import Counter
import re
import requests

try:
    from .config import config
    from .logger import get_logger
except ImportError:
    from modules.config import config
    from modules.logger import get_logger

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = get_logger("summarizer")


class ClusterSummary(str):
    """
    Subclass of str representing a generated cluster summary.
    Acts as a standard string (full_summary) for 100% backward compatibility,
    while exposing .dense for high-density 1-2 sentence context injection (Item 21).
    """
    full: str
    dense: str
    
    def __new__(cls, full: str, dense: Optional[str] = None):
        full_clean = full.strip() if full else ""
        obj = super().__new__(cls, full_clean)
        obj.full = full_clean
        obj.dense = dense.strip() if dense else cls._extract_dense_fallback(full_clean)
        return obj

    @staticmethod
    def _extract_dense_fallback(full_text: str) -> str:
        """Extract first 1-2 sentences as dense summary fallback."""
        if not full_text:
            return ""
        clean_text = re.sub(r'(?i)^FULL SUMMARY\s*:\s*', '', full_text).strip()
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if s.strip()]
        if len(sentences) >= 2:
            return ' '.join(sentences[:2])
        elif sentences:
            return sentences[0]
        return clean_text[:200]


def parse_summary_response(raw_text: str) -> Tuple[str, str]:
    """Parse raw LLM response into full_summary and dense_summary."""
    if not raw_text:
        return "", ""
    
    dense_match = re.search(r'(?i)\bDENSE SUMMARY\s*:\s*(.+)$', raw_text, re.DOTALL)
    full_match = re.search(r'(?i)(?:FULL SUMMARY\s*:\s*)?(.+?)(?=\bDENSE SUMMARY\s*:|$)', raw_text, re.DOTALL)
    
    if dense_match and full_match:
        full_part = full_match.group(1).strip()
        dense_part = dense_match.group(1).strip()
        dense_part = re.split(r'\n\n+', dense_part)[0].strip()
        if full_part and dense_part:
            return full_part, dense_part
    
    full_clean = re.sub(r'(?i)^FULL SUMMARY\s*:\s*', '', raw_text).strip()
    dense_clean = ClusterSummary._extract_dense_fallback(full_clean)
    return full_clean, dense_clean


class OllamaSummarizer:
    """Handles cluster summarization using local Ollama LLM with retry backoff and fallback heuristics."""
    
    def __init__(self, 
                 ollama_model: Optional[str] = None, 
                 ollama_url: Optional[str] = None,
                 timeout: Optional[int] = None,
                 max_retries: Optional[int] = None,
                 retry_delay: Optional[float] = None):
        self.ollama_model = ollama_model or config.ollama.generation_model
        self.ollama_url = (ollama_url or config.ollama.url).rstrip('/')
        self.timeout = timeout or config.ollama.timeout
        self.max_retries = max_retries or config.ollama.max_retries
        self.retry_delay = retry_delay or config.ollama.retry_delay
    
    def generate_summary(self, 
                         texts: List[str], 
                         max_chunks: int = 15,
                         source_doc: Optional[str] = None,
                         page_range: Optional[Tuple[int, int]] = None) -> ClusterSummary:
        """
        Generate dual full & dense summary using Ollama local LLM with source attribution constraints.
        
        Args:
            texts: List of text chunk strings from the cluster.
            max_chunks: Maximum chunks to sample.
            source_doc: Source filename (e.g. 'The Book of Five Rings.pdf') for strict attribution.
            page_range: Optional (min_page, max_page) tuple for citation context.
        """
        sample_size = min(max_chunks, len(texts))
        sampled_texts = texts[:sample_size]
        
        combined_text = "\n\n---\n\n".join(sampled_texts)
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        doc_label = f"from document '{source_doc}'" if source_doc else "from the provided document"
        pages_label = f" (Pages {page_range[0]}-{page_range[1]})" if (page_range and page_range[0] and page_range[1]) else ""
        
        prompt = f"""You are an expert document summarizer. Summarize the following excerpts {doc_label}{pages_label}.

CONSTRAINTS:
1. STRICT ATTRIBUTION: Attribute all concepts strictly to '{source_doc or 'the source text'}'. Never blend concepts, characters, or authors from other books.
2. ACCURACY: Identify the central themes, key arguments, and specific terminology in these excerpts.
3. OUTPUT FORMAT: You must format your response with two clear labeled sections:
FULL SUMMARY:
[1-2 focused paragraphs explaining the main topic, context, and key details]

DENSE SUMMARY:
[Exactly 1-2 concise, high-density sentences summarizing the core takeaway for retrieval citations]

Content to summarize:
{combined_text}

FULL SUMMARY:
"""
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 400
                        }
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    raw_summary = result.get('response', '').strip()
                    if raw_summary:
                        full_s, dense_s = parse_summary_response(raw_summary)
                        return ClusterSummary(full=full_s, dense=dense_s)
                    logger.warning("Empty response received from Ollama. Falling back to heuristic summary.")
                    return self.generate_fallback_summary(texts, source_doc=source_doc)
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Ollama generation attempt {attempt}/{self.max_retries} returned error: {last_error}")
                    
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout after {self.timeout}s: {e}"
                logger.warning(f"Ollama attempt {attempt}/{self.max_retries} timed out. Model may be loading.")
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning(f"Ollama attempt {attempt}/{self.max_retries} failed to connect to {self.ollama_url}.")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error in Ollama generation: {e}")
                break
                
            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (1.5 ** (attempt - 1))
                time.sleep(sleep_time)

        logger.error(f"All {self.max_retries} Ollama summarization attempts failed ({last_error}). Using fallback summary.")
        return self.generate_fallback_summary(texts, source_doc=source_doc)
    
    def generate_fallback_summary(self, texts: List[str], source_doc: Optional[str] = None) -> ClusterSummary:
        """Fallback summary generation using frequency analysis if Ollama is unavailable."""
        all_text = ' '.join(texts)
        words = all_text.lower().split()
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'is', 'are', 'was', 'were', 'of', 'with', 'that', 'this'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(10)
        
        doc_note = f" in '{source_doc}'" if source_doc else ""
        dense = f"Cluster of {len(texts)} chunks{doc_note} covering key concepts: {', '.join([w for w, _ in top_keywords[:4]])}."
        full = f"""Cluster Summary (Heuristic Fallback):
- Source: {source_doc or 'Various'}
- Total chunks: {len(texts)}
- Key themes: {', '.join([word for word, _ in top_keywords[:8]])}
- Sample content: {texts[0][:200]}...

This cluster contains content related to these key concepts."""
        return ClusterSummary(full=full.strip(), dense=dense.strip())
    
    def check_health(self, required_models: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Check if Ollama service is reachable and has the required models."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False, f"Ollama returned status {response.status_code}"
            
            available_models = [m.get('name', '') for m in response.json().get('models', [])]
            
            if required_models:
                for req in required_models:
                    if not any(req in name for name in available_models):
                        return False, f"Model '{req}' not found. Run 'ollama pull {req}'"
            
            return True, "Ollama is ready"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Ollama on {self.ollama_url}. Run 'ollama serve' first."
        except Exception as e:
            return False, f"Ollama health check failed: {str(e)}"

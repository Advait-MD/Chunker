import sys
from typing import List, Tuple, Optional
from collections import Counter
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class OllamaSummarizer:
    """Handles cluster summarization using local Ollama LLM with fallback heuristics."""
    
    def __init__(self, ollama_model: str = "gemma3:1b", ollama_url: str = "http://localhost:11434"):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip('/')
    
    def generate_summary(self, texts: List[str], max_chunks: int = 15) -> str:
        """Generate summary using Ollama local LLM."""
        sample_size = min(max_chunks, len(texts))
        sampled_texts = texts[:sample_size]
        
        combined_text = "\n\n---\n\n".join(sampled_texts)
        
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        prompt = f"""You are a document summarizer. Summarize the following content into a concise, informative summary that captures the main themes and key points.

Content to summarize:
{combined_text}

Provide a clear summary (3-5 paragraphs) that:
1. Identifies the main topic or theme
2. Highlights key concepts discussed
3. Mentions important details
4. Uses clear, concise language

Summary:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get('response', '').strip()
                return summary if summary else self.generate_fallback_summary(texts)
            else:
                print(f"Ollama API error: {response.status_code}")
                return self.generate_fallback_summary(texts)
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self.generate_fallback_summary(texts)
    
    def generate_fallback_summary(self, texts: List[str]) -> str:
        """Fallback summary generation using frequency analysis if Ollama fails."""
        all_text = ' '.join(texts)
        words = all_text.lower().split()
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'is', 'are', 'was', 'were', 'of', 'with'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(10)
        
        summary = f"""Cluster Summary:
- Total chunks: {len(texts)}
- Key themes: {', '.join([f"{word}" for word, _ in top_keywords[:8]])}
- Sample content: {texts[0][:200]}...

This cluster contains content related to these key concepts."""
        return summary.strip()
    
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
                        return False, f"Model '{req}' not found in Ollama"
            
            return True, "Ollama is ready"
        except Exception as e:
            return False, f"Cannot connect to Ollama: {str(e)}"
